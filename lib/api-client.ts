import type { AnalysisResponse, FrameAnalysisResponse } from "./types";

const API_BASE = "/api";
const CHUNK_SIZE = 8 * 1024 * 1024; // 8MB per chunk

// ML backend URL for direct chunk uploads (bypasses Next.js proxy).
// NEXT_PUBLIC_ vars are baked at build time. If missing, derive from hostname.
function getMLBackendUrl(): string {
  if (process.env.NEXT_PUBLIC_ML_BACKEND_URL) {
    return process.env.NEXT_PUBLIC_ML_BACKEND_URL;
  }
  if (typeof window !== "undefined") {
    // Production: demo.gipeace.com → peace-ml.fly.dev
    if (window.location.hostname === "demo.gipeace.com" ||
        window.location.hostname === "peace-frontend.fly.dev") {
      return "https://peace-ml.fly.dev";
    }
  }
  return ""; // Local dev — falls back to /ml-api/ rewrite proxy
}

const ML_BACKEND_URL = getMLBackendUrl();

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error");
    throw new ApiError(response.status, text);
  }
  return response.json();
}

export function uploadVideo(
  file: File,
  onProgress?: (fraction: number) => void,
): { promise: Promise<{ analysis_id: string }>; abort: () => void } {
  const controller = new AbortController();

  const promise = (async (): Promise<{ analysis_id: string }> => {
    // Step 1: Initialize upload via Next.js (auth + quota check)
    const initRes = await fetch(`${API_BASE}/upload/init`, {
      method: "POST",
      signal: controller.signal,
    });

    if (!initRes.ok) {
      let message = "Upload initialization failed";
      try {
        const data = await initRes.json();
        message = data.message || data.error || message;
      } catch {}
      throw new ApiError(initRes.status, message);
    }

    const { uploadId } = await initRes.json();

    // Step 2: Send chunks directly to ML backend (bypasses Next.js)
    // This avoids Next.js body size limits and keeps the 512MB frontend lean.
    const chunkUrl = ML_BACKEND_URL
      ? `${ML_BACKEND_URL}/api/v1/upload/chunk`
      : `/ml-api/upload/chunk`; // fallback to Next.js rewrite proxy for local dev

    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    let lastResult: { analysis_id: string } | null = null;

    for (let i = 0; i < totalChunks; i++) {
      if (controller.signal.aborted) {
        throw new ApiError(0, "Upload aborted");
      }

      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const chunkRes = await fetch(chunkUrl, {
        method: "POST",
        headers: {
          "x-upload-id": uploadId,
          "x-chunk-index": String(i),
          "x-total-chunks": String(totalChunks),
          "x-filename": file.name,
        },
        body: chunk,
        signal: controller.signal,
      });

      if (!chunkRes.ok) {
        let message = "Chunk upload failed";
        try {
          const data = await chunkRes.json();
          message = data.message || data.error || message;
        } catch {}
        throw new ApiError(chunkRes.status, message);
      }

      const chunkData = await chunkRes.json();

      // Report progress after each chunk
      if (onProgress) {
        onProgress((i + 1) / totalChunks);
      }

      // Last chunk returns the analysis result
      if (chunkData.analysis_id) {
        lastResult = chunkData;
      }
    }

    if (!lastResult) {
      throw new ApiError(0, "Upload completed but no analysis ID received");
    }

    // Step 3: Register the analysis in the Next.js DB
    const completeRes = await fetch(`${API_BASE}/upload/complete`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        uploadId,
        analysisId: lastResult.analysis_id,
        filename: file.name,
      }),
      signal: controller.signal,
    });

    if (!completeRes.ok) {
      // Non-fatal — analysis still runs, just may not show in dashboard
      console.error("Failed to register analysis:", await completeRes.text().catch(() => ""));
    }

    return lastResult;
  })();

  return { promise, abort: () => controller.abort() };
}

export async function getAnalysis(id: string): Promise<AnalysisResponse> {
  const endpoint = id.startsWith("live_")
    ? `${API_BASE}/analysis/live/${id}`
    : `${API_BASE}/analysis/${id}`;
  const response = await fetch(endpoint);
  return handleResponse(response);
}

export async function analyzeFrame(
  frame: Blob,
  previousFrame?: Blob,
): Promise<FrameAnalysisResponse> {
  const formData = new FormData();
  formData.append("frame", frame);
  if (previousFrame) {
    formData.append("previous_frame", previousFrame);
  }

  const response = await fetch(`${API_BASE}/analyze-frame`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(response);
}

export async function saveLiveAnalysis(data: {
  filename: string;
  overallScore: number | null;
  minScore: number | null;
  maxScore: number | null;
  avgScore: number | null;
  framesAnalyzed: number;
  duration: number | null;
  timeline: unknown[];
  videoFile?: File;
}): Promise<{ id: string; analysisId: string }> {
  const formData = new FormData();
  formData.append(
    "metadata",
    JSON.stringify({
      filename: data.filename,
      overallScore: data.overallScore,
      minScore: data.minScore,
      maxScore: data.maxScore,
      avgScore: data.avgScore,
      framesAnalyzed: data.framesAnalyzed,
      duration: data.duration,
      timeline: data.timeline,
    }),
  );
  if (data.videoFile) {
    formData.append("video", data.videoFile);
  }

  const response = await fetch(`${API_BASE}/analysis/save-live`, {
    method: "POST",
    body: formData,
  });
  return handleResponse(response);
}

export function createLiveWebSocket(): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/live`;
  return new WebSocket(wsUrl);
}

export { ApiError };
