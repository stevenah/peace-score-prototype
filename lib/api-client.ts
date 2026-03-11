import type { AnalysisResponse, FrameAnalysisResponse } from "./types";

const API_BASE = "/api";

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
    // Step 1: Initialize upload via Next.js (auth + quota + presigned S3 URL)
    const initRes = await fetch(`${API_BASE}/upload/init`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ filename: file.name }),
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

    const { uploadId, presignedUrl, s3Key } = await initRes.json();

    // Step 2: Upload directly to S3 using presigned PUT URL
    const xhr = new XMLHttpRequest();
    await new Promise<void>((resolve, reject) => {
      xhr.open("PUT", presignedUrl);
      xhr.setRequestHeader("Content-Type", "video/mp4");

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(e.loaded / e.total);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve();
        } else {
          reject(new ApiError(xhr.status, "S3 upload failed"));
        }
      };
      xhr.onerror = () => reject(new ApiError(0, "S3 upload failed"));
      xhr.onabort = () => reject(new ApiError(0, "Upload aborted"));

      controller.signal.addEventListener("abort", () => xhr.abort());
      xhr.send(file);
    });

    // Step 3: Notify backend — registers in DB and kicks off ML analysis
    const completeRes = await fetch(`${API_BASE}/upload/complete`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        uploadId,
        s3Key,
        filename: file.name,
      }),
      signal: controller.signal,
    });

    if (!completeRes.ok) {
      const text = await completeRes.text().catch(() => "Upload complete failed");
      throw new ApiError(completeRes.status, text);
    }

    const result = await completeRes.json();
    return { analysis_id: result.analysisId };
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
