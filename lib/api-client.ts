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

export async function uploadVideo(file: File): Promise<{ analysis_id: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });

  return handleResponse(response);
}

export async function getAnalysis(id: string): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE}/analysis/${id}`);
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

export function createLiveWebSocket(): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/live`;
  return new WebSocket(wsUrl);
}

export { ApiError };
