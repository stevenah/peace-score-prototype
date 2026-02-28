import type { AnatomicalRegion, MotionDirection, PeaceScore } from "./types";

export const PEACE_SCORE_LABELS: Record<PeaceScore, string> = {
  0: "Poor",
  1: "Inadequate",
  2: "Adequate",
  3: "Excellent",
};

export const PEACE_SCORE_DESCRIPTIONS: Record<PeaceScore, string> = {
  0: "Substantial fluid/content completely obscuring evaluation",
  1: "Substantial opaque fluid/content, some parts not evaluable",
  2: "Small amount of hazy fluid/foam, most mucosa visible",
  3: "Clean mucosa or minor transparent fluid, no impediment",
};

export const PEACE_SCORE_COLORS: Record<PeaceScore, string> = {
  0: "#ef4444", // red-500
  1: "#f97316", // orange-500
  2: "#84cc16", // lime-500
  3: "#22c55e", // green-500
};

export const PEACE_SCORE_BG_COLORS: Record<PeaceScore, string> = {
  0: "#fef2f2", // red-50
  1: "#fff7ed", // orange-50
  2: "#f7fee7", // lime-50
  3: "#f0fdf4", // green-50
};

export const MOTION_LABELS: Record<MotionDirection, string> = {
  insertion: "Inserting",
  retraction: "Retracting",
  stationary: "Stationary",
};

export const MOTION_COLORS: Record<MotionDirection, string> = {
  insertion: "#3b82f6", // blue-500
  retraction: "#f97316", // orange-500
  stationary: "#6b7280", // gray-500
};

export const REGION_LABELS: Record<AnatomicalRegion, string> = {
  esophagus: "Esophagus",
  stomach: "Stomach",
  duodenum: "Duodenum",
};

export const REGION_ORDER: AnatomicalRegion[] = [
  "esophagus",
  "stomach",
  "duodenum",
];

export const ALLOWED_VIDEO_TYPES = [
  "video/mp4",
  "video/quicktime",
  "video/x-msvideo",
  "video/x-matroska",
];

export const MAX_FILE_SIZE_MB = 500;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

export const ML_BACKEND_URL =
  process.env.ML_BACKEND_URL || "http://localhost:8000";

export function getWsUrl(): string {
  if (typeof window === "undefined") return "";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return process.env.NEXT_PUBLIC_WS_URL || `${protocol}//${window.location.host}/api/live`;
}
