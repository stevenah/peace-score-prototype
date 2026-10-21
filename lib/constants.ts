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

/** The PEACE scale as hex, for SVG attributes and inline styles. Keep in step
 *  with --peace-0..3 in app/globals.css, which is the same scale as tokens. */
export const PEACE_SCORE_COLORS: Record<PeaceScore, string> = {
  0: "#ef4444",
  1: "#f97316",
  2: "#84cc16",
  3: "#22c55e",
};

export const MOTION_LABELS: Record<MotionDirection, string> = {
  insertion: "Inserting",
  withdrawal: "Withdrawal",
  stationary: "Stationary",
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

export const MAX_FILE_SIZE_MB = 1024;
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
