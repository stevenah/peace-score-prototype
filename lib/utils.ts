import type { PeaceScore } from "./types";

import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS } from "./constants";

export function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function formatConfidence(confidence: number): string {
  return `${Math.round(confidence * 100)}%`;
}

export function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins}:${secs.padStart(4, "0")}`;
}

export function getScoreColor(score: PeaceScore): string {
  return PEACE_SCORE_COLORS[score];
}

export function getScoreLabel(score: PeaceScore): string {
  return PEACE_SCORE_LABELS[score];
}

// export function cn(...classes: (string | undefined | false | null)[]): string {
//   return classes.filter(Boolean).join(" ");
// }

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}


export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Compute min, max, avg from an array of PEACE scores (0-3). */
export function computeScoreStats(scores: number[]): {
  minScore: number | null;
  maxScore: number | null;
  avgScore: number | null;
} {
  if (scores.length === 0) return { minScore: null, maxScore: null, avgScore: null };
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  const avg = scores.reduce((sum, s) => sum + s, 0) / scores.length;
  return {
    minScore: min,
    maxScore: max,
    avgScore: Math.round(avg * 100) / 100,
  };
}

/**
 * Round half to even ("banker's rounding"), matching Python's built-in round().
 *
 * The ML backend aggregates per-region scores with `round(sum(scores) / len(scores))`
 * (ml-backend/app/ml/pipeline.py). Ties are common with small integer score sets
 * (e.g. [2, 3] -> 2.5), and JS Math.round rounds half *up*, so using it here would
 * make the live page and the batch results page disagree on the same video.
 */
export function roundHalfToEven(value: number): number {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff > 0.5) return floor + 1;
  if (diff < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Mean PEACE score for a region, rounded the same way the batch pipeline does. */
export function meanScore(scores: number[]): number {
  return roundHalfToEven(scores.reduce((sum, s) => sum + s, 0) / scores.length);
}
