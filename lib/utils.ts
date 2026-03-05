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
