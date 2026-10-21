/**
 * Types and small helpers for the admin Models tab.
 *
 * The shapes mirror the ML backend's model admin API (ml-backend/app/api/models.py
 * and app/services/evals.py), reached through /api/admin/ml/*.
 */

import { STATION_LABELS, STATION_SHORT } from "@/lib/live/stations";
import type { StationKey } from "@/lib/types";

export type ServingMode = "off" | "shadow" | "display";

export interface Serving {
  mode: ServingMode;
  /** True when the mode was chosen on this page rather than taken from the env. */
  mode_saved: boolean;
  mock: boolean;
  selected_version: string | null;
  default: { kind: "env" | "lock" | "none"; version: string | null };
  loaded_version: string | null;
  device: string | null;
  failure: string | null;
}

export interface ModelInfo {
  version: string;
  path: string;
  source: string;
  servable: boolean;
  problem: string | null;
  arch: string | null;
  input_size: number | null;
  layouts_version: string | null;
  calibrated: boolean;
  warning: string | null;
  size_bytes: number | null;
  modified_at: string | null;
  training: {
    mode: string | null;
    fold: number | null;
    candidate: string | null;
    variant: string | null;
    run: string | null;
  };
  metrics: Record<string, unknown>;
}

export interface Experiment {
  name: string;
  candidate: string | null;
  input_size: number | null;
  folds: { fold: number; clip_macro_f1: number | null; has_oof: boolean }[];
  mean_clip_macro_f1: number | null;
  has_all_dev: boolean;
  has_calibration: boolean;
  has_tracker: boolean;
  oof_macro_f1: Interval | null;
  outcome: string | null;
  test_scored: boolean;
}

export type EvalKind =
  | "selftest"
  | "latency"
  | "accuracy"
  | "oof_metrics"
  | "gates_report"
  | "replay";

export interface EvalKindInfo {
  kind: EvalKind;
  label: string;
  target: "bundle" | "experiment";
  summary: string;
  available: boolean;
  reason: string | null;
}

export interface ModelsResponse {
  admin_enabled: boolean;
  serving: Serving;
  stations: StationKey[];
  models: ModelInfo[];
  experiments: Experiment[];
  eval_kinds: EvalKindInfo[];
}

export type EvalStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface EvalJob {
  id: string;
  kind: EvalKind;
  label: string;
  target: { type: "bundle" | "experiment"; id: string; path: string };
  params: Record<string, string | number>;
  status: EvalStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  progress: { done: number; total: number; label: string } | null;
  headline: string | null;
  verdict: "pass" | "fail" | null;
  error: string | null;
  /** Only on a single-job fetch. */
  result?: unknown;
  log?: string;
}

// -- results ------------------------------------------------------------------

export interface Interval {
  value: number;
  lo: number | null;
  hi: number | null;
}

export interface SelftestResult {
  ok: boolean;
  version: string;
  seconds: number;
  checks: {
    name: string;
    label: string;
    status: "passed" | "failed" | "skipped";
    detail: string;
  }[];
}

interface Percentiles {
  p50: number | null;
  p95: number | null;
  p99: number | null;
}

export interface LatencyRow {
  config: string;
  streams: number;
  frames: number;
  frame_ms: Percentiles;
  landmark_ms: Percentiles;
  drift: number | null;
  rss_peak_mb: number;
}

export interface LatencyResult {
  host: { platform: string; machine: string; torch: string; torch_threads: number };
  mode: "screen" | "sustained";
  streams: number;
  rate: number | null;
  peace: "real" | "mock";
  baseline: LatencyRow;
  model: LatencyRow;
}

export interface PerClass {
  n: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
}

export interface ClipMetrics {
  n_clips: number;
  n_patients: number;
  coverage: number | null;
  macro_f1: Interval;
  bal_acc: Interval;
  top2_acc?: number | null;
  per_class: Partial<Record<StationKey, PerClass>>;
  /** Rows are the true station, columns the predicted one, in station order. */
  confusion: number[][];
}

export interface FrameMetrics {
  centre_quality: { n: number; macro_f1: number | null; accuracy: number | null };
  all_frames: { n: number; macro_f1: number | null; accuracy: number | null };
  calibration: { n: number; ece: number | null; brier: number | null; nll: number | null };
}

/** bundle_eval.py; evaluate.py's eval_oof.json carries the same clip and frame blocks. */
export interface AccuracyResult {
  stations: StationKey[];
  clip: ClipMetrics;
  frame: FrameMetrics | null;
  patient_level?: { macro_f1: number | null } | null;
  // bundle_eval only
  split?: {
    name: "heldout" | "shift";
    label: string;
    n_clips: number;
    n_patients: number;
    n_frames: number;
    skipped_clips: number;
  };
  device?: string;
  calibration?: { temperature: number; q_min: number; quality_gate: boolean };
  acceptance?: {
    n_frames: number;
    share: { ok: number; uncertain: number; low_quality: number };
    precision_ok: number | null;
  };
  warnings?: string[];
  seconds?: number;
  // evaluate.py only
  folds?: number[];
  clip_merged?: { macro_f1: Interval | null; n_clips: number | null };
  per_fold_macro_f1?: Record<string, number>;
}

export interface GatesResult {
  outcome: string | null;
  gates: Record<string, { status: string | null }>;
  auto_enabled: boolean[] | null;
  stations: StationKey[];
  clip_macro_f1: Interval | null;
  report: string | null;
}

export interface ReplayScenario {
  n_patients?: number;
  n_streams?: number;
  recall_mean?: number | null;
  recall_mean_lo?: number | null;
  recall_mean_hi?: number | null;
  tto_median?: number | null;
  fdr_overall?: number | null;
  fdr_overall_lo?: number | null;
  fdr_overall_hi?: number | null;
  flicker_per_min?: number | null;
  best_frame_purity?: number | null;
}

export interface ReplayResult {
  transit: "placeholder" | "model";
  stations: StationKey[];
  scenarios: Record<string, ReplayScenario>;
}

// -- helpers ------------------------------------------------------------------

export const MODE_LABELS: Record<ServingMode, string> = {
  off: "Off",
  shadow: "Shadow",
  display: "On screen",
};

export const MODE_DESCRIPTIONS: Record<ServingMode, string> = {
  off: "Stations are not computed.",
  shadow: "Stations are computed and logged. The workstation shows nothing.",
  display: "Stations are computed and shown in the live workstation.",
};

/** The live workstation sends 2 frames a second, so each frame has 500 ms. */
export const LIVE_FRAME_INTERVAL_MS = 500;

export function isActive(job: Pick<EvalJob, "status">): boolean {
  return job.status === "queued" || job.status === "running";
}

/** A metric in [0, 1] to three decimals, the way the training reports print it. */
export function fmtScore(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? "–" : v.toFixed(3);
}

export function fmtInterval(i: Interval | null | undefined): string {
  if (!i || i.lo == null || i.hi == null) return "";
  return `${fmtScore(i.lo)} to ${fmtScore(i.hi)}`;
}

export function fmtPercent(v: number | null | undefined, digits = 0): string {
  return v == null || !Number.isFinite(v) ? "–" : `${(v * 100).toFixed(digits)}%`;
}

export function fmtMs(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "–";
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ms`;
}

/** "42 s", "3 min 5 s", "1 h 12 min". */
export function fmtElapsed(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}

/** Run time of a job: start to finish, or start to `now` while it runs. */
export function jobSeconds(job: EvalJob, now: number = Date.now()): number | null {
  if (!job.started_at) return null;
  const end = job.finished_at ? Date.parse(job.finished_at) : now;
  return (end - Date.parse(job.started_at)) / 1000;
}

export function progressFraction(job: EvalJob): number | null {
  const p = job.progress;
  if (!p || !(p.total > 0)) return null;
  return Math.min(1, Math.max(0, p.done / p.total));
}

export function stationLabel(key: string): string {
  return STATION_LABELS[key as StationKey] ?? key;
}

export function stationShort(key: string): string {
  return STATION_SHORT[key as StationKey] ?? key;
}

/** What a bundle was trained on, in a few words. */
export function trainingSummary(model: ModelInfo): string {
  const t = model.training;
  const parts: string[] = [];
  if (t.candidate) parts.push(t.candidate);
  if (t.mode === "all_dev") parts.push("all dev patients");
  else if (t.mode === "smoke") parts.push("smoke run");
  else if (t.mode === "fold" && t.fold != null) parts.push(`fold ${t.fold} held out`);
  else if (t.run) parts.push(t.run);
  return parts.join(", ");
}

/**
 * Why held-out accuracy cannot be run on this bundle, or null when it can.
 * Mirrors the backend's refusal so the menu can say so before the click.
 */
export function heldOutBlocker(model: ModelInfo): string | null {
  const t = model.training;
  if (t.mode === "all_dev") return "Trained on every dev patient: no fold is held out.";
  if ((t.mode !== "fold" && t.mode !== "smoke") || t.fold == null)
    return "Its metadata does not say which fold it held out.";
  return null;
}

/** Each row of a confusion matrix as shares of that row's total. */
export function rowShares(confusion: number[][]): number[][] {
  return confusion.map((row) => {
    const total = row.reduce((a, b) => a + b, 0);
    return row.map((v) => (total > 0 ? v / total : 0));
  });
}

/** One short line saying what a job ran on and how. */
export function describeParams(job: EvalJob): string {
  const p = job.params;
  switch (job.kind) {
    case "latency":
      return p.mode === "sustained"
        ? `${p.streams} streams at 2 frames/s for ${p.minutes} min`
        : `${p.frames} frames back to back`;
    case "accuracy":
      return p.split === "shift" ? "legacy shift set" : "held-out dev fold";
    case "replay":
      return `${p.transit} transit frames`;
    default:
      return "";
  }
}
