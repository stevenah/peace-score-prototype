// === PEACE Score Types ===

export type PeaceScore = 0 | 1 | 2 | 3;

export type AnatomicalRegion = "esophagus" | "stomach" | "duodenum";

export type MotionDirection = "insertion" | "withdrawal" | "stationary";

export type AnalysisStatus = "queued" | "processing" | "completed" | "failed";

export interface PeaceScoreResult {
  score: PeaceScore;
  label: string;
  confidence: number;
  /**
   * Full smoothed softmax over the four PEACE classes. Present with the real
   * models, absent with the mock backend — always guard before using.
   */
  probs?: number[];
  /** Confidence of the region prediction, when the backend reports it. */
  region_confidence?: number;
}

export interface FrameScoreEntry {
  frame_index: number;
  timestamp: number;
  score: PeaceScore;
  confidence: number;
}

export interface RegionScore extends PeaceScoreResult {
  region: AnatomicalRegion;
  frame_scores: FrameScoreEntry[];
}

// === Motion Types ===

export interface MotionSegment {
  start_time: number;
  end_time: number;
  direction: MotionDirection;
  confidence: number;
}

export interface MotionResult {
  direction: MotionDirection;
  confidence: number;
  optical_flow_magnitude: number;
}

// === Analysis Types ===

export interface VideoMetadata {
  duration_seconds: number;
  fps: number;
  resolution: [number, number];
  total_frames: number;
  analyzed_frames: number;
}

export interface TimelineEntry {
  timestamp: number;
  frame_index: number;
  motion: MotionDirection;
  region: AnatomicalRegion;
  peace_score: PeaceScore;
  confidence: number;
  /**
   * Landmark fields, written only for frames that carried a landmark block, so a
   * timeline saved with the feature off is identical to one saved before it.
   */
  landmark_top?: StationKey | null;
  landmark_confidence?: number | null;
  landmark_status?: LandmarkStatus;
  station_current?: StationKey | null;
  station_events?: LandmarkEvents;
}

export interface MotionAnalysis {
  segments: MotionSegment[];
}

export interface PeaceScores {
  overall: PeaceScoreResult;
  by_region: Partial<Record<AnatomicalRegion, RegionScore>>;
}

export interface AnalysisResults {
  motion_analysis: MotionAnalysis;
  peace_scores: PeaceScores;
  timeline: TimelineEntry[];
}

export interface AnalysisResponse {
  analysis_id: string;
  status: AnalysisStatus;
  progress: number;
  video_metadata?: VideoMetadata;
  results?: AnalysisResults;
  created_at: string;
  completed_at?: string;
  error?: string;
  video_url?: string | null;
  /** Live analyses only, when ESGE station tracking ran. */
  stations?: StationsSummary;
}

// === Frame Analysis ===

export interface FrameAnalysisResponse {
  peace_score: PeaceScoreResult;
  motion?: MotionResult;
  region?: AnatomicalRegion;
  processing_time_ms: number;
}

// === Detections ===

/**
 * Axis-aligned box, NORMALISED against the submitted source frame (origin
 * top-left, all values 0..1).
 *
 * Normalised rather than pixel coordinates because the backend resizes every
 * frame to 224x224 before inference, so pixel coords would be expressed in a
 * resolution nothing downstream knows about. Note that resize is currently NOT
 * aspect-preserving, so normalised coords map straight back to the full frame by
 * simple multiplication — if preprocessing ever switches to letterboxing, this
 * mapping must change with it.
 */
export interface NormalizedBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export type DetectionKind = "lesion" | "landmark" | "instrument" | "artifact";

export interface Detection {
  id: string;
  kind: DetectionKind;
  box: NormalizedBox;
  confidence: number;
  label?: string;
}

// === ESGE stations (landmark classifier) ===

/**
 * The 10 ESGE upper-GI photodocumentation stations, keyed as in
 * contracts/stations.json. Every per-station array on the wire is in this ESGE
 * order — see lib/live/stations.ts for the order and labels.
 */
export type StationKey =
  | "esophagus_proximal"
  | "esophagus_distal"
  | "z_line"
  | "duodenal_bulb"
  | "duodenum_descending"
  | "antrum"
  | "cardia_fundus_retroflex"
  | "lesser_curvature_retroflex"
  | "incisura"
  | "corpus_greater_curvature";

/** Model state per station. "observed" is sticky for the session. */
export type StationStatus = "unseen" | "candidate" | "observed";

/**
 * Per-frame landmark status. Only "ok" frames count as evidence; the rest are
 * evaluated (or skipped) but qualify for no station.
 */
export type LandmarkStatus =
  | "ok"
  | "uncertain"
  | "low_quality"
  | "unsupported_layout"
  | "skipped"
  | "error";

export interface LandmarkEvents {
  /** Stations that became observed on THIS frame. Lossy — prefer the levels. */
  observed: StationKey[];
  /** Stations for which THIS frame is a new best view. */
  best_frame: StationKey[];
}

/**
 * The additive `landmark` block of a live frame result (schema esge10.v1, see
 * contracts/live_frame_result.v2.example.json). Absent when the backend runs
 * with the landmark feature off.
 */
export interface LandmarkResult {
  schema: "esge10.v1";
  model_version: string;
  /** Backend display flag. false = shadow mode: record, but show nothing. */
  display: boolean;
  status: LandmarkStatus;
  /** Present iff the video layout was recognised. */
  layout?: string;
  mode?: string;
  /** Present iff status is ok | uncertain | low_quality. ESGE order. */
  probs?: number[];
  top?: StationKey;
  confidence?: number;
  quality?: number;
  current: StationKey | null;
  /** 10 entries, ESGE order. */
  stations: StationStatus[];
  /** 10 entries, ESGE order. false = manual-tick only for that station. */
  auto_enabled: boolean[];
  events: LandmarkEvents;
  landmark_ms?: number;
}

export type ManualStationMark = "confirmed" | "rejected" | null;

/** What a saved live analysis keeps about stations (AnalysisSession.stationsData). */
export interface StationsSummary {
  schema: "esge10.v1";
  model_version: string | null;
  display: boolean;
  availability: "absent" | "ok" | "unsupported_layout" | "error";
  /** Model state, sticky union over the session. ESGE order. */
  status: StationStatus[];
  /** Clinician overrides; they win over the model. ESGE order. */
  manual: ManualStationMark[];
  auto_enabled: boolean[];
  /** Video time at which each station first counted as observed, else null. */
  observed_at_t: (number | null)[];
}

// === Live Feed ===

export interface LiveFrameResult {
  type: "frame_result";
  /**
   * WARNING: this is a Unix epoch from the server (time.time()), not a media
   * time. Never build a timeline from it — pair results with the client-side
   * video time instead (see hooks/useLiveFeed).
   */
  timestamp: number;
  frame_index: number;
  peace_score: PeaceScoreResult;
  motion?: MotionResult;
  region?: AnatomicalRegion;
  processing_time_ms: number;
  /**
   * Present only once a detection head is deployed. The current model
   * (ml-backend/app/ml/model.py) has two classification heads and no box
   * regression, so this is always absent today.
   */
  detections?: Detection[];
  /**
   * ESGE station classifier output. Raw wire data: always run it through
   * parseLandmark (lib/live/wire.ts) before use — a malformed block is dropped
   * there while the PEACE fields above keep flowing.
   */
  landmark?: LandmarkResult;
}

export interface LiveErrorMessage {
  type: "error";
  /** Index of the frame that failed, so its pending ticket can be retired. */
  frame_index?: number;
  message?: string;
}

export type LiveMessage = LiveFrameResult | LiveErrorMessage;

// === Colon Segments ===

export type ColonSegment =
  | "rectum"
  | "sigmoid"
  | "descending"
  | "splenic_flexure"
  | "transverse"
  | "hepatic_flexure"
  | "ascending"
  | "cecum";

export const SEGMENT_LABELS: Record<ColonSegment, string> = {
  rectum: "Rectum",
  sigmoid: "Sigmoid",
  descending: "Descending",
  splenic_flexure: "Splenic Flexure",
  transverse: "Transverse",
  hepatic_flexure: "Hepatic Flexure",
  ascending: "Ascending",
  cecum: "Cecum",
};

// === Dashboard Records ===

export interface AnalysisRecord {
  id: string;
  analysisId: string;
  filename: string;
  status: string;
  overallScore: number | null;
  minScore: number | null;
  maxScore: number | null;
  avgScore: number | null;
  framesAnalyzed: number | null;
  duration: number | null;
  createdAt: string;
  completedAt: string | null;
}

// === Config ===

export interface AnalysisConfig {
  sample_rate_fps: number;
  enable_motion_detection: boolean;
  enable_peace_scoring: boolean;
  regions: AnatomicalRegion[];
}

// === Batch Upload ===

export type BatchItemStatus =
  | "pending"
  | "uploading"
  | "queued"
  | "processing"
  | "completed"
  | "failed";

export interface BatchItem {
  /** Client-generated ID for tracking before analysisId is assigned */
  id: string;
  file: File;
  status: BatchItemStatus;
  /** Overall progress 0-1 across upload + processing phases */
  progress: number;
  /** Set after successful upload */
  analysisId: string | null;
  /** Analysis response from polling */
  analysis: AnalysisResponse | null;
  /** Upload or polling error */
  error: string | null;
}
