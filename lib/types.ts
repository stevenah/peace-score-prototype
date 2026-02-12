// === PEACE Score Types ===

export type PeaceScore = 0 | 1 | 2 | 3;

export type AnatomicalRegion = "esophagus" | "stomach" | "duodenum";

export type MotionDirection = "insertion" | "retraction" | "stationary";

export type AnalysisStatus = "queued" | "processing" | "completed" | "failed";

export interface PeaceScoreResult {
  score: PeaceScore;
  label: string;
  confidence: number;
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
}

// === Frame Analysis ===

export interface FrameAnalysisResponse {
  peace_score: PeaceScoreResult;
  motion?: MotionResult;
  region?: AnatomicalRegion;
  processing_time_ms: number;
}

// === Live Feed ===

export interface LiveFrameResult {
  type: "frame_result";
  timestamp: number;
  frame_index: number;
  peace_score: PeaceScoreResult;
  motion?: MotionResult;
  region?: AnatomicalRegion;
  processing_time_ms: number;
}

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

// === Config ===

export interface AnalysisConfig {
  sample_rate_fps: number;
  enable_motion_detection: boolean;
  enable_peace_scoring: boolean;
  regions: AnatomicalRegion[];
}
