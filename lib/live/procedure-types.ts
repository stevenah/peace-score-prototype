import type {
  AnatomicalRegion,
  LandmarkStatus,
  ManualStationMark,
  MotionDirection,
  PeaceScore,
  StationKey,
  StationStatus,
} from "@/lib/types";

/**
 * One analysed frame, with its score paired back to the *video* time at which it
 * was captured.
 *
 * `t` is the only clock this module trusts. LiveFrameResult.timestamp from the
 * backend is a Unix epoch (ml-backend/app/api/websocket.py), not a media time,
 * and must never be read here.
 */
export interface FrameSample {
  readonly seq: number;
  readonly t: number;
  readonly score: PeaceScore;
  readonly scoreConfidence: number;
  /**
   * Expected score, sum(i * p(i)) over the class distribution — a genuinely
   * continuous value, unlike the argmax. Null when the backend sends no
   * distribution (mock models).
   */
  readonly expectedScore: number | null;
  readonly region: AnatomicalRegion | null;
  readonly motion: MotionDirection;
  readonly motionConfidence: number;
  readonly frameIndex: number;
  readonly processingTimeMs: number;

  // --- ESGE stations. All optional: absent when the backend sends no (valid)
  // landmark block, which leaves every PEACE-only path exactly as before. -----
  readonly landmarkStatus?: LandmarkStatus;
  readonly landmarkTop?: StationKey | null;
  readonly landmarkConfidence?: number | null;
  readonly landmarkQuality?: number | null;
  readonly landmarkModelVersion?: string;
  readonly stationsDisplay?: boolean;
  readonly stationCurrent?: StationKey | null;
  /** 10 entries, ESGE order. */
  readonly stationStatus?: readonly StationStatus[];
  /** 10 entries, ESGE order. */
  readonly stationAutoEnabled?: readonly boolean[];
  /** Edge events as sent. Lossy: the store derives edges from the levels. */
  readonly stationObservedEvents?: readonly StationKey[];
  /** This frame is a new best view of these stations. */
  readonly stationBestEvents?: readonly StationKey[];
}

export interface TimersSlice {
  readonly totalElapsedS: number;
  readonly totalTargetS: number;
  /** 0..1, clamped. */
  readonly totalProgress: number;
  readonly currentRegionDwellS: number;
  readonly perRegionS: Readonly<Record<AnatomicalRegion, number>>;
}

export interface RegionsSlice {
  readonly current: AnatomicalRegion | null;
  /** Consensus-smoothed, so it does not flicker frame to frame. */
  readonly motion: MotionDirection;
  readonly visited: readonly AnatomicalRegion[];
}

export interface CoverageSlice {
  readonly regionsVisited: number;
  readonly regionsTotal: number;
  /** The hero number. Real, monotonic, and means exactly what it says. */
  readonly framesAnalysed: number;
  /** 0..100 per region: dwell as a share of the total procedure target. */
  readonly fillByRegion: Readonly<Record<AnatomicalRegion, number>>;
}

export interface ScoreSlice {
  /** Latest raw score. */
  readonly current: PeaceScore | null;
  /** Settled score for the ring — resists single-frame misclassification. */
  readonly display: PeaceScore | null;
  readonly displayLabel: string | null;
  /** Continuous session mean, 0..3. */
  readonly meanScore: number | null;
  /** EMA of recent frames, 0..3. Drives the visibility readout. */
  readonly recentMean: number | null;
  readonly min: PeaceScore | null;
  readonly max: PeaceScore | null;
  readonly perRegionMean: Readonly<Partial<Record<AnatomicalRegion, PeaceScore>>>;
  readonly meanConfidence: number | null;
  /** Latest expected score 0..3, smoothed. Moves continuously. */
  readonly expected: number | null;
}

export type VisibilityLevel = "adequate" | "degraded" | "obscured";

export interface VisibilitySlice {
  /**
   * null when there is no data, or when mean confidence is below the gate — we
   * do not report a status derived from predictions the model is unsure of.
   */
  readonly level: VisibilityLevel | null;
  readonly detail: string | null;
}

export interface FeedSlice {
  readonly framesReceived: number;
  readonly latencyMsEma: number | null;
  readonly ended: boolean;
}

export type StationAvailability = "absent" | "ok" | "unsupported_layout" | "error";

export type ManualMark = ManualStationMark;

/**
 * ESGE station checklist. Flat primitives and flat primitive arrays only, so
 * keepIfEqual can keep the slice referentially stable between frames.
 *
 * Every per-station array has 10 entries in ESGE order (lib/live/stations.ts).
 */
export interface StationsSlice {
  /**
   * "absent" until the first evaluated (not skipped) landmark frame, and
   * forever when the backend runs with the feature off. Latched with
   * hysteresis over evaluated frames — see STATION_AVAILABILITY_RELEASE_FRAMES.
   */
  readonly availability: StationAvailability;
  /** Backend display flag; false = shadow mode. */
  readonly display: boolean;
  /**
   * THE gate. availability === "ok" && display. Every station-driven behaviour —
   * checklist, hero, capture trigger, coaching, announcement, gallery — checks
   * this and nothing else, so shadow mode cannot leak into the UI.
   */
  readonly visible: boolean;
  readonly modelVersion: string | null;
  readonly current: StationKey | null;
  /** Model status, sticky union: once observed, observed for the session. */
  readonly status: readonly StationStatus[];
  /** Clinician overrides. The model never clears them. */
  readonly manual: readonly ManualMark[];
  /** manual ?? (status === "observed"). */
  readonly observed: readonly boolean[];
  readonly autoEnabled: readonly boolean[];
  /** Video time (sample.t) at which each station first counted as observed. */
  readonly observedAtT: readonly (number | null)[];
  readonly observedCount: number;
  readonly total: number;
  /**
   * Auto-enabled stations 4-10 still unobserved while the scope is back in the
   * esophagus after reaching the stomach or duodenum — i.e. the operator is on
   * the way out. Empty during insertion by construction.
   */
  readonly exitingMissing: readonly StationKey[];
}

/** What ProcedureStore.ingest reports back about the frame it just folded in. */
export interface IngestOutcome {
  /**
   * Stations whose effective "observed" turned true on this frame, derived by
   * comparing level arrays — so a lost `events.observed` message loses nothing.
   */
  readonly stationEdges: readonly StationKey[];
}

/**
 * The complete presentational shape. Panels take slices of this and nothing else,
 * so the same panels can later be driven from a saved analysis (see
 * snapshotFromTimeline) rather than only from a live socket.
 */
export interface ProcedureSnapshot {
  readonly rev: number;
  readonly timers: TimersSlice;
  readonly regions: RegionsSlice;
  readonly coverage: CoverageSlice;
  readonly score: ScoreSlice;
  readonly visibility: VisibilitySlice;
  readonly feed: FeedSlice;
  readonly stations: StationsSlice;
}
