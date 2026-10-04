import { PEACE_SCORE_LABELS, REGION_ORDER } from "@/lib/constants";
import { roundHalfToEven } from "@/lib/utils";
import type {
  AnatomicalRegion,
  MotionDirection,
  PeaceScore,
  StationKey,
  StationStatus,
} from "@/lib/types";
import {
  MAX_SAMPLE_GAP_S,
  MOTION_CONFIDENCE_THRESHOLD,
  MOTION_CONSENSUS_FRAMES,
  RECENT_SCORE_ALPHA,
  SCORE_SETTLE_FRAMES,
  SCORE_SETTLE_HOLD_MS,
  STATION_AVAILABILITY_RELEASE_FRAMES,
  TOTAL_PROCEDURE_TARGET_S,
  VISIBILITY_CONFIDENCE_GATE,
} from "./config";
import type {
  CoverageSlice,
  FeedSlice,
  FrameSample,
  IngestOutcome,
  ManualMark,
  ProcedureSnapshot,
  RegionsSlice,
  ScoreSlice,
  StationAvailability,
  StationsSlice,
  TimersSlice,
  VisibilityLevel,
  VisibilitySlice,
} from "./procedure-types";
import {
  GASTRODUODENAL,
  STATION_INDEX,
  STATION_ORDER,
  STATION_REGION,
  STATIONS_TOTAL,
  isObserved,
  missingStations,
} from "./stations";

const SCORE_SETTLE_HOLD_S = SCORE_SETTLE_HOLD_MS / 1000;

function zeroPerRegion(): Record<AnatomicalRegion, number> {
  return { esophagus: 0, stomach: 0, duodenum: 0 };
}

function clamp01(n: number): number {
  return n < 0 ? 0 : n > 1 ? 1 : n;
}

/** Reuse `prev` when every own key matches, so selectors stay referentially stable. */
function keepIfEqual<T extends object>(prev: T | undefined, next: T): T {
  if (!prev) return next;
  const keys = Object.keys(next) as (keyof T)[];
  for (const k of keys) {
    const a = prev[k];
    const b = next[k];
    if (a === b) continue;
    // One level deep for the plain per-region records.
    if (
      a && b &&
      typeof a === "object" && typeof b === "object" &&
      !Array.isArray(a) && !Array.isArray(b)
    ) {
      const ak = Object.keys(a as object);
      const bk = Object.keys(b as object);
      if (ak.length === bk.length &&
          ak.every((kk) => (a as Record<string, unknown>)[kk] === (b as Record<string, unknown>)[kk])) {
        continue;
      }
    }
    if (Array.isArray(a) && Array.isArray(b) &&
        a.length === b.length && a.every((v, i) => v === b[i])) {
      continue;
    }
    return next;
  }
  return prev;
}

/**
 * Accumulates live frame samples into the numbers the workstation displays.
 *
 * Every metric is a fold: `acc' = f(acc, sample)`. No array is ever rescanned, so
 * ingest is O(1) regardless of procedure length. This matters less for speed than
 * for correctness — dwell and coverage are naturally folds, and writing them that
 * way is what forces the seek/pause discontinuity handling to be explicit.
 */
export class ProcedureStore {
  private samples: FrameSample[] = [];
  private listeners = new Set<() => void>();
  private snapshot: ProcedureSnapshot;
  private rev = 0;

  // --- accumulators -------------------------------------------------------
  private elapsedS = 0;
  private prev: FrameSample | null = null;

  private perRegionS = zeroPerRegion();
  private currentRegion: AnatomicalRegion | null = null;
  private currentRegionDwellS = 0;
  private visited: AnatomicalRegion[] = [];

  private scoreSum = 0;
  private scoreN = 0;
  private confSum = 0;
  private recentMean: number | null = null;
  private expected: number | null = null;
  private minScore: PeaceScore | null = null;
  private maxScore: PeaceScore | null = null;
  private perRegionScoreSum = zeroPerRegion();
  private perRegionScoreN = zeroPerRegion();

  private motionRing: MotionDirection[] = [];
  private smoothedMotion: MotionDirection = "stationary";

  private currentScore: PeaceScore | null = null;
  private displayScore: PeaceScore | null = null;
  private displayScoreSetAtT = 0;
  private candidateScore: PeaceScore | null = null;
  private candidateRun = 0;

  private latencyEma: number | null = null;
  private ended = false;

  // --- ESGE stations. Internal arrays may be mutated in place: build() always
  // copies them, so a published array is never one that later changes. -------
  private stAvailability: StationAvailability = "absent";
  /** Consecutive unsupported/error frames (skips ignored), for the availability latch. */
  private stBadRun = 0;
  private stDisplay = false;
  private stModelVersion: string | null = null;
  private stCurrent: StationKey | null = null;
  private stStatus: StationStatus[] = Array(STATIONS_TOTAL).fill("unseen");
  private stManual: ManualMark[] = Array(STATIONS_TOTAL).fill(null);
  private stAutoEnabled: boolean[] = Array(STATIONS_TOTAL).fill(true);
  private stModelObservedAtT: (number | null)[] = Array(STATIONS_TOTAL).fill(null);
  private stManualAtT: (number | null)[] = Array(STATIONS_TOTAL).fill(null);
  /** Latched once the model has placed the scope past the Z-line. */
  private stBeyondEsophagus = false;
  private stFoldFailed = false;

  constructor(private readonly totalTargetS: number = TOTAL_PROCEDURE_TARGET_S) {
    this.snapshot = this.build();
  }

  // --- public API ---------------------------------------------------------

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): ProcedureSnapshot => this.snapshot;

  getSamples(): readonly FrameSample[] {
    return this.samples;
  }

  /** Binary search for the sample nearest a video time. Post-run scrubbing only. */
  findSampleAtTime(t: number): FrameSample | null {
    const a = this.samples;
    if (a.length === 0) return null;
    let lo = 0;
    let hi = a.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (a[mid].t < t) lo = mid + 1;
      else hi = mid;
    }
    const at = a[lo];
    const before = lo > 0 ? a[lo - 1] : null;
    if (before && Math.abs(before.t - t) <= Math.abs(at.t - t)) return before;
    return at;
  }

  markEnded(): void {
    if (this.ended) return;
    this.ended = true;
    this.publish();
  }

  /**
   * Clinician override for one station (ESGE index 0..9). A mark wins over the
   * model in both directions and the model never clears it; null hands the
   * station back to the model.
   */
  setManualStation(index: number, mark: ManualMark): void {
    if (!Number.isInteger(index) || index < 0 || index >= STATIONS_TOTAL) return;
    if (this.stManual[index] === mark) return;
    this.stManual[index] = mark;
    // A manual confirm with no model observation is timed at the procedure
    // clock when it was made.
    this.stManualAtT[index] = mark === "confirmed" ? (this.prev?.t ?? null) : null;
    this.publish();
  }

  ingest(sample: FrameSample): IngestOutcome {
    this.samples.push(sample);

    // --- elapsed + dwell, credited to the region we were *in* --------------
    // Elapsed is the sum of *credited* intervals rather than (last - first), so
    // that total time always equals the sum of per-region dwell. A seek would
    // otherwise inflate the total without crediting any region.
    if (this.prev !== null) {
      const dt = sample.t - this.prev.t;
      // A backwards jump (seek) or a long gap (seek forward, or a backpressure
      // stall) is a discontinuity: credit no time and re-anchor.
      if (dt >= 0 && dt <= MAX_SAMPLE_GAP_S) {
        this.elapsedS += dt;
        const region = this.prev.region;
        if (region) this.perRegionS[region] += dt;
        if (region && region === this.currentRegion) {
          this.currentRegionDwellS += dt;
        }
      } else {
        this.currentRegionDwellS = 0;
      }
    }

    // --- region transition --------------------------------------------------
    if (sample.region && sample.region !== this.currentRegion) {
      this.currentRegion = sample.region;
      this.currentRegionDwellS = 0;
      if (!this.visited.includes(sample.region)) {
        this.visited = [...this.visited, sample.region];
      }
    }

    // --- score folds --------------------------------------------------------
    this.scoreSum += sample.score;
    this.scoreN += 1;
    this.confSum += sample.scoreConfidence;
    this.recentMean =
      this.recentMean === null
        ? sample.score
        : RECENT_SCORE_ALPHA * sample.score +
          (1 - RECENT_SCORE_ALPHA) * this.recentMean;
    this.minScore =
      this.minScore === null || sample.score < this.minScore
        ? sample.score
        : this.minScore;
    this.maxScore =
      this.maxScore === null || sample.score > this.maxScore
        ? sample.score
        : this.maxScore;
    if (sample.region) {
      this.perRegionScoreSum[sample.region] += sample.score;
      this.perRegionScoreN[sample.region] += 1;
    }
    this.currentScore = sample.score;
    if (sample.expectedScore !== null) {
      this.expected =
        this.expected === null
          ? sample.expectedScore
          : 0.35 * sample.expectedScore + 0.65 * this.expected;
    }

    // --- settled display score ---------------------------------------------
    // Require N consecutive agreeing frames AND a minimum hold, so one
    // misclassified frame never reaches the ring.
    if (sample.score === this.candidateScore) {
      this.candidateRun += 1;
    } else {
      this.candidateScore = sample.score;
      this.candidateRun = 1;
    }
    const heldLongEnough =
      this.displayScore === null ||
      sample.t - this.displayScoreSetAtT >= SCORE_SETTLE_HOLD_S;
    if (
      this.candidateScore !== this.displayScore &&
      this.candidateRun >= SCORE_SETTLE_FRAMES &&
      heldLongEnough
    ) {
      this.displayScore = this.candidateScore;
      this.displayScoreSetAtT = sample.t;
    }

    // --- motion consensus ---------------------------------------------------
    this.motionRing.push(
      sample.motionConfidence >= MOTION_CONFIDENCE_THRESHOLD
        ? sample.motion
        : "stationary",
    );
    if (this.motionRing.length > MOTION_CONSENSUS_FRAMES) this.motionRing.shift();
    const first = this.motionRing[0];
    this.smoothedMotion =
      this.motionRing.length === MOTION_CONSENSUS_FRAMES &&
      first !== "stationary" &&
      this.motionRing.every((m) => m === first)
        ? first
        : "stationary";

    // --- feed ---------------------------------------------------------------
    this.latencyEma =
      this.latencyEma === null
        ? sample.processingTimeMs
        : 0.2 * sample.processingTimeMs + 0.8 * this.latencyEma;

    // --- ESGE stations ------------------------------------------------------
    // Last, and fenced: whatever happens here, the PEACE folds above are kept
    // and published.
    let stationEdges: StationKey[] = [];
    if (sample.landmarkStatus !== undefined) {
      try {
        stationEdges = this.foldStations(sample);
      } catch (err) {
        if (!this.stFoldFailed) {
          this.stFoldFailed = true;
          console.warn("Station tracking fold failed; PEACE continues", err);
        }
      }
    }

    this.prev = sample;
    this.publish();
    return { stationEdges };
  }

  /** O(10). Returns the stations whose effective "observed" just turned true. */
  private foldStations(sample: FrameSample): StationKey[] {
    const before = this.effectiveObserved();

    // Availability: on after one usable frame, off only after a run of bad ones.
    // uncertain / low_quality are the model working normally. A skipped frame
    // was never evaluated, so it says nothing about the source or the model and
    // must not touch the latch: with landmark_every_n >= 2 an unsupported source
    // alternates unsupported/skipped, and resetting here would never let it latch.
    const status = sample.landmarkStatus;
    if (status === "unsupported_layout" || status === "error") {
      this.stBadRun += 1;
      if (this.stBadRun >= STATION_AVAILABILITY_RELEASE_FRAMES) {
        this.stAvailability = status;
      }
    } else if (status !== "skipped") {
      this.stBadRun = 0;
      this.stAvailability = "ok";
    }

    if (sample.stationsDisplay !== undefined) this.stDisplay = sample.stationsDisplay;
    if (sample.landmarkModelVersion !== undefined) {
      this.stModelVersion = sample.landmarkModelVersion;
    }
    // An error frame may carry a blank tracker (no current, nothing
    // auto-enabled) when the backend could not recover its state; taking that at
    // face value would flash "manual" on every row. Keep the last good values.
    const trusted = status !== "error";
    if (trusted && sample.stationCurrent !== undefined) this.stCurrent = sample.stationCurrent;

    const incoming = sample.stationStatus;
    if (incoming && incoming.length === STATIONS_TOTAL) {
      for (let i = 0; i < STATIONS_TOTAL; i++) {
        // Sticky union: the model can promote a station, never demote one.
        if (this.stStatus[i] === "observed") continue;
        this.stStatus[i] = incoming[i];
        if (incoming[i] === "observed") this.stModelObservedAtT[i] = sample.t;
      }
    }
    const auto = sample.stationAutoEnabled;
    if (trusted && auto && auto.length === STATIONS_TOTAL) {
      for (let i = 0; i < STATIONS_TOTAL; i++) this.stAutoEnabled[i] = auto[i];
    }

    if (
      (this.stCurrent !== null && STATION_REGION[this.stCurrent] !== "esophagus") ||
      GASTRODUODENAL.some((k) => this.stStatus[STATION_INDEX[k]] === "observed")
    ) {
      this.stBeyondEsophagus = true;
    }

    const after = this.effectiveObserved();
    return STATION_ORDER.filter((_, i) => !before[i] && after[i]);
  }

  private effectiveObserved(): boolean[] {
    return this.stStatus.map((s, i) => isObserved(s, this.stManual[i]));
  }

  // --- snapshot -----------------------------------------------------------

  private publish(): void {
    this.rev += 1;
    this.snapshot = this.build(this.snapshot);
    for (const l of this.listeners) l();
  }

  private build(prev?: ProcedureSnapshot): ProcedureSnapshot {
    const totalElapsedS = this.elapsedS;

    const timers: TimersSlice = keepIfEqual(prev?.timers, {
      totalElapsedS,
      totalTargetS: this.totalTargetS,
      totalProgress: clamp01(totalElapsedS / this.totalTargetS),
      currentRegionDwellS: this.currentRegionDwellS,
      perRegionS: { ...this.perRegionS },
    });

    const regions: RegionsSlice = keepIfEqual(prev?.regions, {
      current: this.currentRegion,
      motion: this.smoothedMotion,
      visited: [...this.visited],
    });

    const fillByRegion = zeroPerRegion();
    for (const r of REGION_ORDER) {
      fillByRegion[r] = clamp01(this.perRegionS[r] / this.totalTargetS) * 100;
    }
    const coverage: CoverageSlice = keepIfEqual(prev?.coverage, {
      regionsVisited: this.visited.length,
      regionsTotal: REGION_ORDER.length,
      framesAnalysed: this.scoreN,
      fillByRegion,
    });

    const perRegionMean: Partial<Record<AnatomicalRegion, PeaceScore>> = {};
    for (const r of REGION_ORDER) {
      const n = this.perRegionScoreN[r];
      if (n > 0) {
        perRegionMean[r] = roundHalfToEven(
          this.perRegionScoreSum[r] / n,
        ) as PeaceScore;
      }
    }
    const meanConfidence = this.scoreN > 0 ? this.confSum / this.scoreN : null;
    const score: ScoreSlice = keepIfEqual(prev?.score, {
      current: this.currentScore,
      display: this.displayScore,
      displayLabel:
        this.displayScore === null ? null : PEACE_SCORE_LABELS[this.displayScore],
      meanScore: this.scoreN > 0 ? this.scoreSum / this.scoreN : null,
      recentMean: this.recentMean,
      min: this.minScore,
      max: this.maxScore,
      perRegionMean,
      meanConfidence,
      expected: this.expected,
    });

    const visibility: VisibilitySlice = keepIfEqual(
      prev?.visibility,
      deriveVisibility(this.recentMean, meanConfidence, this.smoothedMotion),
    );

    const feed: FeedSlice = keepIfEqual(prev?.feed, {
      framesReceived: this.scoreN,
      latencyMsEma: this.latencyEma,
      ended: this.ended,
    });

    const stations: StationsSlice = keepIfEqual(prev?.stations, this.buildStations());

    return {
      rev: this.rev,
      timers,
      regions,
      coverage,
      score,
      visibility,
      feed,
      stations,
    };
  }

  /** Always fresh arrays: keepIfEqual then decides, element-wise, whether to reuse. */
  private buildStations(): StationsSlice {
    const observed = this.effectiveObserved();
    const observedAtT = observed.map((o, i) => {
      if (!o) return null;
      const model = this.stModelObservedAtT[i];
      if (this.stManual[i] !== "confirmed") return model;
      // Confirmed: whichever came first, so a later model observation cannot
      // move the time the station first counted as observed.
      const manual = this.stManualAtT[i];
      if (model === null) return manual;
      if (manual === null) return model;
      return Math.min(model, manual);
    });
    const inEsophagus =
      this.stCurrent !== null && STATION_REGION[this.stCurrent] === "esophagus";
    return {
      availability: this.stAvailability,
      display: this.stDisplay,
      visible: this.stAvailability === "ok" && this.stDisplay,
      modelVersion: this.stModelVersion,
      current: this.stCurrent,
      status: [...this.stStatus],
      manual: [...this.stManual],
      observed,
      autoEnabled: [...this.stAutoEnabled],
      observedAtT,
      observedCount: observed.filter(Boolean).length,
      total: STATIONS_TOTAL,
      exitingMissing:
        inEsophagus && this.stBeyondEsophagus
          ? missingStations(observed, this.stAutoEnabled, GASTRODUODENAL)
          : [],
    };
  }
}

/**
 * Visibility readout.
 *
 * Deliberately NOT called "risk": the model scores mucosal cleanliness, not
 * pathology, and there is no patient context in this system. Reporting a "risk
 * level" would claim an assessment the software cannot make.
 */
export function deriveVisibility(
  recentMean: number | null,
  meanConfidence: number | null,
  motion: MotionDirection,
): VisibilitySlice {
  if (recentMean === null || meanConfidence === null) {
    return { level: null, detail: null };
  }
  if (meanConfidence < VISIBILITY_CONFIDENCE_GATE) {
    return { level: null, detail: "Low model confidence" };
  }

  let level: VisibilityLevel;
  if (recentMean >= 2.5) level = "adequate";
  else if (recentMean >= 1.5) level = "degraded";
  else level = "obscured";

  let detail: string | null = null;
  if (level !== "adequate" && motion === "withdrawal") {
    detail = "Withdrawing with impaired view";
  } else if (level === "obscured") {
    detail = "Mucosa substantially obscured";
  } else if (level === "degraded") {
    detail = "Fluid or foam limiting view";
  }
  return { level, detail };
}
