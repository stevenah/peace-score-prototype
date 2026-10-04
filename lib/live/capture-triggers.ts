import { FILMSTRIP_MIN_GAP_S } from "./config";
import type { FrameSample, StationsSlice } from "./procedure-types";
import type { AnatomicalRegion, PeaceScore, StationKey } from "@/lib/types";

export type CaptureReason =
  | "station-observed"
  | "region-entry"
  | "quality-alert"
  | "best-in-region"
  | "detection";

/** Store-derived context for the frame being evaluated. */
export interface CaptureContext {
  /** The station slice AFTER this frame was ingested. Its `visible` is the gate. */
  readonly stations?: Pick<StationsSlice, "visible">;
  /** ProcedureStore.ingest's level-derived observed edges for this frame. */
  readonly stationEdges?: readonly StationKey[];
}

export interface TriggerState {
  readonly lastCaptureT: number | null;
  readonly lastRegion: AnatomicalRegion | null;
  /**
   * Whether the current alert episode has already produced a thumbnail.
   *
   * Deliberately not a plain "were we in an alert last frame" edge: the edge can
   * land inside the cooldown, and a pure edge would then be swallowed and the
   * whole episode would yield nothing. This latches only on an actual capture and
   * clears when the condition does, so every episode yields exactly one.
   */
  readonly alertCaptured: boolean;
  readonly bestByRegion: Readonly<Partial<Record<AnatomicalRegion, PeaceScore>>>;
}

export function initialTriggerState(): TriggerState {
  return {
    lastCaptureT: null,
    lastRegion: null,
    alertCaptured: false,
    bestByRegion: {},
  };
}

/**
 * Decides whether a frame is worth keeping in the filmstrip.
 *
 * All triggers are edge-triggered and share one cooldown, so a bad stretch yields
 * a single thumbnail rather than one every 500ms. There is deliberately no
 * "detection" trigger firing today — the model has no detection head.
 */
export function evaluateCapture(
  state: TriggerState,
  sample: FrameSample,
  ctx: CaptureContext = {},
): { reason: CaptureReason | null; station: StationKey | null; next: TriggerState } {
  // A station just became observed on this frame. Gated on visibility, so shadow
  // mode never produces a thumbnail the clinician did not ask for. When several
  // edges land together, prefer the one this frame actually shows.
  const edges = ctx.stations?.visible ? (ctx.stationEdges ?? []) : [];
  const station =
    edges.length === 0
      ? null
      : sample.landmarkTop && edges.includes(sample.landmarkTop)
        ? sample.landmarkTop
        : edges[0];

  const region = sample.region;
  const enteredRegion = region !== null && region !== state.lastRegion;
  const isAlert = sample.motion === "withdrawal" && sample.score < 2;
  const alertWanted = isAlert && !state.alertCaptured;

  const best = region ? state.bestByRegion[region] : undefined;
  const isBestInRegion =
    region !== null && best !== undefined && sample.score > best;

  const cooledDown =
    state.lastCaptureT === null ||
    sample.t - state.lastCaptureT >= FILMSTRIP_MIN_GAP_S;

  let reason: CaptureReason | null = null;
  // Priority: a newly observed station beats a new region beats an alert beats a
  // new best.
  if (station) reason = "station-observed";
  else if (enteredRegion) reason = "region-entry";
  else if (alertWanted) reason = "quality-alert";
  else if (isBestInRegion) reason = "best-in-region";

  // Region entry and alerts are the interesting events, so they bypass nothing —
  // but everything still respects the shared cooldown except the very first
  // frame of a new region, which is the whole point of the strip. A station
  // observation happens once per station per procedure, so it bypasses too.
  const allowed =
    reason === "region-entry" || reason === "station-observed" ? true : cooledDown;
  if (!allowed) reason = null;

  const next: TriggerState = {
    lastCaptureT: reason ? sample.t : state.lastCaptureT,
    lastRegion: region ?? state.lastRegion,
    // Latches on capture, clears when the condition clears.
    alertCaptured: isAlert
      ? state.alertCaptured || reason === "quality-alert"
      : false,
    bestByRegion: region
      ? {
          ...state.bestByRegion,
          [region]:
            best === undefined || sample.score > best ? sample.score : best,
        }
      : state.bestByRegion,
  };

  return { reason, station: reason === "station-observed" ? station : null, next };
}
