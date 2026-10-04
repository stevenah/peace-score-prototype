import { z } from "zod";
import type {
  LandmarkResult,
  LandmarkStatus,
  LiveFrameResult,
  StationsSummary,
  StationStatus,
  TimelineEntry,
} from "@/lib/types";
import type { FrameSample, StationsSlice } from "./procedure-types";
import { STATION_ORDER, STATION_SCHEMA, STATIONS_TOTAL } from "./stations";

/**
 * Wire adapters for the live feed and for what a live analysis persists.
 *
 * Only the `landmark` sub-object is validated. PEACE fields flow into the
 * FrameSample exactly as they always have, so a malformed or unknown landmark
 * block can cost us the station checklist but never the cleanliness score.
 */

const StationKeySchema = z.enum(STATION_ORDER);
const StationStatusSchema = z.enum(["unseen", "candidate", "observed"] satisfies StationStatus[]);
const LandmarkStatusSchema = z.enum([
  "ok",
  "uncertain",
  "low_quality",
  "unsupported_layout",
  "skipped",
  "error",
] satisfies LandmarkStatus[]);

const perStation = <T extends z.ZodType>(item: T) => z.array(item).length(STATIONS_TOTAL);

/** esge10.v1, as in contracts/live_frame_result.v2.example.json. */
export const LandmarkResultSchema = z.object({
  schema: z.literal(STATION_SCHEMA),
  model_version: z.string(),
  display: z.boolean(),
  status: LandmarkStatusSchema,
  layout: z.string().nullish(),
  mode: z.string().nullish(),
  probs: perStation(z.number()).nullish(),
  top: StationKeySchema.nullish(),
  confidence: z.number().nullish(),
  quality: z.number().nullish(),
  current: StationKeySchema.nullable(),
  stations: perStation(StationStatusSchema),
  auto_enabled: perStation(z.boolean()),
  // Events are lossy by nature and nothing depends on them for correctness (the
  // store derives edges from the level arrays), so a missing list is not worth
  // dropping the frame over.
  events: z
    .object({
      observed: z.array(StationKeySchema).default([]),
      best_frame: z.array(StationKeySchema).default([]),
    })
    .default({ observed: [], best_frame: [] }),
  landmark_ms: z.number().nullish(),
});

type ParsedLandmark = z.infer<typeof LandmarkResultSchema>;

function normalize(p: ParsedLandmark): LandmarkResult {
  const out: LandmarkResult = {
    schema: STATION_SCHEMA,
    model_version: p.model_version,
    display: p.display,
    status: p.status,
    current: p.current,
    stations: p.stations,
    auto_enabled: p.auto_enabled,
    events: { observed: p.events.observed, best_frame: p.events.best_frame },
  };
  // Optional fields are omitted rather than set to undefined/null.
  if (p.layout != null) out.layout = p.layout;
  if (p.mode != null) out.mode = p.mode;
  if (p.probs != null) out.probs = p.probs;
  if (p.top != null) out.top = p.top;
  if (p.confidence != null) out.confidence = p.confidence;
  if (p.quality != null) out.quality = p.quality;
  if (p.landmark_ms != null) out.landmark_ms = p.landmark_ms;
  return out;
}

export type LandmarkParser = (raw: unknown) => LandmarkResult | undefined;

/**
 * A parser that warns at most once for its lifetime — create one per procedure
 * run. At 2 Hz a per-frame warning would bury the console within seconds.
 */
export function createLandmarkParser(
  warn: (message: string, detail?: unknown) => void = (m, d) => console.warn(m, d),
): LandmarkParser {
  let warned = false;
  const warnOnce = (message: string, detail?: unknown) => {
    if (warned) return;
    warned = true;
    warn(message, detail);
  };

  return (raw) => {
    // Feature off: the key is simply absent. Nothing to say.
    if (raw === undefined || raw === null) return undefined;
    if (
      typeof raw === "object" &&
      "schema" in raw &&
      (raw as { schema: unknown }).schema !== STATION_SCHEMA
    ) {
      warnOnce(
        `Ignoring landmark block with schema ${JSON.stringify((raw as { schema: unknown }).schema)}; expected ${STATION_SCHEMA}`,
      );
      return undefined;
    }
    const parsed = LandmarkResultSchema.safeParse(raw);
    if (!parsed.success) {
      warnOnce(
        "Dropping malformed landmark block; PEACE results are unaffected",
        parsed.error.issues,
      );
      return undefined;
    }
    return normalize(parsed.data);
  };
}

/** Module-wide parser (warns once per page load). Prefer a per-run instance. */
export const parseLandmark: LandmarkParser = createLandmarkParser();

/** sum(i * p(i)) over the PEACE class distribution, when the backend sends one. */
export function expectedFromProbs(probs: number[] | undefined): number | null {
  if (!probs || probs.length === 0) return null;
  const total = probs.reduce((a, b) => a + b, 0);
  if (total <= 0) return null;
  return probs.reduce((acc, p, i) => acc + i * p, 0) / total;
}

/** The optional FrameSample fields carried by one landmark block. */
export function landmarkSampleFields(lm: LandmarkResult) {
  return {
    landmarkStatus: lm.status,
    landmarkTop: lm.top ?? null,
    landmarkConfidence: lm.confidence ?? null,
    landmarkQuality: lm.quality ?? null,
    landmarkModelVersion: lm.model_version,
    stationsDisplay: lm.display,
    stationCurrent: lm.current,
    stationStatus: lm.stations,
    stationAutoEnabled: lm.auto_enabled,
    stationObservedEvents: lm.events.observed,
    stationBestEvents: lm.events.best_frame,
  } satisfies Partial<FrameSample>;
}

/**
 * The single wire -> sample adapter.
 *
 * `ticket` is the client-side send ticket this result was paired with (see
 * hooks/useLiveFeed), which is what thumbnails and gallery frames are keyed by.
 * The landmark mapping is fenced so that it can never cost us the PEACE sample.
 */
export function frameSampleFromResult(
  result: LiveFrameResult,
  videoTime: number,
  ticket: number,
  parse: LandmarkParser = parseLandmark,
): FrameSample {
  const sample: FrameSample = {
    seq: ticket,
    t: videoTime,
    score: result.peace_score.score,
    scoreConfidence: result.peace_score.confidence,
    expectedScore: expectedFromProbs(result.peace_score.probs),
    region: result.region ?? null,
    motion: result.motion?.direction ?? "stationary",
    motionConfidence: result.motion?.confidence ?? 0,
    frameIndex: result.frame_index,
    processingTimeMs: result.processing_time_ms,
  };
  if (result.landmark === undefined) return sample;
  try {
    const lm = parse(result.landmark);
    return lm ? { ...sample, ...landmarkSampleFields(lm) } : sample;
  } catch {
    return sample;
  }
}

// --- persistence ------------------------------------------------------------

/**
 * One saved timeline entry. Landmark fields are written only for frames that
 * carried a landmark block, so a feature-off timeline is unchanged byte for byte.
 */
export function timelineEntryFromSample(s: FrameSample): TimelineEntry {
  const entry: TimelineEntry = {
    timestamp: s.t,
    frame_index: s.frameIndex,
    motion: s.motion,
    region: s.region ?? "stomach",
    peace_score: s.score,
    confidence: s.scoreConfidence,
  };
  if (s.landmarkStatus === undefined) return entry;
  entry.landmark_top = s.landmarkTop ?? null;
  entry.landmark_confidence = s.landmarkConfidence ?? null;
  entry.landmark_status = s.landmarkStatus;
  entry.station_current = s.stationCurrent ?? null;
  const observed = s.stationObservedEvents ?? [];
  const best = s.stationBestEvents ?? [];
  if (observed.length > 0 || best.length > 0) {
    entry.station_events = { observed: [...observed], best_frame: [...best] };
  }
  return entry;
}

/**
 * What AnalysisSession.stationsData stores. Undefined while stations were never
 * available (feature off), so such a save is identical to one made before the
 * feature existed. Shadow-mode sessions ARE saved — that is the point of them.
 */
export function stationsSummary(s: StationsSlice): StationsSummary | undefined {
  if (s.availability === "absent") return undefined;
  return {
    schema: STATION_SCHEMA,
    model_version: s.modelVersion,
    display: s.display,
    availability: s.availability,
    status: [...s.status],
    manual: [...s.manual],
    auto_enabled: [...s.autoEnabled],
    observed_at_t: [...s.observedAtT],
  };
}

/** Body of PATCH /api/analysis/live/[id]/stations: overrides made after the save. */
export const StationsPatchSchema = z.object({
  manual: perStation(z.enum(["confirmed", "rejected"]).nullable()),
  observed_at_t: perStation(z.number().nonnegative().nullable()),
});

export type StationsPatch = z.infer<typeof StationsPatchSchema>;

/** A saved stations summary, as read back from the database. Tolerant of extras. */
export const StationsSummarySchema = z.object({
  schema: z.literal(STATION_SCHEMA),
  model_version: z.string().nullable(),
  display: z.boolean(),
  availability: z.enum(["absent", "ok", "unsupported_layout", "error"]),
  status: perStation(StationStatusSchema),
  manual: perStation(z.enum(["confirmed", "rejected"]).nullable()),
  auto_enabled: perStation(z.boolean()),
  observed_at_t: perStation(z.number().nullable()),
});

/** A stored AnalysisSession.stationsData, or null when absent or unreadable. */
export function parseStationsData(raw: string | null | undefined): StationsSummary | null {
  if (!raw) return null;
  try {
    const parsed = StationsSummarySchema.safeParse(JSON.parse(raw));
    return parsed.success ? parsed.data : null;
  } catch {
    return null;
  }
}
