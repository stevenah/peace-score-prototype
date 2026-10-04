import type { MotionDirection, PeaceScore, StationKey } from "@/lib/types";
import { ALERT_TIMING, COACH_TIMING } from "./config";
import { STATION_SHORT } from "./stations";

export type CoachLevel = "info" | "guide" | "warn" | "alert";

export interface CoachPrompt {
  readonly id: string;
  readonly level: CoachLevel;
  readonly text: string;
}

export type ConnectionPhase =
  | "idle"
  | "connecting"
  | "ready"
  | "streaming"
  | "stalled"
  | "disconnected"
  | "error"
  | "ended";

export interface CoachInputs {
  readonly phase: ConnectionPhase;
  readonly recentMean: number | null;
  readonly motion: MotionDirection;
  readonly displayScore: PeaceScore | null;
  /**
   * Stations still missing on the way out (StationsSlice.exitingMissing). The
   * caller passes it only when stations are visible; absent = no station rules.
   */
  readonly stationsMissing?: readonly StationKey[];
  /**
   * Auto-enabled stations 4-10 still unobserved, wherever the scope is. Read only
   * while a stations-missing prompt clears, to keep its wording true: the exit
   * list also empties when the tracker merely loses its current station, which
   * is no reason to drop the prompt at once. Absent = fall back to
   * stationsMissing.
   */
  readonly stationsPending?: readonly StationKey[];
}

export interface CoachState {
  readonly active: CoachPrompt | null;
  readonly activeSinceMs: number;
  readonly candidateId: string | null;
  readonly candidateRun: number;
  /** When the active prompt's condition first stopped holding. */
  readonly clearingSinceMs: number | null;
  readonly lastChangeMs: number;
  /** id -> wall time at which it was last dismissed, for REPEAT_MS. */
  readonly lastDismissedMs: Readonly<Record<string, number>>;
}

export function initialCoachState(): CoachState {
  return {
    active: null,
    activeSinceMs: 0,
    candidateId: null,
    candidateRun: 0,
    clearingSinceMs: null,
    lastChangeMs: Number.NEGATIVE_INFINITY,
    lastDismissedMs: {},
  };
}

/**
 * Connection-phase prompts are reported as-is: they describe a state we know
 * exactly, so they need no hysteresis. Data-derived prompts do — the score is
 * argmax of a smoothed distribution and still crosses the 1<->2 boundary
 * repeatedly when the truth sits near 1.5, and motion is not smoothed by the
 * backend at all.
 */
interface Rule {
  readonly id: string;
  readonly level: CoachLevel;
  /** A function for prompts whose wording tracks the inputs while shown. */
  readonly text: string | ((i: CoachInputs) => string);
  /**
   * Wording while the prompt clears, or null when nothing is left to say — the
   * prompt then goes at once rather than repeating a stale claim through the
   * clear window. Without it, a clearing prompt keeps its last wording.
   */
  readonly clearingText?: (i: CoachInputs) => string | null;
  readonly instant: boolean;
  test(i: CoachInputs): boolean;
}

function ruleText(rule: Rule, inputs: CoachInputs): string {
  return typeof rule.text === "function" ? rule.text(inputs) : rule.text;
}

/** "Not yet observed: Lesser curve, Incisura (+1)". */
export function missingStationsText(missing: readonly StationKey[]): string {
  const shown = missing.slice(0, 2).map((k) => STATION_SHORT[k]).join(", ");
  const more = missing.length - 2;
  return `Not yet observed: ${shown}${more > 0 ? ` (+${more})` : ""}`;
}

const RULES: readonly Rule[] = [
  {
    id: "error",
    level: "alert",
    text: "Analysis unavailable",
    instant: true,
    test: (i) => i.phase === "error",
  },
  {
    id: "dirty-withdrawal",
    level: "alert",
    text: "Insufficient cleaning — withdrawing",
    instant: false,
    test: (i) =>
      i.phase === "streaming" &&
      i.motion === "withdrawal" &&
      i.displayScore !== null &&
      i.displayScore < 2,
  },
  {
    id: "disconnected",
    level: "warn",
    text: "Connection lost — analysis stopped",
    instant: true,
    test: (i) => i.phase === "disconnected",
  },
  {
    id: "stalled",
    level: "warn",
    text: "Analysis stalled — no results from the server",
    instant: true,
    test: (i) => i.phase === "stalled",
  },
  {
    // Fires only on the way OUT: back in the esophagus after reaching the stomach
    // or duodenum, with auto-enabled stations unobserved. That is the last moment
    // the operator can still act on it, and it can never fire during insertion.
    id: "stations-missing",
    level: "guide",
    text: (i) => missingStationsText(i.stationsMissing ?? []),
    // Once the last station is observed the prompt would name one that has been
    // seen; drop it at once instead.
    clearingText: (i) => {
      const left = i.stationsPending ?? i.stationsMissing ?? [];
      return left.length > 0 ? missingStationsText(left) : null;
    },
    instant: false,
    test: (i) =>
      i.phase === "streaming" && (i.stationsMissing?.length ?? 0) > 0,
  },
  {
    id: "connecting",
    level: "info",
    text: "Connecting to analysis server…",
    instant: true,
    test: (i) => i.phase === "connecting",
  },
  {
    id: "ready",
    level: "guide",
    text: "Ready — press play to begin",
    instant: true,
    test: (i) => i.phase === "ready",
  },
  {
    id: "ended",
    level: "info",
    text: "Procedure ended",
    instant: true,
    test: (i) => i.phase === "ended",
  },
  {
    id: "obscured",
    level: "guide",
    text: "View obscured — consider washing and re-inspecting",
    instant: false,
    test: (i) =>
      i.phase === "streaming" && i.recentMean !== null && i.recentMean < 1.5,
  },
];

function timingFor(level: CoachLevel) {
  return level === "alert" ? ALERT_TIMING : COACH_TIMING;
}

function dismiss(
  prev: CoachState,
  active: CoachPrompt,
  candidateId: string | null,
  candidateRun: number,
  nowMs: number,
): CoachState {
  return {
    ...prev,
    active: null,
    candidateId,
    candidateRun,
    clearingSinceMs: null,
    lastChangeMs: nowMs,
    lastDismissedMs: { ...prev.lastDismissedMs, [active.id]: nowMs },
  };
}

/**
 * Pure state machine. Call once per input change with a wall clock.
 *
 * Arm slowly, clear slowly, and arm alerts faster than guidance. The asymmetric
 * arm/clear windows are what stop a condition oscillating around a threshold
 * from producing a strobe rather than a single message.
 */
export function nextCoachState(
  prev: CoachState,
  inputs: CoachInputs,
  nowMs: number,
): CoachState {
  const firing = RULES.find((r) => r.test(inputs)) ?? null;

  // --- track how long the candidate has been asking to be shown -----------
  const candidateId = firing?.id ?? null;
  const candidateRun =
    candidateId !== null && candidateId === prev.candidateId
      ? prev.candidateRun + 1
      : candidateId === null
        ? 0
        : 1;

  const active = prev.active;

  // --- the active prompt still holds --------------------------------------
  // Its wording may have moved with the inputs (e.g. a missing station has since
  // been observed); refresh it without resetting its dwell.
  if (active && firing && firing.id === active.id) {
    const text = ruleText(firing, inputs);
    return {
      ...prev,
      active: text === active.text ? active : { ...active, text },
      candidateId,
      candidateRun,
      clearingSinceMs: null,
    };
  }

  // --- the active prompt no longer holds: clear it, slowly ----------------
  // ...unless it has nothing true left to say, which clears it at once.
  if (active) {
    const clearing = RULES.find((r) => r.id === active.id)?.clearingText;
    const text = clearing ? clearing(inputs) : active.text;
    if (text === null) return dismiss(prev, active, candidateId, candidateRun, nowMs);
    const shown = text === active.text ? active : { ...active, text };

    const timing = timingFor(active.level);
    const clearingSinceMs = prev.clearingSinceMs ?? nowMs;
    const heldLongEnough = nowMs - prev.activeSinceMs >= timing.MIN_DWELL_MS;
    const clearedLongEnough = nowMs - clearingSinceMs >= timing.CLEAR_MS;

    if (!heldLongEnough || !clearedLongEnough) {
      return { ...prev, active: shown, candidateId, candidateRun, clearingSinceMs };
    }
    return dismiss(prev, active, candidateId, candidateRun, nowMs);
  }

  // --- nothing active: consider promoting the candidate -------------------
  if (!firing) {
    return { ...prev, candidateId, candidateRun, clearingSinceMs: null };
  }

  const timing = timingFor(firing.level);

  if (!firing.instant && candidateRun < timing.ARM_FRAMES) {
    return { ...prev, candidateId, candidateRun, clearingSinceMs: null };
  }
  if (
    timing.RATE_LIMIT_MS > 0 &&
    nowMs - prev.lastChangeMs < timing.RATE_LIMIT_MS
  ) {
    return { ...prev, candidateId, candidateRun, clearingSinceMs: null };
  }
  const dismissedAt = prev.lastDismissedMs[firing.id];
  if (
    timing.REPEAT_MS > 0 &&
    dismissedAt !== undefined &&
    nowMs - dismissedAt < timing.REPEAT_MS
  ) {
    return { ...prev, candidateId, candidateRun, clearingSinceMs: null };
  }

  return {
    ...prev,
    active: { id: firing.id, level: firing.level, text: ruleText(firing, inputs) },
    activeSinceMs: nowMs,
    candidateId,
    candidateRun,
    clearingSinceMs: null,
    lastChangeMs: nowMs,
  };
}
