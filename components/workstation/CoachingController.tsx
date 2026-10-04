"use client";

import { createContext, useContext, useMemo, type ReactNode } from "react";
import { useCoaching } from "@/hooks/useCoaching";
import {
  selectRegions,
  selectScore,
  useProcedureSlice,
} from "@/hooks/useProcedureMetrics";
import type { CoachPrompt, ConnectionPhase } from "@/lib/live/coaching";
import type { ProcedureSnapshot } from "@/lib/live/procedure-types";
import { GASTRODUODENAL, missingStations } from "@/lib/live/stations";
import type { StationKey } from "@/lib/types";

/**
 * The missing-on-exit stations as a primitive, so this component re-renders only
 * when the set changes, not on every station update. Gated on visibility: in
 * shadow mode the coach never speaks about stations.
 */
const selectMissingKey = (s: ProcedureSnapshot) =>
  s.stations.visible ? s.stations.exitingMissing.join(",") : "";

/**
 * Stations 4-10 still unobserved wherever the scope is, likewise as a primitive:
 * it keeps a clearing "not yet observed" prompt truthful (lib/live/coaching.ts).
 */
const selectPendingKey = (s: ProcedureSnapshot) =>
  s.stations.visible
    ? missingStations(s.stations.observed, s.stations.autoEnabled, GASTRODUODENAL).join(",")
    : "";

function keysFrom(key: string): StationKey[] | undefined {
  return key ? (key.split(",") as StationKey[]) : undefined;
}

const CoachContext = createContext<CoachPrompt | null>(null);

export function useCoachPrompt(): CoachPrompt | null {
  return useContext(CoachContext);
}

/**
 * Subscribes to the store, runs the coaching state machine, and publishes the
 * result by context.
 *
 * `children` is passed as a prop rather than rendered inline by this component,
 * so when the store updates at 2Hz React reuses the identical children element
 * and skips that whole subtree — the video never re-renders because a score
 * changed. Only the two components that read the context re-render, and only when
 * the prompt itself changes.
 */
export function CoachingController({
  phase,
  children,
}: {
  phase: ConnectionPhase;
  children: ReactNode;
}) {
  const score = useProcedureSlice(selectScore);
  const regions = useProcedureSlice(selectRegions);
  const missingKey = useProcedureSlice(selectMissingKey);
  const pendingKey = useProcedureSlice(selectPendingKey);

  const inputs = useMemo(
    () => ({
      phase,
      recentMean: score.recentMean,
      motion: regions.motion,
      displayScore: score.display,
      stationsMissing: keysFrom(missingKey),
      // Nothing pending falls back to the exit list, which is then empty too.
      stationsPending: keysFrom(pendingKey),
    }),
    [phase, score.recentMean, score.display, regions.motion, missingKey, pendingKey],
  );

  const prompt = useCoaching(inputs);

  return (
    <CoachContext.Provider value={prompt}>{children}</CoachContext.Provider>
  );
}
