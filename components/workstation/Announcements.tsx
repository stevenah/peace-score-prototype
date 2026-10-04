"use client";

import { useEffect, useState } from "react";
import { useProcedureSlice } from "@/hooks/useProcedureMetrics";
import { STATION_ANNOUNCE_DEBOUNCE_MS } from "@/lib/live/config";
import type { ProcedureSnapshot } from "@/lib/live/procedure-types";
import { useCoachPrompt } from "./CoachingController";

/** A primitive, so the region re-renders only when the sentence changes. */
const selectStationSentence = (s: ProcedureSnapshot) =>
  s.stations.visible && s.stations.observedCount > 0
    ? `${s.stations.observedCount} of ${s.stations.total} stations observed`
    : "";

/**
 * Speaks `text` once it has held still for `delayMs`, so a burst of stations
 * observed in quick succession becomes one announcement, not a queue of them.
 */
function useSettled(text: string, delayMs: number): string {
  const [settled, setSettled] = useState("");
  useEffect(() => {
    const id = setTimeout(() => setSettled(text), delayMs);
    return () => clearTimeout(id);
  }, [text, delayMs]);
  return settled;
}

/**
 * Three always-mounted live regions: coach guidance (polite), coach alerts
 * (assertive), and the station count (polite, debounced). The count has its own
 * region so it can never clobber a coach prompt, or vice versa.
 *
 * Swapping aria-live on a single node is unreliable across screen readers, and so
 * is mounting a region and filling it in the same tick. Timers and the score ring
 * are deliberately NOT live — a screen reader announcing a ticking clock makes the
 * page unusable.
 */
export function Announcements() {
  const coach = useCoachPrompt();
  const isAlert = coach?.level === "alert";
  const stationSentence = useSettled(
    useProcedureSlice(selectStationSentence),
    STATION_ANNOUNCE_DEBOUNCE_MS,
  );

  return (
    <>
      <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
        {coach && !isAlert ? coach.text : ""}
      </div>
      <div role="alert" aria-atomic="true" className="sr-only">
        {isAlert ? coach.text : ""}
      </div>
      <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
        {stationSentence}
      </div>
    </>
  );
}
