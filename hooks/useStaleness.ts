"use client";

import { useEffect, useState } from "react";
import { STALE_AFTER_MS } from "@/lib/live/config";

/**
 * True when the analysis feed has gone quiet while we are supposedly streaming.
 *
 * Without this the workstation keeps showing the last frame's numbers with the
 * timers still climbing after the socket has died — an instrument asserting
 * liveness it does not have. That is a worse failure than showing nothing.
 */
export function useStaleness(
  lastResultAtMs: number | null,
  active: boolean,
  timeoutMs: number = STALE_AFTER_MS,
): boolean {
  const [stale, setStale] = useState(false);

  // Evaluated only from the interval, never synchronously in the effect body:
  // a 400ms granularity is immaterial against a multi-second staleness threshold.
  useEffect(() => {
    const evaluate = () => {
      if (!active || lastResultAtMs === null) {
        setStale(false);
        return;
      }
      setStale(Date.now() - lastResultAtMs > timeoutMs);
    };
    const id = setInterval(evaluate, 400);
    return () => clearInterval(id);
  }, [lastResultAtMs, active, timeoutMs]);

  return stale;
}
