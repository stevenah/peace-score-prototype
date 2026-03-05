"use client";

import { useCallback, useEffect, useRef } from "react";

interface UsePollingOptions {
  /** Initial delay between polls in ms */
  interval: number;
  /** If true, increase delay after each poll (1.5x, capped at maxInterval) */
  backoff?: boolean;
  /** Max delay in ms when backoff is enabled (default: 30000) */
  maxInterval?: number;
  /** Whether polling is active */
  enabled: boolean;
}

/**
 * Shared polling hook with optional exponential backoff.
 * Calls `fn` immediately when enabled, then repeats on a timer.
 * Stops when `enabled` becomes false or the component unmounts.
 */
export function usePolling(fn: () => Promise<void>, options: UsePollingOptions) {
  const { interval, backoff = false, maxInterval = 30000, enabled } = options;
  const fnRef = useRef(fn);

  useEffect(() => {
    fnRef.current = fn;
  });

  const poll = useCallback(() => {
    return fnRef.current();
  }, []);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let delay = interval;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      if (cancelled) return;
      await poll();
      if (cancelled) return;
      if (backoff) {
        delay = Math.min(delay * 1.5, maxInterval);
      }
      timer = setTimeout(tick, delay);
    };

    // Immediate first call, then schedule
    poll().then(() => {
      if (!cancelled) {
        timer = setTimeout(tick, delay);
      }
    });

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [enabled, interval, backoff, maxInterval, poll]);
}
