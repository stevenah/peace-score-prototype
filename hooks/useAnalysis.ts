"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getAnalysis } from "@/lib/api-client";
import type { AnalysisResponse } from "@/lib/types";

const POLL_INTERVAL = 2000;

export function useAnalysis(analysisId: string | null) {
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const poll = useCallback(async () => {
    if (!analysisId) return;
    try {
      const result = await getAnalysis(analysisId);
      setData(result);

      // Stop polling when complete or failed
      if (result.status === "completed" || result.status === "failed") {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to fetch analysis";
      setError(message);
    }
  }, [analysisId]);

  useEffect(() => {
    if (!analysisId) {
      setData(null);
      return;
    }

    // Immediately fetch
    poll();

    // Start polling
    intervalRef.current = setInterval(poll, POLL_INTERVAL);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [analysisId, poll]);

  return { data, error };
}
