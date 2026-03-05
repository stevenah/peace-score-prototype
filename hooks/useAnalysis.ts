"use client";

import { useCallback, useState } from "react";
import { getAnalysis } from "@/lib/api-client";
import { usePolling } from "./usePolling";
import type { AnalysisResponse } from "@/lib/types";

export function useAnalysis(analysisId: string | null) {
  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isTerminal = data?.status === "completed" || data?.status === "failed";

  const poll = useCallback(async () => {
    if (!analysisId) return;
    try {
      const result = await getAnalysis(analysisId);
      setData(result);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to fetch analysis";
      setError(message);
    }
  }, [analysisId]);

  usePolling(poll, {
    interval: 2000,
    enabled: !!analysisId && !isTerminal,
  });

  return { data, error };
}
