"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { uploadVideo, getAnalysis } from "@/lib/api-client";
import type { BatchItem, AnalysisResponse } from "@/lib/types";

const POLL_INTERVAL = 2000;

export function useBatchUpload() {
  const [items, setItems] = useState<BatchItem[]>([]);
  const isUploadingRef = useRef(false);
  const pollIntervalsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(
    new Map(),
  );

  const updateItem = useCallback((id: string, patch: Partial<BatchItem>) => {
    setItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...patch } : item)),
    );
  }, []);

  const startPolling = useCallback(
    (itemId: string, analysisId: string) => {
      if (pollIntervalsRef.current.has(itemId)) return;

      const poll = async () => {
        try {
          const result: AnalysisResponse = await getAnalysis(analysisId);
          updateItem(itemId, {
            status: result.status,
            analysis: result,
            error:
              result.status === "failed"
                ? result.error || "Analysis failed"
                : null,
          });

          if (result.status === "completed" || result.status === "failed") {
            const interval = pollIntervalsRef.current.get(itemId);
            if (interval) {
              clearInterval(interval);
              pollIntervalsRef.current.delete(itemId);
            }
          }
        } catch (e) {
          updateItem(itemId, {
            error: e instanceof Error ? e.message : "Polling failed",
          });
        }
      };

      poll();
      const interval = setInterval(poll, POLL_INTERVAL);
      pollIntervalsRef.current.set(itemId, interval);
    },
    [updateItem],
  );

  // Sequential upload processor
  const processNextRef = useRef<() => void>(undefined);
  processNextRef.current = async () => {
    if (isUploadingRef.current) return;

    const nextPending = items.find((item) => item.status === "pending");
    if (!nextPending) return;

    isUploadingRef.current = true;
    updateItem(nextPending.id, { status: "uploading" });

    try {
      const result = await uploadVideo(nextPending.file);
      updateItem(nextPending.id, {
        status: "queued",
        analysisId: result.analysis_id,
      });
      startPolling(nextPending.id, result.analysis_id);
    } catch (e) {
      updateItem(nextPending.id, {
        status: "failed",
        error: e instanceof Error ? e.message : "Upload failed",
      });
    } finally {
      isUploadingRef.current = false;
    }
  };

  // Trigger upload processing whenever items change
  useEffect(() => {
    processNextRef.current?.();
  }, [items]);

  // Cleanup polling intervals on unmount
  useEffect(() => {
    return () => {
      pollIntervalsRef.current.forEach((interval) => clearInterval(interval));
      pollIntervalsRef.current.clear();
    };
  }, []);

  const addFiles = useCallback((files: File[]) => {
    const newItems: BatchItem[] = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      status: "pending",
      analysisId: null,
      analysis: null,
      error: null,
    }));
    setItems((prev) => [...prev, ...newItems]);
  }, []);

  const removeItem = useCallback((id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
    const interval = pollIntervalsRef.current.get(id);
    if (interval) {
      clearInterval(interval);
      pollIntervalsRef.current.delete(id);
    }
  }, []);

  const reset = useCallback(() => {
    pollIntervalsRef.current.forEach((interval) => clearInterval(interval));
    pollIntervalsRef.current.clear();
    setItems([]);
    isUploadingRef.current = false;
  }, []);

  const hasActive = items.some(
    (item) => item.status !== "completed" && item.status !== "failed",
  );
  const allComplete =
    items.length > 0 &&
    items.every(
      (item) => item.status === "completed" || item.status === "failed",
    );
  const completedCount = items.filter(
    (item) => item.status === "completed",
  ).length;

  return {
    items,
    addFiles,
    removeItem,
    reset,
    hasActive,
    allComplete,
    completedCount,
    totalCount: items.length,
  };
}
