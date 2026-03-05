"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { uploadVideo, getAnalysis } from "@/lib/api-client";
import type { BatchItem, AnalysisResponse } from "@/lib/types";

const POLL_INTERVAL = 2000;
const MAX_CONCURRENT_UPLOADS = 3;

// Progress is split across phases:
//  Upload:     0% – 40%
//  Queued:     40% – 50%  (animated drift)
//  Processing: 50% – 100% (backend progress mapped)
const UPLOAD_WEIGHT = 0.4;
const QUEUED_START = 0.4;
const QUEUED_END = 0.5;
const PROCESS_START = 0.5;

export function useBatchUpload() {
  const [items, setItems] = useState<BatchItem[]>([]);
  const activeUploadsRef = useRef(new Set<string>());
  const pollIntervalsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(
    new Map(),
  );
  // Tracks animated "drift" progress for queued items
  const driftIntervalsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(
    new Map(),
  );

  const updateItem = useCallback((id: string, patch: Partial<BatchItem>) => {
    setItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...patch } : item)),
    );
  }, []);

  const stopDrift = useCallback((itemId: string) => {
    const interval = driftIntervalsRef.current.get(itemId);
    if (interval) {
      clearInterval(interval);
      driftIntervalsRef.current.delete(itemId);
    }
  }, []);

  // Slowly increment progress while queued to give visual feedback
  const startDrift = useCallback(
    (itemId: string) => {
      if (driftIntervalsRef.current.has(itemId)) return;

      const interval = setInterval(() => {
        setItems((prev) =>
          prev.map((item) => {
            if (item.id !== itemId) return item;
            // Drift from QUEUED_START toward QUEUED_END, decelerating
            const remaining = QUEUED_END - item.progress;
            if (remaining <= 0.001) return item;
            const step = remaining * 0.08; // ease-out: smaller steps as we approach target
            return { ...item, progress: item.progress + step };
          }),
        );
      }, 200);
      driftIntervalsRef.current.set(itemId, interval);
    },
    [],
  );

  const startPolling = useCallback(
    (itemId: string, analysisId: string) => {
      if (pollIntervalsRef.current.has(itemId)) return;

      const poll = async () => {
        try {
          const result: AnalysisResponse = await getAnalysis(analysisId);

          // Map backend progress (0-1) into our 50%-100% range
          const backendProgress = result.progress ?? 0;
          const mappedProgress =
            result.status === "completed"
              ? 1
              : result.status === "processing"
                ? PROCESS_START + backendProgress * (1 - PROCESS_START)
                : undefined;

          updateItem(itemId, {
            status: result.status,
            analysis: result,
            ...(mappedProgress !== undefined && { progress: mappedProgress }),
            error:
              result.status === "failed"
                ? result.error || "Analysis failed"
                : null,
          });

          if (result.status === "processing") {
            stopDrift(itemId);
          }

          if (result.status === "completed" || result.status === "failed") {
            stopDrift(itemId);
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
    [updateItem, stopDrift],
  );

  // Upload a single item (called concurrently for up to MAX_CONCURRENT_UPLOADS)
  const uploadItemRef = useRef<(id: string, file: File) => Promise<void>>(undefined);
  uploadItemRef.current = async (itemId: string, file: File) => {
    updateItem(itemId, { status: "uploading", progress: 0 });

    try {
      const result = await uploadVideo(file, (fraction) => {
        updateItem(itemId, { progress: fraction * UPLOAD_WEIGHT });
      });
      updateItem(itemId, {
        status: "queued",
        analysisId: result.analysis_id,
        progress: QUEUED_START,
      });
      startDrift(itemId);
      startPolling(itemId, result.analysis_id);
    } catch (e) {
      updateItem(itemId, {
        status: "failed",
        error: e instanceof Error ? e.message : "Upload failed",
      });
    } finally {
      activeUploadsRef.current.delete(itemId);
    }
  };

  // Schedule pending uploads up to concurrency limit
  const scheduleRef = useRef<() => void>(undefined);
  scheduleRef.current = () => {
    const slots = MAX_CONCURRENT_UPLOADS - activeUploadsRef.current.size;
    if (slots <= 0) return;

    const pending = items.filter(
      (item) =>
        item.status === "pending" && !activeUploadsRef.current.has(item.id),
    );

    pending.slice(0, slots).forEach((item) => {
      activeUploadsRef.current.add(item.id);
      uploadItemRef.current?.(item.id, item.file);
    });
  };

  // Trigger scheduling whenever items change
  useEffect(() => {
    scheduleRef.current?.();
  }, [items]);

  // Cleanup on unmount
  useEffect(() => {
    const polls = pollIntervalsRef.current;
    const drifts = driftIntervalsRef.current;
    return () => {
      polls.forEach((interval) => clearInterval(interval));
      polls.clear();
      drifts.forEach((interval) => clearInterval(interval));
      drifts.clear();
    };
  }, []);

  const addFiles = useCallback((files: File[]) => {
    const newItems: BatchItem[] = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      status: "pending",
      progress: 0,
      analysisId: null,
      analysis: null,
      error: null,
    }));
    setItems((prev) => [...prev, ...newItems]);
  }, []);

  const removeItem = useCallback(
    (id: string) => {
      setItems((prev) => prev.filter((item) => item.id !== id));
      stopDrift(id);
      activeUploadsRef.current.delete(id);
      const interval = pollIntervalsRef.current.get(id);
      if (interval) {
        clearInterval(interval);
        pollIntervalsRef.current.delete(id);
      }
    },
    [stopDrift],
  );

  const reset = useCallback(() => {
    pollIntervalsRef.current.forEach((interval) => clearInterval(interval));
    pollIntervalsRef.current.clear();
    driftIntervalsRef.current.forEach((interval) => clearInterval(interval));
    driftIntervalsRef.current.clear();
    activeUploadsRef.current.clear();
    setItems([]);
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
