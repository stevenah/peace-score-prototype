"use client";

import { useCallback, useMemo, useSyncExternalStore } from "react";
import { uploadStore } from "@/lib/upload-store";

export function useBatchUpload() {
  const items = useSyncExternalStore(
    uploadStore.subscribe,
    uploadStore.getSnapshot,
    uploadStore.getSnapshot,
  );

  const addFiles = useCallback((files: File[]) => {
    uploadStore.addFiles(files);
  }, []);

  const removeItem = useCallback((id: string) => {
    uploadStore.removeItem(id);
  }, []);

  const reset = useCallback(() => {
    uploadStore.reset();
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

  return useMemo(
    () => ({
      items,
      addFiles,
      removeItem,
      reset,
      hasActive,
      allComplete,
      completedCount,
      totalCount: items.length,
    }),
    [items, addFiles, removeItem, reset, hasActive, allComplete, completedCount],
  );
}
