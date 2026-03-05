/**
 * Module-level upload store that persists across route navigations.
 * React components subscribe via useSyncExternalStore in useBatchUpload.
 */
import { uploadVideo, getAnalysis } from "@/lib/api-client";
import type { BatchItem, AnalysisResponse } from "@/lib/types";

const POLL_INTERVAL = 2000;
const MAX_CONCURRENT_UPLOADS = 3;

// Progress phase weights
const UPLOAD_WEIGHT = 0.4;
const QUEUED_START = 0.4;
const QUEUED_END = 0.5;
const PROCESS_START = 0.5;

// --- Module-level state (survives component unmounts) ---
let items: BatchItem[] = [];
const listeners = new Set<() => void>();
const activeUploads = new Set<string>();
const pollIntervals = new Map<string, ReturnType<typeof setInterval>>();
const driftIntervals = new Map<string, ReturnType<typeof setInterval>>();
const xhrAbortFns = new Map<string, () => void>();

function notify() {
  for (const listener of listeners) {
    listener();
  }
}

function updateItem(id: string, patch: Partial<BatchItem>) {
  items = items.map((item) =>
    item.id === id ? { ...item, ...patch } : item,
  );
  notify();
}

// --- Drift (queued visual progress) ---

function stopDrift(itemId: string) {
  const interval = driftIntervals.get(itemId);
  if (interval) {
    clearInterval(interval);
    driftIntervals.delete(itemId);
  }
}

function startDrift(itemId: string) {
  if (driftIntervals.has(itemId)) return;
  const interval = setInterval(() => {
    const item = items.find((i) => i.id === itemId);
    if (!item) {
      stopDrift(itemId);
      return;
    }
    const remaining = QUEUED_END - item.progress;
    if (remaining <= 0.001) return;
    const step = remaining * 0.08;
    updateItem(itemId, { progress: item.progress + step });
  }, 200);
  driftIntervals.set(itemId, interval);
}

// --- Polling ---

function stopPolling(itemId: string) {
  const interval = pollIntervals.get(itemId);
  if (interval) {
    clearInterval(interval);
    pollIntervals.delete(itemId);
  }
}

function startPolling(itemId: string, analysisId: string) {
  if (pollIntervals.has(itemId)) return;

  const poll = async () => {
    try {
      const result: AnalysisResponse = await getAnalysis(analysisId);

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
        stopPolling(itemId);
      }
    } catch (e) {
      updateItem(itemId, {
        error: e instanceof Error ? e.message : "Polling failed",
      });
    }
  };

  poll();
  const interval = setInterval(poll, POLL_INTERVAL);
  pollIntervals.set(itemId, interval);
}

// --- Upload a single item ---

async function uploadItem(itemId: string, file: File) {
  updateItem(itemId, { status: "uploading", progress: 0 });

  try {
    const { promise, abort } = uploadVideo(file, (fraction) => {
      updateItem(itemId, { progress: fraction * UPLOAD_WEIGHT });
    });
    xhrAbortFns.set(itemId, abort);

    const result = await promise;
    xhrAbortFns.delete(itemId);

    updateItem(itemId, {
      status: "queued",
      analysisId: result.analysis_id,
      progress: QUEUED_START,
    });
    startDrift(itemId);
    startPolling(itemId, result.analysis_id);
  } catch (e) {
    xhrAbortFns.delete(itemId);
    updateItem(itemId, {
      status: "failed",
      error: e instanceof Error ? e.message : "Upload failed",
    });
  } finally {
    activeUploads.delete(itemId);
    schedule();
  }
}

// --- Scheduler ---

function schedule() {
  const slots = MAX_CONCURRENT_UPLOADS - activeUploads.size;
  if (slots <= 0) return;

  const pending = items.filter(
    (item) => item.status === "pending" && !activeUploads.has(item.id),
  );

  pending.slice(0, slots).forEach((item) => {
    activeUploads.add(item.id);
    uploadItem(item.id, item.file);
  });
}

// --- Public API ---

export const uploadStore = {
  getSnapshot(): BatchItem[] {
    return items;
  },

  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },

  addFiles(files: File[]) {
    const newItems: BatchItem[] = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      status: "pending",
      progress: 0,
      analysisId: null,
      analysis: null,
      error: null,
    }));
    items = [...items, ...newItems];
    notify();
    schedule();
  },

  removeItem(id: string) {
    // Abort active upload if any
    const abortFn = xhrAbortFns.get(id);
    if (abortFn) {
      abortFn();
      xhrAbortFns.delete(id);
    }
    activeUploads.delete(id);
    stopDrift(id);
    stopPolling(id);
    items = items.filter((item) => item.id !== id);
    notify();
  },

  reset() {
    // Abort all active uploads
    for (const [id, abortFn] of xhrAbortFns) {
      abortFn();
      xhrAbortFns.delete(id);
    }
    pollIntervals.forEach((interval) => clearInterval(interval));
    pollIntervals.clear();
    driftIntervals.forEach((interval) => clearInterval(interval));
    driftIntervals.clear();
    activeUploads.clear();
    items = [];
    notify();
  },
};
