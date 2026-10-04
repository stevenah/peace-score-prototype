import type { ManualStationMark } from "@/lib/types";

/** First retry delay after a failed PATCH; doubles per consecutive failure. */
export const MARKS_RETRY_BASE_MS = 1000;
export const MARKS_RETRY_MAX_MS = 30_000;

export interface StationMarksPatch {
  manual: ManualStationMark[];
  observed_at_t: (number | null)[];
}

export interface StationMarksSyncOptions {
  /** The marks as they stand now, or null while stations were never available. */
  read(): {
    readonly manual: readonly ManualStationMark[];
    readonly observedAtT: readonly (number | null)[];
  } | null;
  /** PATCH /api/analysis/live/[id]/stations. */
  send(analysisId: string, patch: StationMarksPatch): Promise<unknown>;
  /** Outcome of each PATCH, for the save indicator. */
  onResult?(ok: boolean): void;
  retryBaseMs?: number;
  retryMaxMs?: number;
}

function sameMarks(
  a: readonly ManualStationMark[],
  b: readonly ManualStationMark[],
): boolean {
  return a.length === b.length && a.every((m, i) => m === b[i]);
}

/**
 * Keeps the saved record's manual station marks in step with the checklist once
 * the procedure has been saved.
 *
 * The server's copy only advances when a PATCH succeeds. A failed PATCH is
 * retried on its own backoff timer, so an override made after the save survives
 * a transient 5xx or network blip instead of being dropped because the client
 * had already recorded it as sent. PATCHes are serialised, so a fast double
 * toggle cannot land out of order.
 */
export class StationMarksSync {
  private analysisId: string | null = null;
  /** As acknowledged by the server (the save, or the last good PATCH). */
  private persisted: readonly ManualStationMark[] | null = null;
  /** Queued or in flight, so a publish meanwhile does not send it again. */
  private sending: readonly ManualStationMark[] | null = null;
  private failures = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private chain: Promise<unknown> = Promise.resolve();
  /** While stopped (unmounted), failures are not retried. */
  private stopped = false;

  constructor(private readonly opts: StationMarksSyncOptions) {}

  /** The save landed carrying `marks`; from here on changes are PATCHed. */
  saved(analysisId: string, marks: readonly ManualStationMark[] | null): void {
    this.analysisId = analysisId;
    this.persisted = marks;
    // Overrides made while the save was in flight.
    this.sync();
  }

  /**
   * Sends the marks if they differ from what the server has or is being sent.
   * Cheap when nothing changed, so it can run on every store publish. While a
   * retry is scheduled only `force` (exit) sends early.
   */
  sync(force = false): void {
    const analysisId = this.analysisId;
    if (!analysisId) return;
    if (this.retryTimer !== null && !force) return;
    const now = this.opts.read();
    if (!now) return;
    const target = this.sending ?? this.persisted;
    if (target && sameMarks(target, now.manual)) return;
    this.clearRetry();

    const manual = [...now.manual];
    const patch = { manual, observed_at_t: [...now.observedAtT] };
    this.sending = manual;
    const settle = () => {
      if (this.sending === manual) this.sending = null;
    };
    this.chain = this.chain
      .then(() => this.opts.send(analysisId, patch))
      .then(
        () => {
          settle();
          this.persisted = manual;
          this.failures = 0;
          this.opts.onResult?.(true);
          // The server is back: anything held behind a retry goes now.
          if (this.retryTimer !== null) {
            this.clearRetry();
            this.sync();
          }
        },
        () => {
          settle();
          this.failures += 1;
          this.opts.onResult?.(false);
          this.scheduleRetry();
        },
      );
  }

  /** Allow retries again (a StrictMode remount stops, then starts). */
  start(): void {
    this.stopped = false;
  }

  /** Cancels any scheduled retry; a PATCH already in flight still lands. */
  stop(): void {
    this.stopped = true;
    this.clearRetry();
  }

  private scheduleRetry(): void {
    if (this.stopped) return;
    this.clearRetry();
    const base = this.opts.retryBaseMs ?? MARKS_RETRY_BASE_MS;
    const max = this.opts.retryMaxMs ?? MARKS_RETRY_MAX_MS;
    const delay = Math.min(max, base * 2 ** (this.failures - 1));
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.sync();
    }, delay);
  }

  private clearRetry(): void {
    if (this.retryTimer === null) return;
    clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }
}
