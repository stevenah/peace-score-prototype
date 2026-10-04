import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import {
  MARKS_RETRY_BASE_MS,
  MARKS_RETRY_MAX_MS,
  StationMarksSync,
  type StationMarksPatch,
} from "@/lib/live/station-marks-sync";
import type { ManualStationMark } from "@/lib/types";

/** A controllable server: each send parks until the test settles it. */
function harness() {
  let manual: ManualStationMark[] = Array(10).fill(null);
  const observedAtT: (number | null)[] = Array(10).fill(null);
  const calls: Array<{ id: string; patch: StationMarksPatch }> = [];
  const pending: Array<{ resolve: () => void; reject: () => void }> = [];
  const results: boolean[] = [];
  const sync = new StationMarksSync({
    read: () => ({ manual, observedAtT }),
    send: (id, patch) => {
      calls.push({ id, patch });
      return new Promise<void>((resolve, reject) =>
        pending.push({ resolve, reject: () => reject(new Error("503")) }),
      );
    },
    onResult: (ok) => results.push(ok),
  });
  return {
    sync,
    calls,
    results,
    mark(i: number, m: ManualStationMark) {
      manual = manual.map((v, j) => (j === i ? m : v));
      if (m === "confirmed") observedAtT[i] = 42;
    },
    async ok() {
      pending.shift()!.resolve();
      await vi.advanceTimersByTimeAsync(0);
    },
    async fail() {
      pending.shift()!.reject();
      await vi.advanceTimersByTimeAsync(0);
    },
  };
}

/** Lets queued PATCHes reach the server (sends run behind a promise chain). */
const flush = () => vi.advanceTimersByTimeAsync(0);

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe("StationMarksSync", () => {
  it("sends nothing before the save lands, nor when the save already carried the marks", async () => {
    const h = harness();
    h.mark(8, "confirmed");
    h.sync.sync();
    await flush();
    expect(h.calls).toHaveLength(0);
    h.sync.saved("live_1", [...Array(8).fill(null), "confirmed", null]);
    h.sync.sync();
    await flush();
    expect(h.calls).toHaveLength(0);
  });

  it("sends marks changed during the save, once, while a publish repeats", async () => {
    const h = harness();
    h.mark(8, "confirmed");
    h.sync.saved("live_1", Array(10).fill(null));
    await flush();
    expect(h.calls).toHaveLength(1);
    expect(h.calls[0].id).toBe("live_1");
    expect(h.calls[0].patch.manual[8]).toBe("confirmed");
    expect(h.calls[0].patch.observed_at_t[8]).toBe(42);
    h.sync.sync(); // publishes while in flight
    await flush();
    h.sync.sync();
    await flush();
    expect(h.calls).toHaveLength(1);
    await h.ok();
    h.sync.sync();
    await flush();
    expect(h.calls).toHaveLength(1);
    expect(h.results).toEqual([true]);
  });

  it("retries a failed PATCH on a backoff timer until it lands", async () => {
    const h = harness();
    h.sync.saved("live_1", Array(10).fill(null));
    await flush();
    h.mark(8, "confirmed");
    h.sync.sync();
    await flush();
    await h.fail();
    expect(h.results).toEqual([false]);

    // Publishes during the backoff do not hammer the server...
    h.sync.sync();
    await flush();
    expect(h.calls).toHaveLength(1);
    // ...the timer retries on its own, with no further publish needed.
    await vi.advanceTimersByTimeAsync(MARKS_RETRY_BASE_MS);
    expect(h.calls).toHaveLength(2);
    expect(h.calls[1].patch.manual[8]).toBe("confirmed");
    await h.fail();
    await vi.advanceTimersByTimeAsync(MARKS_RETRY_BASE_MS);
    expect(h.calls).toHaveLength(2); // doubled: not yet
    await vi.advanceTimersByTimeAsync(MARKS_RETRY_BASE_MS);
    expect(h.calls).toHaveLength(3);
    await h.ok();
    expect(h.results).toEqual([false, false, true]);

    await vi.advanceTimersByTimeAsync(MARKS_RETRY_MAX_MS);
    h.sync.sync();
    await flush();
    expect(h.calls).toHaveLength(3);
  });

  it("caps the backoff", async () => {
    const h = harness();
    h.sync.saved("live_1", Array(10).fill(null));
    await flush();
    h.mark(0, "rejected");
    h.sync.sync();
    await flush();
    for (let i = 0; i < 8; i++) {
      await h.fail();
      await vi.advanceTimersByTimeAsync(MARKS_RETRY_MAX_MS);
    }
    expect(h.calls).toHaveLength(9);
  });

  it("force (exit) sends at once, even while a retry is waiting", async () => {
    const h = harness();
    h.sync.saved("live_1", Array(10).fill(null));
    await flush();
    h.mark(8, "confirmed");
    h.sync.sync();
    await flush();
    await h.fail();
    h.mark(9, "confirmed");
    h.sync.sync(true);
    await flush();
    expect(h.calls).toHaveLength(2);
    expect(h.calls[1].patch.manual.slice(8)).toEqual(["confirmed", "confirmed"]);
    await h.ok();
    // The superseded retry does not fire a redundant PATCH.
    await vi.advanceTimersByTimeAsync(MARKS_RETRY_MAX_MS);
    expect(h.calls).toHaveLength(2);
  });

  it("serialises PATCHes, and a later success supersedes an earlier failure", async () => {
    const h = harness();
    h.sync.saved("live_1", Array(10).fill(null));
    await flush();
    h.mark(8, "confirmed");
    h.sync.sync();
    await flush();
    h.mark(8, "rejected");
    h.sync.sync();
    await flush();
    // The second waits for the first.
    expect(h.calls).toHaveLength(1);
    await h.fail();
    expect(h.calls).toHaveLength(2);
    expect(h.calls[1].patch.manual[8]).toBe("rejected");
    await h.ok();
    await vi.advanceTimersByTimeAsync(MARKS_RETRY_MAX_MS);
    expect(h.calls).toHaveLength(2);
  });

  it("stop cancels the retry; start re-arms it (StrictMode remount)", async () => {
    const h = harness();
    h.sync.saved("live_1", Array(10).fill(null));
    await flush();
    h.mark(8, "confirmed");
    h.sync.sync();
    await flush();
    await h.fail();
    h.sync.stop();
    await vi.advanceTimersByTimeAsync(MARKS_RETRY_MAX_MS);
    expect(h.calls).toHaveLength(1);

    h.sync.start();
    h.sync.sync();
    await flush();
    expect(h.calls).toHaveLength(2);
    await h.fail();
    await vi.advanceTimersByTimeAsync(MARKS_RETRY_MAX_MS);
    expect(h.calls).toHaveLength(3);
  });

  it("does nothing while stations were never available", async () => {
    const send = vi.fn();
    const sync = new StationMarksSync({ read: () => null, send });
    sync.saved("live_1", null);
    sync.sync(true);
    await flush();
    expect(send).not.toHaveBeenCalled();
  });
});
