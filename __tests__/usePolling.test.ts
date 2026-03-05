import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { usePolling } from "@/hooks/usePolling";

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => {
  vi.useRealTimers();
});

describe("usePolling", () => {
  it("calls fn immediately when enabled", async () => {
    const fn = vi.fn().mockResolvedValue(undefined);

    renderHook(() => usePolling(fn, { interval: 1000, enabled: true }));

    // Flush the immediate call (microtask)
    await act(async () => {});

    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("does not call fn when disabled", async () => {
    const fn = vi.fn().mockResolvedValue(undefined);

    renderHook(() => usePolling(fn, { interval: 1000, enabled: false }));
    await act(async () => {});

    expect(fn).not.toHaveBeenCalled();
  });

  it("calls fn on interval after initial call", async () => {
    const fn = vi.fn().mockResolvedValue(undefined);

    renderHook(() => usePolling(fn, { interval: 1000, enabled: true }));

    // Flush immediate call
    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(1);

    // Advance one interval
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    // Flush the tick's promise
    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("stops polling when enabled becomes false", async () => {
    const fn = vi.fn().mockResolvedValue(undefined);

    const { rerender } = renderHook(
      ({ enabled }) => usePolling(fn, { interval: 1000, enabled }),
      { initialProps: { enabled: true } },
    );

    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(1);

    // Disable polling
    rerender({ enabled: false });

    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    await act(async () => {});

    // Should still be 1 (no more calls after disable)
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("applies exponential backoff when enabled", async () => {
    const fn = vi.fn().mockResolvedValue(undefined);

    renderHook(() =>
      usePolling(fn, {
        interval: 1000,
        backoff: true,
        maxInterval: 5000,
        enabled: true,
      }),
    );

    // Initial call
    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(1);

    // First interval: 1000ms
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(2);

    // Second interval: 1000 * 1.5 = 1500ms
    await act(async () => {
      vi.advanceTimersByTime(1499);
    });
    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(2); // not yet

    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("caps backoff at maxInterval", async () => {
    const fn = vi.fn().mockResolvedValue(undefined);

    renderHook(() =>
      usePolling(fn, {
        interval: 3000,
        backoff: true,
        maxInterval: 5000,
        enabled: true,
      }),
    );

    await act(async () => {}); // initial
    // interval 1: 3000
    await act(async () => { vi.advanceTimersByTime(3000); });
    await act(async () => {});
    // interval 2: 4500
    await act(async () => { vi.advanceTimersByTime(4500); });
    await act(async () => {});
    // interval 3: min(6750, 5000) = 5000 (capped)
    await act(async () => { vi.advanceTimersByTime(5000); });
    await act(async () => {});

    expect(fn).toHaveBeenCalledTimes(4);
  });

  it("cleans up timer on unmount", async () => {
    const fn = vi.fn().mockResolvedValue(undefined);

    const { unmount } = renderHook(() =>
      usePolling(fn, { interval: 1000, enabled: true }),
    );

    await act(async () => {});
    expect(fn).toHaveBeenCalledTimes(1);

    unmount();

    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(fn).toHaveBeenCalledTimes(1);
  });
});
