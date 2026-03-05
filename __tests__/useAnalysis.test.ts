import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";

// Mock the api-client module before importing the hook
vi.mock("@/lib/api-client", () => ({
  getAnalysis: vi.fn(),
}));

import { useAnalysis } from "@/hooks/useAnalysis";
import { getAnalysis } from "@/lib/api-client";
import type { AnalysisResponse } from "@/lib/types";

const mockGetAnalysis = vi.mocked(getAnalysis);

function makeResponse(overrides: Partial<AnalysisResponse> = {}): AnalysisResponse {
  return {
    analysis_id: "test-123",
    status: "processing",
    progress: 0.5,
    created_at: "2025-01-01",
    ...overrides,
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  mockGetAnalysis.mockReset();
});
afterEach(() => {
  vi.useRealTimers();
});

describe("useAnalysis", () => {
  it("returns null data when no analysisId", async () => {
    const { result } = renderHook(() => useAnalysis(null));
    await act(async () => {});
    expect(result.current.data).toBeNull();
    expect(mockGetAnalysis).not.toHaveBeenCalled();
  });

  it("fetches analysis immediately when given an ID", async () => {
    const response = makeResponse();
    mockGetAnalysis.mockResolvedValue(response);

    const { result } = renderHook(() => useAnalysis("test-123"));
    await act(async () => {});

    expect(mockGetAnalysis).toHaveBeenCalledWith("test-123");
    expect(result.current.data?.status).toBe("processing");
  });

  it("stops polling when status is completed", async () => {
    const completed = makeResponse({ status: "completed", progress: 1 });
    mockGetAnalysis.mockResolvedValue(completed);

    const { result } = renderHook(() => useAnalysis("test-123"));
    await act(async () => {});

    expect(result.current.data?.status).toBe("completed");

    // Advance time - should not poll again
    mockGetAnalysis.mockClear();
    await act(async () => {
      vi.advanceTimersByTime(10000);
    });
    await act(async () => {});

    expect(mockGetAnalysis).not.toHaveBeenCalled();
  });

  it("stops polling when status is failed", async () => {
    const failed = makeResponse({ status: "failed", error: "Something broke" });
    mockGetAnalysis.mockResolvedValue(failed);

    const { result } = renderHook(() => useAnalysis("test-123"));
    await act(async () => {});

    expect(result.current.data?.status).toBe("failed");

    mockGetAnalysis.mockClear();
    await act(async () => {
      vi.advanceTimersByTime(10000);
    });
    await act(async () => {});

    expect(mockGetAnalysis).not.toHaveBeenCalled();
  });

  it("sets error on fetch failure", async () => {
    mockGetAnalysis.mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useAnalysis("test-123"));
    await act(async () => {});

    expect(result.current.error).toBe("Network error");
  });

  it("continues polling while processing", async () => {
    const processing = makeResponse({ status: "processing", progress: 0.3 });
    mockGetAnalysis.mockResolvedValue(processing);

    renderHook(() => useAnalysis("test-123"));

    // Initial call
    await act(async () => {});
    expect(mockGetAnalysis).toHaveBeenCalledTimes(1);

    // After interval (2000ms)
    await act(async () => {
      vi.advanceTimersByTime(2000);
    });
    await act(async () => {});
    expect(mockGetAnalysis).toHaveBeenCalledTimes(2);
  });
});
