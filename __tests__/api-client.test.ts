import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ApiError, getAnalysis, analyzeFrame, saveLiveAnalysis } from "@/lib/api-client";

// -- ApiError ---------------------------------------------------------------

describe("ApiError", () => {
  it("has correct name and properties", () => {
    const err = new ApiError(404, "Not found");
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("ApiError");
    expect(err.status).toBe(404);
    expect(err.message).toBe("Not found");
  });

  it("is catchable as Error", () => {
    try {
      throw new ApiError(500, "Server error");
    } catch (e) {
      expect(e).toBeInstanceOf(Error);
      expect((e as ApiError).status).toBe(500);
    }
  });
});

// -- getAnalysis ------------------------------------------------------------

describe("getAnalysis", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("calls the batch endpoint for normal IDs", async () => {
    const mockResponse = {
      analysis_id: "abc-123",
      status: "completed",
      progress: 1,
      created_at: "2025-01-01",
    };
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify(mockResponse), { status: 200 }),
    );

    const result = await getAnalysis("abc-123");
    expect(fetch).toHaveBeenCalledWith("/api/analysis/abc-123");
    expect(result.analysis_id).toBe("abc-123");
  });

  it("calls the live endpoint for live_ prefixed IDs", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ analysis_id: "live_xyz", status: "completed", progress: 1, created_at: "" }), { status: 200 }),
    );

    await getAnalysis("live_xyz");
    expect(fetch).toHaveBeenCalledWith("/api/analysis/live/live_xyz");
  });

  it("throws ApiError on non-OK response", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response("Not found", { status: 404 }),
    );

    await expect(getAnalysis("bad-id")).rejects.toThrow(ApiError);
    await expect(getAnalysis("bad-id")).rejects.toMatchObject({ status: 404 });
  });
});

// -- analyzeFrame -----------------------------------------------------------

describe("analyzeFrame", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends frame as FormData", async () => {
    const mockResult = {
      peace_score: { score: 2, label: "Adequate", confidence: 0.8 },
      processing_time_ms: 50,
    };
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify(mockResult), { status: 200 }),
    );

    const frame = new Blob(["frame-data"], { type: "image/jpeg" });
    const result = await analyzeFrame(frame);

    expect(fetch).toHaveBeenCalledOnce();
    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("/api/analyze-frame");
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeInstanceOf(FormData);
    expect(result.peace_score.score).toBe(2);
  });

  it("includes previous_frame when provided", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ peace_score: { score: 1, label: "Inadequate", confidence: 0.7 }, processing_time_ms: 30 }), { status: 200 }),
    );

    const frame = new Blob(["frame"], { type: "image/jpeg" });
    const prev = new Blob(["prev"], { type: "image/jpeg" });
    await analyzeFrame(frame, prev);

    const body = vi.mocked(fetch).mock.calls[0][1]?.body as FormData;
    expect(body.has("frame")).toBe(true);
    expect(body.has("previous_frame")).toBe(true);
  });
});

// -- saveLiveAnalysis -------------------------------------------------------

describe("saveLiveAnalysis", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("posts metadata as JSON in FormData", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ id: "db-1", analysisId: "live_1" }), { status: 200 }),
    );

    const result = await saveLiveAnalysis({
      filename: "test.mp4",
      overallScore: 2,
      framesAnalyzed: 50,
      duration: 30,
      timeline: [],
    });

    expect(result.analysisId).toBe("live_1");

    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("/api/analysis/save-live");
    expect(init?.method).toBe("POST");

    const body = init?.body as FormData;
    const metadata = JSON.parse(body.get("metadata") as string);
    expect(metadata.filename).toBe("test.mp4");
    expect(metadata.overallScore).toBe(2);
  });

  it("appends video file when provided", async () => {
    vi.mocked(fetch).mockResolvedValue(
      new Response(JSON.stringify({ id: "db-1", analysisId: "live_1" }), { status: 200 }),
    );

    const videoFile = new File(["video"], "test.mp4", { type: "video/mp4" });
    await saveLiveAnalysis({
      filename: "test.mp4",
      overallScore: null,
      framesAnalyzed: 10,
      duration: null,
      timeline: [],
      videoFile,
    });

    const body = vi.mocked(fetch).mock.calls[0][1]?.body as FormData;
    expect(body.has("video")).toBe(true);
  });
});
