import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { stationToPromote, useStationGallery } from "@/hooks/useStationGallery";
import { STATION_GALLERY_RING } from "@/lib/live/config";
import type { FrameSample } from "@/lib/live/procedure-types";
import { STATION_INDEX, STATION_ORDER } from "@/lib/live/stations";
import type { StationKey, StationStatus } from "@/lib/types";

let urlCounter = 0;
const createObjectURL = vi.fn(() => `blob:mock/${++urlCounter}`);
const revokeObjectURL = vi.fn();
const original = {
  createObjectURL: URL.createObjectURL,
  revokeObjectURL: URL.revokeObjectURL,
};

beforeEach(() => {
  urlCounter = 0;
  createObjectURL.mockClear();
  revokeObjectURL.mockClear();
  URL.createObjectURL = createObjectURL;
  URL.revokeObjectURL = revokeObjectURL;
});
afterEach(() => {
  URL.createObjectURL = original.createObjectURL;
  URL.revokeObjectURL = original.revokeObjectURL;
});

const blob = (n: number) => new Blob([`frame-${n}`], { type: "image/jpeg" });

describe("useStationGallery", () => {
  it("evicts the oldest parked frame beyond the ring size", () => {
    expect(STATION_GALLERY_RING).toBe(8);
    const { result } = renderHook(() => useStationGallery());
    for (let t = 0; t <= STATION_GALLERY_RING; t++) result.current.offer(t, blob(t), t / 2);

    let promoted = true;
    act(() => {
      promoted = result.current.promote(0, "antrum");
    });
    expect(promoted).toBe(false);
    act(() => {
      promoted = result.current.promote(1, "antrum");
    });
    expect(promoted).toBe(true);
    expect(result.current.frames.antrum).toMatchObject({ ticket: 1, videoTime: 0.5 });
  });

  it("copies on promote: the ring entry stays available", () => {
    const { result } = renderHook(() => useStationGallery());
    result.current.offer(5, blob(5), 2.5);
    act(() => {
      expect(result.current.promote(5, "antrum")).toBe(true);
      expect(result.current.promote(5, "incisura")).toBe(true);
    });
    expect(result.current.frames.antrum?.ticket).toBe(5);
    expect(result.current.frames.incisura?.ticket).toBe(5);
    expect(result.current.has("antrum")).toBe(true);
    expect(result.current.has("z_line")).toBe(false);
  });

  it("revokes a station's old URL when a better frame replaces it", () => {
    const { result } = renderHook(() => useStationGallery());
    result.current.offer(1, blob(1), 1);
    result.current.offer(2, blob(2), 2);
    act(() => {
      result.current.promote(1, "antrum");
    });
    const first = result.current.frames.antrum?.url;
    expect(revokeObjectURL).not.toHaveBeenCalled();

    act(() => {
      result.current.promote(2, "antrum");
    });
    expect(revokeObjectURL).toHaveBeenCalledWith(first);
    expect(result.current.frames.antrum?.url).not.toBe(first);
  });

  it("revokes every URL it created on unmount", () => {
    const { result, unmount } = renderHook(() => useStationGallery());
    result.current.offer(1, blob(1), 1);
    act(() => {
      result.current.promote(1, "antrum");
      result.current.promote(1, "incisura");
    });
    const urls = createObjectURL.mock.results.map((r) => r.value);
    expect(urls).toHaveLength(2);
    unmount();
    for (const url of urls) expect(revokeObjectURL).toHaveBeenCalledWith(url);
  });

  it("reset revokes and empties; clearRing keeps promoted frames", () => {
    const { result } = renderHook(() => useStationGallery());
    result.current.offer(1, blob(1), 1);
    act(() => {
      result.current.promote(1, "antrum");
    });
    result.current.clearRing();
    expect(result.current.frames.antrum).toBeDefined();
    let promoted = true;
    act(() => {
      promoted = result.current.promote(1, "incisura");
    });
    expect(promoted).toBe(false);

    act(() => result.current.reset());
    expect(result.current.frames).toEqual({});
    expect(revokeObjectURL).toHaveBeenCalledTimes(1);
  });
});

describe("stationToPromote (level-driven)", () => {
  const status = (observed: StationKey[]): StationStatus[] =>
    STATION_ORDER.map((k) => (observed.includes(k) ? "observed" : "unseen"));

  function frame(opts: Partial<FrameSample>): FrameSample {
    return {
      seq: 0,
      t: 0,
      score: 2,
      scoreConfidence: 0.9,
      expectedScore: null,
      region: "stomach",
      motion: "stationary",
      motionConfidence: 0.9,
      frameIndex: 0,
      processingTimeMs: 20,
      landmarkStatus: "ok",
      landmarkTop: "antrum",
      ...opts,
    };
  }

  const none = () => false;
  const all = () => true;
  const visible = (observed: StationKey[]) => ({ visible: true, status: status(observed) });

  it("fills an empty slot from a confident frame of an observed station", () => {
    expect(stationToPromote(frame({}), visible(["antrum"]), none)).toBe("antrum");
  });

  it("works even if the observed/best_frame events were lost", () => {
    expect(
      stationToPromote(frame({ stationBestEvents: [] }), visible(["antrum"]), none),
    ).toBe("antrum");
  });

  it("replaces a filled slot only on a best_frame event", () => {
    expect(stationToPromote(frame({}), visible(["antrum"]), all)).toBeNull();
    expect(
      stationToPromote(frame({ stationBestEvents: ["antrum"] }), visible(["antrum"]), all),
    ).toBe("antrum");
  });

  it("skips stations not yet observed, frames without a top, and non-ok frames", () => {
    expect(stationToPromote(frame({}), visible([]), none)).toBeNull();
    expect(stationToPromote(frame({ landmarkTop: null }), visible(["antrum"]), none)).toBeNull();
    expect(
      stationToPromote(frame({ landmarkStatus: "low_quality" }), visible(["antrum"]), none),
    ).toBeNull();
  });

  it("does nothing while stations are not visible (shadow mode)", () => {
    const hidden = { visible: false, status: status(["antrum"]) };
    expect(
      stationToPromote(frame({ stationBestEvents: ["antrum"] }), hidden, none),
    ).toBeNull();
  });

  it("indexes stations in ESGE order", () => {
    expect(STATION_INDEX.antrum).toBe(5);
  });
});
