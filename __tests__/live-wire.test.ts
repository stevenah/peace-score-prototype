import { describe, it, expect, vi } from "vitest";
import v1Fixture from "@/contracts/live_frame_result.v1.example.json";
import v2Fixture from "@/contracts/live_frame_result.v2.example.json";
import {
  StationsPatchSchema,
  createLandmarkParser,
  frameSampleFromResult,
  parseStationsData,
  stationsSummary,
  timelineEntryFromSample,
} from "@/lib/live/wire";
import { ProcedureStore } from "@/lib/live/procedure-store";
import type { LiveFrameResult } from "@/lib/types";

const v1 = v1Fixture as LiveFrameResult;
const v2 = v2Fixture as LiveFrameResult;
const landmark = v2Fixture.landmark;

function withLandmark(lm: unknown): LiveFrameResult {
  return { ...v1, landmark: lm } as LiveFrameResult;
}

describe("parseLandmark", () => {
  it("returns undefined, silently, when the block is absent (feature off)", () => {
    const warn = vi.fn();
    const parse = createLandmarkParser(warn);
    expect(parse(v1.landmark)).toBeUndefined();
    expect(parse(undefined)).toBeUndefined();
    expect(warn).not.toHaveBeenCalled();
  });

  it("parses the v2 fixture losslessly", () => {
    const parse = createLandmarkParser(() => {});
    expect(parse(landmark)).toEqual(landmark);
  });

  it("rejects probs of the wrong length", () => {
    const parse = createLandmarkParser(() => {});
    expect(parse({ ...landmark, probs: landmark.probs.slice(0, 9) })).toBeUndefined();
  });

  it("rejects stations / auto_enabled of the wrong length", () => {
    const parse = createLandmarkParser(() => {});
    expect(parse({ ...landmark, stations: landmark.stations.slice(1) })).toBeUndefined();
    expect(
      parse({ ...landmark, auto_enabled: [...landmark.auto_enabled, true] }),
    ).toBeUndefined();
  });

  it("rejects unknown station keys and statuses", () => {
    const parse = createLandmarkParser(() => {});
    expect(parse({ ...landmark, top: "pylorus" })).toBeUndefined();
    expect(parse({ ...landmark, status: "great" })).toBeUndefined();
    expect(
      parse({ ...landmark, stations: ["seen", ...landmark.stations.slice(1)] }),
    ).toBeUndefined();
  });

  it("ignores a block with another schema", () => {
    const warn = vi.fn();
    const parse = createLandmarkParser(warn);
    expect(parse({ ...landmark, schema: "esge10.v2" })).toBeUndefined();
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toMatch(/esge10\.v2/);
  });

  it("warns once per parser, however many bad frames arrive", () => {
    const warn = vi.fn();
    const parse = createLandmarkParser(warn);
    for (let i = 0; i < 20; i++) parse({ schema: "esge10.v1", junk: i });
    expect(warn).toHaveBeenCalledTimes(1);
    // A fresh parser (a new procedure) may warn again.
    createLandmarkParser(warn)({ schema: "esge10.v1" });
    expect(warn).toHaveBeenCalledTimes(2);
  });

  it("accepts a non-classified frame: no probs/top, null current, no layout", () => {
    const parse = createLandmarkParser(() => {});
    const { probs, top, confidence, quality, layout, mode, ...rest } = landmark;
    void probs; void top; void confidence; void quality; void layout; void mode;
    const lm = parse({ ...rest, status: "unsupported_layout", current: null });
    expect(lm?.status).toBe("unsupported_layout");
    expect(lm?.current).toBeNull();
    expect(lm && "probs" in lm).toBe(false);
    // Explicit nulls are normalised away too.
    const lm2 = parse({ ...landmark, status: "error", top: null, probs: null });
    expect(lm2 && "top" in lm2).toBe(false);
  });

  it("defaults missing events rather than dropping the frame", () => {
    const parse = createLandmarkParser(() => {});
    const { events, ...rest } = landmark;
    void events;
    expect(parse(rest)?.events).toEqual({ observed: [], best_frame: [] });
  });
});

describe("frameSampleFromResult — PEACE always survives", () => {
  it("drops a malformed landmark block and keeps the PEACE sample intact", () => {
    const warn = vi.fn();
    const good = frameSampleFromResult(v1, 3, 7, createLandmarkParser(warn));
    const bad = frameSampleFromResult(
      withLandmark({ schema: "esge10.v1", status: "ok", probs: "nope" }),
      3,
      7,
      createLandmarkParser(warn),
    );
    expect(bad).toEqual(good);
    expect(bad.score).toBe(2);
    expect(bad.landmarkStatus).toBeUndefined();
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("survives a parser that throws", () => {
    const sample = frameSampleFromResult(v2, 3, 7, () => {
      throw new Error("boom");
    });
    expect(sample.score).toBe(2);
    expect(sample.landmarkStatus).toBeUndefined();
  });

  it("keys the sample by the client ticket, not the server index", () => {
    const sample = frameSampleFromResult(v2, 3, 99, createLandmarkParser(() => {}));
    expect(sample.seq).toBe(99);
    expect(sample.frameIndex).toBe(412);
  });
});

describe("persistence wire format", () => {
  const parse = () => createLandmarkParser(() => {});

  it("writes a feature-off timeline entry exactly as before", () => {
    const entry = timelineEntryFromSample(frameSampleFromResult(v1, 3, 0, parse()));
    expect(entry).toEqual({
      timestamp: 3,
      frame_index: 412,
      motion: "stationary",
      region: "stomach",
      peace_score: 2,
      confidence: 0.81,
    });
  });

  it("adds landmark fields when the frame carried a landmark block", () => {
    const entry = timelineEntryFromSample(frameSampleFromResult(v2, 3, 0, parse()));
    expect(entry).toMatchObject({
      landmark_top: "antrum",
      landmark_confidence: 0.912,
      landmark_status: "ok",
      station_current: "antrum",
      station_events: { observed: ["antrum"], best_frame: ["antrum"] },
    });
  });

  it("omits station_events when a frame had none", () => {
    const quiet = { ...landmark, events: { observed: [], best_frame: [] } };
    const entry = timelineEntryFromSample(
      frameSampleFromResult(withLandmark(quiet), 3, 0, parse()),
    );
    expect(entry.landmark_status).toBe("ok");
    expect("station_events" in entry).toBe(false);
  });

  it("summarises stations only once they were ever available", () => {
    const store = new ProcedureStore();
    expect(stationsSummary(store.getSnapshot().stations)).toBeUndefined();

    store.ingest(frameSampleFromResult(v2, 3, 0, parse()));
    const summary = stationsSummary(store.getSnapshot().stations);
    expect(summary).toMatchObject({
      schema: "esge10.v1",
      model_version: "lm-1.0.0",
      display: true,
      availability: "ok",
      manual: Array(10).fill(null),
      auto_enabled: Array(10).fill(true),
    });
    expect(summary?.status.slice(0, 6)).toEqual(Array(6).fill("observed"));
    expect(summary?.observed_at_t.slice(0, 6)).toEqual(Array(6).fill(3));
    // Round-trips through the stored-column reader.
    expect(parseStationsData(JSON.stringify(summary))).toEqual(summary);
  });

  it("reads back nothing from an empty or corrupt stationsData column", () => {
    expect(parseStationsData(null)).toBeNull();
    expect(parseStationsData("{not json")).toBeNull();
    expect(parseStationsData(JSON.stringify({ schema: "esge10.v1" }))).toBeNull();
  });

  it("validates PATCH bodies: 10 marks and 10 times", () => {
    const ok = {
      manual: ["confirmed", null, "rejected", null, null, null, null, null, null, null],
      observed_at_t: [1, null, null, null, null, null, null, null, null, null],
    };
    expect(StationsPatchSchema.safeParse(ok).success).toBe(true);
    expect(
      StationsPatchSchema.safeParse({ ...ok, manual: ok.manual.slice(1) }).success,
    ).toBe(false);
    expect(
      StationsPatchSchema.safeParse({ ...ok, manual: ["maybe", ...ok.manual.slice(1)] })
        .success,
    ).toBe(false);
    expect(
      StationsPatchSchema.safeParse({ ...ok, observed_at_t: [-1, ...ok.observed_at_t.slice(1)] })
        .success,
    ).toBe(false);
  });
});
