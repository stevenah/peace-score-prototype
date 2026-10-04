import { describe, it, expect } from "vitest";
import stationsContract from "@/contracts/stations.json";
import v1Fixture from "@/contracts/live_frame_result.v1.example.json";
import v2Fixture from "@/contracts/live_frame_result.v2.example.json";
import {
  ESOPHAGEAL,
  GASTRODUODENAL,
  STATION_ESGE_NO,
  STATION_GROUPS,
  STATION_INDEX,
  STATION_LABELS,
  STATION_ORDER,
  STATION_REGION,
  STATION_SCHEMA,
  STATION_SHORT,
  STATIONS_TOTAL,
  isObserved,
  isStationKey,
  missingStations,
} from "@/lib/live/stations";
import { createLandmarkParser, frameSampleFromResult } from "@/lib/live/wire";
import type { LiveFrameResult } from "@/lib/types";

describe("lib/live/stations mirrors contracts/stations.json", () => {
  it("has the same schema id", () => {
    expect(STATION_SCHEMA).toBe(stationsContract.schema);
  });

  it("has the same stations, in the same (ESGE) order, with the same metadata", () => {
    const ours = STATION_ORDER.map((key) => ({
      key,
      esge: STATION_ESGE_NO[key],
      region: STATION_REGION[key],
      label: STATION_LABELS[key],
      short: STATION_SHORT[key],
    }));
    const theirs = stationsContract.stations.map((s) => ({
      key: s.key,
      esge: s.esge,
      region: s.region,
      label: s.label,
      short: s.short,
    }));
    expect(ours).toEqual(theirs);
    expect(STATIONS_TOTAL).toBe(10);
  });

  it("numbers stations 1..10 and indexes them 0..9", () => {
    STATION_ORDER.forEach((k, i) => {
      expect(STATION_INDEX[k]).toBe(i);
      expect(STATION_ESGE_NO[k]).toBe(i + 1);
    });
  });

  it("splits esophageal (1-3) from gastroduodenal (4-10)", () => {
    expect(ESOPHAGEAL).toEqual(STATION_ORDER.slice(0, 3));
    expect(GASTRODUODENAL).toEqual(STATION_ORDER.slice(3));
  });

  it("groups the checklist esophagus / duodenum / stomach, covering every station once", () => {
    expect(STATION_GROUPS.map((g) => g.region)).toEqual([
      "esophagus",
      "duodenum",
      "stomach",
    ]);
    expect(STATION_GROUPS.flatMap((g) => g.stations)).toEqual([...STATION_ORDER]);
  });

  it("recognises station keys, and nothing inherited", () => {
    expect(isStationKey("antrum")).toBe(true);
    expect(isStationKey("toString")).toBe(false);
    expect(isStationKey(3)).toBe(false);
  });
});

describe("station helpers", () => {
  it("lets a manual mark win in both directions", () => {
    expect(isObserved("observed", null)).toBe(true);
    expect(isObserved("observed", "rejected")).toBe(false);
    expect(isObserved("unseen", "confirmed")).toBe(true);
    expect(isObserved("candidate", null)).toBe(false);
  });

  it("reports missing stations in ESGE order, never manual-only ones", () => {
    const observed = Array(10).fill(false);
    observed[STATION_INDEX.antrum] = true;
    const auto = Array(10).fill(true);
    auto[STATION_INDEX.incisura] = false;
    expect(missingStations(observed, auto, GASTRODUODENAL)).toEqual([
      "duodenal_bulb",
      "duodenum_descending",
      "cardia_fundus_retroflex",
      "lesser_curvature_retroflex",
      "corpus_greater_curvature",
    ]);
  });
});

describe("contract fixtures through the wire -> sample adapter", () => {
  const quiet = () => createLandmarkParser(() => {});

  it("v1 (feature off) yields a PEACE-only sample with no landmark fields", () => {
    const sample = frameSampleFromResult(
      v1Fixture as LiveFrameResult,
      12.5,
      412,
      quiet(),
    );
    expect(sample).toEqual({
      seq: 412,
      t: 12.5,
      score: 2,
      scoreConfidence: 0.81,
      expectedScore: expect.closeTo(2.13, 10),
      region: "stomach",
      motion: "stationary",
      motionConfidence: 0.8,
      frameIndex: 412,
      processingTimeMs: 131.4,
    });
  });

  it("v2 (feature on) maps into the optional FrameSample station fields", () => {
    const sample = frameSampleFromResult(
      v2Fixture as LiveFrameResult,
      12.5,
      412,
      quiet(),
    );
    // PEACE part identical to v1.
    const peaceOnly = frameSampleFromResult(v1Fixture as LiveFrameResult, 12.5, 412, quiet());
    expect(sample).toMatchObject(peaceOnly);
    expect(sample).toMatchObject({
      landmarkStatus: "ok",
      landmarkTop: "antrum",
      landmarkConfidence: 0.912,
      landmarkQuality: 0.82,
      landmarkModelVersion: "lm-1.0.0",
      stationsDisplay: true,
      stationCurrent: "antrum",
      stationStatus: [
        "observed", "observed", "observed", "observed", "observed", "observed",
        "candidate", "unseen", "unseen", "unseen",
      ],
      stationAutoEnabled: Array(10).fill(true),
      stationObservedEvents: ["antrum"],
      stationBestEvents: ["antrum"],
    });
  });
});
