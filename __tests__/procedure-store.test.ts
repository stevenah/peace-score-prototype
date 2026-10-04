import { describe, it, expect } from "vitest";
import { ProcedureStore, deriveVisibility } from "@/lib/live/procedure-store";
import {
  STATION_AVAILABILITY_RELEASE_FRAMES,
  TOTAL_PROCEDURE_TARGET_S,
} from "@/lib/live/config";
import type { FrameSample } from "@/lib/live/procedure-types";
import { ESOPHAGEAL, STATION_INDEX, STATION_ORDER } from "@/lib/live/stations";
import type {
  AnatomicalRegion,
  LandmarkStatus,
  MotionDirection,
  PeaceScore,
  StationKey,
} from "@/lib/types";

let seq = 0;
function sample(
  t: number,
  score: PeaceScore,
  region: AnatomicalRegion | null = "stomach",
  motion: MotionDirection = "stationary",
  opts: {
    scoreConfidence?: number;
    motionConfidence?: number;
    expectedScore?: number;
  } = {},
): FrameSample {
  return {
    seq: seq++,
    t,
    score,
    scoreConfidence: opts.scoreConfidence ?? 0.9,
    expectedScore: opts.expectedScore ?? null,
    region,
    motion,
    motionConfidence: opts.motionConfidence ?? 0.9,
    frameIndex: seq,
    processingTimeMs: 25,
  };
}

/** Steady 2 Hz run: one sample every 0.5s. */
function feed(
  store: ProcedureStore,
  specs: Array<[PeaceScore, AnatomicalRegion | null]>,
  step = 0.5,
) {
  specs.forEach(([score, region], i) => store.ingest(sample(i * step, score, region)));
}

describe("ProcedureStore — timing", () => {
  it("starts empty", () => {
    const s = new ProcedureStore().getSnapshot();
    expect(s.timers.totalElapsedS).toBe(0);
    expect(s.coverage.framesAnalysed).toBe(0);
    expect(s.score.meanScore).toBeNull();
    expect(s.visibility.level).toBeNull();
  });

  it("credits no time for a single sample", () => {
    const store = new ProcedureStore();
    store.ingest(sample(0, 2));
    expect(store.getSnapshot().timers.totalElapsedS).toBe(0);
    expect(store.getSnapshot().coverage.framesAnalysed).toBe(1);
  });

  it("per-region dwell sums to total elapsed", () => {
    const store = new ProcedureStore();
    feed(store, [
      [2, "esophagus"], [2, "esophagus"], [3, "esophagus"],
      [3, "stomach"], [2, "stomach"], [1, "stomach"], [2, "stomach"],
      [3, "duodenum"], [3, "duodenum"],
    ]);
    const { timers } = store.getSnapshot();
    const sum =
      timers.perRegionS.esophagus +
      timers.perRegionS.stomach +
      timers.perRegionS.duodenum;
    expect(sum).toBeCloseTo(timers.totalElapsedS, 10);
    expect(timers.totalElapsedS).toBeCloseTo(4.0, 10); // 9 samples * 0.5s - 0.5
  });

  it("credits dwell to the region the frame was IN, not the next one", () => {
    const store = new ProcedureStore();
    // 0.0 esophagus, 0.5 esophagus, 1.0 stomach
    store.ingest(sample(0.0, 2, "esophagus"));
    store.ingest(sample(0.5, 2, "esophagus"));
    store.ingest(sample(1.0, 2, "stomach"));
    const { perRegionS } = store.getSnapshot().timers;
    // The 0.5->1.0 interval belongs to esophagus: we were still there.
    expect(perRegionS.esophagus).toBeCloseTo(1.0, 10);
    expect(perRegionS.stomach).toBeCloseTo(0, 10);
  });

  it("credits nothing across a backward seek", () => {
    const store = new ProcedureStore();
    store.ingest(sample(10.0, 2));
    store.ingest(sample(10.5, 2));
    store.ingest(sample(2.0, 2)); // user scrubbed backwards
    store.ingest(sample(2.5, 2));
    const { timers } = store.getSnapshot();
    // 0.5 + (discontinuity) + 0.5
    expect(timers.totalElapsedS).toBeCloseTo(1.0, 10);
  });

  it("credits nothing across a gap longer than MAX_SAMPLE_GAP_S", () => {
    const store = new ProcedureStore();
    store.ingest(sample(0, 2));
    store.ingest(sample(0.5, 2));
    store.ingest(sample(30.0, 2)); // long stall or forward seek
    store.ingest(sample(30.5, 2));
    expect(store.getSnapshot().timers.totalElapsedS).toBeCloseTo(1.0, 10);
  });

  it("resets current-region dwell after a discontinuity", () => {
    const store = new ProcedureStore();
    store.ingest(sample(0, 2, "stomach"));
    store.ingest(sample(0.5, 2, "stomach"));
    expect(store.getSnapshot().timers.currentRegionDwellS).toBeCloseTo(0.5, 10);
    store.ingest(sample(60, 2, "stomach"));
    expect(store.getSnapshot().timers.currentRegionDwellS).toBe(0);
  });

  it("reports total progress against the ESGE-derived target", () => {
    const store = new ProcedureStore();
    const n = Math.round(TOTAL_PROCEDURE_TARGET_S / 0.5) + 1;
    for (let i = 0; i < n; i++) store.ingest(sample(i * 0.5, 3));
    const { timers } = store.getSnapshot();
    expect(timers.totalTargetS).toBe(TOTAL_PROCEDURE_TARGET_S);
    expect(timers.totalProgress).toBe(1); // clamped
  });
});

describe("ProcedureStore — regions and coverage", () => {
  it("tracks visit order without duplicates", () => {
    const store = new ProcedureStore();
    feed(store, [
      [2, "esophagus"], [2, "stomach"], [2, "esophagus"], [2, "duodenum"],
    ]);
    const { regions, coverage } = store.getSnapshot();
    expect(regions.visited).toEqual(["esophagus", "stomach", "duodenum"]);
    expect(coverage.regionsVisited).toBe(3);
    expect(coverage.regionsTotal).toBe(3);
  });

  it("framesAnalysed counts every ingested frame", () => {
    const store = new ProcedureStore();
    feed(store, Array.from({ length: 40 }, () => [2, "stomach"] as [PeaceScore, AnatomicalRegion]));
    expect(store.getSnapshot().coverage.framesAnalysed).toBe(40);
  });

  it("fill is dwell as a share of the total target, never over 100", () => {
    const store = new ProcedureStore();
    const n = Math.round(TOTAL_PROCEDURE_TARGET_S / 0.5) + 20;
    for (let i = 0; i < n; i++) store.ingest(sample(i * 0.5, 3, "stomach"));
    const fill = store.getSnapshot().coverage.fillByRegion;
    expect(fill.stomach).toBeLessThanOrEqual(100);
    expect(fill.stomach).toBeCloseTo(100, 5);
    expect(fill.esophagus).toBe(0);
  });
});

describe("ProcedureStore — scoring", () => {
  it("uses a ROUNDED MEAN per region, matching the batch pipeline", () => {
    const store = new ProcedureStore();
    // scores 3,3,3,0 -> mean 2.25 -> 2.  The old UI used Math.min, giving 0.
    feed(store, [[3, "stomach"], [3, "stomach"], [3, "stomach"], [0, "stomach"]]);
    expect(store.getSnapshot().score.perRegionMean.stomach).toBe(2);
  });

  it("rounds half to even, like Python's round()", () => {
    const store = new ProcedureStore();
    feed(store, [[2, "stomach"], [3, "stomach"]]); // mean 2.5 -> 2, not 3
    expect(store.getSnapshot().score.perRegionMean.stomach).toBe(2);
  });

  it("keeps the session mean continuous", () => {
    const store = new ProcedureStore();
    feed(store, [[3, "stomach"], [3, "stomach"], [3, "stomach"], [0, "stomach"]]);
    expect(store.getSnapshot().score.meanScore).toBeCloseTo(2.25, 10);
  });

  it("tracks min and max", () => {
    const store = new ProcedureStore();
    feed(store, [[2, "stomach"], [0, "stomach"], [3, "stomach"], [1, "stomach"]]);
    const { score } = store.getSnapshot();
    expect(score.min).toBe(0);
    expect(score.max).toBe(3);
  });

  it("does not let a single bad frame reach the display score", () => {
    const store = new ProcedureStore();
    // settle on 3, then one stray 0
    for (let i = 0; i < 8; i++) store.ingest(sample(i * 0.5, 3));
    expect(store.getSnapshot().score.display).toBe(3);
    store.ingest(sample(4.0, 0));
    expect(store.getSnapshot().score.current).toBe(0);
    expect(store.getSnapshot().score.display).toBe(3); // held
  });

  it("does adopt a score that persists", () => {
    const store = new ProcedureStore();
    for (let i = 0; i < 8; i++) store.ingest(sample(i * 0.5, 3));
    for (let i = 8; i < 16; i++) store.ingest(sample(i * 0.5, 1));
    expect(store.getSnapshot().score.display).toBe(1);
  });

  it("does not thrash when the score alternates every frame", () => {
    const store = new ProcedureStore();
    const seen = new Set<PeaceScore | null>();
    for (let i = 0; i < 40; i++) {
      store.ingest(sample(i * 0.5, (i % 2 === 0 ? 1 : 2) as PeaceScore));
      seen.add(store.getSnapshot().score.display);
    }
    // Never settles on a second value, because no value ever holds long enough.
    expect(seen.size).toBeLessThanOrEqual(2);
  });
});

describe("ProcedureStore — expected score", () => {
  it("is null when the backend sends no distribution", () => {
    const store = new ProcedureStore();
    store.ingest(sample(0, 2));
    expect(store.getSnapshot().score.expected).toBeNull();
  });

  it("tracks the continuous expected score, unlike the stepped argmax", () => {
    const store = new ProcedureStore();
    store.ingest(sample(0.0, 2, "stomach", "stationary", { expectedScore: 2.4 }));
    const first = store.getSnapshot().score.expected;
    expect(first).toBeCloseTo(2.4, 10);

    store.ingest(sample(0.5, 2, "stomach", "stationary", { expectedScore: 2.9 }));
    const second = store.getSnapshot().score.expected!;
    // Smoothed, so it moves toward the new value without jumping to it.
    expect(second).toBeGreaterThan(2.4);
    expect(second).toBeLessThan(2.9);
    // ...while the discrete score has not changed at all.
    expect(store.getSnapshot().score.current).toBe(2);
  });
});

describe("ProcedureStore — motion consensus", () => {
  it("stays stationary until enough frames agree", () => {
    const store = new ProcedureStore();
    store.ingest(sample(0.0, 2, "stomach", "withdrawal"));
    expect(store.getSnapshot().regions.motion).toBe("stationary");
    store.ingest(sample(0.5, 2, "stomach", "withdrawal"));
    expect(store.getSnapshot().regions.motion).toBe("stationary");
    store.ingest(sample(1.0, 2, "stomach", "withdrawal"));
    expect(store.getSnapshot().regions.motion).toBe("withdrawal");
  });

  it("ignores low-confidence motion", () => {
    const store = new ProcedureStore();
    for (let i = 0; i < 5; i++) {
      store.ingest(
        sample(i * 0.5, 2, "stomach", "withdrawal", { motionConfidence: 0.3 }),
      );
    }
    expect(store.getSnapshot().regions.motion).toBe("stationary");
  });
});

describe("ProcedureStore — snapshot identity", () => {
  it("keeps unchanged slices referentially stable", () => {
    const store = new ProcedureStore();
    feed(store, [[2, "stomach"], [2, "stomach"], [2, "stomach"]]);
    const a = store.getSnapshot();
    store.ingest(sample(1.5, 2, "stomach"));
    const b = store.getSnapshot();
    // coverage changes every frame (framesAnalysed), regions does not.
    expect(b.regions).toBe(a.regions);
    expect(b.coverage).not.toBe(a.coverage);
  });

  it("notifies subscribers on ingest", () => {
    const store = new ProcedureStore();
    let calls = 0;
    const unsub = store.subscribe(() => calls++);
    store.ingest(sample(0, 2));
    store.ingest(sample(0.5, 2));
    expect(calls).toBe(2);
    unsub();
    store.ingest(sample(1.0, 2));
    expect(calls).toBe(2);
  });
});

describe("ProcedureStore — findSampleAtTime", () => {
  it("returns null when empty", () => {
    expect(new ProcedureStore().findSampleAtTime(1)).toBeNull();
  });

  it("finds the nearest sample", () => {
    const store = new ProcedureStore();
    feed(store, [[0, "stomach"], [1, "stomach"], [2, "stomach"], [3, "stomach"]]);
    expect(store.findSampleAtTime(-5)?.t).toBe(0);
    expect(store.findSampleAtTime(0.4)?.t).toBe(0.5);
    expect(store.findSampleAtTime(1.0)?.t).toBe(1.0);
    expect(store.findSampleAtTime(99)?.t).toBe(1.5);
  });
});

describe("deriveVisibility", () => {
  it("reports nothing without data", () => {
    expect(deriveVisibility(null, null, "stationary").level).toBeNull();
  });

  it("is gated on model confidence", () => {
    const v = deriveVisibility(2.9, 0.2, "stationary");
    expect(v.level).toBeNull();
    expect(v.detail).toBe("Low model confidence");
  });

  it("maps the recent mean onto three states", () => {
    expect(deriveVisibility(3.0, 0.9, "stationary").level).toBe("adequate");
    expect(deriveVisibility(2.0, 0.9, "stationary").level).toBe("degraded");
    expect(deriveVisibility(1.0, 0.9, "stationary").level).toBe("obscured");
  });

  it("calls out withdrawal with an impaired view", () => {
    expect(deriveVisibility(1.2, 0.9, "withdrawal").detail).toBe(
      "Withdrawing with impaired view",
    );
  });
});

// --- ESGE stations ---------------------------------------------------------

type St = "unseen" | "candidate" | "observed";

/** Station status array from the observed keys (everything else unseen). */
function levels(
  observed: readonly StationKey[] = [],
  candidates: readonly StationKey[] = [],
): St[] {
  return STATION_ORDER.map((k) =>
    observed.includes(k) ? "observed" : candidates.includes(k) ? "candidate" : "unseen",
  );
}

function lmSample(
  t: number,
  opts: {
    status?: LandmarkStatus;
    top?: StationKey | null;
    current?: StationKey | null;
    observed?: readonly StationKey[];
    candidates?: readonly StationKey[];
    display?: boolean;
    auto?: readonly boolean[];
    version?: string;
    region?: AnatomicalRegion | null;
  } = {},
): FrameSample {
  return {
    ...sample(t, 2, opts.region ?? "stomach"),
    landmarkStatus: opts.status ?? "ok",
    landmarkTop: opts.top ?? null,
    landmarkConfidence: opts.top ? 0.9 : null,
    landmarkQuality: opts.top ? 0.8 : null,
    landmarkModelVersion: opts.version ?? "lm-1.0.0",
    stationsDisplay: opts.display ?? true,
    stationCurrent: opts.current ?? null,
    stationStatus: levels(opts.observed, opts.candidates),
    stationAutoEnabled: opts.auto ?? Array(10).fill(true),
    stationObservedEvents: [],
    stationBestEvents: [],
  };
}

const I = STATION_INDEX;

describe("ProcedureStore — stations: feature off", () => {
  it("stays absent and invisible, and the stations slice never changes", () => {
    const store = new ProcedureStore();
    const initial = store.getSnapshot().stations;
    expect(initial).toMatchObject({
      availability: "absent",
      display: false,
      visible: false,
      modelVersion: null,
      current: null,
      observedCount: 0,
      total: 10,
      exitingMissing: [],
    });
    expect(initial.status).toEqual(Array(10).fill("unseen"));
    expect(initial.manual).toEqual(Array(10).fill(null));
    expect(initial.observed).toEqual(Array(10).fill(false));
    expect(initial.observedAtT).toEqual(Array(10).fill(null));

    feed(store, [[2, "esophagus"], [3, "stomach"], [1, "duodenum"]]);
    const snap = store.getSnapshot();
    expect(snap.stations).toBe(initial);
    // ...and everything else behaves exactly as before.
    expect(snap.coverage.framesAnalysed).toBe(3);
    expect(snap.regions.visited).toEqual(["esophagus", "stomach", "duodenum"]);
  });

  it("reports no edges for PEACE-only frames", () => {
    const store = new ProcedureStore();
    expect(store.ingest(sample(0, 2)).stationEdges).toEqual([]);
  });
});

describe("ProcedureStore — stations: status", () => {
  it("keeps observed sticky even if the model later says unseen", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(10, { observed: ["antrum"] }));
    store.ingest(lmSample(10.5, { observed: [] })); // e.g. a backend restart
    const st = store.getSnapshot().stations;
    expect(st.status[I.antrum]).toBe("observed");
    expect(st.observed[I.antrum]).toBe(true);
    expect(st.observedCount).toBe(1);
  });

  it("times each station by the video clock of its first observation", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(3.0, { candidates: ["antrum"] }));
    store.ingest(lmSample(3.5, { observed: ["antrum"] }));
    store.ingest(lmSample(4.0, { observed: ["antrum", "incisura"] }));
    const st = store.getSnapshot().stations;
    expect(st.observedAtT[I.antrum]).toBe(3.5);
    expect(st.observedAtT[I.incisura]).toBe(4.0);
    expect(st.observedAtT[I.corpus_greater_curvature]).toBeNull();
    expect(st.status[I.antrum]).toBe("observed");
  });

  it("passes candidates through but never demotes an observed station to one", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { candidates: ["z_line"] }));
    expect(store.getSnapshot().stations.status[I.z_line]).toBe("candidate");
    store.ingest(lmSample(0.5, { observed: ["z_line"] }));
    store.ingest(lmSample(1.0, { candidates: ["z_line"] }));
    expect(store.getSnapshot().stations.status[I.z_line]).toBe("observed");
  });

  it("returns level-derived edges, once per station", () => {
    const store = new ProcedureStore();
    expect(store.ingest(lmSample(0, { candidates: ["antrum"] })).stationEdges).toEqual([]);
    // The backend's events list is empty (lost) — the edge still appears.
    expect(store.ingest(lmSample(0.5, { observed: ["antrum"] })).stationEdges).toEqual([
      "antrum",
    ]);
    expect(store.ingest(lmSample(1.0, { observed: ["antrum"] })).stationEdges).toEqual([]);
  });

  it("follows the latest current station and model version", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { current: "z_line", version: "mock-0.1" }));
    expect(store.getSnapshot().stations.current).toBe("z_line");
    expect(store.getSnapshot().stations.modelVersion).toBe("mock-0.1");
    store.ingest(lmSample(0.5, { current: null }));
    expect(store.getSnapshot().stations.current).toBeNull();
  });
});

describe("ProcedureStore — stations: manual override", () => {
  it("a rejection beats the model and survives later observed frames", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { observed: ["antrum"] }));
    store.setManualStation(I.antrum, "rejected");
    let st = store.getSnapshot().stations;
    expect(st.manual[I.antrum]).toBe("rejected");
    expect(st.observed[I.antrum]).toBe(false);
    expect(st.observedAtT[I.antrum]).toBeNull();
    expect(st.observedCount).toBe(0);

    for (let i = 1; i < 6; i++) store.ingest(lmSample(i * 0.5, { observed: ["antrum"] }));
    st = store.getSnapshot().stations;
    expect(st.manual[I.antrum]).toBe("rejected");
    expect(st.observed[I.antrum]).toBe(false);
    // The model's own record is untouched.
    expect(st.status[I.antrum]).toBe("observed");
  });

  it("a confirmation counts as observed, timed at the procedure clock", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, {}));
    store.ingest(lmSample(42, {}));
    store.setManualStation(I.incisura, "confirmed");
    const st = store.getSnapshot().stations;
    expect(st.observed[I.incisura]).toBe(true);
    expect(st.observedAtT[I.incisura]).toBe(42);
    expect(st.observedCount).toBe(1);
  });

  it("a confirmed station keeps the model's time when the model saw it", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(5, { observed: ["antrum"] }));
    store.ingest(lmSample(60, { observed: ["antrum"] }));
    store.setManualStation(I.antrum, "confirmed");
    expect(store.getSnapshot().stations.observedAtT[I.antrum]).toBe(5);
  });

  it("a confirmed station keeps its manual time when the model sees it later", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, {}));
    store.ingest(lmSample(100, {}));
    store.setManualStation(I.incisura, "confirmed");
    expect(store.getSnapshot().stations.observedAtT[I.incisura]).toBe(100);
    store.ingest(lmSample(200, { observed: ["incisura"] }));
    const st = store.getSnapshot().stations;
    expect(st.status[I.incisura]).toBe("observed");
    expect(st.observedAtT[I.incisura]).toBe(100);
  });

  it("clearing the mark hands the station back to the model", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { observed: ["antrum"] }));
    store.setManualStation(I.antrum, "rejected");
    store.setManualStation(I.antrum, null);
    expect(store.getSnapshot().stations.observed[I.antrum]).toBe(true);
  });

  it("does not report an edge for a station the clinician rejected or already confirmed", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, {}));
    store.setManualStation(I.antrum, "rejected");
    store.setManualStation(I.incisura, "confirmed");
    const { stationEdges } = store.ingest(
      lmSample(0.5, { observed: ["antrum", "incisura"] }),
    );
    expect(stationEdges).toEqual([]);
  });

  it("publishes on change, and ignores no-ops and bad indices", () => {
    const store = new ProcedureStore();
    let calls = 0;
    store.subscribe(() => calls++);
    store.setManualStation(I.antrum, "confirmed");
    expect(calls).toBe(1);
    store.setManualStation(I.antrum, "confirmed");
    store.setManualStation(-1, "confirmed");
    store.setManualStation(10, "confirmed");
    store.setManualStation(1.5, "confirmed");
    expect(calls).toBe(1);
  });
});

describe("ProcedureStore — stations: snapshot identity", () => {
  it("keeps the stations slice when nothing in it changed", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { observed: ["antrum"], current: "antrum" }));
    const a = store.getSnapshot().stations;
    store.ingest(lmSample(0.5, { observed: ["antrum"], current: "antrum" }));
    const b = store.getSnapshot().stations;
    expect(b).toBe(a);
  });

  it("publishes a new slice with NEW arrays when something changed", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { observed: ["antrum"] }));
    const a = store.getSnapshot().stations;
    store.ingest(lmSample(0.5, { observed: ["antrum", "incisura"] }));
    const b = store.getSnapshot().stations;
    expect(b).not.toBe(a);
    expect(b.status).not.toBe(a.status);
    expect(b.observed).not.toBe(a.observed);
    // The old snapshot was not mutated behind its readers' backs.
    expect(a.status[I.incisura]).toBe("unseen");
    expect(a.observedCount).toBe(1);
  });

  it("publishes a new slice on a manual mark", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, {}));
    const a = store.getSnapshot().stations;
    store.setManualStation(I.antrum, "confirmed");
    const b = store.getSnapshot().stations;
    expect(b).not.toBe(a);
    expect(a.manual[I.antrum]).toBeNull();
  });
});

describe("ProcedureStore — stations: availability", () => {
  it("turns on after one usable frame; uncertain/low_quality count as usable", () => {
    for (const status of ["ok", "uncertain", "low_quality"] as const) {
      const store = new ProcedureStore();
      store.ingest(lmSample(0, { status }));
      expect(store.getSnapshot().stations.availability).toBe("ok");
      expect(store.getSnapshot().stations.visible).toBe(true);
    }
  });

  it("a skipped frame is not evidence: it neither turns availability on nor resets a bad run", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { status: "skipped" }));
    expect(store.getSnapshot().stations.availability).toBe("absent");
    expect(store.getSnapshot().stations.visible).toBe(false);
    store.ingest(lmSample(0.5, { status: "ok" }));
    expect(store.getSnapshot().stations.availability).toBe("ok");
    store.ingest(lmSample(1, { status: "skipped" }));
    expect(store.getSnapshot().stations.availability).toBe("ok");
  });

  // landmark_every_n = 2 (the G-SERVE fallback): every other frame is skipped
  // without looking at the layout, so a bad source alternates bad/skipped.
  for (const bad of ["unsupported_layout", "error"] as const) {
    it(`latches ${bad} after ${STATION_AVAILABILITY_RELEASE_FRAMES} bad evaluated frames, skips interleaved`, () => {
      const store = new ProcedureStore();
      let t = 0;
      let evaluated = 0;
      for (let i = 0; i < 40; i++) {
        const status = i % 2 === 0 ? bad : "skipped";
        store.ingest(lmSample((t += 0.5), { status }));
        if (status === bad) evaluated += 1;
        const st = store.getSnapshot().stations;
        if (evaluated < STATION_AVAILABILITY_RELEASE_FRAMES) {
          expect(st.availability).toBe("absent");
        } else {
          expect(st.availability).toBe(bad);
          expect(st.visible).toBe(false);
        }
      }
      // Once latched, only a usable evaluated frame releases it — not a skip.
      store.ingest(lmSample((t += 0.5), { status: "skipped" }));
      expect(store.getSnapshot().stations.availability).toBe(bad);
      store.ingest(lmSample((t += 0.5), { status: "uncertain" }));
      expect(store.getSnapshot().stations.availability).toBe("ok");
    });

    it(`leaves ok for ${bad} mid-procedure even when every other frame is skipped`, () => {
      const store = new ProcedureStore();
      store.ingest(lmSample(0, { status: "ok" }));
      let t = 0;
      for (let i = 0; i < 2 * STATION_AVAILABILITY_RELEASE_FRAMES - 1; i++) {
        store.ingest(lmSample((t += 0.5), { status: i % 2 === 0 ? bad : "skipped" }));
      }
      expect(store.getSnapshot().stations.availability).toBe(bad);
    });
  }

  it("only leaves ok after a run of unsupported/error frames", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { status: "ok" }));
    const n = STATION_AVAILABILITY_RELEASE_FRAMES;
    for (let i = 1; i < n; i++) {
      store.ingest(lmSample(i * 0.5, { status: i % 2 ? "unsupported_layout" : "error" }));
      expect(store.getSnapshot().stations.availability).toBe("ok");
    }
    store.ingest(lmSample(n * 0.5, { status: "unsupported_layout" }));
    expect(store.getSnapshot().stations.availability).toBe("unsupported_layout");
    expect(store.getSnapshot().stations.visible).toBe(false);

    // One usable frame brings it straight back.
    store.ingest(lmSample(n * 0.5 + 0.5, { status: "low_quality" }));
    expect(store.getSnapshot().stations.availability).toBe("ok");
  });

  it("an interrupted bad run resets the count", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { status: "ok" }));
    let t = 0;
    for (let round = 0; round < 4; round++) {
      for (let i = 0; i < STATION_AVAILABILITY_RELEASE_FRAMES - 1; i++) {
        store.ingest(lmSample((t += 0.5), { status: "error" }));
      }
      store.ingest(lmSample((t += 0.5), { status: "ok" }));
    }
    expect(store.getSnapshot().stations.availability).toBe("ok");
  });

  it("a source that is never supported ends up unsupported, not flickering", () => {
    const store = new ProcedureStore();
    for (let i = 0; i < STATION_AVAILABILITY_RELEASE_FRAMES; i++) {
      expect(store.getSnapshot().stations.availability).toBe("absent");
      store.ingest(lmSample(i * 0.5, { status: "unsupported_layout" }));
    }
    expect(store.getSnapshot().stations.availability).toBe("unsupported_layout");
  });

  it("keeps current and auto_enabled from the last good frame across error frames", () => {
    const store = new ProcedureStore();
    const auto = Array(10).fill(true);
    auto[I.incisura] = false;
    store.ingest(lmSample(0, { current: "antrum", observed: ["antrum"], auto }));
    // The backend's blank-tracker error block: nothing current, nothing enabled.
    store.ingest(
      lmSample(0.5, { status: "error", current: null, auto: Array(10).fill(false) }),
    );
    const st = store.getSnapshot().stations;
    expect(st.current).toBe("antrum");
    expect(st.autoEnabled).toEqual(auto);
    expect(st.observed[I.antrum]).toBe(true);
  });

  it("display=false (shadow mode) is never visible, but still tracks", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { display: false, observed: ["antrum"] }));
    const st = store.getSnapshot().stations;
    expect(st.availability).toBe("ok");
    expect(st.display).toBe(false);
    expect(st.visible).toBe(false);
    expect(st.observed[I.antrum]).toBe(true);
  });
});

describe("ProcedureStore — stations: exitingMissing", () => {
  it("is empty throughout insertion, then lists what is missing on the way out", () => {
    const store = new ProcedureStore();
    let t = 0;
    // Insertion: esophagus stations, current esophageal, nothing gastric yet.
    for (const k of ["esophagus_proximal", "esophagus_distal", "z_line"] as const) {
      store.ingest(lmSample((t += 0.5), { current: k, observed: ESOPHAGEAL }));
      expect(store.getSnapshot().stations.exitingMissing).toEqual([]);
    }
    // Stomach and duodenum: some stations observed.
    const seen: StationKey[] = [...ESOPHAGEAL, "duodenal_bulb", "duodenum_descending", "antrum"];
    store.ingest(lmSample((t += 0.5), { current: "antrum", observed: seen }));
    expect(store.getSnapshot().stations.exitingMissing).toEqual([]);

    // Back in the esophagus: the rest is missing.
    store.ingest(lmSample((t += 0.5), { current: "esophagus_distal", observed: seen }));
    expect(store.getSnapshot().stations.exitingMissing).toEqual([
      "cardia_fundus_retroflex",
      "lesser_curvature_retroflex",
      "incisura",
      "corpus_greater_curvature",
    ]);

    // A manual confirmation removes a station from the list.
    store.setManualStation(I.incisura, "confirmed");
    expect(store.getSnapshot().stations.exitingMissing).not.toContain("incisura");
  });

  it("never lists manual-only (not auto-enabled) stations", () => {
    const store = new ProcedureStore();
    const auto = Array(10).fill(true);
    auto[I.lesser_curvature_retroflex] = false;
    store.ingest(lmSample(0, { current: "antrum", observed: ["antrum"], auto }));
    store.ingest(lmSample(0.5, { current: "z_line", observed: ["antrum"], auto }));
    const missing = store.getSnapshot().stations.exitingMissing;
    expect(missing).toContain("incisura");
    expect(missing).not.toContain("lesser_curvature_retroflex");
    expect(store.getSnapshot().stations.autoEnabled[I.lesser_curvature_retroflex]).toBe(false);
  });

  it("is empty when every gastroduodenal station is observed", () => {
    const store = new ProcedureStore();
    store.ingest(lmSample(0, { current: "antrum", observed: STATION_ORDER }));
    store.ingest(lmSample(0.5, { current: "z_line", observed: STATION_ORDER }));
    expect(store.getSnapshot().stations.exitingMissing).toEqual([]);
    expect(store.getSnapshot().stations.observedCount).toBe(10);
  });
});
