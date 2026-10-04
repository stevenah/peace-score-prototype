import { describe, it, expect } from "vitest";
import {
  evaluateCapture,
  initialTriggerState,
  type TriggerState,
} from "@/lib/live/capture-triggers";
import { FILMSTRIP_MIN_GAP_S } from "@/lib/live/config";
import type { FrameSample } from "@/lib/live/procedure-types";
import type {
  AnatomicalRegion,
  MotionDirection,
  PeaceScore,
  StationKey,
} from "@/lib/types";

function s(
  t: number,
  score: PeaceScore,
  region: AnatomicalRegion | null = "stomach",
  motion: MotionDirection = "stationary",
): FrameSample {
  return {
    seq: 0,
    t,
    score,
    scoreConfidence: 0.9,
    expectedScore: null,
    region,
    motion,
    motionConfidence: 0.9,
    frameIndex: 0,
    processingTimeMs: 20,
  };
}

describe("capture triggers", () => {
  it("captures on entering a region", () => {
    const r = evaluateCapture(initialTriggerState(), s(0, 2, "esophagus"));
    expect(r.reason).toBe("region-entry");
  });

  it("does not capture again while in the same region", () => {
    let st = initialTriggerState();
    let r = evaluateCapture(st, s(0, 2, "esophagus"));
    st = r.next;
    r = evaluateCapture(st, s(0.5, 2, "esophagus"));
    expect(r.reason).toBeNull();
  });

  it("captures the first frame of a bad-withdrawal episode, once", () => {
    let st = initialTriggerState();
    st = evaluateCapture(st, s(0, 3, "stomach")).next; // region entry
    let captured = 0;
    // 20 consecutive bad frames over 10 seconds
    for (let i = 1; i <= 20; i++) {
      const r = evaluateCapture(st, s(i * 0.5, 1, "stomach", "withdrawal"));
      st = r.next;
      if (r.reason === "quality-alert") captured++;
    }
    expect(captured).toBe(1);
  });

  it("respects the cooldown between non-region captures", () => {
    let st = initialTriggerState();
    st = evaluateCapture(st, s(0, 0, "stomach")).next;
    // improving scores would each be a new best, but the cooldown gates them
    let captured = 0;
    for (let i = 1; i <= 6; i++) {
      const r = evaluateCapture(st, s(i * 0.5, Math.min(3, i) as PeaceScore, "stomach"));
      st = r.next;
      if (r.reason) captured++;
    }
    // 6 frames over 3s, under the 5s cooldown
    expect(FILMSTRIP_MIN_GAP_S).toBeGreaterThan(3);
    expect(captured).toBe(0);
  });

  it("region entry is never suppressed by the cooldown", () => {
    let st: TriggerState = initialTriggerState();
    st = evaluateCapture(st, s(0, 2, "esophagus")).next;
    const r = evaluateCapture(st, s(0.5, 2, "stomach"));
    expect(r.reason).toBe("region-entry");
  });

  it("never fires a detection trigger — there is no detector", () => {
    let st = initialTriggerState();
    const reasons: (string | null)[] = [];
    for (let i = 0; i < 40; i++) {
      const r = evaluateCapture(
        st,
        s(i * 0.5, (i % 4) as PeaceScore, "stomach", i % 3 === 0 ? "withdrawal" : "insertion"),
      );
      st = r.next;
      reasons.push(r.reason);
    }
    expect(reasons).not.toContain("detection");
  });
});

describe("capture triggers — ESGE stations", () => {
  const VISIBLE = { visible: true };
  const HIDDEN = { visible: false };

  function withTop(sample: FrameSample, top: StationKey | null): FrameSample {
    return { ...sample, landmarkStatus: "ok", landmarkTop: top };
  }

  it("captures a newly observed station, ahead of a region entry", () => {
    const r = evaluateCapture(initialTriggerState(), withTop(s(0, 2, "stomach"), "antrum"), {
      stations: VISIBLE,
      stationEdges: ["antrum"],
    });
    expect(r.reason).toBe("station-observed");
    expect(r.station).toBe("antrum");
  });

  it("bypasses the shared cooldown", () => {
    let st = initialTriggerState();
    st = evaluateCapture(st, s(0, 2, "stomach")).next; // region entry at t=0
    const r = evaluateCapture(st, withTop(s(1, 2, "stomach"), "incisura"), {
      stations: VISIBLE,
      stationEdges: ["incisura"],
    });
    expect(1).toBeLessThan(FILMSTRIP_MIN_GAP_S);
    expect(r.reason).toBe("station-observed");
    expect(r.next.lastCaptureT).toBe(1);
  });

  it("prefers the edge this frame shows when several land together", () => {
    const r = evaluateCapture(initialTriggerState(), withTop(s(0, 2), "incisura"), {
      stations: VISIBLE,
      stationEdges: ["antrum", "incisura"],
    });
    expect(r.station).toBe("incisura");
  });

  it("never fires while stations are not visible (shadow mode)", () => {
    let st = initialTriggerState();
    st = evaluateCapture(st, s(0, 2, "stomach")).next;
    const r = evaluateCapture(st, withTop(s(1, 2, "stomach"), "antrum"), {
      stations: HIDDEN,
      stationEdges: ["antrum"],
    });
    expect(r.reason).toBeNull();
    expect(r.station).toBeNull();
  });

  it("behaves exactly as before without landmarks", () => {
    const frames = Array.from({ length: 60 }, (_, i) =>
      s(
        i * 0.5,
        (i % 4) as PeaceScore,
        (["esophagus", "stomach", "duodenum"] as const)[Math.floor(i / 20)],
        i % 3 === 0 ? "withdrawal" : "insertion",
      ),
    );
    let a = initialTriggerState();
    let b = initialTriggerState();
    for (const f of frames) {
      const ra = evaluateCapture(a, f);
      const rb = evaluateCapture(b, f, { stations: VISIBLE, stationEdges: [] });
      expect(rb.reason).toBe(ra.reason);
      expect(rb.next).toEqual(ra.next);
      expect(ra.station).toBeNull();
      a = ra.next;
      b = rb.next;
    }
  });
});
