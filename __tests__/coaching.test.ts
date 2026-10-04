import { afterEach, beforeEach, describe, it, expect, vi } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { useCoaching } from "@/hooks/useCoaching";
import {
  initialCoachState,
  missingStationsText,
  nextCoachState,
  type CoachInputs,
  type CoachState,
} from "@/lib/live/coaching";
import type { StationKey } from "@/lib/types";

const STREAM: CoachInputs = {
  phase: "streaming",
  recentMean: 3,
  motion: "stationary",
  displayScore: 3,
};

const DIRTY_WITHDRAWAL: CoachInputs = {
  phase: "streaming",
  recentMean: 1.2,
  motion: "withdrawal",
  displayScore: 1,
};

/** Drive the machine at 2 Hz for `frames`, returning every state. */
function run(
  inputsAt: (frame: number) => CoachInputs,
  frames: number,
  startState: CoachState = initialCoachState(),
  stepMs = 500,
): CoachState[] {
  const out: CoachState[] = [];
  let s = startState;
  for (let i = 0; i < frames; i++) {
    s = nextCoachState(s, inputsAt(i), i * stepMs);
    out.push(s);
  }
  return out;
}

function transitions(states: CoachState[]): number {
  let n = 0;
  let last: string | null = null;
  for (const s of states) {
    const id = s.active?.id ?? null;
    if (id !== last) n++;
    last = id;
  }
  return n - 1; // don't count the initial state
}

describe("coaching — arming", () => {
  it("shows nothing while everything is fine", () => {
    const states = run(() => STREAM, 20);
    expect(states.every((s) => s.active === null)).toBe(true);
  });

  it("does not fire an alert on a single bad frame", () => {
    let s = initialCoachState();
    s = nextCoachState(s, DIRTY_WITHDRAWAL, 0);
    expect(s.active).toBeNull();
  });

  it("fires the alert once the condition persists", () => {
    const states = run(() => DIRTY_WITHDRAWAL, 4);
    expect(states[0].active).toBeNull();
    expect(states[1].active?.id).toBe("dirty-withdrawal");
    expect(states[1].active?.level).toBe("alert");
  });

  it("shows connection phases immediately — they need no hysteresis", () => {
    let s = initialCoachState();
    s = nextCoachState(s, { ...STREAM, phase: "connecting" }, 0);
    expect(s.active?.id).toBe("connecting");
  });
});

describe("coaching — the strobe test", () => {
  it("makes at most one transition per 3s under a 2Hz alternating input", () => {
    // The pathological case: the score sits exactly on the 1|2 boundary and
    // flips every single frame for 60 seconds.
    const states = run(
      (i) => (i % 2 === 0 ? DIRTY_WITHDRAWAL : STREAM),
      120, // 60s at 2Hz
    );
    const t = transitions(states);
    const seconds = 60;
    expect(t).toBeLessThanOrEqual(Math.ceil(seconds / 3));
    // And in practice it should be far quieter than the bound.
    expect(t).toBeLessThanOrEqual(4);
  });

  it("holds an alert for its minimum dwell even if the condition clears at once", () => {
    // Arm the alert...
    let s = initialCoachState();
    s = nextCoachState(s, DIRTY_WITHDRAWAL, 0);
    s = nextCoachState(s, DIRTY_WITHDRAWAL, 500);
    expect(s.active?.id).toBe("dirty-withdrawal");
    const shownAt = 500;

    // ...then the condition immediately goes away.
    let t = shownAt;
    while (t < shownAt + 2500) {
      t += 500;
      s = nextCoachState(s, STREAM, t);
      expect(s.active?.id).toBe("dirty-withdrawal");
    }
    // Past MIN_DWELL (3000) and CLEAR (1500) it finally goes.
    t += 2000;
    s = nextCoachState(s, STREAM, t);
    expect(s.active).toBeNull();
  });
});

describe("coaching — priority", () => {
  it("prefers the alert over lower-severity guidance", () => {
    const states = run(
      () => ({
        phase: "streaming",
        recentMean: 1.0, // would also trigger "obscured"
        motion: "withdrawal",
        displayScore: 1,
      }),
      6,
    );
    expect(states.at(-1)?.active?.id).toBe("dirty-withdrawal");
  });

  it("prefers a hard error over everything", () => {
    let s = initialCoachState();
    s = nextCoachState(s, { ...DIRTY_WITHDRAWAL, phase: "error" }, 0);
    expect(s.active?.id).toBe("error");
  });
});

describe("coaching — repeat suppression", () => {
  it("does not re-nag with the same guidance inside REPEAT_MS", () => {
    const obscured: CoachInputs = {
      phase: "streaming",
      recentMean: 1.0,
      motion: "stationary",
      displayScore: 1,
    };
    // Show it, clear it, then re-trigger 5s later.
    let s = initialCoachState();
    let t = 0;
    for (let i = 0; i < 4; i++, t += 500) s = nextCoachState(s, obscured, t);
    expect(s.active?.id).toBe("obscured");

    for (let i = 0; i < 20; i++, t += 500) s = nextCoachState(s, STREAM, t);
    expect(s.active).toBeNull();
    const dismissedAt = s.lastDismissedMs["obscured"];
    expect(dismissedAt).toBeDefined();

    // Re-trigger well inside the 20s repeat window.
    for (let i = 0; i < 6; i++, t += 500) s = nextCoachState(s, obscured, t);
    expect(t - dismissedAt).toBeLessThan(20000);
    expect(s.active).toBeNull();
  });
});

describe("coaching — stations missing on exit", () => {
  const MISSING: CoachInputs = {
    ...STREAM,
    stationsMissing: ["lesser_curvature_retroflex", "incisura", "corpus_greater_curvature"],
  };

  it("never fires without a missing list (insertion, shadow mode, feature off)", () => {
    const states = run(() => ({ ...STREAM, stationsMissing: [] }), 20);
    expect(states.every((s) => s.active === null)).toBe(true);
    const off = run(() => STREAM, 20);
    expect(off.every((s) => s.active === null)).toBe(true);
  });

  it("arms like guidance (not instantly) and names what is missing", () => {
    const states = run(() => MISSING, 6);
    expect(states[0].active).toBeNull();
    const shown = states.find((s) => s.active)?.active;
    expect(shown?.id).toBe("stations-missing");
    expect(shown?.level).toBe("guide");
    expect(shown?.text).toBe("Not yet observed: Lesser curve, Incisura (+1)");
  });

  it("only fires while streaming", () => {
    const states = run(() => ({ ...MISSING, phase: "stalled" }), 6);
    expect(states.at(-1)?.active?.id).toBe("stalled");
  });

  it("refreshes its text as stations get observed, without resetting its dwell", () => {
    let s = initialCoachState();
    let t = 0;
    for (let i = 0; i < 4; i++, t += 500) s = nextCoachState(s, MISSING, t);
    expect(s.active?.id).toBe("stations-missing");
    const since = s.activeSinceMs;

    s = nextCoachState(s, { ...STREAM, stationsMissing: ["corpus_greater_curvature"] }, t);
    expect(s.active?.text).toBe("Not yet observed: Corpus");
    expect(s.activeSinceMs).toBe(since);

    // Unchanged inputs keep the very same prompt object.
    const before = s.active;
    s = nextCoachState(s, { ...STREAM, stationsMissing: ["corpus_greater_curvature"] }, t + 500);
    expect(s.active).toBe(before);
  });

  /** Armed and showing "Not yet observed: <keys>", at 2 Hz from t=0. */
  function armed(keys: StationKey[]): { s: CoachState; t: number } {
    let s = initialCoachState();
    let t = 0;
    const inputs = { ...STREAM, stationsMissing: keys, stationsPending: keys };
    for (let i = 0; i < 4; i++, t += 500) s = nextCoachState(s, inputs, t);
    expect(s.active?.id).toBe("stations-missing");
    return { s, t };
  }

  it("drops the prompt at once when its last missing station is observed", () => {
    const { s: shown, t } = armed(["incisura"]);
    // Incisura observed: the exit list empties and nothing is pending.
    const s = nextCoachState(shown, { ...STREAM, stationsPending: [] }, t);
    expect(s.active).toBeNull();
    expect(s.lastDismissedMs["stations-missing"]).toBe(t);
  });

  it("without a pending list, an emptied exit list clears at once too", () => {
    const { s: shown, t } = armed(["incisura"]);
    expect(nextCoachState(shown, STREAM, t).active).toBeNull();
  });

  it("keeps a true wording while clearing after the tracker loses its current station", () => {
    let { s, t } = armed(["incisura", "corpus_greater_curvature"]);
    // current -> null: the exit list empties, but both stations are still missing.
    const lost = (pending: StationKey[]): CoachInputs => ({ ...STREAM, stationsPending: pending });
    s = nextCoachState(s, lost(["incisura", "corpus_greater_curvature"]), (t += 250));
    expect(s.active?.text).toBe("Not yet observed: Incisura, Corpus");
    expect(s.clearingSinceMs).toBe(t);

    // Incisura observed mid-clear: the wording follows at once.
    s = nextCoachState(s, lost(["corpus_greater_curvature"]), (t += 250));
    expect(s.active?.text).toBe("Not yet observed: Corpus");

    // Back in the esophagus within the clear window: the same prompt holds.
    const back = { ...STREAM, stationsMissing: ["corpus_greater_curvature"] as StationKey[] };
    s = nextCoachState(s, { ...back, stationsPending: back.stationsMissing }, (t += 250));
    expect(s.active?.id).toBe("stations-missing");
    expect(s.clearingSinceMs).toBeNull();
  });

  it("never names an observed station, replayed at 250 ms ticks", () => {
    // The scenario from review: "Not yet observed: Incisura" is up on exit,
    // then Incisura is observed at t=5000.
    let s = initialCoachState();
    const texts: Array<[number, string | null]> = [];
    for (let t = 0; t <= 8000; t += 250) {
      const observed = t >= 5000;
      const keys: StationKey[] = observed ? [] : ["incisura"];
      s = nextCoachState(s, { ...STREAM, stationsMissing: keys, stationsPending: keys }, t);
      texts.push([t, s.active?.text ?? null]);
    }
    expect(texts.some(([t, text]) => t < 5000 && text === "Not yet observed: Incisura")).toBe(true);
    expect(texts.filter(([t]) => t >= 5000).every(([, text]) => text === null)).toBe(true);
  });

  it("yields to the dirty-withdrawal alert, and outranks 'obscured'", () => {
    const dirty = run(() => ({ ...DIRTY_WITHDRAWAL, stationsMissing: MISSING.stationsMissing }), 6);
    expect(dirty.at(-1)?.active?.id).toBe("dirty-withdrawal");

    const obscured = run(
      () => ({ ...STREAM, recentMean: 1.0, stationsMissing: MISSING.stationsMissing }),
      6,
    );
    expect(obscured.at(-1)?.active?.id).toBe("stations-missing");
  });
});

describe("missingStationsText", () => {
  it("names up to two stations and counts the rest", () => {
    expect(missingStationsText(["incisura"])).toBe("Not yet observed: Incisura");
    expect(missingStationsText(["antrum", "incisura"])).toBe(
      "Not yet observed: Antrum, Incisura",
    );
    expect(
      missingStationsText(["duodenal_bulb", "antrum", "incisura", "corpus_greater_curvature"]),
    ).toBe("Not yet observed: Bulb, Antrum (+2)");
  });
});

describe("useCoaching — prompt text follows the inputs", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("re-publishes the prompt when only its wording changes", () => {
    const missing = (keys: StationKey[]): CoachInputs => ({ ...STREAM, stationsMissing: keys });
    const { result, rerender } = renderHook(({ inputs }) => useCoaching(inputs), {
      initialProps: { inputs: missing(["incisura", "antrum"]) },
    });
    // Tick until armed (ARM_FRAMES evaluations at 250 ms).
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(result.current?.id).toBe("stations-missing");
    expect(result.current?.text).toBe("Not yet observed: Incisura, Antrum");

    rerender({ inputs: missing(["antrum"]) });
    expect(result.current?.id).toBe("stations-missing");
    expect(result.current?.text).toBe("Not yet observed: Antrum");
  });
});
