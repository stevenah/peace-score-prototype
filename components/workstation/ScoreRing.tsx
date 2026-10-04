"use client";

import type { PeaceScore } from "@/lib/types";

const RING_STROKE: Record<PeaceScore, string> = {
  0: "stroke-peace-0",
  1: "stroke-peace-1",
  2: "stroke-peace-2",
  3: "stroke-peace-3",
};

const RING_TEXT: Record<PeaceScore, string> = {
  0: "text-peace-0",
  1: "text-peace-1",
  2: "text-peace-2",
  3: "text-peace-3",
};

interface ScoreRingProps {
  /** Settled score — see ProcedureStore's display-score debounce. */
  score: PeaceScore | null;
  label: string | null;
  /** Continuous session mean 0..3, drawn as a quiet inner arc. */
  sessionMean: number | null;
  size?: number;
  dimmed?: boolean;
}

/**
 * Four discrete segments rather than a score/3 arc.
 *
 * The value is ordinal with four states, and "three of four lit" is countable at
 * a glance without reading the numeral. Decisively: score 0 drawn as an arc is an
 * empty ring, indistinguishable from "not reading" — as a lit red segment it is
 * unmistakably "reading, and it is bad".
 *
 * All lit segments take the current score's colour, so the ring reads as one
 * value at distance instead of as a legend.
 */
export function ScoreRing({
  score,
  label,
  sessionMean,
  size = 108,
  dimmed = false,
}: ScoreRingProps) {
  const has = score !== null;

  return (
    <div
      className={`relative shrink-0 transition-opacity duration-300 ${dimmed ? "opacity-50" : ""}`}
      style={{ width: size, height: size }}
    >
      <svg
        viewBox="0 0 120 120"
        className="h-full w-full -rotate-90"
        role="img"
        aria-label={
          has
            ? `PEACE score ${score} of 3${label ? `, ${label}` : ""}`
            : "PEACE score not yet available"
        }
      >
        {/* 4 segments: 22% arc + 3% gap = 25% pitch. pathLength normalises the
            circumference to 100 units, so no 2*pi*r arithmetic is involved. */}
        {[0, 1, 2, 3].map((i) => {
          const lit = has && i <= (score as number);
          return (
            <circle
              key={i}
              cx="60"
              cy="60"
              r="50"
              pathLength={100}
              fill="none"
              strokeWidth={9}
              strokeDasharray="22 78"
              strokeDashoffset={-25 * i}
              className={`transition-[stroke] duration-200 ease-out motion-reduce:transition-none ${
                lit ? RING_STROKE[score as PeaceScore] : "stroke-white/15"
              }`}
            />
          );
        })}

        {sessionMean !== null && (
          <circle
            cx="60"
            cy="60"
            r="38"
            pathLength={100}
            fill="none"
            strokeWidth={2}
            strokeLinecap="butt"
            strokeDasharray={`${(sessionMean / 3) * 100} 100`}
            className="stroke-white/25 transition-[stroke-dasharray] duration-700 ease-out motion-reduce:transition-none"
          />
        )}
      </svg>

      <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-1">
        {has ? (
          <>
            <div className="flex items-baseline gap-0.5">
              <span className={`ws-num-md ${RING_TEXT[score as PeaceScore]}`}>
                {score}
              </span>
              <span className="font-mono text-[13px] font-light text-ws-faint">
                /3
              </span>
            </div>
            <span
              className={`text-[11px] font-medium leading-none ${RING_TEXT[score as PeaceScore]}`}
            >
              {label}
            </span>
          </>
        ) : (
          <>
            <span className="ws-num-md text-ws-faint">–</span>
            <span className="text-[11px] font-medium leading-none text-ws-faint">
              Awaiting
            </span>
          </>
        )}
      </div>
    </div>
  );
}
