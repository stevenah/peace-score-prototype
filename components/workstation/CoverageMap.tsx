"use client";

import { useId } from "react";
import { REGION_LABELS, REGION_ORDER } from "@/lib/constants";
import type { AnatomicalRegion, PeaceScore } from "@/lib/types";

/**
 * Upper GI tract, drawn for a dark rail.
 *
 * The esophagus and duodenum are genuinely tubular and are stroked centrelines.
 * The stomach is a sac and is a filled silhouette — a thick stroked centreline
 * renders as a bent sausage with a crescent hollow, which does not read as a
 * stomach at any size.
 *
 * Coverage is shown by clipping each region to a rect that grows downward, so the
 * fill advances in the same direction the scope travels (down the esophagus, into
 * the stomach, on to the duodenum). One consistent metaphor across all three.
 *
 * Drawn rather than reusing /public/{esophagus,stomach,duodenum}.png: those are
 * three different crops at three different scales, their fills are baked so colour
 * cannot be driven from the score, and they include the colon, which is not scored
 * here.
 */
const TRACT: Record<
  AnatomicalRegion,
  {
    /** Vertical extent used to grow the coverage clip, [top, bottom]. */
    span: [number, number];
    label: { x: number; y: number; anchor: "start" | "middle" | "end" };
  }
> = {
  esophagus: { span: [6, 82], label: { x: 120, y: 46, anchor: "start" } },
  stomach: { span: [66, 186], label: { x: 14, y: 210, anchor: "start" } },
  duodenum: { span: [140, 212], label: { x: 196, y: 231, anchor: "end" } },
};

const ESOPHAGUS_D = "M 92 14 C 91 38 90 58 90 74";
const STOMACH_D =
  "M 90 72 C 64 68 42 88 39 118 C 36 150 55 178 86 183 C 108 186 127 174 138 158 C 141 154 144 152 148 150 C 139 145 131 142 123 136 C 108 123 98 99 96 74 Z";
const DUODENUM_D = "M 148 150 C 165 151 175 165 173 182 C 171 199 156 208 141 203";

const ESOPHAGUS_W = 15;
const DUODENUM_W = 18;

const FILL_CLASS: Record<PeaceScore, string> = {
  0: "text-peace-0",
  1: "text-peace-1",
  2: "text-peace-2",
  3: "text-peace-3",
};

export interface RegionCoverageState {
  /** 0..100 — dwell as a share of the total procedure target. */
  fill: number;
  active: boolean;
  score: PeaceScore | null;
}

interface CoverageMapProps {
  states: Record<AnatomicalRegion, RegionCoverageState>;
  className?: string;
  dimmed?: boolean;
}

export function CoverageMap({ states, className, dimmed }: CoverageMapProps) {
  const uid = useId().replace(/:/g, "");

  const clipHeight = (region: AnatomicalRegion) => {
    const [top, bottom] = TRACT[region].span;
    const pct = Math.max(0, Math.min(100, states[region].fill)) / 100;
    return top + (bottom - top) * pct;
  };

  return (
    <svg
      viewBox="0 0 200 240"
      className={`${className ?? ""} transition-opacity duration-300 ${dimmed ? "opacity-50" : ""}`}
      role="img"
      aria-label={describeCoverage(states)}
    >
      <defs>
        {REGION_ORDER.map((region) => (
          <clipPath key={region} id={`${uid}-${region}`}>
            <rect x="0" y="0" width="200" height={clipHeight(region)} />
          </clipPath>
        ))}
      </defs>

      {/* Silhouette. Always drawn — a missing organ reads as broken, not absent. */}
      <g opacity={0.72}>
        <path
          d={ESOPHAGUS_D}
          fill="none"
          stroke="var(--ws-organ-idle)"
          strokeWidth={ESOPHAGUS_W}
          strokeLinecap="round"
        />
        <path d={STOMACH_D} fill="var(--ws-organ-idle)" />
        <path
          d={DUODENUM_D}
          fill="none"
          stroke="var(--ws-organ-idle)"
          strokeWidth={DUODENUM_W}
          strokeLinecap="round"
        />
      </g>

      {/* Coverage fill, one clipped group per region. */}
      {states.esophagus.score !== null && (
        <g
          clipPath={`url(#${uid}-esophagus)`}
          className={FILL_CLASS[states.esophagus.score]}
        >
          <path
            d={ESOPHAGUS_D}
            fill="none"
            stroke="currentColor"
            strokeWidth={ESOPHAGUS_W}
            strokeLinecap="round"
            opacity={0.95}
          />
        </g>
      )}
      {states.stomach.score !== null && (
        <g
          clipPath={`url(#${uid}-stomach)`}
          className={FILL_CLASS[states.stomach.score]}
        >
          <path d={STOMACH_D} fill="currentColor" opacity={0.95} />
        </g>
      )}
      {states.duodenum.score !== null && (
        <g
          clipPath={`url(#${uid}-duodenum)`}
          className={FILL_CLASS[states.duodenum.score]}
        >
          <path
            d={DUODENUM_D}
            fill="none"
            stroke="currentColor"
            strokeWidth={DUODENUM_W}
            strokeLinecap="round"
            opacity={0.95}
          />
        </g>
      )}

      {REGION_ORDER.map((region) => {
        const { label } = TRACT[region];
        return (
          <text
            key={region}
            x={label.x}
            y={label.y}
            textAnchor={label.anchor}
            className={`text-[9px] font-medium uppercase tracking-[0.1em] ${
              states[region].active ? "fill-white/80" : "fill-white/35"
            }`}
          >
            {REGION_LABELS[region]}
          </text>
        );
      })}
    </svg>
  );
}

function describeCoverage(
  states: Record<AnatomicalRegion, RegionCoverageState>,
): string {
  const parts = REGION_ORDER.map((r) => {
    const s = states[r];
    if (s.score === null) return `${REGION_LABELS[r]} not yet inspected`;
    return `${REGION_LABELS[r]}, score ${s.score} of 3`;
  });
  return `Coverage: ${parts.join(". ")}.`;
}
