"use client";

import { useId } from "react";
import { REGION_LABELS, REGION_ORDER } from "@/lib/constants";
import { STATION_ORDER } from "@/lib/live/stations";
import type { AnatomicalRegion, PeaceScore, StationKey } from "@/lib/types";
import type { StationRowState } from "./StationChecklist";

/**
 * Upper GI tract, drawn for a dark rail, in the standard anterior view: fundus
 * and greater curvature to the viewer's right, pylorus and duodenum to the left.
 * That is the orientation of every endoscopy atlas, and of the organ plates on
 * the results page.
 *
 * The esophagus and duodenum are genuinely tubular and are stroked centrelines.
 * Both run off the edge of the frame, because both continue past what is scored.
 * The stomach is a sac and is a filled silhouette — a thick stroked centreline
 * renders as a bent sausage, which does not read as a stomach at any size. It is
 * drawn last, so each junction is the stomach's own outline rather than a tube's
 * end cap.
 *
 * A scored region takes its PEACE colour twice: dimly all over, so "inspected,
 * and how clean" reads from the whole organ, and at full strength down to its
 * coverage level. The level is a clip that grows downward, the direction the
 * scope travels — one consistent metaphor across all three regions.
 *
 * Drawn rather than reusing /public/{esophagus,stomach,duodenum}.png: those are
 * three different crops at three different scales, their fills are baked so colour
 * cannot be driven from the score, and they include the colon, which is not scored
 * here.
 */
const VIEW_W = 240;
const VIEW_H = 250;

const ESOPHAGUS_D = "M 99 -6 C 98 28 100 58 116 88";
const STOMACH_D =
  "M 107 86 C 110 81 117 79 122 80 C 124 61 144 50 165 52 C 190 54 207 78 208 110 C 209 142 203 172 187 196 C 171 220 137 233 105 227 C 88 224 74 216 66 204 L 66 185 C 82 187 97 184 109 172 C 116 150 113 118 107 86 Z";
const DUODENUM_D = "M 68 194 C 55 190 41 190 34 201 C 26 213 27 232 30 258";

/** Gastric folds along the greater curvature. Linework only, no shading. */
const RUGAE_D = [
  "M 190 104 C 193 136 188 166 173 190",
  "M 172 92 C 177 128 172 160 156 186",
  "M 153 92 C 158 124 154 152 140 176",
];

const ESOPHAGUS_W = 13;
const DUODENUM_W = 14;

const TRACT: Record<
  AnatomicalRegion,
  {
    /** Vertical extent used to grow the coverage clip, [top, bottom]. */
    span: [number, number];
    label: { x: number; y: number; anchor: "start" | "middle" | "end" };
  }
> = {
  esophagus: { span: [0, 88], label: { x: 116, y: 23, anchor: "start" } },
  stomach: { span: [50, 232], label: { x: 236, y: 246, anchor: "end" } },
  duodenum: { span: [183, 250], label: { x: 2, y: 168, anchor: "start" } },
};

/** Where each ESGE station sits on the drawing. */
const STATION_AT: Record<StationKey, readonly [number, number]> = {
  esophagus_proximal: [99, 20],
  esophagus_distal: [101, 52],
  z_line: [113, 83],
  duodenal_bulb: [47, 191],
  duodenum_descending: [29, 228],
  antrum: [92, 206],
  cardia_fundus_retroflex: [162, 76],
  lesser_curvature_retroflex: [132, 126],
  incisura: [123, 167],
  corpus_greater_curvature: [178, 156],
};

const FILL_CLASS: Record<PeaceScore, string> = {
  0: "text-peace-0",
  1: "text-peace-1",
  2: "text-peace-2",
  3: "text-peace-3",
};

type OpenRowState = Exclude<StationRowState, "observed">;

const NODE_RING: Record<OpenRowState, string> = {
  unseen: "stroke-white/40",
  candidate: "stroke-ws-fg-2",
  confirmed: "stroke-peace-3",
  rejected: "stroke-ws-label",
};

const NODE_INK: Record<OpenRowState, string> = {
  unseen: "fill-white/60",
  candidate: "fill-ws-fg-2",
  confirmed: "fill-peace-3",
  rejected: "fill-ws-label",
};

export interface RegionCoverageState {
  /** 0..100 — dwell as a share of the total procedure target. */
  fill: number;
  active: boolean;
  score: PeaceScore | null;
}

/** One station marker; the same reading its checklist row gives. */
export interface StationMark {
  state: StationRowState;
  /** The station the classifier puts the scope at now. */
  current: boolean;
  /** The procedure has ended and the station is still unobserved. */
  overdue: boolean;
}

interface CoverageMapProps {
  states: Record<AnatomicalRegion, RegionCoverageState>;
  /** 10 marks in ESGE order. Omit to draw the tract without stations. */
  stations?: readonly StationMark[];
  className?: string;
  dimmed?: boolean;
}

export function CoverageMap({ states, stations, className, dimmed }: CoverageMapProps) {
  const uid = useId().replace(/:/g, "");

  const clipHeight = (region: AnatomicalRegion) => {
    const [top, bottom] = TRACT[region].span;
    const pct = Math.max(0, Math.min(100, states[region].fill)) / 100;
    return top + (bottom - top) * pct;
  };
  const clip = (region: AnatomicalRegion) => `url(#${uid}-${region})`;
  const outline = (region: AnatomicalRegion) =>
    states[region].active ? "stroke-white/90" : "stroke-white/25";

  const tube = (region: AnatomicalRegion, d: string, width: number) => {
    const { active, score } = states[region];
    return (
      <g>
        {/* Casing: a wider stroke underneath is the tube's outline. */}
        <path
          d={d}
          fill="none"
          strokeWidth={width + (active ? 3.2 : 2.4)}
          className={outline(region)}
        />
        <path d={d} fill="none" strokeWidth={width} className="stroke-ws-organ-idle" />
        {score !== null && (
          <g className={FILL_CLASS[score]}>
            <path d={d} fill="none" stroke="currentColor" strokeWidth={width} opacity={0.3} />
            <path
              d={d}
              fill="none"
              stroke="currentColor"
              strokeWidth={width}
              opacity={0.92}
              clipPath={clip(region)}
            />
          </g>
        )}
      </g>
    );
  };

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      className={`${className ?? ""} transition-opacity duration-300 ${dimmed ? "opacity-50" : ""}`}
      role="img"
      aria-label={describeCoverage(states)}
    >
      <defs>
        {REGION_ORDER.map((region) => (
          <clipPath key={region} id={`${uid}-${region}`}>
            <rect x="0" y="0" width={VIEW_W} height={clipHeight(region)} />
          </clipPath>
        ))}
      </defs>

      {/* Silhouettes are always drawn — a missing organ reads as broken, not absent. */}
      {tube("esophagus", ESOPHAGUS_D, ESOPHAGUS_W)}
      {tube("duodenum", DUODENUM_D, DUODENUM_W)}

      <path d={STOMACH_D} className="fill-ws-organ-idle" />
      {states.stomach.score !== null && (
        <g className={FILL_CLASS[states.stomach.score]}>
          <path d={STOMACH_D} fill="currentColor" opacity={0.3} />
          <path d={STOMACH_D} fill="currentColor" opacity={0.92} clipPath={clip("stomach")} />
        </g>
      )}
      {RUGAE_D.map((d) => (
        <path
          key={d}
          d={d}
          fill="none"
          strokeWidth={1.4}
          strokeLinecap="round"
          className="stroke-black/20"
        />
      ))}
      <path
        d={STOMACH_D}
        fill="none"
        strokeWidth={states.stomach.active ? 1.6 : 1.2}
        strokeLinejoin="round"
        className={outline("stomach")}
      />

      {stations &&
        STATION_ORDER.map((key, i) => {
          const mark = stations[i];
          if (!mark) return null;
          const [x, y] = STATION_AT[key];
          return (
            <g key={key} transform={`translate(${x} ${y})`} data-station={key} data-state={mark.state}>
              <StationNode n={i + 1} mark={mark} />
            </g>
          );
        })}

      {REGION_ORDER.map((region) => {
        const { label } = TRACT[region];
        return (
          <text
            key={region}
            x={label.x}
            y={label.y}
            textAnchor={label.anchor}
            className={`text-[10.5px] font-medium uppercase tracking-[0.1em] ${
              states[region].active ? "fill-white/85" : "fill-white/40"
            }`}
          >
            {REGION_LABELS[region]}
          </text>
        );
      })}
    </svg>
  );
}

/**
 * The checklist's five states, as map pins: solid / outline / dotted / hollow /
 * struck. Every pin sits on a dark disc or a dark casing, so it stays legible
 * over whatever colour its organ has taken.
 */
function StationNode({ n, mark }: { n: number; mark: StationMark }) {
  const { state, current, overdue } = mark;
  const halo = current && (
    <circle r={13} fill="none" strokeWidth={1.8} className="stroke-ws-detect" />
  );
  const numeral =
    "font-mono text-[10px] font-semibold [dominant-baseline:central] [text-anchor:middle]";

  if (state === "observed") {
    return (
      <>
        {halo}
        <circle r={9} strokeWidth={2} className="fill-peace-3 stroke-ws-void" />
        <text className={`${numeral} fill-ws-void`}>{n}</text>
      </>
    );
  }

  // Amber outranks the model's own state once the procedure is over, exactly
  // as it does on the station's checklist row.
  const ring = overdue ? "stroke-ws-coach" : NODE_RING[state];
  const ink = overdue ? "fill-ws-coach" : NODE_INK[state];
  return (
    <>
      {halo}
      <circle
        r={9}
        strokeWidth={1.4}
        strokeDasharray={state === "candidate" ? "2 2.6" : undefined}
        className={`fill-black/80 ${ring}`}
      />
      {state === "rejected" && (
        <path d="M -6 6 L 6 -6" strokeWidth={1.4} strokeLinecap="round" className={ring} />
      )}
      <text className={`${numeral} ${ink}`}>{n}</text>
    </>
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
