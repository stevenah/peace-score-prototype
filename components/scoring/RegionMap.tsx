"use client";

import {
  PEACE_SCORE_COLORS,
  PEACE_SCORE_LABELS,
  REGION_LABELS,
  REGION_ORDER,
} from "@/lib/constants";
import type { AnatomicalRegion, PeaceScore, RegionScore } from "@/lib/types";

interface RegionMapProps {
  byRegion: Partial<Record<AnatomicalRegion, RegionScore>>;
}

const regionPaths: Record<AnatomicalRegion, { d: string; labelY: number }> = {
  esophagus: {
    d: "M 120,30 C 120,30 115,50 115,80 C 115,110 125,110 125,80 C 125,50 120,30 120,30",
    labelY: 55,
  },
  stomach: {
    d: "M 115,110 C 100,120 80,140 75,170 C 70,200 85,230 110,235 C 135,240 155,220 160,190 C 165,160 155,130 125,110",
    labelY: 170,
  },
  duodenum: {
    d: "M 160,190 C 170,195 185,195 190,185 C 195,175 195,160 185,155 C 175,150 165,155 160,165",
    labelY: 180,
  },
};

export function RegionMap({ byRegion }: RegionMapProps) {
  return (
    <div className="flex items-center justify-center gap-8">
      <svg viewBox="40 10 200 260" className="h-56 w-44">
        {REGION_ORDER.map((region) => {
          const data = byRegion[region];
          const path = regionPaths[region];
          const color = data
            ? PEACE_SCORE_COLORS[data.score as PeaceScore]
            : "#d4d4d4";

          return (
            <g key={region}>
              <path
                d={path.d}
                fill={color}
                fillOpacity={0.3}
                stroke={color}
                strokeWidth={2}
              />
              <text
                x={region === "duodenum" ? 195 : 120}
                y={path.labelY}
                textAnchor="middle"
                className="text-[9px] font-medium"
                fill="currentColor"
              >
                {REGION_LABELS[region]}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="space-y-2.5">
        {REGION_ORDER.map((region) => {
          const data = byRegion[region];
          if (!data) return null;
          const color = PEACE_SCORE_COLORS[data.score as PeaceScore];
          return (
            <div key={region} className="flex items-center gap-2.5">
              <div
                className="h-3 w-3 rounded-full"
                style={{ backgroundColor: color }}
              />
              <span className="text-sm text-foreground/80">
                {REGION_LABELS[region]}:{" "}
                <span className="font-medium" style={{ color }}>
                  {data.score}/3
                </span>{" "}
                <span className="text-muted-foreground">
                  {PEACE_SCORE_LABELS[data.score as PeaceScore]}
                </span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
