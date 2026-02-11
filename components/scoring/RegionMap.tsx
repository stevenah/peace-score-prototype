"use client";

import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  PEACE_SCORE_COLORS,
  PEACE_SCORE_LABELS,
  REGION_LABELS,
  REGION_ORDER,
} from "@/lib/constants";
import { formatConfidence } from "@/lib/utils";
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
    <Card>
      <CardHeader>
        <CardTitle>Anatomical Region Scores</CardTitle>
      </CardHeader>
      <div className="flex items-center justify-center gap-8">
        <svg viewBox="40 10 200 260" className="h-64 w-48">
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

        <div className="space-y-3">
          {REGION_ORDER.map((region) => {
            const data = byRegion[region];
            if (!data) return null;
            const color = PEACE_SCORE_COLORS[data.score as PeaceScore];
            return (
              <div key={region} className="flex items-center gap-3">
                <div
                  className="h-4 w-4 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <div>
                  <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                    {REGION_LABELS[region]}: {data.score}/3
                  </p>
                  <p className="text-xs text-neutral-500">
                    {PEACE_SCORE_LABELS[data.score as PeaceScore]} &middot;{" "}
                    {formatConfidence(data.confidence)}
                  </p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Card>
  );
}
