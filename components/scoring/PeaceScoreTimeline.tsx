"use client";

import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { ScoreLegend } from "./ScoreLegend";
import { PEACE_SCORE_COLORS, MOTION_COLORS } from "@/lib/constants";
import { formatTimestamp } from "@/lib/utils";
import type { PeaceScore, TimelineEntry } from "@/lib/types";

interface PeaceScoreTimelineProps {
  timeline: TimelineEntry[];
  totalDuration?: number;
  currentTime?: number;
  onSeek?: (time: number) => void;
}

export function PeaceScoreTimeline({ timeline, totalDuration, currentTime, onSeek }: PeaceScoreTimelineProps) {
  const maxTime = totalDuration || (timeline.length > 0 ? timeline[timeline.length - 1].timestamp : 0);

  if (maxTime === 0) return null;

  const points = timeline.map((entry) => ({
    ...entry,
    time: entry.timestamp,
    scoreColor: PEACE_SCORE_COLORS[entry.peace_score as PeaceScore],
    motionColor: MOTION_COLORS[entry.motion],
  }));

  // Recharts needs data points to render axes/grid, so add invisible boundaries
  const data = points.length > 0
    ? points
    : [{ time: 0 }, { time: maxTime }];

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
          Score Timeline
        </h3>
        <ScoreLegend />
      </div>
      <div className={`h-48 rounded-xl border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900 ${onSeek ? "cursor-pointer" : ""}`}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 5, right: 10, bottom: 5, left: -10 }}
            onClick={onSeek ? (state) => {
              const label = state?.activeLabel;
              if (label != null) onSeek(typeof label === "number" ? label : parseFloat(String(label)));
            } : undefined}
          >
            <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
            <XAxis
              dataKey="time"
              type="number"
              domain={[0, maxTime]}
              tickFormatter={(v: number) => formatTimestamp(v)}
              tick={{ fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 3]}
              ticks={[0, 1, 2, 3]}
              tick={{ fontSize: 10 }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="rounded-lg border border-neutral-200 bg-white px-3 py-2 text-xs shadow-md dark:border-neutral-700 dark:bg-neutral-800">
                    <p className="font-medium" style={{ color: d.scoreColor }}>
                      Score: {d.peace_score}
                    </p>
                    <p style={{ color: d.motionColor }}>
                      {d.motion} &middot; {d.region}
                    </p>
                  </div>
                );
              }}
            />
            <ReferenceLine y={2} stroke="#84cc16" strokeDasharray="3 3" opacity={0.4} />
            {currentTime != null && currentTime > 0 && (
              <ReferenceLine x={currentTime} stroke="#3b82f6" strokeWidth={1.5} opacity={0.6} />
            )}
            <Area
              type="stepAfter"
              dataKey="peace_score"
              stroke="none"
              fill="#3b82f6"
              fillOpacity={0.08}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="peace_score"
              stroke="#3b82f6"
              strokeWidth={2}
              isAnimationActive={false}
              dot={(props: Record<string, unknown>) => {
                const { cx, cy, payload } = props as {
                  cx: number;
                  cy: number;
                  payload: { peace_score: PeaceScore };
                };
                return (
                  <circle
                    key={`dot-${cx}-${cy}`}
                    cx={cx}
                    cy={cy}
                    r={3}
                    fill={PEACE_SCORE_COLORS[payload.peace_score as PeaceScore]}
                    stroke="none"
                  />
                );
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
