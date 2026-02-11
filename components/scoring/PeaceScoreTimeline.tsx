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
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { ScoreLegend } from "./ScoreLegend";
import { PEACE_SCORE_COLORS, MOTION_COLORS } from "@/lib/constants";
import { formatTimestamp } from "@/lib/utils";
import type { PeaceScore, TimelineEntry } from "@/lib/types";

interface PeaceScoreTimelineProps {
  timeline: TimelineEntry[];
}

export function PeaceScoreTimeline({ timeline }: PeaceScoreTimelineProps) {
  if (timeline.length === 0) return null;

  const data = timeline.map((entry) => ({
    ...entry,
    time: formatTimestamp(entry.timestamp),
    scoreColor: PEACE_SCORE_COLORS[entry.peace_score as PeaceScore],
    motionColor: MOTION_COLORS[entry.motion],
  }));

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>Score Timeline</CardTitle>
          <ScoreLegend />
        </div>
      </CardHeader>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 11 }}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 3]}
              ticks={[0, 1, 2, 3]}
              tick={{ fontSize: 11 }}
              label={{
                value: "PEACE Score",
                angle: -90,
                position: "insideLeft",
                style: { fontSize: 11 },
              }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="rounded-lg border border-neutral-200 bg-white p-3 shadow-lg dark:border-neutral-700 dark:bg-neutral-800">
                    <p className="text-xs text-neutral-500">{d.time}</p>
                    <p className="font-medium" style={{ color: d.scoreColor }}>
                      Score: {d.peace_score}
                    </p>
                    <p className="text-xs" style={{ color: d.motionColor }}>
                      {d.motion} | {d.region}
                    </p>
                  </div>
                );
              }}
            />
            <ReferenceLine y={2} stroke="#84cc16" strokeDasharray="3 3" opacity={0.5} />
            <Area
              type="stepAfter"
              dataKey="peace_score"
              stroke="none"
              fill="#3b82f6"
              fillOpacity={0.1}
            />
            <Line
              type="monotone"
              dataKey="peace_score"
              stroke="#3b82f6"
              strokeWidth={2}
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
    </Card>
  );
}
