"use client";

import { lazy, memo, Suspense, useCallback, useMemo } from "react";
import { ScoreLegend } from "./ScoreLegend";
import { PEACE_SCORE_COLORS, MOTION_COLORS } from "@/lib/constants";
import { formatTimestamp } from "@/lib/utils";
import type { PeaceScore, TimelineEntry } from "@/lib/types";

const LazyRecharts = lazy(() =>
  import("recharts").then((mod) => ({
    default: ({ data, maxTime, currentTime, onSeek, renderDot, renderTooltip }: {
      data: Record<string, unknown>[];
      maxTime: number;
      currentTime?: number;
      onSeek?: (time: number) => void;
      renderDot: (props: Record<string, unknown>) => React.ReactElement;
      renderTooltip: (props: Record<string, unknown>) => React.ReactNode;
    }) => (
      <mod.ResponsiveContainer width="100%" height="100%">
        <mod.ComposedChart
          data={data}
          margin={{ top: 5, right: 10, bottom: 5, left: -10 }}
          onClick={onSeek ? (state: { activeLabel?: string | number }) => {
            const label = state?.activeLabel;
            if (label != null) onSeek(typeof label === "number" ? label : parseFloat(String(label)));
          } : undefined}
        >
          <mod.CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <mod.XAxis
            dataKey="time"
            type="number"
            domain={[0, maxTime]}
            tickFormatter={(v: number) => formatTimestamp(v)}
            tick={{ fontSize: 10 }}
            interval="preserveStartEnd"
          />
          <mod.YAxis
            domain={[0, 3]}
            ticks={[0, 1, 2, 3]}
            tick={{ fontSize: 10 }}
          />
          <mod.Tooltip content={renderTooltip} />
          <mod.ReferenceLine y={2} stroke="#84cc16" strokeDasharray="3 3" opacity={0.4} />
          {currentTime != null && currentTime > 0 && (
            <mod.ReferenceLine x={currentTime} stroke="#3b82f6" strokeWidth={1.5} opacity={0.6} />
          )}
          <mod.Area
            type="stepAfter"
            dataKey="peace_score"
            stroke="none"
            fill="#3b82f6"
            fillOpacity={0.08}
            isAnimationActive={false}
          />
          <mod.Line
            type="monotone"
            dataKey="peace_score"
            stroke="#3b82f6"
            strokeWidth={2}
            isAnimationActive={false}
            dot={renderDot}
          />
        </mod.ComposedChart>
      </mod.ResponsiveContainer>
    ),
  }))
);

interface PeaceScoreTimelineProps {
  timeline: TimelineEntry[];
  totalDuration?: number;
  currentTime?: number;
  onSeek?: (time: number) => void;
}

export const PeaceScoreTimeline = memo(function PeaceScoreTimeline({ timeline, totalDuration, currentTime, onSeek }: PeaceScoreTimelineProps) {
  const maxTime = totalDuration || (timeline.length > 0 ? timeline[timeline.length - 1].timestamp : 0);

  if (maxTime === 0) return null;

  const data = useMemo(() => {
    const points = timeline.map((entry) => ({
      ...entry,
      time: entry.timestamp,
      scoreColor: PEACE_SCORE_COLORS[entry.peace_score as PeaceScore],
      motionColor: MOTION_COLORS[entry.motion],
    }));
    return points.length > 0 ? points : [{ time: 0 }, { time: maxTime }];
  }, [timeline, maxTime]);

  const renderDot = useCallback((props: Record<string, unknown>) => {
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
  }, []);

  const renderTooltip = useCallback((props: Record<string, unknown>) => {
    const { active, payload } = props as { active?: boolean; payload?: readonly { payload: Record<string, unknown> }[] };
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 text-xs shadow-md">
        <p className="font-medium" style={{ color: d.scoreColor as string }}>
          Score: {d.peace_score as number}
        </p>
        <p style={{ color: d.motionColor as string }}>
          {d.motion as string} &middot; {d.region as string}
        </p>
      </div>
    );
  }, []);

  return (
    <div>
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground/80">
          Score Timeline
        </h3>
        <ScoreLegend />
      </div>
      <div className={`h-48 rounded-xl border border-border bg-card p-4 ring-1 ring-black/3 dark:ring-white/3 ${onSeek ? "cursor-pointer" : ""}`}>
        <Suspense fallback={<div className="flex h-full items-center justify-center text-xs text-muted-foreground">Loading chart...</div>}>
          <LazyRecharts
            data={data}
            maxTime={maxTime}
            currentTime={currentTime}
            onSeek={onSeek}
            renderDot={renderDot}
            renderTooltip={renderTooltip}
          />
        </Suspense>
      </div>
    </div>
  );
});
