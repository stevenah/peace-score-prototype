"use client";

import { memo, useCallback, useMemo, useRef, useState } from "react";
import { ScoreLegend } from "./ScoreLegend";
import { PEACE_SCORE_COLORS, MOTION_LABELS, REGION_LABELS } from "@/lib/constants";
import type { PeaceScore, TimelineEntry, MotionDirection, AnatomicalRegion } from "@/lib/types";
import { formatDuration } from "@/lib/utils";

interface Segment {
  startTime: number;
  endTime: number;
  score: PeaceScore;
  motion: MotionDirection;
  region: AnatomicalRegion;
  isAlert: boolean;
}

interface ScoreTimelineBarProps {
  timeline: TimelineEntry[];
  totalDuration: number;
  currentTime?: number;
  onSeek?: (time: number) => void;
  className?: string;
}

export const ScoreTimelineBar = memo(function ScoreTimelineBar({
  timeline,
  totalDuration,
  currentTime,
  onSeek,
  className,
}: ScoreTimelineBarProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    x: number;
    time: number;
    segment: Segment;
  } | null>(null);

  const maxTime = totalDuration || (timeline.length > 0 ? timeline[timeline.length - 1].timestamp : 0);

  const segments = useMemo(() => {
    if (timeline.length === 0 || maxTime === 0) return [];
    return timeline.map((entry, i): Segment => {
      const nextTime = i < timeline.length - 1 ? timeline[i + 1].timestamp : maxTime;
      return {
        startTime: entry.timestamp,
        endTime: nextTime,
        score: entry.peace_score,
        motion: entry.motion,
        region: entry.region,
        isAlert: entry.motion === "withdrawal" && entry.peace_score < 2,
      };
    });
  }, [timeline, maxTime]);

  const alerts = useMemo(
    () => segments.filter((s) => s.isAlert),
    [segments],
  );

  const findSegmentAtTime = useCallback(
    (time: number): Segment | null => {
      if (segments.length === 0) return null;
      for (let i = segments.length - 1; i >= 0; i--) {
        if (time >= segments[i].startTime) return segments[i];
      }
      return segments[0];
    },
    [segments],
  );

  const getTimeFromEvent = useCallback(
    (e: React.MouseEvent | MouseEvent): number => {
      const bar = barRef.current;
      if (!bar || maxTime === 0) return 0;
      const rect = bar.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      return fraction * maxTime;
    },
    [maxTime],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const bar = barRef.current;
      if (!bar) return;
      const rect = bar.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const time = getTimeFromEvent(e);
      const segment = findSegmentAtTime(time);
      if (segment) {
        setHoverInfo({ x, time, segment });
      }
    },
    [getTimeFromEvent, findSegmentAtTime],
  );

  const handleMouseLeave = useCallback(() => setHoverInfo(null), []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (!onSeek) return;
      e.preventDefault();
      const time = getTimeFromEvent(e);
      onSeek(time);

      const onMove = (ev: MouseEvent) => {
        const t = getTimeFromEvent(ev);
        onSeek(t);
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    },
    [onSeek, getTimeFromEvent],
  );

  if (maxTime === 0) return null;

  const progress = currentTime != null ? Math.max(0, Math.min(100, (currentTime / maxTime) * 100)) : null;

  // Calculate tooltip position — flip if near right edge
  let tooltipStyle: React.CSSProperties | undefined;
  if (hoverInfo) {
    const barWidth = barRef.current?.getBoundingClientRect().width ?? 400;
    const nearRightEdge = hoverInfo.x > barWidth - 160;
    tooltipStyle = nearRightEdge
      ? { right: barWidth - hoverInfo.x, left: "auto" }
      : { left: hoverInfo.x };
  }

  return (
    <div className={className}>
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium text-foreground/80">Score Timeline</h3>
          {alerts.length > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-red-500/10 px-2 py-0.5 text-xs font-medium text-red-500">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500" />
              {alerts.length} alert{alerts.length !== 1 ? "s" : ""}
            </span>
          )}
        </div>
        <ScoreLegend />
      </div>

      {/* Bar container with alert markers + segments + playhead */}
      <div className="relative">
        {/* Alert markers row */}
        {alerts.length > 0 && (
          <div className="relative mb-1 h-2">
            {alerts.map((alert, i) => {
              const left = (alert.startTime / maxTime) * 100;
              return (
                <div
                  key={i}
                  className="absolute -translate-x-1/2"
                  style={{ left: `${left}%`, top: 0 }}
                >
                  <svg width="8" height="8" viewBox="0 0 8 8">
                    <polygon points="4,0 8,8 0,8" fill="#ef4444" />
                  </svg>
                </div>
              );
            })}
          </div>
        )}

        {/* Segmented bar */}
        <div
          ref={barRef}
          className={`relative h-3 overflow-hidden rounded-full bg-muted ${onSeek ? "cursor-pointer" : ""}`}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onMouseDown={handleMouseDown}
        >
          {segments.map((seg, i) => {
            const leftPct = (seg.startTime / maxTime) * 100;
            const widthPct = ((seg.endTime - seg.startTime) / maxTime) * 100;
            return (
              <div
                key={i}
                className="absolute top-0 h-full transition-opacity"
                style={{
                  left: `${leftPct}%`,
                  width: `${Math.max(widthPct, 0.2)}%`,
                  backgroundColor: PEACE_SCORE_COLORS[seg.score],
                  opacity: hoverInfo?.segment === seg ? 1 : 0.75,
                }}
              />
            );
          })}
          {/* Playhead */}
          {progress != null && (
            <div
              className="pointer-events-none absolute top-1/2 z-10 -translate-x-1/2 -translate-y-1/2"
              style={{ left: `${progress}%`, transition: "left 100ms linear" }}
            >
              <div className="flex flex-col items-center">
                <div className="h-5 w-0.5 rounded-full bg-white shadow-[0_0_3px_rgba(0,0,0,0.5)]" />
              </div>
            </div>
          )}
        </div>

        {/* Hover tooltip */}
        {hoverInfo && (
          <div
            className="pointer-events-none absolute z-20 mt-2 rounded-lg border border-border bg-card px-3 py-2 text-xs shadow-md"
            style={tooltipStyle}
          >
            {hoverInfo.segment.isAlert && (
              <p className="mb-1 font-semibold text-red-500">
                Warning: Insufficient cleaning during withdrawal
              </p>
            )}
            <p className="font-medium" style={{ color: PEACE_SCORE_COLORS[hoverInfo.segment.score] }}>
              Score: {hoverInfo.segment.score}
            </p>
            <p className="text-muted-foreground">
              {MOTION_LABELS[hoverInfo.segment.motion]} &middot; {REGION_LABELS[hoverInfo.segment.region]}
            </p>
            <p className="text-muted-foreground/60">{formatDuration(hoverInfo.time)}</p>
          </div>
        )}
      </div>
    </div>
  );
});
