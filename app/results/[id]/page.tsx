"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import { Play, Pause, RotateCcw } from "lucide-react";
import Link from "next/link";
import { AnalysisProgress } from "@/components/analysis/AnalysisProgress";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { MotionVisual } from "@/components/analysis/MotionVisual";
import { RegionHighlight } from "@/components/scoring/RegionHighlight";
import { PeaceScoreGrid } from "@/components/scoring/PeaceScoreGrid";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  VideoPlaybackPlayer,
  type VideoPlaybackPlayerHandle,
} from "@/components/video/VideoPlaybackPlayer";
import { useAnalysis } from "@/hooks/useAnalysis";
import { formatDuration } from "@/lib/utils";
import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS } from "@/lib/constants";
import type { PeaceScore, MotionDirection, AnatomicalRegion, TimelineEntry } from "@/lib/types";

export default function ResultsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: analysis, error } = useAnalysis(id);

  const playerRef = useRef<VideoPlaybackPlayerHandle>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [replayTime, setReplayTime] = useState(0);
  const animRef = useRef<number | null>(null);
  const lastTickRef = useRef<number>(0);

  const results = analysis?.results;
  const hasVideo = !!analysis?.video_url;
  const totalDuration = results?.timeline?.length
    ? results.timeline[results.timeline.length - 1].timestamp
    : 0;

  const activeEntry: TimelineEntry | null =
    results?.timeline?.length && replayTime > 0
      ? results.timeline.reduce((best, entry) =>
          Math.abs(entry.timestamp - replayTime) <
          Math.abs(best.timestamp - replayTime)
            ? entry
            : best,
        )
      : null;

  // Keep totalDuration in a ref so the animation loop always reads the latest value
  const totalDurationRef = useRef(totalDuration);
  useEffect(() => {
    totalDurationRef.current = totalDuration;
  }, [totalDuration]);

  // Animation-based replay (fallback when no video)
  useEffect(() => {
    if (hasVideo || !isPlaying) return;

    lastTickRef.current = 0;

    function tick(now: number) {
      if (!lastTickRef.current) lastTickRef.current = now;
      const delta = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      setReplayTime((prev) => {
        const next = prev + delta;
        if (next >= totalDurationRef.current) {
          setIsPlaying(false);
          return totalDurationRef.current;
        }
        return next;
      });
      animRef.current = requestAnimationFrame(tick);
    }

    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isPlaying, hasVideo]);

  const handleRestart = useCallback(() => {
    setReplayTime(0);
    setIsPlaying(true);
  }, []);

  if (error) {
    return (
      <div className="py-12 text-center">
        <p className="text-red-600">{error}</p>
        <Link href="/analyze" className="mt-4 text-sm text-primary">
          Back to analysis
        </Link>
      </div>
    );
  }

  if (!analysis) {
    return <AnalysisProgress status="processing" progress={0} />;
  }

  const isComplete = analysis.status === "completed";

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">
            Analysis Results
          </h1>
          <p className="text-xs text-muted-foreground">ID: {id}</p>
        </div>
      </div>

      {!isComplete && (
        <AnalysisProgress
          status={analysis.status}
          progress={analysis.progress}
          error={analysis.error || undefined}
        />
      )}

      {isComplete && results && (
        <div className="space-y-6">
          {/* Video + Live Scores layout (matches LiveAnalysis) */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 lg:items-stretch">
            <div className="lg:col-span-3">
              {hasVideo ? (
                <VideoPlaybackPlayer
                  ref={playerRef}
                  src={analysis.video_url!}
                  onTimeUpdate={setReplayTime}
                  peaceScore={
                    activeEntry
                      ? (activeEntry.peace_score as PeaceScore)
                      : null
                  }
                  motionDirection={activeEntry?.motion ?? null}
                  region={activeEntry?.region ?? null}
                />
              ) : (
                <Card className="flex h-full items-center justify-center">
                  <div className="space-y-3 text-center">
                    <CardHeader>
                      <CardTitle>Video Details</CardTitle>
                    </CardHeader>
                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                      <div>
                        <p className="text-xs text-muted-foreground">Duration</p>
                        <p className="text-lg font-semibold">
                          {analysis.video_metadata &&
                          analysis.video_metadata.duration_seconds > 0
                            ? formatDuration(
                                analysis.video_metadata.duration_seconds,
                              )
                            : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Resolution</p>
                        <p className="text-lg font-semibold">
                          {analysis.video_metadata &&
                          analysis.video_metadata.resolution[0] > 0
                            ? `${analysis.video_metadata.resolution[0]}x${analysis.video_metadata.resolution[1]}`
                            : "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">
                          Frames Analyzed
                        </p>
                        <p className="text-lg font-semibold">
                          {analysis.video_metadata?.analyzed_frames || "—"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Segments</p>
                        <p className="text-lg font-semibold">
                          {results.motion_analysis.segments.length || "—"}
                        </p>
                      </div>
                    </div>
                  </div>
                </Card>
              )}
            </div>

            <div className="flex flex-col gap-4 lg:min-h-0 lg:overflow-hidden">
              {activeEntry ? (
                <>
                  <Card className="flex min-h-0 flex-1 flex-col items-center justify-between overflow-hidden pb-4 pt-3 text-center">
                    <p className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      PEACE Score
                    </p>
                    <div className="flex items-baseline gap-1.5">
                      <span
                        className="text-4xl font-bold"
                        style={{ color: PEACE_SCORE_COLORS[activeEntry.peace_score as PeaceScore] }}
                      >
                        {activeEntry.peace_score}
                      </span>
                      <span className="text-sm text-muted-foreground/60">/ 3</span>
                    </div>
                    <p
                      className="shrink-0 text-sm font-medium"
                      style={{ color: PEACE_SCORE_COLORS[activeEntry.peace_score as PeaceScore] }}
                    >
                      {PEACE_SCORE_LABELS[activeEntry.peace_score as PeaceScore]}
                    </p>
                  </Card>

                  <Card className="flex min-h-0 flex-1 flex-col items-center justify-between overflow-hidden pb-4 pt-3 text-center">
                    <p className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Motion
                    </p>
                    {activeEntry.motion ? (
                      <MotionVisual
                        direction={activeEntry.motion as MotionDirection}
                      />
                    ) : (
                      <p className="text-sm font-medium text-muted-foreground/30">—</p>
                    )}
                    <span />
                  </Card>

                  <Card className="flex min-h-0 flex-1 flex-col items-center justify-between overflow-hidden pb-4 pt-3 text-center">
                    <p className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Region
                    </p>
                    {activeEntry.region ? (
                      <RegionHighlight activeRegion={activeEntry.region as AnatomicalRegion} />
                    ) : (
                      <p className="text-sm font-medium text-muted-foreground/30">—</p>
                    )}
                    <p className="shrink-0 text-sm font-medium capitalize text-foreground">
                      {activeEntry.region || "\u00A0"}
                    </p>
                  </Card>
                </>
              ) : (
                <>
                  <Card className="flex min-h-0 flex-1 flex-col items-center justify-between overflow-hidden pb-4 pt-3 text-center">
                    <p className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      PEACE Score
                    </p>
                    <div className="flex items-baseline gap-1.5">
                      <span
                        className="text-4xl font-bold"
                        style={{ color: PEACE_SCORE_COLORS[results.peace_scores.overall.score as PeaceScore] }}
                      >
                        {results.peace_scores.overall.score}
                      </span>
                      <span className="text-sm text-muted-foreground/60">/ 3</span>
                    </div>
                    <p
                      className="shrink-0 text-sm font-medium"
                      style={{ color: PEACE_SCORE_COLORS[results.peace_scores.overall.score as PeaceScore] }}
                    >
                      {results.peace_scores.overall.label || PEACE_SCORE_LABELS[results.peace_scores.overall.score as PeaceScore]}
                    </p>
                  </Card>

                  <Card className="flex flex-1 flex-col items-center text-center">
                    <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Motion
                    </p>
                    <p className="text-sm font-medium text-muted-foreground/30">—</p>
                  </Card>

                  <Card className="flex flex-1 flex-col items-center text-center">
                    <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Region
                    </p>
                    <p className="text-sm font-medium text-muted-foreground/30">—</p>
                  </Card>
                </>
              )}
            </div>
          </div>

          {Object.keys(results.peace_scores.by_region).length > 0 && (
            <PeaceScoreGrid byRegion={results.peace_scores.by_region} />
          )}

          {results.timeline.length > 0 && (
            <div className="space-y-3">
              <PeaceScoreTimeline
                timeline={results.timeline}
                totalDuration={totalDuration}
                currentTime={replayTime}
                onSeek={(t) => {
                  if (hasVideo) {
                    playerRef.current?.seekTo(t);
                  } else {
                    setReplayTime(t);
                    setIsPlaying(false);
                  }
                }}
              />
              {/* Only show animation controls when there's no video player */}
              {!hasVideo && (
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => setIsPlaying((p) => !p)}
                    className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-card text-foreground/70 transition-colors hover:bg-accent"
                  >
                    {isPlaying ? (
                      <Pause className="h-4 w-4" />
                    ) : (
                      <Play className="h-4 w-4" />
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={handleRestart}
                    className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-card text-foreground/70 transition-colors hover:bg-accent"
                  >
                    <RotateCcw className="h-4 w-4" />
                  </button>
                  <span className="text-xs text-muted-foreground">
                    {formatDuration(replayTime)} /{" "}
                    {formatDuration(totalDuration)}
                  </span>
                  {activeEntry && (
                    <span
                      className="ml-auto text-sm font-bold"
                      style={{
                        color: PEACE_SCORE_COLORS[activeEntry.peace_score as PeaceScore],
                      }}
                    >
                      Score: {activeEntry.peace_score}/3 &middot;{" "}
                      {activeEntry.motion} &middot; {activeEntry.region}
                    </span>
                  )}
                </div>
              )}
            </div>
          )}

          {results.motion_analysis.segments.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Motion Segments</CardTitle>
              </CardHeader>
              <div className="flex flex-wrap gap-2">
                {results.motion_analysis.segments.map((seg, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-2 rounded-lg bg-muted/50 px-3 py-1.5"
                  >
                    <MotionIndicator
                      direction={seg.direction as MotionDirection}
                    />
                    <span className="text-xs text-muted-foreground">
                      {formatDuration(seg.start_time)} → {formatDuration(seg.end_time)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
