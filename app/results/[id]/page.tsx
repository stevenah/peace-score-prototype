"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Play, Pause, RotateCcw } from "lucide-react";
import Link from "next/link";
import { AnalysisProgress } from "@/components/analysis/AnalysisProgress";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { PeaceScoreGrid } from "@/components/scoring/PeaceScoreGrid";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import {
  VideoPlaybackPlayer,
  type VideoPlaybackPlayerHandle,
} from "@/components/video/VideoPlaybackPlayer";
import { useAnalysis } from "@/hooks/useAnalysis";
import { formatDuration } from "@/lib/utils";
import { PEACE_SCORE_COLORS } from "@/lib/constants";
import type { PeaceScore, MotionDirection, TimelineEntry } from "@/lib/types";

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

  // Animation-based replay (fallback when no video)
  const tick = useCallback(
    (now: number) => {
      if (!lastTickRef.current) lastTickRef.current = now;
      const delta = (now - lastTickRef.current) / 1000;
      lastTickRef.current = now;
      setReplayTime((prev) => {
        const next = prev + delta;
        if (next >= totalDuration) {
          setIsPlaying(false);
          return totalDuration;
        }
        return next;
      });
      animRef.current = requestAnimationFrame(tick);
    },
    [totalDuration],
  );

  useEffect(() => {
    if (hasVideo) return; // video drives time, not animation
    if (isPlaying) {
      lastTickRef.current = 0;
      animRef.current = requestAnimationFrame(tick);
    }
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [isPlaying, tick, hasVideo]);

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
          {/* Video + Live Scores layout (when video available) */}
          {hasVideo ? (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 lg:items-stretch">
              <div className="lg:col-span-2">
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
              </div>

              <div className="flex flex-col gap-4">
                <PeaceScoreCard
                  score={results.peace_scores.overall.score as PeaceScore}
                  label={results.peace_scores.overall.label}
                  size="lg"
                />

                {activeEntry?.motion && (
                  <Card>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Motion
                    </p>
                    <MotionIndicator
                      direction={activeEntry.motion as MotionDirection}
                    />
                  </Card>
                )}

                {activeEntry?.region && (
                  <Card>
                    <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      Region
                    </p>
                    <p className="text-base font-medium capitalize text-foreground">
                      {activeEntry.region}
                    </p>
                  </Card>
                )}
              </div>

              <div className="lg:col-span-1">
                <Card className="h-full">
                  <CardHeader>
                    <CardTitle>Video Details</CardTitle>
                  </CardHeader>
                  <div className="space-y-3">
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
                </Card>
              </div>
            </div>
          ) : (
            /* Original layout (no video) */
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
              <PeaceScoreCard
                score={results.peace_scores.overall.score as PeaceScore}
                label={results.peace_scores.overall.label}
                size="lg"
              />
              <div className="lg:col-span-3">
                <Card>
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
                </Card>
              </div>
            </div>
          )}

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
              <div className="space-y-2">
                {results.motion_analysis.segments.map((seg, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded-lg bg-muted/50 px-4 py-2"
                  >
                    <MotionIndicator
                      direction={seg.direction as MotionDirection}
                    />
                    <span className="text-xs text-muted-foreground">
                      {formatDuration(seg.start_time)} →{" "}
                      {formatDuration(seg.end_time)}
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
