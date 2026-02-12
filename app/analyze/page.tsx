"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSession } from "next-auth/react";
import { saveLiveAnalysis } from "@/lib/api-client";
import { VideoUploader } from "@/components/video/VideoUploader";
import { VideoStreamPlayer, type VideoStreamPlayerHandle } from "@/components/video/VideoStreamPlayer";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { PeaceScoreGrid } from "@/components/scoring/PeaceScoreGrid";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { ColonMap } from "@/components/scoring/ColonMap";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import { PEACE_SCORE_LABELS, REGION_LABELS, REGION_ORDER } from "@/lib/constants";
import { formatDuration } from "@/lib/utils";
import type {
  PeaceScore,
  MotionDirection,
  AnatomicalRegion,
  TimelineEntry,
  RegionScore,
  LiveFrameResult,
} from "@/lib/types";

function computeRegionScores(
  results: LiveFrameResult[],
): Partial<Record<AnatomicalRegion, RegionScore>> {
  const byRegion: Partial<Record<AnatomicalRegion, LiveFrameResult[]>> = {};
  for (const r of results) {
    const region = (r.region || "stomach") as AnatomicalRegion;
    if (!byRegion[region]) byRegion[region] = [];
    byRegion[region]!.push(r);
  }

  const scores: Partial<Record<AnatomicalRegion, RegionScore>> = {};
  for (const region of REGION_ORDER) {
    const frames = byRegion[region];
    if (!frames || frames.length === 0) continue;
    const minScore = Math.min(
      ...frames.map((f) => f.peace_score.score),
    ) as PeaceScore;
    const avgConf =
      frames.reduce((s, f) => s + f.peace_score.confidence, 0) / frames.length;
    scores[region] = {
      score: minScore,
      label: PEACE_SCORE_LABELS[minScore],
      confidence: Math.round(avgConf * 100) / 100,
      region,
      frame_scores: frames.map((f, i) => ({
        frame_index: f.frame_index,
        timestamp: i,
        score: f.peace_score.score as PeaceScore,
        confidence: f.peace_score.confidence,
      })),
    };
  }
  return scores;
}

export default function AnalyzePage() {
  const playerRef = useRef<VideoStreamPlayerHandle>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [videoEnded, setVideoEnded] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentVideoTime, setCurrentVideoTime] = useState(0);

  const {
    isConnected,
    latestResult,
    results,
    sendFrame,
  } = useLiveFeed({
    enabled: isAnalyzing,
    wsUrl: "ws://localhost:8000/api/v1/ws/live",
  });

  const { data: sessionData } = useSession();
  const [isSaved, setIsSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setVideoEnded(false);
    setIsAnalyzing(true);
    setIsSaved(false);
    setSaveError(null);
  }, []);

  // Auto-save when video ends and user is logged in
  useEffect(() => {
    if (!videoEnded || results.length === 0 || isSaved) return;
    if (!sessionData?.user) return;

    const overallScore = Math.min(...results.map((r) => r.peace_score.score));

    saveLiveAnalysis({
      filename: selectedFile?.name ?? "live-analysis",
      overallScore,
      framesAnalyzed: results.length,
      duration: videoDuration > 0 ? videoDuration : null,
      timeline: results.map((r, i) => ({
        timestamp: i * 0.5,
        frame_index: r.frame_index,
        motion: r.motion?.direction || "stationary",
        region: r.region || "stomach",
        peace_score: r.peace_score.score,
        confidence: r.peace_score.confidence,
      })),
      videoFile: selectedFile ?? undefined,
    })
      .then(() => setIsSaved(true))
      .catch((err) => setSaveError(err.message));
  }, [videoEnded, results, isSaved, sessionData, selectedFile, videoDuration]);

  const timeline: TimelineEntry[] = useMemo(
    () =>
      results.map((r, i) => ({
        timestamp: i * 0.5,
        frame_index: r.frame_index,
        motion: (r.motion?.direction || "stationary") as MotionDirection,
        region: (r.region || "stomach") as AnatomicalRegion,
        peace_score: r.peace_score.score as PeaceScore,
        confidence: r.peace_score.confidence,
      })),
    [results],
  );

  // When scrubbing or replaying, show the result closest to the current video time
  const displayResult = useMemo(() => {
    if (results.length === 0) return null;
    // During live analysis (not ended), always show latest
    if (!videoEnded) return latestResult;
    // Find the result closest to current playback time
    let best = results[0];
    let bestDist = Math.abs(best.frame_index * 0.5 - currentVideoTime);
    for (let i = 1; i < results.length; i++) {
      const dist = Math.abs(results[i].frame_index * 0.5 - currentVideoTime);
      if (dist < bestDist) {
        best = results[i];
        bestDist = dist;
      }
    }
    return best;
  }, [results, latestResult, videoEnded, currentVideoTime]);

  const regionScores = useMemo(() => computeRegionScores(results), [results]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
          Video Analysis
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Upload an endoscopy video for real-time PEACE score analysis
        </p>
        {isSaved && (
          <p className="mt-2 text-sm text-green-600 dark:text-green-400">
            Analysis saved to your dashboard.
          </p>
        )}
        {saveError && (
          <p className="mt-2 text-sm text-red-600 dark:text-red-400">
            Failed to save analysis: {saveError}
          </p>
        )}
      </div>

      {/* Video + Live Scores + Colon Map */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 lg:items-stretch">
        <div className="lg:col-span-2">
          {selectedFile ? (
            <VideoStreamPlayer
              ref={playerRef}
              file={selectedFile}
              isAnalyzing={isAnalyzing && isConnected}
              onFrameCapture={sendFrame}
              onVideoEnd={() => setVideoEnded(true)}
              onVideoReady={(d) => setVideoDuration(d)}
              onTimeUpdate={setCurrentVideoTime}
              captureIntervalMs={500}
              peaceScore={displayResult ? displayResult.peace_score.score as PeaceScore : null}
              motionDirection={displayResult?.motion ? displayResult.motion.direction as MotionDirection : null}
              region={displayResult?.region || null}
            />
          ) : (
            <VideoUploader onFileSelect={handleFileSelect} />
          )}
        </div>

        <div className="flex flex-col gap-4">
          {displayResult ? (
            <>
              <PeaceScoreCard
                score={displayResult.peace_score.score as PeaceScore}
                label={displayResult.peace_score.label}
                size="lg"
              />

              {displayResult.motion && (
                <Card>
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400">
                    Motion
                  </p>
                  <MotionIndicator
                    direction={displayResult.motion.direction as MotionDirection}
                  />
                </Card>
              )}

              {displayResult.region && (
                <Card>
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400">
                    Region
                  </p>
                  <p className="text-base font-medium capitalize text-neutral-900 dark:text-neutral-100">
                    {displayResult.region}
                  </p>
                </Card>
              )}
            </>
          ) : (
            <>
              <Card>
                <p className="mb-1 text-xs font-medium uppercase tracking-wider text-neutral-400">
                  PEACE Score
                </p>
                <div className="flex items-baseline gap-2">
                  <span className="text-4xl font-bold text-neutral-300 dark:text-neutral-600">
                    —
                  </span>
                  <span className="text-sm text-neutral-300 dark:text-neutral-600">/ 3</span>
                </div>
                <p className="mt-0.5 text-sm font-medium text-neutral-300 dark:text-neutral-600">
                  No data
                </p>
              </Card>

              <Card>
                <p className="mb-1 text-xs font-medium uppercase tracking-wider text-neutral-400">
                  Motion
                </p>
                <p className="text-base font-medium text-neutral-300 dark:text-neutral-600">
                  —
                </p>
                <p className="mt-0.5 text-sm text-neutral-300 dark:text-neutral-600">
                  No data
                </p>
              </Card>

              <Card className="flex-1">
                <p className="mb-1 text-xs font-medium uppercase tracking-wider text-neutral-400">
                  Region
                </p>
                <p className="text-base font-medium text-neutral-300 dark:text-neutral-600">
                  —
                </p>
                <p className="mt-0.5 text-sm text-neutral-300 dark:text-neutral-600">
                  No data
                </p>
              </Card>
            </>
          )}
        </div>

        {/* Colon Map */}
        <div className="flex">
          <ColonMap />
        </div>
      </div>

      {/* Video Details */}
      {selectedFile && (
        <Card>
          <CardHeader>
            <CardTitle>Video Details</CardTitle>
          </CardHeader>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div>
              <p className="text-xs text-neutral-500">Filename</p>
              <p className="text-lg font-semibold truncate">{selectedFile.name}</p>
            </div>
            <div>
              <p className="text-xs text-neutral-500">Duration</p>
              <p className="text-lg font-semibold">
                {videoDuration > 0 ? formatDuration(videoDuration) : "—"}
              </p>
            </div>
            <div>
              <p className="text-xs text-neutral-500">Frames Analyzed</p>
              <p className="text-lg font-semibold">{results.length || "—"}</p>
            </div>
            <div>
              <p className="text-xs text-neutral-500">Status</p>
              <p className="text-lg font-semibold">
                {videoEnded ? "Complete" : isAnalyzing && isConnected ? "Analyzing" : "Waiting"}
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Score Timeline */}
      {videoDuration > 0 ? (
        <PeaceScoreTimeline
          timeline={timeline}
          totalDuration={videoDuration}
          currentTime={currentVideoTime}
          onSeek={(t) => playerRef.current?.seekTo(t)}
        />
      ) : (
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
              Score Timeline
            </h3>
          </div>
          <Card className="flex h-48 items-center justify-center">
            <p className="text-sm text-neutral-400">
              Timeline will appear as frames are analyzed
            </p>
          </Card>
        </div>
      )}

      {/* Region Scores */}
      <div>
        <div className="mb-3">
          <h3 className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
            Region Scores
          </h3>
        </div>
        {Object.keys(regionScores).length > 0 ? (
          <PeaceScoreGrid byRegion={regionScores} />
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {REGION_ORDER.map((region) => (
              <Card key={region}>
                <p className="mb-1 text-xs font-medium uppercase tracking-wider text-neutral-400">
                  {REGION_LABELS[region]}
                </p>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-neutral-300 dark:text-neutral-600">
                    —
                  </span>
                  <span className="text-sm text-neutral-300 dark:text-neutral-600">/ 3</span>
                </div>
                <p className="mt-0.5 text-sm font-medium text-neutral-300 dark:text-neutral-600">
                  No data
                </p>
              </Card>
            ))}
          </div>
        )}
      </div>

    </div>
  );
}
