"use client";

import { useCallback, useMemo, useState } from "react";
import { VideoUploader } from "@/components/video/VideoUploader";
import { VideoStreamPlayer } from "@/components/video/VideoStreamPlayer";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { PeaceScoreGrid } from "@/components/scoring/PeaceScoreGrid";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { RegionMap } from "@/components/scoring/RegionMap";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import { PEACE_SCORE_LABELS, REGION_ORDER } from "@/lib/constants";
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
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [videoEnded, setVideoEnded] = useState(false);

  const {
    isConnected,
    latestResult,
    results,
    sendFrame,
    reset,
  } = useLiveFeed({
    enabled: isAnalyzing,
    wsUrl: "ws://localhost:8000/api/v1/ws/live",
  });

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setVideoEnded(false);
    setIsAnalyzing(true);
  }, []);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setIsAnalyzing(false);
    setVideoEnded(false);
    reset();
  }, [reset]);

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

  const regionScores = useMemo(() => computeRegionScores(results), [results]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
          Video Analysis
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Upload an endoscopy video for real-time PEACE score analysis
        </p>
      </div>

      {!selectedFile && <VideoUploader onFileSelect={handleFileSelect} />}

      {selectedFile && (
        <div className="space-y-6">
          {/* Video + Live Scores */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <VideoStreamPlayer
                file={selectedFile}
                isAnalyzing={isAnalyzing && isConnected}
                onFrameCapture={sendFrame}
                onVideoEnd={() => setVideoEnded(true)}
                captureIntervalMs={500}
              />
            </div>

            <div className="space-y-4">
              {latestResult ? (
                <>
                  <PeaceScoreCard
                    score={latestResult.peace_score.score as PeaceScore}
                    label={latestResult.peace_score.label}
                    size="lg"
                  />

                  {latestResult.motion && (
                    <div className="rounded-xl border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
                      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400">
                        Motion
                      </p>
                      <MotionIndicator
                        direction={latestResult.motion.direction as MotionDirection}
                      />
                    </div>
                  )}

                  {latestResult.region && (
                    <div className="rounded-xl border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
                      <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400">
                        Region
                      </p>
                      <p className="text-base font-medium capitalize text-neutral-900 dark:text-neutral-100">
                        {latestResult.region}
                      </p>
                    </div>
                  )}
                </>
              ) : (
                <div className="rounded-xl border border-neutral-200 bg-white p-8 text-center dark:border-neutral-800 dark:bg-neutral-900">
                  <p className="text-sm text-neutral-400">
                    {isConnected ? "Press play to begin" : "Connecting..."}
                  </p>
                </div>
              )}
            </div>
          </div>

          {timeline.length > 2 && <PeaceScoreTimeline timeline={timeline} />}

          {Object.keys(regionScores).length > 0 && (
            <PeaceScoreGrid byRegion={regionScores} />
          )}

          {Object.keys(regionScores).length >= 2 && (
            <RegionMap byRegion={regionScores} />
          )}

          {videoEnded && results.length > 0 && (
            <div className="rounded-xl border border-neutral-200 bg-white p-5 text-center dark:border-neutral-800 dark:bg-neutral-900">
              <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                Analysis complete
              </p>
              <p className="mt-1 text-xs text-neutral-400">
                {results.length} frames analyzed
              </p>
            </div>
          )}

          <div className="text-center">
            <button
              onClick={handleReset}
              className="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400"
            >
              Analyze another video
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
