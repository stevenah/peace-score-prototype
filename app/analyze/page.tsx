"use client";

import { useCallback, useMemo, useState } from "react";
import { RotateCcw } from "lucide-react";
import { VideoUploader } from "@/components/video/VideoUploader";
import { VideoStreamPlayer } from "@/components/video/VideoStreamPlayer";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { PeaceScoreGrid } from "@/components/scoring/PeaceScoreGrid";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { ColonMap } from "@/components/scoring/ColonMap";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import { PEACE_SCORE_LABELS, REGION_LABELS, REGION_ORDER } from "@/lib/constants";
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
  const [videoDuration, setVideoDuration] = useState(0);

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
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
          Video Analysis
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Upload an endoscopy video for real-time PEACE score analysis
        </p>
      </div>

      {/* Video + Live Scores + Colon Map */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 lg:items-stretch">
        <div className="lg:col-span-2">
          {selectedFile ? (
            <VideoStreamPlayer
              file={selectedFile}
              isAnalyzing={isAnalyzing && isConnected}
              onFrameCapture={sendFrame}
              onVideoEnd={() => setVideoEnded(true)}
              onVideoReady={(d) => setVideoDuration(d)}
              captureIntervalMs={500}
            />
          ) : (
            <VideoUploader onFileSelect={handleFileSelect} />
          )}
        </div>

        <div className="flex flex-col gap-4">
          {latestResult ? (
            <>
              <PeaceScoreCard
                score={latestResult.peace_score.score as PeaceScore}
                label={latestResult.peace_score.label}
                size="lg"
              />

              {latestResult.motion && (
                <Card>
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400">
                    Motion
                  </p>
                  <MotionIndicator
                    direction={latestResult.motion.direction as MotionDirection}
                  />
                </Card>
              )}

              {latestResult.region && (
                <Card>
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400">
                    Region
                  </p>
                  <p className="text-base font-medium capitalize text-neutral-900 dark:text-neutral-100">
                    {latestResult.region}
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

      {/* Score Timeline */}
      {videoDuration > 0 ? (
        <PeaceScoreTimeline timeline={timeline} totalDuration={videoDuration} />
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

      {videoEnded && results.length > 0 && (
        <Card className="text-center">
          <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
            Analysis complete
          </p>
          <p className="mt-1 text-xs text-neutral-400">
            {results.length} frames analyzed
          </p>
        </Card>
      )}

      {selectedFile && (
        <div className="text-center">
          <Button variant="ghost" onClick={handleReset}>
            <RotateCcw className="h-4 w-4" />
            Analyze another video
          </Button>
        </div>
      )}
    </div>
  );
}
