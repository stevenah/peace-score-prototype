"use client";

import { useCallback, useMemo, useState } from "react";
import { Activity } from "lucide-react";
import { VideoUploader } from "@/components/video/VideoUploader";
import { VideoStreamPlayer } from "@/components/video/VideoStreamPlayer";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { PeaceScoreGrid } from "@/components/scoring/PeaceScoreGrid";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { RegionMap } from "@/components/scoring/RegionMap";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import { formatDuration } from "@/lib/utils";
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
  const [videoDuration, setVideoDuration] = useState(0);

  const {
    isConnected,
    latestResult,
    results,
    framesProcessed,
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

  const handleVideoEnd = useCallback(() => {
    setVideoEnded(true);
  }, []);

  const handlePlayStateChange = useCallback(
    (playing: boolean) => {
      // Keep WebSocket alive even when paused so we don't lose connection
      if (!playing && videoEnded) {
        // Video finished - keep results
      }
    },
    [videoEnded],
  );

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setIsAnalyzing(false);
    setVideoEnded(false);
    setVideoDuration(0);
    reset();
  }, [reset]);

  // Build timeline from streaming results
  const timeline: TimelineEntry[] = useMemo(
    () =>
      results.map((r, i) => ({
        timestamp: i * 0.5, // Approximate based on capture interval
        frame_index: r.frame_index,
        motion: (r.motion?.direction || "stationary") as MotionDirection,
        region: (r.region || "stomach") as AnatomicalRegion,
        peace_score: r.peace_score.score as PeaceScore,
        confidence: r.peace_score.confidence,
      })),
    [results],
  );

  // Compute running region scores
  const regionScores = useMemo(() => computeRegionScores(results), [results]);

  // Overall score = min across all region scores
  const overallScore: PeaceScore | null = useMemo(() => {
    const regionValues = Object.values(regionScores);
    if (regionValues.length === 0) return null;
    return Math.min(
      ...regionValues.map((r) => r!.score),
    ) as PeaceScore;
  }, [regionScores]);

  const overallConfidence = useMemo(() => {
    if (results.length === 0) return 0;
    return (
      Math.round(
        (results.reduce((s, r) => s + r.peace_score.confidence, 0) /
          results.length) *
          100,
      ) / 100
    );
  }, [results]);

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

      {/* Upload Section */}
      {!selectedFile && (
        <VideoUploader onFileSelect={handleFileSelect} />
      )}

      {/* Streaming Analysis */}
      {selectedFile && (
        <div className="space-y-6">
          {/* Connection status */}
          <div className="flex items-center justify-between">
            <Badge
              color={isConnected ? "#22c55e" : "#f97316"}
              bgColor={isConnected ? "#f0fdf4" : "#fff7ed"}
            >
              <Activity className="h-3 w-3" />
              {isConnected ? "Analyzing" : "Connecting..."}
            </Badge>
            <span className="text-xs text-neutral-500">
              {framesProcessed} frames analyzed
            </span>
          </div>

          {/* Video + Live Scores side by side */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {/* Video Player */}
            <div className="lg:col-span-2">
              <VideoStreamPlayer
                file={selectedFile}
                isAnalyzing={isAnalyzing && isConnected}
                onFrameCapture={sendFrame}
                onPlayStateChange={handlePlayStateChange}
                onVideoEnd={handleVideoEnd}
                onVideoReady={(d) => setVideoDuration(d)}
                captureIntervalMs={500}
              />
            </div>

            {/* Live Score Panel */}
            <div className="space-y-4">
              {latestResult ? (
                <>
                  <PeaceScoreCard
                    score={latestResult.peace_score.score as PeaceScore}
                    confidence={latestResult.peace_score.confidence}
                    label={latestResult.peace_score.label}
                    size="lg"
                  />

                  {latestResult.motion && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Camera Motion</CardTitle>
                      </CardHeader>
                      <MotionIndicator
                        direction={
                          latestResult.motion.direction as MotionDirection
                        }
                        confidence={latestResult.motion.confidence}
                      />
                    </Card>
                  )}

                  {latestResult.region && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Current Region</CardTitle>
                      </CardHeader>
                      <p className="text-lg font-semibold capitalize text-neutral-900 dark:text-neutral-100">
                        {latestResult.region}
                      </p>
                    </Card>
                  )}

                  <Card>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-neutral-500">Latency</span>
                      <span className="font-medium text-neutral-900 dark:text-neutral-100">
                        {Math.round(latestResult.processing_time_ms)}ms
                      </span>
                    </div>
                  </Card>
                </>
              ) : (
                <Card className="py-12 text-center">
                  <p className="text-sm text-neutral-500">
                    {isConnected
                      ? "Press play to begin analysis..."
                      : "Connecting to analysis server..."}
                  </p>
                </Card>
              )}
            </div>
          </div>

          {/* Live Timeline - shows as data comes in */}
          {timeline.length > 2 && <PeaceScoreTimeline timeline={timeline} />}

          {/* Running Region Scores */}
          {Object.keys(regionScores).length > 0 && (
            <div>
              <h2 className="mb-4 text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                Scores by Region {!videoEnded && "(updating...)"}
              </h2>
              <PeaceScoreGrid byRegion={regionScores} />
            </div>
          )}

          {/* Region Map - once we have enough data */}
          {Object.keys(regionScores).length >= 2 && (
            <RegionMap byRegion={regionScores} />
          )}

          {/* Final Summary - shown when video ends */}
          {videoEnded && overallScore !== null && (
            <Card className="border-blue-200 dark:border-blue-800">
              <CardHeader>
                <CardTitle>Analysis Complete</CardTitle>
              </CardHeader>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <div>
                  <p className="text-xs text-neutral-500">Overall Score</p>
                  <p className="text-2xl font-bold" style={{
                    color: overallScore >= 2 ? "#22c55e" : overallScore >= 1 ? "#f97316" : "#ef4444"
                  }}>
                    {overallScore}/3
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500">Avg Confidence</p>
                  <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    {Math.round(overallConfidence * 100)}%
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500">Frames Analyzed</p>
                  <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    {framesProcessed}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-neutral-500">Video Duration</p>
                  <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                    {formatDuration(videoDuration)}
                  </p>
                </div>
              </div>
            </Card>
          )}

          {/* Restart */}
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
