"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { saveLiveAnalysis } from "@/lib/api-client";
import { VideoUploader } from "@/components/video/VideoUploader";
import { VideoStreamPlayer, type VideoStreamPlayerHandle } from "@/components/video/VideoStreamPlayer";
import { MotionVisual } from "@/components/analysis/MotionVisual";
import { PeaceScoreGrid } from "@/components/scoring/PeaceScoreGrid";
import { RegionHighlight } from "@/components/scoring/RegionHighlight";
import { Card } from "@/components/ui/Card";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS } from "@/lib/constants";
import type {
  PeaceScore,
  MotionDirection,
  AnatomicalRegion,
  RegionScore,
} from "@/lib/types";

const CAPTURE_INTERVAL_MS = 500;
const CAPTURE_INTERVAL_S = CAPTURE_INTERVAL_MS / 1000;
/** Max frames awaiting backend response before pausing video to stay in sync */
const MAX_IN_FLIGHT_FRAMES = 2;

/** Minimum confidence to trust a motion direction from the backend */
const MOTION_CONFIDENCE_THRESHOLD = 0.6;
/**
 * Number of consecutive frames that must agree on a non-stationary direction
 * before the UI switches away from "stationary".  Prevents flickering.
 */
const MOTION_CONSENSUS_FRAMES = 3;

export type ConnectionStatus = {
  isAnalyzing: boolean;
  isConnected: boolean;
  isConnecting: boolean;
  connectionError: string | null;
};

export type SaveStatus = {
  isSaving: boolean;
  isSaved: boolean;
  saveError: string | null;
};

export function LiveAnalysis({
  onConnectionStatus,
  onSaveStatus,
}: {
  onConnectionStatus?: (status: ConnectionStatus) => void;
  onSaveStatus?: (status: SaveStatus) => void;
}) {
  const playerRef = useRef<VideoStreamPlayerHandle>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // Whether analysis is actively running (WS connected, frames being captured)
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [currentVideoTime, setCurrentVideoTime] = useState(0);

  const {
    isConnected,
    isConnecting,
    connectionError,
    results,
    inFlightFrames,
    sendFrame,
  } = useLiveFeed({
    enabled: isAnalyzing,
  });

  // Report connection status to parent
  useEffect(() => {
    onConnectionStatus?.({ isAnalyzing, isConnected, isConnecting, connectionError });
  }, [onConnectionStatus, isAnalyzing, isConnected, isConnecting, connectionError]);

  // Track actual video time for each captured frame so timeline stays in sync
  const [captureTimes, setCaptureTimes] = useState<number[]>([]);

  const [isSaved, setIsSaved] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Report save status to parent
  useEffect(() => {
    onSaveStatus?.({ isSaving, isSaved, saveError });
  }, [onSaveStatus, isSaving, isSaved, saveError]);

  const handleFrameCapture = useCallback((blob: Blob, videoTime: number) => {
    const sent = sendFrame(blob);
    if (sent) {
      setCaptureTimes((prev) => [...prev, videoTime]);
    }
  }, [sendFrame]);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setIsAnalyzing(true);
    setIsSaved(false);
    setIsSaving(false);
    setSaveError(null);
    setCaptureTimes([]);
  }, []);

  // Auto-save analysis when video ends
  const handleVideoEnd = useCallback(() => {
    setIsAnalyzing(false);

    if (results.length === 0 || isSaving || isSaved) return;

    setIsSaving(true);
    setSaveError(null);

    const scores = results.map((r) => r.peace_score.score as number);
    const overallScore = Math.min(...scores);
    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    const avgScore = Math.round((scores.reduce((sum, s) => sum + s, 0) / scores.length) * 100) / 100;

    saveLiveAnalysis({
      filename: selectedFile?.name ?? "live-analysis",
      overallScore,
      minScore,
      maxScore,
      avgScore,
      framesAnalyzed: results.length,
      duration: videoDuration > 0 ? videoDuration : null,
      timeline: results.map((r, i) => ({
        timestamp: captureTimes[i] ?? i * CAPTURE_INTERVAL_S,
        frame_index: r.frame_index,
        motion: r.motion?.direction || "stationary",
        region: r.region || "stomach",
        peace_score: r.peace_score.score,
        confidence: r.peace_score.confidence,
      })),
      videoFile: selectedFile ?? undefined,
    })
      .then(() => {
        setIsSaved(true);
        setIsSaving(false);
      })
      .catch((err) => {
        setSaveError(err.message);
        setIsSaving(false);
      });
  }, [results, isSaving, isSaved, selectedFile, videoDuration, captureTimes]);

  // Compute per-region PEACE scores from all results seen so far
  const byRegion = useMemo(() => {
    const regionFrames: Record<string, { scores: number[]; confidences: number[]; labels: string[] }> = {};
    for (const r of results) {
      const region = r.region || "stomach";
      if (!regionFrames[region]) {
        regionFrames[region] = { scores: [], confidences: [], labels: [] };
      }
      regionFrames[region].scores.push(r.peace_score.score as number);
      regionFrames[region].confidences.push(r.peace_score.confidence);
      if (r.peace_score.label) regionFrames[region].labels.push(r.peace_score.label);
    }

    const byRegionMap: Partial<Record<AnatomicalRegion, RegionScore>> = {};
    for (const [region, data] of Object.entries(regionFrames)) {
      const score = Math.min(...data.scores) as PeaceScore;
      const avgConfidence = data.confidences.reduce((a, b) => a + b, 0) / data.confidences.length;
      byRegionMap[region as AnatomicalRegion] = {
        score,
        label: PEACE_SCORE_LABELS[score],
        confidence: avgConfidence,
        region: region as AnatomicalRegion,
        frame_scores: [],
      };
    }
    return byRegionMap;
  }, [results]);

  // Show the result whose actual capture time is closest to the current video time.
  // This keeps the score card, motion, and region in sync with the video position
  // during both live playback and post-analysis scrubbing.
  const displayResultIdx = useMemo(() => {
    if (results.length === 0) return -1;
    let bestIdx = 0;
    let bestDist = Math.abs((captureTimes[0] ?? 0) - currentVideoTime);
    for (let i = 1; i < results.length; i++) {
      const t = captureTimes[i] ?? i * CAPTURE_INTERVAL_S;
      const dist = Math.abs(t - currentVideoTime);
      if (dist < bestDist) {
        bestIdx = i;
        bestDist = dist;
      }
    }
    return bestIdx;
  }, [results, captureTimes, currentVideoTime]);

  const displayResult = displayResultIdx >= 0 ? results[displayResultIdx] : null;

  // Smooth the motion direction: require MOTION_CONSENSUS_FRAMES consecutive
  // high-confidence frames agreeing on a non-stationary direction before
  // switching away from "stationary". This prevents flickering.
  const smoothedMotionDirection: MotionDirection | null = useMemo(() => {
    if (displayResultIdx < 0) return null;

    // Look at the window of frames ending at the current display frame
    const windowStart = Math.max(0, displayResultIdx - MOTION_CONSENSUS_FRAMES + 1);
    const window = results.slice(windowStart, displayResultIdx + 1);

    // Count how many recent frames agree on each non-stationary direction
    // with sufficient confidence
    let consecutiveDir: MotionDirection | null = null;
    let streak = 0;

    for (const r of window) {
      const dir = r.motion?.direction ?? "stationary";
      const conf = r.motion?.confidence ?? 0;

      if (dir !== "stationary" && conf >= MOTION_CONFIDENCE_THRESHOLD) {
        if (dir === consecutiveDir) {
          streak++;
        } else {
          consecutiveDir = dir;
          streak = 1;
        }
      } else {
        consecutiveDir = null;
        streak = 0;
      }
    }

    if (consecutiveDir && streak >= MOTION_CONSENSUS_FRAMES) {
      return consecutiveDir;
    }

    return "stationary";
  }, [results, displayResultIdx]);

  const handleStepFrame = useCallback((delta: number) => {
    if (results.length === 0) return;
    const nextIdx = Math.max(0, Math.min(results.length - 1, displayResultIdx + delta));
    const time = captureTimes[nextIdx] ?? nextIdx * CAPTURE_INTERVAL_S;
    playerRef.current?.seekTo(time);
    playerRef.current?.pause();
  }, [results.length, displayResultIdx, captureTimes]);

  const hasFrames = results.length > 0;

  const frameStepper = (
    <div className={`flex items-center gap-2 ${!hasFrames ? "opacity-30" : ""}`}>
      <span className="mx-1 h-4 w-px bg-border" />
      <button
        type="button"
        onClick={() => handleStepFrame(-1)}
        disabled={!hasFrames || displayResultIdx <= 0}
        className="flex h-7 w-7 items-center justify-center rounded-md border border-border bg-card text-foreground/70 transition-colors hover:bg-accent disabled:opacity-30 disabled:pointer-events-none"
      >
        <ChevronLeft className="h-3.5 w-3.5" />
      </button>
      <span className="text-xs tabular-nums text-muted-foreground">
        {hasFrames ? `${displayResultIdx + 1} / ${results.length}` : "— / —"}
      </span>
      <button
        type="button"
        onClick={() => handleStepFrame(1)}
        disabled={!hasFrames || displayResultIdx >= results.length - 1}
        className="flex h-7 w-7 items-center justify-center rounded-md border border-border bg-card text-foreground/70 transition-colors hover:bg-accent disabled:opacity-30 disabled:pointer-events-none"
      >
        <ChevronRight className="h-3.5 w-3.5" />
      </button>
    </div>
  );

  return (
    <div className="space-y-8">
      {/* Video + Live Scores */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-4 lg:items-stretch">
        <div className="lg:col-span-3">
          {selectedFile ? (
            <VideoStreamPlayer
              ref={playerRef}
              file={selectedFile}
              isAnalyzing={isAnalyzing && isConnected}
              onFrameCapture={handleFrameCapture}
              onVideoEnd={handleVideoEnd}
              onVideoReady={(d) => setVideoDuration(d)}
              onTimeUpdate={setCurrentVideoTime}
              captureIntervalMs={CAPTURE_INTERVAL_MS}
              syncPause={isAnalyzing && inFlightFrames >= MAX_IN_FLIGHT_FRAMES}
              peaceScore={displayResult ? displayResult.peace_score.score as PeaceScore : null}
              motionDirection={smoothedMotionDirection}
              region={displayResult?.region || null}
              controlsRight={frameStepper}
            />
          ) : (
            <VideoUploader onFilesSelect={(files) => handleFileSelect(files[0])} />
          )}
        </div>

        <div className="flex flex-col gap-4 lg:min-h-0 lg:overflow-hidden">
          {displayResult ? (
            <>
              <Card className="flex min-h-0 flex-1 flex-col items-center justify-between overflow-hidden pb-4 pt-3 text-center">
                <p className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  PEACE Score
                </p>
                <div className="flex items-baseline gap-1.5">
                  <span
                    className="text-4xl font-bold"
                    style={{ color: PEACE_SCORE_COLORS[displayResult.peace_score.score as PeaceScore] }}
                  >
                    {displayResult.peace_score.score}
                  </span>
                  <span className="text-sm text-muted-foreground/60">/ 3</span>
                </div>
                <p
                  className="shrink-0 text-sm font-medium"
                  style={{ color: PEACE_SCORE_COLORS[displayResult.peace_score.score as PeaceScore] }}
                >
                  {displayResult.peace_score.label || PEACE_SCORE_LABELS[displayResult.peace_score.score as PeaceScore]}
                </p>
              </Card>

              <Card className="flex min-h-0 flex-1 flex-col items-center justify-between overflow-hidden pb-4 pt-3 text-center">
                <p className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Motion
                </p>
                {smoothedMotionDirection && smoothedMotionDirection !== "stationary" ? (
                  <MotionVisual
                    direction={smoothedMotionDirection}
                  />
                ) : displayResult.motion ? (
                  <MotionVisual direction="stationary" />
                ) : (
                  <p className="text-sm font-medium text-muted-foreground/30">—</p>
                )}
                <span />
              </Card>

              <Card className="flex min-h-0 flex-1 flex-col items-center justify-between overflow-hidden pb-4 pt-3 text-center">
                <p className="shrink-0 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Region
                </p>
                {displayResult.region ? (
                  <RegionHighlight activeRegion={displayResult.region as AnatomicalRegion} />
                ) : (
                  <p className="text-sm font-medium text-muted-foreground/30">—</p>
                )}
                <p className="shrink-0 text-sm font-medium capitalize text-foreground">
                  {displayResult.region || "\u00A0"}
                </p>
              </Card>
            </>
          ) : (
            <>
              <Card className="flex flex-1 flex-col items-center text-center">
                <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  PEACE Score
                </p>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-4xl font-bold text-muted-foreground/30">
                    —
                  </span>
                  <span className="text-sm text-muted-foreground/30">/ 3</span>
                </div>
                <p className="mt-1 text-sm font-medium text-muted-foreground/30">
                  No data
                </p>
              </Card>

              <Card className="flex flex-1 flex-col items-center text-center">
                <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Motion
                </p>
                <p className="text-sm font-medium text-muted-foreground/30">
                  —
                </p>
              </Card>

              <Card className="flex flex-1 flex-col items-center text-center">
                <p className="mb-3 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Region
                </p>
                <p className="text-sm font-medium text-muted-foreground/30">
                  —
                </p>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* Per-region PEACE status cards */}
      <PeaceScoreGrid byRegion={byRegion} />
    </div>
  );
}
