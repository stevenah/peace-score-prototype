"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Loader2, CheckCircle2 } from "lucide-react";
import { saveLiveAnalysis } from "@/lib/api-client";
import { VideoUploader } from "@/components/video/VideoUploader";
import { VideoStreamPlayer, type VideoStreamPlayerHandle } from "@/components/video/VideoStreamPlayer";
import { MotionVisual } from "@/components/analysis/MotionVisual";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { RegionHighlight } from "@/components/scoring/RegionHighlight";
import { Card } from "@/components/ui/Card";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS } from "@/lib/constants";
import type {
  PeaceScore,
  MotionDirection,
  AnatomicalRegion,
  TimelineEntry,
} from "@/lib/types";

const CAPTURE_INTERVAL_MS = 500;
const CAPTURE_INTERVAL_S = CAPTURE_INTERVAL_MS / 1000;

export type ConnectionStatus = {
  isAnalyzing: boolean;
  isConnected: boolean;
  isConnecting: boolean;
  connectionError: string | null;
};

export function LiveAnalysis({
  onConnectionStatus,
}: {
  onConnectionStatus?: (status: ConnectionStatus) => void;
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

    const overallScore = Math.min(...results.map((r) => r.peace_score.score));

    saveLiveAnalysis({
      filename: selectedFile?.name ?? "live-analysis",
      overallScore,
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

  const timeline: TimelineEntry[] = useMemo(
    () =>
      results.map((r, i) => ({
        timestamp: captureTimes[i] ?? i * CAPTURE_INTERVAL_S,
        frame_index: r.frame_index,
        motion: (r.motion?.direction || "stationary") as MotionDirection,
        region: (r.region || "stomach") as AnatomicalRegion,
        peace_score: r.peace_score.score as PeaceScore,
        confidence: r.peace_score.confidence,
      })),
    [results, captureTimes],
  );

  // Show the result whose actual capture time is closest to the current video time.
  // This keeps the score card, motion, and region in sync with the video position
  // during both live playback and post-analysis scrubbing.
  const displayResult = useMemo(() => {
    if (results.length === 0) return null;
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
    return results[bestIdx];
  }, [results, captureTimes, currentVideoTime]);


  return (
    <div className="space-y-8">
      {/* Auto-save status */}
      {selectedFile && (isSaving || isSaved || saveError) && (
        <div className="flex items-center gap-2">
          {isSaving && (
            <>
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Saving analysis...</p>
            </>
          )}
          {isSaved && (
            <>
              <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
              <p className="text-sm text-green-600 dark:text-green-400">
                Analysis saved to your dashboard.
              </p>
            </>
          )}
          {saveError && (
            <p className="text-sm text-red-600 dark:text-red-400">
              Failed to save: {saveError}
            </p>
          )}
        </div>
      )}

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
              peaceScore={displayResult ? displayResult.peace_score.score as PeaceScore : null}
              motionDirection={displayResult?.motion ? displayResult.motion.direction as MotionDirection : null}
              region={displayResult?.region || null}
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
                {displayResult.motion ? (
                  <MotionVisual
                    direction={displayResult.motion.direction as MotionDirection}
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
            <h3 className="text-sm font-medium text-foreground/80">
              Score Timeline
            </h3>
          </div>
          <Card className="flex h-48 items-center justify-center">
            <p className="text-sm text-muted-foreground/60">
              Timeline will appear as frames are analyzed
            </p>
          </Card>
        </div>
      )}
    </div>
  );
}
