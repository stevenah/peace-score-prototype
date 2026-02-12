"use client";

import { useEffect, useRef, useState, useCallback, useImperativeHandle, forwardRef } from "react";
import { Play, Pause, RotateCcw, Maximize, Minimize, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { formatDuration } from "@/lib/utils";
import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS, MOTION_LABELS } from "@/lib/constants";
import type { PeaceScore, MotionDirection } from "@/lib/types";

export interface VideoStreamPlayerHandle {
  seekTo: (time: number) => void;
}

interface VideoStreamPlayerProps {
  file: File;
  isAnalyzing: boolean;
  onFrameCapture: (blob: Blob) => void;
  onPlayStateChange?: (playing: boolean) => void;
  onVideoEnd?: () => void;
  onVideoReady?: (duration: number) => void;
  onTimeUpdate?: (time: number) => void;
  captureIntervalMs?: number;
  /** Current PEACE score for fullscreen HUD overlay */
  peaceScore?: PeaceScore | null;
  /** Current motion direction for fullscreen HUD overlay */
  motionDirection?: MotionDirection | null;
  /** Current anatomical region for fullscreen HUD overlay */
  region?: string | null;
}

export const VideoStreamPlayer = forwardRef<VideoStreamPlayerHandle, VideoStreamPlayerProps>(function VideoStreamPlayer({
  file,
  isAnalyzing,
  onFrameCapture,
  onPlayStateChange,
  onVideoEnd,
  onVideoReady,
  onTimeUpdate,
  captureIntervalMs = 500,
  peaceScore,
  motionDirection,
  region,
}, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scrubRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Retraction warning: retracting with insufficient cleaning (score < 2)
  const showRetractionWarning =
    motionDirection === "retraction" &&
    peaceScore != null &&
    peaceScore < 2;

  useImperativeHandle(ref, () => ({
    seekTo(time: number) {
      const video = videoRef.current;
      if (!video) return;
      video.currentTime = time;
      setCurrentTime(time);
      onTimeUpdate?.(time);
    },
  }), [onTimeUpdate]);

  // Create object URL for the file
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Frame capture interval
  useEffect(() => {
    if (!isPlaying || !isAnalyzing) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    intervalRef.current = setInterval(() => {
      captureFrame();
    }, captureIntervalMs);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, isAnalyzing, captureIntervalMs]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.paused || video.ended) return;

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (blob) onFrameCapture(blob);
      },
      "image/jpeg",
      0.8,
    );
  }, [onFrameCapture]);

  function handlePlay() {
    const video = videoRef.current;
    if (!video) return;

    if (video.ended) {
      video.currentTime = 0;
    }

    video.play();
    setIsPlaying(true);
    onPlayStateChange?.(true);
  }

  function handlePause() {
    videoRef.current?.pause();
    setIsPlaying(false);
    onPlayStateChange?.(false);
  }

  function handleRestart() {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    video.play();
    setIsPlaying(true);
    onPlayStateChange?.(true);
  }

  const seekTo = useCallback((fraction: number) => {
    const video = videoRef.current;
    if (!video || duration === 0) return;
    const t = Math.max(0, Math.min(1, fraction)) * duration;
    video.currentTime = t;
    setCurrentTime(t);
    onTimeUpdate?.(t);
  }, [duration, onTimeUpdate]);

  const handleScrubFromEvent = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      const bar = scrubRef.current;
      if (!bar) return;
      const rect = bar.getBoundingClientRect();
      const fraction = (e.clientX - rect.left) / rect.width;
      seekTo(fraction);
    },
    [seekTo],
  );

  const handleScrubStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsScrubbing(true);
      const wasPlaying = isPlaying;
      videoRef.current?.pause();
      handleScrubFromEvent(e);

      const onMove = (ev: MouseEvent) => handleScrubFromEvent(ev);
      const onUp = () => {
        setIsScrubbing(false);
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        if (wasPlaying) {
          videoRef.current?.play();
        }
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    },
    [isPlaying, handleScrubFromEvent],
  );

  // Fullscreen toggle
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  }, []);

  // Sync fullscreen state with browser API
  useEffect(() => {
    const handleChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", handleChange);
    return () => document.removeEventListener("fullscreenchange", handleChange);
  }, []);

  // Escape key exits fullscreen (browser handles this, but we also allow 'f' to toggle)
  useEffect(() => {
    if (!isFullscreen) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "f" || e.key === "F") {
        toggleFullscreen();
      } else if (e.key === " " || e.key === "k" || e.key === "K") {
        e.preventDefault();
        if (isPlaying) handlePause();
        else handlePlay();
      }
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [isFullscreen, isPlaying, toggleFullscreen]);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const scoreColor = peaceScore != null ? PEACE_SCORE_COLORS[peaceScore] : undefined;
  const scoreLabel = peaceScore != null ? PEACE_SCORE_LABELS[peaceScore] : undefined;

  return (
    <div ref={containerRef} className={isFullscreen ? "flex h-screen w-screen flex-col bg-black" : "space-y-3"}>
      <div className={`relative overflow-hidden bg-black ${isFullscreen ? "flex-1" : "rounded-xl"}`}>
        {videoUrl && (
          <video
            ref={videoRef}
            src={videoUrl}
            className={isFullscreen ? "h-full w-full object-contain" : "w-full"}
            style={isFullscreen ? undefined : { aspectRatio: "16/9" }}
            playsInline
            muted
            onLoadedMetadata={(e) => {
              const d = e.currentTarget.duration;
              setDuration(d);
              onVideoReady?.(d);
            }}
            onTimeUpdate={(e) => {
              const t = e.currentTarget.currentTime;
              setCurrentTime(t);
              if (!isScrubbing) onTimeUpdate?.(t);
            }}
            onPlay={() => {
              setIsPlaying(true);
              onPlayStateChange?.(true);
            }}
            onPause={() => {
              setIsPlaying(false);
              onPlayStateChange?.(false);
            }}
            onEnded={() => {
              setIsPlaying(false);
              onPlayStateChange?.(false);
              onVideoEnd?.();
            }}
          />
        )}

        {/* Red flash overlay for retraction warning */}
        {showRetractionWarning && (
          <div className="pointer-events-none absolute inset-0 animate-[flash-red_1s_ease-in-out_infinite] border-[6px] border-red-500/0">
            <div className="absolute inset-0 bg-red-500/0 animate-[flash-red-bg_1s_ease-in-out_infinite]" />
          </div>
        )}

        {/* Retraction warning badge */}
        {showRetractionWarning && (
          <div className="absolute left-1/2 top-6 -translate-x-1/2 flex items-center gap-2 rounded-full bg-red-600/90 px-4 py-2 text-white shadow-lg backdrop-blur-sm animate-pulse">
            <AlertTriangle className="h-5 w-5" />
            <span className="text-sm font-semibold">Insufficient cleaning — retracting</span>
          </div>
        )}

        {/* Fullscreen HUD overlay */}
        {isFullscreen && (
          <div className="pointer-events-none absolute inset-0">
            {/* Top-left: PEACE score */}
            {peaceScore != null && (
              <div className="absolute left-5 top-5 rounded-xl bg-black/60 px-4 py-3 backdrop-blur-sm">
                <p className="text-[10px] font-semibold uppercase tracking-widest text-white/60">
                  PEACE Score
                </p>
                <div className="flex items-baseline gap-1.5">
                  <span className="text-4xl font-bold" style={{ color: scoreColor }}>
                    {peaceScore}
                  </span>
                  <span className="text-sm text-white/50">/ 3</span>
                </div>
                <p className="text-sm font-medium" style={{ color: scoreColor }}>
                  {scoreLabel}
                </p>
              </div>
            )}

            {/* Top-right: Motion + Region */}
            <div className="absolute right-5 top-5 flex flex-col items-end gap-2">
              {motionDirection && (
                <div className="rounded-lg bg-black/60 px-3 py-2 backdrop-blur-sm">
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-white/60">
                    Motion
                  </p>
                  <p className="text-sm font-medium text-white">
                    {MOTION_LABELS[motionDirection]}
                  </p>
                </div>
              )}
              {region && (
                <div className="rounded-lg bg-black/60 px-3 py-2 backdrop-blur-sm">
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-white/60">
                    Region
                  </p>
                  <p className="text-sm font-medium capitalize text-white">{region}</p>
                </div>
              )}
            </div>

            {/* Bottom: time + progress */}
            <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/70 to-transparent px-5 pb-4 pt-10">
              <div className="mb-2 h-1 w-full overflow-hidden rounded-full bg-white/20">
                <div
                  className="h-full rounded-full bg-white/80 transition-[width] duration-100 ease-linear"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-xs text-white/70">
                <span className="tabular-nums">
                  {formatDuration(currentTime)} / {formatDuration(duration)}
                </span>
                <span className="text-[10px] uppercase tracking-wider text-white/40">
                  Press F to exit · Space to play/pause
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Playback overlay when paused (non-fullscreen) */}
        {!isPlaying && videoUrl && !isFullscreen && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30">
            <button
              onClick={handlePlay}
              className="flex h-16 w-16 items-center justify-center rounded-full bg-white/90 text-neutral-900 shadow-lg transition-transform hover:scale-110"
            >
              <Play className="ml-1 h-7 w-7" />
            </button>
          </div>
        )}

        {/* Playback overlay when paused (fullscreen) */}
        {!isPlaying && videoUrl && isFullscreen && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <button
              onClick={handlePlay}
              className="flex h-20 w-20 items-center justify-center rounded-full bg-white/90 text-neutral-900 shadow-xl transition-transform hover:scale-110"
            >
              <Play className="ml-1 h-9 w-9" />
            </button>
          </div>
        )}

        {/* Fullscreen toggle button (always visible, top-right in non-fullscreen) */}
        {videoUrl && !isFullscreen && (
          <button
            onClick={toggleFullscreen}
            className="absolute right-3 top-3 rounded-lg bg-black/50 p-2 text-white/80 backdrop-blur-sm transition-colors hover:bg-black/70 hover:text-white"
            title="Enter fullscreen (F)"
          >
            <Maximize className="h-4 w-4" />
          </button>
        )}

        {/* Exit fullscreen button */}
        {isFullscreen && (
          <button
            onClick={toggleFullscreen}
            className="absolute right-5 bottom-14 z-10 rounded-lg bg-black/50 p-2 text-white/80 backdrop-blur-sm transition-colors hover:bg-black/70 hover:text-white"
            title="Exit fullscreen (F)"
          >
            <Minimize className="h-5 w-5" />
          </button>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {/* Controls (hidden in fullscreen) */}
      {!isFullscreen && (
        <div className="flex items-center gap-3">
          {isPlaying ? (
            <Button variant="secondary" size="sm" onClick={handlePause}>
              <Pause className="h-4 w-4" />
              Pause
            </Button>
          ) : (
            <Button size="sm" onClick={handlePlay}>
              <Play className="h-4 w-4" />
              {currentTime > 0 && currentTime < duration ? "Resume" : "Play"}
            </Button>
          )}
          <Button variant="ghost" size="sm" onClick={handleRestart}>
            <RotateCcw className="h-4 w-4" />
          </Button>

          <div
            ref={scrubRef}
            className="group relative flex-1 cursor-pointer py-1"
            onMouseDown={handleScrubStart}
          >
            <div className="h-2 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
              <div
                className="h-full rounded-full transition-[width] duration-100 ease-linear"
                style={{
                  width: `${progress}%`,
                  backgroundColor: isAnalyzing ? "#3b82f6" : "#9ca3af",
                }}
              />
            </div>
            {/* Scrub thumb */}
            <div
              className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-neutral-600 opacity-0 shadow transition-opacity group-hover:opacity-100 dark:border-neutral-800 dark:bg-neutral-300"
              style={{ left: `${progress}%` }}
            />
          </div>

          <span className="text-xs tabular-nums text-neutral-500">
            {formatDuration(currentTime)} / {formatDuration(duration)}
          </span>
        </div>
      )}
    </div>
  );
});
