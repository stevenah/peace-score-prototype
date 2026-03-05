"use client";

import { useEffect, useRef, useState, useCallback, useImperativeHandle, forwardRef } from "react";
import Hls from "hls.js";
import { Play, Pause, RotateCcw, Maximize, Minimize, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { formatDuration } from "@/lib/utils";
import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS, MOTION_LABELS } from "@/lib/constants";
import type { PeaceScore, MotionDirection } from "@/lib/types";

export interface VideoStreamPlayerHandle {
  seekTo: (time: number) => void;
  pause: () => void;
}

interface VideoStreamPlayerProps {
  source: File | string;
  isAnalyzing: boolean;
  onFrameCapture: (blob: Blob, videoTime: number) => void;
  onPlayStateChange?: (playing: boolean) => void;
  onVideoEnd?: () => void;
  onVideoReady?: (duration: number) => void;
  onTimeUpdate?: (time: number) => void;
  captureIntervalMs?: number;
  /** When true, pauses video to let analysis catch up. Resumes automatically when false. */
  syncPause?: boolean;
  /** Current PEACE score for fullscreen HUD overlay */
  peaceScore?: PeaceScore | null;
  /** Current motion direction for fullscreen HUD overlay */
  motionDirection?: MotionDirection | null;
  /** Current anatomical region for fullscreen HUD overlay */
  region?: string | null;
  /** Extra content rendered at the right end of the controls bar */
  controlsRight?: React.ReactNode;
  /** Hides scrub bar/duration and shows LIVE indicator */
  isLiveStream?: boolean;
}

export const VideoStreamPlayer = forwardRef<VideoStreamPlayerHandle, VideoStreamPlayerProps>(function VideoStreamPlayer({
  source,
  isAnalyzing,
  onFrameCapture,
  onPlayStateChange,
  onVideoEnd,
  onVideoReady,
  onTimeUpdate,
  captureIntervalMs = 500,
  syncPause = false,
  peaceScore,
  motionDirection,
  region,
  controlsRight,
  isLiveStream = false,
}, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scrubRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const analyzedBucketsRef = useRef<Set<number>>(new Set());
  const syncPausedRef = useRef(false);
  const hlsRef = useRef<Hls | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [corsError, setCorsError] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Withdrawal warning: withdrawing with insufficient cleaning (score < 2)
  const showWithdrawalWarning =
    motionDirection === "withdrawal" &&
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
    pause() {
      videoRef.current?.pause();
      setIsPlaying(false);
      onPlayStateChange?.(false);
    },
  }), [onTimeUpdate, onPlayStateChange]);

  // Resolve source to a playable URL (File blob, HLS, or direct URL)
  useEffect(() => {
    const video = videoRef.current;

    // Cleanup previous HLS instance
    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }

    if (source instanceof File) {
      const url = URL.createObjectURL(source);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    }

    // String URL
    const url = source;
    const isHls = url.includes(".m3u8");

    if (isHls && Hls.isSupported() && video) {
      const hls = new Hls({ enableWorker: true, lowLatencyMode: true });
      hlsRef.current = hls;
      hls.loadSource(url);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        setVideoUrl(url);
      });
      hls.on(Hls.Events.ERROR, (_event, data) => {
        if (data.fatal) {
          console.error("HLS fatal error:", data);
        }
      });
      return () => {
        hls.destroy();
        hlsRef.current = null;
      };
    }

    // Safari native HLS or direct URL
    setVideoUrl(url);
  }, [source]);

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

  // Pause/resume video to keep it in sync with backend processing
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !isAnalyzing) return;

    if (syncPause && !video.paused && isPlaying) {
      // Backend is behind — pause video until it catches up
      syncPausedRef.current = true;
      video.pause();
    } else if (!syncPause && syncPausedRef.current) {
      // Backend caught up — resume playback
      syncPausedRef.current = false;
      video.play();
    }
  }, [syncPause, isAnalyzing, isPlaying]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.paused || video.ended) return;

    // Quantize to capture interval to avoid re-analyzing the same time bucket
    const bucket = Math.floor(video.currentTime * 1000 / captureIntervalMs);
    if (analyzedBucketsRef.current.has(bucket)) return;
    analyzedBucketsRef.current.add(bucket);

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    try {
      ctx.drawImage(video, 0, 0);
    } catch {
      // SecurityError from cross-origin stream without CORS headers
      setCorsError(true);
      return;
    }
    // Snapshot the time now — toBlob is async and video.currentTime may advance
    const captureTime = video.currentTime;
    canvas.toBlob(
      (blob) => {
        if (blob) onFrameCapture(blob, captureTime);
      },
      "image/jpeg",
      0.8,
    );
  }, [onFrameCapture, captureIntervalMs]);

  const handlePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.ended) {
      video.currentTime = 0;
    }

    video.play();
    setIsPlaying(true);
    onPlayStateChange?.(true);
  }, [onPlayStateChange]);

  const handlePause = useCallback(() => {
    videoRef.current?.pause();
    setIsPlaying(false);
    onPlayStateChange?.(false);
  }, [onPlayStateChange]);

  const handleRestart = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    video.play();
    setIsPlaying(true);
    onPlayStateChange?.(true);
  }, [onPlayStateChange]);

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
  }, [isFullscreen, isPlaying, toggleFullscreen, handlePause, handlePlay]);

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  const scoreColor = peaceScore != null ? PEACE_SCORE_COLORS[peaceScore] : undefined;
  const scoreLabel = peaceScore != null ? PEACE_SCORE_LABELS[peaceScore] : undefined;

  return (
    <div ref={containerRef} className={isFullscreen ? "flex h-screen w-screen flex-col bg-black" : "space-y-3"}>
      <div className={`relative overflow-hidden bg-black ${isFullscreen ? "flex-1" : "rounded-xl"}`}>
        {videoUrl && (
          <video
            ref={videoRef}
            src={hlsRef.current ? undefined : videoUrl}
            crossOrigin={typeof source === "string" ? "anonymous" : undefined}
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

        {/* Red flash overlay for withdrawal warning */}
        {showWithdrawalWarning && (
          <div className="pointer-events-none absolute inset-0 animate-[flash-red_1s_ease-in-out_infinite] border-[6px] border-red-500/0">
            <div className="absolute inset-0 bg-red-500/0 animate-[flash-red-bg_1s_ease-in-out_infinite]" />
          </div>
        )}

        {/* Withdrawal warning badge */}
        {showWithdrawalWarning && (
          <div className="absolute left-1/2 top-6 -translate-x-1/2 flex items-center gap-2 rounded-full bg-red-600/90 px-4 py-2 text-white shadow-lg backdrop-blur-sm animate-pulse">
            <AlertTriangle className="h-5 w-5" />
            <span className="text-sm font-semibold">Insufficient cleaning — withdrawing</span>
          </div>
        )}

        {/* CORS error banner */}
        {corsError && (
          <div className="absolute left-1/2 bottom-6 -translate-x-1/2 rounded-lg bg-amber-600/90 px-4 py-2 text-white shadow-lg backdrop-blur-sm">
            <span className="text-sm font-medium">Cannot capture frames — stream blocked by CORS policy</span>
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
              {!isLiveStream && (
                <div className="mb-2 h-1 w-full overflow-hidden rounded-full bg-white/20">
                  <div
                    className="h-full rounded-full bg-white/80 transition-[width] duration-100 ease-linear"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              )}
              <div className="flex items-center justify-between text-xs text-white/70">
                {isLiveStream ? (
                  <div className="flex items-center gap-2">
                    <span className="relative flex h-2.5 w-2.5">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                      <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-red-500" />
                    </span>
                    <span className="text-xs font-medium text-red-400">LIVE</span>
                  </div>
                ) : (
                  <span className="tabular-nums">
                    {formatDuration(currentTime)} / {formatDuration(duration)}
                  </span>
                )}
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
          {!isLiveStream && (
            <Button variant="ghost" size="sm" onClick={handleRestart}>
              <RotateCcw className="h-4 w-4" />
            </Button>
          )}

          {isLiveStream ? (
            <div className="flex items-center gap-2">
              <span className="relative flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-red-500" />
              </span>
              <span className="text-xs font-medium text-red-500">LIVE</span>
            </div>
          ) : (
            <>
              <div
                ref={scrubRef}
                className="group relative flex-1 cursor-pointer py-1"
                onMouseDown={handleScrubStart}
              >
                <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
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
                  className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-primary opacity-0 shadow transition-opacity group-hover:opacity-100 dark:border-card"
                  style={{ left: `${progress}%` }}
                />
              </div>

              <span className="text-xs tabular-nums text-muted-foreground">
                {formatDuration(currentTime)} / {formatDuration(duration)}
              </span>
            </>
          )}

          {controlsRight}
        </div>
      )}
    </div>
  );
});
