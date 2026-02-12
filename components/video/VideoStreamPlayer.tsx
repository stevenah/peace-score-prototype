"use client";

import { useEffect, useRef, useState, useCallback, useImperativeHandle, forwardRef } from "react";
import { Play, Pause, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { formatDuration } from "@/lib/utils";

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
}, ref) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scrubRef = useRef<HTMLDivElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isScrubbing, setIsScrubbing] = useState(false);

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

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="space-y-3">
      <div className="relative overflow-hidden rounded-xl bg-black">
        {videoUrl && (
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full"
            style={{ aspectRatio: "16/9" }}
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

        {/* Playback overlay when paused */}
        {!isPlaying && videoUrl && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30">
            <button
              onClick={handlePlay}
              className="flex h-16 w-16 items-center justify-center rounded-full bg-white/90 text-neutral-900 shadow-lg transition-transform hover:scale-110"
            >
              <Play className="ml-1 h-7 w-7" />
            </button>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" />

      {/* Controls */}
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
    </div>
  );
});
