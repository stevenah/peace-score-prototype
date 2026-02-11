"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { Play, Pause, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Progress } from "@/components/ui/Progress";
import { formatDuration } from "@/lib/utils";

interface VideoStreamPlayerProps {
  file: File;
  isAnalyzing: boolean;
  onFrameCapture: (blob: Blob) => void;
  onPlayStateChange?: (playing: boolean) => void;
  onVideoEnd?: () => void;
  onVideoReady?: (duration: number) => void;
  captureIntervalMs?: number;
}

export function VideoStreamPlayer({
  file,
  isAnalyzing,
  onFrameCapture,
  onPlayStateChange,
  onVideoEnd,
  onVideoReady,
  captureIntervalMs = 500,
}: VideoStreamPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

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
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
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

        <div className="flex-1">
          <Progress value={progress} color={isAnalyzing ? "#3b82f6" : "#9ca3af"} />
        </div>

        <span className="text-xs tabular-nums text-neutral-500">
          {formatDuration(currentTime)} / {formatDuration(duration)}
        </span>
      </div>
    </div>
  );
}
