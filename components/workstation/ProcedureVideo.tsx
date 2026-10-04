"use client";

import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import Hls from "hls.js";
import { Pause, Play, RotateCcw } from "lucide-react";
import { DetectionOverlay } from "@/components/video/DetectionOverlay";
import { THUMB_QUALITY, THUMB_WIDTH } from "@/lib/live/config";
import type { Detection } from "@/lib/types";

export interface ProcedureVideoHandle {
  seekTo(time: number): void;
  pause(): void;
  play(): void;
}

interface ProcedureVideoProps {
  source: File | string;
  isLiveStream: boolean;
  isAnalyzing: boolean;
  /** Pauses playback to let the backend catch up. */
  syncPause: boolean;
  captureIntervalMs: number;
  detections?: readonly Detection[];
  onFrameCapture: (analysis: Blob, thumbnail: string, videoTime: number) => void;
  onVideoEnd?: () => void;
  onVideoReady?: (duration: number) => void;
  onPlayStateChange?: (playing: boolean) => void;
  onTimeUpdate?: (time: number) => void;
  onCorsError?: () => void;
}

export const ProcedureVideo = forwardRef<
  ProcedureVideoHandle,
  ProcedureVideoProps
>(function ProcedureVideo(
  {
    source,
    isLiveStream,
    isAnalyzing,
    syncPause,
    captureIntervalMs,
    detections,
    onFrameCapture,
    onVideoEnd,
    onVideoReady,
    onTimeUpdate,
    onPlayStateChange,
    onCorsError,
  },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const fitRef = useRef<HTMLDivElement>(null);
  const analysisCanvasRef = useRef<HTMLCanvasElement>(null);
  const thumbCanvasRef = useRef<HTMLCanvasElement>(null);
  const scrubRef = useRef<HTMLDivElement>(null);
  const hlsRef = useRef<Hls | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastBucketRef = useRef<number | null>(null);
  const syncPausedRef = useRef(false);
  const captureRef = useRef(onFrameCapture);
  useEffect(() => {
    captureRef.current = onFrameCapture;
  }, [onFrameCapture]);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useImperativeHandle(
    ref,
    () => ({
      seekTo(time: number) {
        const v = videoRef.current;
        if (!v) return;
        v.currentTime = time;
        setCurrentTime(time);
        onTimeUpdate?.(time);
      },
      pause() {
        videoRef.current?.pause();
      },
      play() {
        void videoRef.current?.play().catch(() => {});
      },
    }),
    [onTimeUpdate],
  );

  // Resolve the source to something playable (File blob, HLS, or direct URL).
  useEffect(() => {
    const video = videoRef.current;
    lastBucketRef.current = null;

    if (hlsRef.current) {
      hlsRef.current.destroy();
      hlsRef.current = null;
    }

    if (source instanceof File) {
      const url = URL.createObjectURL(source);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    }

    const url = source;
    if (url.includes(".m3u8") && Hls.isSupported() && video) {
      const hls = new Hls({ enableWorker: true, lowLatencyMode: true });
      hlsRef.current = hls;
      hls.loadSource(url);
      hls.attachMedia(video);
      hls.on(Hls.Events.MANIFEST_PARSED, () => setVideoUrl(url));
      return () => {
        hls.destroy();
        hlsRef.current = null;
      };
    }

    setVideoUrl(url);
  }, [source]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current;
    const analysisCanvas = analysisCanvasRef.current;
    const thumbCanvas = thumbCanvasRef.current;
    if (!video || !analysisCanvas || !thumbCanvas) return;
    if (video.paused || video.ended) return;

    // Only guard against double-capture inside one interval tick. Remembering
    // every bucket ever seen would make restart and seek-back emit no frames.
    const bucket = Math.floor((video.currentTime * 1000) / captureIntervalMs);
    if (lastBucketRef.current === bucket) return;
    lastBucketRef.current = bucket;

    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    analysisCanvas.width = w;
    analysisCanvas.height = h;
    const ctx = analysisCanvas.getContext("2d");
    if (!ctx) return;

    try {
      ctx.drawImage(video, 0, 0);
    } catch {
      // SecurityError: cross-origin stream without CORS headers taints the canvas.
      onCorsError?.();
      return;
    }

    // A second, much smaller draw for the filmstrip. Retaining the full-size
    // analysis frames instead would be tens of MB over a long procedure.
    const scale = THUMB_WIDTH / w;
    thumbCanvas.width = THUMB_WIDTH;
    thumbCanvas.height = Math.max(1, Math.round(h * scale));
    const tctx = thumbCanvas.getContext("2d");
    let thumbnail = "";
    if (tctx) {
      tctx.drawImage(video, 0, 0, thumbCanvas.width, thumbCanvas.height);
      try {
        thumbnail = thumbCanvas.toDataURL("image/jpeg", THUMB_QUALITY);
      } catch {
        thumbnail = "";
      }
    }

    const captureTime = video.currentTime;
    analysisCanvas.toBlob(
      (blob) => {
        if (blob) captureRef.current(blob, thumbnail, captureTime);
      },
      "image/jpeg",
      0.8,
    );
  }, [captureIntervalMs, onCorsError]);

  useEffect(() => {
    if (!isPlaying || !isAnalyzing) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    intervalRef.current = setInterval(captureFrame, captureIntervalMs);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, isAnalyzing, captureIntervalMs, captureFrame]);

  // Backpressure: hold the video while the backend catches up.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !isAnalyzing) return;

    if (syncPause && !video.paused) {
      syncPausedRef.current = true;
      video.pause();
    } else if (!syncPause && syncPausedRef.current) {
      syncPausedRef.current = false;
      // play() rejects if the element was detached or interrupted. Without this
      // catch the flag is already cleared and playback would never resume.
      void video.play().catch(() => {
        syncPausedRef.current = true;
      });
    }
  }, [syncPause, isAnalyzing]);

  const handlePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.ended) video.currentTime = 0;
    void video.play().catch(() => {});
  }, []);

  const handlePause = useCallback(() => videoRef.current?.pause(), []);

  const handleRestart = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    lastBucketRef.current = null;
    video.currentTime = 0;
    void video.play().catch(() => {});
  }, []);

  const seekFraction = useCallback(
    (fraction: number) => {
      const video = videoRef.current;
      if (!video || duration === 0) return;
      const t = Math.max(0, Math.min(1, fraction)) * duration;
      video.currentTime = t;
      setCurrentTime(t);
      onTimeUpdate?.(t);
    },
    [duration, onTimeUpdate],
  );

  const handleScrub = useCallback(
    (e: React.MouseEvent) => {
      const bar = scrubRef.current;
      if (!bar) return;
      e.preventDefault();
      const apply = (clientX: number) => {
        const rect = bar.getBoundingClientRect();
        seekFraction((clientX - rect.left) / rect.width);
      };
      apply(e.clientX);
      const move = (ev: MouseEvent) => apply(ev.clientX);
      const up = () => {
        document.removeEventListener("mousemove", move);
        document.removeEventListener("mouseup", up);
      };
      document.addEventListener("mousemove", move);
      document.addEventListener("mouseup", up);
    },
    [seekFraction],
  );

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="grid h-full min-h-0 min-w-0 grid-rows-[minmax(0,1fr)_auto]">
      <section
        className="ws-stage relative grid min-h-0 min-w-0 place-items-center bg-ws-void"
        role="region"
        aria-label="Endoscopy video"
      >
        <div ref={fitRef} className="ws-fit relative">
          {videoUrl && (
            <video
              ref={videoRef}
              src={hlsRef.current ? undefined : videoUrl}
              crossOrigin={typeof source === "string" ? "anonymous" : undefined}
              className="absolute inset-0 h-full w-full bg-ws-void object-contain"
              playsInline
              muted
              aria-label="Live endoscopy feed"
              onLoadedMetadata={(e) => {
                const v = e.currentTarget;
                if (v.videoWidth && v.videoHeight) {
                  // Unitless ratio: calc() cannot consume "16/9".
                  fitRef.current?.style.setProperty(
                    "--ar",
                    String(v.videoWidth / v.videoHeight),
                  );
                }
                setDuration(v.duration);
                onVideoReady?.(v.duration);
              }}
              onTimeUpdate={(e) => {
                const t = e.currentTarget.currentTime;
                setCurrentTime(t);
                onTimeUpdate?.(t);
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
          <DetectionOverlay detections={detections} />
        </div>

        {!isPlaying && videoUrl && (
          <button
            onClick={handlePlay}
            className="absolute inset-0 grid place-items-center bg-black/30 transition-colors hover:bg-black/20"
            aria-label="Play"
          >
            <span className="grid size-16 place-items-center rounded-full bg-white/90 text-neutral-900">
              <Play className="ml-1 size-7" />
            </span>
          </button>
        )}
      </section>

      {/* Control strip. Deliberately below the video, not floating over it. */}
      <div className="flex h-9 items-center gap-3 px-3">
        <button
          onClick={isPlaying ? handlePause : handlePlay}
          className="grid size-6 shrink-0 place-items-center text-ws-label transition-colors hover:text-ws-fg"
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
        </button>

        {isLiveStream ? (
          <div className="flex items-center gap-2">
            <span className="size-1.5 rounded-full bg-peace-3" aria-hidden />
            <span className="ws-micro uppercase text-peace-3">Live</span>
          </div>
        ) : (
          <>
            <button
              onClick={handleRestart}
              className="grid size-6 shrink-0 place-items-center text-ws-label transition-colors hover:text-ws-fg"
              aria-label="Restart"
            >
              <RotateCcw className="size-3.5" />
            </button>
            <div
              ref={scrubRef}
              onMouseDown={handleScrub}
              className="group relative h-6 flex-1 cursor-pointer"
            >
              <div className="absolute top-1/2 h-[3px] w-full -translate-y-1/2 bg-white/12">
                <div
                  className="h-full bg-white/55"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
            <span className="ws-micro shrink-0 tabular-nums">
              {formatClock(currentTime)} / {formatClock(duration)}
            </span>
          </>
        )}
      </div>

      <canvas ref={analysisCanvasRef} className="hidden" />
      <canvas ref={thumbCanvasRef} className="hidden" />
    </div>
  );
});

function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
