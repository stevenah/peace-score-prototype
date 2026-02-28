"use client";

import { useEffect, useRef, useState } from "react";
import { Camera, CameraOff } from "lucide-react";
import { Button } from "@/components/ui/Button";

interface LiveFeedViewerProps {
  isActive: boolean;
  onFrameCapture?: (blob: Blob) => void;
  captureIntervalMs?: number;
}

export function LiveFeedViewer({
  isActive,
  onFrameCapture,
  captureIntervalMs = 1000,
}: LiveFeedViewerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isActive) {
      stopStream();
      return;
    }

    startStream();

    return () => {
      stopStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive]);

  useEffect(() => {
    if (!isActive || !onFrameCapture || !hasPermission) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      return;
    }

    intervalRef.current = setInterval(() => {
      captureFrame();
    }, captureIntervalMs);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, hasPermission, captureIntervalMs, onFrameCapture]);

  async function startStream() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setHasPermission(true);
      setError(null);
    } catch {
      setHasPermission(false);
      setError("Camera access denied. Please allow camera permissions.");
    }
  }

  function stopStream() {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  function captureFrame() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !onFrameCapture) return;

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
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border p-12">
        <CameraOff className="mb-4 h-10 w-10 text-muted-foreground/60" />
        <p className="mb-2 text-sm text-red-600 dark:text-red-400">{error}</p>
        <Button variant="secondary" size="sm" onClick={() => startStream()}>
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-xl bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full"
        style={{ aspectRatio: "4/3" }}
      />
      <canvas ref={canvasRef} className="hidden" />
      {!hasPermission && isActive && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <div className="flex flex-col items-center">
            <Camera className="mb-2 h-8 w-8 animate-pulse text-white" />
            <p className="text-sm text-white">Requesting camera access...</p>
          </div>
        </div>
      )}
    </div>
  );
}
