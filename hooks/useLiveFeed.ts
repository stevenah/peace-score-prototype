"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { LiveFrameResult } from "@/lib/types";

interface UseLiveFeedOptions {
  enabled: boolean;
  wsUrl?: string;
}

export function useLiveFeed({ enabled, wsUrl }: UseLiveFeedOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [latestResult, setLatestResult] = useState<LiveFrameResult | null>(null);
  const [results, setResults] = useState<LiveFrameResult[]>([]);
  const [framesProcessed, setFramesProcessed] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsConnected(false);
      return;
    }

    const protocol = typeof window !== "undefined"
      ? (window.location.protocol === "https:" ? "wss:" : "ws:")
      : "ws:";
    const host = typeof window !== "undefined" ? window.location.host : "localhost:3000";
    const url = wsUrl || `${protocol}//${host}/api/live`;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);

    ws.onmessage = (event) => {
      try {
        const result: LiveFrameResult = JSON.parse(event.data);
        setLatestResult(result);
        setResults((prev) => [...prev, result]);
        setFramesProcessed((c) => c + 1);
      } catch {
        // Ignore parse errors
      }
    };

    return () => {
      ws.close();
    };
  }, [enabled, wsUrl]);

  const sendFrame = useCallback((blob: Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then((buffer) => {
        wsRef.current?.send(buffer);
      });
    }
  }, []);

  const reset = useCallback(() => {
    setResults([]);
    setLatestResult(null);
    setFramesProcessed(0);
  }, []);

  return {
    isConnected,
    latestResult,
    results,
    framesProcessed,
    sendFrame,
    reset,
  };
}
