"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { LiveFrameResult } from "@/lib/types";

interface UseLiveFeedOptions {
  enabled: boolean;
}

export function useLiveFeed({ enabled }: UseLiveFeedOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [latestResult, setLatestResult] = useState<LiveFrameResult | null>(null);
  const [results, setResults] = useState<LiveFrameResult[]>([]);
  const [framesProcessed, setFramesProcessed] = useState(0);
  const [inFlightFrames, setInFlightFrames] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    async function connect() {
      setIsConnecting(true);
      setConnectionError(null);

      // Fetch the WS URL from the server (uses runtime env var ML_BACKEND_URL)
      let wsUrl: string;
      try {
        const res = await fetch("/api/live");
        const data = await res.json();
        wsUrl = data.ws_url;
      } catch {
        wsUrl = "ws://localhost:8000/api/v1/ws/live";
      }

      if (cancelled) return;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelled) return;
        setIsConnected(true);
        setIsConnecting(false);
        setConnectionError(null);
      };

      ws.onclose = () => {
        if (cancelled) return;
        setIsConnected(false);
        setIsConnecting(false);
      };

      ws.onerror = () => {
        if (cancelled) return;
        setIsConnected(false);
        setIsConnecting(false);
        setConnectionError("Failed to connect to analysis server");
      };

      ws.onmessage = (event) => {
        try {
          const result: LiveFrameResult = JSON.parse(event.data);
          setLatestResult(result);
          setResults((prev) => [...prev, result]);
          setFramesProcessed((c) => c + 1);
          setInFlightFrames((c) => Math.max(0, c - 1));
        } catch {
          // Ignore parse errors
        }
      };
    }

    connect();

    return () => {
      cancelled = true;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setIsConnected(false);
      setIsConnecting(false);
    };
  }, [enabled]);

  const sendFrame = useCallback((blob: Blob): boolean => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      setInFlightFrames((c) => c + 1);
      blob.arrayBuffer().then((buffer) => {
        wsRef.current?.send(buffer);
      });
      return true;
    }
    return false;
  }, []);

  const reset = useCallback(() => {
    setResults([]);
    setLatestResult(null);
    setFramesProcessed(0);
    setInFlightFrames(0);
    setConnectionError(null);
  }, []);

  return {
    isConnected,
    isConnecting,
    connectionError,
    latestResult,
    results,
    framesProcessed,
    inFlightFrames,
    sendFrame,
    reset,
  };
}
