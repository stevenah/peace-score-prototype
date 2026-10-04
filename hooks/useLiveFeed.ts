"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { IN_FLIGHT_TIMEOUT_MS } from "@/lib/live/config";
import type { LiveFrameResult, LiveMessage } from "@/lib/types";

interface UseLiveFeedOptions {
  enabled: boolean;
  /**
   * Called once per analysed frame, with the *video* time at which that frame was
   * captured and the ticket sendFrame returned for it. Prefer this over reading
   * `results`: LiveFrameResult.timestamp is a Unix epoch from the server, not a
   * media time. Key anything captured alongside the frame by `ticket`, never by
   * `result.frame_index` — the two differ once any frame fails to go out.
   */
  onResult?: (result: LiveFrameResult, videoTime: number, ticket: number) => void;
}

interface PendingFrame {
  ticket: number;
  videoTime: number;
  sentAt: number;
}

export function useLiveFeed({ enabled, onResult }: UseLiveFeedOptions) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [latestResult, setLatestResult] = useState<LiveFrameResult | null>(null);
  const [results, setResults] = useState<LiveFrameResult[]>([]);
  const [framesProcessed, setFramesProcessed] = useState(0);
  const [inFlightFrames, setInFlightFrames] = useState(0);
  const [lastResultAtMs, setLastResultAtMs] = useState<number | null>(null);
  const [frameErrors, setFrameErrors] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  /**
   * server frame_index -> pending frame. The client predicts the server's
   * frame_index from its own count of frames actually put on the wire: both start
   * at 0 per connection and increment once per frame in the backend's sequential
   * receive loop.
   *
   * Keying on the index rather than using a plain FIFO matters — a single dropped
   * reply would otherwise shift every later pairing by one frame, permanently.
   */
  const pendingRef = useRef<Map<number, PendingFrame>>(new Map());
  /**
   * Frames actually sent on the current connection == the next server
   * frame_index. Advanced only after ws.send() succeeds: a frame whose
   * arrayBuffer() rejects never reaches the server, and counting it would leave
   * every later pairing off by one for the rest of the procedure.
   */
  const sentCountRef = useRef(0);
  /**
   * Client tickets, handed out synchronously by sendFrame. Monotonic for the
   * hook's lifetime (never reused across connections), so rings keyed by ticket
   * cannot collide.
   */
  const ticketCounterRef = useRef(0);
  /** Accepted by sendFrame but not yet on the wire; counts toward in-flight. */
  const queuedRef = useRef(0);
  /** Serialises sends so the client's counter matches the server's receive order. */
  const sendChainRef = useRef<Promise<unknown>>(Promise.resolve());
  const onResultRef = useRef(onResult);
  useEffect(() => {
    onResultRef.current = onResult;
  }, [onResult]);

  const syncInFlight = useCallback(() => {
    setInFlightFrames(pendingRef.current.size + queuedRef.current);
  }, []);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    async function connect() {
      setIsConnecting(true);
      setConnectionError(null);

      // Fetch the WS URL from the server (uses runtime env, see app/api/live).
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
      // The backend restarts frame_index at 0 for each connection.
      sentCountRef.current = 0;
      pendingRef.current.clear();
      syncInFlight();

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
        let message: LiveMessage;
        try {
          message = JSON.parse(event.data);
        } catch {
          return;
        }

        if (message.type === "error") {
          // A per-frame failure, not a connection failure. Retire the pending
          // ticket so the in-flight count decays, and keep going — latching this
          // as a connection error would strand the UI on "analysis unavailable"
          // for the rest of the procedure after one unreadable frame.
          if (typeof message.frame_index === "number") {
            pendingRef.current.delete(message.frame_index);
            syncInFlight();
          }
          setFrameErrors((c) => c + 1);
          setLastResultAtMs(Date.now());
          return;
        }
        const parsed: LiveFrameResult = message;

        const idx = parsed.frame_index;
        const pending = pendingRef.current.get(idx);
        pendingRef.current.delete(idx);
        // Anything older than what just arrived is never coming back.
        for (const key of pendingRef.current.keys()) {
          if (key < idx) pendingRef.current.delete(key);
        }
        syncInFlight();

        setLatestResult(parsed);
        setResults((prev) => [...prev, parsed]);
        setFramesProcessed((c) => c + 1);
        setLastResultAtMs(Date.now());

        if (pending !== undefined) {
          onResultRef.current?.(parsed, pending.videoTime, pending.ticket);
        }
      };
    }

    connect();

    const pending = pendingRef.current;
    return () => {
      cancelled = true;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      pending.clear();
      setIsConnected(false);
      setIsConnecting(false);
      setInFlightFrames(0);
    };
  }, [enabled, syncInFlight]);

  // Write off frames whose replies never arrived, so the in-flight count decays
  // and backpressure cannot latch the video paused indefinitely.
  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => {
      const now = Date.now();
      let expired = false;
      for (const [key, entry] of pendingRef.current) {
        if (now - entry.sentAt > IN_FLIGHT_TIMEOUT_MS) {
          pendingRef.current.delete(key);
          expired = true;
        }
      }
      if (expired) syncInFlight();
    }, 1000);
    return () => clearInterval(id);
  }, [enabled, syncInFlight]);

  /**
   * Queues a frame for analysis. Returns its ticket — the value onResult will
   * report for it — or null when there is no open connection to send on.
   */
  const sendFrame = useCallback(
    (blob: Blob, videoTime = 0): number | null => {
      const ws = wsRef.current;
      if (ws?.readyState !== WebSocket.OPEN) return null;

      const ticket = ticketCounterRef.current++;
      queuedRef.current += 1;
      syncInFlight();

      // Chained so that concurrent arrayBuffer() resolutions cannot reorder
      // sends — the index scheme depends on the server receiving them in order.
      sendChainRef.current = sendChainRef.current
        .then(() => blob.arrayBuffer())
        .then((buffer) => {
          // Only on the connection the frame was captured for: a new socket
          // numbers its frames from 0 again.
          if (wsRef.current !== ws || ws.readyState !== WebSocket.OPEN) return;
          ws.send(buffer);
          // After send(), not before: if it throws, nothing was counted.
          const index = sentCountRef.current++;
          pendingRef.current.set(index, { ticket, videoTime, sentAt: Date.now() });
        })
        .catch(() => {
          // Unreadable blob or a failed send: the frame never reached the
          // server and consumed no index, so later pairings stay aligned.
        })
        .finally(() => {
          queuedRef.current = Math.max(0, queuedRef.current - 1);
          syncInFlight();
        });

      return ticket;
    },
    [syncInFlight],
  );

  const reset = useCallback(() => {
    setResults([]);
    setLatestResult(null);
    setFramesProcessed(0);
    setInFlightFrames(0);
    setConnectionError(null);
    setLastResultAtMs(null);
    setFrameErrors(0);
    pendingRef.current.clear();
    sentCountRef.current = 0;
  }, []);

  return {
    isConnected,
    isConnecting,
    connectionError,
    latestResult,
    results,
    framesProcessed,
    inFlightFrames,
    lastResultAtMs,
    frameErrors,
    sendFrame,
    reset,
  };
}
