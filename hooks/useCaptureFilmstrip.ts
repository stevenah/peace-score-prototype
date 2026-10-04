"use client";

import { useCallback, useRef, useState } from "react";
import {
  FILMSTRIP_MAX_RETAINED,
  THUMB_RING_SIZE,
} from "@/lib/live/config";
import type { CaptureReason } from "@/lib/live/capture-triggers";
import type { AnatomicalRegion, PeaceScore, StationKey } from "@/lib/types";

export interface Capture {
  readonly id: string;
  readonly videoTime: number;
  /** Small JPEG data URL. */
  readonly url: string;
  readonly reason: CaptureReason;
  readonly region: AnatomicalRegion | null;
  readonly score: PeaceScore;
  /** Set for "station-observed" captures: the station this frame showed. */
  readonly station?: StationKey;
}

interface PendingThumb {
  url: string;
  videoTime: number;
}

/**
 * Keeps a short strip of notable frames.
 *
 * The frame belonging to a result is already gone by the time that result comes
 * back from the server (~200ms later, video still playing), so thumbnails are
 * parked in a small ring at capture time — keyed by the send ticket from
 * useLiveFeed — and promoted only once a trigger fires.
 *
 * Thumbnails are small JPEG data URLs with a hard cap, so there is no object-URL
 * revocation lifecycle to get wrong and nothing to leak across procedures.
 */
export function useCaptureFilmstrip(maxRetained = FILMSTRIP_MAX_RETAINED) {
  const ringRef = useRef<Map<number, PendingThumb>>(new Map());
  const [captures, setCaptures] = useState<Capture[]>([]);

  const offerThumb = useCallback(
    (ticket: number, url: string, videoTime: number) => {
      if (!url) return;
      const ring = ringRef.current;
      ring.set(ticket, { url, videoTime });
      while (ring.size > THUMB_RING_SIZE) {
        const oldest = ring.keys().next().value;
        if (oldest === undefined) break;
        ring.delete(oldest);
      }
    },
    [],
  );

  const commit = useCallback(
    (
      ticket: number,
      meta: {
        reason: CaptureReason;
        region: AnatomicalRegion | null;
        score: PeaceScore;
        station?: StationKey | null;
      },
    ) => {
      const pending = ringRef.current.get(ticket);
      if (!pending) return;
      ringRef.current.delete(ticket);
      setCaptures((prev) => {
        const next: Capture[] = [
          {
            id: `${ticket}`,
            videoTime: pending.videoTime,
            url: pending.url,
            reason: meta.reason,
            region: meta.region,
            score: meta.score,
            ...(meta.station ? { station: meta.station } : {}),
          },
          ...prev,
        ];
        return next.length > maxRetained ? next.slice(0, maxRetained) : next;
      });
    },
    [maxRetained],
  );

  const clear = useCallback(() => {
    ringRef.current.clear();
    setCaptures([]);
  }, []);

  return { offerThumb, commit, captures, clear };
}
