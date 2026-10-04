"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { STATION_GALLERY_RING } from "@/lib/live/config";
import type { FrameSample, StationsSlice } from "@/lib/live/procedure-types";
import { STATION_INDEX } from "@/lib/live/stations";
import type { StationKey } from "@/lib/types";

export interface StationFrame {
  /** Object URL owned by the hook; revoked when replaced and on unmount. */
  readonly url: string;
  readonly videoTime: number;
  readonly ticket: number;
}

export type StationFrames = Readonly<Partial<Record<StationKey, StationFrame>>>;

interface RingEntry {
  blob: Blob;
  videoTime: number;
}

/**
 * Which station, if any, this frame should become the gallery view of.
 *
 * Driven by the LEVEL signal rather than by events, so a lost `best_frame`
 * message costs at most an upgrade, never a station's only frame:
 *  - the frame's top class is k, and k is observed (sticky store status), and
 *  - the backend flagged it a new best view of k, or k has no frame yet and this
 *    one is a confident ("ok") frame.
 */
export function stationToPromote(
  sample: FrameSample,
  stations: Pick<StationsSlice, "visible" | "status">,
  hasFrame: (station: StationKey) => boolean,
): StationKey | null {
  if (!stations.visible) return null;
  const k = sample.landmarkTop;
  if (!k) return null;
  if (stations.status[STATION_INDEX[k]] !== "observed") return null;
  if (sample.stationBestEvents?.includes(k)) return k;
  if (!hasFrame(k) && sample.landmarkStatus === "ok") return k;
  return null;
}

/**
 * Best full-resolution frame per ESGE station, for the end-of-procedure gallery.
 *
 * Deliberately separate from the filmstrip: the filmstrip's 240px q0.5 thumbnails
 * are not documentation grade, and its commit() deletes ring entries, whereas one
 * frame can be both a capture and a station's best view.
 *
 * Like the filmstrip, a frame is gone by the time its result arrives, so analysis
 * blobs are parked in a small ring keyed by send ticket at capture time and
 * copied out (not moved) on promotion. Memory is bounded at ring + 10 frames.
 */
export function useStationGallery(ringSize = STATION_GALLERY_RING) {
  const ringRef = useRef<Map<number, RingEntry>>(new Map());
  const urlsRef = useRef<Map<StationKey, string>>(new Map());
  const [frames, setFrames] = useState<StationFrames>({});

  const offer = useCallback(
    (ticket: number, blob: Blob, videoTime: number) => {
      const ring = ringRef.current;
      ring.delete(ticket);
      ring.set(ticket, { blob, videoTime });
      while (ring.size > ringSize) {
        const oldest = ring.keys().next().value;
        if (oldest === undefined) break;
        ring.delete(oldest);
      }
    },
    [ringSize],
  );

  const has = useCallback(
    (station: StationKey) => urlsRef.current.has(station),
    [],
  );

  /** Copies ring entry `ticket` into `station`'s slot. False if it has left the ring. */
  const promote = useCallback((ticket: number, station: StationKey): boolean => {
    const entry = ringRef.current.get(ticket);
    if (!entry) return false;
    const url = URL.createObjectURL(entry.blob);
    const previous = urlsRef.current.get(station);
    urlsRef.current.set(station, url);
    if (previous) URL.revokeObjectURL(previous);
    setFrames((prev) => ({
      ...prev,
      [station]: { url, videoTime: entry.videoTime, ticket },
    }));
    return true;
  }, []);

  /** Drops parked frames (e.g. on reconnect); promoted views are kept. */
  const clearRing = useCallback(() => {
    ringRef.current.clear();
  }, []);

  const reset = useCallback(() => {
    ringRef.current.clear();
    for (const url of urlsRef.current.values()) URL.revokeObjectURL(url);
    urlsRef.current.clear();
    setFrames({});
  }, []);

  useEffect(() => {
    const urls = urlsRef.current;
    const ring = ringRef.current;
    return () => {
      for (const url of urls.values()) URL.revokeObjectURL(url);
      urls.clear();
      ring.clear();
    };
  }, []);

  return { offer, promote, has, frames, clearRing, reset };
}
