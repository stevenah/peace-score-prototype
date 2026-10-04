"use client";

import { Camera } from "lucide-react";
import { FILMSTRIP_VISIBLE_SLOTS } from "@/lib/live/config";
import { STATION_LABELS, STATION_SHORT } from "@/lib/live/stations";
import type { Capture } from "@/hooks/useCaptureFilmstrip";
import type { PeaceScore } from "@/lib/types";

const BAR: Record<PeaceScore, string> = {
  0: "bg-peace-0",
  1: "bg-peace-1",
  2: "bg-peace-2",
  3: "bg-peace-3",
};

/**
 * Newest at the top, so a new capture pushes older ones down and out rather than
 * shifting the position the eye is already reading.
 *
 * Empty slots occupy the exact final geometry, so the first real capture replaces
 * a slot instead of expanding the strip — no layout shift mid-procedure.
 *
 * No detection boxes are drawn: there is no detector, and inventing boxes on
 * specific frames would be fabricating findings. Each thumb instead carries its
 * PEACE colour and its timestamp, which are real — plus, for a frame on which an
 * ESGE station became observed, that station's chip.
 */
export function CaptureFilmstrip({ captures }: { captures: readonly Capture[] }) {
  const slots = Array.from({ length: FILMSTRIP_VISIBLE_SLOTS }, (_, i) =>
    captures[i] ?? null,
  );

  return (
    <ol className="flex gap-2 xl:flex-col" aria-label="Recent captures">
      {slots.map((cap, i) =>
        cap ? (
          <li
            key={cap.id}
            className="relative aspect-video w-full min-w-0 flex-1 bg-ws-void animate-in fade-in slide-in-from-top-2 duration-200 ease-out motion-reduce:animate-none motion-reduce:opacity-100 xl:flex-none"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={cap.url}
              alt=""
              className="h-full w-full object-contain"
              aria-hidden
            />
            <span
              className="absolute left-1 top-1 font-mono text-[10px] font-light leading-none text-white/70"
              aria-hidden
            >
              {formatClock(cap.videoTime)}
            </span>
            {cap.station && (
              <>
                <span
                  className="absolute right-1 top-1 max-w-[70%] truncate bg-black/70 px-1 py-[2px] text-[10px] font-medium leading-none text-peace-3"
                  aria-hidden
                >
                  {STATION_SHORT[cap.station]}
                </span>
                <span className="sr-only">
                  {`${STATION_LABELS[cap.station]} observed at ${formatClock(cap.videoTime)}`}
                </span>
              </>
            )}
            <span
              className={`absolute bottom-0 left-0 h-[3px] w-full ${BAR[cap.score]}`}
              aria-hidden
            />
          </li>
        ) : (
          <li
            key={`empty-${i}`}
            className="relative aspect-video w-full min-w-0 flex-1 border border-dashed border-white/10 bg-white/[0.02] xl:flex-none"
          >
            <Camera
              className="absolute inset-0 m-auto size-4 text-white/15"
              aria-hidden
            />
          </li>
        ),
      )}
    </ol>
  );
}

function formatClock(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}
