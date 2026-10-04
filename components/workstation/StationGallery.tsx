"use client";

import { X } from "lucide-react";
import type { StationFrames } from "@/hooks/useStationGallery";
import { selectStations, useProcedureSlice } from "@/hooks/useProcedureMetrics";
import type { StationsSlice } from "@/lib/live/procedure-types";
import { STATION_LABELS, STATION_ORDER, STATION_SHORT } from "@/lib/live/stations";
import {
  SimulatedBadge,
  STATIONS_DISCLAIMER,
  formatStationClock,
  isSimulatedModel,
  stationRowState,
  type StationRowState,
} from "./StationChecklist";

const STATE_TEXT: Record<StationRowState, string> = {
  unseen: "Not observed",
  candidate: "Not observed",
  observed: "Observed",
  confirmed: "Confirmed",
  rejected: "Rejected",
};

interface StationGalleryProps {
  stations: StationsSlice;
  frames: StationFrames;
  onClose: () => void;
  className?: string;
}

/**
 * End-of-procedure review: the best kept frame per ESGE station, 2 x 5 in ESGE
 * order.
 *
 * An overlay over the video rather than a panel in the right rail, which is far
 * too narrow for ten legible frames. A station with no frame says so plainly —
 * "No frame kept" is not the same claim as "not observed", and the tile's state
 * line says which applies.
 */
export function StationGallery({
  stations,
  frames,
  onClose,
  className = "absolute inset-0",
}: StationGalleryProps) {
  return (
    <section
      aria-label="Station gallery"
      className={`${className} z-20 flex min-h-0 flex-col gap-3 bg-ws-void/92 p-4 animate-in fade-in duration-200 ease-out motion-reduce:animate-none`}
    >
      <header className="flex shrink-0 items-center gap-3">
        <h2 className="text-[15px] font-medium leading-none text-ws-fg-2">
          Station views
        </h2>
        <span className="ws-micro tabular-nums">
          {stations.observedCount} / {stations.total} observed
        </span>
        {isSimulatedModel(stations.modelVersion) && <SimulatedBadge />}
        <button
          type="button"
          onClick={onClose}
          className="ml-auto grid size-6 place-items-center text-ws-label transition-colors hover:text-ws-fg"
          aria-label="Close station gallery"
        >
          <X className="size-4" />
        </button>
      </header>

      <ol className="grid min-h-0 flex-1 grid-cols-5 grid-rows-2 gap-2">
        {STATION_ORDER.map((key, i) => {
          const frame = frames[key];
          const state = stationRowState(stations, i);
          const observed = stations.observed[i];
          return (
            <li
              key={key}
              className="flex min-h-0 min-w-0 flex-col bg-ws-surface"
              aria-label={`${i + 1}. ${STATION_LABELS[key]}`}
            >
              <div className="relative min-h-0 flex-1">
                {frame ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={frame.url}
                    alt={`${STATION_LABELS[key]}, frame at ${formatStationClock(frame.videoTime)}`}
                    className="absolute inset-0 h-full w-full object-contain"
                  />
                ) : (
                  <div className="absolute inset-0 grid place-items-center border border-dashed border-white/10">
                    <span className="ws-micro text-ws-faint">No frame kept</span>
                  </div>
                )}
              </div>
              <div className="flex min-w-0 items-center gap-1.5 px-2 py-1.5">
                <span className="font-mono text-[11px] tabular-nums text-ws-faint">
                  {i + 1}
                </span>
                <span className="truncate text-[12px] text-ws-fg-2">
                  {STATION_SHORT[key]}
                </span>
                <span
                  className={`ml-auto shrink-0 text-[10px] font-medium uppercase tracking-[0.06em] ${
                    observed ? "text-peace-3" : "text-ws-coach"
                  }`}
                >
                  {STATE_TEXT[state]}
                </span>
              </div>
            </li>
          );
        })}
      </ol>

      <p className="shrink-0 text-[11px] leading-snug text-ws-faint">
        {STATIONS_DISCLAIMER}
      </p>
    </section>
  );
}

/**
 * Store-connected wrapper. Shows only when asked to AND stations are visible —
 * never in shadow mode or with the feature off.
 */
export function StationGalleryOverlay({
  open,
  frames,
  onClose,
  className,
}: {
  open: boolean;
  frames: StationFrames;
  onClose: () => void;
  className?: string;
}) {
  const stations = useProcedureSlice(selectStations);
  if (!open || !stations.visible) return null;
  return (
    <StationGallery
      stations={stations}
      frames={frames}
      onClose={onClose}
      className={className}
    />
  );
}
