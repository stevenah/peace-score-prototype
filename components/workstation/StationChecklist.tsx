"use client";

import { Component, useId, type ErrorInfo, type ReactNode } from "react";
import { REGION_LABELS } from "@/lib/constants";
import type { ManualMark, StationsSlice } from "@/lib/live/procedure-types";
import {
  STATION_GROUPS,
  STATION_INDEX,
  STATION_LABELS,
  STATION_SHORT,
} from "@/lib/live/stations";
import type { StationKey } from "@/lib/types";

/** What one checklist row shows. Manual marks win over the model. */
export type StationRowState =
  | "unseen"
  | "candidate"
  | "observed"
  | "confirmed"
  | "rejected";

export function stationRowState(
  stations: Pick<StationsSlice, "status" | "manual">,
  index: number,
): StationRowState {
  const manual = stations.manual[index];
  if (manual === "confirmed") return "confirmed";
  if (manual === "rejected") return "rejected";
  return stations.status[index] ?? "unseen";
}

/** One press cycles: model -> confirmed -> rejected -> model. */
export function nextManualMark(mark: ManualMark): ManualMark {
  if (mark === null) return "confirmed";
  if (mark === "confirmed") return "rejected";
  return null;
}

/**
 * Station UI is an add-on to the PEACE instrument; a render fault in it must
 * take down only itself, never the rail or the video around it.
 */
export class StationsErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn("Station UI failed; hiding it", error, info.componentStack);
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

/** Mock models must never pass for clinical output. */
export function isSimulatedModel(version: string | null): boolean {
  return version !== null && version.startsWith("mock");
}

export const STATIONS_DISCLAIMER =
  "AI-observed views — not a record of photo documentation.";

export function SimulatedBadge() {
  return (
    <span
      className="inline-flex items-center border border-ws-coach/50 px-1.5 py-0.5 text-[10px] font-semibold leading-none tracking-[0.08em] text-ws-coach"
      title="Mock landmark model — output is simulated, not clinical"
    >
      SIMULATED
    </span>
  );
}

/** "3:12", as on the filmstrip. */
export function formatStationClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;
}

/** "03:12", for spoken names. */
function formatSpokenClock(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return `${m.toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;
}

/**
 * The row's name for assistive tech. It starts with the visible short label
 * (WCAG 2.5.3 Label in Name), so "click D2" works under voice control, then
 * adds the full label unless the short one already opens it.
 */
export function stationSpokenName(key: StationKey): string {
  const short = STATION_SHORT[key];
  const label = STATION_LABELS[key];
  return label.toLowerCase().startsWith(short.toLowerCase())
    ? label
    : `${short}, ${label}`;
}

function accessibleName(
  key: StationKey,
  state: StationRowState,
  at: number | null,
  autoEnabled: boolean,
): string {
  const label = stationSpokenName(key);
  let what: string;
  switch (state) {
    case "observed":
      what = at === null ? "observed" : `observed at ${formatSpokenClock(at)}`;
      break;
    case "confirmed":
      what = "confirmed manually";
      break;
    case "rejected":
      what = "rejected manually";
      break;
    case "candidate":
      what = "candidate, not yet observed";
      break;
    default:
      what = "not yet observed";
  }
  return `${label} — ${what}${autoEnabled ? "" : ", manual only"}`;
}

/**
 * Status glyphs, drawn rather than taken from the icon set so the five states
 * differ in shape as well as colour: ring / dotted ring / solid check / outline
 * check / struck ring.
 */
function StationGlyph({ state }: { state: StationRowState }) {
  const common = {
    width: 14,
    height: 14,
    viewBox: "0 0 14 14",
    "aria-hidden": true,
    className: "shrink-0",
  } as const;
  switch (state) {
    case "observed":
      return (
        <svg {...common} data-glyph="observed">
          <circle cx="7" cy="7" r="6" className="fill-peace-3" />
          <path
            d="M4 7.2 6.1 9.2 10 5"
            fill="none"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="stroke-ws-void"
          />
        </svg>
      );
    case "confirmed":
      return (
        <svg {...common} data-glyph="confirmed">
          <circle cx="7" cy="7" r="5.5" fill="none" strokeWidth="1.2" className="stroke-peace-3" />
          <path
            d="M4 7.2 6.1 9.2 10 5"
            fill="none"
            strokeWidth="1.4"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="stroke-peace-3"
          />
        </svg>
      );
    case "rejected":
      return (
        <svg {...common} data-glyph="rejected">
          <circle cx="7" cy="7" r="5.5" fill="none" strokeWidth="1.2" className="stroke-ws-label" />
          <path d="M3 11 11 3" strokeWidth="1.4" strokeLinecap="round" className="stroke-ws-label" />
        </svg>
      );
    case "candidate":
      return (
        <svg {...common} data-glyph="candidate">
          <circle
            cx="7"
            cy="7"
            r="5.5"
            fill="none"
            strokeWidth="1.4"
            strokeDasharray="1.6 2.2"
            className="stroke-ws-fg-2"
          />
        </svg>
      );
    default:
      return (
        <svg {...common} data-glyph="unseen">
          <circle cx="7" cy="7" r="5.5" fill="none" strokeWidth="1.2" className="stroke-current" />
        </svg>
      );
  }
}

interface StationChecklistProps {
  stations: StationsSlice;
  /** After the procedure ends, stations still unobserved turn amber. */
  ended?: boolean;
  /** Called with the ESGE index (0..9) when a row is pressed. */
  onToggle?: (index: number) => void;
}

/**
 * The 10 ESGE stations in ESGE order, grouped esophagus / duodenum / stomach.
 *
 * "Observed" is the model's (or clinician's) statement that the view was seen in
 * the feed — never that it was photo-documented, which is not observable here.
 * Each row is a button that cycles a manual mark, so stations the model cannot
 * recognise (manual only) remain usable and every override becomes a label.
 *
 * Renders nothing unless stations are visible: in shadow mode, or with the
 * feature off, the rail is exactly what it was before this component existed.
 */
export function StationChecklist({ stations, ended = false, onToggle }: StationChecklistProps) {
  const hintId = useId();

  if (!stations.visible) {
    if (!stations.display) return null;
    if (stations.availability === "unsupported_layout") {
      return (
        <p className="ws-micro leading-snug">
          Station tracking unavailable for this video source
        </p>
      );
    }
    if (stations.availability === "error") {
      return <p className="ws-micro leading-snug">Station tracking interrupted</p>;
    }
    return null;
  }

  return (
    <section aria-label="ESGE station checklist" className="space-y-2.5">
      {isSimulatedModel(stations.modelVersion) && <SimulatedBadge />}
      <span id={hintId} className="sr-only">
        Press to cycle the manual mark: confirm, reject, clear.
      </span>

      {STATION_GROUPS.map((group) => (
        <div key={group.region} className="space-y-1">
          <div className="ws-micro uppercase text-ws-faint">
            {REGION_LABELS[group.region]}
          </div>
          <ol aria-label={REGION_LABELS[group.region]} className="space-y-px">
            {group.stations.map((key) => {
              const i = STATION_INDEX[key];
              const state = stationRowState(stations, i);
              const observed = stations.observed[i];
              const at = stations.observedAtT[i];
              const auto = stations.autoEnabled[i] !== false;
              const isCurrent = stations.current === key;
              const overdue = ended && !observed;
              return (
                <li key={key}>
                  <button
                    type="button"
                    aria-pressed={stations.manual[i] !== null}
                    aria-current={isCurrent ? "step" : undefined}
                    aria-label={accessibleName(key, state, at, auto)}
                    aria-describedby={hintId}
                    onClick={() => onToggle?.(i)}
                    className={`grid w-full grid-cols-[1.1rem_minmax(0,1fr)_14px_2.4rem] items-center gap-2 border-l-2 px-1.5 py-[5px] text-left transition-colors hover:bg-ws-raised focus-visible:outline focus-visible:outline-1 focus-visible:outline-ws-detect ${
                      isCurrent ? "border-ws-detect bg-ws-surface" : "border-transparent"
                    } ${overdue ? "text-ws-coach" : "text-ws-faint"}`}
                  >
                    <span className="font-mono text-[11px] tabular-nums">{i + 1}</span>
                    <span
                      className={`truncate text-[13px] leading-tight ${
                        overdue
                          ? "text-ws-coach"
                          : observed
                            ? "text-ws-fg-2"
                            : "text-ws-label"
                      } ${state === "rejected" ? "line-through" : ""}`}
                    >
                      {STATION_SHORT[key]}
                      {!auto && (
                        <span className="ml-1.5 text-[10px] font-medium uppercase tracking-[0.06em] text-ws-faint no-underline">
                          manual
                        </span>
                      )}
                    </span>
                    <StationGlyph state={state} />
                    <span className="text-right font-mono text-[11px] tabular-nums text-ws-label">
                      {observed && at !== null ? formatStationClock(at) : ""}
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
        </div>
      ))}

      <p className="text-[11px] leading-snug text-ws-faint">{STATIONS_DISCLAIMER}</p>
    </section>
  );
}
