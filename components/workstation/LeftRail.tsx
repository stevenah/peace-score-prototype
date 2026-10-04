"use client";

import { useCallback } from "react";
import { REGION_LABELS, REGION_ORDER } from "@/lib/constants";
import {
  selectCoverage,
  selectRegions,
  selectScore,
  selectStations,
  selectTimers,
  selectVisibility,
  useProcedureSlice,
  useProcedureStore,
} from "@/hooks/useProcedureMetrics";
import { STATION_REGION } from "@/lib/live/stations";
import { CoverageMap, type RegionCoverageState } from "./CoverageMap";
import { ScoreRing } from "./ScoreRing";
import {
  StationChecklist,
  StationsErrorBoundary,
  nextManualMark,
} from "./StationChecklist";
import type { AnatomicalRegion } from "@/lib/types";
import type {
  ProcedureSnapshot,
  VisibilityLevel,
} from "@/lib/live/procedure-types";

const VISIBILITY_TEXT: Record<VisibilityLevel, string> = {
  adequate: "Adequate",
  degraded: "Degraded",
  obscured: "Obscured",
};

const VISIBILITY_CLASS: Record<VisibilityLevel, string> = {
  adequate: "text-peace-3",
  degraded: "text-peace-1",
  obscured: "text-peace-0",
};

const selectEnded = (s: ProcedureSnapshot) => s.feed.ended;

export function LeftRail({ dimmed }: { dimmed: boolean }) {
  const timers = useProcedureSlice(selectTimers);
  const regions = useProcedureSlice(selectRegions);
  const coverage = useProcedureSlice(selectCoverage);
  const score = useProcedureSlice(selectScore);
  const visibility = useProcedureSlice(selectVisibility);
  const stations = useProcedureSlice(selectStations);
  const ended = useProcedureSlice(selectEnded);
  const store = useProcedureStore();

  const hasData = coverage.framesAnalysed > 0;
  const onTarget = timers.totalProgress >= 1;

  const toggleStation = useCallback(
    (i: number) => {
      const current = store.getSnapshot().stations.manual[i] ?? null;
      store.setManualStation(i, nextManualMark(current));
    },
    [store],
  );

  // With stations visible, the region label follows the station classifier, so
  // the rail never says "Stomach" beside a checklist whose current row is the
  // Z-line. The dwell clock is PEACE-region dwell, so it is shown only while the
  // two agree.
  const stationRegion =
    stations.visible && stations.current ? STATION_REGION[stations.current] : null;
  const regionLabel = stationRegion ?? regions.current;
  const showDwell =
    regions.current !== null && (stationRegion === null || stationRegion === regions.current);

  const coverageStates = REGION_ORDER.reduce(
    (acc, r) => {
      acc[r] = {
        fill: coverage.fillByRegion[r],
        active: regions.current === r,
        score: score.perRegionMean[r] ?? null,
      };
      return acc;
    },
    {} as Record<AnatomicalRegion, RegionCoverageState>,
  );

  return (
    <aside
      className={`flex min-h-0 select-none flex-col gap-5 overflow-y-auto px-5 py-4 transition-opacity duration-300 ${
        dimmed ? "opacity-50" : ""
      }`}
    >
      {/* Total procedure time, against the ESGE >=7 min examination measure. */}
      <div className="space-y-1.5">
        <div className="ws-label">Timing:</div>
        <div className="ws-num-lg">
          {hasData ? formatHms(timers.totalElapsedS) : "--:--:--"}
        </div>
        <div className="h-[5px] w-full bg-ws-raised" aria-hidden>
          <div
            className={`h-full transition-[width] duration-500 ease-linear motion-reduce:transition-none ${
              onTarget ? "bg-peace-3" : "bg-ws-coach"
            }`}
            style={{ width: `${timers.totalProgress * 100}%` }}
          />
        </div>
        <div className="ws-micro">
          Target {formatHms(timers.totalTargetS)} · ESGE examination time
        </div>
      </div>

      {/* Current region + its dwell. No per-region target: none is established. */}
      <div className="space-y-1">
        <div className="ws-label">
          {regionLabel ? `${REGION_LABELS[regionLabel]}:` : "Region:"}
        </div>
        <div className="ws-num-lg">
          {showDwell ? formatMs(timers.currentRegionDwellS) : "--:--"}
        </div>
      </div>

      <div className="space-y-1">
        <div className="ws-label">Visibility:</div>
        <div
          className={`ws-value ${
            visibility.level ? VISIBILITY_CLASS[visibility.level] : "text-ws-faint"
          }`}
        >
          {visibility.level ? VISIBILITY_TEXT[visibility.level] : "—"}
        </div>
        {visibility.detail && (
          <div className="ws-micro leading-snug">{visibility.detail}</div>
        )}
      </div>

      <div className="flex items-center gap-4">
        <ScoreRing
          score={score.display}
          label={score.displayLabel}
          sessionMean={score.meanScore}
          dimmed={false}
        />
        <div className="min-w-0 space-y-1">
          <div className="ws-label">Cleanliness</div>
          {typeof score.expected === "number" && (
            <div className="ws-micro leading-snug">
              Now <span className="text-ws-fg-2">{score.expected.toFixed(2)}</span>
              {" / 3"}
            </div>
          )}
          <div className="ws-micro leading-snug">
            Mean{" "}
            <span className="text-ws-fg-2">
              {typeof score.meanScore === "number" ? score.meanScore.toFixed(2) : "—"}
            </span>
            {" / 3"}
          </div>
          <div className="ws-micro leading-snug">
            Range{" "}
            <span className="text-ws-fg-2">
              {score.min == null ? "—" : `${score.min}–${score.max}`}
            </span>
          </div>
        </div>
      </div>

      {/*
        The reference shows "Observed Sites: 26" from a 26-station blind-spot
        protocol. When the ESGE station classifier is on and displayed, the hero
        is its 10-station count — "observed" in the feed, never "documented".
        Otherwise (feature off, shadow mode, unsupported source) the hero is the
        thing actually being accumulated: frames analysed.
      */}
      {stations.visible ? (
        <div className="space-y-1">
          <div className="ws-label">Stations observed:</div>
          <div className="flex items-baseline gap-2">
            <span className="ws-num-xl">{stations.observedCount}</span>
            <span className="ws-num-md text-ws-label">/ {stations.total}</span>
          </div>
          <div className="ws-micro">
            Frames analysed{" "}
            <span className="text-ws-fg-2">{coverage.framesAnalysed}</span>
            {" · "}Regions{" "}
            <span className="text-ws-fg-2">
              {coverage.regionsVisited} / {coverage.regionsTotal}
            </span>
          </div>
        </div>
      ) : (
        <div className="space-y-1">
          <div className="ws-label">Frames analysed:</div>
          <div className="ws-num-xl">{coverage.framesAnalysed}</div>
          <div className="ws-micro">
            Regions{" "}
            <span className="text-ws-fg-2">
              {coverage.regionsVisited} / {coverage.regionsTotal}
            </span>
          </div>
        </div>
      )}

      <StationsErrorBoundary>
        <StationChecklist
          stations={stations}
          ended={ended}
          onToggle={toggleStation}
        />
      </StationsErrorBoundary>

      <div className="mt-auto flex min-h-0 shrink-0 justify-center pt-3">
        <CoverageMap
          states={coverageStates}
          // Smaller while the checklist shares the rail, so both fit at 1080p.
          className={stations.visible ? "h-[150px] w-auto" : "h-[200px] w-auto"}
        />
      </div>
    </aside>
  );
}

function formatHms(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return [h, m, sec].map((n) => n.toString().padStart(2, "0")).join(":");
}

function formatMs(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  return `${m.toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;
}

