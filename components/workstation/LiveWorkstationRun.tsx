"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { saveLiveAnalysis, updateLiveStations } from "@/lib/api-client";
import { useCaptureFilmstrip } from "@/hooks/useCaptureFilmstrip";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import { ProcedureStoreProvider } from "@/hooks/useProcedureMetrics";
import { stationToPromote, useStationGallery } from "@/hooks/useStationGallery";
import { useStaleness } from "@/hooks/useStaleness";
import {
  evaluateCapture,
  initialTriggerState,
  type TriggerState,
} from "@/lib/live/capture-triggers";
import { ProcedureStore } from "@/lib/live/procedure-store";
import { StationMarksSync } from "@/lib/live/station-marks-sync";
import type { ConnectionPhase } from "@/lib/live/coaching";
import {
  createLandmarkParser,
  frameSampleFromResult,
  stationsSummary,
  timelineEntryFromSample,
} from "@/lib/live/wire";
import type { LiveFrameResult } from "@/lib/types";
import { Announcements } from "./Announcements";
import { CaptureFilmstrip } from "./CaptureFilmstrip";
import { CoachingController } from "./CoachingController";
import { LeftRail } from "./LeftRail";
import { ProcedureVideo } from "./ProcedureVideo";
import { StageAlert } from "./StageAlert";
import { StationsErrorBoundary } from "./StationChecklist";
import { StationGalleryOverlay } from "./StationGallery";
import { WorkstationTopBar } from "./WorkstationTopBar";
import type { ProcedureSource } from "./PreProcedureEntry";

const CAPTURE_INTERVAL_MS = 500;
/** Frames awaiting a reply before playback is held so analysis can catch up. */
const MAX_IN_FLIGHT_FRAMES = 2;

export function LiveWorkstationRun({
  session,
  onExit,
}: {
  session: ProcedureSource;
  onExit: () => void;
}) {
  // Lazily-initialised state rather than a ref: one instance for the life of the
  // run, created without touching a ref during render.
  const [store] = useState(() => new ProcedureStore());
  // One parser per run, so a malformed landmark block warns once per procedure.
  const [parseLandmark] = useState(() => createLandmarkParser());
  const [saveState, setSaveState] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  /** Manual station marks made after the save, PATCHed (and retried) until acknowledged. */
  const [marksSync] = useState(
    () =>
      new StationMarksSync({
        read: () => {
          const st = store.getSnapshot().stations;
          return st.availability === "absent" ? null : st;
        },
        send: (id, patch) => updateLiveStations(id, patch),
        onResult: (ok) => setSaveState(ok ? "saved" : "error"),
      }),
  );

  const triggerRef = useRef<TriggerState>(initialTriggerState());
  /** A save is in flight or has landed; cleared again if it fails. */
  const savedRef = useRef(false);
  const durationRef = useRef(0);

  const [corsError, setCorsError] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [ended, setEnded] = useState(false);
  const [galleryDismissed, setGalleryDismissed] = useState(false);

  const { offerThumb, commit, captures } = useCaptureFilmstrip();
  const {
    offer: offerGalleryFrame,
    promote: promoteGalleryFrame,
    has: galleryHasFrame,
    frames: galleryFrames,
  } = useStationGallery();

  const handleResult = useCallback(
    (result: LiveFrameResult, videoTime: number, ticket: number) => {
      // PEACE fields map exactly as before; the landmark block is parsed and
      // fenced inside the adapter, so it can never cost us this sample.
      const sample = frameSampleFromResult(result, videoTime, ticket, parseLandmark);
      const { stationEdges } = store.ingest(sample);
      const stations = store.getSnapshot().stations;

      const { reason, station, next } = evaluateCapture(triggerRef.current, sample, {
        stations,
        stationEdges,
      });
      triggerRef.current = next;
      if (reason) {
        commit(ticket, {
          reason,
          region: sample.region,
          score: sample.score,
          station,
        });
      }

      try {
        const best = stationToPromote(sample, stations, galleryHasFrame);
        if (best) promoteGalleryFrame(ticket, best);
      } catch {
        // The gallery is a nicety; the analysis stream is not.
      }
    },
    [store, commit, parseLandmark, galleryHasFrame, promoteGalleryFrame],
  );

  const {
    isConnected,
    isConnecting,
    connectionError,
    inFlightFrames,
    lastResultAtMs,
    sendFrame,
  } = useLiveFeed({ enabled: true, onResult: handleResult });

  // Arm the watchdog whenever results are expected. Crucially that includes the
  // case where backpressure has paused the video: frames are still outstanding,
  // and gating on isPlaying alone would disarm the watchdog exactly when the
  // backend has gone quiet — the failure it exists to catch.
  const isStale = useStaleness(
    lastResultAtMs,
    isConnected && !ended && (isPlaying || inFlightFrames > 0),
  );

  // A socket that closes mid-procedure is not "idle": the video is still playing
  // and every panel is still showing the last numbers we received. Having ever
  // received a result is proof the connection was working, so no extra state is
  // needed to tell "not connected yet" from "connection lost".
  const lostConnection = lastResultAtMs !== null && !isConnected && !ended;
  const degraded = isStale || lostConnection;

  const handleFrameCapture = useCallback(
    (analysis: Blob, thumbnail: string, videoTime: number) => {
      // Everything captured with this frame is keyed by the ticket the feed
      // assigned it — the same value onResult reports back.
      const ticket = sendFrame(analysis, videoTime);
      if (ticket === null) return;
      offerThumb(ticket, thumbnail, videoTime);
      // Full-resolution frames are worth parking only while stations may be
      // shown: before the first result, or while they are visible.
      const snap = store.getSnapshot();
      if (snap.feed.framesReceived === 0 || snap.stations.visible) {
        offerGalleryFrame(ticket, analysis, videoTime);
      }
    },
    [sendFrame, offerThumb, offerGalleryFrame, store],
  );

  // Before the save lands there is nothing to patch: the save itself carries
  // the marks. After it, every publish checks for changed marks.
  useEffect(() => store.subscribe(() => marksSync.sync()), [store, marksSync]);
  useEffect(() => {
    marksSync.start();
    return () => marksSync.stop();
  }, [marksSync]);

  /**
   * Saves the procedure once. Called when the video ends AND on exit: URL/HLS
   * streams never fire onEnded, so without the exit path a live-stream procedure
   * would persist nothing. A failed save is retried by the next of either.
   */
  const persist = useCallback(() => {
    if (savedRef.current) return;
    const samples = store.getSamples();
    if (samples.length === 0) return;
    savedRef.current = true;
    setSaveState("saving");

    const snap = store.getSnapshot();
    const scores = samples.map((s) => s.score as number);
    const mean =
      Math.round((scores.reduce((a, b) => a + b, 0) / scores.length) * 100) / 100;
    const stations = stationsSummary(snap.stations);

    saveLiveAnalysis({
      filename: session.label,
      overallScore: mean,
      minScore: snap.score.min ?? 0,
      maxScore: snap.score.max ?? 0,
      avgScore: mean,
      framesAnalyzed: samples.length,
      duration: durationRef.current > 0 ? durationRef.current : null,
      timeline: samples.map(timelineEntryFromSample),
      stations,
      videoFile: session.source instanceof File ? session.source : undefined,
    })
      .then(
        ({ analysisId }) => {
          setSaveState("saved");
          marksSync.saved(analysisId, stations?.manual ?? null);
        },
        () => {
          // Not saved: let the next video end or the exit try again.
          savedRef.current = false;
          setSaveState("error");
        },
      );
  }, [store, session, marksSync]);

  const handleVideoEnd = useCallback(() => {
    setEnded(true);
    setGalleryDismissed(false);
    store.markEnded();
    persist();
  }, [store, persist]);

  const handleExit = useCallback(() => {
    persist();
    // Last chance for overrides whose PATCH failed and is waiting on a retry.
    marksSync.sync(true);
    onExit();
  }, [persist, marksSync, onExit]);

  const handleVideoReady = useCallback((d: number) => {
    durationRef.current = d;
  }, []);

  const handlePlayStateChange = useCallback((playing: boolean) => {
    setIsPlaying(playing);
    if (playing) setEnded(false);
  }, []);

  const handleCorsError = useCallback(() => setCorsError(true), []);

  const phase: ConnectionPhase = connectionError
    ? "error"
    : ended
      ? "ended"
      : lostConnection
        ? "disconnected"
        : isConnecting
          ? "connecting"
          : isStale
            ? "stalled"
            : isConnected
              ? "streaming"
              : "idle";

  const dot = connectionError
    ? "error"
    : degraded
      ? "stale"
      : isConnected
        ? "live"
        : isConnecting
          ? "connecting"
          : "idle";

  return (
    <ProcedureStoreProvider store={store}>
      <CoachingController phase={phase}>
        <div className="grid h-full min-h-0 grid-rows-[2rem_minmax(0,1fr)]">
          <WorkstationTopBar dot={dot} onExit={handleExit} />

          <div
            className="grid min-h-0 grid-cols-1 grid-rows-[minmax(0,1fr)_auto]
                       md:grid-cols-[clamp(200px,15.5vw,288px)_minmax(0,1fr)] md:grid-rows-[minmax(0,1fr)_auto]
                       xl:grid-cols-[clamp(200px,15.5vw,288px)_minmax(0,1fr)_clamp(112px,9vw,176px)] xl:grid-rows-[minmax(0,1fr)]"
          >
            <div className="relative row-start-1 min-h-0 min-w-0 md:col-start-2 md:row-start-1">
              <ProcedureVideo
                source={session.source}
                isLiveStream={session.mode === "url"}
                isAnalyzing={isConnected}
                syncPause={inFlightFrames >= MAX_IN_FLIGHT_FRAMES}
                captureIntervalMs={CAPTURE_INTERVAL_MS}
                onFrameCapture={handleFrameCapture}
                onVideoEnd={handleVideoEnd}
                onVideoReady={handleVideoReady}
                onPlayStateChange={handlePlayStateChange}
                onCorsError={handleCorsError}
              />

              <StageAlert />

              {/* Above the video (and its play button) but clear of the control
                  strip, so the procedure can still be replayed. */}
              <StationsErrorBoundary>
                <StationGalleryOverlay
                  open={ended && !galleryDismissed}
                  frames={galleryFrames}
                  onClose={() => setGalleryDismissed(true)}
                  className="absolute inset-x-0 top-0 bottom-9"
                />
              </StationsErrorBoundary>

              {corsError && (
                <div className="absolute bottom-14 left-1/2 -translate-x-1/2 select-text bg-ws-coach px-4 py-2 text-[13px] font-medium text-black">
                  Cannot capture frames — stream blocked by CORS policy
                </div>
              )}

              {saveState !== "idle" && (
                <div className="ws-micro absolute bottom-14 right-4 z-30">
                  {saveState === "saving" && "Saving…"}
                  {saveState === "saved" && "Saved to dashboard"}
                  {saveState === "error" && (
                    <span className="text-peace-0">Save failed</span>
                  )}
                </div>
              )}
            </div>

            <LeftRail dimmed={degraded} />

            <aside className="row-start-2 min-w-0 select-none px-3 py-3 md:col-start-2 md:row-start-2 xl:col-start-3 xl:row-start-1">
              <CaptureFilmstrip captures={captures} />
            </aside>
          </div>

          <Announcements />
        </div>
      </CoachingController>
    </ProcedureStoreProvider>
  );
}
