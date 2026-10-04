"use client";

import { Activity, X } from "lucide-react";
import { useCoachPrompt } from "./CoachingController";

type DotState = "idle" | "connecting" | "live" | "stale" | "error";

const DOT: Record<DotState, string> = {
  idle: "bg-ws-faint",
  connecting: "bg-ws-coach",
  live: "bg-peace-3",
  stale: "bg-ws-coach",
  error: "bg-ws-alert",
};

interface Props {
  dot: DotState;
  onExit: () => void;
}

/**
 * grid-cols-[1fr_auto_1fr] rather than justify-between, so the coaching pill stays
 * optically centred and the side clusters do not shift as it mounts and unmounts.
 */
export function WorkstationTopBar({ dot, onExit }: Props) {
  // Alerts render over the video instead; the pill carries routine guidance only.
  const coach = useCoachPrompt();
  const pill = coach && coach.level !== "alert" ? coach : null;

  return (
    <header className="grid h-8 shrink-0 select-none grid-cols-[1fr_auto_1fr] items-center gap-4 bg-ws-void px-3">
      <div className="flex min-w-0 items-center gap-2">
        <Activity className="size-3.5 shrink-0 text-ws-label" aria-hidden />
        <span className="ws-micro truncate uppercase">PEACE Live</span>
        <span
          className={`ml-1 size-1.5 shrink-0 rounded-full ${DOT[dot]}`}
          aria-hidden
        />
      </div>

      <div className="flex min-w-0 justify-center">
        {pill && (
          <div
            className={`flex min-w-0 items-center rounded-full px-4 py-1 animate-in fade-in slide-in-from-top-1 duration-200 ease-out motion-reduce:animate-none ${
              pill.level === "warn"
                ? "bg-ws-coach/12 font-semibold text-ws-coach"
                : pill.level === "info"
                  ? "bg-ws-surface font-medium text-ws-label"
                  : "bg-ws-surface font-medium text-ws-coach"
            }`}
          >
            <span className="truncate text-[13px] leading-none">{pill.text}</span>
          </div>
        )}
      </div>

      <div className="flex min-w-0 items-center justify-end gap-3">
        <span className="ws-micro whitespace-nowrap uppercase text-ws-faint">
          Research prototype — not for clinical use
        </span>
        <button
          onClick={onExit}
          className="grid size-6 shrink-0 place-items-center text-ws-label transition-colors hover:text-ws-fg"
          aria-label="Exit workstation"
        >
          <X className="size-4" />
        </button>
      </div>
    </header>
  );
}
