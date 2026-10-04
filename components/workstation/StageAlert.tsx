"use client";

import { AlertTriangle } from "lucide-react";
import { useCoachPrompt } from "./CoachingController";

/**
 * Alerts render over the video, at 17px, where the eye already is — the top-bar
 * pill is far too quiet for the one message that matters.
 *
 * The wash animation runs ONCE. A red flash repeating over live mucosa induces
 * afterimages in a dark-adapted eye and masks the very tissue the alert is
 * pointing at, which is why the old 1Hz infinite flash is gone.
 */
export function StageAlert() {
  const coach = useCoachPrompt();
  if (!coach || coach.level !== "alert") return null;

  return (
    <div
      className="pointer-events-none absolute left-1/2 top-6 z-10 flex -translate-x-1/2 items-center gap-2.5 bg-ws-alert px-5 py-2.5
                 animate-in fade-in slide-in-from-top-2 duration-200 ease-out
                 [animation:ws-alert-wash_700ms_ease-out_1_forwards]
                 motion-reduce:animate-none"
    >
      <AlertTriangle className="size-5 shrink-0 text-white" aria-hidden />
      <span className="text-[17px] font-semibold leading-none text-white">
        {coach.text}
      </span>
    </div>
  );
}
