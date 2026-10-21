"use client";

import { MOTION_LABELS } from "@/lib/constants";
import type { MotionDirection } from "@/lib/types";

interface MotionVisualProps {
  direction: MotionDirection;
}

/** Same tones as MotionIndicator, as text colour the SVG inherits. */
const TONE: Record<MotionDirection, string> = {
  insertion: "text-primary",
  withdrawal: "text-warning",
  stationary: "text-muted-foreground",
};

export function MotionVisual({ direction }: MotionVisualProps) {
  return (
    <div className={`flex flex-col items-center gap-2 ${TONE[direction]}`}>
      <svg viewBox="0 0 80 120" className="h-28 w-16" aria-hidden="true">
        {direction === "insertion" && (
          <>
            {/* Single downward arrow */}
            <line x1="40" y1="25" x2="40" y2="95" stroke="currentColor" strokeWidth={4} strokeLinecap="round" />
            <polyline points="28,78 40,95 52,78" fill="none" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
          </>
        )}

        {direction === "withdrawal" && (
          <>
            {/* Single upward arrow */}
            <line x1="40" y1="25" x2="40" y2="95" stroke="currentColor" strokeWidth={4} strokeLinecap="round" />
            <polyline points="28,42 40,25 52,42" fill="none" stroke="currentColor" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
          </>
        )}

        {direction === "stationary" && (
          <>
            {/* Scope tube (stationary only) */}
            <line x1="40" y1="10" x2="40" y2="110" stroke="currentColor" strokeOpacity={0.2} strokeWidth={6} strokeLinecap="round" />
            {/* Pulse rings */}
            <circle cx="40" cy="60" r="6" fill="currentColor" />
            <circle cx="40" cy="60" r="14" fill="none" stroke="currentColor" strokeWidth={1.5} strokeOpacity={0.4} />
            <circle cx="40" cy="60" r="22" fill="none" stroke="currentColor" strokeWidth={1} strokeOpacity={0.2} />
          </>
        )}
      </svg>
      <span className="text-sm font-medium">{MOTION_LABELS[direction]}</span>
    </div>
  );
}
