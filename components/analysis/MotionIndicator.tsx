import { ArrowDown, ArrowUp, Pause } from "lucide-react";
import { MOTION_COLORS, MOTION_LABELS } from "@/lib/constants";
import type { MotionDirection } from "@/lib/types";

interface MotionIndicatorProps {
  direction: MotionDirection;
}

const icons = {
  insertion: ArrowDown,
  retraction: ArrowUp,
  stationary: Pause,
};

export function MotionIndicator({ direction }: MotionIndicatorProps) {
  const Icon = icons[direction];
  const color = MOTION_COLORS[direction];

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium"
      style={{ color, backgroundColor: `${color}15` }}
    >
      <Icon className="h-3.5 w-3.5" />
      {MOTION_LABELS[direction]}
    </span>
  );
}
