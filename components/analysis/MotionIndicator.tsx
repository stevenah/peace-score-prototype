import { ArrowDown, ArrowUp, Pause } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { MOTION_COLORS, MOTION_LABELS } from "@/lib/constants";
import { formatConfidence } from "@/lib/utils";
import type { MotionDirection } from "@/lib/types";

interface MotionIndicatorProps {
  direction: MotionDirection;
  confidence?: number;
  size?: "sm" | "md";
}

const icons = {
  insertion: ArrowDown,
  retraction: ArrowUp,
  stationary: Pause,
};

export function MotionIndicator({
  direction,
  confidence,
  size = "md",
}: MotionIndicatorProps) {
  const Icon = icons[direction];
  const color = MOTION_COLORS[direction];

  return (
    <Badge color={color} bgColor={`${color}15`}>
      <Icon className={size === "sm" ? "h-3 w-3" : "h-4 w-4"} />
      <span>{MOTION_LABELS[direction]}</span>
      {confidence !== undefined && (
        <span className="opacity-60">{formatConfidence(confidence)}</span>
      )}
    </Badge>
  );
}
