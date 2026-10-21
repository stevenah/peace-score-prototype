import { ArrowDown, ArrowUp, Pause } from "lucide-react";
import { Badge, type BadgeVariant } from "@/components/ui/Badge";
import { MOTION_LABELS } from "@/lib/constants";
import type { MotionDirection } from "@/lib/types";

interface MotionIndicatorProps {
  direction: MotionDirection;
}

const icons = {
  insertion: ArrowDown,
  withdrawal: ArrowUp,
  stationary: Pause,
};

/** Withdrawal is the phase that carries the cleaning alert, so it takes the
 *  warning tone; insertion is routine (primary); stationary is neutral. */
export const MOTION_VARIANTS: Record<MotionDirection, BadgeVariant> = {
  insertion: "primary",
  withdrawal: "warning",
  stationary: "default",
};

export function MotionIndicator({ direction }: MotionIndicatorProps) {
  const Icon = icons[direction];

  return (
    <Badge variant={MOTION_VARIANTS[direction]} className="px-3 py-1 text-sm [&>svg]:size-3.5">
      <Icon aria-hidden="true" />
      {MOTION_LABELS[direction]}
    </Badge>
  );
}
