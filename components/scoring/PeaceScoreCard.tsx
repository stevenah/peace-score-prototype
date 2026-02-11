import { Card } from "@/components/ui/Card";
import {
  PEACE_SCORE_BG_COLORS,
  PEACE_SCORE_COLORS,
  PEACE_SCORE_DESCRIPTIONS,
  PEACE_SCORE_LABELS,
} from "@/lib/constants";
import { formatConfidence } from "@/lib/utils";
import type { PeaceScore } from "@/lib/types";

interface PeaceScoreCardProps {
  score: PeaceScore;
  confidence: number;
  label?: string;
  region?: string;
  size?: "sm" | "lg";
}

export function PeaceScoreCard({
  score,
  confidence,
  label,
  region,
  size = "sm",
}: PeaceScoreCardProps) {
  const color = PEACE_SCORE_COLORS[score];
  const bgColor = PEACE_SCORE_BG_COLORS[score];
  const scoreLabel = label || PEACE_SCORE_LABELS[score];

  return (
    <Card
      className="relative overflow-hidden"
    >
      <div
        className="absolute inset-0 opacity-30"
        style={{ backgroundColor: bgColor }}
      />
      <div className="relative">
        {region && (
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-neutral-500">
            {region}
          </p>
        )}
        <div className="flex items-baseline gap-3">
          <span
            className={`font-bold ${size === "lg" ? "text-5xl" : "text-3xl"}`}
            style={{ color }}
          >
            {score}
          </span>
          <span className="text-sm font-medium text-neutral-600 dark:text-neutral-400">
            / 3
          </span>
        </div>
        <p
          className={`mt-1 font-semibold ${size === "lg" ? "text-lg" : "text-sm"}`}
          style={{ color }}
        >
          {scoreLabel}
        </p>
        <p className="mt-1 text-xs text-neutral-500">
          {PEACE_SCORE_DESCRIPTIONS[score]}
        </p>
        <p className="mt-2 text-xs text-neutral-400">
          Confidence: {formatConfidence(confidence)}
        </p>
      </div>
    </Card>
  );
}
