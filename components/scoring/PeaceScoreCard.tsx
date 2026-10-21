import { memo } from "react";
import { Card, CardEyebrow } from "@/components/ui/Card";
import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS } from "@/lib/constants";
import type { PeaceScore } from "@/lib/types";

interface PeaceScoreCardProps {
  score: PeaceScore;
  label?: string;
  region?: string;
  size?: "sm" | "lg";
}

export const PeaceScoreCard = memo(function PeaceScoreCard({
  score,
  label,
  region,
  size = "sm",
}: PeaceScoreCardProps) {
  const color = PEACE_SCORE_COLORS[score];
  const scoreLabel = label || PEACE_SCORE_LABELS[score];

  return (
    <Card>
      <CardEyebrow className="mb-2">{region || "PEACE Score"}</CardEyebrow>
      <div className="flex items-baseline gap-2">
        <span
          className={`font-bold ${size === "lg" ? "text-4xl" : "text-2xl"}`}
          style={{ color }}
        >
          {score}
        </span>
        <span className="text-sm text-muted-foreground/60">/ 3</span>
      </div>
      <p
        className={`mt-0.5 font-medium ${size === "lg" ? "text-base" : "text-sm"}`}
        style={{ color }}
      >
        {scoreLabel}
      </p>
    </Card>
  );
});
