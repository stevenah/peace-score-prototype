import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS } from "@/lib/constants";
import type { PeaceScore } from "@/lib/types";

export function ScoreLegend() {
  const scores: PeaceScore[] = [0, 1, 2, 3];

  return (
    <div className="flex flex-wrap items-center gap-4">
      {scores.map((score) => (
        <div key={score} className="flex items-center gap-2">
          <div
            className="h-3 w-3 rounded-full"
            style={{ backgroundColor: PEACE_SCORE_COLORS[score] }}
          />
          <span className="text-xs text-neutral-600 dark:text-neutral-400">
            {score} - {PEACE_SCORE_LABELS[score]}
          </span>
        </div>
      ))}
    </div>
  );
}
