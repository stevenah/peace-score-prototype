import { PeaceScoreCard } from "./PeaceScoreCard";
import { REGION_LABELS, REGION_ORDER } from "@/lib/constants";
import type { AnatomicalRegion, PeaceScore, RegionScore } from "@/lib/types";

interface PeaceScoreGridProps {
  byRegion: Partial<Record<AnatomicalRegion, RegionScore>>;
}

export function PeaceScoreGrid({ byRegion }: PeaceScoreGridProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {REGION_ORDER.map((region) => {
        const data = byRegion[region];
        if (!data) {
          return (
            <div
              key={region}
              className="rounded-xl border border-neutral-200 bg-white p-5 dark:border-neutral-800 dark:bg-neutral-900"
            >
              <p className="mb-1 text-xs font-medium uppercase tracking-wider text-neutral-400">
                {REGION_LABELS[region]}
              </p>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-neutral-300 dark:text-neutral-600">
                  —
                </span>
                <span className="text-sm text-neutral-300 dark:text-neutral-600">/ 3</span>
              </div>
              <p className="mt-0.5 text-sm font-medium text-neutral-300 dark:text-neutral-600">
                No data
              </p>
            </div>
          );
        }
        return (
          <PeaceScoreCard
            key={region}
            score={data.score as PeaceScore}
            label={data.label}
            region={REGION_LABELS[region]}
          />
        );
      })}
    </div>
  );
}
