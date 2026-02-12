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
        if (!data) return null;
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
