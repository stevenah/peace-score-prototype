import { memo } from "react";
import { PeaceScoreCard } from "./PeaceScoreCard";
import { Card, CardEyebrow } from "@/components/ui/Card";
import { REGION_LABELS, REGION_ORDER } from "@/lib/constants";
import type { AnatomicalRegion, PeaceScore, RegionScore } from "@/lib/types";

interface PeaceScoreGridProps {
  byRegion: Partial<Record<AnatomicalRegion, RegionScore>>;
}

export const PeaceScoreGrid = memo(function PeaceScoreGrid({ byRegion }: PeaceScoreGridProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      {REGION_ORDER.map((region) => {
        const data = byRegion[region];
        if (!data) {
          return (
            <Card key={region}>
              <CardEyebrow className="mb-2">{REGION_LABELS[region]}</CardEyebrow>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold text-muted-foreground/30">
                  —
                </span>
                <span className="text-sm text-muted-foreground/30">/ 3</span>
              </div>
              <p className="mt-0.5 text-sm font-medium text-muted-foreground/30">
                No data
              </p>
            </Card>
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
});
