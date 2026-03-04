"use client";

import type { AnatomicalRegion } from "@/lib/types";

interface RegionHighlightProps {
  activeRegion: AnatomicalRegion;
}

const regionImages: Record<AnatomicalRegion, string> = {
  esophagus: "/esophagus.png",
  stomach: "/stomach.png",
  duodenum: "/duodenum.png",
};

export function RegionHighlight({ activeRegion }: RegionHighlightProps) {
  return (
    <div className="relative my-2 min-h-0 flex-1 self-stretch overflow-hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={regionImages[activeRegion]}
        alt={activeRegion}
        className="absolute inset-0 m-auto h-[115%] w-2/3 object-contain"
      />
    </div>
  );
}
