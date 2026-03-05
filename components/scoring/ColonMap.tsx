'use client';

import { SEGMENT_LABELS, type ColonSegment } from '@/lib/types';

// SVG path data for colon segments (simplified anatomical representation)
const SEGMENT_PATHS: Record<ColonSegment, { path: string; labelX: number; labelY: number }> = {
  rectum: {
    path: 'M 200 340 Q 200 360 200 380 L 200 400',
    labelX: 200,
    labelY: 420,
  },
  sigmoid: {
    path: 'M 200 340 Q 160 340 140 310 Q 120 280 140 250',
    labelX: 110,
    labelY: 295,
  },
  descending: {
    path: 'M 140 250 L 140 150',
    labelX: 100,
    labelY: 200,
  },
  splenic_flexure: {
    path: 'M 140 150 Q 140 100 180 100',
    labelX: 130,
    labelY: 80,
  },
  transverse: {
    path: 'M 180 100 L 280 100',
    labelX: 230,
    labelY: 80,
  },
  hepatic_flexure: {
    path: 'M 280 100 Q 320 100 320 150',
    labelX: 340,
    labelY: 80,
  },
  ascending: {
    path: 'M 320 150 L 320 280',
    labelX: 360,
    labelY: 215,
  },
  cecum: {
    path: 'M 320 280 Q 320 320 300 340 Q 280 360 280 340',
    labelX: 340,
    labelY: 340,
  },
};

interface ColonMapProps {
  segmentsVisited?: ColonSegment[];
  currentSegment?: ColonSegment | null;
  qualityScores?: Partial<Record<ColonSegment, number>>;
}

export function ColonMap({
  segmentsVisited = [],
  currentSegment = null,
  qualityScores = {},
}: ColonMapProps) {
  const getSegmentColor = (segment: ColonSegment) => {
    if (currentSegment === segment) return '#3b82f6'; // Blue - current
    if (segmentsVisited.includes(segment)) {
      const score = qualityScores[segment] ?? 0;
      if (score >= 0.8) return '#22c55e'; // Green
      if (score >= 0.6) return '#eab308'; // Yellow
      return '#f97316'; // Orange
    }
    return '#e5e7eb'; // Gray - not visited
  };

  return (
    <div className="flex flex-1 flex-col bg-card rounded-xl border border-border p-4">
      <h3 className="text-sm font-medium text-muted-foreground mb-2">Colon Map</h3>

      <svg viewBox="0 0 460 460" className="w-full h-auto">
        {/* Background */}
        <rect width="460" height="460" fill="none" rx="8" />

        {/* Title */}
        <text x="230" y="30" textAnchor="middle" className="text-sm font-medium fill-foreground/80">
          Colonoscopy Progress
        </text>

        {/* Draw all segments */}
        {(Object.entries(SEGMENT_PATHS) as [ColonSegment, typeof SEGMENT_PATHS[ColonSegment]][]).map(
          ([segment, { path }]) => (
            <path
              key={segment}
              d={path}
              fill="none"
              stroke={getSegmentColor(segment)}
              strokeWidth={currentSegment === segment ? 20 : 16}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="transition-all duration-300"
            />
          )
        )}

        {/* Segment labels */}
        {(Object.entries(SEGMENT_PATHS) as [ColonSegment, typeof SEGMENT_PATHS[ColonSegment]][]).map(
          ([segment, { labelX, labelY }]) => (
            <text
              key={`label-${segment}`}
              x={labelX}
              y={labelY}
              textAnchor="middle"
              className={`text-xs ${
                segmentsVisited.includes(segment as ColonSegment)
                  ? 'fill-foreground/80'
                  : 'fill-muted-foreground/60'
              }`}
            >
              {SEGMENT_LABELS[segment as ColonSegment]}
            </text>
          )
        )}

        {/* Current position marker */}
        {currentSegment && (
          <circle
            cx={SEGMENT_PATHS[currentSegment].labelX}
            cy={SEGMENT_PATHS[currentSegment].labelY - 30}
            r="8"
            fill="#ef4444"
            className="animate-pulse"
          />
        )}

        {/* Legend */}
        <g transform="translate(20, 430)">
          <rect x="0" y="0" width="12" height="12" fill="#22c55e" rx="2" />
          <text x="18" y="10" className="text-xs fill-muted-foreground">Visited (Good)</text>

          <rect x="120" y="0" width="12" height="12" fill="#3b82f6" rx="2" />
          <text x="138" y="10" className="text-xs fill-muted-foreground">Current</text>

          <rect x="210" y="0" width="12" height="12" fill="#e5e7eb" rx="2" />
          <text x="228" y="10" className="text-xs fill-muted-foreground">Not Visited</text>
        </g>
      </svg>

      {/* Progress stats */}
      <div className="mt-3 flex justify-between text-sm">
        <span className="text-muted-foreground">
          Segments: {segmentsVisited.length}/8
        </span>
        <span className={`font-medium ${
          segmentsVisited.includes('cecum') ? 'text-green-600' : 'text-muted-foreground'
        }`}>
          {segmentsVisited.includes('cecum') ? '✓ Cecum Reached' : 'Cecum Not Reached'}
        </span>
      </div>
    </div>
  );
}
