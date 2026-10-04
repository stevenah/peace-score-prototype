"use client";

import type { Detection } from "@/lib/types";

interface DetectionOverlayProps {
  /** Absent today: the current model has no detection head. */
  detections?: readonly Detection[];
  minConfidence?: number;
}

/**
 * Draws detection boxes over the video.
 *
 * Always mounted, even with nothing to draw, so the stacking context and
 * compositing layer already exist when the first box arrives. It renders no
 * placeholder, no frame and no legend — an empty overlay must be invisible, not
 * "empty-looking".
 *
 * Coordinates are normalised 0..1 (see NormalizedBox). viewBox 0 0 100 100 with
 * preserveAspectRatio="none" lets them pass straight through; the parent .ws-fit
 * wrapper is exactly the video's content box, so no letterbox maths is needed.
 */
export function DetectionOverlay({
  detections,
  minConfidence = 0,
}: DetectionOverlayProps) {
  const visible = (detections ?? []).filter(
    (d) => d.confidence >= minConfidence,
  );

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      aria-hidden="true"
    >
      {visible.map((d) => (
        <rect
          key={d.id}
          x={d.box.x * 100}
          y={d.box.y * 100}
          width={d.box.w * 100}
          height={d.box.h * 100}
          fill="none"
          stroke="var(--ws-detect)"
          strokeWidth={1.5}
          /* Without this the non-uniform viewBox scaling stretches the stroke,
             giving thick horizontal and thin vertical edges. */
          vectorEffect="non-scaling-stroke"
          shapeRendering="crispEdges"
        />
      ))}
    </svg>
  );
}
