"use client";

import { memo } from "react";
import Link from "next/link";
import { ArrowRight, X } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Progress } from "@/components/ui/Progress";
import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/analysis/StatusBadge";
import { formatFileSize } from "@/lib/utils";
import { PEACE_SCORE_COLORS } from "@/lib/constants";
import type { BatchItem, PeaceScore } from "@/lib/types";

export const BatchItemCard = memo(function BatchItemCard({
  item,
  onRemove,
}: {
  item: BatchItem;
  onRemove: (id: string) => void;
}) {
  const score = item.analysis?.results?.peace_scores.overall
    .score as PeaceScore | undefined;
  const isActive =
    item.status === "uploading" ||
    item.status === "queued" ||
    item.status === "processing";
  const canRemove =
    item.status === "pending" ||
    item.status === "completed" ||
    item.status === "failed";

  // Upload-phase detail: show uploaded / total + percentage
  const uploadFraction =
    item.status === "uploading" ? item.progress / 0.4 : 0; // UPLOAD_WEIGHT = 0.4
  const uploadedBytes = Math.round(uploadFraction * item.file.size);
  const uploadPct = Math.round(uploadFraction * 100);

  // Status detail text
  const detailText =
    item.status === "uploading"
      ? `${formatFileSize(uploadedBytes)} / ${formatFileSize(item.file.size)} (${uploadPct}%)`
      : item.status === "queued"
        ? "Waiting for processing..."
        : item.status === "processing"
          ? `Analyzing — ${Math.round(item.progress * 100)}%`
          : null;

  return (
    <Card className="flex items-center gap-4 px-4 py-3">
      {/* File info */}
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-foreground">
          {item.file.name}
        </p>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{formatFileSize(item.file.size)}</span>
          {detailText && (
            <span className="text-muted-foreground/70">{detailText}</span>
          )}
          {item.error && (
            <span className="truncate text-destructive" title={item.error}>
              {item.error}
            </span>
          )}
        </div>
      </div>

      {/* Progress bar for active states */}
      {isActive && (
        <div className="w-32">
          <Progress value={item.progress * 100} />
        </div>
      )}

      {/* Score for completed */}
      {item.status === "completed" && score !== undefined && (
        <span
          className="text-lg font-bold"
          style={{ color: PEACE_SCORE_COLORS[score] }}
        >
          {score}
          <span className="text-xs text-muted-foreground">/3</span>
        </span>
      )}

      {/* Status badge */}
      <StatusBadge status={item.status} />

      {/* View results link */}
      {item.status === "completed" && item.analysisId && (
        <Button variant="ghost" size="icon-xs" asChild>
          <Link href={`/results/${item.analysisId}`} aria-label="View results">
            <ArrowRight />
          </Link>
        </Button>
      )}

      {/* Remove button */}
      {canRemove && (
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={() => onRemove(item.id)}
          aria-label="Remove"
        >
          <X />
        </Button>
      )}
    </Card>
  );
});
