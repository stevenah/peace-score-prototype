"use client";

import { memo } from "react";
import Link from "next/link";
import {
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  ArrowRight,
  X,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import { Button } from "@/components/ui/Button";
import { formatFileSize } from "@/lib/utils";
import { PEACE_SCORE_COLORS } from "@/lib/constants";
import type { BatchItem, PeaceScore } from "@/lib/types";

const STATUS_CONFIG: Record<
  string,
  { icon: React.ReactNode; label: string; bgColor: string; color: string }
> = {
  pending: {
    icon: <Clock className="h-3 w-3" />,
    label: "Pending",
    bgColor: "#f3f4f6",
    color: "#6b7280",
  },
  uploading: {
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
    label: "Uploading",
    bgColor: "#e0e7ff",
    color: "#3730a3",
  },
  queued: {
    icon: <Clock className="h-3 w-3" />,
    label: "Queued",
    bgColor: "#fef3c7",
    color: "#92400e",
  },
  processing: {
    icon: <Loader2 className="h-3 w-3 animate-spin" />,
    label: "Processing",
    bgColor: "#dbeafe",
    color: "#1e40af",
  },
  completed: {
    icon: <CheckCircle2 className="h-3 w-3" />,
    label: "Complete",
    bgColor: "#dcfce7",
    color: "#166534",
  },
  failed: {
    icon: <XCircle className="h-3 w-3" />,
    label: "Failed",
    bgColor: "#fee2e2",
    color: "#991b1b",
  },
};

export const BatchItemCard = memo(function BatchItemCard({
  item,
  onRemove,
}: {
  item: BatchItem;
  onRemove: (id: string) => void;
}) {
  const config = STATUS_CONFIG[item.status];
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

  return (
    <Card className="flex items-center gap-4 px-4 py-3">
      {/* File info */}
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-foreground">
          {item.file.name}
        </p>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>{formatFileSize(item.file.size)}</span>
          {item.error && (
            <span
              className="truncate text-red-600 dark:text-red-400"
              title={item.error}
            >
              {item.error}
            </span>
          )}
        </div>
      </div>

      {/* Progress bar for active states */}
      {isActive && (
        <div className="w-24">
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
      <Badge bgColor={config.bgColor} color={config.color}>
        {config.icon}
        {config.label}
      </Badge>

      {/* View results link */}
      {item.status === "completed" && item.analysisId && (
        <Link href={`/results/${item.analysisId}`}>
          <Button variant="ghost" size="icon-xs">
            <ArrowRight className="h-3.5 w-3.5" />
          </Button>
        </Link>
      )}

      {/* Remove button */}
      {canRemove && (
        <button
          type="button"
          onClick={() => onRemove(item.id)}
          className="rounded-md p-1 text-muted-foreground transition-colors hover:text-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </Card>
  );
});
