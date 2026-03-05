"use client";

import { memo } from "react";
import {
  AlertTriangle,
  Clock,
  Loader2,
  Upload,
  X,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import { formatFileSize } from "@/lib/utils";
import type { BatchItem } from "@/lib/types";

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
  failed: {
    icon: <AlertTriangle className="h-3 w-3" />,
    label: "Failed",
    bgColor: "#fee2e2",
    color: "#991b1b",
  },
};

export const UploadCard = memo(function UploadCard({
  item,
  onRemove,
}: {
  item: BatchItem;
  onRemove: (id: string) => void;
}) {
  const config = STATUS_CONFIG[item.status];
  if (!config) return null;

  const progressPercent = Math.round(item.progress * 100);

  return (
    <Card className="flex h-full flex-col justify-between">
      <div>
        {/* Header: filename + status */}
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-foreground">
              {item.file.name}
            </p>
            <p className="text-xs text-muted-foreground">
              {formatFileSize(item.file.size)}
            </p>
          </div>
          <Badge bgColor={config.bgColor} color={config.color}>
            {config.icon}
            {config.label}
          </Badge>
        </div>

        {/* Status-specific content */}
        <div className="mt-4">
          {item.status === "pending" && (
            <p className="text-xs text-muted-foreground">
              Waiting to upload...
            </p>
          )}

          {item.status === "uploading" && (
            <div>
              <Progress value={progressPercent} />
              <p className="mt-1 text-xs text-muted-foreground">
                Uploading... {progressPercent}%
              </p>
            </div>
          )}

          {item.status === "queued" && (
            <div>
              <Progress value={progressPercent} color="#ca8a04" />
              <p className="mt-1 text-xs text-muted-foreground">
                Waiting for available worker...
              </p>
            </div>
          )}

          {item.status === "processing" && (
            <div>
              <Progress value={progressPercent} />
              <p className="mt-1 text-xs text-muted-foreground">
                Analyzing frames... {progressPercent}%
              </p>
            </div>
          )}

          {item.error && (
            <p className="mt-1 text-xs text-red-500">{item.error}</p>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="mt-4 flex items-center justify-between border-t border-border pt-3">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {item.status === "failed" ? (
            <>
              <AlertTriangle className="h-3 w-3 text-red-500" />
              <span className="text-red-500">Upload failed</span>
            </>
          ) : (
            <>
              <Upload className="h-3 w-3" />
              In progress
            </>
          )}
        </div>
        <button
          type="button"
          onClick={() => onRemove(item.id)}
          className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/50 dark:hover:text-red-400"
          title={item.status === "failed" ? "Dismiss" : "Cancel"}
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </Card>
  );
});
