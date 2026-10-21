"use client";

import { memo } from "react";
import { AlertTriangle, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Progress } from "@/components/ui/Progress";
import { StatusBadge } from "@/components/analysis/StatusBadge";
import { formatFileSize } from "@/lib/utils";
import type { BatchItem } from "@/lib/types";

export const UploadCard = memo(function UploadCard({
  item,
  onRemove,
}: {
  item: BatchItem;
  onRemove: (id: string) => void;
}) {
  if (item.status === "completed") return null;

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
          <StatusBadge status={item.status} />
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
              <Progress value={progressPercent} tone="warning" />
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
            <p className="mt-1 text-xs text-destructive">{item.error}</p>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="mt-4 flex items-center justify-between border-t border-border pt-3">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          {item.status === "failed" ? (
            <span className="inline-flex items-center gap-1.5 text-destructive">
              <AlertTriangle aria-hidden="true" className="size-3" />
              Upload failed
            </span>
          ) : (
            <>
              <Upload aria-hidden="true" className="size-3" />
              In progress
            </>
          )}
        </div>
        <Button
          variant="ghost-destructive"
          size="icon-xs"
          onClick={() => onRemove(item.id)}
          title={item.status === "failed" ? "Dismiss" : "Cancel"}
          aria-label={item.status === "failed" ? "Dismiss" : "Cancel upload"}
        >
          <X />
        </Button>
      </div>
    </Card>
  );
});
