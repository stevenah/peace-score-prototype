"use client";

import { useRouter } from "next/navigation";
import { FileVideo, RotateCcw } from "lucide-react";
import { VideoUploader } from "@/components/video/VideoUploader";
import { BatchItemCard } from "@/components/video/BatchItemCard";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useBatchUpload } from "@/hooks/useBatchUpload";

export function UploadAnalysis() {
  const router = useRouter();
  const {
    items,
    addFiles,
    removeItem,
    reset,
    hasActive,
    allComplete,
    completedCount,
    totalCount,
  } = useBatchUpload();

  // No items: show the uploader full-size
  if (items.length === 0) {
    return <VideoUploader onFilesSelect={addFiles} />;
  }

  return (
    <div className="space-y-6">
      {/* Summary header */}
      <Card className="flex items-center justify-between px-5 py-4">
        <div className="flex items-center gap-3">
          <FileVideo className="h-5 w-5 text-primary" />
          <div>
            <p className="text-sm font-medium text-foreground">
              {completedCount} of {totalCount} complete
            </p>
            <p className="text-xs text-muted-foreground">
              {hasActive
                ? "Processing videos..."
                : allComplete
                  ? "All analyses finished"
                  : "Ready"}
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {allComplete && (
            <Button variant="secondary" size="sm" onClick={reset}>
              <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
              New Batch
            </Button>
          )}
          <Button
            variant="secondary"
            size="sm"
            onClick={() => router.push("/dashboard")}
          >
            View Dashboard
          </Button>
        </div>
      </Card>

      {/* Item list */}
      <div className="space-y-2">
        {items.map((item) => (
          <BatchItemCard key={item.id} item={item} onRemove={removeItem} />
        ))}
      </div>

      {/* Add more videos (compact uploader) */}
      {!allComplete && (
        <div className="h-32">
          <VideoUploader onFilesSelect={addFiles} />
        </div>
      )}
    </div>
  );
}
