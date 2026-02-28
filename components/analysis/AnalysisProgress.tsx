import { Loader2 } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Progress } from "@/components/ui/Progress";
import type { AnalysisStatus } from "@/lib/types";

interface AnalysisProgressProps {
  status: AnalysisStatus;
  progress: number;
  error?: string;
}

const statusMessages: Record<AnalysisStatus, string> = {
  queued: "Queued for analysis...",
  processing: "Analyzing video frames...",
  completed: "Analysis complete",
  failed: "Analysis failed",
};

export function AnalysisProgress({
  status,
  progress,
  error,
}: AnalysisProgressProps) {
  const isActive = status === "queued" || status === "processing";

  return (
    <Card>
      <div className="flex items-center gap-3">
        {isActive && (
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
        )}
        <div className="flex-1">
          <p className="text-sm font-medium text-foreground">
            {statusMessages[status]}
          </p>
          {isActive && (
            <div className="mt-2">
              <Progress value={progress * 100} />
              <p className="mt-1 text-xs text-muted-foreground">
                {Math.round(progress * 100)}% complete
              </p>
            </div>
          )}
          {error && (
            <p className="mt-1 text-sm text-red-600 dark:text-red-400">
              {error}
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}
