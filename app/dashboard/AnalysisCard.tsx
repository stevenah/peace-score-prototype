"use client";

import Link from "next/link";
import { memo, useState, useTransition } from "react";
import {
  Clock,
  CheckCircle2,
  XCircle,
  Loader2,
  Trash2,
  ExternalLink,
} from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import { Button } from "@/components/ui/Button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { deleteAnalysis } from "./actions";
import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS } from "@/lib/constants";
import { formatDuration } from "@/lib/utils";
import type { PeaceScore, AnalysisRecord } from "@/lib/types";

const StatusBadge = memo(function StatusBadge({ status }: { status: string }) {
  switch (status) {
    case "queued":
      return (
        <Badge bgColor="#fef3c7" color="#92400e">
          <Clock className="h-3 w-3" />
          Queued
        </Badge>
      );
    case "processing":
      return (
        <Badge bgColor="#dbeafe" color="#1e40af">
          <Loader2 className="h-3 w-3 animate-spin" />
          Processing
        </Badge>
      );
    case "completed":
      return (
        <Badge bgColor="#dcfce7" color="#166534">
          <CheckCircle2 className="h-3 w-3" />
          Complete
        </Badge>
      );
    case "failed":
      return (
        <Badge bgColor="#fee2e2" color="#991b1b">
          <XCircle className="h-3 w-3" />
          Failed
        </Badge>
      );
    default:
      return null;
  }
});

export const AnalysisCard = memo(function AnalysisCard({
  analysis,
  onDelete,
}: {
  analysis: AnalysisRecord;
  onDelete: () => void;
}) {
  const [isPending, startTransition] = useTransition();
  const [open, setOpen] = useState(false);
  const isComplete = analysis.status === "completed";
  const score = analysis.overallScore as PeaceScore | null;

  function handleDelete() {
    const formData = new FormData();
    formData.set("id", analysis.id);
    startTransition(async () => {
      await deleteAnalysis(formData);
      setOpen(false);
      onDelete();
    });
  }

  const dateStr =
    new Date(analysis.createdAt).toLocaleDateString() +
    " at " +
    new Date(analysis.createdAt).toLocaleTimeString();

  return (
    <Card className="flex flex-col justify-between">
      <div>
        {/* Header: filename + status */}
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-foreground">
              {analysis.filename}
            </p>
            <p className="text-xs text-muted-foreground">{dateStr}</p>
          </div>
          <StatusBadge status={analysis.status} />
        </div>

        {/* Status-specific content */}
        <div className="mt-4">
          {analysis.status === "processing" && (
            <div>
              <Progress value={50} color="#3b82f6" />
              <p className="mt-1 text-xs text-muted-foreground">
                Analyzing frames...
              </p>
            </div>
          )}

          {analysis.status === "queued" && (
            <p className="text-xs text-muted-foreground">
              Waiting for available worker...
            </p>
          )}

          {isComplete && score !== null && (
            <div className="flex items-center gap-4">
              <div>
                <span
                  className="text-2xl font-bold"
                  style={{ color: PEACE_SCORE_COLORS[score] }}
                >
                  {score}
                </span>
                <span className="text-sm text-muted-foreground"> / 3</span>
                <p
                  className="text-xs font-medium"
                  style={{ color: PEACE_SCORE_COLORS[score] }}
                >
                  {PEACE_SCORE_LABELS[score]}
                </p>
              </div>
              <div className="flex-1 space-y-1 text-xs text-muted-foreground">
                {analysis.framesAnalyzed != null && (
                  <p>{analysis.framesAnalyzed} frames</p>
                )}
                {analysis.duration != null && (
                  <p>{formatDuration(analysis.duration)}</p>
                )}
              </div>
            </div>
          )}

          {analysis.status === "failed" && (
            <p className="text-xs text-red-500">
              Analysis could not be completed
            </p>
          )}
        </div>
      </div>

      {/* Footer: actions */}
      <div className="mt-4 flex items-center justify-between border-t border-border pt-3">
        {isComplete ? (
          <Link href={`/results/${analysis.analysisId}`}>
            <Button variant="ghost" size="sm">
              <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
              View Results
            </Button>
          </Link>
        ) : analysis.status === "processing" || analysis.status === "queued" ? (
          <Link href={`/results/${analysis.analysisId}`}>
            <Button variant="ghost" size="sm">
              View Progress
            </Button>
          </Link>
        ) : (
          <div />
        )}

        <AlertDialog open={open} onOpenChange={setOpen}>
          <AlertDialogTrigger asChild>
            <button
              type="button"
              className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-red-50 hover:text-red-600 dark:hover:bg-red-950/50 dark:hover:text-red-400"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete analysis</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete &ldquo;{analysis.filename}
                &rdquo;? This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={isPending}>Cancel</AlertDialogCancel>
              <AlertDialogAction
                variant="destructive"
                onClick={handleDelete}
                disabled={isPending}
              >
                {isPending ? "Deleting..." : "Delete"}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </Card>
  );
});
