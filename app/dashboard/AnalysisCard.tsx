"use client";

import Link from "next/link";
import { memo, useState, useTransition } from "react";
import { XCircle, Trash2, ExternalLink } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { StatusBadge } from "@/components/analysis/StatusBadge";
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
import { cn, formatDuration } from "@/lib/utils";
import type { PeaceScore, AnalysisRecord } from "@/lib/types";

export const AnalysisCard = memo(function AnalysisCard({
  analysis,
  onDelete,
  selectMode = false,
  selected = false,
  onToggleSelect,
}: {
  analysis: AnalysisRecord;
  onDelete: () => void;
  selectMode?: boolean;
  selected?: boolean;
  onToggleSelect?: () => void;
}) {
  const [isPending, startTransition] = useTransition();
  const [open, setOpen] = useState(false);
  const isComplete = analysis.status === "completed";
  const isActive = analysis.status === "processing" || analysis.status === "queued";
  const score = analysis.overallScore;
  const scoreInt = score !== null ? Math.round(score) as PeaceScore : null;

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
    <div
      className="h-full"
      role={selectMode ? "button" : undefined}
      tabIndex={selectMode ? 0 : undefined}
      onClick={selectMode ? onToggleSelect : undefined}
      onKeyDown={selectMode ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onToggleSelect?.(); } } : undefined}
    >
    <Card
      className={cn(
        "flex h-full flex-col justify-between transition-all",
        selectMode && "cursor-pointer hover:ring-2 hover:ring-primary/40",
        selected && "border-primary/50 bg-primary/10 shadow-md shadow-primary/10 ring-2 ring-primary",
      )}
    >
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
              <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
                <div className="h-full animate-indeterminate rounded-full bg-primary" />
              </div>
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

          {isComplete && score !== null && scoreInt !== null && (
            <div className="flex items-center gap-4">
              <div>
                <span
                  className="text-2xl font-bold"
                  style={{ color: PEACE_SCORE_COLORS[scoreInt] }}
                >
                  {score}
                </span>
                <span className="text-sm text-muted-foreground"> / 3</span>
                <p
                  className="text-xs font-medium"
                  style={{ color: PEACE_SCORE_COLORS[scoreInt] }}
                >
                  {PEACE_SCORE_LABELS[scoreInt]}
                </p>
              </div>
              <div className="flex-1 space-y-1 text-xs text-muted-foreground">
                {analysis.minScore != null && analysis.maxScore != null && analysis.avgScore != null && (
                  <div className="flex gap-3">
                    <span>Min <span className="font-medium text-foreground">{analysis.minScore}</span></span>
                    <span>Max <span className="font-medium text-foreground">{analysis.maxScore}</span></span>
                    <span>Avg <span className="font-medium text-foreground">{analysis.avgScore.toFixed(1)}</span></span>
                  </div>
                )}
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
            <p className="text-xs text-destructive">
              Analysis could not be completed
            </p>
          )}
        </div>
      </div>

      {/* Footer: actions */}
      <div className="mt-4 flex items-center justify-between border-t border-border pt-3">
        {isComplete ? (
          <Button variant="ghost" size="sm" asChild>
            <Link href={`/results/${analysis.analysisId}`}>
              <ExternalLink />
              View Results
            </Link>
          </Button>
        ) : analysis.status === "processing" || analysis.status === "queued" ? (
          <Button variant="ghost" size="sm" asChild>
            <Link href={`/results/${analysis.analysisId}`}>View Progress</Link>
          </Button>
        ) : (
          <div />
        )}

        <AlertDialog open={open} onOpenChange={setOpen}>
          <AlertDialogTrigger asChild>
            <Button
              variant="ghost-destructive"
              size="icon-xs"
              title={isActive ? "Cancel" : "Delete"}
              aria-label={isActive ? "Cancel analysis" : "Delete analysis"}
            >
              {isActive ? <XCircle /> : <Trash2 />}
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>
                {isActive ? "Cancel analysis" : "Delete analysis"}
              </AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to {isActive ? "cancel" : "delete"} &ldquo;{analysis.filename}
                &rdquo;? This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={isPending}>Go Back</AlertDialogCancel>
              <AlertDialogAction
                variant="destructive"
                onClick={(e) => {
                  e.preventDefault();
                  handleDelete();
                }}
                disabled={isPending}
              >
                {isPending
                  ? isActive ? "Canceling..." : "Deleting..."
                  : isActive ? "Cancel Analysis" : "Delete"}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </Card>
    </div>
  );
});
