"use client";

import Link from "next/link";
import { useTransition, useState } from "react";
import { deleteAnalysis } from "./actions";
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

export function AnalysisRow({
  analysis,
}: {
  analysis: { id: string; analysisId: string; filename: string; createdAt: Date };
}) {
  const [isPending, startTransition] = useTransition();
  const [open, setOpen] = useState(false);

  function handleDelete() {
    const formData = new FormData();
    formData.set("id", analysis.id);
    startTransition(async () => {
      await deleteAnalysis(formData);
      setOpen(false);
    });
  }

  return (
    <div className="flex items-center justify-between rounded-lg border border-neutral-200 p-4 transition-colors hover:bg-neutral-50 dark:border-neutral-800 dark:hover:bg-neutral-800/50">
      <Link href={`/results/${analysis.analysisId}`} className="min-w-0 flex-1">
        <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
          {analysis.filename}
        </p>
        <p className="text-xs text-neutral-500">
          {new Date(analysis.createdAt).toLocaleDateString()} at{" "}
          {new Date(analysis.createdAt).toLocaleTimeString()}
        </p>
      </Link>
      <div className="flex items-center gap-2">
        <Link
          href={`/results/${analysis.analysisId}`}
          className="rounded-md bg-neutral-900 px-3 py-1 text-xs font-medium text-white dark:bg-neutral-100 dark:text-neutral-900"
        >
          View
        </Link>
        <AlertDialog open={open} onOpenChange={setOpen}>
          <AlertDialogTrigger asChild>
            <button
              type="button"
              className="rounded-md px-3 py-1 text-xs font-medium text-red-600 transition-colors hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/50"
            >
              Delete
            </button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete analysis</AlertDialogTitle>
              <AlertDialogDescription>
                Are you sure you want to delete &ldquo;{analysis.filename}&rdquo;? This action cannot be undone.
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
    </div>
  );
}
