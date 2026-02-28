"use client";

import { useRouter } from "next/navigation";
import { CheckCircle2, ArrowRight, RotateCcw, Loader2 } from "lucide-react";
import { VideoUploader } from "@/components/video/VideoUploader";
import { AnalysisProgress } from "@/components/analysis/AnalysisProgress";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { useVideoUpload } from "@/hooks/useVideoUpload";
import { useAnalysis } from "@/hooks/useAnalysis";
import type { PeaceScore } from "@/lib/types";

type Phase = "idle" | "uploading" | "processing" | "complete" | "error";

function derivePhase(
  isUploading: boolean,
  analysisId: string | null,
  uploadError: string | null,
  analysisStatus: string | undefined,
  pollError: string | null,
): Phase {
  if (uploadError || pollError || analysisStatus === "failed") return "error";
  if (analysisStatus === "completed") return "complete";
  if (analysisId && !isUploading) return "processing";
  if (isUploading) return "uploading";
  return "idle";
}

export function UploadAnalysis() {
  const router = useRouter();
  const {
    upload,
    isUploading,
    analysisId,
    error: uploadError,
    reset,
  } = useVideoUpload();
  const { data: analysis, error: pollError } = useAnalysis(analysisId);

  const phase = derivePhase(
    isUploading,
    analysisId,
    uploadError,
    analysis?.status,
    pollError,
  );

  const errorMessage =
    uploadError || pollError || analysis?.error || "Analysis failed";

  if (phase === "idle") {
    return <VideoUploader onFileSelect={(file) => upload(file)} />;
  }

  if (phase === "uploading") {
    return (
      <Card>
        <div className="flex items-center gap-3 py-8 justify-center">
          <Loader2 className="h-5 w-5 animate-spin text-blue-600" />
          <div>
            <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
              Uploading video...
            </p>
            <p className="text-xs text-neutral-500">
              Sending video to analysis server
            </p>
          </div>
        </div>
      </Card>
    );
  }

  if (phase === "processing" && analysis) {
    return (
      <div className="space-y-4">
        <AnalysisProgress status={analysis.status} progress={analysis.progress} />
        <p className="text-center text-xs text-neutral-500">
          Analysis will continue even if you leave this page
        </p>
      </div>
    );
  }

  if (phase === "complete" && analysis?.results) {
    const overallScore = analysis.results.peace_scores.overall.score as PeaceScore;
    const overallLabel = analysis.results.peace_scores.overall.label;

    return (
      <div className="space-y-6">
        <Card className="py-8 text-center">
          <CheckCircle2 className="mx-auto h-12 w-12 text-green-500" />
          <h3 className="mt-3 text-lg font-semibold text-neutral-900 dark:text-neutral-100">
            Analysis Complete
          </h3>
          {analysis.video_metadata && (
            <p className="mt-1 text-sm text-neutral-500">
              {analysis.video_metadata.analyzed_frames} frames analyzed
              {analysis.video_metadata.duration_seconds > 0 &&
                ` \u00b7 ${Math.round(analysis.video_metadata.duration_seconds)}s duration`}
            </p>
          )}
        </Card>

        <div className="flex justify-center">
          <PeaceScoreCard score={overallScore} label={overallLabel} size="lg" />
        </div>

        <div className="flex justify-center gap-3">
          <Button onClick={() => router.push(`/results/${analysisId}`)}>
            <ArrowRight className="mr-2 h-4 w-4" />
            View Full Results
          </Button>
          <Button variant="secondary" onClick={reset}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Analyze Another
          </Button>
        </div>
      </div>
    );
  }

  if (phase === "error") {
    return (
      <Card className="py-8 text-center">
        <p className="text-sm font-medium text-red-600 dark:text-red-400">
          {errorMessage}
        </p>
        <Button variant="secondary" onClick={reset} className="mt-4">
          <RotateCcw className="mr-2 h-4 w-4" />
          Try Again
        </Button>
      </Card>
    );
  }

  return null;
}
