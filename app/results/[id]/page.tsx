"use client";

import { use } from "react";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { AnalysisProgress } from "@/components/analysis/AnalysisProgress";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { PeaceScoreGrid } from "@/components/scoring/PeaceScoreGrid";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { RegionMap } from "@/components/scoring/RegionMap";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { useAnalysis } from "@/hooks/useAnalysis";
import { formatDuration } from "@/lib/utils";
import type { PeaceScore, MotionDirection } from "@/lib/types";

export default function ResultsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  const { data: analysis, error } = useAnalysis(id);

  if (error) {
    return (
      <div className="py-12 text-center">
        <p className="text-red-600">{error}</p>
        <Link href="/analyze" className="mt-4 text-sm text-blue-600">
          Back to analysis
        </Link>
      </div>
    );
  }

  if (!analysis) {
    return <AnalysisProgress status="processing" progress={0} />;
  }

  const isComplete = analysis.status === "completed";
  const results = analysis.results;

  return (
    <div className="space-y-8">
      <div className="flex items-center gap-4">
        <Link
          href="/analyze"
          className="rounded-lg p-2 text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800"
        >
          <ArrowLeft className="h-5 w-5" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
            Analysis Results
          </h1>
          <p className="text-xs text-neutral-500">ID: {id}</p>
        </div>
      </div>

      {!isComplete && (
        <AnalysisProgress
          status={analysis.status}
          progress={analysis.progress}
          error={analysis.error || undefined}
        />
      )}

      {isComplete && results && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-4">
            <PeaceScoreCard
              score={results.peace_scores.overall.score as PeaceScore}
              label={results.peace_scores.overall.label}
              size="lg"
            />
            <div className="lg:col-span-3">
              <Card>
                <CardHeader>
                  <CardTitle>Video Details</CardTitle>
                </CardHeader>
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <div>
                    <p className="text-xs text-neutral-500">Duration</p>
                    <p className="text-lg font-semibold">
                      {analysis.video_metadata
                        ? formatDuration(
                            analysis.video_metadata.duration_seconds,
                          )
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-neutral-500">Resolution</p>
                    <p className="text-lg font-semibold">
                      {analysis.video_metadata
                        ? `${analysis.video_metadata.resolution[0]}x${analysis.video_metadata.resolution[1]}`
                        : "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-neutral-500">Frames Analyzed</p>
                    <p className="text-lg font-semibold">
                      {analysis.video_metadata?.analyzed_frames || 0}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-neutral-500">Segments</p>
                    <p className="text-lg font-semibold">
                      {results.motion_analysis.segments.length}
                    </p>
                  </div>
                </div>
              </Card>
            </div>
          </div>

          <PeaceScoreGrid byRegion={results.peace_scores.by_region} />
          <PeaceScoreTimeline timeline={results.timeline} />

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <RegionMap byRegion={results.peace_scores.by_region} />
            <Card>
              <CardHeader>
                <CardTitle>Motion Segments</CardTitle>
              </CardHeader>
              <div className="space-y-2">
                {results.motion_analysis.segments.map((seg, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded-lg bg-neutral-50 px-4 py-2 dark:bg-neutral-800"
                  >
                    <MotionIndicator
                      direction={seg.direction as MotionDirection}
                    />
                    <span className="text-xs text-neutral-500">
                      {formatDuration(seg.start_time)} →{" "}
                      {formatDuration(seg.end_time)}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
