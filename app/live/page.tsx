"use client";

import { useState } from "react";
import { Video, VideoOff } from "lucide-react";
import { LiveFeedViewer } from "@/components/video/LiveFeedViewer";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { Button } from "@/components/ui/Button";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import type { PeaceScore, MotionDirection, TimelineEntry } from "@/lib/types";

export default function LivePage() {
  const [isActive, setIsActive] = useState(false);
  const { latestResult, results, sendFrame, reset } = useLiveFeed({
    enabled: isActive,
    wsUrl: "ws://localhost:8000/api/v1/ws/live",
  });

  function handleToggle() {
    if (isActive) {
      setIsActive(false);
      reset();
    } else {
      setIsActive(true);
    }
  }

  const timeline: TimelineEntry[] = results.map((r, i) => ({
    timestamp: i,
    frame_index: r.frame_index,
    motion: (r.motion?.direction || "stationary") as MotionDirection,
    region: (r.region || "stomach") as TimelineEntry["region"],
    peace_score: r.peace_score.score as PeaceScore,
    confidence: r.peace_score.confidence,
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
            Live Feed
          </h1>
          <p className="mt-1 text-sm text-neutral-500">
            Real-time PEACE score analysis from camera
          </p>
        </div>
        <Button
          onClick={handleToggle}
          variant={isActive ? "secondary" : "default"}
        >
          {isActive ? (
            <>
              <VideoOff className="h-4 w-4" /> Stop
            </>
          ) : (
            <>
              <Video className="h-4 w-4" /> Start
            </>
          )}
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <LiveFeedViewer
            isActive={isActive}
            onFrameCapture={sendFrame}
            captureIntervalMs={1000}
          />
        </div>

        <div className="space-y-4">
          {latestResult ? (
            <>
              <PeaceScoreCard
                score={latestResult.peace_score.score as PeaceScore}
                label={latestResult.peace_score.label}
                size="lg"
              />

              {latestResult.motion && (
                <div className="rounded-xl border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400">
                    Motion
                  </p>
                  <MotionIndicator
                    direction={latestResult.motion.direction as MotionDirection}
                  />
                </div>
              )}

              {latestResult.region && (
                <div className="rounded-xl border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
                  <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400">
                    Region
                  </p>
                  <p className="text-base font-medium capitalize text-neutral-900 dark:text-neutral-100">
                    {latestResult.region}
                  </p>
                </div>
              )}
            </>
          ) : (
            <div className="rounded-xl border border-neutral-200 bg-white p-8 text-center dark:border-neutral-800 dark:bg-neutral-900">
              <p className="text-sm text-neutral-400">
                {isActive ? "Waiting for first frame..." : "Start the feed to begin"}
              </p>
            </div>
          )}
        </div>
      </div>

      {timeline.length > 5 && <PeaceScoreTimeline timeline={timeline} />}
    </div>
  );
}
