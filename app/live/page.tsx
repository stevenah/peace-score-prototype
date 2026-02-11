"use client";

import { useState } from "react";
import { Video, VideoOff, Activity } from "lucide-react";
import { LiveFeedViewer } from "@/components/video/LiveFeedViewer";
import { MotionIndicator } from "@/components/analysis/MotionIndicator";
import { PeaceScoreCard } from "@/components/scoring/PeaceScoreCard";
import { PeaceScoreTimeline } from "@/components/scoring/PeaceScoreTimeline";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { useLiveFeed } from "@/hooks/useLiveFeed";
import type { PeaceScore, MotionDirection, TimelineEntry } from "@/lib/types";

export default function LivePage() {
  const [isActive, setIsActive] = useState(false);
  const { isConnected, latestResult, results, framesProcessed, sendFrame, reset } =
    useLiveFeed({
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

  // Build timeline from live results
  const timeline: TimelineEntry[] = results.map((r, i) => ({
    timestamp: i,
    frame_index: r.frame_index,
    motion: (r.motion?.direction || "stationary") as MotionDirection,
    region: (r.region || "stomach") as TimelineEntry["region"],
    peace_score: r.peace_score.score as PeaceScore,
    confidence: r.peace_score.confidence,
  }));

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
            Live Feed Analysis
          </h1>
          <p className="mt-1 text-sm text-neutral-500">
            Real-time PEACE score analysis from camera feed
          </p>
        </div>
        <div className="flex items-center gap-3">
          {isActive && (
            <Badge
              color={isConnected ? "#22c55e" : "#ef4444"}
              bgColor={isConnected ? "#f0fdf4" : "#fef2f2"}
            >
              <Activity className="h-3 w-3" />
              {isConnected ? "Connected" : "Connecting..."}
            </Badge>
          )}
          <Button
            onClick={handleToggle}
            variant={isActive ? "secondary" : "primary"}
          >
            {isActive ? (
              <>
                <VideoOff className="h-4 w-4" /> Stop
              </>
            ) : (
              <>
                <Video className="h-4 w-4" /> Start Live Feed
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Camera Feed */}
        <div className="lg:col-span-2">
          <LiveFeedViewer
            isActive={isActive}
            onFrameCapture={sendFrame}
            captureIntervalMs={1000}
          />
        </div>

        {/* Live Scores */}
        <div className="space-y-4">
          {latestResult ? (
            <>
              <PeaceScoreCard
                score={latestResult.peace_score.score as PeaceScore}
                confidence={latestResult.peace_score.confidence}
                label={latestResult.peace_score.label}
                size="lg"
              />

              {latestResult.motion && (
                <Card>
                  <CardHeader>
                    <CardTitle>Camera Motion</CardTitle>
                  </CardHeader>
                  <MotionIndicator
                    direction={latestResult.motion.direction as MotionDirection}
                    confidence={latestResult.motion.confidence}
                  />
                </Card>
              )}

              {latestResult.region && (
                <Card>
                  <CardHeader>
                    <CardTitle>Current Region</CardTitle>
                  </CardHeader>
                  <p className="text-lg font-semibold capitalize text-neutral-900 dark:text-neutral-100">
                    {latestResult.region}
                  </p>
                </Card>
              )}

              <Card>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-neutral-500">Frames processed</span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-100">
                    {framesProcessed}
                  </span>
                </div>
                {latestResult.processing_time_ms && (
                  <div className="mt-2 flex items-center justify-between text-sm">
                    <span className="text-neutral-500">Latency</span>
                    <span className="font-medium text-neutral-900 dark:text-neutral-100">
                      {Math.round(latestResult.processing_time_ms)}ms
                    </span>
                  </div>
                )}
              </Card>
            </>
          ) : (
            <Card className="py-12 text-center">
              <p className="text-sm text-neutral-500">
                {isActive
                  ? "Waiting for first frame..."
                  : "Start the live feed to begin analysis"}
              </p>
            </Card>
          )}
        </div>
      </div>

      {/* Live Timeline */}
      {timeline.length > 5 && <PeaceScoreTimeline timeline={timeline} />}
    </div>
  );
}
