import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { PEACE_SCORE_LABELS } from "@/lib/constants";
import type { PeaceScore } from "@/lib/types";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const authSession = await auth();
    if (!authSession?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id } = await params;
    const session = await prisma.analysisSession.findUnique({
      where: { analysisId: id },
    });

    if (!session) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }

    if (session.userId !== authSession.user.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const score = session.overallScore as PeaceScore | null;
    const timeline = session.timelineData
      ? JSON.parse(session.timelineData)
      : [];

    return NextResponse.json({
      analysis_id: session.analysisId,
      status: session.status,
      progress: 100,
      results:
        score !== null
          ? {
              peace_scores: {
                overall: {
                  score,
                  label: PEACE_SCORE_LABELS[score],
                  confidence: 1,
                },
                by_region: {},
              },
              motion_analysis: { segments: [] },
              timeline,
            }
          : undefined,
      video_metadata: {
        duration_seconds: session.duration ?? 0,
        fps: 0,
        resolution: [0, 0],
        total_frames: session.framesAnalyzed ?? 0,
        analyzed_frames: session.framesAnalyzed ?? 0,
      },
      video_url: session.videoPath
        ? `/api/video/${session.analysisId}`
        : null,
      created_at: session.createdAt.toISOString(),
      completed_at: session.completedAt?.toISOString(),
    });
  } catch (error) {
    console.error("Live analysis fetch error:", error);
    return NextResponse.json(
      { error: "Failed to fetch analysis" },
      { status: 500 },
    );
  }
}
