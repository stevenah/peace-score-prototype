import { NextRequest, NextResponse } from "next/server";
import { ML_BACKEND_URL } from "@/lib/constants";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { computeScoreStats } from "@/lib/utils";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { id } = await params;

    // Verify user owns this analysis
    const existing = await prisma.analysisSession.findUnique({
      where: { analysisId: id },
      select: { userId: true },
    });
    if (existing && existing.userId !== session.user.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const response = await fetch(`${ML_BACKEND_URL}/api/v1/analyze/${id}`);

    if (!response.ok) {
      const text = await response.text();
      return NextResponse.json(
        { error: text },
        { status: response.status },
      );
    }

    const data = await response.json();

    // Update analysis session when status changes to completed/failed
    if (data.status === "completed" || data.status === "failed") {
      try {
        const timeline: { peace_score: number }[] = data.results?.timeline ?? [];
        const frameScores = timeline.map((e) => e.peace_score);
        const stats = computeScoreStats(frameScores);

        await prisma.analysisSession.updateMany({
          where: { analysisId: id },
          data: {
            status: data.status,
            completedAt: new Date(),
            overallScore: data.results?.peace_scores?.overall?.score ?? null,
            minScore: stats.minScore,
            maxScore: stats.maxScore,
            avgScore: stats.avgScore,
            framesAnalyzed: data.video_metadata?.analyzed_frames ?? null,
            duration: data.video_metadata?.duration_seconds ?? null,
            // Persist the S3 key from ML backend so the video proxy can find it
            ...(data.video_path ? { videoPath: data.video_path } : {}),
          },
        });
      } catch {
        // Non-critical: don't fail the response if DB update fails
      }
    }

    // Determine if a video is available and point to the proxy route
    const hasVideoFromMl = !!data.video_path || !!data.video_url;
    if (hasVideoFromMl) {
      data.video_url = `/api/video/${id}`;
    } else {
      try {
        const dbSession = await prisma.analysisSession.findUnique({
          where: { analysisId: id },
          select: { videoPath: true },
        });
        if (dbSession?.videoPath) {
          data.video_url = `/api/video/${id}`;
        }
      } catch {
        // Non-critical
      }
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Analysis fetch error:", error);
    return NextResponse.json(
      { error: "Failed to fetch analysis" },
      { status: 500 },
    );
  }
}
