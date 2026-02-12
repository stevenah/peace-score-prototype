import { NextRequest, NextResponse } from "next/server";
import { ML_BACKEND_URL } from "@/lib/constants";
import { prisma } from "@/lib/db";

export async function GET(
  _request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
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
        await prisma.analysisSession.updateMany({
          where: { analysisId: id },
          data: {
            status: data.status,
            completedAt: new Date(),
            overallScore: data.results?.peace_scores?.overall?.score ?? null,
            framesAnalyzed: data.video_metadata?.analyzed_frames ?? null,
            duration: data.video_metadata?.duration_seconds ?? null,
          },
        });
      } catch {
        // Non-critical: don't fail the response if DB update fails
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
