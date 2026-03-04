import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { ML_BACKEND_URL } from "@/lib/constants";

export async function GET() {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const analyses = await prisma.analysisSession.findMany({
    where: { userId: session.user.id },
    orderBy: { createdAt: "desc" },
    take: 50,
  });

  // Sync status from ML backend for any in-progress jobs
  const activeJobs = analyses.filter(
    (a) => a.status === "processing" || a.status === "queued",
  );

  if (activeJobs.length > 0) {
    await Promise.allSettled(
      activeJobs.map(async (job) => {
        const res = await fetch(
          `${ML_BACKEND_URL}/api/v1/analyze/${job.analysisId}`,
        );
        if (!res.ok) return null;
        const data = await res.json();

        if (data.status === "completed" || data.status === "failed") {
          await prisma.analysisSession.updateMany({
            where: { analysisId: job.analysisId },
            data: {
              status: data.status,
              completedAt: new Date(),
              overallScore:
                data.results?.peace_scores?.overall?.score ?? null,
              framesAnalyzed:
                data.video_metadata?.analyzed_frames ?? null,
              duration: data.video_metadata?.duration_seconds ?? null,
            },
          });

          // Update the in-memory record so the response reflects the new data
          const idx = analyses.findIndex((a) => a.id === job.id);
          if (idx !== -1) {
            analyses[idx] = {
              ...analyses[idx],
              status: data.status,
              completedAt: new Date(),
              overallScore:
                data.results?.peace_scores?.overall?.score ?? null,
              framesAnalyzed:
                data.video_metadata?.analyzed_frames ?? null,
              duration:
                data.video_metadata?.duration_seconds ?? null,
            };
          }
        }
      }),
    );
  }

  return NextResponse.json(analyses);
}
