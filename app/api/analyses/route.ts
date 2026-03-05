import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { ML_BACKEND_URL } from "@/lib/constants";

export async function GET(request: NextRequest) {
  const session = await auth();
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = request.nextUrl;
  const page = Math.max(1, parseInt(searchParams.get("page") || "1"));
  const pageSize = Math.min(
    100,
    Math.max(1, parseInt(searchParams.get("pageSize") || "24")),
  );
  const status = searchParams.get("status");
  const search = searchParams.get("search");
  const scores = searchParams.get("scores");
  const dateRange = searchParams.get("dateRange");
  const sort = searchParams.get("sort") || "date-desc";

  // Parse score filter values
  const scoreValues = scores
    ? scores
        .split(",")
        .map(Number)
        .filter((n) => !isNaN(n))
    : [];

  // Parse date cutoff
  let dateCutoff: Date | null = null;
  if (dateRange && dateRange !== "all") {
    const now = new Date();
    switch (dateRange) {
      case "today":
        dateCutoff = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        break;
      case "week":
        dateCutoff = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case "month":
        dateCutoff = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
    }
  }

  // Build where clause
  const where = {
    userId: session.user.id,
    ...(status === "processing" && {
      status: { in: ["processing", "queued"] },
    }),
    ...((status === "completed" || status === "failed") && { status }),
    ...(search && { filename: { contains: search } }),
    ...(scoreValues.length > 0 && { overallScore: { in: scoreValues } }),
    ...(dateCutoff && { createdAt: { gte: dateCutoff } }),
  };

  // Build orderBy
  let orderBy: Record<string, string>;
  switch (sort) {
    case "date-asc":
      orderBy = { createdAt: "asc" };
      break;
    case "score-desc":
      orderBy = { overallScore: "desc" };
      break;
    case "score-asc":
      orderBy = { overallScore: "asc" };
      break;
    case "name-asc":
      orderBy = { filename: "asc" };
      break;
    default:
      orderBy = { createdAt: "desc" };
  }

  const userFilter = { userId: session.user.id };

  // Fetch paginated data + global counts in parallel
  const [
    allCount,
    processingCount,
    completedCount,
    failedCount,
    filteredTotal,
    analyses,
  ] = await Promise.all([
    prisma.analysisSession.count({ where: userFilter }),
    prisma.analysisSession.count({
      where: { ...userFilter, status: { in: ["processing", "queued"] } },
    }),
    prisma.analysisSession.count({
      where: { ...userFilter, status: "completed" },
    }),
    prisma.analysisSession.count({
      where: { ...userFilter, status: "failed" },
    }),
    prisma.analysisSession.count({ where }),
    prisma.analysisSession.findMany({
      where,
      orderBy,
      skip: (page - 1) * pageSize,
      take: pageSize,
    }),
  ]);

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

  return NextResponse.json({
    data: analyses,
    total: filteredTotal,
    page,
    pageSize,
    counts: {
      all: allCount,
      processing: processingCount,
      completed: completedCount,
      failed: failedCount,
    },
  });
}
