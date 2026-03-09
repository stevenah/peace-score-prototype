import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { getPresignedUrl } from "@/lib/s3";

export const runtime = "nodejs";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ analysisId: string }> },
) {
  const authSession = await auth();
  if (!authSession?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { analysisId } = await params;

  const session = await prisma.analysisSession.findUnique({
    where: { analysisId },
  });

  if (!session) {
    return NextResponse.json({ error: "Analysis not found" }, { status: 404 });
  }

  // Verify ownership
  if (session.userId !== authSession.user.id) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  // videoPath stores the S3 key (e.g. "videos/<job_id>.mp4")
  if (!session.videoPath) {
    // Video may not have been uploaded to S3 yet — try fetching from ML backend
    const mlUrl = process.env.ML_BACKEND_URL || "http://localhost:8000";
    try {
      const res = await fetch(`${mlUrl}/api/v1/analyze/${analysisId}`);
      if (res.ok) {
        const data = await res.json();
        if (data.video_path) {
          // Persist the S3 key so we don't have to ask again
          await prisma.analysisSession.update({
            where: { analysisId },
            data: { videoPath: data.video_path },
          });
          const url = await getPresignedUrl(data.video_path);
          return NextResponse.redirect(url);
        }
      }
    } catch {
      // ML backend unreachable — fall through to 404
    }
    return NextResponse.json({ error: "Video not available yet" }, { status: 404 });
  }

  try {
    const url = await getPresignedUrl(session.videoPath);
    return NextResponse.redirect(url);
  } catch {
    return NextResponse.json(
      { error: "Failed to generate video URL" },
      { status: 500 },
    );
  }
}
