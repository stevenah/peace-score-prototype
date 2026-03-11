import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { getObjectStream } from "@/lib/s3";

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

  if (session.userId !== authSession.user.id) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  let videoPath = session.videoPath;

  if (!videoPath) {
    try {
      const { fetchMlBackend } = await import("@/lib/ml-backend");
      const res = await fetchMlBackend(`/api/v1/analyze/${analysisId}`);
      if (res.ok) {
        const data = await res.json();
        if (data.video_path) {
          await prisma.analysisSession.update({
            where: { analysisId },
            data: { videoPath: data.video_path },
          });
          videoPath = data.video_path;
        }
      }
    } catch {
      // ML backend unreachable
    }
    if (!videoPath) {
      return NextResponse.json(
        { error: "Video not available yet" },
        { status: 404 },
      );
    }
  }

  try {
    const range = request.headers.get("range") || undefined;
    const s3Response = await getObjectStream(videoPath, range);

    const headers: Record<string, string> = {
      "Accept-Ranges": "bytes",
      "Content-Type": s3Response.ContentType || "video/mp4",
      "Cache-Control": "private, max-age=3600",
    };

    if (s3Response.ContentLength != null) {
      headers["Content-Length"] = String(s3Response.ContentLength);
    }
    if (s3Response.ContentRange) {
      headers["Content-Range"] = s3Response.ContentRange;
    }

    const status = s3Response.ContentRange ? 206 : 200;

    const body = s3Response.Body;
    if (!body) {
      return NextResponse.json(
        { error: "Empty response from storage" },
        { status: 502 },
      );
    }

    const webStream = body.transformToWebStream();

    return new Response(webStream, { status, headers });
  } catch {
    return NextResponse.json(
      { error: "Failed to stream video" },
      { status: 500 },
    );
  }
}
