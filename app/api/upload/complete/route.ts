import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { logAudit } from "@/lib/audit";
import { ML_BACKEND_URL } from "@/lib/constants";

export async function POST(request: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { uploadId, s3Key, filename } = await request.json();

    if (!s3Key || !filename) {
      return NextResponse.json(
        { error: "Missing s3Key or filename" },
        { status: 400 },
      );
    }

    // Notify ML backend to start analysis from S3
    const mlRes = await fetch(`${ML_BACKEND_URL}/api/v1/analyze/s3`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ s3_key: s3Key }),
    });

    if (!mlRes.ok) {
      const text = await mlRes.text().catch(() => "ML backend error");
      console.error("ML backend error:", text);
      return NextResponse.json(
        { error: "Failed to start analysis" },
        { status: 502 },
      );
    }

    const { analysis_id: analysisId } = await mlRes.json();

    // Prevent duplicates
    const existing = await prisma.analysisSession.findUnique({
      where: { analysisId },
    });

    if (!existing) {
      await prisma.analysisSession.create({
        data: {
          userId: session.user.id,
          analysisId,
          filename,
          status: "processing",
          videoPath: s3Key,
        },
      });
    }

    logAudit({
      actorId: session.user.id,
      actorEmail: session.user.email ?? null,
      action: "VIDEO_UPLOADED",
      targetType: "AnalysisSession",
      targetId: analysisId,
      targetLabel: filename,
    });

    return NextResponse.json({ ok: true, analysisId });
  } catch (error) {
    console.error("Upload complete error:", error);
    return NextResponse.json(
      { error: "Failed to register analysis" },
      { status: 500 },
    );
  }
}
