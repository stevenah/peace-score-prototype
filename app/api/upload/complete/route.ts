import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { logAudit } from "@/lib/audit";

export async function POST(request: NextRequest) {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { analysisId, filename } = await request.json();

    if (!analysisId || !filename) {
      return NextResponse.json(
        { error: "Missing analysisId or filename" },
        { status: 400 },
      );
    }

    // Prevent duplicates
    const existing = await prisma.analysisSession.findUnique({
      where: { analysisId },
    });

    if (existing) {
      return NextResponse.json({ ok: true });
    }

    await prisma.analysisSession.create({
      data: {
        userId: session.user.id,
        analysisId,
        filename,
        status: "processing",
        videoPath: null,
      },
    });

    logAudit({
      actorId: session.user.id,
      actorEmail: session.user.email ?? null,
      action: "VIDEO_UPLOADED",
      targetType: "AnalysisSession",
      targetId: analysisId,
      targetLabel: filename,
    });

    return NextResponse.json({ ok: true });
  } catch (error) {
    console.error("Upload complete error:", error);
    return NextResponse.json(
      { error: "Failed to register analysis" },
      { status: 500 },
    );
  }
}
