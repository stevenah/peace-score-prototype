import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { writeFile, mkdir } from "fs/promises";
import path from "path";

const UPLOADS_DIR = path.join(process.cwd(), "uploads", "videos");

export async function POST(request: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const formData = await request.formData();
    const metadataRaw = formData.get("metadata");
    const videoFile = formData.get("video");

    if (!metadataRaw || typeof metadataRaw !== "string") {
      return NextResponse.json(
        { error: "Missing metadata" },
        { status: 400 },
      );
    }

    const { filename, overallScore, framesAnalyzed, duration, timeline } =
      JSON.parse(metadataRaw);

    if (!filename || typeof framesAnalyzed !== "number") {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 },
      );
    }

    const analysisId = `live_${crypto.randomUUID()}`;
    let videoPath: string | null = null;

    if (videoFile && videoFile instanceof Blob) {
      await mkdir(UPLOADS_DIR, { recursive: true });
      const ext = (filename as string).match(/\.\w+$/)?.[0] || ".mp4";
      const safeFilename = `${analysisId}${ext}`;
      const filePath = path.join(UPLOADS_DIR, safeFilename);
      const buffer = Buffer.from(await videoFile.arrayBuffer());
      await writeFile(filePath, buffer);
      videoPath = `uploads/videos/${safeFilename}`;
    }

    const record = await prisma.analysisSession.create({
      data: {
        userId: session.user.id,
        analysisId,
        filename,
        status: "completed",
        overallScore: typeof overallScore === "number" ? overallScore : null,
        framesAnalyzed,
        duration: typeof duration === "number" ? duration : null,
        timelineData: Array.isArray(timeline)
          ? JSON.stringify(timeline)
          : null,
        videoPath,
        completedAt: new Date(),
      },
    });

    return NextResponse.json({ id: record.id, analysisId });
  } catch (error) {
    console.error("Save live analysis error:", error);
    return NextResponse.json(
      { error: "Failed to save analysis" },
      { status: 500 },
    );
  }
}
