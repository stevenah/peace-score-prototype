import { NextRequest, NextResponse } from "next/server";
import { createReadStream } from "fs";
import { Readable } from "stream";
import { ML_BACKEND_URL } from "@/lib/constants";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { logAudit } from "@/lib/audit";
import {
  getSession,
  appendChunk,
  getFilePath,
  removeSession,
  waitForFlush,
} from "@/lib/upload-sessions";

export const maxDuration = 300;

export async function POST(request: NextRequest) {
  const uploadId = request.headers.get("x-upload-id");
  const chunkIndex = parseInt(request.headers.get("x-chunk-index") || "0", 10);
  const totalChunks = parseInt(request.headers.get("x-total-chunks") || "1", 10);
  const filename = request.headers.get("x-filename") || "upload.mp4";

  if (!uploadId) {
    return NextResponse.json({ error: "Missing upload ID" }, { status: 400 });
  }

  const session = getSession(uploadId);
  if (!session) {
    return NextResponse.json({ error: "Upload session not found or expired" }, { status: 404 });
  }

  // Verify the caller owns this upload session
  const authSession = await auth();
  if (!authSession?.user?.id || authSession.user.id !== session.userId) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    // Read the chunk body as a buffer (each chunk is ~8MB, well within limits)
    const arrayBuffer = await request.arrayBuffer();
    const chunk = Buffer.from(arrayBuffer);

    const { complete } = appendChunk(uploadId, chunk, chunkIndex, totalChunks);

    if (!complete) {
      return NextResponse.json({ ok: true, chunksReceived: chunkIndex + 1 });
    }

    // All chunks received — stream assembled file to ML backend
    const filePath = getFilePath(uploadId);
    if (!filePath) {
      return NextResponse.json({ error: "Upload file not found" }, { status: 500 });
    }

    await waitForFlush(uploadId);

    // Stream the file to ML backend without buffering into memory
    const fileStream = createReadStream(filePath);
    const webStream = Readable.toWeb(fileStream) as ReadableStream;

    const response = await fetch(
      `${ML_BACKEND_URL}/api/v1/analyze/video/stream`,
      {
        method: "POST",
        body: webStream,
        headers: {
          "content-type": "application/octet-stream",
          "x-filename": filename,
        },
        // @ts-expect-error -- duplex required for streaming request bodies
        duplex: "half",
      },
    );

    // Clean up temp file regardless of outcome
    removeSession(uploadId);

    if (!response.ok) {
      const text = await response.text();
      await releaseSlot(session.userId);
      await createFailedRecord(session.userId, filename);
      return NextResponse.json(
        { error: `ML backend error: ${text}` },
        { status: response.status },
      );
    }

    const data = await response.json();

    if (data.analysis_id) {
      await prisma.analysisSession.create({
        data: {
          userId: session.userId,
          analysisId: data.analysis_id,
          filename,
          status: "processing",
          videoPath: null,
        },
      });

      logAudit({
        actorId: session.userId,
        actorEmail: session.userEmail,
        action: "VIDEO_UPLOADED",
        targetType: "AnalysisSession",
        targetId: data.analysis_id,
        targetLabel: filename,
      });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Chunk upload error:", error);
    removeSession(uploadId);
    await releaseSlot(session.userId).catch(() => {});
    await createFailedRecord(session.userId, filename).catch(() => {});
    return NextResponse.json(
      { error: "Failed to process upload" },
      { status: 500 },
    );
  }
}

async function releaseSlot(userId: string) {
  await prisma.$executeRaw`
    UPDATE "User"
    SET "uploadCount" = GREATEST("uploadCount" - 1, 0)
    WHERE "id" = ${userId}
  `;
}

async function createFailedRecord(userId: string, filename: string) {
  const failId = `fail_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  await prisma.analysisSession.create({
    data: {
      userId,
      analysisId: failId,
      filename,
      status: "failed",
      completedAt: new Date(),
    },
  });
}
