import { NextRequest, NextResponse } from "next/server";
import { ML_BACKEND_URL } from "@/lib/constants";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { uploadVideo } from "@/lib/s3";

const MIME_TYPES: Record<string, string> = {
  ".mp4": "video/mp4",
  ".mov": "video/quicktime",
  ".avi": "video/x-msvideo",
  ".mkv": "video/x-matroska",
  ".webm": "video/webm",
};

const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB

// Video magic bytes: ftyp (MP4/MOV), RIFF (AVI), 1A 45 DF A3 (MKV/WebM)
function isValidVideoFile(header: Uint8Array): boolean {
  // MP4/MOV: "ftyp" at offset 4
  if (header.length >= 8 && header[4] === 0x66 && header[5] === 0x74 && header[6] === 0x79 && header[7] === 0x70) return true;
  // AVI: "RIFF" header
  if (header.length >= 4 && header[0] === 0x52 && header[1] === 0x49 && header[2] === 0x46 && header[3] === 0x46) return true;
  // MKV/WebM: EBML header
  if (header.length >= 4 && header[0] === 0x1a && header[1] === 0x45 && header[2] === 0xdf && header[3] === 0xa3) return true;
  return false;
}

export async function POST(request: NextRequest) {
  let userId: string | undefined;
  let filename = "unknown";

  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }
    userId = session.user.id;

    // Parse form data first so we have the filename for error records
    let formData: FormData;
    try {
      formData = await request.formData();
    } catch {
      await createFailedRecord(userId, filename);
      return NextResponse.json(
        { error: "Failed to parse upload body. File may be too large." },
        { status: 400 },
      );
    }

    const file = formData.get("file");

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    filename = file instanceof File ? file.name : "upload.mp4";

    // Quota check — done after form parse so we have the filename for
    // the failed record, but before any heavy processing.
    const currentUser = await prisma.user.findUnique({
      where: { id: userId },
      select: { uploadLimit: true, uploadCount: true },
    });

    if (!currentUser) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    if (
      currentUser.uploadLimit !== -1 &&
      currentUser.uploadCount >= currentUser.uploadLimit
    ) {
      await createFailedRecord(userId, filename);
      return NextResponse.json(
        {
          error: "Upload limit reached",
          message: `You have reached your upload limit of ${currentUser.uploadLimit} analyses. Contact an administrator to increase your quota.`,
          quotaExceeded: true,
        },
        { status: 429 },
      );
    }

    // Reserve a slot atomically: only increment if still under the limit.
    // The WHERE clause ensures concurrent requests can't exceed the limit.
    let reserved = false;
    if (currentUser.uploadLimit === -1) {
      // Unlimited — increment without a cap check
      await prisma.user.update({
        where: { id: userId },
        data: { uploadCount: { increment: 1 } },
      });
      reserved = true;
    } else {
      const result = await prisma.$executeRaw`
        UPDATE "User"
        SET "uploadCount" = "uploadCount" + 1
        WHERE "id" = ${userId}
          AND "uploadCount" < "uploadLimit"
      `;
      reserved = result > 0;
    }

    if (!reserved) {
      await createFailedRecord(userId, filename);
      return NextResponse.json(
        {
          error: "Upload limit reached",
          message: `You have reached your upload limit of ${currentUser.uploadLimit} analyses. Contact an administrator to increase your quota.`,
          quotaExceeded: true,
        },
        { status: 429 },
      );
    }

    if (file.size > MAX_FILE_SIZE) {
      await releaseSlot(userId);
      await createFailedRecord(userId, filename);
      return NextResponse.json({ error: "File too large (max 500MB)" }, { status: 413 });
    }

    // Validate file content via magic bytes
    const headerBytes = new Uint8Array(await file.slice(0, 12).arrayBuffer());
    if (!isValidVideoFile(headerBytes)) {
      await releaseSlot(userId);
      await createFailedRecord(userId, filename);
      return NextResponse.json({ error: "Invalid video file format" }, { status: 400 });
    }

    // Forward to ML backend
    const mlFormData = new FormData();
    mlFormData.append("file", file);

    const response = await fetch(`${ML_BACKEND_URL}/api/v1/analyze/video`, {
      method: "POST",
      body: mlFormData,
    });

    if (!response.ok) {
      const text = await response.text();
      await releaseSlot(userId);
      await createFailedRecord(userId, filename);
      return NextResponse.json(
        { error: `ML backend error: ${text}` },
        { status: response.status },
      );
    }

    const data = await response.json();

    // Upload video to S3 so it can be played back in results
    let videoPath: string | null = null;
    let videoStorageFailed = false;
    if (data.analysis_id) {
      try {
        const ext = filename.match(/\.\w+$/)?.[0] || ".mp4";
        const key = `videos/${data.analysis_id}${ext}`;
        const buffer = Buffer.from(await file.arrayBuffer());
        const contentType = MIME_TYPES[ext] || "video/mp4";
        await uploadVideo(key, buffer, contentType);
        videoPath = key;
      } catch (e) {
        console.error("Failed to upload video to S3:", e);
        videoStorageFailed = true;
      }
    }

    // Save analysis session
    if (data.analysis_id) {
      await prisma.analysisSession.create({
        data: {
          userId: session.user.id,
          analysisId: data.analysis_id,
          filename,
          status: "processing",
          videoPath,
        },
      });
    }

    return NextResponse.json({ ...data, videoStorageFailed });
  } catch (error) {
    console.error("Upload error:", error);
    if (userId) {
      await releaseSlot(userId).catch(() => {});
      await createFailedRecord(userId, filename).catch(() => {});
    }
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
