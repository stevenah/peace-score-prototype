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

export async function POST(request: NextRequest) {
  try {
    const session = await auth();

    const formData = await request.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
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
      return NextResponse.json(
        { error: `ML backend error: ${text}` },
        { status: response.status },
      );
    }

    const data = await response.json();

    // Upload video to S3 so it can be played back in results
    let videoPath: string | null = null;
    if (data.analysis_id) {
      try {
        const filename = file instanceof File ? file.name : "upload.mp4";
        const ext = filename.match(/\.\w+$/)?.[0] || ".mp4";
        const key = `videos/${data.analysis_id}${ext}`;
        const buffer = Buffer.from(await file.arrayBuffer());
        const contentType = MIME_TYPES[ext] || "video/mp4";
        await uploadVideo(key, buffer, contentType);
        videoPath = key;
      } catch (e) {
        console.error("Failed to upload video to S3:", e);
      }
    }

    // Save analysis session if user is authenticated
    if (session?.user?.id && data.analysis_id) {
      await prisma.analysisSession.create({
        data: {
          userId: session.user.id,
          analysisId: data.analysis_id,
          filename: file instanceof File ? file.name : "unknown",
          status: "processing",
          videoPath,
        },
      });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: "Failed to process upload" },
      { status: 500 },
    );
  }
}
