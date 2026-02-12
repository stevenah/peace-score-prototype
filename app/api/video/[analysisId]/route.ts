import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import { stat, open } from "fs/promises";
import { Readable } from "stream";
import path from "path";

export const runtime = "nodejs";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ analysisId: string }> },
) {
  const { analysisId } = await params;

  const session = await prisma.analysisSession.findUnique({
    where: { analysisId },
  });

  if (!session?.videoPath) {
    return NextResponse.json({ error: "Video not found" }, { status: 404 });
  }

  const filePath = path.join(process.cwd(), session.videoPath);

  try {
    const stats = await stat(filePath);
    const fileSize = stats.size;
    const range = request.headers.get("range");

    const ext = path.extname(filePath).toLowerCase();
    const mimeTypes: Record<string, string> = {
      ".mp4": "video/mp4",
      ".mov": "video/quicktime",
      ".avi": "video/x-msvideo",
      ".mkv": "video/x-matroska",
      ".webm": "video/webm",
    };
    const contentType = mimeTypes[ext] || "video/mp4";

    if (range) {
      const parts = range.replace(/bytes=/, "").split("-");
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = end - start + 1;

      const fileHandle = await open(filePath, "r");
      const nodeStream = fileHandle.createReadStream({ start, end });
      const webStream = Readable.toWeb(nodeStream) as ReadableStream;

      return new Response(webStream, {
        status: 206,
        headers: {
          "Content-Range": `bytes ${start}-${end}/${fileSize}`,
          "Accept-Ranges": "bytes",
          "Content-Length": String(chunkSize),
          "Content-Type": contentType,
        },
      });
    }

    const fileHandle = await open(filePath, "r");
    const nodeStream = fileHandle.createReadStream();
    const webStream = Readable.toWeb(nodeStream) as ReadableStream;

    return new Response(webStream, {
      status: 200,
      headers: {
        "Content-Length": String(fileSize),
        "Content-Type": contentType,
        "Accept-Ranges": "bytes",
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Video file not accessible" },
      { status: 404 },
    );
  }
}
