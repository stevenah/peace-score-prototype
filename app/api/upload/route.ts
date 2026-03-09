// Upload is handled by /api/upload/init and /api/upload/chunk.
// This file is kept to avoid 404s if old clients hit this endpoint.
import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json(
    { error: "Use /api/upload/init and /api/upload/chunk for uploads" },
    { status: 410 },
  );
}
