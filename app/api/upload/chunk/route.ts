// Chunks are sent directly to the ML backend, not through Next.js.
// This file exists only to return a helpful error if an old client hits it.
import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json(
    { error: "Chunks should be sent directly to the ML backend" },
    { status: 410 },
  );
}
