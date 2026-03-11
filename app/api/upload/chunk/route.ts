// Uploads now go directly to S3 via presigned URLs.
// This file exists only to return a helpful error if an old client hits it.
import { NextResponse } from "next/server";

export async function POST() {
  return NextResponse.json(
    { error: "Uploads now go directly to S3. Update your client." },
    { status: 410 },
  );
}
