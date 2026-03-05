import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";

function getWsUrlForBrowser(): string {
  // ML_BACKEND_PUBLIC_URL is the browser-reachable address (e.g. http://localhost:8000).
  // ML_BACKEND_URL is the Docker-internal address and must NOT be sent to the browser.
  const publicUrl =
    process.env.ML_BACKEND_PUBLIC_URL || "http://localhost:8000";
  const wsUrl = publicUrl
    .replace(/^https:\/\//, "wss://")
    .replace(/^http:\/\//, "ws://");
  return `${wsUrl}/api/v1/ws/live`;
}

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return NextResponse.json({
    ws_url: getWsUrlForBrowser(),
  });
}
