import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";

function getWsUrlForBrowser(): string {
  // NEXT_PUBLIC_WS_URL is the fully-formed browser-reachable WebSocket URL and
  // takes precedence when set (it is what fly.toml configures in production).
  // ML_BACKEND_PUBLIC_URL is the http(s) origin fallback for local/docker setups.
  // ML_BACKEND_URL is the Docker-internal address and must NOT be sent to the browser.
  const explicitWsUrl = process.env.NEXT_PUBLIC_WS_URL;
  if (explicitWsUrl) return explicitWsUrl;

  const publicUrl = process.env.ML_BACKEND_PUBLIC_URL || "http://localhost:8000";
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
