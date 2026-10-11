import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { signLiveToken } from "@/lib/ml-backend";

function getWsUrlForBrowser(): string {
  // ML_BACKEND_PUBLIC_URL is the browser-reachable http(s) origin of the ML
  // backend. NEXT_PUBLIC_WS_URL, when set, overrides it with a fully-formed
  // WebSocket URL.
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

  const wsUrl = getWsUrlForBrowser();
  const token = signLiveToken();

  return NextResponse.json({
    ws_url: token ? `${wsUrl}?token=${token}` : wsUrl,
  });
}
