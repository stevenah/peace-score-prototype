import { NextResponse } from "next/server";

function getWsUrlFromBackend(): string {
  const backendUrl = process.env.ML_BACKEND_URL || "http://localhost:8000";
  // Convert http(s) to ws(s)
  const wsUrl = backendUrl
    .replace(/^https:\/\//, "wss://")
    .replace(/^http:\/\//, "ws://");
  return `${wsUrl}/api/v1/ws/live`;
}

export async function GET() {
  return NextResponse.json({
    ws_url: getWsUrlFromBackend(),
  });
}
