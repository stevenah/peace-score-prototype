import { NextResponse } from "next/server";

export async function GET() {
  // WebSocket upgrade is handled differently in Next.js
  // For the prototype, the client connects directly to the ML backend WebSocket
  // This route serves as a fallback / health check
  return NextResponse.json({
    message: "Live feed WebSocket is available at the ML backend directly",
    ws_url: `ws://localhost:8000/api/v1/ws/live`,
  });
}
