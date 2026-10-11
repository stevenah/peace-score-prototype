import { createHmac } from "crypto";

const ML_BACKEND_URL =
  process.env.ML_BACKEND_URL || "http://localhost:8000";
// Shared with the ML backend, which reads it as PEACE_API_KEY. Empty leaves
// both sides open, which is the local-dev default.
const ML_API_KEY = process.env.ML_API_KEY || "";

// Long enough to outlast a cold start of the ML machine, which the socket's
// handshake waits through.
const LIVE_TOKEN_TTL_SECONDS = 300;

/**
 * Fetch from the ML backend, authenticating with the shared key when one is set.
 */
export async function fetchMlBackend(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (ML_API_KEY) headers.set("x-api-key", ML_API_KEY);
  return fetch(`${ML_BACKEND_URL}${path}`, { ...init, headers });
}

/**
 * Token for the live WebSocket, which the browser opens straight to the ML
 * backend and cannot attach headers to. Format: `<unix expiry>.<hmac of it>`.
 * Null when no key is configured.
 */
export function signLiveToken(): string | null {
  if (!ML_API_KEY) return null;
  const expires = Math.floor(Date.now() / 1000) + LIVE_TOKEN_TTL_SECONDS;
  const signature = createHmac("sha256", ML_API_KEY)
    .update(String(expires))
    .digest("hex");
  return `${expires}.${signature}`;
}
