import { createHmac } from "crypto";

const ML_BACKEND_URL =
  process.env.ML_BACKEND_URL || "http://localhost:8000";
/**
 * Optional second backend that is tried first: the GPU box, reached over the
 * Fly private network (README, "Using the GPU box for compute"). Unset, every
 * request goes to ML_BACKEND_URL and nothing below applies.
 */
const ML_BACKEND_GPU_URL = process.env.ML_BACKEND_GPU_URL || "";
// Shared with the ML backend, which reads it as PEACE_API_KEY. Empty leaves
// both sides open, which is the local-dev default.
const ML_API_KEY = process.env.ML_API_KEY || "";

// Long enough to outlast a cold start of the ML machine, which the socket's
// handshake waits through.
const LIVE_TOKEN_TTL_SECONDS = 300;

/**
 * How long a request to the GPU box may take before it counts as down. The
 * analysis endpoints answer at once and do the work in the backend's worker,
 * so this is a reachability limit, not a compute budget. The box is usually
 * off, so this is also the most a visitor waits once a minute.
 */
const GPU_TIMEOUT_MS = 2500;
/** After a failure, how long requests skip the GPU box before trying it again. */
const GPU_RETRY_AFTER_MS = 60_000;
let gpuDownUntil = 0;

function withKey(init?: RequestInit): RequestInit {
  const headers = new Headers(init?.headers);
  if (ML_API_KEY) headers.set("x-api-key", ML_API_KEY);
  return { ...init, headers };
}

/** The GPU box's answer, or null when it is off, down, or resting after a failure. */
async function tryGpuBackend(path: string, init: RequestInit): Promise<Response | null> {
  if (!ML_BACKEND_GPU_URL || Date.now() < gpuDownUntil) return null;

  const timeout = AbortSignal.timeout(GPU_TIMEOUT_MS);
  const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
  try {
    const res = await fetch(`${ML_BACKEND_GPU_URL}${path}`, { ...init, signal });
    if (res.status < 500) return res;
    console.warn(`GPU backend answered ${res.status}; using ${ML_BACKEND_URL} for ${GPU_RETRY_AFTER_MS / 1000}s`);
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    console.warn(`GPU backend unreachable (${reason}); using ${ML_BACKEND_URL} for ${GPU_RETRY_AFTER_MS / 1000}s`);
  }
  gpuDownUntil = Date.now() + GPU_RETRY_AFTER_MS;
  return null;
}

/**
 * Fetch from the ML backend, authenticating with the shared key when one is
 * set. With a GPU backend configured it goes there first and falls back to the
 * Fly machine when the box is unreachable. A job is stored by the backend that
 * ran it, so a 404 from one backend is answered by asking the other.
 */
export async function fetchMlBackend(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const request = withKey(init);
  const fromGpu = await tryGpuBackend(path, request);
  if (fromGpu && fromGpu.status !== 404) return fromGpu;

  const fromFly = await fetch(`${ML_BACKEND_URL}${path}`, request);
  return fromGpu && fromFly.status === 404 ? fromGpu : fromFly;
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
