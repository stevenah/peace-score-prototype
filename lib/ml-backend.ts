const ML_BACKEND_URL =
  process.env.ML_BACKEND_URL || "http://localhost:8000";
const ML_BACKEND_FALLBACK_URL = process.env.ML_BACKEND_FALLBACK_URL || "";

/**
 * Fetch from the ML backend with automatic failover to the fallback URL.
 * Tries the primary URL first; if it fails (network error or 5xx), retries
 * against the fallback URL when one is configured.
 */
export async function fetchMlBackend(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const urls = [ML_BACKEND_URL];
  if (ML_BACKEND_FALLBACK_URL) {
    urls.push(ML_BACKEND_FALLBACK_URL);
  }

  let lastError: unknown;

  for (const baseUrl of urls) {
    try {
      const res = await fetch(`${baseUrl}${path}`, init);
      // Retry on server errors (5xx) if we have a fallback
      if (res.status >= 500 && baseUrl !== urls[urls.length - 1]) {
        lastError = new Error(`ML backend returned ${res.status}`);
        console.warn(
          `ML backend ${baseUrl} returned ${res.status}, trying fallback...`,
        );
        continue;
      }
      return res;
    } catch (err) {
      lastError = err;
      if (baseUrl !== urls[urls.length - 1]) {
        console.warn(
          `ML backend ${baseUrl} unreachable, trying fallback...`,
          err,
        );
        continue;
      }
    }
  }

  throw lastError;
}
