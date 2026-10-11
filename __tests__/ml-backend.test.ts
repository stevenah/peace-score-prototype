import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import tokenContract from "@/contracts/live_token.example.json";

// lib/ml-backend reads its configuration once at import, so each test loads a
// fresh copy under the environment it needs.
async function load(env: Record<string, string>) {
  vi.resetModules();
  for (const [name, value] of Object.entries(env)) vi.stubEnv(name, value);
  return import("@/lib/ml-backend");
}

function lastFetchCall() {
  const [url, init] = vi.mocked(fetch).mock.calls.at(-1)!;
  return { url, init, headers: new Headers(init?.headers) };
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("fetchMlBackend", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}")));
  });

  it("sends the shared key to the configured backend", async () => {
    const { fetchMlBackend } = await load({
      ML_BACKEND_URL: "http://ml:8000",
      ML_API_KEY: "test-key",
    });

    await fetchMlBackend("/api/v1/analyze/abc");

    const { url, headers } = lastFetchCall();
    expect(url).toBe("http://ml:8000/api/v1/analyze/abc");
    expect(headers.get("x-api-key")).toBe("test-key");
  });

  it("keeps the caller's method, body and headers", async () => {
    const { fetchMlBackend } = await load({ ML_API_KEY: "test-key" });

    await fetchMlBackend("/api/v1/analyze/s3", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: '{"s3_key":"uploads/x.mp4"}',
    });

    const { init, headers } = lastFetchCall();
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe('{"s3_key":"uploads/x.mp4"}');
    expect(headers.get("content-type")).toBe("application/json");
    expect(headers.get("x-api-key")).toBe("test-key");
  });

  it("sends no key when none is configured", async () => {
    const { fetchMlBackend } = await load({ ML_API_KEY: "" });

    await fetchMlBackend("/api/v1/health");

    expect(lastFetchCall().headers.has("x-api-key")).toBe(false);
  });
});

describe("signLiveToken", () => {
  it("is null when no key is configured", async () => {
    const { signLiveToken } = await load({ ML_API_KEY: "" });
    expect(signLiveToken()).toBeNull();
  });

  it("produces the token in contracts/live_token.example.json", async () => {
    const { signLiveToken } = await load({ ML_API_KEY: tokenContract.key });
    // The contract's expiry is five minutes after this instant.
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime((tokenContract.expires - 300) * 1000);

    expect(signLiveToken()).toBe(tokenContract.token);
  });
});
