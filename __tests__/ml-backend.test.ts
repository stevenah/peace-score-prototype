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

describe("fetchMlBackend with a GPU backend", () => {
  const env = {
    ML_BACKEND_URL: "http://fly:8000",
    ML_BACKEND_GPU_URL: "http://[fdaa::1]:8000",
    ML_API_KEY: "test-key",
  };

  function urls() {
    return vi.mocked(fetch).mock.calls.map(([url]) => String(url));
  }

  it("asks the GPU backend first", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("{}")));
    const { fetchMlBackend } = await load(env);

    const res = await fetchMlBackend("/api/v1/analyze/abc");

    expect(res.status).toBe(200);
    expect(urls()).toEqual(["http://[fdaa::1]:8000/api/v1/analyze/abc"]);
    expect(lastFetchCall().headers.get("x-api-key")).toBe("test-key");
  });

  it("falls back to Fly when the box is unreachable, and rests it for a while", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockRejectedValueOnce(new TypeError("fetch failed"))
        .mockResolvedValue(new Response("{}")),
    );
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(1_000_000);
    const { fetchMlBackend } = await load(env);

    await fetchMlBackend("/api/v1/health");
    await fetchMlBackend("/api/v1/health");
    expect(urls()).toEqual([
      "http://[fdaa::1]:8000/api/v1/health",
      "http://fly:8000/api/v1/health",
      "http://fly:8000/api/v1/health",
    ]);

    vi.setSystemTime(1_000_000 + 61_000);
    await fetchMlBackend("/api/v1/health");
    expect(urls().at(-1)).toBe("http://[fdaa::1]:8000/api/v1/health");
  });

  it("treats a 5xx from the box like an outage", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(new Response("down", { status: 503 }))
        .mockResolvedValue(new Response("{}")),
    );
    const { fetchMlBackend } = await load(env);

    const res = await fetchMlBackend("/api/v1/analyze/abc");

    expect(res.status).toBe(200);
    expect(urls()).toEqual([
      "http://[fdaa::1]:8000/api/v1/analyze/abc",
      "http://fly:8000/api/v1/analyze/abc",
    ]);
  });

  it("asks the other backend on a 404, since a job lives where it ran", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(new Response("no", { status: 404 }))
        .mockResolvedValueOnce(new Response("{}"))
        .mockResolvedValue(new Response("{}")),
    );
    const { fetchMlBackend } = await load(env);

    const res = await fetchMlBackend("/api/v1/analyze/live_1");
    expect(res.status).toBe(200);
    expect(urls()).toEqual([
      "http://[fdaa::1]:8000/api/v1/analyze/live_1",
      "http://fly:8000/api/v1/analyze/live_1",
    ]);

    // A 404 is an answer, not an outage: the box is still asked first.
    await fetchMlBackend("/api/v1/health");
    expect(urls().at(-1)).toBe("http://[fdaa::1]:8000/api/v1/health");
  });

  it("returns the 404 when neither backend knows the id", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("no", { status: 404 })));
    const { fetchMlBackend } = await load(env);

    const res = await fetchMlBackend("/api/v1/analyze/nope");

    expect(res.status).toBe(404);
    expect(urls()).toHaveLength(2);
  });
});
