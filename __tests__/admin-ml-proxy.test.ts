import { beforeEach, describe, expect, it, vi } from "vitest";
import type { NextRequest } from "next/server";

vi.mock("@/lib/auth", () => ({ auth: vi.fn() }));
vi.mock("@/lib/audit", () => ({ logAudit: vi.fn() }));
vi.mock("@/lib/ml-backend", () => ({ fetchMlBackend: vi.fn() }));

import { auth } from "@/lib/auth";
import { logAudit } from "@/lib/audit";
import { fetchMlBackend } from "@/lib/ml-backend";
import { DELETE, GET, POST, PUT } from "@/app/api/admin/ml/[...path]/route";

const mockAuth = auth as unknown as ReturnType<typeof vi.fn>;
const backend = vi.mocked(fetchMlBackend);
const audit = vi.mocked(logAudit);

const ADMIN = { user: { id: "u1", email: "admin@example.com", role: "ADMIN" } };

function request(method: string, path: string, body?: unknown): NextRequest {
  return new Request(`http://test/api/admin/ml/${path}`, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : typeof body === "string" ? body : JSON.stringify(body),
  }) as unknown as NextRequest;
}

function params(path: string) {
  return { params: Promise.resolve({ path: path.split("?")[0].split("/") }) };
}

function answer(status: number, data: unknown) {
  backend.mockResolvedValueOnce(
    new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } }),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockAuth.mockResolvedValue(ADMIN);
});

describe("access", () => {
  it.each([null, { user: { id: "u2", role: "USER" } }])("refuses anyone but an admin", async (session) => {
    mockAuth.mockResolvedValue(session);
    const res = await GET(request("GET", "models"), params("models"));
    expect(res.status).toBe(403);
    expect(backend).not.toHaveBeenCalled();
  });

  it.each(["analyze/s3", "health", "ws/live", "models/../analyze/s3", "evals/.hidden", "models/a b"])(
    "reaches nothing but the model admin API: %s",
    async (path) => {
      const res = await GET(request("GET", "models"), { params: Promise.resolve({ path: path.split("/") }) });
      expect(res.status).toBe(404);
      expect(backend).not.toHaveBeenCalled();
    },
  );
});

describe("forwarding", () => {
  it("passes a read through with its query string", async () => {
    answer(200, { jobs: [] });
    const res = await GET(request("GET", "evals?limit=30"), params("evals"));

    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ jobs: [] });
    const [path, init] = backend.mock.calls[0];
    expect(path).toBe("/api/v1/evals?limit=30");
    expect(init?.method).toBe("GET");
    expect(init?.body).toBeUndefined();
    expect(audit).not.toHaveBeenCalled();
  });

  it("passes a job id through", async () => {
    answer(200, { id: "20261011T200501-abc123" });
    await GET(request("GET", "evals/20261011T200501-abc123"), params("evals/20261011T200501-abc123"));
    expect(backend.mock.calls[0][0]).toBe("/api/v1/evals/20261011T200501-abc123");
  });

  it("sends the body of a change and keeps the backend's status", async () => {
    answer(201, { id: "job1", status: "queued" });
    const body = { kind: "latency", target: "lm-1.0.0", params: { mode: "screen", frames: 60 } };
    const res = await POST(request("POST", "evals", body), params("evals"));

    expect(res.status).toBe(201);
    const [path, init] = backend.mock.calls[0];
    expect(path).toBe("/api/v1/evals");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual(body);
    expect(new Headers(init?.headers).get("content-type")).toBe("application/json");
  });

  it("refuses a body that is not JSON", async () => {
    const res = await POST(request("POST", "evals", "{nope"), params("evals"));
    expect(res.status).toBe(400);
    expect(backend).not.toHaveBeenCalled();
  });
});

describe("errors", () => {
  it("turns the backend's refusal into an error message", async () => {
    answer(409, { detail: "This bundle cannot be loaded: layouts_version 'lay-2' != 'lay-3'" });
    const res = await POST(request("POST", "models/active", { version: "lm-0.0.3" }), params("models/active"));
    expect(res.status).toBe(409);
    expect(await res.json()).toEqual({
      error: "This bundle cannot be loaded: layouts_version 'lay-2' != 'lay-3'",
    });
    expect(audit).not.toHaveBeenCalled();
  });

  it("joins FastAPI validation errors", async () => {
    answer(422, { detail: [{ loc: ["body", "mode"], msg: "Input should be 'off', 'shadow' or 'display'" }] });
    const res = await PUT(request("PUT", "models/mode", { mode: "loud" }), params("models/mode"));
    expect(await res.json()).toEqual({ error: "Input should be 'off', 'shadow' or 'display'" });
  });

  it("reports a refused key as a deployment fault, not as the visitor's", async () => {
    answer(401, { detail: "Invalid or missing API key" });
    const res = await GET(request("GET", "models"), params("models"));
    expect(res.status).toBe(502);
    expect((await res.json()).error).toContain("ML_API_KEY");
  });

  it("says so when the backend is down", async () => {
    backend.mockRejectedValueOnce(new TypeError("fetch failed"));
    const res = await GET(request("GET", "models"), params("models"));
    expect(res.status).toBe(502);
    expect(await res.json()).toEqual({ error: "The ML backend is not reachable." });
  });
});

describe("audit log", () => {
  const actor = { actorId: "u1", actorEmail: "admin@example.com" };

  it("records a model switch", async () => {
    answer(200, { loaded_version: "lm-1.0.0" });
    await POST(request("POST", "models/active", { version: "lm-1.0.0" }), params("models/active"));
    expect(audit).toHaveBeenCalledWith(
      expect.objectContaining({ ...actor, action: "MODEL_ACTIVATED", targetType: "model", targetLabel: "lm-1.0.0" }),
    );
  });

  it("records a return to the default", async () => {
    answer(200, { loaded_version: null });
    await DELETE(request("DELETE", "models/active"), params("models/active"));
    expect(audit).toHaveBeenCalledWith(
      expect.objectContaining({ action: "MODEL_ACTIVATED", targetLabel: "the configured default" }),
    );
    expect(backend.mock.calls[0][1]?.body).toBeUndefined();
  });

  it("records a serving-mode change", async () => {
    answer(200, { mode: "display" });
    await PUT(request("PUT", "models/mode", { mode: "display" }), params("models/mode"));
    expect(audit).toHaveBeenCalledWith(
      expect.objectContaining({ action: "SERVING_MODE_CHANGED", targetLabel: "display" }),
    );
  });

  it("records a started eval with its parameters", async () => {
    answer(201, { id: "job1" });
    await POST(
      request("POST", "evals", { kind: "accuracy", target: "lm-1.0.0", params: { split: "shift" } }),
      params("evals"),
    );
    expect(audit).toHaveBeenCalledWith(
      expect.objectContaining({
        action: "EVAL_STARTED",
        targetType: "eval",
        targetLabel: "accuracy on lm-1.0.0",
        details: { split: "shift" },
      }),
    );
  });

  it("does not record a cancel", async () => {
    answer(200, { id: "job1", status: "cancelled" });
    await POST(request("POST", "evals/job1/cancel"), params("evals/job1/cancel"));
    expect(audit).not.toHaveBeenCalled();
  });
});
