import { beforeEach, describe, it, expect, vi } from "vitest";
import type { NextRequest } from "next/server";

vi.mock("@/lib/auth", () => ({ auth: vi.fn() }));
vi.mock("@/lib/db", () => ({
  prisma: {
    analysisSession: {
      findUnique: vi.fn(),
      update: vi.fn(),
      create: vi.fn(),
    },
  },
}));

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { PATCH } from "@/app/api/analysis/live/[id]/stations/route";
import { POST as saveLive } from "@/app/api/analysis/save-live/route";
import { GET as getLive } from "@/app/api/analysis/live/[id]/route";

const mockAuth = auth as unknown as ReturnType<typeof vi.fn>;
const db = prisma.analysisSession as unknown as {
  findUnique: ReturnType<typeof vi.fn>;
  update: ReturnType<typeof vi.fn>;
  create: ReturnType<typeof vi.fn>;
};

const SUMMARY = {
  schema: "esge10.v1",
  model_version: "mock-flow-1",
  display: true,
  availability: "ok",
  status: ["observed", ...Array(9).fill("unseen")],
  manual: Array(10).fill(null),
  auto_enabled: Array(10).fill(true),
  observed_at_t: [4, ...Array(9).fill(null)],
};

const PATCH_BODY = {
  manual: ["rejected", null, null, null, null, "confirmed", null, null, null, null],
  observed_at_t: [null, null, null, null, null, 30, null, null, null, null],
};

function patchRequest(body: unknown): NextRequest {
  return new Request("http://test/api/analysis/live/live_1/stations", {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: typeof body === "string" ? body : JSON.stringify(body),
  }) as unknown as NextRequest;
}

const params = { params: Promise.resolve({ id: "live_1" }) };

beforeEach(() => {
  vi.clearAllMocks();
  mockAuth.mockResolvedValue({ user: { id: "u1" } });
});

describe("PATCH /api/analysis/live/[id]/stations", () => {
  it("requires a session", async () => {
    mockAuth.mockResolvedValue(null);
    const res = await PATCH(patchRequest(PATCH_BODY), params);
    expect(res.status).toBe(401);
    expect(db.update).not.toHaveBeenCalled();
  });

  it("rejects a malformed body", async () => {
    expect((await PATCH(patchRequest("{nope"), params)).status).toBe(400);
    expect(
      (await PATCH(patchRequest({ ...PATCH_BODY, manual: ["confirmed"] }), params)).status,
    ).toBe(400);
    expect(db.update).not.toHaveBeenCalled();
  });

  it("404s an unknown analysis and 403s someone else's", async () => {
    db.findUnique.mockResolvedValueOnce(null);
    expect((await PATCH(patchRequest(PATCH_BODY), params)).status).toBe(404);
    db.findUnique.mockResolvedValueOnce({ userId: "u2", stationsData: JSON.stringify(SUMMARY) });
    expect((await PATCH(patchRequest(PATCH_BODY), params)).status).toBe(403);
    expect(db.update).not.toHaveBeenCalled();
  });

  it("409s an analysis saved without station data", async () => {
    db.findUnique.mockResolvedValueOnce({ userId: "u1", stationsData: null });
    expect((await PATCH(patchRequest(PATCH_BODY), params)).status).toBe(409);
  });

  it("updates only the clinician-owned fields", async () => {
    db.findUnique.mockResolvedValueOnce({ userId: "u1", stationsData: JSON.stringify(SUMMARY) });
    const res = await PATCH(patchRequest(PATCH_BODY), params);
    expect(res.status).toBe(200);
    const stored = JSON.parse(db.update.mock.calls[0][0].data.stationsData);
    expect(db.update.mock.calls[0][0].where).toEqual({ analysisId: "live_1" });
    expect(stored).toEqual({ ...SUMMARY, ...PATCH_BODY });
    expect(stored.status).toEqual(SUMMARY.status); // the model's record is untouched
    expect((await res.json()).stations).toEqual(stored);
  });
});

describe("POST /api/analysis/save-live — stationsData", () => {
  function saveRequest(metadata: Record<string, unknown>): NextRequest {
    const form = new FormData();
    form.append("metadata", JSON.stringify(metadata));
    return new Request("http://test/api/analysis/save-live", {
      method: "POST",
      body: form,
    }) as unknown as NextRequest;
  }

  const META = { filename: "run.mp4", framesAnalyzed: 3, timeline: [] };

  beforeEach(() => {
    db.create.mockResolvedValue({ id: "db-1" });
  });

  it("stores the stations summary as JSON", async () => {
    const res = await saveLive(saveRequest({ ...META, stations: SUMMARY }));
    expect(res.status).toBe(200);
    const data = db.create.mock.calls[0][0].data;
    expect(JSON.parse(data.stationsData)).toEqual(SUMMARY);
  });

  it("stores null without a summary (feature off)", async () => {
    await saveLive(saveRequest(META));
    expect(db.create.mock.calls[0][0].data.stationsData).toBeNull();
  });

  it("drops a malformed summary but still saves the analysis", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const res = await saveLive(saveRequest({ ...META, stations: { schema: "esge10.v1" } }));
    expect(res.status).toBe(200);
    expect(db.create.mock.calls[0][0].data.stationsData).toBeNull();
  });
});

describe("GET /api/analysis/live/[id] — stations", () => {
  const row = {
    analysisId: "live_1",
    userId: "u1",
    status: "completed",
    overallScore: 2,
    duration: 10,
    framesAnalyzed: 3,
    timelineData: "[]",
    videoPath: null,
    createdAt: new Date(0),
    completedAt: new Date(0),
  };
  const get = () => getLive(new Request("http://test") as unknown as NextRequest, params);

  it("returns the saved summary parsed", async () => {
    db.findUnique.mockResolvedValueOnce({ ...row, stationsData: JSON.stringify(SUMMARY) });
    const body = await (await get()).json();
    expect(body.stations).toEqual(SUMMARY);
  });

  it("omits stations when none were saved", async () => {
    db.findUnique.mockResolvedValueOnce({ ...row, stationsData: null });
    const body = await (await get()).json();
    expect("stations" in body).toBe(false);
    expect(body.analysis_id).toBe("live_1");
  });
});
