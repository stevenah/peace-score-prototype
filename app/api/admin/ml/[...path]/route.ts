import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { logAudit, type AuditAction } from "@/lib/audit";
import { fetchMlBackend } from "@/lib/ml-backend";

/**
 * Admin-only pass-through to the ML backend's model admin API:
 * /api/admin/ml/models/... and /api/admin/ml/evals/... become
 * /api/v1/models/... and /api/v1/evals/... there, with the shared key added.
 * Nothing else on the backend is reachable through here.
 */
const ALLOWED_ROOTS = new Set(["models", "evals"]);
const SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

function errorText(status: number, data: unknown): string {
  const detail = (data as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc, msg, ...}]
    return detail
      .map((d) => (d as { msg?: string }).msg)
      .filter(Boolean)
      .join("; ");
  }
  return `The ML backend answered ${status}`;
}

/** The audit entry a successful change earns, if any. */
function auditFor(
  method: string,
  path: string[],
  body: Record<string, unknown> | null,
): { action: AuditAction; targetType: string; label: string; details?: Record<string, unknown> } | null {
  const route = `${method} ${path.join("/")}`;
  if (route === "POST models/active")
    return { action: "MODEL_ACTIVATED", targetType: "model", label: String(body?.version) };
  if (route === "DELETE models/active")
    return { action: "MODEL_ACTIVATED", targetType: "model", label: "the configured default" };
  if (route === "PUT models/mode")
    return { action: "SERVING_MODE_CHANGED", targetType: "model", label: String(body?.mode) };
  if (route === "POST evals")
    return {
      action: "EVAL_STARTED",
      targetType: "eval",
      label: `${body?.kind} on ${body?.target}`,
      details: (body?.params as Record<string, unknown>) ?? undefined,
    };
  return null;
}

async function forward(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> },
) {
  const session = await auth();
  if (!session?.user || session.user.role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { path } = await params;
  if (!ALLOWED_ROOTS.has(path[0]) || !path.every((s) => SEGMENT.test(s))) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  const method = request.method;
  const init: RequestInit = { method, cache: "no-store" };
  let body: Record<string, unknown> | null = null;
  if (method === "POST" || method === "PUT") {
    const text = await request.text();
    if (text) {
      try {
        body = JSON.parse(text);
      } catch {
        return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
      }
      init.body = text;
      init.headers = { "content-type": "application/json" };
    }
  }

  let res: Response;
  try {
    res = await fetchMlBackend(
      `/api/v1/${path.join("/")}${new URL(request.url).search}`,
      init,
    );
  } catch {
    return NextResponse.json(
      { error: "The ML backend is not reachable." },
      { status: 502 },
    );
  }

  const data = await res.json().catch(() => null);
  if (res.status === 401) {
    // Our own key was refused: a deployment fault, not the visitor's.
    return NextResponse.json(
      { error: "The ML backend refused the web app's key (ML_API_KEY and PEACE_API_KEY differ)." },
      { status: 502 },
    );
  }
  if (!res.ok) {
    return NextResponse.json(
      { error: errorText(res.status, data) },
      { status: res.status },
    );
  }

  const entry = auditFor(method, path, body);
  if (entry) {
    logAudit({
      actorId: session.user.id ?? null,
      actorEmail: session.user.email ?? null,
      action: entry.action,
      targetType: entry.targetType,
      targetLabel: entry.label,
      details: entry.details ?? null,
    });
  }
  return NextResponse.json(data, { status: res.status });
}

export { forward as GET, forward as POST, forward as PUT, forward as DELETE };
