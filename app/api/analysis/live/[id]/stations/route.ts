import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { StationsPatchSchema, parseStationsData } from "@/lib/live/wire";
import type { StationsSummary } from "@/lib/types";

/**
 * Station overrides made after a live analysis was saved.
 *
 * The save fires when the video ends — the same moment the checklist turns amber
 * and the clinician starts ticking — so without this every late override would
 * be lost. Only the clinician-owned fields change; the model's record does not.
 */
export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const authSession = await auth();
    if (!authSession?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    let body: unknown;
    try {
      body = await request.json();
    } catch {
      return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
    }
    const patch = StationsPatchSchema.safeParse(body);
    if (!patch.success) {
      return NextResponse.json(
        { error: "Invalid stations patch" },
        { status: 400 },
      );
    }

    const { id } = await params;
    const session = await prisma.analysisSession.findUnique({
      where: { analysisId: id },
      select: { userId: true, stationsData: true },
    });

    if (!session) {
      return NextResponse.json({ error: "Not found" }, { status: 404 });
    }

    if (session.userId !== authSession.user.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const current = parseStationsData(session.stationsData);
    if (!current) {
      return NextResponse.json(
        { error: "No station data for this analysis" },
        { status: 409 },
      );
    }

    const next: StationsSummary = {
      ...current,
      manual: patch.data.manual,
      observed_at_t: patch.data.observed_at_t,
    };
    await prisma.analysisSession.update({
      where: { analysisId: id },
      data: { stationsData: JSON.stringify(next) },
    });

    return NextResponse.json({ stations: next });
  } catch (error) {
    console.error("Live stations update error:", error);
    return NextResponse.json(
      { error: "Failed to update stations" },
      { status: 500 },
    );
  }
}
