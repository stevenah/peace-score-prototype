import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";

export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user || session.user.role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { id } = await params;
  const body = await request.json();

  const updateData: Record<string, unknown> = {};

  if (body.role !== undefined) {
    if (body.role !== "USER" && body.role !== "ADMIN") {
      return NextResponse.json({ error: "Invalid role" }, { status: 400 });
    }
    if (id === session.user.id && body.role !== "ADMIN") {
      return NextResponse.json(
        { error: "Cannot change your own role" },
        { status: 400 },
      );
    }
    updateData.role = body.role;
  }

  if (body.uploadLimit !== undefined) {
    if (typeof body.uploadLimit !== "number") {
      return NextResponse.json(
        { error: "Invalid uploadLimit" },
        { status: 400 },
      );
    }
    updateData.uploadLimit = body.uploadLimit;
  }

  if (body.uploadCount !== undefined) {
    if (typeof body.uploadCount !== "number") {
      return NextResponse.json(
        { error: "Invalid uploadCount" },
        { status: 400 },
      );
    }
    updateData.uploadCount = body.uploadCount;
  }

  if (Object.keys(updateData).length === 0) {
    return NextResponse.json(
      { error: "No fields to update" },
      { status: 400 },
    );
  }

  const user = await prisma.user.update({
    where: { id },
    data: updateData,
    select: {
      id: true,
      name: true,
      email: true,
      role: true,
      uploadLimit: true,
      uploadCount: true,
    },
  });

  return NextResponse.json({ user });
}
