import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import bcrypt from "bcryptjs";

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

  if (body.name !== undefined) {
    updateData.name = body.name || null;
  }

  if (body.email !== undefined) {
    if (!body.email || typeof body.email !== "string") {
      return NextResponse.json({ error: "Invalid email" }, { status: 400 });
    }
    const existing = await prisma.user.findUnique({
      where: { email: body.email },
    });
    if (existing && existing.id !== id) {
      return NextResponse.json(
        { error: "Email already in use" },
        { status: 409 },
      );
    }
    updateData.email = body.email;
  }

  if (body.password !== undefined) {
    if (typeof body.password !== "string" || body.password.length < 6) {
      return NextResponse.json(
        { error: "Password must be at least 6 characters" },
        { status: 400 },
      );
    }
    updateData.hashedPassword = await bcrypt.hash(body.password, 10);
  }

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

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user || session.user.role !== "ADMIN") {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const { id } = await params;

  if (id === session.user.id) {
    return NextResponse.json(
      { error: "Cannot delete your own account" },
      { status: 400 },
    );
  }

  const user = await prisma.user.findUnique({ where: { id } });
  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  await prisma.user.delete({ where: { id } });

  return NextResponse.json({ success: true });
}
