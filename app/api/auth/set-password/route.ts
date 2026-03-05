import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import bcrypt from "bcryptjs";

export async function POST(request: NextRequest) {
  const { email, password } = await request.json();

  if (!email || !password) {
    return NextResponse.json(
      { error: "Email and password are required" },
      { status: 400 },
    );
  }

  if (password.length < 6) {
    return NextResponse.json(
      { error: "Password must be at least 6 characters" },
      { status: 400 },
    );
  }

  const user = await prisma.user.findUnique({
    where: { email },
    select: { id: true, hashedPassword: true },
  });

  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  if (user.hashedPassword) {
    return NextResponse.json(
      { error: "Password is already set" },
      { status: 400 },
    );
  }

  const hashedPassword = await bcrypt.hash(password, 10);
  await prisma.user.update({
    where: { id: user.id },
    data: { hashedPassword },
  });

  return NextResponse.json({ success: true });
}
