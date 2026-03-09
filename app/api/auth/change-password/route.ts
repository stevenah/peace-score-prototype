import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";
import bcrypt from "bcryptjs";

export async function POST(request: NextRequest) {
  const { email, currentPassword, newPassword } = await request.json();

  if (!email || !currentPassword || !newPassword) {
    return NextResponse.json(
      { error: "Email, current password, and new password are required" },
      { status: 400 },
    );
  }

  if (newPassword.length < 6) {
    return NextResponse.json(
      { error: "New password must be at least 6 characters" },
      { status: 400 },
    );
  }

  const user = await prisma.user.findUnique({
    where: { email },
    select: { id: true, hashedPassword: true },
  });

  if (!user || !user.hashedPassword) {
    return NextResponse.json({ error: "Invalid request" }, { status: 400 });
  }

  const isValid = await bcrypt.compare(currentPassword, user.hashedPassword);
  if (!isValid) {
    return NextResponse.json(
      { error: "Current password is incorrect" },
      { status: 401 },
    );
  }

  const hashedPassword = await bcrypt.hash(newPassword, 10);
  await prisma.user.update({
    where: { id: user.id },
    data: { hashedPassword, forcePasswordChange: false },
  });

  return NextResponse.json({ success: true });
}
