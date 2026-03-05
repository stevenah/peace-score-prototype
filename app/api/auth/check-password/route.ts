import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/db";

export async function POST(request: NextRequest) {
  const { email } = await request.json();

  if (!email) {
    return NextResponse.json(
      { error: "Email is required" },
      { status: 400 },
    );
  }

  const user = await prisma.user.findUnique({
    where: { email },
    select: { hashedPassword: true },
  });

  if (!user) {
    return NextResponse.json({ needsPassword: false });
  }

  return NextResponse.json({ needsPassword: !user.hashedPassword });
}
