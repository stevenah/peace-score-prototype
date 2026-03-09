import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { registerUpload } from "@/lib/upload-sessions";

export async function POST() {
  try {
    const session = await auth();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const userId = session.user.id;

    const currentUser = await prisma.user.findUnique({
      where: { id: userId },
      select: { uploadLimit: true, uploadCount: true },
    });

    if (!currentUser) {
      return NextResponse.json({ error: "User not found" }, { status: 404 });
    }

    if (
      currentUser.uploadLimit !== -1 &&
      currentUser.uploadCount >= currentUser.uploadLimit
    ) {
      return NextResponse.json(
        {
          error: "Upload limit reached",
          message: `You have reached your upload limit of ${currentUser.uploadLimit} analyses. Contact an administrator to increase your quota.`,
          quotaExceeded: true,
        },
        { status: 429 },
      );
    }

    // Reserve a slot atomically
    let reserved = false;
    if (currentUser.uploadLimit === -1) {
      await prisma.user.update({
        where: { id: userId },
        data: { uploadCount: { increment: 1 } },
      });
      reserved = true;
    } else {
      const result = await prisma.$executeRaw`
        UPDATE "User"
        SET "uploadCount" = "uploadCount" + 1
        WHERE "id" = ${userId}
          AND "uploadCount" < "uploadLimit"
      `;
      reserved = result > 0;
    }

    if (!reserved) {
      return NextResponse.json(
        {
          error: "Upload limit reached",
          message: `You have reached your upload limit of ${currentUser.uploadLimit} analyses. Contact an administrator to increase your quota.`,
          quotaExceeded: true,
        },
        { status: 429 },
      );
    }

    const uploadId = registerUpload(userId, session.user.email ?? null);

    return NextResponse.json({ uploadId });
  } catch (error) {
    console.error("Upload init error:", error);
    return NextResponse.json(
      { error: "Failed to initialize upload" },
      { status: 500 },
    );
  }
}
