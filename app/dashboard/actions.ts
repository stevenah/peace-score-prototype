"use server";

import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";
import { revalidatePath } from "next/cache";

export async function deleteAnalysis(formData: FormData) {
  const session = await auth();
  if (!session?.user) throw new Error("Unauthorized");

  const id = formData.get("id") as string;
  if (!id) throw new Error("Missing analysis ID");

  await prisma.analysisSession.deleteMany({
    where: { id, userId: session.user.id },
  });

  revalidatePath("/dashboard");
}

export async function deleteAnalysesBulk(ids: string[]) {
  const session = await auth();
  if (!session?.user) throw new Error("Unauthorized");

  if (!ids.length) throw new Error("No IDs provided");

  await prisma.analysisSession.deleteMany({
    where: { id: { in: ids }, userId: session.user.id },
  });

  revalidatePath("/dashboard");
}
