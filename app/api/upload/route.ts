import { NextRequest, NextResponse } from "next/server";
import { ML_BACKEND_URL } from "@/lib/constants";
import { auth } from "@/lib/auth";
import { prisma } from "@/lib/db";

export async function POST(request: NextRequest) {
  try {
    const session = await auth();

    const formData = await request.formData();
    const file = formData.get("file");

    if (!file || !(file instanceof Blob)) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    // Forward to ML backend
    const mlFormData = new FormData();
    mlFormData.append("file", file);

    const response = await fetch(`${ML_BACKEND_URL}/api/v1/analyze/video`, {
      method: "POST",
      body: mlFormData,
    });

    if (!response.ok) {
      const text = await response.text();
      return NextResponse.json(
        { error: `ML backend error: ${text}` },
        { status: response.status },
      );
    }

    const data = await response.json();

    // Save analysis session if user is authenticated
    if (session?.user?.id && data.analysis_id) {
      await prisma.analysisSession.create({
        data: {
          userId: session.user.id,
          analysisId: data.analysis_id,
          filename: file instanceof File ? file.name : "unknown",
          status: "processing",
        },
      });
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: "Failed to process upload" },
      { status: 500 },
    );
  }
}
