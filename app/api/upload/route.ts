import { NextRequest, NextResponse } from "next/server";
import { ML_BACKEND_URL } from "@/lib/constants";

export async function POST(request: NextRequest) {
  try {
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
    return NextResponse.json(data);
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: "Failed to process upload" },
      { status: 500 },
    );
  }
}
