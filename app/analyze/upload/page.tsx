"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { UploadAnalysis } from "../UploadAnalysis";

export default function UploadAnalysisPage() {
  return (
    <div className="space-y-8">
      <div>
        <Link
          href="/analyze"
          className="mb-3 inline-flex items-center gap-1 text-sm text-muted-foreground transition-colors hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4" />
          Back
        </Link>
        <h1 className="text-2xl font-bold text-foreground">Upload Video</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload one or more videos for batch processing and analysis
        </p>
      </div>

      <UploadAnalysis />
    </div>
  );
}
