"use client";

import { BackLink } from "@/components/layout/BackLink";
import { PageHeader } from "@/components/layout/PageHeader";
import { UploadAnalysis } from "../UploadAnalysis";

export default function UploadAnalysisPage() {
  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <BackLink href="/analyze">Back</BackLink>
        <PageHeader
          title="Upload Video"
          description="Upload one or more videos for batch processing and analysis"
        />
      </div>

      <UploadAnalysis />
    </div>
  );
}
