"use client";

import { useState } from "react";
import { Upload, Radio } from "lucide-react";
import { Tabs } from "@/components/ui/Tabs";
import { UploadAnalysis } from "./UploadAnalysis";
import { LiveAnalysis } from "./LiveAnalysis";

const ANALYSIS_TABS = [
  { id: "upload", label: "Upload Video", icon: <Upload className="h-4 w-4" /> },
  { id: "live", label: "Live Analysis", icon: <Radio className="h-4 w-4" /> },
];

export default function AnalyzePage() {
  const [activeTab, setActiveTab] = useState("upload");

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
          Video Analysis
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          Upload a video for batch processing or perform real-time analysis
        </p>
      </div>

      <Tabs tabs={ANALYSIS_TABS} activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === "upload" ? <UploadAnalysis /> : <LiveAnalysis />}
    </div>
  );
}
