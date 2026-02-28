"use client";

import { useCallback, useState } from "react";
import { uploadVideo } from "@/lib/api-client";

export function useVideoUpload() {
  const [isUploading, setIsUploading] = useState(false);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function upload(file: File) {
    setIsUploading(true);
    setError(null);
    try {
      const result = await uploadVideo(file);
      setAnalysisId(result.analysis_id);
      return result.analysis_id;
    } catch (e) {
      const message = e instanceof Error ? e.message : "Upload failed";
      setError(message);
      return null;
    } finally {
      setIsUploading(false);
    }
  }

  const reset = useCallback(() => {
    setIsUploading(false);
    setAnalysisId(null);
    setError(null);
  }, []);

  return { upload, isUploading, analysisId, error, reset };
}
