"use client";

import { useCallback, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Loader2, CheckCircle2 } from "lucide-react";
import { LiveAnalysis, type ConnectionStatus, type SaveStatus } from "../LiveAnalysis";

export default function LiveAnalysisPage() {
  const [connStatus, setConnStatus] = useState<ConnectionStatus | null>(null);
  const [saveStatus, setSaveStatus] = useState<SaveStatus | null>(null);
  const handleConnectionStatus = useCallback(
    (s: ConnectionStatus) => setConnStatus(s),
    [],
  );
  const handleSaveStatus = useCallback(
    (s: SaveStatus) => setSaveStatus(s),
    [],
  );

  const showStatus = connStatus?.isAnalyzing;

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
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold text-foreground">Live Analysis</h1>
          {showStatus && (
            <div className="flex items-center gap-2 text-sm">
              <span
                className={`inline-block h-2.5 w-2.5 rounded-full ${
                  connStatus.isConnected
                    ? "bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.6)]"
                    : connStatus.connectionError
                      ? "bg-red-500"
                      : "animate-pulse bg-yellow-500"
                }`}
              />
              <span className="text-muted-foreground">
                {connStatus.isConnected
                  ? "Connected to analysis server"
                  : connStatus.connectionError
                    ? connStatus.connectionError
                    : connStatus.isConnecting
                      ? "Connecting..."
                      : "Disconnected"}
              </span>
            </div>
          )}
          {saveStatus?.isSaving && (
            <div className="flex items-center gap-1.5 text-sm">
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              <span className="text-muted-foreground">Saving...</span>
            </div>
          )}
          {saveStatus?.isSaved && (
            <div className="flex items-center gap-1.5 text-sm">
              <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
              <span className="text-green-600 dark:text-green-400">
                Analysis saved to your dashboard.
              </span>
            </div>
          )}
          {saveStatus?.saveError && (
            <span className="text-sm text-red-600 dark:text-red-400">
              Failed to save: {saveStatus.saveError}
            </span>
          )}
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Perform real-time frame-by-frame analysis on a video
        </p>
      </div>

      <LiveAnalysis
        onConnectionStatus={handleConnectionStatus}
        onSaveStatus={handleSaveStatus}
      />
    </div>
  );
}
