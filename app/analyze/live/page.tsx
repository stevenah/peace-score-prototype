"use client";

import { useCallback, useState } from "react";
import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { LiveAnalysis, type ConnectionStatus } from "../LiveAnalysis";

export default function LiveAnalysisPage() {
  const [connStatus, setConnStatus] = useState<ConnectionStatus | null>(null);
  const handleConnectionStatus = useCallback(
    (s: ConnectionStatus) => setConnStatus(s),
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
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Perform real-time frame-by-frame analysis on a video
        </p>
      </div>

      <LiveAnalysis onConnectionStatus={handleConnectionStatus} />
    </div>
  );
}
