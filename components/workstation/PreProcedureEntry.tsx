"use client";

import { useCallback, useState } from "react";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { VideoUploader } from "@/components/video/VideoUploader";
import { Button } from "@/components/ui/Button";

export interface ProcedureSource {
  source: File | string;
  mode: "file" | "url";
  label: string;
}

/**
 * Pre-procedure screen. Sits on the same black ground the video will use, so the
 * feed appears *into* the frame rather than the page changing character.
 */
export function PreProcedureEntry({
  onStart,
}: {
  onStart: (s: ProcedureSource) => void;
}) {
  const [streamUrl, setStreamUrl] = useState("");
  const [urlError, setUrlError] = useState<string | null>(null);

  const submitUrl = useCallback(() => {
    const trimmed = streamUrl.trim();
    if (!trimmed) return;
    try {
      const parsed = new URL(trimmed);
      if (!["http:", "https:"].includes(parsed.protocol)) {
        setUrlError("URL must use HTTP or HTTPS");
        return;
      }
    } catch {
      setUrlError("Enter a valid URL");
      return;
    }
    setUrlError(null);
    onStart({ source: trimmed, mode: "url", label: "live-stream" });
  }, [streamUrl, onStart]);

  return (
    <div className="grid h-full place-items-center px-6">
      <div className="w-full max-w-md space-y-5">
        <div className="space-y-1 text-center">
          <h1 className="text-lg font-semibold text-ws-fg">Live analysis</h1>
          <p className="text-[13px] text-ws-label">
            Load a recording or connect a stream to begin a procedure.
          </p>
        </div>

        <VideoUploader
          onFilesSelect={(files) =>
            onStart({ source: files[0], mode: "file", label: files[0].name })
          }
        />

        <div className="flex items-center gap-3">
          <div className="h-px flex-1 bg-ws-line" />
          <span className="ws-micro uppercase">or paste a link</span>
          <div className="h-px flex-1 bg-ws-line" />
        </div>

        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submitUrl();
          }}
        >
          <input
            type="url"
            value={streamUrl}
            onChange={(e) => {
              setStreamUrl(e.target.value);
              setUrlError(null);
            }}
            placeholder="https://example.com/stream.m3u8"
            className="flex-1 border border-ws-line bg-ws-surface px-3 py-2 text-sm text-ws-fg placeholder:text-ws-faint focus:border-ws-detect focus:outline-none"
          />
          <Button type="submit" disabled={!streamUrl.trim()}>
            Start
          </Button>
        </form>
        {urlError && <p className="text-sm text-peace-0">{urlError}</p>}

        <div className="pt-2 text-center">
          <Link
            href="/analyze"
            className="inline-flex items-center gap-1 text-[13px] text-ws-label transition-colors hover:text-ws-fg"
          >
            <ArrowLeft className="size-3.5" />
            Back
          </Link>
        </div>
      </div>
    </div>
  );
}
