"use client";

import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { ALLOWED_VIDEO_TYPES, MAX_FILE_SIZE_BYTES } from "@/lib/constants";

interface VideoUploaderProps {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export function VideoUploader({
  onFileSelect,
  disabled,
}: VideoUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    if (!ALLOWED_VIDEO_TYPES.includes(file.type) && !file.name.match(/\.(mp4|mov|avi|mkv)$/i)) {
      return "Unsupported file type. Please upload MP4, MOV, AVI, or MKV.";
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      return `File too large. Maximum size is ${MAX_FILE_SIZE_BYTES / (1024 * 1024)}MB.`;
    }
    return null;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
      setError(null);
      onFileSelect(file);
    },
    [validateFile, onFileSelect],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  return (
    <div className="flex h-full flex-col space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        className={cn(
          "flex min-h-0 flex-1 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-12 transition-colors",
          isDragOver
            ? "border-primary bg-primary/5 dark:bg-primary/10"
            : "border-border hover:border-primary/40 dark:border-border dark:hover:border-primary/40",
          disabled && "cursor-not-allowed opacity-50",
        )}
      >
        <Upload
          className={cn(
            "mb-4 h-10 w-10",
            isDragOver ? "text-primary" : "text-muted-foreground/60",
          )}
        />
        <p className="mb-1 text-sm font-medium text-foreground/80">
          {isDragOver ? "Drop video here" : "Drag & drop endoscopy video"}
        </p>
        <p className="text-xs text-muted-foreground">
          MP4, MOV, AVI, MKV up to 500MB
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,.mp4,.mov,.avi,.mkv"
          onChange={handleChange}
          className="hidden"
          disabled={disabled}
        />
      </div>

      {error && (
        <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
      )}

    </div>
  );
}
