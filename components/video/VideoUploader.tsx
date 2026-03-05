"use client";

import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { ALLOWED_VIDEO_TYPES, MAX_FILE_SIZE_BYTES } from "@/lib/constants";

interface VideoUploaderProps {
  onFilesSelect: (files: File[]) => void;
  disabled?: boolean;
}

export function VideoUploader({
  onFilesSelect,
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

  const handleFiles = useCallback(
    (fileList: FileList) => {
      const valid: File[] = [];
      const errors: string[] = [];

      Array.from(fileList).forEach((file) => {
        const err = validateFile(file);
        if (err) {
          errors.push(`${file.name}: ${err}`);
        } else {
          valid.push(file);
        }
      });

      setError(errors.length > 0 ? errors.join("\n") : null);

      if (valid.length > 0) {
        onFilesSelect(valid);
      }
    },
    [validateFile, onFilesSelect],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      if (e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFiles(e.target.files);
      }
    },
    [handleFiles],
  );

  return (
    <div className="flex h-full flex-col space-y-4">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload endoscopy videos — drag and drop or click to browse"
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === "Enter" || e.key === " ") && !disabled) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
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
          {isDragOver ? "Drop videos here" : "Drag & drop endoscopy videos"}
        </p>
        <p className="text-xs text-muted-foreground">
          MP4, MOV, AVI, MKV up to 500MB each
        </p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,.mp4,.mov,.avi,.mkv"
          onChange={handleChange}
          className="hidden"
          disabled={disabled}
        />
      </div>

      {error && (
        <p className="whitespace-pre-line text-sm text-red-600 dark:text-red-400">{error}</p>
      )}

    </div>
  );
}
