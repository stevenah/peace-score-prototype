"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, Film, X } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { cn } from "@/lib/utils";
import { formatFileSize } from "@/lib/utils";
import { ALLOWED_VIDEO_TYPES, MAX_FILE_SIZE_BYTES } from "@/lib/constants";

interface VideoUploaderProps {
  onFileSelect: (file: File) => void;
  isUploading?: boolean;
  disabled?: boolean;
}

export function VideoUploader({
  onFileSelect,
  isUploading,
  disabled,
}: VideoUploaderProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
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
      setSelectedFile(file);
    },
    [validateFile],
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

  const handleClear = () => {
    setSelectedFile(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

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
            ? "border-blue-500 bg-blue-50 dark:bg-blue-950/30"
            : "border-neutral-300 hover:border-neutral-400 dark:border-neutral-700 dark:hover:border-neutral-600",
          disabled && "cursor-not-allowed opacity-50",
        )}
      >
        <Upload
          className={cn(
            "mb-4 h-10 w-10",
            isDragOver ? "text-blue-500" : "text-neutral-400",
          )}
        />
        <p className="mb-1 text-sm font-medium text-neutral-700 dark:text-neutral-300">
          {isDragOver ? "Drop video here" : "Drag & drop endoscopy video"}
        </p>
        <p className="text-xs text-neutral-500">
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

      {selectedFile && !error && (
        <div className="flex items-center justify-between rounded-lg border border-neutral-200 bg-neutral-50 p-4 dark:border-neutral-700 dark:bg-neutral-800">
          <div className="flex items-center gap-3">
            <Film className="h-5 w-5 text-blue-600" />
            <div>
              <p className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                {selectedFile.name}
              </p>
              <p className="text-xs text-neutral-500">
                {formatFileSize(selectedFile.size)}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                handleClear();
              }}
              disabled={isUploading}
            >
              <X className="h-4 w-4" />
            </Button>
            <Button
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onFileSelect(selectedFile);
              }}
              disabled={isUploading}
            >
              {isUploading ? "Uploading..." : "Analyze"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
