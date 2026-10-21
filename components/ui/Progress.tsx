import { cn } from "@/lib/utils";

export type ProgressTone = "primary" | "success" | "warning" | "destructive";

const TONE_CLASS: Record<ProgressTone, string> = {
  primary: "bg-primary",
  success: "bg-success",
  warning: "bg-warning",
  destructive: "bg-destructive",
};

interface ProgressProps {
  value: number; // 0-100
  tone?: ProgressTone;
  className?: string;
}

export function Progress({ value, tone = "primary", className }: ProgressProps) {
  const clampedValue = Math.min(100, Math.max(0, value));

  return (
    <div
      role="progressbar"
      aria-valuenow={clampedValue}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cn("h-2 w-full overflow-hidden rounded-full bg-muted", className)}
    >
      <div
        className={cn(
          "h-full rounded-full transition-all duration-500 ease-out",
          TONE_CLASS[tone],
        )}
        style={{ width: `${clampedValue}%` }}
      />
    </div>
  );
}
