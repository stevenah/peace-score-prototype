import { cn } from "@/lib/utils";

interface BadgeProps {
  children: React.ReactNode;
  color?: string;
  bgColor?: string;
  className?: string;
}

export function Badge({ children, color, bgColor, className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium",
        className,
      )}
      style={{
        color: color || undefined,
        backgroundColor: bgColor || undefined,
      }}
    >
      {children}
    </span>
  );
}
