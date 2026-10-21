import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";
import { Card } from "./Card";

interface EmptyStateProps {
  icon: LucideIcon;
  title?: React.ReactNode;
  description?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}

/** The "nothing here" card: one icon, a line or two, maybe a way forward. */
export function EmptyState({ icon: Icon, title, description, action, className }: EmptyStateProps) {
  return (
    <Card className={cn("flex flex-col items-center gap-3 py-12 text-center", className)}>
      <Icon aria-hidden="true" className="size-10 text-muted-foreground/50" />
      <div className="space-y-1">
        {title && <p className="text-base font-medium text-foreground">{title}</p>}
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
      {action && <div className="mt-1">{action}</div>}
    </Card>
  );
}
