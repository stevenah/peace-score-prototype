import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: React.ReactNode;
  description?: React.ReactNode;
  icon?: LucideIcon;
  /** Sits beside the title: a count badge, an ID. */
  meta?: React.ReactNode;
  /** Buttons at the trailing edge. */
  actions?: React.ReactNode;
  className?: string;
}

/** Every page starts with this: one h1, an optional line under it, actions on the right. */
export function PageHeader({ title, description, icon: Icon, meta, actions, className }: PageHeaderProps) {
  return (
    <div className={cn("flex flex-wrap items-start justify-between gap-4", className)}>
      <div className="min-w-0">
        <div className="flex items-center gap-3">
          {Icon && <Icon aria-hidden="true" className="size-6 shrink-0 text-primary" />}
          <h1 className="text-2xl font-bold tracking-tight text-foreground">{title}</h1>
          {meta}
        </div>
        {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
