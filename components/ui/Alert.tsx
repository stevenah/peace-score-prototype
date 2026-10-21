import { cva, type VariantProps } from "class-variance-authority";
import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

const alertVariants = cva(
  "flex items-start gap-2 rounded-lg px-3 py-2.5 text-sm [&>svg]:mt-0.5 [&>svg]:size-4 [&>svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-muted text-muted-foreground",
        destructive: "bg-destructive/10 text-destructive",
        warning: "bg-warning/12 text-warning",
        success: "bg-success/12 text-success",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

interface AlertProps
  extends Omit<React.ComponentProps<"div">, "title">,
    VariantProps<typeof alertVariants> {
  icon?: LucideIcon;
  /** Rendered at the trailing edge, e.g. a dismiss button. */
  action?: React.ReactNode;
}

/** An inline notice. Errors announce themselves; everything else is polite. */
export function Alert({ className, variant, icon: Icon, action, children, role, ...props }: AlertProps) {
  return (
    <div
      role={role ?? (variant === "destructive" ? "alert" : "status")}
      data-slot="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    >
      {Icon && <Icon aria-hidden="true" />}
      <div className="min-w-0 flex-1">{children}</div>
      {action}
    </div>
  );
}
