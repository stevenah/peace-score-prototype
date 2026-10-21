import { ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";
import { controlClass } from "./Input";

/** A native <select> styled like Input, with its own chevron. */
export function Select({ className, children, ...props }: React.ComponentProps<"select">) {
  return (
    <div className={cn("relative", className)}>
      <select
        data-slot="select"
        className={cn(controlClass, "appearance-none pr-8")}
        {...props}
      >
        {children}
      </select>
      <ChevronDown
        aria-hidden="true"
        className="pointer-events-none absolute right-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
      />
    </div>
  );
}
