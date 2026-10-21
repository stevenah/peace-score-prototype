import { cn } from "@/lib/utils";

/** A card that holds rows: the same frame as Card, with a divider between items. */
export function List({ className, ...props }: React.ComponentProps<"ul">) {
  return (
    <ul
      data-slot="list"
      className={cn(
        "divide-y divide-border overflow-hidden rounded-xl border border-border bg-card shadow-sm",
        className,
      )}
      {...props}
    />
  );
}
