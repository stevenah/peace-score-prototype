import {
  CheckCircle2,
  Clock,
  Loader2,
  XCircle,
  type LucideIcon,
} from "lucide-react";

import { Badge, type BadgeVariant } from "@/components/ui/Badge";

/** Every lifecycle state an upload or analysis can be in, across the batch
 *  uploader, the dashboard cards and the results page. */
export type LifecycleStatus =
  | "pending"
  | "uploading"
  | "queued"
  | "processing"
  | "completed"
  | "failed";

const STATUS: Record<
  LifecycleStatus,
  { icon: LucideIcon; label: string; variant: BadgeVariant; spin?: boolean }
> = {
  pending: { icon: Clock, label: "Pending", variant: "default" },
  uploading: { icon: Loader2, label: "Uploading", variant: "primary", spin: true },
  queued: { icon: Clock, label: "Queued", variant: "warning" },
  processing: { icon: Loader2, label: "Processing", variant: "primary", spin: true },
  completed: { icon: CheckCircle2, label: "Complete", variant: "success" },
  failed: { icon: XCircle, label: "Failed", variant: "destructive" },
};

export function StatusBadge({ status, className }: { status: string; className?: string }) {
  const entry = STATUS[status as LifecycleStatus];
  if (!entry) return null;
  const Icon = entry.icon;
  return (
    <Badge variant={entry.variant} className={className}>
      <Icon aria-hidden="true" className={entry.spin ? "animate-spin" : undefined} />
      {entry.label}
    </Badge>
  );
}
