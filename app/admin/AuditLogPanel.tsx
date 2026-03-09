"use client";

import { useCallback, useEffect, useState } from "react";
import { ClipboardList, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Skeleton } from "@/components/ui/Skeleton";

interface AuditLogEntry {
  id: string;
  timestamp: string;
  actorEmail: string | null;
  action: string;
  targetType: string;
  targetId: string | null;
  targetLabel: string | null;
  details: string | null;
}

const ACTION_COLORS: Record<string, string> = {
  USER_CREATED: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400",
  VIDEO_UPLOADED: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400",
  USER_UPDATED: "bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400",
  ROLE_CHANGED: "bg-blue-100 text-blue-700 dark:bg-blue-950/40 dark:text-blue-400",
  UPLOAD_COUNT_RESET: "bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400",
  PASSWORD_RESET: "bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400",
  USER_DELETED: "bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-400",
  ANALYSIS_DELETED: "bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-400",
};

const ACTION_LABELS: Record<string, string> = {
  USER_CREATED: "User Created",
  USER_UPDATED: "User Updated",
  USER_DELETED: "User Deleted",
  UPLOAD_COUNT_RESET: "Count Reset",
  ROLE_CHANGED: "Role Changed",
  PASSWORD_RESET: "Password Reset",
  VIDEO_UPLOADED: "Video Uploaded",
  ANALYSIS_DELETED: "Analysis Deleted",
};

const ACTION_VERBS: Record<string, string> = {
  USER_CREATED: "created user",
  USER_UPDATED: "updated user",
  USER_DELETED: "deleted user",
  UPLOAD_COUNT_RESET: "reset upload count for",
  ROLE_CHANGED: "changed role for",
  PASSWORD_RESET: "reset password for",
  VIDEO_UPLOADED: "uploaded video",
  ANALYSIS_DELETED: "deleted analysis",
};

const ALL_ACTIONS = [
  "USER_CREATED",
  "USER_UPDATED",
  "USER_DELETED",
  "ROLE_CHANGED",
  "PASSWORD_RESET",
  "UPLOAD_COUNT_RESET",
  "VIDEO_UPLOADED",
  "ANALYSIS_DELETED",
];

function formatDetails(details: Record<string, unknown>): string {
  return Object.entries(details)
    .map(([key, value]) => {
      if (
        value &&
        typeof value === "object" &&
        "from" in value &&
        "to" in value
      ) {
        const v = value as { from: unknown; to: unknown };
        return `${key}: ${v.from ?? "none"} → ${v.to ?? "none"}`;
      }
      return `${key}: ${JSON.stringify(value)}`;
    })
    .join(", ");
}

const inputClass =
  "block w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm shadow-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-ring/30 dark:text-foreground";

export default function AuditLogPanel() {
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [actionFilter, setActionFilter] = useState("");
  const pageSize = 50;

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const fetchLogs = useCallback(async () => {
    setIsLoading(true);
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("pageSize", String(pageSize));
    if (actionFilter) params.set("action", actionFilter);

    const res = await fetch(`/api/admin/audit-logs?${params}`);
    if (res.ok) {
      const data = await res.json();
      setLogs(data.logs);
      setTotal(data.total);
    }
    setIsLoading(false);
  }, [page, actionFilter]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  useEffect(() => {
    setPage(1);
  }, [actionFilter]);

  if (isLoading && logs.length === 0) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-16" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filter */}
      <div className="max-w-xs">
        <select
          value={actionFilter}
          onChange={(e) => setActionFilter(e.target.value)}
          className={inputClass}
        >
          <option value="">All actions</option>
          {ALL_ACTIONS.map((action) => (
            <option key={action} value={action}>
              {ACTION_LABELS[action] || action}
            </option>
          ))}
        </select>
      </div>

      {/* Log entries */}
      {logs.length === 0 ? (
        <Card className="p-12 text-center">
          <ClipboardList className="mx-auto h-10 w-10 text-muted-foreground/50" />
          <p className="mt-3 text-sm text-muted-foreground">
            {actionFilter ? "No logs match this filter" : "No audit logs yet"}
          </p>
        </Card>
      ) : (
        <div className="space-y-2">
          {logs.map((log) => (
            <div
              key={log.id}
              className="flex items-start gap-4 rounded-lg border border-border bg-card p-4"
            >
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge
                    className={`text-xs ${ACTION_COLORS[log.action] || "bg-muted text-muted-foreground"}`}
                  >
                    {ACTION_LABELS[log.action] || log.action}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {new Date(log.timestamp).toLocaleString()}
                  </span>
                </div>
                <p className="text-sm">
                  <span className="font-medium">
                    {log.actorEmail ?? "System"}
                  </span>{" "}
                  {ACTION_VERBS[log.action] || "performed action on"}{" "}
                  <span className="font-medium">
                    {log.targetLabel ?? log.targetId ?? ""}
                  </span>
                </p>
                {log.details && (
                  <p className="text-xs text-muted-foreground">
                    {(() => {
                      try {
                        return formatDetails(JSON.parse(log.details));
                      } catch {
                        return log.details;
                      }
                    })()}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {total > pageSize && (
        <div className="flex items-center justify-between pt-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
          >
            <ChevronLeft className="mr-1 h-4 w-4" />
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            Page {page} of {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
          >
            Next
            <ChevronRight className="ml-1 h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}
