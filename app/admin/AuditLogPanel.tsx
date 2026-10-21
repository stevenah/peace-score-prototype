"use client";

import { useCallback, useEffect, useState } from "react";
import { ClipboardList, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Badge, type BadgeVariant } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { List } from "@/components/ui/List";
import { Select } from "@/components/ui/Select";
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

const ACTION_VARIANTS: Record<string, BadgeVariant> = {
  USER_CREATED: "success",
  VIDEO_UPLOADED: "success",
  EVAL_STARTED: "success",
  USER_UPDATED: "primary",
  ROLE_CHANGED: "primary",
  MODEL_ACTIVATED: "primary",
  SERVING_MODE_CHANGED: "primary",
  UPLOAD_COUNT_RESET: "warning",
  PASSWORD_RESET: "warning",
  USER_DELETED: "destructive",
  ANALYSIS_DELETED: "destructive",
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
  MODEL_ACTIVATED: "Model Activated",
  SERVING_MODE_CHANGED: "Serving Mode",
  EVAL_STARTED: "Eval Started",
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
  MODEL_ACTIVATED: "switched the landmark model to",
  SERVING_MODE_CHANGED: "set station recognition to",
  EVAL_STARTED: "started eval",
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
  "MODEL_ACTIVATED",
  "SERVING_MODE_CHANGED",
  "EVAL_STARTED",
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
      <Select
        aria-label="Filter by action"
        value={actionFilter}
        onChange={(e) => setActionFilter(e.target.value)}
        className="max-w-xs"
      >
        <option value="">All actions</option>
        {ALL_ACTIONS.map((action) => (
          <option key={action} value={action}>
            {ACTION_LABELS[action] || action}
          </option>
        ))}
      </Select>

      {/* Log entries */}
      {logs.length === 0 ? (
        <EmptyState
          icon={ClipboardList}
          description={actionFilter ? "No logs match this filter" : "No audit logs yet"}
        />
      ) : (
        <List>
          {logs.map((log) => (
            <li key={log.id} className="space-y-1 px-5 py-4">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={ACTION_VARIANTS[log.action] ?? "default"}>
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
            </li>
          ))}
        </List>
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
            <ChevronLeft />
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
            <ChevronRight />
          </Button>
        </div>
      )}
    </div>
  );
}
