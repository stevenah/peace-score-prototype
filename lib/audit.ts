import { prisma } from "@/lib/db";

export type AuditAction =
  | "USER_CREATED"
  | "USER_UPDATED"
  | "USER_DELETED"
  | "UPLOAD_COUNT_RESET"
  | "ROLE_CHANGED"
  | "PASSWORD_RESET"
  | "VIDEO_UPLOADED"
  | "ANALYSIS_DELETED";

interface AuditLogEntry {
  actorId: string | null;
  actorEmail: string | null;
  action: AuditAction;
  targetType: string;
  targetId?: string | null;
  targetLabel?: string | null;
  details?: Record<string, unknown> | null;
}

export function logAudit(entry: AuditLogEntry): void {
  prisma.auditLog
    .create({
      data: {
        actorId: entry.actorId,
        actorEmail: entry.actorEmail,
        action: entry.action,
        targetType: entry.targetType,
        targetId: entry.targetId ?? null,
        targetLabel: entry.targetLabel ?? null,
        details: entry.details ? JSON.stringify(entry.details) : null,
      },
    })
    .catch((err) => console.error("Audit log write failed:", err));
}
