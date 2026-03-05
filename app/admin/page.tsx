"use client";

import { useCallback, useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Users, Plus, Search, Shield, Pencil, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import { Skeleton } from "@/components/ui/Skeleton";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface AdminUser {
  id: string;
  name: string | null;
  email: string;
  role: string;
  uploadLimit: number;
  uploadCount: number;
  createdAt: string;
  _count: { analyses: number };
}

const inputClass =
  "mt-1 block w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm shadow-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-ring/30 dark:text-foreground";

const labelClass = "block text-sm font-medium text-foreground/80";

export default function AdminPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState("");

  // Add user dialog
  const [showAddUser, setShowAddUser] = useState(false);
  const [newName, setNewName] = useState("");
  const [newEmail, setNewEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newRole, setNewRole] = useState("USER");
  const [newLimit, setNewLimit] = useState(10);
  const [formError, setFormError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Edit user dialog
  const [editingUser, setEditingUser] = useState<AdminUser | null>(null);
  const [editLimit, setEditLimit] = useState(10);
  const [editRole, setEditRole] = useState("USER");

  const fetchUsers = useCallback(async () => {
    const params = new URLSearchParams();
    if (search) params.set("search", search);
    const res = await fetch(`/api/admin/users?${params}`);
    if (res.ok) {
      const data = await res.json();
      setUsers(data.users);
    }
    setIsLoading(false);
  }, [search]);

  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/login");
      return;
    }
    if (status === "authenticated") {
      if (session?.user?.role !== "ADMIN") {
        router.push("/dashboard");
        return;
      }
      fetchUsers();
    }
  }, [status, session, router, fetchUsers]);

  function resetAddForm() {
    setNewName("");
    setNewEmail("");
    setNewPassword("");
    setNewRole("USER");
    setNewLimit(10);
    setFormError(null);
  }

  async function handleAddUser(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);
    setIsSubmitting(true);

    const res = await fetch("/api/admin/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: newName || null,
        email: newEmail,
        password: newPassword,
        role: newRole,
        uploadLimit: newLimit,
      }),
    });

    setIsSubmitting(false);
    if (res.ok) {
      setShowAddUser(false);
      resetAddForm();
      fetchUsers();
    } else {
      const data = await res.json();
      setFormError(data.error || "Failed to create user");
    }
  }

  async function handleUpdateUser() {
    if (!editingUser) return;
    setIsSubmitting(true);

    const res = await fetch(`/api/admin/users/${editingUser.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: editRole,
        uploadLimit: editLimit,
      }),
    });

    setIsSubmitting(false);
    if (res.ok) {
      setEditingUser(null);
      fetchUsers();
    }
  }

  async function handleResetCount(user: AdminUser) {
    const res = await fetch(`/api/admin/users/${user.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uploadCount: 0 }),
    });
    if (res.ok) fetchUsers();
  }

  function openEditDialog(user: AdminUser) {
    setEditingUser(user);
    setEditRole(user.role);
    setEditLimit(user.uploadLimit);
  }

  function getUsagePercent(user: AdminUser) {
    if (user.uploadLimit === -1) return 0;
    if (user.uploadLimit === 0) return 100;
    return Math.min(100, (user.uploadCount / user.uploadLimit) * 100);
  }

  function getUsageColor(user: AdminUser) {
    const pct = getUsagePercent(user);
    if (pct >= 100) return "var(--color-red-500, #ef4444)";
    if (pct >= 80) return "var(--color-amber-500, #f59e0b)";
    return "var(--clinical-blue, #3b82f6)";
  }

  if (status === "loading" || isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-10 w-28" />
        </div>
        <Skeleton className="h-10 w-full" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-44" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Shield className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold tracking-tight">
            Admin Dashboard
          </h1>
          <Badge className="text-xs bg-muted text-muted-foreground">
            {users.length} user{users.length !== 1 ? "s" : ""}
          </Badge>
        </div>
        <Button onClick={() => setShowAddUser(true)}>
          <Plus className="mr-2 h-4 w-4" />
          Add User
        </Button>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <input
          type="text"
          placeholder="Search users by name or email..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className={`${inputClass} pl-9`}
        />
      </div>

      {/* Users grid */}
      {users.length === 0 ? (
        <Card className="p-12 text-center">
          <Users className="mx-auto h-10 w-10 text-muted-foreground/50" />
          <p className="mt-3 text-sm text-muted-foreground">
            {search ? "No users match your search" : "No users found"}
          </p>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {users.map((user) => (
            <Card key={user.id} className="relative">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="min-w-0 flex-1">
                    <CardTitle className="truncate text-sm font-semibold">
                      {user.name || "Unnamed User"}
                    </CardTitle>
                    <p className="mt-0.5 truncate text-xs text-muted-foreground">
                      {user.email}
                    </p>
                  </div>
                  <Badge
                    className={`ml-2 shrink-0 text-xs ${
                      user.role === "ADMIN"
                        ? "bg-primary/10 text-primary"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {user.role}
                  </Badge>
                </div>
              </CardHeader>
              <div className="space-y-3 px-6 pb-5">
                {/* Usage */}
                <div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>Usage</span>
                    <span>
                      {user.uploadCount} /{" "}
                      {user.uploadLimit === -1
                        ? "Unlimited"
                        : user.uploadLimit}
                    </span>
                  </div>
                  <Progress
                    value={getUsagePercent(user)}
                    color={getUsageColor(user)}
                    className="mt-1.5"
                  />
                </div>

                {/* Stats */}
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{user._count.analyses} total analyses</span>
                  <span>
                    Joined{" "}
                    {new Date(user.createdAt).toLocaleDateString()}
                  </span>
                </div>

                {/* Actions */}
                <div className="flex gap-2 pt-1">
                  <Button
                    variant="outline"
                    size="sm"
                    className="flex-1"
                    onClick={() => openEditDialog(user)}
                  >
                    <Pencil className="mr-1.5 h-3 w-3" />
                    Edit
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleResetCount(user)}
                    title="Reset upload count"
                  >
                    <RotateCcw className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Add User Dialog */}
      <AlertDialog
        open={showAddUser}
        onOpenChange={(open) => {
          if (!open) resetAddForm();
          setShowAddUser(open);
        }}
      >
        <AlertDialogContent>
          <form onSubmit={handleAddUser}>
            <AlertDialogHeader>
              <AlertDialogTitle>Add New User</AlertDialogTitle>
              <AlertDialogDescription>
                Create a new user account with specified role and upload limit.
              </AlertDialogDescription>
            </AlertDialogHeader>

            <div className="my-4 space-y-3">
              {formError && (
                <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600 dark:bg-red-950/30 dark:text-red-400">
                  {formError}
                </p>
              )}

              <div>
                <label htmlFor="add-name" className={labelClass}>
                  Name
                </label>
                <input
                  id="add-name"
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className={inputClass}
                  placeholder="John Doe"
                />
              </div>

              <div>
                <label htmlFor="add-email" className={labelClass}>
                  Email <span className="text-red-500">*</span>
                </label>
                <input
                  id="add-email"
                  type="email"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  required
                  className={inputClass}
                  placeholder="user@example.com"
                />
              </div>

              <div>
                <label htmlFor="add-password" className={labelClass}>
                  Password <span className="text-red-500">*</span>
                </label>
                <input
                  id="add-password"
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  required
                  minLength={6}
                  className={inputClass}
                  placeholder="Min 6 characters"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label htmlFor="add-role" className={labelClass}>
                    Role
                  </label>
                  <select
                    id="add-role"
                    value={newRole}
                    onChange={(e) => setNewRole(e.target.value)}
                    className={inputClass}
                  >
                    <option value="USER">User</option>
                    <option value="ADMIN">Admin</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="add-limit" className={labelClass}>
                    Upload Limit
                  </label>
                  <input
                    id="add-limit"
                    type="number"
                    value={newLimit}
                    onChange={(e) => setNewLimit(parseInt(e.target.value) || 0)}
                    className={inputClass}
                    min={-1}
                  />
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    -1 = unlimited
                  </p>
                </div>
              </div>
            </div>

            <AlertDialogFooter>
              <AlertDialogCancel type="button">Cancel</AlertDialogCancel>
              <AlertDialogAction type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Creating..." : "Create User"}
              </AlertDialogAction>
            </AlertDialogFooter>
          </form>
        </AlertDialogContent>
      </AlertDialog>

      {/* Edit User Dialog */}
      <AlertDialog
        open={!!editingUser}
        onOpenChange={(open) => {
          if (!open) setEditingUser(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Edit User</AlertDialogTitle>
            <AlertDialogDescription>
              {editingUser?.name || editingUser?.email} — Update role and upload
              quota settings.
            </AlertDialogDescription>
          </AlertDialogHeader>

          <div className="my-4 space-y-3">
            <div>
              <label htmlFor="edit-role" className={labelClass}>
                Role
              </label>
              <select
                id="edit-role"
                value={editRole}
                onChange={(e) => setEditRole(e.target.value)}
                className={inputClass}
                disabled={editingUser?.id === session?.user?.id}
              >
                <option value="USER">User</option>
                <option value="ADMIN">Admin</option>
              </select>
              {editingUser?.id === session?.user?.id && (
                <p className="mt-0.5 text-xs text-muted-foreground">
                  You cannot change your own role
                </p>
              )}
            </div>

            <div>
              <label htmlFor="edit-limit" className={labelClass}>
                Upload Limit
              </label>
              <input
                id="edit-limit"
                type="number"
                value={editLimit}
                onChange={(e) => setEditLimit(parseInt(e.target.value) || 0)}
                className={inputClass}
                min={-1}
              />
              <p className="mt-0.5 text-xs text-muted-foreground">
                -1 = unlimited, current usage: {editingUser?.uploadCount ?? 0}
              </p>
            </div>
          </div>

          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleUpdateUser}
              disabled={isSubmitting}
            >
              {isSubmitting ? "Saving..." : "Save Changes"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
