"use client";

import { useCallback, useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Users, Plus, Search, Shield, Pencil, RotateCcw, Dices, Trash2, ClipboardList, Boxes } from "lucide-react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";
import { Card, CardHeader, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { EmptyState } from "@/components/ui/EmptyState";
import { Field } from "@/components/ui/Field";
import { Input } from "@/components/ui/Input";
import { Progress, type ProgressTone } from "@/components/ui/Progress";
import { Select } from "@/components/ui/Select";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs } from "@/components/ui/Tabs";
import AuditLogPanel from "./AuditLogPanel";
import ModelsPanel from "./models/ModelsPanel";
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

export default function AdminPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState("users");
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
  const [newForcePasswordChange, setNewForcePasswordChange] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Delete user dialog
  const [deletingUser, setDeletingUser] = useState<AdminUser | null>(null);

  // Edit user dialog
  const [editingUser, setEditingUser] = useState<AdminUser | null>(null);
  const [editName, setEditName] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [editPassword, setEditPassword] = useState("");
  const [editLimit, setEditLimit] = useState(10);
  const [editRole, setEditRole] = useState("USER");
  const [editError, setEditError] = useState<string | null>(null);

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

  function generatePassword(length = 16) {
    const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%&*";
    const array = new Uint8Array(length);
    crypto.getRandomValues(array);
    return Array.from(array, (b) => chars[b % chars.length]).join("");
  }

  function resetAddForm() {
    setNewName("");
    setNewEmail("");
    setNewPassword("");
    setNewRole("USER");
    setNewLimit(10);
    setNewForcePasswordChange(false);
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
        forcePasswordChange: newForcePasswordChange,
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
    setEditError(null);
    setIsSubmitting(true);

    const payload: Record<string, unknown> = {
      name: editName || null,
      email: editEmail,
      role: editRole,
      uploadLimit: editLimit,
    };
    if (editPassword) {
      payload.password = editPassword;
    }

    const res = await fetch(`/api/admin/users/${editingUser.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    setIsSubmitting(false);
    if (res.ok) {
      setEditingUser(null);
      fetchUsers();
    } else {
      const data = await res.json();
      setEditError(data.error || "Failed to update user");
    }
  }

  async function handleDeleteUser() {
    if (!deletingUser) return;
    setIsSubmitting(true);

    const res = await fetch(`/api/admin/users/${deletingUser.id}`, {
      method: "DELETE",
    });

    setIsSubmitting(false);
    if (res.ok) {
      setDeletingUser(null);
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
    setEditName(user.name || "");
    setEditEmail(user.email);
    setEditPassword("");
    setEditRole(user.role);
    setEditLimit(user.uploadLimit);
    setEditError(null);
  }

  function getUsagePercent(user: AdminUser) {
    if (user.uploadLimit === -1) return 0;
    if (user.uploadLimit === 0) return 100;
    return Math.min(100, (user.uploadCount / user.uploadLimit) * 100);
  }

  function getUsageTone(user: AdminUser): ProgressTone {
    const pct = getUsagePercent(user);
    if (pct >= 100) return "destructive";
    if (pct >= 80) return "warning";
    return "primary";
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
      <PageHeader
        icon={Shield}
        title="Admin Dashboard"
        meta={
          activeTab === "users" && (
            <Badge>
              {users.length} user{users.length !== 1 ? "s" : ""}
            </Badge>
          )
        }
        actions={
          activeTab === "users" && (
            <Button onClick={() => setShowAddUser(true)}>
              <Plus />
              Add User
            </Button>
          )
        }
      />

      {/* Tabs */}
      <Tabs
        tabs={[
          { id: "users", label: "Users", icon: <Users className="size-4" /> },
          { id: "models", label: "Models", icon: <Boxes className="size-4" /> },
          { id: "audit", label: "Audit Log", icon: <ClipboardList className="size-4" /> },
        ]}
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {activeTab === "audit" ? (
        <AuditLogPanel />
      ) : activeTab === "models" ? (
        <ModelsPanel />
      ) : (
      <>
      {/* Search */}
      <div className="relative">
        <Search aria-hidden="true" className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="search"
          aria-label="Search users"
          placeholder="Search users by name or email..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* Users grid */}
      {users.length === 0 ? (
        <EmptyState
          icon={Users}
          description={search ? "No users match your search" : "No users found"}
        />
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
                    variant={user.role === "ADMIN" ? "primary" : "default"}
                    className="ml-2 shrink-0"
                  >
                    {user.role}
                  </Badge>
                </div>
              </CardHeader>
              <div className="space-y-3">
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
                    tone={getUsageTone(user)}
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
                    <Pencil />
                    Edit
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => handleResetCount(user)}
                    title="Reset upload count"
                    aria-label="Reset upload count"
                  >
                    <RotateCcw />
                  </Button>
                  {user.id !== session?.user?.id && (
                    <Button
                      variant="ghost-destructive"
                      size="icon-sm"
                      onClick={() => setDeletingUser(user)}
                      title="Delete user"
                      aria-label="Delete user"
                    >
                      <Trash2 />
                    </Button>
                  )}
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

            <div className="my-4 space-y-4">
              {formError && <Alert variant="destructive">{formError}</Alert>}

              <Field label="Name" htmlFor="add-name">
                <Input
                  id="add-name"
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="John Doe"
                />
              </Field>

              <Field label="Email" htmlFor="add-email" required>
                <Input
                  id="add-email"
                  type="email"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  required
                  placeholder="user@example.com"
                />
              </Field>

              <Field label="Password" htmlFor="add-password">
                <div className="flex gap-2">
                  <Input
                    id="add-password"
                    type={newPassword ? "text" : "password"}
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    minLength={newPassword ? 6 : undefined}
                    placeholder="Leave blank to let user set on first login"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    className="shrink-0"
                    onClick={() => setNewPassword(generatePassword())}
                    title="Generate random password"
                    aria-label="Generate random password"
                  >
                    <Dices />
                  </Button>
                </div>
              </Field>

              <div className="grid grid-cols-2 gap-3">
                <Field label="Role" htmlFor="add-role">
                  <Select
                    id="add-role"
                    value={newRole}
                    onChange={(e) => setNewRole(e.target.value)}
                  >
                    <option value="USER">User</option>
                    <option value="ADMIN">Admin</option>
                  </Select>
                </Field>

                <Field label="Upload Limit" htmlFor="add-limit" hint="-1 = unlimited">
                  <Input
                    id="add-limit"
                    type="number"
                    value={newLimit}
                    onChange={(e) => setNewLimit(parseInt(e.target.value) || 0)}
                    min={-1}
                  />
                </Field>
              </div>

              <label className="flex cursor-pointer items-center gap-2 text-sm text-foreground">
                <input
                  type="checkbox"
                  checked={newForcePasswordChange}
                  onChange={(e) => setNewForcePasswordChange(e.target.checked)}
                  className="size-4 accent-primary"
                />
                Require password change on first login
              </label>
            </div>

            <AlertDialogFooter>
              <AlertDialogCancel type="button">Cancel</AlertDialogCancel>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Creating..." : "Create User"}
              </Button>
            </AlertDialogFooter>
          </form>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete User Dialog */}
      <AlertDialog
        open={!!deletingUser}
        onOpenChange={(open) => {
          if (!open) setDeletingUser(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete User</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete{" "}
              <span className="font-medium text-foreground">
                {deletingUser?.name || deletingUser?.email}
              </span>
              ? This will permanently remove their account and all{" "}
              {deletingUser?._count.analyses ?? 0} analyses. This action cannot
              be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={handleDeleteUser}
              disabled={isSubmitting}
            >
              {isSubmitting ? "Deleting..." : "Delete User"}
            </AlertDialogAction>
          </AlertDialogFooter>
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
              Update user account details for{" "}
              {editingUser?.name || editingUser?.email}.
            </AlertDialogDescription>
          </AlertDialogHeader>

          <div className="my-4 space-y-4">
            {editError && <Alert variant="destructive">{editError}</Alert>}

            <Field label="Name" htmlFor="edit-name">
              <Input
                id="edit-name"
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                placeholder="John Doe"
              />
            </Field>

            <Field label="Email" htmlFor="edit-email" required>
              <Input
                id="edit-email"
                type="email"
                value={editEmail}
                onChange={(e) => setEditEmail(e.target.value)}
                required
                placeholder="user@example.com"
              />
            </Field>

            <Field label="New Password" htmlFor="edit-password">
              <div className="flex gap-2">
                <Input
                  id="edit-password"
                  type={editPassword ? "text" : "password"}
                  value={editPassword}
                  onChange={(e) => setEditPassword(e.target.value)}
                  minLength={6}
                  placeholder="Leave blank to keep current"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="shrink-0"
                  onClick={() => setEditPassword(generatePassword())}
                  title="Generate random password"
                  aria-label="Generate random password"
                >
                  <Dices />
                </Button>
              </div>
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field
                label="Role"
                htmlFor="edit-role"
                hint={editingUser?.id === session?.user?.id ? "You cannot change your own role" : undefined}
              >
                <Select
                  id="edit-role"
                  value={editRole}
                  onChange={(e) => setEditRole(e.target.value)}
                  disabled={editingUser?.id === session?.user?.id}
                >
                  <option value="USER">User</option>
                  <option value="ADMIN">Admin</option>
                </Select>
              </Field>

              <Field
                label="Upload Limit"
                htmlFor="edit-limit"
                hint={`-1 = unlimited, usage: ${editingUser?.uploadCount ?? 0}`}
              >
                <Input
                  id="edit-limit"
                  type="number"
                  value={editLimit}
                  onChange={(e) => setEditLimit(parseInt(e.target.value) || 0)}
                  min={-1}
                />
              </Field>
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
      </>
      )}
    </div>
  );
}
