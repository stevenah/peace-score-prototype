import { BackLink } from "@/components/layout/BackLink";

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-4">
      <BackLink href="/dashboard">Back to Dashboard</BackLink>
      {children}
    </div>
  );
}
