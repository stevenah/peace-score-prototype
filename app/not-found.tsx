import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { FileQuestion } from "lucide-react";

export default function NotFound() {
  return (
    <div className="flex min-h-[50vh] flex-col items-center justify-center gap-4 text-center">
      <FileQuestion className="h-12 w-12 text-muted-foreground" />
      <h2 className="text-xl font-semibold">Page not found</h2>
      <p className="text-muted-foreground">
        The page you&apos;re looking for doesn&apos;t exist.
      </p>
      <Button asChild>
        <Link href="/dashboard">Go to Dashboard</Link>
      </Button>
    </div>
  );
}
