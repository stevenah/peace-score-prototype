"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Activity } from "lucide-react";
import { Button } from "@/components/ui/Button";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    const result = await signIn("credentials", {
      email,
      password,
      redirect: false,
    });

    setIsLoading(false);

    if (result?.error) {
      setError("Invalid email or password");
    } else {
      router.push("/dashboard");
      router.refresh();
    }
  }

  return (
    <div className="flex min-h-[70vh] items-center justify-center">
      <div className="w-full max-w-sm space-y-6 rounded-2xl border border-border bg-card p-8 shadow-lg shadow-black/3 ring-1 ring-black/2 dark:ring-white/2">
        <div className="text-center">
          <Activity className="mx-auto h-10 w-10 text-primary" />
          <h1 className="mt-4 text-2xl font-bold text-foreground">
            Sign in to PEACE Analyzer
          </h1>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <p className="rounded-lg bg-red-50 p-3 text-sm text-red-600 dark:bg-red-950/30 dark:text-red-400">
              {error}
            </p>
          )}

          <div>
            <label
              htmlFor="email"
              className="block text-sm font-medium text-foreground/80"
            >
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="mt-1 block w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm shadow-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-ring/30 dark:text-foreground"
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-foreground/80"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="mt-1 block w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm shadow-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-ring/30 dark:text-foreground"
              placeholder="••••••••"
            />
          </div>

          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading ? "Signing in..." : "Sign in"}
          </Button>
        </form>

        <p className="text-center text-sm text-muted-foreground">
          No account?{" "}
          <Link
            href="/register"
            className="text-primary hover:text-primary/80"
          >
            Register
          </Link>
        </p>
      </div>
    </div>
  );
}
