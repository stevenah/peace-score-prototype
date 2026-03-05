"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Activity } from "lucide-react";
import { Button } from "@/components/ui/Button";

type Step = "login" | "set-password";

export default function LoginPage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const inputClass =
    "mt-1 block w-full rounded-lg border border-border bg-background px-3 py-2.5 text-sm shadow-sm transition-colors focus:border-primary focus:outline-none focus:ring-2 focus:ring-ring/30 dark:text-foreground";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    if (step === "login") {
      // Check if user needs to set a password
      const checkRes = await fetch("/api/auth/check-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });

      if (checkRes.ok) {
        const { needsPassword } = await checkRes.json();
        if (needsPassword) {
          setIsLoading(false);
          setStep("set-password");
          return;
        }
      }

      // Normal login
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
    } else {
      // Set password step
      if (password.length < 6) {
        setError("Password must be at least 6 characters");
        setIsLoading(false);
        return;
      }

      if (password !== confirmPassword) {
        setError("Passwords do not match");
        setIsLoading(false);
        return;
      }

      const setRes = await fetch("/api/auth/set-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      if (!setRes.ok) {
        const data = await setRes.json();
        setError(data.error || "Failed to set password");
        setIsLoading(false);
        return;
      }

      // Auto sign in after setting password
      const result = await signIn("credentials", {
        email,
        password,
        redirect: false,
      });

      setIsLoading(false);

      if (result?.error) {
        setError("Password set but sign in failed. Please try again.");
        setStep("login");
      } else {
        router.push("/dashboard");
        router.refresh();
      }
    }
  }

  return (
    <div className="flex min-h-[70vh] items-center justify-center">
      <div className="w-full max-w-sm space-y-6 rounded-2xl border border-border bg-card p-8 shadow-lg shadow-black/3 ring-1 ring-black/2 dark:ring-white/2">
        <div className="text-center">
          <Activity className="mx-auto h-10 w-10 text-primary" />
          <h1 className="mt-4 text-2xl font-bold text-foreground">
            {step === "login"
              ? "Sign in to PEACE Analyzer"
              : "Set Your Password"}
          </h1>
          {step === "set-password" && (
            <p className="mt-2 text-sm text-muted-foreground">
              Your account requires a password. Please create one to continue.
            </p>
          )}
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
              disabled={step === "set-password"}
              className={`${inputClass} ${step === "set-password" ? "opacity-60" : ""}`}
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label
              htmlFor="password"
              className="block text-sm font-medium text-foreground/80"
            >
              {step === "login" ? "Password" : "New Password"}
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required={step === "set-password"}
              minLength={step === "set-password" ? 6 : undefined}
              className={inputClass}
              placeholder={
                step === "login" ? "••••••••" : "Min 6 characters"
              }
            />
          </div>

          {step === "set-password" && (
            <div>
              <label
                htmlFor="confirm-password"
                className="block text-sm font-medium text-foreground/80"
              >
                Confirm Password
              </label>
              <input
                id="confirm-password"
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                minLength={6}
                className={inputClass}
                placeholder="Re-enter your password"
              />
            </div>
          )}

          <Button type="submit" className="w-full" disabled={isLoading}>
            {isLoading
              ? step === "login"
                ? "Signing in..."
                : "Setting password..."
              : step === "login"
                ? "Sign in"
                : "Set Password & Sign in"}
          </Button>

          {step === "set-password" && (
            <button
              type="button"
              onClick={() => {
                setStep("login");
                setPassword("");
                setConfirmPassword("");
                setError(null);
              }}
              className="w-full text-center text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              Back to sign in
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
