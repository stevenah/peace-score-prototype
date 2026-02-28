"use client";

import { useCallback, useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { Upload, Clock, CheckCircle2, AlertTriangle } from "lucide-react";
import { Tabs } from "@/components/ui/Tabs";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import { AnalysisCard } from "./AnalysisCard";
import Link from "next/link";
import type { AnalysisRecord } from "@/lib/types";

type StatusFilter = "all" | "processing" | "completed" | "failed";

const EMPTY_STATES: Record<
  StatusFilter,
  { icon: React.ReactNode; title: string; description: string }
> = {
  all: {
    icon: <Upload className="h-12 w-12 text-neutral-300" />,
    title: "No analyses yet",
    description: "Upload an endoscopy video to get started with PEACE scoring.",
  },
  processing: {
    icon: <Clock className="h-12 w-12 text-neutral-300" />,
    title: "No active analyses",
    description: "All your analyses have completed processing.",
  },
  completed: {
    icon: <CheckCircle2 className="h-12 w-12 text-neutral-300" />,
    title: "No completed analyses",
    description: "Your analyses are still being processed.",
  },
  failed: {
    icon: <AlertTriangle className="h-12 w-12 text-neutral-300" />,
    title: "No failed analyses",
    description: "All your analyses completed successfully.",
  },
};

export default function DashboardPage() {
  const { data: session, status: authStatus } = useSession();
  const router = useRouter();
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<StatusFilter>("all");

  const fetchAnalyses = useCallback(async () => {
    try {
      const res = await fetch("/api/analyses");
      if (res.ok) {
        const data = await res.json();
        setAnalyses(data);
      }
    } catch {
      // Silently fail on poll errors
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Auth redirect + initial fetch
  useEffect(() => {
    if (authStatus === "unauthenticated") {
      router.push("/login");
      return;
    }
    if (authStatus === "authenticated") {
      fetchAnalyses();
    }
  }, [authStatus, router, fetchAnalyses]);

  // Auto-refresh every 5s if any jobs are in-progress
  const hasActiveJobs = analyses.some(
    (a) => a.status === "processing" || a.status === "queued",
  );

  useEffect(() => {
    if (!hasActiveJobs) return;
    const interval = setInterval(fetchAnalyses, 5000);
    return () => clearInterval(interval);
  }, [hasActiveJobs, fetchAnalyses]);

  // Filter
  const filtered =
    filter === "all"
      ? analyses
      : analyses.filter((a) => {
          if (filter === "processing")
            return a.status === "processing" || a.status === "queued";
          return a.status === filter;
        });

  // Counts for tab labels
  const counts: Record<StatusFilter, number> = {
    all: analyses.length,
    processing: analyses.filter(
      (a) => a.status === "processing" || a.status === "queued",
    ).length,
    completed: analyses.filter((a) => a.status === "completed").length,
    failed: analyses.filter((a) => a.status === "failed").length,
  };

  const filterTabs = [
    { id: "all", label: `All (${counts.all})` },
    { id: "processing", label: `Processing (${counts.processing})` },
    { id: "completed", label: `Completed (${counts.completed})` },
    { id: "failed", label: `Failed (${counts.failed})` },
  ];

  if (isLoading || authStatus === "loading") {
    return (
      <div className="space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-5 w-40" />
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 rounded-xl" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">
            Welcome back, {session?.user?.name || session?.user?.email}
          </h1>
          <p className="mt-1 text-sm text-neutral-500">
            {analyses.length} total{" "}
            {analyses.length === 1 ? "analysis" : "analyses"}
          </p>
        </div>
        <Link href="/analyze">
          <Button>
            <Upload className="mr-2 h-4 w-4" />
            New Analysis
          </Button>
        </Link>
      </div>

      {/* Filter tabs */}
      <Tabs
        tabs={filterTabs}
        activeTab={filter}
        onChange={(id) => setFilter(id as StatusFilter)}
      />

      {/* Cards grid or empty state */}
      {filtered.length === 0 ? (
        <Card className="py-12 text-center">
          <div className="flex flex-col items-center gap-3">
            {EMPTY_STATES[filter].icon}
            <h3 className="text-lg font-medium text-neutral-700 dark:text-neutral-300">
              {EMPTY_STATES[filter].title}
            </h3>
            <p className="text-sm text-neutral-500">
              {EMPTY_STATES[filter].description}
            </p>
            {filter === "all" && (
              <Link href="/analyze">
                <Button className="mt-2">
                  <Upload className="mr-2 h-4 w-4" />
                  Start Analysis
                </Button>
              </Link>
            )}
          </div>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filtered.map((analysis) => (
            <AnalysisCard
              key={analysis.id}
              analysis={analysis}
              onDelete={fetchAnalyses}
            />
          ))}
        </div>
      )}
    </div>
  );
}
