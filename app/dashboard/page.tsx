"use client";

import { useCallback, useEffect, useMemo, useState, useTransition } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import {
  Upload,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Search,
  ArrowUpDown,
  Calendar,
  X,
  CheckSquare,
  Trash2,
} from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { Tabs } from "@/components/ui/Tabs";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import { AnalysisCard } from "./AnalysisCard";
import { deleteAnalysesBulk } from "./actions";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { PEACE_SCORE_COLORS, PEACE_SCORE_LABELS } from "@/lib/constants";
import type { AnalysisRecord, PeaceScore } from "@/lib/types";
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

type StatusFilter = "all" | "processing" | "completed" | "failed";
type DateRange = "all" | "today" | "week" | "month";
type SortOption =
  | "date-desc"
  | "date-asc"
  | "score-desc"
  | "score-asc"
  | "name-asc";

const DATE_RANGE_LABELS: Record<DateRange, string> = {
  all: "All Time",
  today: "Today",
  week: "This Week",
  month: "This Month",
};

const SORT_LABELS: Record<SortOption, string> = {
  "date-desc": "Newest",
  "date-asc": "Oldest",
  "score-desc": "Score (High \u2192 Low)",
  "score-asc": "Score (Low \u2192 High)",
  "name-asc": "Name (A \u2192 Z)",
};

const EMPTY_STATES: Record<
  StatusFilter,
  { icon: React.ReactNode; title: string; description: string }
> = {
  all: {
    icon: <Upload className="h-12 w-12 text-muted-foreground/40" />,
    title: "No analyses yet",
    description: "Upload an endoscopy video to get started with PEACE scoring.",
  },
  processing: {
    icon: <Clock className="h-12 w-12 text-muted-foreground/40" />,
    title: "No active analyses",
    description: "All your analyses have completed processing.",
  },
  completed: {
    icon: <CheckCircle2 className="h-12 w-12 text-muted-foreground/40" />,
    title: "No completed analyses",
    description: "Your analyses are still being processed.",
  },
  failed: {
    icon: <AlertTriangle className="h-12 w-12 text-muted-foreground/40" />,
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
  const [search, setSearch] = useState("");
  const [scoreFilter, setScoreFilter] = useState<PeaceScore[]>([]);
  const [dateRange, setDateRange] = useState<DateRange>("all");
  const [sortBy, setSortBy] = useState<SortOption>("date-desc");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectMode, setSelectMode] = useState(false);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [isBulkDeleting, startBulkDelete] = useTransition();

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

  // Auto-refresh if any jobs are in-progress, with backoff.
  // Also poll briefly after mount to catch uploads that were in-flight
  // when navigating here (XHRs complete in background, creating DB records).
  const hasActiveJobs = analyses.some(
    (a) => a.status === "processing" || a.status === "queued",
  );
  const [recentMount, setRecentMount] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setRecentMount(false), 15_000);
    return () => clearTimeout(timer);
  }, []);

  usePolling(fetchAnalyses, {
    interval: hasActiveJobs ? 5000 : 3000,
    backoff: hasActiveJobs,
    maxInterval: 30000,
    enabled: hasActiveJobs || recentMount,
  });

  // Filter & sort
  const hasActiveFilters =
    search !== "" ||
    scoreFilter.length > 0 ||
    dateRange !== "all" ||
    sortBy !== "date-desc";

  function clearFilters() {
    setSearch("");
    setScoreFilter([]);
    setDateRange("all");
    setSortBy("date-desc");
  }

  function toggleSelect(id: string) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function selectAll() {
    setSelectedIds(new Set(filtered.map((a) => a.id)));
  }

  function deselectAll() {
    setSelectedIds(new Set());
  }

  function exitSelectMode() {
    setSelectMode(false);
    setSelectedIds(new Set());
  }

  function handleBulkDelete() {
    startBulkDelete(async () => {
      await deleteAnalysesBulk([...selectedIds]);
      setBulkDeleteOpen(false);
      setSelectedIds(new Set());
      setSelectMode(false);
      fetchAnalyses();
    });
  }

  function toggleScore(score: PeaceScore) {
    setScoreFilter((prev) =>
      prev.includes(score) ? prev.filter((s) => s !== score) : [...prev, score],
    );
  }

  const filtered = useMemo(() => {
    let result = [...analyses];

    // Status filter
    if (filter !== "all") {
      result = result.filter((a) => {
        if (filter === "processing")
          return a.status === "processing" || a.status === "queued";
        return a.status === filter;
      });
    }

    // Search by filename
    if (search) {
      const q = search.toLowerCase();
      result = result.filter((a) => a.filename.toLowerCase().includes(q));
    }

    // Score filter
    if (scoreFilter.length > 0) {
      result = result.filter(
        (a) =>
          a.overallScore !== null &&
          scoreFilter.includes(a.overallScore as PeaceScore),
      );
    }

    // Date range
    if (dateRange !== "all") {
      const now = new Date();
      let cutoff: Date;
      switch (dateRange) {
        case "today":
          cutoff = new Date(now.getFullYear(), now.getMonth(), now.getDate());
          break;
        case "week":
          cutoff = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
          break;
        case "month":
          cutoff = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
          break;
      }
      result = result.filter((a) => new Date(a.createdAt) >= cutoff);
    }

    // Sort
    result.sort((a, b) => {
      switch (sortBy) {
        case "date-desc":
          return (
            new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
          );
        case "date-asc":
          return (
            new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
          );
        case "score-desc":
          return (b.overallScore ?? -1) - (a.overallScore ?? -1);
        case "score-asc":
          return (a.overallScore ?? -1) - (b.overallScore ?? -1);
        case "name-asc":
          return a.filename.localeCompare(b.filename);
        default:
          return 0;
      }
    });

    return result;
  }, [analyses, filter, search, scoreFilter, dateRange, sortBy]);

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
          <h1 className="text-2xl font-bold text-foreground">
            Welcome back, {session?.user?.name || session?.user?.email}
          </h1>
        </div>
        <div className="flex items-center gap-2">
          {analyses.length > 0 && (
            <Button
              variant={selectMode ? "secondary" : "outline"}
              onClick={selectMode ? exitSelectMode : () => setSelectMode(true)}
            >
              <CheckSquare className="mr-2 h-4 w-4" />
              {selectMode ? "Cancel" : "Select"}
            </Button>
          )}
          <Link href="/analyze">
            <Button>
              <Upload className="mr-2 h-4 w-4" />
              New Analysis
            </Button>
          </Link>
        </div>
      </div>

      {/* Filter tabs */}
      <Tabs
        tabs={filterTabs}
        activeTab={filter}
        onChange={(id) => setFilter(id as StatusFilter)}
      />

      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search files..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="h-9 rounded-md border border-border bg-card pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          />
        </div>

        {/* Score pills */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground">Score:</span>
          {([0, 1, 2, 3] as PeaceScore[]).map((score) => (
            <button
              key={score}
              type="button"
              onClick={() => toggleScore(score)}
              className={cn(
                "flex h-7 min-w-8 items-center justify-center rounded-full px-2 text-xs font-medium transition-colors",
                scoreFilter.includes(score)
                  ? "text-white shadow-sm"
                  : "bg-muted text-muted-foreground hover:text-foreground",
              )}
              style={
                scoreFilter.includes(score)
                  ? { backgroundColor: PEACE_SCORE_COLORS[score] }
                  : undefined
              }
              title={PEACE_SCORE_LABELS[score]}
            >
              {score}
            </button>
          ))}
        </div>

        {/* Date range dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              className={cn(
                "inline-flex h-9 items-center gap-1.5 rounded-md border border-border px-3 text-sm transition-colors hover:bg-muted",
                dateRange !== "all"
                  ? "border-primary/50 text-foreground"
                  : "text-muted-foreground",
              )}
            >
              <Calendar className="h-3.5 w-3.5" />
              {DATE_RANGE_LABELS[dateRange]}
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            <DropdownMenuLabel>Date Range</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuRadioGroup
              value={dateRange}
              onValueChange={(v) => setDateRange(v as DateRange)}
            >
              {(Object.keys(DATE_RANGE_LABELS) as DateRange[]).map((key) => (
                <DropdownMenuRadioItem key={key} value={key}>
                  {DATE_RANGE_LABELS[key]}
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Sort dropdown */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              className={cn(
                "inline-flex h-9 items-center gap-1.5 rounded-md border border-border px-3 text-sm transition-colors hover:bg-muted",
                sortBy !== "date-desc"
                  ? "border-primary/50 text-foreground"
                  : "text-muted-foreground",
              )}
            >
              <ArrowUpDown className="h-3.5 w-3.5" />
              {SORT_LABELS[sortBy]}
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            <DropdownMenuLabel>Sort By</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuRadioGroup
              value={sortBy}
              onValueChange={(v) => setSortBy(v as SortOption)}
            >
              {(Object.keys(SORT_LABELS) as SortOption[]).map((key) => (
                <DropdownMenuRadioItem key={key} value={key}>
                  {SORT_LABELS[key]}
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Clear filters */}
        {hasActiveFilters && (
          <button
            type="button"
            onClick={clearFilters}
            className="inline-flex h-9 items-center gap-1 rounded-md px-2 text-xs text-muted-foreground transition-colors hover:text-foreground"
          >
            <X className="h-3.5 w-3.5" />
            Clear
          </button>
        )}
      </div>

      {/* Bulk action bar */}
      {selectMode && filtered.length > 0 && (
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-2.5">
          <Button
            variant="ghost"
            size="sm"
            onClick={selectedIds.size === filtered.length ? deselectAll : selectAll}
          >
            {selectedIds.size === filtered.length ? "Deselect all" : "Select all"}
          </Button>
          <span className="text-sm text-muted-foreground">
            {selectedIds.size} selected
          </span>
          <div className="flex-1" />
          <Button
            variant="destructive"
            size="sm"
            disabled={selectedIds.size === 0}
            onClick={() => setBulkDeleteOpen(true)}
          >
            <Trash2 className="mr-1.5 h-3.5 w-3.5" />
            Delete {selectedIds.size > 0 ? `(${selectedIds.size})` : ""}
          </Button>
        </div>
      )}

      {/* Bulk delete confirmation dialog */}
      <AlertDialog open={bulkDeleteOpen} onOpenChange={setBulkDeleteOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {selectedIds.size} {selectedIds.size === 1 ? "analysis" : "analyses"}</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete {selectedIds.size} selected {selectedIds.size === 1 ? "analysis" : "analyses"}? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isBulkDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={handleBulkDelete}
              disabled={isBulkDeleting}
            >
              {isBulkDeleting ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Cards grid or empty state */}
      {filtered.length === 0 ? (
        <Card className="py-12 text-center">
          <div className="flex flex-col items-center gap-3">
            {analyses.length > 0 && hasActiveFilters ? (
              <>
                <Search className="h-12 w-12 text-muted-foreground/40" />
                <h3 className="text-lg font-medium text-foreground/70">
                  No matching analyses
                </h3>
                <p className="text-sm text-muted-foreground">
                  Try adjusting your filters to find what you&apos;re looking
                  for.
                </p>
                <Button variant="outline" className="mt-2" onClick={clearFilters}>
                  <X className="mr-2 h-4 w-4" />
                  Clear Filters
                </Button>
              </>
            ) : (
              <>
                {EMPTY_STATES[filter].icon}
                <h3 className="text-lg font-medium text-foreground/70">
                  {EMPTY_STATES[filter].title}
                </h3>
                <p className="text-sm text-muted-foreground">
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
              </>
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
              selectMode={selectMode}
              selected={selectedIds.has(analysis.id)}
              onToggleSelect={() => toggleSelect(analysis.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
