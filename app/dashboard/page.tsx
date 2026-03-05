"use client";

import { useCallback, useEffect, useState, useTransition } from "react";
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
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { usePolling } from "@/hooks/usePolling";
import { useBatchUpload } from "@/hooks/useBatchUpload";
import { UploadCard } from "./UploadCard";
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
import { deleteAnalysesBulk, deleteFailedAnalyses } from "./actions";
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

const PAGE_SIZE_OPTIONS = [12, 24, 48, 96] as const;

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
  const { items: uploadItems, removeItem: removeUploadItem } = useBatchUpload();
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<StatusFilter>("all");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [scoreFilter, setScoreFilter] = useState<PeaceScore[]>([]);
  const [dateRange, setDateRange] = useState<DateRange>("all");
  const [sortBy, setSortBy] = useState<SortOption>("date-desc");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectMode, setSelectMode] = useState(false);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const [isBulkDeleting, startBulkDelete] = useTransition();
  const [deleteFailedOpen, setDeleteFailedOpen] = useState(false);
  const [isDeletingFailed, startDeleteFailed] = useTransition();

  // Pagination state
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState<number>(24);
  const [totalItems, setTotalItems] = useState(0);
  const [counts, setCounts] = useState<Record<StatusFilter, number>>({
    all: 0,
    processing: 0,
    completed: 0,
    failed: 0,
  });
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  const fetchAnalyses = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      params.set("page", String(page));
      params.set("pageSize", String(pageSize));
      if (filter !== "all") params.set("status", filter);
      if (debouncedSearch) params.set("search", debouncedSearch);
      if (scoreFilter.length > 0) params.set("scores", scoreFilter.join(","));
      if (dateRange !== "all") params.set("dateRange", dateRange);
      params.set("sort", sortBy);

      const res = await fetch(`/api/analyses?${params}`);
      if (res.ok) {
        const json = await res.json();
        setAnalyses(json.data);
        setTotalItems(json.total);
        setCounts(json.counts);
        // If page is beyond available pages (e.g. after deletion), go to last page
        const maxPage = Math.max(1, Math.ceil(json.total / pageSize));
        if (page > maxPage) {
          setPage(maxPage);
        }
      }
    } catch {
      // Silently fail on poll errors
    } finally {
      setIsLoading(false);
    }
  }, [page, pageSize, filter, debouncedSearch, scoreFilter, dateRange, sortBy]);

  // Auth redirect + fetch on param changes
  useEffect(() => {
    if (authStatus === "unauthenticated") {
      router.push("/login");
      return;
    }
    if (authStatus === "authenticated") {
      fetchAnalyses();
    }
  }, [authStatus, router, fetchAnalyses]);

  // Active uploads not yet in the server-fetched list.
  // Failed items are excluded — the server creates DB records for all
  // failures, so they appear as AnalysisCards after the next poll.
  const serverAnalysisIds = new Set(analyses.map((a) => a.analysisId));
  const activeUploads = uploadItems.filter(
    (item) =>
      item.status !== "completed" &&
      item.status !== "failed" &&
      !(item.analysisId && serverAnalysisIds.has(item.analysisId)),
  );

  // Auto-refresh if any jobs are in-progress, with backoff.
  // Also poll briefly after mount to catch uploads that were in-flight
  // when navigating here (XHRs complete in background, creating DB records).
  const hasActiveJobs = counts.processing > 0;
  const [recentMount, setRecentMount] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setRecentMount(false), 15_000);
    return () => clearTimeout(timer);
  }, []);

  usePolling(fetchAnalyses, {
    interval: hasActiveJobs ? 5000 : 3000,
    backoff: hasActiveJobs,
    maxInterval: 30000,
    enabled: hasActiveJobs || recentMount || activeUploads.length > 0,
  });

  // Filter & sort helpers
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
    setPage(1);
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
    setSelectedIds(new Set(analyses.map((a) => a.id)));
  }

  function deselectAll() {
    setSelectedIds(new Set());
  }

  function exitSelectMode() {
    setSelectMode(false);
    setSelectedIds(new Set());
  }

  function handleDeleteFailed() {
    startDeleteFailed(async () => {
      await deleteFailedAnalyses();
      setDeleteFailedOpen(false);
      fetchAnalyses();
    });
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
    setPage(1);
  }

  // Counts for tab labels (include client-side uploads not yet in the database)
  const filterTabs = [
    { id: "all", label: `All (${counts.all + activeUploads.length})` },
    { id: "processing", label: `Processing (${counts.processing + activeUploads.length})` },
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
          {counts.failed > 0 && (
            <Button
              variant="outline"
              onClick={() => setDeleteFailedOpen(true)}
              className="text-red-600 border-red-200 hover:bg-red-50 dark:text-red-400 dark:border-red-800 dark:hover:bg-red-950/50"
            >
              <Trash2 className="mr-2 h-4 w-4" />
              Delete Failed ({counts.failed})
            </Button>
          )}
          {counts.all > 0 && (
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
        onChange={(id) => {
          setFilter(id as StatusFilter);
          setPage(1);
        }}
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
              onValueChange={(v) => {
                setDateRange(v as DateRange);
                setPage(1);
              }}
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
              onValueChange={(v) => {
                setSortBy(v as SortOption);
                setPage(1);
              }}
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

        {/* Page size */}
        <div className="ml-auto">
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
            className="h-9 rounded-md border border-border bg-card px-2 text-sm text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
          >
            {PAGE_SIZE_OPTIONS.map((size) => (
              <option key={size} value={size}>
                {size} / page
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Bulk action bar */}
      {selectMode && analyses.length > 0 && (
        <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-2.5">
          <Button
            variant="ghost"
            size="sm"
            onClick={selectedIds.size === analyses.length ? deselectAll : selectAll}
          >
            {selectedIds.size === analyses.length ? "Deselect all" : "Select all"}
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

      {/* Delete failed confirmation dialog */}
      <AlertDialog open={deleteFailedOpen} onOpenChange={setDeleteFailedOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete all failed analyses</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete {counts.failed} failed {counts.failed === 1 ? "analysis" : "analyses"}? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeletingFailed}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={handleDeleteFailed}
              disabled={isDeletingFailed}
            >
              {isDeletingFailed ? "Deleting..." : "Delete All Failed"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Cards grid or empty state */}
      {analyses.length === 0 && (filter === "all" || filter === "processing" ? activeUploads.length === 0 : true) ? (
        <Card className="py-12 text-center">
          <div className="flex flex-col items-center gap-3">
            {counts.all > 0 && hasActiveFilters ? (
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
        <>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {(filter === "all" || filter === "processing") &&
              activeUploads.map((item) => (
                <UploadCard
                  key={item.id}
                  item={item}
                  onRemove={removeUploadItem}
                />
              ))}
            {analyses.map((analysis) => (
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

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-4 pt-2">
              <Button
                variant="outline"
                size="sm"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                <ChevronLeft className="mr-1 h-4 w-4" />
                Previous
              </Button>
              <span className="text-sm text-muted-foreground">
                Page {page} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
                <ChevronRight className="ml-1 h-4 w-4" />
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
