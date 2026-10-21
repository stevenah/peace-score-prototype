"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  FlaskConical,
  Loader2,
  RefreshCw,
  XCircle,
} from "lucide-react";
import { Alert } from "@/components/ui/Alert";
import { Button } from "@/components/ui/Button";
import { List } from "@/components/ui/List";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Progress } from "@/components/ui/Progress";
import { Skeleton } from "@/components/ui/Skeleton";
import { Tabs } from "@/components/ui/Tabs";
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { usePolling } from "@/hooks/usePolling";
import { formatFileSize } from "@/lib/utils";
import {
  MODE_DESCRIPTIONS,
  MODE_LABELS,
  describeParams,
  fmtElapsed,
  fmtScore,
  heldOutBlocker,
  isActive,
  jobSeconds,
  progressFraction,
  trainingSummary,
  type EvalJob,
  type EvalKind,
  type EvalKindInfo,
  type Experiment,
  type ModelInfo,
  type ModelsResponse,
  type Serving,
  type ServingMode,
} from "@/lib/model-admin";
import { EvalResult } from "./EvalResult";

const POLL_MS = 2000;

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/admin/ml/${path}`, {
    ...init,
    headers: init?.body ? { "content-type": "application/json" } : undefined,
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.error || `Request failed (${res.status})`);
  return data as T;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong";
}

// -- eval menus -----------------------------------------------------------------

interface EvalChoice {
  key: string;
  kind: EvalKind;
  label: string;
  hint: string;
  params: Record<string, string | number>;
  /** Why it cannot be started, shown in place of the hint. */
  blocked: string | null;
}

function kindBlocker(kinds: EvalKindInfo[], kind: EvalKind): string | null {
  const info = kinds.find((k) => k.kind === kind);
  if (!info) return "Not offered by this backend.";
  return info.available ? null : info.reason;
}

function modelChoices(model: ModelInfo, kinds: EvalKindInfo[]): EvalChoice[] {
  const unloadable = model.servable ? null : "This bundle cannot be loaded.";
  return [
    {
      key: "selftest",
      kind: "selftest",
      label: "Self-test",
      hint: "File hashes, reference logits, live path. A few seconds.",
      params: {},
      blocked: kindBlocker(kinds, "selftest"),
    },
    {
      key: "latency-quick",
      kind: "latency",
      label: "Latency, quick",
      hint: "60 synthetic frames back to back. About a minute.",
      params: { mode: "screen", frames: 60 },
      blocked: kindBlocker(kinds, "latency") ?? unloadable,
    },
    {
      key: "latency-sustained",
      kind: "latency",
      label: "Latency, sustained",
      hint: "2 streams at 2 frames/s, 2 minutes with and without the model.",
      params: { mode: "sustained", minutes: 2, streams: 2 },
      blocked: kindBlocker(kinds, "latency") ?? unloadable,
    },
    {
      key: "accuracy-heldout",
      kind: "accuracy",
      label: "Held-out accuracy",
      hint:
        model.training.fold != null
          ? `Dev fold ${model.training.fold}, which this model never trained on.`
          : "The dev fold this model never trained on.",
      params: { split: "heldout" },
      blocked: kindBlocker(kinds, "accuracy") ?? unloadable ?? heldOutBlocker(model),
    },
    {
      key: "accuracy-shift",
      kind: "accuracy",
      label: "Accuracy on the shift set",
      hint: "Legacy-cohort clips from another processor layout.",
      params: { split: "shift" },
      blocked: kindBlocker(kinds, "accuracy") ?? unloadable,
    },
  ];
}

function experimentChoices(exp: Experiment, kinds: EvalKindInfo[]): EvalChoice[] {
  const uncalibrated = exp.has_calibration ? null : "Needs calibration.json: run calibrate.py on this run first.";
  return [
    {
      key: "oof",
      kind: "oof_metrics",
      label: "Out-of-fold metrics",
      hint: "evaluate.py on the saved predictions.",
      params: {},
      blocked: kindBlocker(kinds, "oof_metrics") ?? uncalibrated,
    },
    {
      key: "gates",
      kind: "gates_report",
      label: "Gates report",
      hint: "evaluate.py --report: gates and the GO / NO-GO outcome.",
      params: {},
      blocked: kindBlocker(kinds, "gates_report") ?? uncalibrated,
    },
    {
      key: "replay-placeholder",
      kind: "replay",
      label: "Procedure replay",
      hint: "stream_eval.py with placeholder transit frames.",
      params: { transit: "placeholder" },
      blocked: kindBlocker(kinds, "replay") ?? uncalibrated,
    },
    {
      key: "replay-model",
      kind: "replay",
      label: "Procedure replay, model transit",
      hint: "The fold models classify the transit frames. Slower.",
      params: { transit: "model" },
      blocked: kindBlocker(kinds, "replay") ?? uncalibrated,
    },
  ];
}

function EvalMenu({
  choices,
  disabled,
  onStart,
}: {
  choices: EvalChoice[];
  disabled: boolean;
  onStart: (choice: EvalChoice) => void;
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" disabled={disabled}>
          <FlaskConical />
          Evaluate
          <ChevronDown />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-80">
        {choices.map((choice) => (
          <DropdownMenuItem
            key={choice.key}
            disabled={choice.blocked != null}
            onSelect={() => onStart(choice)}
            className="flex-col items-start gap-0.5"
          >
            <span className="font-medium">{choice.label}</span>
            <span className="text-xs text-muted-foreground">{choice.blocked ?? choice.hint}</span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

// -- serving --------------------------------------------------------------------

function ServingCard({
  serving,
  models,
  canChange,
  busy,
  onMode,
  onUseDefault,
}: {
  serving: Serving;
  models: ModelInfo[];
  canChange: boolean;
  busy: boolean;
  onMode: (mode: ServingMode) => void;
  onUseDefault: () => void;
}) {
  const loaded = models.find((m) => m.version === serving.loaded_version);
  const waiting = serving.selected_version ?? serving.default.version;
  // With nothing chosen and no default, "no model" is the expected state, not a fault.
  const failure = serving.mock || !waiting ? null : serving.failure;

  let model: React.ReactNode;
  if (serving.mock) {
    model = "Mock models are in use (PEACE_USE_MOCK_MODELS), so no bundle is loaded.";
  } else if (serving.loaded_version) {
    model = (
      <>
        <span className="font-mono font-medium text-foreground">{serving.loaded_version}</span>
        {loaded?.arch && ` · ${loaded.arch} at ${loaded.input_size} px`}
        {serving.device && ` · loaded on ${serving.device}`}
      </>
    );
  } else if (failure) {
    model = null;
  } else if (waiting) {
    model = (
      <>
        <span className="font-mono font-medium text-foreground">{waiting}</span> · loads{" "}
        {serving.mode === "off" ? "when station recognition is switched on" : "with the next live session"}
      </>
    );
  } else {
    model = "None. Choose one below.";
  }

  const source = serving.selected_version
    ? "Chosen on this page."
    : serving.default.kind === "lock"
      ? "The default pinned by BUNDLE.lock."
      : serving.default.kind === "env"
        ? "The default set by PEACE_LANDMARK_BUNDLE_DIR."
        : null;

  return (
    <Card className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
        <div>
          <h2 className="text-sm font-semibold">Station recognition</h2>
          <p className="mt-1 max-w-xl text-sm text-muted-foreground">
            {MODE_DESCRIPTIONS[serving.mode]} A change applies to live sessions started from then on.
          </p>
        </div>
        <Tabs
          tabs={(Object.keys(MODE_LABELS) as ServingMode[]).map((id) => ({ id, label: MODE_LABELS[id] }))}
          activeTab={serving.mode}
          onChange={(id) => id !== serving.mode && onMode(id as ServingMode)}
          className={canChange && !busy ? "" : "pointer-events-none opacity-60"}
        />
      </div>

      <div className="border-t border-border pt-4">
        <h2 className="text-sm font-semibold">Model in use</h2>
        {model && <p className="mt-1 text-sm text-muted-foreground">{model}</p>}
        {failure && (
          <Alert variant="destructive" icon={XCircle} className="mt-2">
            No landmark model is loaded: {failure}
          </Alert>
        )}
        {(source || serving.selected_version) && !serving.mock && (
          <p className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            {source}
            {serving.selected_version && (
              <Button variant="link" size="xs" className="h-auto p-0" disabled={!canChange || busy} onClick={onUseDefault}>
                {serving.default.version ? `Go back to the default (${serving.default.version})` : "Clear the choice"}
              </Button>
            )}
          </p>
        )}
      </div>
    </Card>
  );
}

// -- rows -----------------------------------------------------------------------

function ModelRow({
  model,
  active,
  kinds,
  canChange,
  canActivate,
  busy,
  onActivate,
  onStart,
}: {
  model: ModelInfo;
  active: boolean;
  kinds: EvalKindInfo[];
  canChange: boolean;
  canActivate: boolean;
  busy: boolean;
  onActivate: () => void;
  onStart: (choice: EvalChoice) => void;
}) {
  const facts = [
    model.arch && `${model.arch} at ${model.input_size} px`,
    trainingSummary(model),
    model.size_bytes != null && formatFileSize(model.size_bytes),
    model.modified_at && new Date(model.modified_at).toLocaleDateString(),
  ].filter(Boolean);

  return (
    <li className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm font-semibold">{model.version}</span>
          {active && <Badge variant="primary">Active</Badge>}
          {!model.servable && (
            <Badge variant="destructive">
              <Ban className="h-3 w-3" aria-hidden="true" />
              Cannot be served
            </Badge>
          )}
          {model.servable && !model.calibrated && (
            <Badge variant="warning">
              <AlertTriangle className="h-3 w-3" aria-hidden="true" />
              Uncalibrated
            </Badge>
          )}
        </div>
        <p className="text-sm text-muted-foreground">{facts.join(" · ")}</p>
        {model.problem && <p className="text-sm text-destructive">{model.problem}</p>}
        <p className="truncate font-mono text-xs text-muted-foreground/80" title={model.path}>
          {model.path}
        </p>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <EvalMenu choices={modelChoices(model, kinds)} disabled={!canChange || busy} onStart={onStart} />
        {!active && (
          <Button size="sm" disabled={!canActivate || !model.servable || busy} onClick={onActivate}>
            Make active
          </Button>
        )}
      </div>
    </li>
  );
}

function ExperimentRow({
  exp,
  kinds,
  canChange,
  busy,
  onStart,
}: {
  exp: Experiment;
  kinds: EvalKindInfo[];
  canChange: boolean;
  busy: boolean;
  onStart: (choice: EvalChoice) => void;
}) {
  const folds = exp.folds.length;
  const facts = [
    exp.candidate,
    `${folds} fold${folds === 1 ? "" : "s"}${exp.has_all_dev ? " and the all-dev model" : ""}`,
    exp.mean_clip_macro_f1 != null && `held-out clip macro-F1 ${fmtScore(exp.mean_clip_macro_f1)} (training log)`,
    exp.outcome && `gates outcome ${exp.outcome}`,
  ].filter(Boolean);

  return (
    <li className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="min-w-0 flex-1 space-y-1">
        <span className="font-mono text-sm font-semibold">{exp.name}</span>
        <p className="text-sm text-muted-foreground">{facts.join(" · ")}</p>
        {!exp.has_calibration && (
          <p className="text-xs text-muted-foreground">
            Not calibrated yet, so the script evals cannot run on it.
          </p>
        )}
      </div>
      <div className="shrink-0">
        <EvalMenu choices={experimentChoices(exp, kinds)} disabled={!canChange || busy} onStart={onStart} />
      </div>
    </li>
  );
}

function JobStatus({ job }: { job: EvalJob }) {
  const fraction = progressFraction(job);
  const [Icon, tone, label] =
    job.status === "queued"
      ? [Clock, "text-muted-foreground", "Queued"]
      : job.status === "running"
        ? [Loader2, "text-primary", fraction != null ? `Running, ${Math.round(fraction * 100)}%` : "Running"]
        : job.status === "cancelled"
          ? [Ban, "text-muted-foreground", "Cancelled"]
          : job.status === "failed"
            ? [XCircle, "text-destructive", "Did not finish"]
            : job.verdict === "fail"
              ? [XCircle, "text-destructive", "Failed"]
              : [CheckCircle2, "text-success", job.verdict === "pass" ? "Passed" : "Done"];
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${tone}`}>
      <Icon className={`h-4 w-4 shrink-0 ${job.status === "running" ? "animate-spin" : ""}`} aria-hidden="true" />
      {label}
    </span>
  );
}

function JobRow({
  job,
  detail,
  open,
  now,
  canChange,
  onToggle,
  onCancel,
}: {
  job: EvalJob;
  detail: EvalJob | null;
  open: boolean;
  now: number;
  canChange: boolean;
  onToggle: () => void;
  onCancel: () => void;
}) {
  const seconds = jobSeconds(job, now);
  const fraction = progressFraction(job);
  const params = describeParams(job);

  return (
    <li>
      <div className="flex flex-col gap-2 px-5 py-3 sm:flex-row sm:items-center sm:gap-4">
        <button
          type="button"
          onClick={onToggle}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-start gap-2 rounded-md text-left outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {open ? (
            <ChevronDown className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
          ) : (
            <ChevronRight className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
          )}
          <span className="min-w-0 space-y-0.5">
            <span className="block text-sm">
              <span className="font-medium">{job.label}</span>
              <span className="text-muted-foreground"> on </span>
              <span className="font-mono text-[13px]">{job.target.id}</span>
              {params && <span className="text-muted-foreground"> · {params}</span>}
            </span>
            <span className="block text-sm text-muted-foreground">
              {job.headline ?? job.error ?? (job.status === "queued" ? "Waiting for the eval ahead of it." : "")}
            </span>
          </span>
        </button>
        <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 pl-6 sm:pl-0">
          <span className="text-xs text-muted-foreground tabular-nums">
            {seconds != null && `${fmtElapsed(seconds)} · `}
            {new Date(job.created_at).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}
          </span>
          <JobStatus job={job} />
          {isActive(job) && (
            <Button variant="outline" size="sm" disabled={!canChange} onClick={onCancel}>
              Cancel
            </Button>
          )}
        </div>
      </div>

      {job.status === "running" && (
        <div className="px-5 pb-3">
          {fraction != null ? (
            <Progress value={fraction * 100} />
          ) : (
            <div className="h-2 w-full animate-pulse rounded-full bg-muted" />
          )}
          {job.progress && (
            <p className="mt-1 text-xs text-muted-foreground tabular-nums">
              {job.progress.done.toLocaleString()} of {job.progress.total.toLocaleString()} {job.progress.label}
            </p>
          )}
        </div>
      )}

      {open && (
        <div className="space-y-4 border-t border-border bg-card px-5 py-4">
          {!detail ? (
            <Skeleton className="h-24" />
          ) : (
            <>
              <EvalResult job={detail} />
              {detail.log && (
                <details open={detail.status !== "succeeded"}>
                  <summary className="cursor-pointer text-sm font-medium">Log</summary>
                  <pre className="mt-2 max-h-72 overflow-auto rounded-lg bg-muted p-3 text-xs leading-relaxed whitespace-pre-wrap">
                    {detail.log}
                  </pre>
                </details>
              )}
            </>
          )}
        </div>
      )}
    </li>
  );
}

function Section({ title, count, children }: { title: string; count?: number; children: React.ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="flex items-center gap-2 text-base font-semibold">
        {title}
        {count != null && <Badge>{count}</Badge>}
      </h2>
      {children}
    </section>
  );
}

// -- the panel ------------------------------------------------------------------

export default function ModelsPanel() {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [jobs, setJobs] = useState<EvalJob[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);
  const [detail, setDetail] = useState<EvalJob | null>(null);
  const [confirming, setConfirming] = useState<ModelInfo | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const openIdRef = useRef(openId);
  const hadActive = useRef(false);

  useEffect(() => {
    openIdRef.current = openId;
  }, [openId]);

  const loadModels = useCallback(async () => {
    try {
      setData(await api<ModelsResponse>("models"));
      setLoadError(null);
    } catch (err) {
      setLoadError(message(err));
    }
  }, []);

  const loadDetail = useCallback(async (id: string) => {
    try {
      const job = await api<EvalJob>(`evals/${id}`);
      if (openIdRef.current === id) setDetail(job);
    } catch {
      // The row still shows the job's status; the detail fills in on the next poll.
    }
  }, []);

  /** Jobs, the open job's detail and the clock, in one tick. */
  const refreshJobs = useCallback(async () => {
    let list: EvalJob[];
    try {
      list = (await api<{ jobs: EvalJob[] }>("evals?limit=30")).jobs;
    } catch {
      return;
    }
    setJobs(list);
    setNow(Date.now());
    if (openIdRef.current) await loadDetail(openIdRef.current);
    const active = list.some(isActive);
    if (hadActive.current && !active) loadModels(); // a script eval may have written new reports
    hadActive.current = active;
  }, [loadDetail, loadModels]);

  // Polls until the first load succeeds (backing off while the backend is
  // down), then only while an eval is queued or running: an idle tab must not
  // keep a scale-to-zero backend awake.
  const loaded = data !== null;
  const tick = useCallback(async () => {
    if (!loaded) await loadModels();
    await refreshJobs();
  }, [loaded, loadModels, refreshJobs]);

  usePolling(tick, { interval: POLL_MS, backoff: !loaded, enabled: !loaded || jobs.some(isActive) });

  /** Run one change against the backend; a refusal becomes the notice. */
  async function act(change: () => Promise<void>) {
    setBusy(true);
    setNotice(null);
    try {
      await change();
    } catch (err) {
      setNotice(message(err));
    } finally {
      setBusy(false);
    }
  }

  const applyServing = (serving: Serving) => setData((d) => (d ? { ...d, serving } : d));

  const setMode = (mode: ServingMode) =>
    act(async () =>
      applyServing(await api<Serving>("models/mode", { method: "PUT", body: JSON.stringify({ mode }) })),
    );

  const activate = (model: ModelInfo) =>
    act(async () =>
      applyServing(
        await api<Serving>("models/active", { method: "POST", body: JSON.stringify({ version: model.version }) }),
      ),
    );

  const useDefault = () =>
    act(async () => applyServing(await api<Serving>("models/active", { method: "DELETE" })));

  const startEval = (target: string, choice: EvalChoice) =>
    act(async () => {
      const job = await api<EvalJob>("evals", {
        method: "POST",
        body: JSON.stringify({ kind: choice.kind, target, params: choice.params }),
      });
      setOpenId(job.id);
      setDetail(null);
      await refreshJobs();
    });

  const cancel = (job: EvalJob) =>
    act(async () => {
      await api<EvalJob>(`evals/${job.id}/cancel`, { method: "POST" });
      await refreshJobs();
    });

  function toggle(job: EvalJob) {
    if (openId === job.id) {
      setOpenId(null);
      return;
    }
    setOpenId(job.id);
    setDetail(null);
    loadDetail(job.id);
  }

  if (!data && !loadError) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-40" />
        <Skeleton className="h-56" />
      </div>
    );
  }

  if (!data) {
    return (
      <Card className="space-y-3 p-10 text-center">
        <XCircle className="mx-auto h-9 w-9 text-muted-foreground/60" aria-hidden="true" />
        <p className="text-sm font-medium">The model list could not be loaded</p>
        <p className="text-sm text-muted-foreground">{loadError}</p>
        <Button variant="outline" size="sm" onClick={loadModels}>
          Try again
        </Button>
      </Card>
    );
  }

  const { serving, models, experiments, eval_kinds: kinds } = data;
  const canChange = data.admin_enabled;
  const activeVersion = serving.mock
    ? null
    : (serving.selected_version ?? serving.loaded_version ?? serving.default.version);

  return (
    <div className="space-y-8">
      {!canChange && (
        <Alert icon={Ban}>
          Model management is switched off on this backend (PEACE_MODEL_ADMIN=false). This page shows what is
          served; changing it and running evals happen on a backend that has it on.
        </Alert>
      )}
      {notice && (
        <Alert
          variant="destructive"
          action={
            <Button variant="link" size="xs" className="h-auto p-0 text-current" onClick={() => setNotice(null)}>
              Dismiss
            </Button>
          }
        >
          {notice}
        </Alert>
      )}

      <ServingCard
        serving={serving}
        models={models}
        canChange={canChange}
        busy={busy}
        onMode={setMode}
        onUseDefault={useDefault}
      />

      <Section title="Landmark models" count={models.length}>
        {models.length === 0 ? (
          <Card className="p-8 text-center text-sm text-muted-foreground">
            No bundles found. Export one with{" "}
            <code className="font-mono text-xs">python -m training.landmarks.export_bundle</code>, or put a bundle
            folder in <code className="font-mono text-xs">ml-backend/app/models/landmarks/</code>.
          </Card>
        ) : (
          <List>
            {models.map((model) => (
              <ModelRow
                key={model.version}
                model={model}
                active={model.version === activeVersion}
                kinds={kinds}
                canChange={canChange}
                canActivate={canChange && !serving.mock}
                busy={busy}
                onActivate={() => (model.calibrated ? activate(model) : setConfirming(model))}
                onStart={(choice) => startEval(model.version, choice)}
              />
            ))}
          </List>
        )}
      </Section>

      {experiments.length > 0 && (
        <Section title="Training runs" count={experiments.length}>
          <List>
            {experiments.map((exp) => (
              <ExperimentRow
                key={exp.name}
                exp={exp}
                kinds={kinds}
                canChange={canChange}
                busy={busy}
                onStart={(choice) => startEval(exp.name, choice)}
              />
            ))}
          </List>
        </Section>
      )}

      <Section title="Evaluations" count={jobs.length || undefined}>
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1">
          <p className="text-sm text-muted-foreground">
            One eval runs at a time. Start one from a model&apos;s or a run&apos;s Evaluate menu.
          </p>
          <Button variant="ghost" size="sm" onClick={() => { loadModels(); refreshJobs(); }}>
            <RefreshCw />
            Refresh
          </Button>
        </div>
        {jobs.length === 0 ? (
          <Card className="p-8 text-center text-sm text-muted-foreground">No evals have been run yet.</Card>
        ) : (
          <List>
            {jobs.map((job) => (
              <JobRow
                key={job.id}
                job={job}
                detail={openId === job.id ? detail : null}
                open={openId === job.id}
                now={now}
                canChange={canChange}
                onToggle={() => toggle(job)}
                onCancel={() => cancel(job)}
              />
            ))}
          </List>
        )}
      </Section>

      <AlertDialog open={!!confirming} onOpenChange={(open) => !open && setConfirming(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Serve an uncalibrated model?</AlertDialogTitle>
            <AlertDialogDescription>
              <span className="font-mono text-foreground">{confirming?.version}</span> was exported without
              calibration: its temperature and thresholds are defaults, so the stations it marks as observed
              are not validated. Use it for testing only.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (confirming) activate(confirming);
                setConfirming(null);
              }}
            >
              Make active
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
