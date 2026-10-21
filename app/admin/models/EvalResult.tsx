"use client";

import { AlertTriangle, CheckCircle2, MinusCircle, XCircle } from "lucide-react";
import { Alert } from "@/components/ui/Alert";
import {
  LIVE_FRAME_INTERVAL_MS,
  fmtElapsed,
  fmtInterval,
  fmtMs,
  fmtPercent,
  fmtScore,
  stationLabel,
  type AccuracyResult,
  type EvalJob,
  type GatesResult,
  type LatencyResult,
  type LatencyRow,
  type ReplayResult,
  type SelftestResult,
} from "@/lib/model-admin";
import { ConfusionMatrix } from "./ConfusionMatrix";

const th = "px-2 py-1.5 text-left text-xs font-medium text-muted-foreground";
const td = "px-2 py-1.5 text-sm";
const num = "text-right tabular-nums";

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-lg border border-border px-3 py-2.5">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-0.5 text-2xl font-semibold tracking-tight">{value}</p>
      {note && <p className="mt-0.5 text-xs text-muted-foreground">{note}</p>}
    </div>
  );
}

/** Status is never colour alone: an icon and a word, coloured to match. */
function Status({ state, children }: { state: "good" | "bad" | "neutral"; children: React.ReactNode }) {
  const Icon = state === "good" ? CheckCircle2 : state === "bad" ? XCircle : MinusCircle;
  const tone =
    state === "good"
      ? "text-success"
      : state === "bad"
        ? "text-destructive"
        : "text-muted-foreground";
  return (
    <span className={`inline-flex items-center gap-1.5 text-sm font-medium ${tone}`}>
      <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
      {children}
    </span>
  );
}

/** A share of 0..1 as a thin bar on a lighter step of the same accent. */
function ShareBar({ value }: { value: number | null | undefined }) {
  const pct = value == null || !Number.isFinite(value) ? 0 : Math.min(1, Math.max(0, value)) * 100;
  return (
    <span aria-hidden="true" className="block h-1.5 w-full min-w-16 bg-primary/15">
      <span className="block h-full rounded-r-[3px] bg-primary" style={{ width: `${pct}%` }} />
    </span>
  );
}

// -- self-test ------------------------------------------------------------------

function SelftestView({ result }: { result: SelftestResult }) {
  return (
    <ul className="divide-y divide-border">
      {result.checks.map((check) => (
        <li key={check.name} className="grid gap-x-4 gap-y-0.5 py-2 sm:grid-cols-[14rem_1fr]">
          <Status state={check.status === "passed" ? "good" : check.status === "failed" ? "bad" : "neutral"}>
            <span>
              {check.label}{" "}
              <span className="font-normal">
                {check.status === "passed" ? "passed" : check.status === "failed" ? "failed" : "not run"}
              </span>
            </span>
          </Status>
          <p className="text-sm text-muted-foreground">{check.detail}</p>
        </li>
      ))}
    </ul>
  );
}

// -- latency --------------------------------------------------------------------

function LatencyView({ result }: { result: LatencyResult }) {
  const { baseline, model, host } = result;
  const p95 = model.frame_ms.p95;
  const budget = p95 == null ? null : p95 / LIVE_FRAME_INTERVAL_MS;
  const fits = budget != null && budget <= 1;
  const rows: [string, LatencyRow][] = [
    ["PEACE alone", baseline],
    [`PEACE and ${model.config}`, model],
  ];

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Per frame, median" value={fmtMs(model.frame_ms.p50)} note={`PEACE alone ${fmtMs(baseline.frame_ms.p50)}`} />
        <Stat label="Per frame, 95th percentile" value={fmtMs(p95)} note={`PEACE alone ${fmtMs(baseline.frame_ms.p95)}`} />
        <Stat label="Landmark step, median" value={fmtMs(model.landmark_ms.p50)} note={`95th percentile ${fmtMs(model.landmark_ms.p95)}`} />
        <div className="rounded-lg border border-border px-3 py-2.5">
          <p className="text-xs text-muted-foreground">95th percentile, share of the frame interval</p>
          <p className="mt-0.5 text-2xl font-semibold tracking-tight">{fmtPercent(budget)}</p>
          {budget != null && (
            <>
              <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-primary/15">
                <div
                  className={`h-full rounded-full ${fits ? "bg-primary" : "bg-destructive"}`}
                  style={{ width: `${Math.min(100, budget * 100)}%` }}
                />
              </div>
              <div className="mt-1.5">
                <Status state={fits ? "good" : "bad"}>
                  <span className="text-xs font-normal">
                    {fits ? "Fits" : "Does not fit"} the {LIVE_FRAME_INTERVAL_MS} ms between live frames
                  </span>
                </Status>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className={th}>Configuration</th>
              <th className={`${th} ${num}`}>Frames</th>
              <th className={`${th} ${num}`}>Median</th>
              <th className={`${th} ${num}`}>p95</th>
              <th className={`${th} ${num}`}>p99</th>
              <th className={`${th} ${num}`}>Drift</th>
              <th className={`${th} ${num}`}>Peak memory</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, row]) => (
              <tr key={label} className="border-b border-border last:border-0">
                <td className={td}>{label}</td>
                <td className={`${td} ${num}`}>{row.frames}</td>
                <td className={`${td} ${num}`}>{fmtMs(row.frame_ms.p50)}</td>
                <td className={`${td} ${num}`}>{fmtMs(row.frame_ms.p95)}</td>
                <td className={`${td} ${num}`}>{fmtMs(row.frame_ms.p99)}</td>
                <td className={`${td} ${num}`}>{row.drift == null ? "–" : `${row.drift.toFixed(2)}×`}</td>
                <td className={`${td} ${num}`}>{Math.round(row.rss_peak_mb)} MB</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-muted-foreground">
        Measured on this machine&apos;s CPU ({host.machine}, {host.torch_threads} torch threads, torch {host.torch})
        with synthetic 1920×1080 frames
        {result.mode === "sustained"
          ? `, ${result.streams} streams at ${result.rate} frames/s each.`
          : ", one stream, back to back."}{" "}
        Drift is the last third&apos;s median over the first third&apos;s; above 1 means it slowed down. Peak
        memory is cumulative over the run.
        {result.peace === "mock" && " PEACE ran as a mock here, so its share of the time is not the real model's."}
      </p>
    </div>
  );
}

// -- accuracy (bundle_eval) and out-of-fold metrics (evaluate.py) -------------------

function AccuracyView({ result }: { result: AccuracyResult }) {
  const { clip, frame, split, acceptance } = result;
  const stations = result.stations.filter((key) => clip.per_class[key]);

  return (
    <div className="space-y-5">
      {result.warnings?.map((warning) => (
        <Alert key={warning} variant="warning" icon={AlertTriangle}>
          {warning}
        </Alert>
      ))}

      <p className="text-sm text-muted-foreground">
        {clip.n_clips.toLocaleString()} clips from {clip.n_patients} patients
        {split ? `, ${split.label}` : result.folds ? `, out of fold across ${result.folds.length} folds` : ""}
        {split ? `. ${split.n_frames.toLocaleString()} frames` : ""}
        {result.device ? ` on ${result.device}` : ""}
        {result.seconds != null ? ` in ${fmtElapsed(result.seconds)}` : ""}.
        {result.calibration && !result.calibration.quality_gate && " Quality gate off."}
      </p>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Clip macro-F1" value={fmtScore(clip.macro_f1.value)} note={`95% CI ${fmtInterval(clip.macro_f1)}`} />
        <Stat label="Balanced accuracy" value={fmtScore(clip.bal_acc.value)} note={`95% CI ${fmtInterval(clip.bal_acc)}`} />
        {clip.top2_acc != null && (
          <Stat label="Top-2 accuracy" value={fmtScore(clip.top2_acc)} note="True station among the two most likely" />
        )}
        {result.patient_level?.macro_f1 != null && (
          <Stat label="Patient-level macro-F1" value={fmtScore(result.patient_level.macro_f1)} note="Every patient weighs the same" />
        )}
        {result.clip_merged?.macro_f1 && (
          <Stat label="Merged-hierarchy macro-F1" value={fmtScore(result.clip_merged.macro_f1.value)} note={`95% CI ${fmtInterval(result.clip_merged.macro_f1)}`} />
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className={th}>Station</th>
              <th className={`${th} ${num}`}>Clips</th>
              <th className={th} colSpan={2}>Precision</th>
              <th className={th} colSpan={2}>Recall</th>
              <th className={`${th} ${num}`}>F1</th>
            </tr>
          </thead>
          <tbody>
            {stations.map((key) => {
              const row = clip.per_class[key]!;
              return (
                <tr key={key} className="border-b border-border last:border-0">
                  <td className={td}>
                    <span className="inline-block w-6 text-muted-foreground tabular-nums">
                      {result.stations.indexOf(key) + 1}
                    </span>
                    {stationLabel(key)}
                  </td>
                  <td className={`${td} ${num}`}>{row.n}</td>
                  <td className={`${td} ${num} w-14`}>{fmtScore(row.precision)}</td>
                  <td className="w-28 px-2"><ShareBar value={row.precision} /></td>
                  <td className={`${td} ${num} w-14`}>{fmtScore(row.recall)}</td>
                  <td className="w-28 px-2"><ShareBar value={row.recall} /></td>
                  <td className={`${td} ${num}`}>{fmtScore(row.f1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <ConfusionMatrix stations={result.stations} confusion={clip.confusion} />

      {(frame || acceptance) && (
        <div className="space-y-1 text-sm text-muted-foreground">
          {frame && (
            <p>
              <span className="font-medium text-foreground">Frames.</span> Accuracy{" "}
              {fmtScore(frame.centre_quality.accuracy)} on {frame.centre_quality.n.toLocaleString()}{" "}
              quality-passing centre frames, {fmtScore(frame.all_frames.accuracy)} on all{" "}
              {frame.all_frames.n.toLocaleString()}. Calibration error (ECE) {fmtScore(frame.calibration.ece)},
              NLL {fmtScore(frame.calibration.nll)}.
            </p>
          )}
          {acceptance && (
            <p>
              <span className="font-medium text-foreground">Served thresholds.</span>{" "}
              {fmtPercent(acceptance.share.ok)} of frames pass as ok, {fmtPercent(acceptance.share.uncertain)}{" "}
              are uncertain and {fmtPercent(acceptance.share.low_quality)} low quality. Of the ok frames,{" "}
              {fmtPercent(acceptance.precision_ok)} name the right station. Every frame here shows one of
              the ten stations, so a live stream will do worse.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

// -- gates report ---------------------------------------------------------------

function gateState(status: string | null): "good" | "bad" | "neutral" {
  if (status === "PASS" || status === "GO") return "good";
  if (status === "FAIL" || status === "NO-GO") return "bad";
  return "neutral";
}

function GatesView({ result }: { result: GatesResult }) {
  const manualOnly = result.auto_enabled
    ? result.stations.filter((_, i) => result.auto_enabled![i] === false)
    : [];
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-1">
        <Status state={gateState(result.outcome)}>Outcome {result.outcome ?? "unknown"}</Status>
        {result.clip_macro_f1 && (
          <span className="text-sm text-muted-foreground">
            Clip macro-F1 {fmtScore(result.clip_macro_f1.value)} (95% CI {fmtInterval(result.clip_macro_f1)})
          </span>
        )}
      </div>
      <table className="w-full max-w-md">
        <thead>
          <tr className="border-b border-border">
            <th className={th}>Gate</th>
            <th className={th}>Status</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(result.gates).map(([name, gate]) => (
            <tr key={name} className="border-b border-border last:border-0">
              <td className={`${td} font-mono text-xs`}>{name}</td>
              <td className={td}>
                <Status state={gateState(gate.status)}>{gate.status ?? "–"}</Status>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {result.auto_enabled && (
        <p className="text-sm text-muted-foreground">
          {manualOnly.length === 0
            ? "Every station may be marked observed automatically."
            : `Manual only: ${manualOnly.map(stationLabel).join(", ")}.`}
        </p>
      )}
      {result.report && (
        <details>
          <summary className="cursor-pointer text-sm font-medium">gates_report.md</summary>
          <pre className="mt-2 max-h-96 overflow-auto rounded-lg bg-muted p-3 text-xs leading-relaxed">{result.report}</pre>
        </details>
      )}
    </div>
  );
}

// -- procedure replay -----------------------------------------------------------

function withInterval(value?: number | null, lo?: number | null, hi?: number | null): string {
  if (value == null) return "–";
  return lo != null && hi != null ? `${fmtScore(value)} (${fmtScore(lo)} to ${fmtScore(hi)})` : fmtScore(value);
}

function ReplayView({ result }: { result: ReplayResult }) {
  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className={th}>Scenario</th>
              <th className={`${th} ${num}`}>Patients</th>
              <th className={`${th} ${num}`}>Mean station recall</th>
              <th className={`${th} ${num}`}>Time to observed</th>
              <th className={`${th} ${num}`}>False-detection rate</th>
              <th className={`${th} ${num}`}>Flicker / min</th>
              <th className={`${th} ${num}`}>Best-frame purity</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(result.scenarios).map(([name, s]) => (
              <tr key={name} className="border-b border-border last:border-0">
                <td className={`${td} font-mono text-xs`}>{name}</td>
                <td className={`${td} ${num}`}>{s.n_patients ?? "–"}</td>
                <td className={`${td} ${num}`}>{withInterval(s.recall_mean, s.recall_mean_lo, s.recall_mean_hi)}</td>
                <td className={`${td} ${num}`}>{s.tto_median == null ? "–" : `${s.tto_median.toFixed(1)} s`}</td>
                <td className={`${td} ${num}`}>{withInterval(s.fdr_overall, s.fdr_overall_lo, s.fdr_overall_hi)}</td>
                <td className={`${td} ${num}`}>{s.flicker_per_min == null ? "–" : s.flicker_per_min.toFixed(2)}</td>
                <td className={`${td} ${num}`}>{fmtScore(s.best_frame_purity)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted-foreground">
        S0: clips abut. S2: gaps filled with synthetic transit frames. S4: one station&apos;s clips removed, so
        observing it is a false detection. S5: S2 with the clip order shuffled. Intervals are 95% patient
        bootstraps.
        {result.transit === "placeholder" &&
          " Transit frames were placeholders that never qualify, which is optimistic; run with model transit for the real figure."}
      </p>
    </div>
  );
}

// -- dispatch -------------------------------------------------------------------

export function EvalResult({ job }: { job: EvalJob }) {
  if (job.result == null) return null;
  switch (job.kind) {
    case "selftest":
      return <SelftestView result={job.result as SelftestResult} />;
    case "latency":
      return <LatencyView result={job.result as LatencyResult} />;
    case "accuracy":
    case "oof_metrics":
      return <AccuracyView result={job.result as AccuracyResult} />;
    case "gates_report":
      return <GatesView result={job.result as GatesResult} />;
    case "replay":
      return <ReplayView result={job.result as ReplayResult} />;
  }
}
