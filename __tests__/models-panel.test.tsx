import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import ModelsPanel from "@/app/admin/models/ModelsPanel";
import { EvalResult } from "@/app/admin/models/EvalResult";
import { STATION_ORDER } from "@/lib/live/stations";
import {
  describeParams,
  fmtElapsed,
  fmtInterval,
  fmtMs,
  fmtPercent,
  fmtScore,
  heldOutBlocker,
  jobSeconds,
  progressFraction,
  rowShares,
  trainingSummary,
  type AccuracyResult,
  type EvalJob,
  type ModelInfo,
  type ModelsResponse,
  type Serving,
} from "@/lib/model-admin";

// -- fixtures -----------------------------------------------------------------

function model(overrides: Partial<ModelInfo> & { version: string }): ModelInfo {
  return {
    path: `runs/exp/fold0/bundle/${overrides.version}`,
    source: "runs",
    servable: true,
    problem: null,
    arch: "effb0",
    input_size: 224,
    layouts_version: "lay-3",
    calibrated: true,
    warning: null,
    size_bytes: 16_300_000,
    modified_at: "2026-09-28T19:00:00+00:00",
    training: { mode: "fold", fold: 0, candidate: "effb0_in1k", variant: "raw", run: "fold0" },
    metrics: {},
    ...overrides,
  };
}

const SERVING: Serving = {
  mode: "off",
  mode_saved: false,
  mock: false,
  selected_version: null,
  default: { kind: "none", version: null },
  loaded_version: null,
  device: null,
  failure: null,
};

function payload(overrides: Partial<ModelsResponse> = {}): ModelsResponse {
  return {
    admin_enabled: true,
    serving: SERVING,
    stations: [...STATION_ORDER],
    models: [
      model({ version: "lm-0.0.4-mpsfold0", calibrated: false }),
      model({ version: "lm-0.0.3-mpsfold0", servable: false, problem: "layouts_version 'lay-2' != 'lay-3'" }),
    ],
    experiments: [],
    eval_kinds: [
      { kind: "selftest", label: "Self-test", target: "bundle", summary: "", available: true, reason: null },
      { kind: "latency", label: "Latency", target: "bundle", summary: "", available: true, reason: null },
      { kind: "accuracy", label: "Held-out accuracy", target: "bundle", summary: "", available: true, reason: null },
    ],
    ...overrides,
  };
}

function job(overrides: Partial<EvalJob> = {}): EvalJob {
  return {
    id: "20261011T200501-abc123",
    kind: "selftest",
    label: "Self-test",
    target: { type: "bundle", id: "lm-0.0.4-mpsfold0", path: "runs/x" },
    params: {},
    status: "succeeded",
    created_at: "2026-10-11T20:05:01+00:00",
    started_at: "2026-10-11T20:05:01+00:00",
    finished_at: "2026-10-11T20:05:05+00:00",
    progress: null,
    headline: "All 4 checks passed",
    verdict: "pass",
    error: null,
    ...overrides,
  };
}

/** Routes /api/admin/ml/* to canned answers and records the calls. */
function mockApi(routes: Record<string, (init?: RequestInit) => [number, unknown]>) {
  const calls: { key: string; body: unknown }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const key = `${init?.method ?? "GET"} ${url.replace("/api/admin/ml/", "").split("?")[0]}`;
      calls.push({ key, body: init?.body ? JSON.parse(init.body as string) : undefined });
      const route = routes[key];
      if (!route) throw new Error(`unexpected request ${key}`);
      const [status, data] = route(init);
      return new Response(JSON.stringify(data), { status });
    }),
  );
  return calls;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// -- the panel ------------------------------------------------------------------

describe("ModelsPanel", () => {
  beforeEach(() => {
    // Radix menus measure and scroll their items; jsdom implements neither.
    Element.prototype.scrollIntoView = vi.fn();
    Element.prototype.hasPointerCapture = vi.fn(() => false);
    Element.prototype.releasePointerCapture = vi.fn();
  });

  it("lists the bundles and says why one cannot be served", async () => {
    mockApi({ "GET models": () => [200, payload()], "GET evals": () => [200, { jobs: [] }] });
    render(<ModelsPanel />);

    expect(await screen.findByText("lm-0.0.4-mpsfold0")).toBeInTheDocument();
    expect(screen.getByText("Uncalibrated")).toBeInTheDocument();
    const broken = screen.getByText("lm-0.0.3-mpsfold0").closest("li")!;
    expect(within(broken).getByText("Cannot be served")).toBeInTheDocument();
    expect(within(broken).getByText("layouts_version 'lay-2' != 'lay-3'")).toBeInTheDocument();
    expect(within(broken).getByRole("button", { name: "Make active" })).toBeDisabled();
    expect(screen.getByText("None. Choose one below.")).toBeInTheDocument();
    expect(screen.getByText("No evals have been run yet.")).toBeInTheDocument();
  });

  it("asks before serving an uncalibrated bundle, then switches", async () => {
    const calls = mockApi({
      "GET models": () => [200, payload()],
      "GET evals": () => [200, { jobs: [] }],
      "POST models/active": () => [
        200,
        { ...SERVING, selected_version: "lm-0.0.4-mpsfold0", loaded_version: "lm-0.0.4-mpsfold0", device: "mps" },
      ],
    });
    render(<ModelsPanel />);
    const row = (await screen.findByText("lm-0.0.4-mpsfold0")).closest("li")!;

    fireEvent.click(within(row).getByRole("button", { name: "Make active" }));
    expect(calls.some((c) => c.key === "POST models/active")).toBe(false);
    const dialog = await screen.findByRole("alertdialog");
    expect(within(dialog).getByText("Serve an uncalibrated model?")).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole("button", { name: "Make active" }));

    await waitFor(() => expect(within(row).getByText("Active")).toBeInTheDocument());
    expect(calls.find((c) => c.key === "POST models/active")?.body).toEqual({ version: "lm-0.0.4-mpsfold0" });
    expect(within(row).queryByRole("button", { name: "Make active" })).not.toBeInTheDocument();
    expect(screen.getByText(/loaded on mps/)).toBeInTheDocument();
    expect(screen.getByText("Chosen on this page.")).toBeInTheDocument();
  });

  it("switches the serving mode", async () => {
    const calls = mockApi({
      "GET models": () => [200, payload()],
      "GET evals": () => [200, { jobs: [] }],
      "PUT models/mode": () => [200, { ...SERVING, mode: "shadow", mode_saved: true }],
    });
    render(<ModelsPanel />);
    await screen.findByText("lm-0.0.4-mpsfold0");
    expect(screen.getByText(/Stations are not computed\./)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Shadow" }));
    expect(await screen.findByText(/The workstation shows nothing\./)).toBeInTheDocument();
    expect(calls.find((c) => c.key === "PUT models/mode")?.body).toEqual({ mode: "shadow" });
  });

  it("shows a refusal from the backend", async () => {
    mockApi({
      "GET models": () => [200, payload({ models: [model({ version: "lm-1.0.0" })] })],
      "GET evals": () => [200, { jobs: [] }],
      "POST models/active": () => [409, { error: "landmark bundle: sha256 mismatch for model.pt" }],
    });
    render(<ModelsPanel />);
    fireEvent.click(await screen.findByRole("button", { name: "Make active" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("sha256 mismatch for model.pt");
  });

  it("starts an eval from the menu and opens it", async () => {
    const running = job({ status: "running", headline: null, verdict: null, finished_at: null });
    let jobs: EvalJob[] = [];
    const calls = mockApi({
      "GET models": () => [200, payload()],
      "GET evals": () => [200, { jobs }],
      "POST evals": () => {
        jobs = [running];
        return [201, running];
      },
      [`GET evals/${running.id}`]: () => [200, { ...running, result: null, log: "$ python -m app.ml.landmarks.selfcheck" }],
    });
    render(<ModelsPanel />);
    const row = (await screen.findByText("lm-0.0.4-mpsfold0")).closest("li")!;

    fireEvent.pointerDown(within(row).getByRole("button", { name: "Evaluate" }), { button: 0, ctrlKey: false });
    fireEvent.click(await screen.findByRole("menuitem", { name: /Self-test/ }));

    expect(await screen.findByText("Running")).toBeInTheDocument();
    expect(calls.find((c) => c.key === "POST evals")?.body).toEqual({
      kind: "selftest",
      target: "lm-0.0.4-mpsfold0",
      params: {},
    });
    expect(await screen.findByText("$ python -m app.ml.landmarks.selfcheck")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled();
  });

  it("explains in the menu why an eval cannot run", async () => {
    const allDev = model({
      version: "lm-1.0.0",
      training: { mode: "all_dev", fold: null, candidate: "effb0_in1k", variant: "ema", run: "all_dev" },
    });
    mockApi({
      "GET models": () => [
        200,
        payload({
          models: [allDev],
          eval_kinds: [
            { kind: "selftest", label: "Self-test", target: "bundle", summary: "", available: true, reason: null },
            { kind: "latency", label: "Latency", target: "bundle", summary: "", available: true, reason: null },
            { kind: "accuracy", label: "Held-out accuracy", target: "bundle", summary: "", available: true, reason: null },
          ],
        }),
      ],
      "GET evals": () => [200, { jobs: [] }],
    });
    render(<ModelsPanel />);
    await screen.findByText("lm-1.0.0");
    fireEvent.pointerDown(screen.getByRole("button", { name: "Evaluate" }), { button: 0, ctrlKey: false });

    const heldOut = await screen.findByRole("menuitem", { name: /Held-out accuracy/ });
    expect(heldOut).toHaveAttribute("aria-disabled", "true");
    expect(heldOut).toHaveTextContent("Trained on every dev patient: no fold is held out.");
    expect(screen.getByRole("menuitem", { name: /Accuracy on the shift set/ })).not.toHaveAttribute("aria-disabled");
  });

  it("is read-only when model management is off", async () => {
    mockApi({ "GET models": () => [200, payload({ admin_enabled: false })], "GET evals": () => [200, { jobs: [] }] });
    render(<ModelsPanel />);
    expect(await screen.findByText(/Model management is switched off/)).toBeInTheDocument();
    for (const button of screen.getAllByRole("button", { name: /Make active|Evaluate/ })) {
      expect(button).toBeDisabled();
    }
  });

  it("says so when the backend cannot be reached", async () => {
    mockApi({ "GET models": () => [502, { error: "The ML backend is not reachable." }], "GET evals": () => [200, { jobs: [] }] });
    render(<ModelsPanel />);
    expect(await screen.findByText("The model list could not be loaded")).toBeInTheDocument();
    expect(screen.getByText("The ML backend is not reachable.")).toBeInTheDocument();
  });
});

// -- result views ---------------------------------------------------------------

function accuracy(): AccuracyResult {
  const confusion: number[][] = STATION_ORDER.map((_, r) => STATION_ORDER.map((__, c) => (r === c ? 8 : 0)));
  confusion[5][8] = 2; // two antrum clips taken for the incisura
  const per_class = Object.fromEntries(
    STATION_ORDER.map((key) => [key, { n: 8, precision: 0.9, recall: 0.8, f1: 0.85 }]),
  );
  return {
    stations: [...STATION_ORDER],
    clip: {
      n_clips: 82,
      n_patients: 15,
      coverage: 1,
      macro_f1: { value: 0.7053, lo: 0.6218, hi: 0.7819 },
      bal_acc: { value: 0.7154, lo: 0.6518, hi: 0.786 },
      top2_acc: 0.8855,
      per_class,
      confusion,
    },
    frame: {
      centre_quality: { n: 7265, macro_f1: 0.598, accuracy: 0.629 },
      all_frames: { n: 13950, macro_f1: 0.579, accuracy: 0.605 },
      calibration: { n: 9737, ece: 0.081, brier: 0.508, nll: 1.119 },
    },
    patient_level: { macro_f1: 0.7299 },
    split: { name: "heldout", label: "dev fold 0 (held out)", n_clips: 82, n_patients: 15, n_frames: 13965, skipped_clips: 0 },
    device: "mps",
    calibration: { temperature: 1, q_min: 0.2, quality_gate: true },
    acceptance: { n_frames: 13950, share: { ok: 0.457, uncertain: 0.421, low_quality: 0.121 }, precision_ok: 0.804 },
    warnings: ["Uncalibrated export: temperature, tau and q_min are defaults, not fitted values."],
    seconds: 58.3,
  };
}

describe("EvalResult", () => {
  it("renders held-out accuracy with its interval, table and matrix", () => {
    render(<EvalResult job={job({ kind: "accuracy", result: accuracy() })} />);

    expect(screen.getByText("0.705")).toBeInTheDocument();
    expect(screen.getByText("95% CI 0.622 to 0.782")).toBeInTheDocument();
    expect(screen.getByText(/82 clips from 15 patients, dev fold 0 \(held out\)/)).toBeInTheDocument();
    expect(screen.getByText(/Uncalibrated export/)).toBeInTheDocument();
    expect(screen.getByText("Gastric corpus (greater curvature)")).toBeInTheDocument();
    expect(screen.getByText(/46% of frames pass as ok/)).toBeInTheDocument();

    const matrix = screen.getByText("Confusion matrix").closest("figure")!;
    const cell = within(matrix).getAllByText("2").find((el) => el.tagName === "TD")!;
    fireEvent.mouseEnter(cell);
    expect(within(matrix).getByText("2 of 10")).toBeInTheDocument();
    expect(matrix).toHaveTextContent("Antrum clips (20%) were taken for Incisura angularis.");
  });

  it("renders self-test checks as words, not colour alone", () => {
    const result = {
      ok: false,
      version: "lm-0.0.3",
      seconds: 0.4,
      checks: [
        { name: "meta", label: "Metadata", status: "passed", detail: "effb0 at 224 px" },
        { name: "files", label: "File hashes", status: "failed", detail: "sha256 mismatch for model.pt" },
        { name: "selftest", label: "Reference logits", status: "skipped", detail: "" },
      ],
    };
    render(<EvalResult job={job({ result })} />);
    expect(screen.getByText("Metadata")).toHaveTextContent("Metadata passed");
    expect(screen.getByText("File hashes")).toHaveTextContent("File hashes failed");
    expect(screen.getByText("Reference logits")).toHaveTextContent("Reference logits not run");
    expect(screen.getByText("sha256 mismatch for model.pt")).toBeInTheDocument();
  });

  it("puts latency against the interval between live frames", () => {
    const row = (config: string, p50: number, p95: number, lm: number | null) => ({
      config,
      streams: 1,
      frames: 60,
      frame_ms: { p50, p95, p99: p95 + 5 },
      landmark_ms: { p50: lm, p95: lm, p99: lm },
      drift: 1.02,
      rss_peak_mb: 812.4,
    });
    const result = {
      host: { platform: "macOS", machine: "arm64", torch: "2.10.0", torch_threads: 2 },
      mode: "screen",
      streams: 1,
      rate: null,
      peace: "real",
      baseline: row("peace_only", 31.2, 40, null),
      model: row("lm-1.0.0", 58.6, 75, 26.1),
    };
    const { rerender } = render(<EvalResult job={job({ kind: "latency", result })} />);
    expect(screen.getByText("15%")).toBeInTheDocument(); // 75 of 500 ms
    expect(screen.getByText(/Fits the 500 ms between live frames/)).toBeInTheDocument();
    expect(screen.getByText("PEACE and lm-1.0.0")).toBeInTheDocument();

    rerender(<EvalResult job={job({ kind: "latency", result: { ...result, model: row("lm-1.0.0", 480, 640, 440) } })} />);
    expect(screen.getByText("128%")).toBeInTheDocument();
    expect(screen.getByText(/Does not fit the 500 ms between live frames/)).toBeInTheDocument();
  });

  it("renders a gates outcome with the manual-only stations", () => {
    const result = {
      outcome: "PARTIAL",
      gates: { "G-MODEL": { status: "PASS" }, "G-STREAM": { status: "FAIL" }, "G-TEST": { status: "NOT_RUN" } },
      auto_enabled: STATION_ORDER.map((_, i) => i !== 1),
      stations: [...STATION_ORDER],
      clip_macro_f1: { value: 0.81, lo: 0.76, hi: 0.85 },
      report: "# Gates report: exp\n",
    };
    render(<EvalResult job={job({ kind: "gates_report", result })} />);
    expect(screen.getByText("Outcome PARTIAL")).toBeInTheDocument();
    expect(screen.getByText("Manual only: Distal esophagus.")).toBeInTheDocument();
    expect(screen.getByText("NOT_RUN")).toBeInTheDocument();
  });

  it("flags placeholder transit as optimistic in a replay", () => {
    const result = {
      transit: "placeholder",
      stations: [...STATION_ORDER],
      scenarios: {
        S2: { n_patients: 72, recall_mean: 0.912, recall_mean_lo: 0.88, recall_mean_hi: 0.94, tto_median: 3.5 },
        S4: { n_patients: 72, fdr_overall: 0.041, fdr_overall_lo: 0.02, fdr_overall_hi: 0.07 },
      },
    };
    render(<EvalResult job={job({ kind: "replay", result })} />);
    expect(screen.getByText("0.912 (0.880 to 0.940)")).toBeInTheDocument();
    expect(screen.getByText("0.041 (0.020 to 0.070)")).toBeInTheDocument();
    expect(screen.getByText(/which is optimistic/)).toBeInTheDocument();
  });

  it("renders nothing for a job without a result", () => {
    const { container } = render(<EvalResult job={job({ status: "failed", result: null })} />);
    expect(container).toBeEmptyDOMElement();
  });
});

// -- helpers ----------------------------------------------------------------------

describe("model-admin helpers", () => {
  it("formats metrics the way the training reports do", () => {
    expect(fmtScore(0.72814)).toBe("0.728");
    expect(fmtScore(null)).toBe("–");
    expect(fmtScore(Number.NaN)).toBe("–");
    expect(fmtInterval({ value: 0.7, lo: 0.62, hi: 0.78 })).toBe("0.620 to 0.780");
    expect(fmtInterval({ value: 0.7, lo: null, hi: null })).toBe("");
    expect(fmtPercent(0.4573)).toBe("46%");
    expect(fmtPercent(0.4573, 1)).toBe("45.7%");
    expect(fmtMs(4.26)).toBe("4.3 ms");
    expect(fmtMs(58.6)).toBe("59 ms");
    expect(fmtMs(null)).toBe("–");
  });

  it("formats run times", () => {
    expect(fmtElapsed(4.4)).toBe("4 s");
    expect(fmtElapsed(185)).toBe("3 min 5 s");
    expect(fmtElapsed(4330)).toBe("1 h 12 min");
    expect(jobSeconds(job())).toBe(4);
    expect(jobSeconds(job({ started_at: null }))).toBeNull();
    const started = "2026-10-11T20:05:01+00:00";
    expect(jobSeconds(job({ started_at: started, finished_at: null }), Date.parse(started) + 30_000)).toBe(30);
  });

  it("reads progress as a fraction", () => {
    expect(progressFraction(job({ progress: { done: 50, total: 200, label: "frames" } }))).toBe(0.25);
    expect(progressFraction(job({ progress: { done: 5, total: 0, label: "" } }))).toBeNull();
    expect(progressFraction(job())).toBeNull();
  });

  it("normalises each confusion row by its own total", () => {
    expect(rowShares([[3, 1], [0, 0]])).toEqual([[0.75, 0.25], [0, 0]]);
  });

  it("describes where a bundle came from and whether a fold is held out", () => {
    const fold = model({ version: "a" });
    expect(trainingSummary(fold)).toBe("effb0_in1k, fold 0 held out");
    expect(heldOutBlocker(fold)).toBeNull();

    const allDev = model({ version: "b", training: { ...fold.training, mode: "all_dev", fold: null } });
    expect(trainingSummary(allDev)).toBe("effb0_in1k, all dev patients");
    expect(heldOutBlocker(allDev)).toMatch(/no fold is held out/);

    const unknown = model({ version: "c", training: { mode: null, fold: null, candidate: null, variant: null, run: "handmade" } });
    expect(trainingSummary(unknown)).toBe("handmade");
    expect(heldOutBlocker(unknown)).toMatch(/does not say which fold/);
  });

  it("summarises a job's parameters", () => {
    expect(describeParams(job({ kind: "latency", params: { mode: "screen", frames: 60 } }))).toBe("60 frames back to back");
    expect(describeParams(job({ kind: "latency", params: { mode: "sustained", minutes: 2, streams: 2 } }))).toBe(
      "2 streams at 2 frames/s for 2 min",
    );
    expect(describeParams(job({ kind: "accuracy", params: { split: "shift", device: "auto" } }))).toBe("legacy shift set");
    expect(describeParams(job({ kind: "replay", params: { transit: "model" } }))).toBe("model transit frames");
    expect(describeParams(job())).toBe("");
  });
});
