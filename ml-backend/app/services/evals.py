"""Evals started from the admin page, run one at a time as subprocesses.

Every eval is ``python -m <module> ...`` launched from ml-backend/. A separate
process keeps torch's thread count, device and memory out of the serving
process, and makes cancelling a matter of terminating it.

    kind          target      module
    selftest      bundle      app.ml.landmarks.selfcheck
    latency       bundle      app.ml.landmarks.bench --bundle
    accuracy      bundle      training.landmarks.bundle_eval
    oof_metrics   experiment  training.landmarks.evaluate --exp
    gates_report  experiment  training.landmarks.evaluate --exp --report
    replay        experiment  training.landmarks.stream_eval --exp

The last four need the ``training`` package and its extras, and the private
frame cache or a run's saved predictions, so they exist only where the data
was prepared; ``unavailable`` says why elsewhere. No kind can score the locked
test set: nothing here builds ``evaluate.py --test``.

A job is a directory under ``settings.eval_dir``:

    <id>/job.json     status, target, params, progress, headline
    <id>/log.txt      the subprocess's stdout and stderr
    <id>/out.json     what the module wrote (bundle kinds)
    <id>/result.json  what the admin page shows

The files are the only state. A restart loses nothing but the job that was
running, which ``start`` marks as failed. Results hold aggregate numbers only.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.config import settings
from app.ml.landmarks import progress, registry
from app.ml.landmarks.stations import STATION_ORDER

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")
KEEP_JOBS = 200
_LOG_TAIL_BYTES = 64 * 1024
_LOG_TAIL_LINES = 200
_ID_RE = re.compile(r"^\d{8}T\d{6}-[0-9a-f]{6}$")


class EvalError(Exception):
    """A request the runner refuses; ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


# -- parameters ---------------------------------------------------------------


def _only(params: dict, *known: str) -> None:
    unknown = sorted(set(params) - set(known))
    if unknown:
        raise ValueError(f"unknown parameter{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}")


def _choice(params: dict, key: str, allowed: tuple[str, ...]) -> str:
    value = params.get(key, allowed[0])
    if value not in allowed:
        raise ValueError(f"{key} must be one of {', '.join(allowed)}")
    return value


def _number(params: dict, key: str, default: float, lo: float, hi: float, integer: bool = False):
    value = params.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
        raise ValueError(f"{key} must be a number from {lo:g} to {hi:g}")
    if integer and value != int(value):
        raise ValueError(f"{key} must be a whole number")
    return int(value) if integer else float(value)


def _no_params(params: dict) -> dict:
    _only(params)
    return {}


def _latency_params(params: dict) -> dict:
    _only(params, "mode", "frames", "minutes", "streams")
    mode = _choice(params, "mode", ("screen", "sustained"))
    if mode == "screen":
        _only(params, "mode", "frames")
        return {"mode": mode, "frames": _number(params, "frames", 60, 10, 1000, integer=True)}
    _only(params, "mode", "minutes", "streams")
    return {"mode": mode, "minutes": _number(params, "minutes", 2, 0.5, 30),
            "streams": _number(params, "streams", 2, 1, 4, integer=True)}


def _accuracy_params(params: dict) -> dict:
    _only(params, "split", "device")
    return {"split": _choice(params, "split", ("heldout", "shift")),
            "device": _choice(params, "device", ("auto", "cpu"))}


def _replay_params(params: dict) -> dict:
    _only(params, "transit")
    return {"transit": _choice(params, "transit", ("placeholder", "model"))}


# -- commands -----------------------------------------------------------------


def _selftest_args(target: Path, params: dict, out: Path) -> list[str]:
    return ["--bundle", str(target), "--out", str(out)]


def _latency_args(target: Path, params: dict, out: Path) -> list[str]:
    args = ["--bundle", str(target), "--mode", params["mode"], "--progress", "--out", str(out),
            "--peace", "mock" if settings.use_mock_models else "real"]
    if params["mode"] == "screen":
        return args + ["--frames", str(params["frames"])]
    return args + ["--minutes", str(params["minutes"]), "--streams", str(params["streams"]), "--rate", "2"]


def _accuracy_args(target: Path, params: dict, out: Path) -> list[str]:
    return ["--bundle", str(target), "--split", params["split"], "--device", params["device"], "--out", str(out)]


def _oof_args(target: Path, params: dict, out: Path) -> list[str]:
    return ["--exp", str(target)]


def _gates_args(target: Path, params: dict, out: Path) -> list[str]:
    return ["--exp", str(target), "--report"]


def _replay_args(target: Path, params: dict, out: Path) -> list[str]:
    return ["--exp", str(target), "--transit", params["transit"]]


# -- results ------------------------------------------------------------------


def _required(path: Path) -> dict:
    data = _read_json(path)
    if data is None:
        raise FileNotFoundError(f"{path.name} was not written")
    return data


def _out_result(target: Path, params: dict, out: Path) -> dict:
    return _required(out)


def _latency_result(target: Path, params: dict, out: Path) -> dict:
    report = _required(out)
    baseline, model = report["results"][0], report["results"][1]
    return {
        "host": report["host"],
        "mode": report["mode"],
        "streams": model["streams"],
        "rate": report["params"]["rate"] if report["mode"] == "sustained" else None,
        "peace": report["params"]["peace"],
        "baseline": baseline,
        "model": model,
    }


def _oof_result(target: Path, params: dict, out: Path) -> dict:
    ev = _required(target / "eval_oof.json")
    return {
        "exp": ev.get("exp"),
        "folds": ev.get("folds"),
        "stations": list(STATION_ORDER),
        "clip": ev["clip"],
        "clip_merged": {k: ev["clip_merged"].get(k) for k in ("macro_f1", "bal_acc", "n_clips")},
        "patient_level": ev.get("patient_level"),
        "frame": ev.get("frame"),
        "per_fold_macro_f1": ev.get("per_fold_macro_f1"),
        "confusable": ev.get("confusable"),
    }


def _gates_result(target: Path, params: dict, out: Path) -> dict:
    gates = _required(target / "gates_result.json")
    ev = _read_json(target / "eval_oof.json") or {}
    try:
        report = (target / "gates_report.md").read_text()
    except OSError:
        report = None
    return {
        "outcome": gates.get("outcome"),
        "gates": {name: {"status": g.get("status")} for name, g in (gates.get("gates") or {}).items()
                  if isinstance(g, dict)},
        "auto_enabled": gates.get("auto_enabled"),
        "stations": list(STATION_ORDER),
        "clip_macro_f1": (ev.get("clip") or {}).get("macro_f1"),
        "report": report,
    }


def _replay_result(target: Path, params: dict, out: Path) -> dict:
    scenarios = _required(target / f"stream_eval_{params['transit']}.json")
    return {"transit": params["transit"], "stations": list(STATION_ORDER), "scenarios": scenarios}


# -- headlines ----------------------------------------------------------------


def _selftest_headline(result: dict) -> tuple[str, str | None]:
    checks = result["checks"]
    failed = next((c for c in checks if c["status"] == "failed"), None)
    if failed:
        return f"Failed at {failed['label'].lower()}", "fail"
    return f"All {len(checks)} checks passed", "pass"


def _latency_headline(result: dict) -> tuple[str, str | None]:
    model, base = result["model"]["frame_ms"], result["baseline"]["frame_ms"]
    return (f"{model['p50']:.0f} ms median, {model['p95']:.0f} ms p95 per frame "
            f"(PEACE alone {base['p50']:.0f} ms)"), None


def _macro_f1_text(f1: dict) -> str:
    return f"Clip macro-F1 {f1['value']:.3f} [{f1['lo']:.3f}, {f1['hi']:.3f}]"


def _accuracy_headline(result: dict) -> tuple[str, str | None]:
    split = result["split"]
    return f"{_macro_f1_text(result['clip']['macro_f1'])} on {split['n_clips']} clips, {split['label']}", None


def _oof_headline(result: dict) -> tuple[str, str | None]:
    return f"{_macro_f1_text(result['clip']['macro_f1'])} on {result['clip']['n_clips']} out-of-fold clips", None


def _gates_headline(result: dict) -> tuple[str, str | None]:
    outcome = result.get("outcome")
    return f"Outcome {outcome}", {"GO": "pass", "NO-GO": "fail"}.get(outcome)


def _replay_headline(result: dict) -> tuple[str, str | None]:
    scenarios = result["scenarios"]
    parts = []
    recall = (scenarios.get("S2") or {}).get("recall_mean")
    if recall is not None:
        parts.append(f"S2 recall {recall:.3f}")
    fdr = (scenarios.get("S4") or {}).get("fdr_overall")
    if fdr is not None:
        parts.append(f"S4 false-detection rate {fdr:.3f}")
    return ", ".join(parts) or f"{len(scenarios)} scenarios replayed", None


# -- availability -------------------------------------------------------------


def _training_unavailable() -> str | None:
    try:
        missing = [m for m in ("training.landmarks.evaluate", "pandas") if importlib.util.find_spec(m) is None]
    except ModuleNotFoundError:
        missing = ["training"]
    if missing:
        return ("Needs the training/ folder and its extras (uv pip install -e '.[train]'), "
                "which this backend does not have.")
    return None


def _data_unavailable() -> str | None:
    reason = _training_unavailable()
    if reason:
        return reason
    from training.landmarks.common import CACHE_ROOT, MANIFEST_PATH

    if not MANIFEST_PATH.is_file() or not (CACHE_ROOT / "frames.csv.gz").is_file():
        return ("Needs the private manifest and frame cache (data/landmarks_private/ and "
                "training/landmarks/cache/v1/), which exist only where the data was prepared.")
    return None


# -- per-target checks --------------------------------------------------------


def _loadable(target: Path, params: dict) -> str | None:
    problem = registry.describe(target).problem
    return f"This bundle cannot be loaded: {problem}" if problem else None


def _accuracy_ready(target: Path, params: dict) -> str | None:
    info = registry.describe(target)
    if info.problem:
        return f"This bundle cannot be loaded: {info.problem}"
    if params["split"] == "heldout":
        run = registry.training_split(info.meta, target)
        if run["mode"] == "all_dev":
            return ("This model trained on every dev patient, so no dev fold is held out. "
                    "Score it on the shift set; the locked test set is scored once, with evaluate.py --test.")
        if run["mode"] not in ("fold", "smoke") or run["fold"] is None:
            return "Cannot tell which dev fold this bundle held out: its metadata names no fold run."
    return None


def _calibrated(target: Path, params: dict) -> str | None:
    if not (target / "calibration.json").is_file():
        return (f"{target.name} has no calibration.json. Run "
                f"python -m training.landmarks.calibrate --exp runs/{target.name} first.")
    return None


# -- kinds ----------------------------------------------------------------------


@dataclass(frozen=True)
class Kind:
    name: str
    label: str
    target: str  # "bundle" | "experiment"
    summary: str
    module: str
    clean: Callable[[dict], dict]  # validate request params; ValueError on a bad one
    args: Callable[[Path, dict, Path], list[str]]  # (target dir, params, out file) -> module arguments
    result: Callable[[Path, dict, Path], dict]
    headline: Callable[[dict], tuple[str, str | None]]  # result -> (text, pass | fail | None)
    unavailable: Callable[[], str | None] = lambda: None  # why this backend cannot run it at all
    refuse: Callable[[Path, dict], str | None] = lambda target, params: None  # why not on this target
    timeout_s: Callable[[dict], float] = lambda params: 3 * 3600.0


def _latency_timeout(params: dict) -> float:
    # Sustained mode runs the PEACE baseline and the model for --minutes each.
    return 900.0 if params["mode"] == "screen" else 2 * 60 * params["minutes"] + 600


KINDS: dict[str, Kind] = {k.name: k for k in (
    Kind("selftest", "Self-test", "bundle",
         "File hashes, reference logits and a pass through the live path on synthetic frames.",
         "app.ml.landmarks.selfcheck", _no_params, _selftest_args, _out_result, _selftest_headline,
         timeout_s=lambda params: 600.0),
    Kind("latency", "Latency", "bundle",
         "Per-frame latency through the serving path on synthetic frames, next to PEACE alone. CPU only.",
         "app.ml.landmarks.bench", _latency_params, _latency_args, _latency_result, _latency_headline,
         refuse=_loadable, timeout_s=_latency_timeout),
    Kind("accuracy", "Held-out accuracy", "bundle",
         "Clip and frame metrics on labelled clips the model never trained on. Never the locked test set.",
         "training.landmarks.bundle_eval", _accuracy_params, _accuracy_args, _out_result, _accuracy_headline,
         unavailable=_data_unavailable, refuse=_accuracy_ready),
    Kind("oof_metrics", "Out-of-fold metrics", "experiment",
         "evaluate.py on the run's saved out-of-fold predictions; writes eval_oof.json.",
         "training.landmarks.evaluate", _no_params, _oof_args, _oof_result, _oof_headline,
         unavailable=_data_unavailable, refuse=_calibrated),
    Kind("gates_report", "Gates report", "experiment",
         "evaluate.py --report: the pre-registered gates and the GO / PARTIAL / NO-GO outcome.",
         "training.landmarks.evaluate", _no_params, _gates_args, _gates_result, _gates_headline,
         unavailable=_data_unavailable, refuse=_calibrated),
    Kind("replay", "Procedure replay", "experiment",
         "stream_eval.py: the replay benchmark (S0, S2, S4, S5) through the serving tracker.",
         "training.landmarks.stream_eval", _replay_params, _replay_args, _replay_result, _replay_headline,
         unavailable=_data_unavailable, refuse=_calibrated),
)}


def describe_kinds(kinds: dict[str, Kind] = KINDS) -> list[dict]:
    out = []
    for kind in kinds.values():
        reason = kind.unavailable()
        out.append({"kind": kind.name, "label": kind.label, "target": kind.target, "summary": kind.summary,
                    "available": reason is None, "reason": reason})
    return out


# -- the runner -------------------------------------------------------------------


class EvalRunner:
    """Queues eval jobs and runs them one at a time on a worker thread."""

    def __init__(self, root: str | Path | None = None, kinds: dict[str, Kind] | None = None,
                 python: str | None = None):
        self._root = Path(root) if root else None
        self._kinds = kinds if kinds is not None else KINDS
        self._python = python or sys.executable
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.RLock()  # job.json writes
        self._stop = threading.Event()
        self._cancelled: set[str] = set()
        self._thread: threading.Thread | None = None

    @property
    def root(self) -> Path:
        return self._root or Path(settings.eval_dir)

    @property
    def kinds(self) -> dict[str, Kind]:
        return self._kinds

    # -- lifecycle

    def start(self) -> None:
        """Fail the jobs a previous process left unfinished, then start the worker."""
        for job in self._all_jobs():
            if job.get("status") in ACTIVE:
                self._update(job["id"], status="failed", finished_at=_now(),
                             error="Interrupted: the ML backend restarted before this eval finished.")
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="eval-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None

    # -- requests

    def submit(self, kind_name: str, target_id: str, params: dict | None = None) -> dict:
        kind = self._kinds.get(kind_name)
        if kind is None:
            raise EvalError(404, f"Unknown eval {kind_name!r}")
        reason = kind.unavailable()
        if reason:
            raise EvalError(409, reason)
        try:
            clean = kind.clean(dict(params or {}))
        except ValueError as exc:
            raise EvalError(422, str(exc)) from exc
        target = self._resolve(kind.target, target_id)
        reason = kind.refuse(target, clean)
        if reason:
            raise EvalError(409, reason)

        job_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{secrets.token_hex(3)}"
        job = {
            "id": job_id,
            "kind": kind.name,
            "label": kind.label,
            "target": {"type": kind.target, "id": target_id, "path": registry.display_path(target)},
            "target_path": str(target),
            "params": clean,
            "status": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "progress": None,
            "headline": None,
            "verdict": None,
            "error": None,
        }
        (self.root / job_id).mkdir(parents=True, exist_ok=True)
        with self._lock:
            _write_json(self.root / job_id / "job.json", job)
        self._prune()
        self._queue.put(job_id)
        return self._public(job)

    def list(self, limit: int = 50) -> list[dict]:
        """Newest first."""
        return [self._public(j) for j in self._newest_first(self._all_jobs())[:limit]]

    def get(self, job_id: str) -> dict | None:
        """One job with its result and the tail of its log."""
        job = self._read(job_id)
        if job is None:
            return None
        out = self._public(job)
        out["result"] = _read_json(self.root / job_id / "result.json")
        out["log"] = self._log_tail(job_id)
        return out

    def cancel(self, job_id: str) -> dict | None:
        """Stop a queued or running job; a finished one is returned unchanged."""
        job = self._read(job_id)
        if job is None:
            return None
        if job["status"] == "queued":
            job = self._update(job_id, status="cancelled", finished_at=_now())
        elif job["status"] == "running":
            self._cancelled.add(job_id)  # the worker terminates the process on its next tick
        return self._public(job)

    # -- internals

    def _resolve(self, target_type: str, target_id: str) -> Path:
        if target_type == "bundle":
            info = registry.find(target_id)
            if info is None:
                raise EvalError(404, f"No bundle {target_id!r}")
            return info.path
        path = registry.experiment_dir(target_id)
        if path is None:
            raise EvalError(404, f"No training run {target_id!r}")
        return path

    @staticmethod
    def _public(job: dict) -> dict:
        return {k: v for k, v in job.items() if k != "target_path"}

    def _read(self, job_id: str) -> dict | None:
        if not _ID_RE.match(job_id):
            return None
        return _read_json(self.root / job_id / "job.json")

    def _all_jobs(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        jobs = (self._read(d.name) for d in self.root.iterdir() if d.is_dir())
        return [j for j in jobs if j is not None]

    def _update(self, job_id: str, **fields) -> dict:
        with self._lock:
            job = self._read(job_id) or {"id": job_id}
            job.update(fields)
            _write_json(self.root / job_id / "job.json", job)
            return job

    @staticmethod
    def _newest_first(jobs: list[dict]) -> list[dict]:
        # created_at has microseconds; the id's timestamp only seconds.
        return sorted(jobs, key=lambda j: (j.get("created_at") or "", j["id"]), reverse=True)

    def _prune(self) -> None:
        done = [j for j in self._newest_first(self._all_jobs()) if j.get("status") not in ACTIVE]
        for job in done[KEEP_JOBS:]:
            shutil.rmtree(self.root / job["id"], ignore_errors=True)

    def _log_lines(self, job_id: str) -> list[str]:
        try:
            with open(self.root / job_id / "log.txt", "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - _LOG_TAIL_BYTES))
                return f.read().decode("utf-8", "replace").splitlines()
        except OSError:
            return []

    def _log_tail(self, job_id: str) -> str:
        lines = [line for line in self._log_lines(job_id) if not line.startswith(progress.PREFIX)]
        return "\n".join(lines[-_LOG_TAIL_LINES:])

    def _latest_progress(self, job_id: str) -> dict | None:
        for line in reversed(self._log_lines(job_id)):
            update = progress.parse(line)
            if update is not None:
                return update
        return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run(job_id)
            except Exception as exc:  # a broken job must not end the worker
                logger.exception("Eval %s crashed in the runner", job_id)
                self._update(job_id, status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}")

    def _run(self, job_id: str) -> None:
        job = self._read(job_id)
        if job is None or job["status"] != "queued":  # cancelled while it waited
            return
        kind = self._kinds[job["kind"]]
        target, params = Path(job["target_path"]), job["params"]
        out = self.root / job_id / "out.json"
        args = kind.args(target, params, out)
        self._update(job_id, status="running", started_at=_now())
        logger.info("Eval %s: %s on %s", job_id, kind.name, job["target"]["id"])

        deadline = time.monotonic() + kind.timeout_s(params)
        why = None  # set when the runner ends the process itself
        with open(self.root / job_id / "log.txt", "wb") as log:
            log.write(f"$ python -m {kind.module} {shlex.join(args)}\n".encode())
            log.flush()
            proc = subprocess.Popen(
                [self._python, "-u", "-m", kind.module, *args],
                cwd=registry.ML_BACKEND, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,  # its own process group, so DataLoader workers end with it
            )
            shown = None
            while True:
                try:
                    code = proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                latest = self._latest_progress(job_id)
                if latest != shown:
                    shown = latest
                    self._update(job_id, progress=latest)
                if job_id in self._cancelled:
                    why = "cancelled"
                elif self._stop.is_set():
                    why = "stopped"
                elif time.monotonic() > deadline:
                    why = "timeout"
                if why:
                    code = _terminate(proc)
                    break

        self._cancelled.discard(job_id)
        done = {"finished_at": _now(), "progress": self._latest_progress(job_id)}
        if why == "cancelled":
            self._update(job_id, status="cancelled", **done)
        elif why == "stopped":
            self._update(job_id, status="failed", error="The ML backend stopped while this eval was running.", **done)
        elif why == "timeout":
            minutes = kind.timeout_s(params) / 60
            self._update(job_id, status="failed", error=f"Stopped after {minutes:.0f} minutes without finishing.",
                         **done)
        elif code != 0:
            self._update(job_id, status="failed", error=self._last_line(job_id) or f"Exited with code {code}.",
                         **done)
        else:
            try:
                result = kind.result(target, params, out)
                headline, verdict = kind.headline(result)
            except Exception as exc:
                self._update(job_id, status="failed", **done,
                             error=f"Finished, but its result could not be read: {type(exc).__name__}: {exc}")
                return
            _write_json(self.root / job_id / "result.json", result)
            self._update(job_id, status="succeeded", headline=headline, verdict=verdict, **done)

    def _last_line(self, job_id: str) -> str | None:
        for line in reversed(self._log_lines(job_id)):
            if line.strip() and not line.startswith(progress.PREFIX):
                return line.strip()[:500]
        return None


def _terminate(proc: subprocess.Popen) -> int:
    """SIGTERM the process group, SIGKILL it if it lingers; the exit code."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    return proc.wait()
