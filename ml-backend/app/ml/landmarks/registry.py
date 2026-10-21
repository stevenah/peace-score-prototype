"""What the landmark backend could serve, and what the admin page chose.

Bundles. A bundle is a directory holding ``meta.json`` and ``model.pt`` (see
``bundle.py``). ``discover`` lists every one under the search roots: the
folder ``BUNDLE.lock`` lives in, the training runs
(``runs/<exp>/<run>/bundle/``), the S3 bundle cache,
``PEACE_LANDMARK_BUNDLE_DIR`` and any extra root. Listing validates
``meta.json`` only; file hashes and the selftest run when a bundle is loaded.

Selection. The admin page's choice lives in ``ACTIVE.json``
(``settings.landmark_state_file``)::

    {"version": "lm-1.0.0", "path": "/abs/bundle/dir", "mode": "shadow"}

``version`` and ``path`` name the bundle to serve instead of the configured
default (``PEACE_LANDMARK_BUNDLE_DIR``, else ``BUNDLE.lock``). ``mode`` is
``off`` | ``shadow`` | ``display`` and overrides ``landmarks_enabled`` and
``landmarks_display``. Either part may be absent. With
``settings.model_admin`` off the file is ignored altogether.

Experiments. ``discover_experiments`` summarises the training runs that have
finished fold runs, for the script evals. It reads only the JSON a run wrote.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.config import settings
from app.ml.landmarks.bundle import (
    META_FILE,
    MODEL_FILE,
    VERSION_RE,
    LandmarkUnavailable,
    check_meta,
    read_meta,
)

logger = logging.getLogger(__name__)

ML_BACKEND = Path(__file__).resolve().parents[3]

# mode -> (landmarks_enabled, landmarks_display)
MODES: dict[str, tuple[bool, bool]] = {
    "off": (False, False),
    "shadow": (True, False),
    "display": (True, True),
}

# runs/<exp>/<run>/bundle/<version>/ is four levels down; leave some slack.
_MAX_DEPTH = 6
_SKIP_DIRS = frozenset({"__pycache__", "frames", "ui_evals"})
_FOLD_RUN = re.compile(r"^fold(\d)$")
_VERSION_PARTS = re.compile(r"^lm-(\d+)\.(\d+)\.(\d+)(.*)$")


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        return None
    return float(v)


def display_path(path: Path) -> str:
    """Path relative to ml-backend/ when it is inside it, else absolute."""
    try:
        return str(Path(path).resolve().relative_to(ML_BACKEND))
    except ValueError:
        return str(path)


# -- bundles -----------------------------------------------------------------


def training_split(meta: Mapping[str, Any], bundle_dir: Path | None = None) -> dict:
    """The training run a bundle was exported from: ``{"mode", "fold"}``.

    ``mode`` is train.py's (fold | all_dev | smoke | holdout); ``fold`` is the
    outer fold a fold or smoke run never trained on. Taken from ``meta.train``
    (export_bundle records both), else from the run-directory name recorded
    there, else from the ``metrics.json`` of the run directory the bundle
    still sits in. What cannot be established is None.
    """
    train = meta.get("train")
    train = train if isinstance(train, dict) else {}
    mode, fold, run = train.get("mode"), train.get("fold"), train.get("run")
    if mode is None and isinstance(run, str):
        m = _FOLD_RUN.match(run)
        if m:
            mode, fold = "fold", int(m.group(1))
        elif run in ("all_dev", "smoke", "holdout"):
            mode = run
    if bundle_dir is not None and (mode is None or (mode in ("fold", "smoke") and fold is None)):
        met = _read_json(Path(bundle_dir).parent.parent / "metrics.json")  # <run>/bundle/<version>/
        if met:
            mode = mode or met.get("mode")
            fold = met.get("fold") if fold is None else fold
    if mode == "smoke" and fold is None:
        fold = 0  # train.py holds out fold 0 when a smoke run names none
    return {
        "mode": mode if isinstance(mode, str) else None,
        "fold": fold if isinstance(fold, int) and not isinstance(fold, bool) else None,
    }


@dataclass(frozen=True)
class BundleInfo:
    version: str
    path: Path
    source: str  # which search root it was found under
    meta: dict = field(default_factory=dict)
    problem: str | None = None  # why it cannot be served; None = servable

    @property
    def servable(self) -> bool:
        return self.problem is None

    def summary(self) -> dict:
        """JSON-ready description for the admin page."""
        meta = self.meta
        train = meta.get("train") if isinstance(meta.get("train"), dict) else {}
        metrics = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
        quality = meta.get("quality") if isinstance(meta.get("quality"), dict) else {}
        weights = self.path / MODEL_FILE
        try:
            stat = weights.stat()
            size, modified = stat.st_size, datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        except OSError:
            size, modified = None, None
        calibration = train.get("calibration_source")
        return {
            "version": self.version,
            "path": display_path(self.path),
            "source": self.source,
            "servable": self.servable,
            "problem": self.problem,
            "arch": meta.get("arch"),
            "input_size": meta.get("input_size"),
            "layouts_version": meta.get("layouts_version"),
            "preprocess_version": meta.get("preprocess_version"),
            "serve_layouts": meta.get("serve_layouts"),
            "temperature": _number(meta.get("temperature")),
            "tau": meta.get("tau"),
            "q_min": _number(quality.get("q_min")),
            "auto_enabled": meta.get("auto_enabled"),
            "calibrated": not (isinstance(calibration, str) and calibration.startswith("none")),
            "warning": metrics.get("WARNING"),
            "size_bytes": size,
            "modified_at": modified,
            "training": {
                **training_split(meta, self.path),
                "candidate": train.get("candidate"),
                "variant": train.get("variant"),
                "run": train.get("run"),
                "git_sha": train.get("git_sha"),
                "git_dirty": train.get("git_dirty"),
                "device": train.get("device"),
            },
            "metrics": {k: v for k, v in metrics.items() if k != "WARNING"},
        }


def describe(bundle_dir: str | os.PathLike, source: str = "") -> BundleInfo:
    path = Path(bundle_dir)
    prefix = f"landmark bundle {path}: "
    try:
        meta = read_meta(path)
    except LandmarkUnavailable as exc:
        return BundleInfo(path.name, path, source, {}, str(exc).removeprefix(prefix))
    version = meta.get("version")
    version = version if isinstance(version, str) and VERSION_RE.match(version) else path.name
    try:
        check_meta(path)
        problem = None
    except LandmarkUnavailable as exc:
        problem = str(exc).removeprefix(prefix)
    return BundleInfo(version, path, source, meta, problem)


def _cache_root() -> Path:
    """Where ``bundle.resolve_cache_dir`` puts fetched bundles, without creating it."""
    cache = Path(settings.landmark_cache_dir)
    anchor = cache
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    if anchor.is_dir() and os.access(anchor, os.W_OK):
        return cache
    return Path(tempfile.gettempdir()) / "peace-landmarks"


def search_roots() -> list[tuple[str, Path]]:
    """(label, folder) in priority order: a version found twice keeps the first."""
    roots: list[tuple[str, Path]] = []
    if settings.landmark_bundle_dir:
        roots.append(("env", Path(settings.landmark_bundle_dir)))
    roots.append(("models", Path(settings.landmark_bundle_lock).parent))
    roots.append(("runs", Path(settings.runs_dir)))
    roots.append(("cache", _cache_root()))
    roots += [("extra", Path(p)) for p in settings.landmark_bundle_roots.split(os.pathsep) if p]
    return roots


def _bundle_dirs(root: Path) -> list[Path]:
    """Every directory under ``root`` (itself included) holding meta.json and model.pt."""
    if not root.is_dir():
        return []
    found = []
    base = len(root.parts)
    for current, dirs, files in os.walk(root):
        here = Path(current)
        if META_FILE in files and MODEL_FILE in files:
            found.append(here)
            dirs[:] = []
            continue
        if len(here.parts) - base >= _MAX_DEPTH:
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in _SKIP_DIRS)
    return found


def _version_key(version: str) -> tuple:
    m = _VERSION_PARTS.match(version)
    if not m:
        return (0, 0, 0, 0, False, version)
    major, minor, patch, suffix = m.groups()
    # A release sorts above its own pre-releases (lm-1.0.0 > lm-1.0.0-rc1).
    return (1, int(major), int(minor), int(patch), suffix == "", suffix)


def discover() -> list[BundleInfo]:
    """Every bundle under the search roots, newest version first."""
    seen: dict[str, BundleInfo] = {}
    for label, root in search_roots():
        for path in _bundle_dirs(root):
            info = describe(path, label)
            seen.setdefault(info.version, info)
    return sorted(seen.values(), key=lambda b: _version_key(b.version), reverse=True)


def find(version: str) -> BundleInfo | None:
    return next((b for b in discover() if b.version == version), None)


# -- the saved selection -----------------------------------------------------


@dataclass(frozen=True)
class ServingState:
    version: str | None = None
    path: str | None = None
    mode: str | None = None  # None = follow the env settings


def read_state() -> ServingState:
    """The saved selection; empty when there is none, it is unreadable or admin is off."""
    if not settings.model_admin:
        return ServingState()
    path = Path(settings.landmark_state_file)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return ServingState()
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable landmark selection %s: %s", path, exc)
        return ServingState()
    if not isinstance(data, dict):
        logger.warning("Ignoring malformed landmark selection %s", path)
        return ServingState()
    version, bundle, mode = data.get("version"), data.get("path"), data.get("mode")
    if not (isinstance(version, str) and isinstance(bundle, str) and bundle):
        version = bundle = None
    return ServingState(version, bundle, mode if mode in MODES else None)


def write_state(state: ServingState) -> None:
    """Persist atomically; an empty state removes the file."""
    path = Path(settings.landmark_state_file)
    data = {k: v for k, v in asdict(state).items() if v is not None}
    if not data:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def selected_bundle_dir() -> Path | None:
    """The bundle the admin page chose, if any (not checked to exist)."""
    chosen = read_state().path
    return Path(chosen) if chosen else None


def current_mode() -> str:
    if not settings.landmarks_enabled:
        return "off"
    return "display" if settings.landmarks_display else "shadow"


def apply_mode(mode: str) -> None:
    """Live sockets opened from now on follow ``mode``; open ones keep theirs."""
    settings.landmarks_enabled, settings.landmarks_display = MODES[mode]


def apply_saved_mode() -> str | None:
    """Startup: a saved mode overrides the env settings. Returns it, or None."""
    mode = read_state().mode
    if mode:
        apply_mode(mode)
    return mode


# -- training runs -----------------------------------------------------------


def _experiment(exp_dir: Path) -> dict | None:
    folds = []
    for run in sorted(exp_dir.glob("fold[0-9]")):
        metrics = _read_json(run / "metrics.json")
        if metrics is None:
            continue
        k = int(run.name[4:])
        best = metrics.get("best") if isinstance(metrics.get("best"), dict) else {}
        oof = metrics.get("oof") if isinstance(metrics.get("oof"), dict) else {}
        folds.append({
            "fold": k,
            "clip_macro_f1": _number(oof.get("clip_macro_f1")),
            "variant": best.get("variant"),
            "step": best.get("step"),
            "has_oof": (run / f"oof_fold{k}.npz").exists(),
        })
    if not folds:
        return None
    config = _read_json(exp_dir / f"fold{folds[0]['fold']}" / "config.json") or {}
    scores = [f["clip_macro_f1"] for f in folds if f["clip_macro_f1"] is not None]
    report = _read_json(exp_dir / "eval_oof.json") or {}
    clip = report.get("clip") if isinstance(report.get("clip"), dict) else {}
    gates = _read_json(exp_dir / "gates_result.json") or {}
    return {
        "name": exp_dir.name,
        "candidate": config.get("candidate"),
        "input_size": (config.get("input") or {}).get("size") if isinstance(config.get("input"), dict) else None,
        "folds": folds,
        "mean_clip_macro_f1": sum(scores) / len(scores) if scores else None,
        "has_all_dev": (exp_dir / "all_dev" / "metrics.json").exists(),
        "has_calibration": (exp_dir / "calibration.json").exists(),
        "has_tracker": (exp_dir / "tracker.json").exists(),
        "oof_macro_f1": clip.get("macro_f1") if isinstance(clip.get("macro_f1"), dict) else None,
        "outcome": gates.get("outcome"),
        "test_scored": (exp_dir / "TEST_SCORED").exists(),
    }


def discover_experiments() -> list[dict]:
    """Training runs under ``runs_dir`` with at least one finished fold run."""
    root = Path(settings.runs_dir)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        exp = _experiment(d)
        if exp:
            out.append(exp)
    return out


def experiment_dir(name: str) -> Path | None:
    """The directory of a listed experiment; None for anything else."""
    if any(e["name"] == name for e in discover_experiments()):
        return Path(settings.runs_dir) / name
    return None
