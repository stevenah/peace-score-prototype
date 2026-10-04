"""Landmark model bundles: export format, verified loading and S3 fetch.

A bundle is a directory named after its version::

    lm-<semver>/
        model.pt    fp32 state_dict of architectures.build(arch)
        meta.json   everything serving needs besides the weights (below)

``meta.json`` keys:

    schema              "esge10.v1"
    version             "lm-<semver>", equal to the directory/lock version
    stations            STATION_ORDER (the logit order)
    arch, input_size    architectures.build(arch); square input side in px
    mean, std           ImageNet statistics (the only normalisation served)
    preprocess_version  preprocess.PREPROCESS_VERSION the model was trained on
    layouts_version     layouts.LAYOUTS_VERSION the model was trained on
    serve_layouts       layout names this bundle may be served on
    temperature         softmax temperature T (probs = softmax(logits / T))
    tau                 [10] per-station acceptance threshold on calibrated p
    quality             QualityParams fields plus "q_min" (the quality gate)
    tracker             TrackerParams fields
    auto_enabled        [10] stations allowed to become "observed" by themselves
    rejection           {"method": "msp"}
    metrics, train      free-form summaries (aggregate numbers only)
    selftest            {seed, input_size, logits[10], rtol, atol}: CPU fp32
                        logits of the deterministic input selftest_input(seed)
    files               {filename: sha256} for every weight file

Loading refuses (``LandmarkUnavailable``) on any sha256, schema, station,
version or selftest mismatch, and on tracker/quality parameters of the wrong
type or range (unknown keys included). The selftest always runs on CPU in
fp32, which is also where its reference logits were computed at export, so
GPU/TF32 or MPS numerics can never disable a good bundle.

Weights are never committed (the repo is public). They are shipped as a
tarball of the bundle directory in PRIVATE S3, pinned by the committed
``app/models/landmarks/BUNDLE.lock``::

    {"version": "lm-1.0.0",
     "s3_key": "models/landmarks/landmarks-lm-1.0.0.tar.gz",
     "sha256": "<hex sha256 of the tarball>"}

``fetch_bundle`` downloads it once into ``<cache_dir>/<version>/`` (atomically,
via a temporary directory and a rename) and fails hard on any mismatch. It
writes ``.bundle-source.json`` = {s3_key, sha256 (the lock's), meta_sha256}
next to the bundle; a cached directory is reused only if that marker exists,
names the lock's sha256 and still matches meta.json, so the served thresholds
stay bound to the locked tarball.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from app.ml.landmarks.layouts import LAYOUTS_BY_NAME, LAYOUTS_VERSION, OLYMPUS_1920
from app.ml.landmarks.preprocess import IMAGENET_MEAN, IMAGENET_STD, PREPROCESS_VERSION
from app.ml.landmarks.quality import QualityParams
from app.ml.landmarks.stations import NUM_STATIONS, SCHEMA, STATION_ORDER
from app.ml.landmarks.tracker import StationTracker, TrackerParams

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

MODEL_FILE = "model.pt"
META_FILE = "meta.json"
SOURCE_FILE = ".bundle-source.json"  # written by fetch_bundle; not part of the bundle

VERSION_RE = re.compile(r"^lm-\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_Q_MIN = 0.2
DEFAULT_TAU = 0.5
SELFTEST_SEED = 1234
SELFTEST_RTOL = 1e-3
SELFTEST_ATOL = 1e-3


class LandmarkUnavailable(RuntimeError):
    """The landmark model cannot be served (missing, corrupt or mismatched)."""


@dataclass(frozen=True)
class LandmarkBundle:
    path: Path
    version: str
    arch: str
    input_size: int
    temperature: float
    tau: tuple[float, ...]
    quality: QualityParams
    q_min: float
    tracker: TrackerParams
    auto_enabled: tuple[bool, ...]
    serve_layouts: tuple[str, ...]
    meta: dict
    model: "nn.Module"


@dataclass(frozen=True)
class BundleLock:
    version: str
    s3_key: str
    sha256: str


# -- hashing and the selftest input -----------------------------------------


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def selftest_input(seed: int, size: int):
    """Deterministic (1, 3, size, size) float32 input in [-1, 1).

    Pure integer arithmetic, so it is bit-identical on every platform and
    torch version (unlike a seeded RNG).
    """
    import torch

    n = 3 * size * size
    i = torch.arange(n, dtype=torch.int64)
    v = (i * 2654435761 + int(seed) * 40503) % 65521
    return (v.to(torch.float64) / 32760.5 - 1.0).to(torch.float32).reshape(1, 3, size, size)


def _selftest_logits(model: "nn.Module", seed: int, size: int) -> np.ndarray:
    import torch

    with torch.inference_mode():
        logits, _ = model(selftest_input(seed, size))
    return logits[0].detach().cpu().double().numpy()


# -- export -----------------------------------------------------------------


def _fp32_state_dict(model: "nn.Module") -> dict:
    return {
        k: (v.detach().float() if v.is_floating_point() else v.detach()).cpu().clone()
        for k, v in model.state_dict().items()
    }


def default_meta(version: str, arch: str, input_size: int = 224) -> dict:
    """Serving defaults for a bundle; export overrides the calibrated fields."""
    return {
        "schema": SCHEMA,
        "version": version,
        "stations": list(STATION_ORDER),
        "arch": arch,
        "input_size": int(input_size),
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "preprocess_version": PREPROCESS_VERSION,
        "layouts_version": LAYOUTS_VERSION,
        "serve_layouts": [OLYMPUS_1920.name],
        "temperature": 1.0,
        "tau": [DEFAULT_TAU] * NUM_STATIONS,
        "quality": {**QualityParams().to_dict(), "q_min": DEFAULT_Q_MIN},
        "tracker": TrackerParams().to_dict(),
        "auto_enabled": [True] * NUM_STATIONS,
        "rejection": {"method": "msp"},
        "metrics": {},
        "train": {},
    }


# Fields a caller may not set: they are derived from the code or the weights.
_DERIVED = ("schema", "stations", "mean", "std", "preprocess_version",
            "layouts_version", "files")


def write_bundle(bundle_dir: str | os.PathLike, model: "nn.Module", meta_fields: Mapping[str, Any]) -> Path:
    """Write ``model.pt`` + ``meta.json`` into ``bundle_dir`` and return it.

    ``meta_fields`` must contain ``version``; ``arch`` defaults to
    ``model.arch``. The selftest logits are computed by reloading the written
    weights into a fresh CPU fp32 model (the exact serving load path), and are
    checked against the given model's own eager output.
    """
    from app.ml.landmarks.architectures import build

    bad = [k for k in _DERIVED if k in meta_fields]
    if bad:
        raise ValueError(f"meta fields {bad} are derived and cannot be set")
    arch = meta_fields.get("arch", getattr(model, "arch", None))
    if arch is None:
        raise ValueError("meta_fields must name the arch")
    version = meta_fields["version"]
    if not VERSION_RE.match(version):
        raise ValueError(f"bundle version {version!r} is not lm-<semver>")

    meta = default_meta(version, arch, meta_fields.get("input_size", 224))
    meta.update({k: v for k, v in meta_fields.items() if k != "selftest"})
    st = dict(meta_fields.get("selftest", {}))
    seed = int(st.get("seed", SELFTEST_SEED))
    rtol = float(st.get("rtol", SELFTEST_RTOL))
    atol = float(st.get("atol", SELFTEST_ATOL))

    out = Path(bundle_dir)
    out.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(_fp32_state_dict(model), out / MODEL_FILE)

    fresh = build(arch, NUM_STATIONS)
    fresh.load_state_dict(torch.load(out / MODEL_FILE, map_location="cpu", weights_only=True), strict=True)
    fresh.eval()
    logits = _selftest_logits(fresh, seed, meta["input_size"])
    eager = copy.deepcopy(model).cpu().float().eval()
    if not np.allclose(_selftest_logits(eager, seed, meta["input_size"]), logits, rtol=rtol, atol=atol):
        raise ValueError("reloaded weights disagree with the eager model; state_dict is incomplete")

    meta["selftest"] = {
        "seed": seed,
        "input_size": meta["input_size"],
        "logits": [float(x) for x in logits],
        "rtol": rtol,
        "atol": atol,
    }
    meta["files"] = {MODEL_FILE: sha256_file(out / MODEL_FILE)}
    (out / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return out


# -- verified loading --------------------------------------------------------


def _fail(path: Path, msg: str) -> LandmarkUnavailable:
    return LandmarkUnavailable(f"landmark bundle {path}: {msg}")


def read_meta(bundle_dir: str | os.PathLike) -> dict:
    path = Path(bundle_dir)
    try:
        meta = json.loads((path / META_FILE).read_text())
    except FileNotFoundError:
        raise _fail(path, f"missing {META_FILE}") from None
    except (OSError, ValueError) as exc:
        raise _fail(path, f"unreadable {META_FILE}: {exc}") from exc
    if not isinstance(meta, dict):
        raise _fail(path, f"{META_FILE} is not an object")
    return meta


def verify_files(bundle_dir: str | os.PathLike, meta: Mapping[str, Any]) -> None:
    """Every listed file exists and matches its sha256; model.pt is listed."""
    path = Path(bundle_dir)
    files = meta.get("files")
    if not isinstance(files, dict) or MODEL_FILE not in files:
        raise _fail(path, f"meta.files must list {MODEL_FILE}")
    for name, digest in files.items():
        if Path(name).name != name or name in ("", ".", ".."):
            raise _fail(path, f"illegal file name {name!r}")
        f = path / name
        if not f.is_file():
            raise _fail(path, f"missing file {name}")
        if sha256_file(f) != digest:
            raise _fail(path, f"sha256 mismatch for {name}")


def _floats(path: Path, meta: Mapping, key: str, n: int | None = None) -> list[float]:
    vals = meta.get(key)
    if not isinstance(vals, list) or (n is not None and len(vals) != n):
        raise _fail(path, f"{key} must be a list of {n} numbers")
    out = [float(v) for v in vals]
    if not all(np.isfinite(out)):
        raise _fail(path, f"{key} has non-finite values")
    return out


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))


def _known_keys(path: Path, name: str, d: Mapping, allowed: set[str]) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise _fail(path, f"unknown {name} keys {unknown}")


def _parse_tracker(path: Path, d: Mapping[str, Any]) -> TrackerParams:
    """TrackerParams with every value checked (JSON floats would build a
    tracker that raises on the first live socket)."""
    _known_keys(path, "tracker", d, set(TrackerParams.__dataclass_fields__))
    p = TrackerParams.from_dict(dict(d))
    counts = {k: getattr(p, k) for k in ("evidence_needed", "evidence_window", "current_k",
                                         "current_window", "current_release")}
    if not all(_is_int(v) and v >= 1 for v in counts.values()):
        raise _fail(path, f"tracker counts must be positive integers: {counts}")
    if not _is_int(p.best_cooldown) or p.best_cooldown < 0:
        raise _fail(path, f"bad tracker.best_cooldown {p.best_cooldown!r}")
    if not _is_num(p.best_margin) or p.best_margin < 1.0:
        raise _fail(path, f"bad tracker.best_margin {p.best_margin!r} (finite, >= 1)")
    if p.evidence_needed > p.evidence_window or p.current_k > p.current_window:
        raise _fail(path, "tracker needs evidence_needed <= evidence_window and current_k <= current_window")
    return p


def _parse_quality(path: Path, d: Mapping[str, Any]) -> tuple[QualityParams, float]:
    """(QualityParams, q_min) with every value checked."""
    _known_keys(path, "quality", d, set(QualityParams.__dataclass_fields__) | {"q_min"})
    q_min = d.get("q_min", DEFAULT_Q_MIN)
    if not _is_num(q_min) or not 0.0 <= q_min <= 1.0:
        raise _fail(path, f"bad quality.q_min {q_min!r}")
    p = QualityParams.from_dict({k: v for k, v in d.items() if k != "q_min"})
    if not _is_num(p.sharp_ref) or p.sharp_ref <= 0:
        raise _fail(path, f"bad quality.sharp_ref {p.sharp_ref!r}")
    if not all(_is_num(v) and 0.0 <= v <= 255.0 for v in (p.dark_y, p.sat_y)):
        raise _fail(path, "quality.dark_y/sat_y must be luminance levels in [0, 255]")
    if not all(_is_num(v) and 0.0 < v <= 1.0 for v in (p.dark_max, p.sat_max, p.redout_max)):
        raise _fail(path, "quality.dark_max/sat_max/redout_max must be fractions in (0, 1]")
    # A few px in practice; bounded so a typo cannot allocate a huge kernel per frame.
    if not _is_int(p.edge_erode_px) or not 0 <= p.edge_erode_px <= 32:
        raise _fail(path, f"bad quality.edge_erode_px {p.edge_erode_px!r} (integer 0..32)")
    return p, float(q_min)


def _parse_meta(path: Path, meta: Mapping[str, Any]) -> dict:
    from app.ml.landmarks.architectures import ARCHS

    if meta.get("schema") != SCHEMA:
        raise _fail(path, f"schema {meta.get('schema')!r} != {SCHEMA!r}")
    version = meta.get("version")
    if not isinstance(version, str) or not VERSION_RE.match(version):
        raise _fail(path, f"bad version {version!r}")
    if list(meta.get("stations") or []) != list(STATION_ORDER):
        raise _fail(path, "station list/order differs from STATION_ORDER")
    if meta.get("preprocess_version") != PREPROCESS_VERSION:
        raise _fail(path, f"preprocess_version {meta.get('preprocess_version')!r} != {PREPROCESS_VERSION!r}")
    if meta.get("layouts_version") != LAYOUTS_VERSION:
        raise _fail(path, f"layouts_version {meta.get('layouts_version')!r} != {LAYOUTS_VERSION!r}")
    arch = meta.get("arch")
    if arch not in ARCHS:
        raise _fail(path, f"unknown arch {arch!r}")
    input_size = meta.get("input_size")
    if not isinstance(input_size, int) or not 32 <= input_size <= 1024:
        raise _fail(path, f"bad input_size {input_size!r}")
    if not np.allclose(_floats(path, meta, "mean", 3), IMAGENET_MEAN) or not np.allclose(
        _floats(path, meta, "std", 3), IMAGENET_STD
    ):
        raise _fail(path, "mean/std differ from the ImageNet statistics served by preprocess.normalize")
    serve = meta.get("serve_layouts")
    if not isinstance(serve, list) or not serve or any(
        n not in LAYOUTS_BY_NAME or not LAYOUTS_BY_NAME[n].served for n in serve
    ):
        raise _fail(path, f"serve_layouts {serve!r} must name served layouts")
    temperature = float(meta.get("temperature", 0.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise _fail(path, f"bad temperature {temperature!r}")
    tau = _floats(path, meta, "tau", NUM_STATIONS)
    if not all(0.0 < t <= 1.0 for t in tau):
        raise _fail(path, "tau values must be in (0, 1]")
    auto = meta.get("auto_enabled")
    if not isinstance(auto, list) or len(auto) != NUM_STATIONS or not all(isinstance(a, bool) for a in auto):
        raise _fail(path, f"auto_enabled must be {NUM_STATIONS} booleans")
    if (meta.get("rejection") or {}).get("method") != "msp":
        raise _fail(path, "only rejection.method == 'msp' is served")
    quality = meta.get("quality")
    tracker = meta.get("tracker")
    if not isinstance(quality, dict) or not isinstance(tracker, dict):
        raise _fail(path, "quality and tracker must be objects")
    quality_params, q_min = _parse_quality(path, quality)
    tracker_params = _parse_tracker(path, tracker)
    try:  # exactly what every live socket will build
        StationTracker(tracker_params, auto)
    except Exception as exc:
        raise _fail(path, f"tracker does not build: {type(exc).__name__}: {exc}") from exc
    st = meta.get("selftest")
    if not isinstance(st, dict) or st.get("input_size") != input_size:
        raise _fail(path, "selftest missing or run at a different input_size")
    return {
        "version": version,
        "arch": arch,
        "input_size": input_size,
        "temperature": temperature,
        "tau": tuple(tau),
        "quality": quality_params,
        "q_min": q_min,
        "tracker": tracker_params,
        "auto_enabled": tuple(auto),
        "serve_layouts": tuple(serve),
    }


def _run_selftest(path: Path, model: "nn.Module", st: Mapping[str, Any]) -> None:
    expected = np.asarray(_floats(path, st, "logits", NUM_STATIONS))
    rtol, atol = float(st.get("rtol", SELFTEST_RTOL)), float(st.get("atol", SELFTEST_ATOL))
    got = _selftest_logits(model, int(st.get("seed", SELFTEST_SEED)), int(st["input_size"]))
    if not np.allclose(got, expected, rtol=rtol, atol=atol):
        diff = float(np.max(np.abs(got - expected)))
        raise _fail(path, f"selftest logits differ (max |d|={diff:.3g}, rtol={rtol}, atol={atol})")


def load_bundle(bundle_dir: str | os.PathLike, device: str = "cpu") -> LandmarkBundle:
    """Verify and load a bundle; raise LandmarkUnavailable on any mismatch."""
    path = Path(bundle_dir)
    try:
        return _load_bundle(path, device)
    except LandmarkUnavailable:
        raise
    except Exception as exc:  # torch/IO/shape errors all mean "not servable"
        raise _fail(path, f"{type(exc).__name__}: {exc}") from exc


def _load_bundle(path: Path, device: str) -> LandmarkBundle:
    import torch

    from app.ml.landmarks.architectures import build

    meta = read_meta(path)
    parsed = _parse_meta(path, meta)
    verify_files(path, meta)

    model = build(parsed["arch"], NUM_STATIONS)
    state = torch.load(path / MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.float().eval()
    _run_selftest(path, model, meta["selftest"])  # always CPU fp32

    dev = torch.device(device)
    model.to(dev)
    logger.info("Loaded landmark bundle %s (%s, %dpx) on %s", parsed["version"],
                parsed["arch"], parsed["input_size"], dev)
    return LandmarkBundle(path=path, meta=dict(meta), model=model, **parsed)


# -- BUNDLE.lock and the S3 cache --------------------------------------------


def read_lock(lock_path: str | os.PathLike) -> BundleLock:
    path = Path(lock_path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise LandmarkUnavailable(f"no bundle lock at {path}") from None
    except (OSError, ValueError) as exc:
        raise LandmarkUnavailable(f"unreadable bundle lock {path}: {exc}") from exc
    try:
        lock = BundleLock(version=data["version"], s3_key=data["s3_key"], sha256=data["sha256"].lower())
    except (KeyError, TypeError, AttributeError) as exc:
        raise LandmarkUnavailable(f"bundle lock {path} needs version, s3_key, sha256") from exc
    if not VERSION_RE.match(lock.version) or not _SHA256_RE.match(lock.sha256) or not lock.s3_key:
        raise LandmarkUnavailable(f"bundle lock {path} is malformed")
    return lock


def write_lock(lock_path: str | os.PathLike, lock: BundleLock) -> None:
    """Used by training/landmarks/export_bundle.py after the S3 upload."""
    data = {"version": lock.version, "s3_key": lock.s3_key, "sha256": lock.sha256}
    Path(lock_path).write_text(json.dumps(data, indent=2) + "\n")


def pack_bundle(bundle_dir: str | os.PathLike, tarball: str | os.PathLike) -> str:
    """Tar+gzip ``bundle_dir`` (as ``<version>/...``); return the tarball sha256."""
    src = Path(bundle_dir)
    version = read_meta(src)["version"]
    with tarfile.open(tarball, "w:gz") as tar:
        for name in sorted(p.name for p in src.iterdir() if p.is_file() and p.name != SOURCE_FILE):
            tar.add(src / name, arcname=f"{version}/{name}", recursive=False)
    return sha256_file(tarball)


def resolve_cache_dir(cache_dir: str | os.PathLike) -> Path:
    """``cache_dir`` if it is (or can be made) writable, else a temp dir."""
    path = Path(cache_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        if os.access(path, os.W_OK):
            return path
    except OSError:
        pass
    fallback = Path(tempfile.gettempdir()) / "peace-landmarks"
    fallback.mkdir(parents=True, exist_ok=True)
    logger.warning("Landmark cache dir %s is not writable; using %s", path, fallback)
    return fallback


def _source_marker(bundle_dir: Path, lock: BundleLock) -> dict:
    return {"s3_key": lock.s3_key, "sha256": lock.sha256, "meta_sha256": sha256_file(bundle_dir / META_FILE)}


def _cached_ok(target: Path, lock: BundleLock) -> bool:
    """Reuse the cache only if its marker binds it to this lock and meta.json.

    A directory without the marker (hand-copied, left over) or with an edited
    meta.json is refetched: meta.json carries every served threshold and is
    otherwise covered by no hash.
    """
    try:
        source = json.loads((target / SOURCE_FILE).read_text())
        if not isinstance(source, dict) or source != _source_marker(target, lock):
            return False
        meta = read_meta(target)
        if meta.get("version") != lock.version:
            return False
        verify_files(target, meta)
        return True
    except (LandmarkUnavailable, OSError, ValueError):
        return False


def _safe_extract(tarball: Path, dest: Path) -> None:
    with tarfile.open(tarball, "r:*") as tar:
        for m in tar.getmembers():
            parts = Path(m.name).parts
            if m.name.startswith("/") or ".." in parts or not (m.isfile() or m.isdir()):
                raise LandmarkUnavailable(f"refusing tar member {m.name!r}")
        tar.extractall(dest, filter="data")


def fetch_bundle(
    lock_path: str | os.PathLike,
    cache_dir: str | os.PathLike,
    download: Callable[[str, str], bool] | None = None,
) -> Path:
    """Return ``<cache>/<version>/`` for the locked bundle, downloading if needed.

    A cached copy is reused only if its source marker (lock sha256 and
    meta.json sha256), version and file sha256s all match. Otherwise the tarball is downloaded from private S3
    (``download(s3_key, local_path) -> bool``, default
    ``app.services.s3.download_from_s3``), checked against the lock's sha256,
    extracted into a temporary directory and renamed into place. Any failure
    raises LandmarkUnavailable; nothing half-written is left in the cache.
    """
    lock = read_lock(lock_path)
    cache = resolve_cache_dir(cache_dir)
    target = cache / lock.version
    if target.is_dir() and _cached_ok(target, lock):
        return target

    if download is None:
        from app.services import s3

        download = s3.download_from_s3

    tmp = Path(tempfile.mkdtemp(prefix=f".{lock.version}.tmp-", dir=cache))
    try:
        tarball = tmp / "bundle.tar.gz"
        if not download(lock.s3_key, str(tarball)) or not tarball.is_file():
            raise LandmarkUnavailable(f"could not download landmark bundle {lock.s3_key} (S3 configured?)")
        digest = sha256_file(tarball)
        if digest != lock.sha256:
            raise LandmarkUnavailable(f"landmark bundle tarball sha256 {digest} != lock {lock.sha256}")
        extracted = tmp / "x"
        _safe_extract(tarball, extracted)
        root = extracted / lock.version if (extracted / lock.version / META_FILE).is_file() else extracted
        meta = read_meta(root)
        if meta.get("version") != lock.version:
            raise LandmarkUnavailable(f"bundle version {meta.get('version')!r} != lock {lock.version!r}")
        verify_files(root, meta)
        (root / SOURCE_FILE).write_text(json.dumps(_source_marker(root, lock)))
        if target.exists():
            shutil.rmtree(target)
        os.replace(root, target)
        logger.info("Fetched landmark bundle %s into %s", lock.version, target)
        return target
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
