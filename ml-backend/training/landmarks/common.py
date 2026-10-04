"""Paths, clip discovery and ffmpeg helpers shared by the landmark data pipeline.

Everything that reads the raw dataset goes through ``iter_clips`` so the
full-length recordings in ``data/landmarks/Full lenght videos/`` (sic) can
never enter the pipeline: that folder is skipped at listing time, and every
audit table ingested from elsewhere is passed through ``drop_full_length``.

Privacy: the repo is public. Anything derived from the dataset (manifest,
overrides, OCR, reports, frame cache) is written under the gitignored
``data/landmarks_private/`` or ``training/landmarks/cache/``. Only aggregate
counts (``manifest_summary_v1.json``) are written inside the tracked tree.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from app.ml.landmarks.stations import FOLDER_TO_STATION

ML_BACKEND = Path(__file__).resolve().parents[2]
REPO_ROOT = ML_BACKEND.parent
DATA_ROOT = REPO_ROOT / "data" / "landmarks"
PRIVATE_ROOT = REPO_ROOT / "data" / "landmarks_private"
AUDIT_DIR = PRIVATE_ROOT / "audit"
OVERRIDES_DIR = PRIVATE_ROOT / "overrides"
ADJUDICATION_DIR = PRIVATE_ROOT / "adjudication"
REPORTS_DIR = PRIVATE_ROOT / "reports"
MANIFEST_PATH = PRIVATE_ROOT / "manifest_v1.csv"
CACHE_ROOT = ML_BACKEND / "training" / "landmarks" / "cache" / "v1"
SUMMARY_PATH = ML_BACKEND / "training" / "landmarks" / "manifest_summary_v1.json"

# The dataset folder name is misspelled on disk; match it exactly and also
# anything that merely looks like it, so a rename can't let it slip through.
FULL_LENGTH_DIR = "Full lenght videos"
_FULL_LENGTH_RE = re.compile(r"full\s*le[nght]+\s*video|(^|/)REC_\d+", re.I)

# Explicit BT.709 limited-range -> full-range RGB, matching how browsers
# decode these bt709/tv clips. Never rely on ffmpeg's default matrix.
FFMPEG_COLOR_VF = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"


def is_full_length(value: str | os.PathLike) -> bool:
    """True for any path, folder or file name belonging to the full-length set."""
    s = str(value)
    return FULL_LENGTH_DIR in s or bool(_FULL_LENGTH_RE.search(s))


def drop_full_length(rows: Iterable[dict], keys: tuple[str, ...] = ("cls", "file", "rel_path")) -> list[dict]:
    """Filter audit/manifest rows: drop every row whose key fields look full-length."""
    return [r for r in rows if not any(is_full_length(r.get(k) or "") for k in keys)]


@dataclass(frozen=True)
class ClipFile:
    rel_path: str  # relative to DATA_ROOT, e.g. "Antrum/HT123-a.mp4"
    folder: str
    file: str

    @property
    def path(self) -> Path:
        return DATA_ROOT / self.rel_path

    @property
    def folder_class(self) -> str:
        return FOLDER_TO_STATION[self.folder]


def iter_clips(root: Path = DATA_ROOT) -> list[ClipFile]:
    """All station clips, sorted. Only the 10 known class folders are listed.

    The full-length folder is never opened (not even listed), and unknown
    folders are ignored, so nothing outside the station set can leak in.
    """
    out: list[ClipFile] = []
    for folder in sorted(os.listdir(root)):
        if is_full_length(folder) or folder not in FOLDER_TO_STATION:
            continue
        d = root / folder
        if not d.is_dir():
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".mp4") and not f.startswith(".") and not is_full_length(f):
                out.append(ClipFile(f"{folder}/{f}", folder, f))
    return out


# --- hashing -----------------------------------------------------------------

def file_digest(path: Path, algo: str = "md5", chunk: int = 1 << 22) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    return file_digest(path, "sha256")


def _salt() -> bytes:
    """Per-installation secret salt, so hashed serials can't be brute-forced
    (a scope serial has only ~10^7 values). Created once, never committed."""
    p = PRIVATE_ROOT / ".salt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(secrets.token_hex(16))
    return p.read_text().strip().encode()


def private_hash(kind: str, value: str, n: int = 10) -> str:
    """Salted, truncated hash for identifiers that must not appear in clear."""
    if not value:
        return ""
    return hashlib.sha256(_salt() + kind.encode() + b"\0" + value.encode()).hexdigest()[:n]


# --- OSD time ------------------------------------------------------------------

_OSD_TS = re.compile(r"(\d{2})/(\d{2})/(\d{4})\D{1,3}(\d{1,2})\D(\d{2})\D(\d{2})")


def parse_osd_datetime(text: str) -> datetime | None:
    """Parse the processor clock ("dd/mm/yyyy HH:MM:SS") from OCR text."""
    m = _OSD_TS.search(text or "")
    if not m:
        return None
    d, mo, y, hh, mm, ss = map(int, m.groups())
    try:
        return datetime(y, mo, d, hh, mm, ss)
    except ValueError:
        return None


# --- ffmpeg ------------------------------------------------------------------

def ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
    return out.splitlines()[0] if out else "unknown"


def probe(path: Path) -> dict:
    """width, height, n_frames, fps and dur_s of the first video stream."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    info = json.loads(res.stdout or "{}")
    st = (info.get("streams") or [{}])[0]
    num, _, den = str(st.get("r_frame_rate", "30/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    dur = float(st.get("duration") or info.get("format", {}).get("duration") or 0.0)
    nb = st.get("nb_frames")
    n_frames = int(nb) if nb and str(nb).isdigit() else int(round(dur * fps))
    return {"width": int(st.get("width", 0)), "height": int(st.get("height", 0)),
            "n_frames": n_frames, "fps": round(fps, 4), "dur_s": round(dur, 3)}


def decode_frames(path: Path, width: int, height: int, every: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (source frame index, HxWx3 uint8 RGB) for every ``every``-th frame.

    Uses the explicit BT.709 colour chain. Frame selection happens inside
    ffmpeg (``select`` on the decoded frame number, passthrough timing) so
    skipped frames never cross the pipe and index k maps to source k*every.
    """
    vf = f"select='not(mod(n\\,{every}))'," + FFMPEG_COLOR_VF if every > 1 else FFMPEG_COLOR_VF
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-i", str(path), "-vf", vf,
           "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    size = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=size)
    try:
        k = 0
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield k * every, np.frombuffer(buf, np.uint8).reshape(height, width, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def grab_frame(path: Path, width: int, height: int, t_s: float = 0.0) -> np.ndarray | None:
    """One RGB frame at time ``t_s`` (seek before input; fast, keyframe-accurate)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(t_s, 0):.3f}", "-i", str(path),
           "-frames:v", "1", "-vf", FFMPEG_COLOR_VF, "-f", "rawvideo", "-"]
    buf = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    if len(buf) < width * height * 3:
        return None
    return np.frombuffer(buf[: width * height * 3], np.uint8).reshape(height, width, 3)


# --- tables ------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    """Atomic CSV write (tmp + rename); ``.gz`` paths are gzip-compressed."""
    fields = fields or (list(rows[0]) if rows else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: _fmt(r.get(k)) for k in fields})
    tmp = path.with_name(path.name + ".tmp")
    data = buf.getvalue().encode("utf-8")
    if str(path).endswith(".gz"):
        data = gzip.compress(data, mtime=0)
    tmp.write_bytes(data)
    tmp.replace(path)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "" if v != v else f"{v:.6g}"
    return str(v)


def as_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False, allow_nan=False) + "\n")
    tmp.replace(path)
