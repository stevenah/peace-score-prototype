"""Build the clip manifest: data/landmarks_private/manifest_v1.csv.

One row per clip file, except that byte-identical copies filed under two
different station folders (the contested Incisura / Lesser-curvature clips)
collapse into ONE row carrying a soft label. Nothing under
``Full lenght videos/`` is ever listed (``common.iter_clips``).

    python -m training.landmarks.build_manifest import-audit <scratchpad>
    python -m training.landmarks.build_manifest            # build / rebuild

Two phases, one command. Phase A needs only the clips: ffprobe + md5,
filename IDs and suffixes, OSD clock (audit OCR), processor fingerprint
(tesseract on the OSD panel), duplicates, grouping, time overlaps and the
exclusion policy. Phase B runs when the frame cache exists
(``extract_cache``): per-clip NBI fraction, frozen and cap flags, and
near-duplicates, all measured on the canonical frames the model will see.
Re-run ``folds`` after every rebuild: this script resets split/fold.

Policy (plan M1.3):
- exact duplicates in one folder: keep the canonical name, others dup_exact
- identical file in two folders: one row, soft label over both stations,
  eval_exclude (clinician adjudication pending)
- re-encoded near duplicates: dup_near; <= 2 frames: too_short
- unresolvable patient: id_uncertain; suffix names another station:
  quarantine_suffix_mismatch; "+" combos: combo (kept for replay)
- legacy 1350x1080 cohort: split=shift, never trained on
- group_id: the patient, merged with another patient only when a clip's OSD
  time falls inside that patient's (date, processor) window +-2 min. Two
  rooms record at once, so time proximity alone never merges.

No absolute dates or scope serials are written: days, processors and scopes
are salted hashes, times are relative (``rel_t_s``) or time-of-day.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.ml.landmarks.layouts import LAYOUTS_VERSION, LEGACY_1350, OLYMPUS_1920
from app.ml.landmarks.preprocess import PREPROCESS_VERSION
from app.ml.landmarks.stations import SCHEMA, STATION_INDEX, STATION_ORDER
from training.landmarks.common import (
    AUDIT_DIR,
    CACHE_ROOT,
    MANIFEST_PATH,
    OVERRIDES_DIR,
    SUMMARY_PATH,
    ClipFile,
    as_bool,
    drop_full_length,
    file_digest,
    is_full_length,
    iter_clips,
    parse_osd_datetime,
    private_hash,
    probe,
    read_csv,
    write_csv,
    write_json,
)
from training.landmarks.ids import load_id_overrides, parse_clip_name, resolve_patient_id

MANIFEST_VERSION = "v1"

MANIFEST_FIELDS = [
    # identity and label
    "clip_uid", "rel_path", "alt_rel_path", "folder_class", "suffix", "suffix_class",
    "label", "label_source", "soft_label",
    # exclusion
    "exclude_reason", "eval_exclude", "flags", "dup_group", "near_dup_group",
    # patient, grouping, recording context (hashed)
    "patient_id", "patient_id_raw", "id_resolution", "group_id",
    "room_hash", "scope_hash", "day_key", "tod_s",
    # recording
    "cohort", "layout", "width", "height", "n_frames", "fps", "dur_s", "md5",
    # measured on cached canonical frames (phase B)
    "nbi_frac", "mode_major", "cap", "cap_score", "frozen", "n_cached", "n_unique",
    # chronology
    "rel_t_s", "order_in_patient", "overlap_head_s", "overlap_tail_s", "overlap_with",
    # split
    "split", "fold",
]

# Exclusion reasons, most important first; a row carries the first that applies.
EXCLUDE_PRIORITY = (
    "dup_exact", "too_short", "dup_near", "id_uncertain", "manual",
    "adjudicated_exclude", "combo", "quarantine_suffix_mismatch",
)
# Rows with these reasons have no safe patient group: split=excluded. Combo
# and quarantined rows keep their patient's split so replays can use them.
NO_GROUP_REASONS = {"dup_exact", "too_short", "dup_near", "id_uncertain"}

MIN_FRAMES = 3  # clips with <= 2 frames are excluded
WINDOW_PAD = timedelta(minutes=2)  # (date, room) window padding for merges
OSD_TOL_S = 2.0  # |t_last - t_first - duration| tolerated for a consistent OCR pair

# Phase-B thresholds (measured on the cache; see README "Numbers").
NBI_MAJOR, WL_MAJOR = 0.8, 0.2
FROZEN_UNIQUE_FRAC = 0.25
FROZEN_MIN_UNIQUE = 4


# --- audit import ------------------------------------------------------------

AUDIT_IMPORTS = {
    # dst name: (relative source path in the audit scratchpad, row keys to filter on)
    "clips_audit.csv": "audit_ds/clips_audit.csv",
    "ocr.csv": "audit_ds/ocr.csv",
    "nbi_flags.csv": "audit_ds/nbi_flags.csv",
    "landmarks_probe.csv": "landmarks_probe.csv",
    "scan.csv": "scan.csv",
    "md5_sorted.txt": "md5_sorted.txt",
}


def import_audit(src: Path) -> dict[str, tuple[int, int]]:
    """Copy the read-only audit tables into AUDIT_DIR, dropping every
    full-length (REC_*) row, so the pipeline never depends on the scratchpad.
    Returns {file: (rows_in, rows_kept)}."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for dst, rel in AUDIT_IMPORTS.items():
        p = src / rel
        if not p.exists():
            continue
        if dst.endswith(".csv"):
            rows = read_csv(p)
            kept = drop_full_length(rows)
            write_csv(AUDIT_DIR / dst, kept, list(rows[0]) if rows else [])
        else:
            lines = p.read_text().splitlines()
            kept = [l for l in lines if not is_full_length(l)]
            (AUDIT_DIR / dst).write_text("\n".join(kept) + "\n")
            rows = lines
        out[dst] = (len(rows), len(kept))
    return out


# --- probe + md5 (cached) ----------------------------------------------------

PROBE_CACHE = AUDIT_DIR / "probe_cache.csv"
PROBE_FIELDS = ["rel_path", "size", "mtime_ns", "md5", "width", "height", "n_frames", "fps", "dur_s"]


def probe_all(clips: list[ClipFile], workers: int = 8) -> dict[str, dict]:
    cache = {r["rel_path"]: r for r in read_csv(PROBE_CACHE)} if PROBE_CACHE.exists() else {}

    def one(c: ClipFile) -> dict:
        st = c.path.stat()
        hit = cache.get(c.rel_path)
        if hit and int(hit["size"]) == st.st_size and int(hit["mtime_ns"]) == st.st_mtime_ns:
            return hit
        return {"rel_path": c.rel_path, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                "md5": file_digest(c.path), **probe(c.path)}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, clips))
    write_csv(PROBE_CACHE, rows, PROBE_FIELDS)
    out = {}
    for r in rows:
        out[r["rel_path"]] = {"md5": r["md5"], "width": int(r["width"]), "height": int(r["height"]),
                              "n_frames": int(r["n_frames"]), "fps": float(r["fps"]),
                              "dur_s": float(r["dur_s"])}
    return out


# --- OSD clock ---------------------------------------------------------------

def osd_times(clips: list[ClipFile], info: dict[str, dict],
              pid: dict[str, str | None]) -> dict[str, tuple[datetime | None, bool]]:
    """Clip start time from the audit OCR of the first and last frame.

    Returns rel_path -> (start time or None, firm). A time is *firm* when the
    first-frame reading agrees with the last-frame reading minus the
    duration. Otherwise the candidate closest to the patient's median firm
    time is used, but it is not firm (the legacy recorder's OSD is small and
    misreads digits) and never drives a group merge.
    """
    ocr = {}
    if (AUDIT_DIR / "ocr.csv").exists():
        for r in drop_full_length(read_csv(AUDIT_DIR / "ocr.csv")):
            ocr[f"{r['cls']}/{r['file']}"] = r
    cands: dict[str, list[datetime]] = {}
    firm: dict[str, datetime] = {}
    for c in clips:
        r = ocr.get(c.rel_path)
        if not r:
            cands[c.rel_path] = []
            continue
        dur = info[c.rel_path]["dur_s"]
        t0 = parse_osd_datetime(r["ocr_first"])
        t1 = parse_osd_datetime(r["ocr_last"])
        t1s = t1 - timedelta(seconds=round(dur)) if t1 else None
        if t0 and t1 and abs((t1 - t0).total_seconds() - dur) <= OSD_TOL_S:
            firm[c.rel_path] = t0
        cands[c.rel_path] = [t for t in (t0, t1s) if t]
    by_patient: dict[str, list[float]] = defaultdict(list)
    for rp, t in firm.items():
        if pid.get(rp):
            by_patient[pid[rp]].append(t.timestamp())
    out: dict[str, tuple[datetime | None, bool]] = {}
    for rp, cs in cands.items():
        if rp in firm:
            out[rp] = (firm[rp], True)
        elif cs:
            ref = by_patient.get(pid.get(rp) or "")
            med = float(np.median(ref)) if ref else cs[0].timestamp()
            out[rp] = (min(cs, key=lambda t: abs(t.timestamp() - med)), False)
        else:
            out[rp] = (None, False)
    return out


# --- OSD panel: processor (room) and scope fingerprints --------------------------

PANEL_CACHE = AUDIT_DIR / "osd_panel.csv"
PANEL_FIELDS = ["md5", "rel_path", "model", "serial", "comment", "button2", "processor", "raw"]
# Scope/serial/comment/button lines of the Olympus OSD panel (reference px).
_PANEL_BOX = (20, 420, 540, 980)


def _ocr_panel_text(path: Path, width: int, t_s: float) -> str:
    s = width / OLYMPUS_1920.ref_width
    x0, y0, x1, y1 = (int(round(v * s)) for v in _PANEL_BOX)
    vf = f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale={2 * (x1 - x0)}:{2 * (y1 - y0)},format=gray,negate"
    png = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{t_s:.2f}", "-i", str(path),
                          "-frames:v", "1", "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True, timeout=60).stdout
    if not png:
        return ""
    res = subprocess.run(["tesseract", "stdin", "stdout", "--psm", "6"], input=png,
                         capture_output=True, timeout=60)
    return res.stdout.decode(errors="ignore")


def parse_panel(text: str) -> dict:
    """Scope model/serial and the processor fingerprint from panel OCR text.

    Two processors record concurrently. They differ in configuration, not in
    scopes (scopes move between rooms): processor A shows the empty comment
    placeholder ("Komentarz") and assigns scope button 2 to "Oswietlenie
    obserwacyjne"; processor B has a filled-in comment and button 2 = "NBI".
    The button is decisive; the placeholder only confirms A (the text of a
    filled-in comment is site-specific and is not matched in this public code).
    """
    model = m.group(1) if (m := re.search(r"GIF-?\s*([A-Z0-9]{3,8})", text)) else ""
    serial = m.group(1) if (m := re.search(r"\b(\d{7})\b", text)) else ""
    comment = "placeholder" if re.search(r"Komentar", text) else ""
    if re.search(r"wietlenie|opcjonal", text):
        button2 = "light"
    elif re.search(r"^\W*\S?\s*NBI\s*$", text, re.M) or re.search(r"Pompa", text):
        button2 = "nbi"
    else:
        button2 = ""
    votes = {("placeholder", "light"): "A", ("placeholder", ""): "A", ("", "light"): "A", ("", "nbi"): "B"}
    processor = votes.get((comment, button2), "")  # contradictory or empty -> unknown
    return {"model": model, "serial": serial, "comment": comment, "button2": button2, "processor": processor}


def osd_panels(clips: list[ClipFile], info: dict[str, dict], workers: int = 8) -> dict[str, dict]:
    """Panel fingerprint per md5 (cached). Legacy clips have no panel: their
    OSD sits inside the image and a single recorder was used."""
    cache = {r["md5"]: r for r in read_csv(PANEL_CACHE)} if PANEL_CACHE.exists() else {}
    for r in cache.values():  # the parser, not the cached fields, is authoritative
        r.update(parse_panel(r["raw"].replace(" / ", "\n")))
    todo = {}
    for c in clips:
        i = info[c.rel_path]
        if i["md5"] not in cache and i["width"] == OLYMPUS_1920.ref_width:
            todo[i["md5"]] = c

    def one(c: ClipFile) -> dict:
        i = info[c.rel_path]
        text = _ocr_panel_text(c.path, i["width"], 0.0)
        p = parse_panel(text)
        if not p["processor"] and i["dur_s"] > 1:  # retry on the middle frame
            text2 = _ocr_panel_text(c.path, i["width"], i["dur_s"] / 2)
            p2 = parse_panel(text2)
            if p2["processor"]:
                p, text = p2, text2
        return {"md5": i["md5"], "rel_path": c.rel_path, **p, "raw": " / ".join(text.split("\n"))}

    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for r in ex.map(one, todo.values()):
                cache[r["md5"]] = r
        write_csv(PANEL_CACHE, sorted(cache.values(), key=lambda r: r["rel_path"]), PANEL_FIELDS)
    return cache


# --- manifest rows -------------------------------------------------------------

@dataclass
class Overrides:
    labels: dict[str, dict]
    exclusions: dict[str, dict]


def load_overrides() -> Overrides:
    labels = {r["clip_uid"]: r for r in read_csv(OVERRIDES_DIR / "label_overrides.csv")} \
        if (OVERRIDES_DIR / "label_overrides.csv").exists() else {}
    excl = {r["key"]: r for r in read_csv(OVERRIDES_DIR / "exclusions.csv")} \
        if (OVERRIDES_DIR / "exclusions.csv").exists() else {}
    for r in labels.values():
        _parse_label_value(r["label"])  # validate early
    return Overrides(labels, excl)


def _parse_label_value(v: str) -> tuple[str, str]:
    """label_overrides 'label' -> (hard label or "", soft label or "").

    Accepts a station key, "exclude", "keep" (confirm the folder label), or a
    soft label "k1:0.5|k2:0.5" whose weights sum to 1.
    """
    v = v.strip()
    if v in ("exclude", "keep") or v in STATION_INDEX:
        return v, ""
    parts = [p.split(":") for p in v.split("|")]
    if all(len(p) == 2 and p[0] in STATION_INDEX for p in parts):
        if abs(sum(float(w) for _, w in parts) - 1.0) < 1e-6:
            return "", v
    raise ValueError(f"bad label override value {v!r}")


def soft_label(stations: list[str]) -> str:
    keys = sorted(set(stations), key=STATION_INDEX.__getitem__)
    w = 1.0 / len(keys)
    return "|".join(f"{k}:{w:.4g}" for k in keys)


def _canonical_sort_key(c: ClipFile, parsed) -> tuple:
    # Prefer no copy markers, then the shorter / alphabetically first name.
    return (len(parsed.copies), len(c.file), c.file)


def _row_canonical_key(r: dict) -> tuple:
    file = r["rel_path"].rsplit("/", 1)[-1]
    return (len(parse_clip_name(file).copies), len(file), file)


def resolve_ids(clips: list[ClipFile], id_overrides) -> dict[str, tuple[str | None, str]]:
    return {c.rel_path: resolve_patient_id(parse_clip_name(c.file), id_overrides) for c in clips}


def build_rows(clips: list[ClipFile], info: dict[str, dict], panels: dict[str, dict], ov: Overrides,
               id_overrides, times: dict[str, tuple[datetime | None, bool]] | None = None) -> list[dict]:
    """One row per clip (identical cross-folder copies collapsed) with the
    exclusion policy applied. ``times``: rel_path -> (OSD start, firm)."""
    parsed = {c.rel_path: parse_clip_name(c.file) for c in clips}
    pid_res = resolve_ids(clips, id_overrides)
    times = times or {}

    by_md5: dict[str, list[ClipFile]] = defaultdict(list)
    for c in clips:
        by_md5[info[c.rel_path]["md5"]].append(c)

    rows: list[dict] = []
    for md5, members in sorted(by_md5.items(), key=lambda kv: kv[1][0].rel_path):
        folders = sorted({c.folder_class for c in members}, key=STATION_INDEX.__getitem__)
        cross_folder = len(folders) > 1

        def canon_key(c: ClipFile) -> tuple:
            p = parsed[c.rel_path]
            # For cross-folder copies, the copy whose folder matches the suffix wins.
            return (p.suffix_class != c.folder_class,) + _canonical_sort_key(c, p)

        members = sorted(members, key=canon_key)
        canon = members[0]
        dup_group = md5[:12] if len(members) > 1 else ""
        emitted = [canon] if cross_folder else members
        for k, c in enumerate(emitted):
            p = parsed[c.rel_path]
            i = info[c.rel_path]
            pid, how = pid_res[c.rel_path]
            flags: list[str] = []
            reasons: list[str] = []
            row = {
                "clip_uid": md5[:12] if k == 0 else f"{md5[:12]}-dup{k}",
                "rel_path": c.rel_path,
                "alt_rel_path": "|".join(m.rel_path for m in members[1:]) if cross_folder else "",
                "folder_class": c.folder_class,
                "suffix": p.suffix,
                "suffix_class": p.suffix_class,
                "label": c.folder_class,
                "label_source": "folder",
                "soft_label": "",
                "dup_group": dup_group,
                "patient_id": pid or "",
                "patient_id_raw": p.raw_id or "",
                "id_resolution": how,
                "width": i["width"], "height": i["height"], "n_frames": i["n_frames"],
                "fps": i["fps"], "dur_s": i["dur_s"], "md5": md5,
                "_t": times.get(c.rel_path, (None, False))[0],
                "_t_firm": times.get(c.rel_path, (None, False))[1],
            }
            if i["width"] == OLYMPUS_1920.ref_width and i["height"] == OLYMPUS_1920.ref_height:
                row.update(cohort="new_1920", layout=OLYMPUS_1920.name)
            elif i["width"] == LEGACY_1350.ref_width and i["height"] == LEGACY_1350.ref_height:
                row.update(cohort="legacy_1350", layout=LEGACY_1350.name)
            else:
                row.update(cohort="unknown", layout="")
                reasons.append("manual")
                flags.append("unknown_layout")
            if k > 0:
                reasons.append("dup_exact")
            if i["n_frames"] < MIN_FRAMES:
                reasons.append("too_short")
            if pid is None:
                reasons.append("id_uncertain")
            if cross_folder:
                row["soft_label"] = soft_label(folders)
                flags.append("contested")
            elif p.is_combo:
                reasons.append("combo")
            elif p.suffix_class and p.suffix_class != c.folder_class:
                reasons.append("quarantine_suffix_mismatch")
            if p.is_combo:
                flags.append("combo")
            if p.unknown_tokens and not p.is_combo:
                flags.append("suffix_ambiguous")
            # Reviewable overrides (by clip_uid; exclusions also by rel_path).
            ex = ov.exclusions.get(row["clip_uid"]) or ov.exclusions.get(c.rel_path)
            if ex:
                reasons.append("manual")
                flags.append(f"manual:{ex['exclude_reason']}")
            lo = ov.labels.get(row["clip_uid"])
            if lo:
                hard, soft = _parse_label_value(lo["label"])
                reasons = [r for r in reasons if r not in ("combo", "quarantine_suffix_mismatch")]
                row["label_source"] = "adjudicated"
                if hard == "exclude":
                    reasons.append("adjudicated_exclude")
                elif hard == "keep":
                    row["soft_label"] = ""
                elif hard:
                    row["label"], row["soft_label"] = hard, ""
                else:
                    row["soft_label"] = soft
                flags.append("adjudicated")
            reason = next((r for r in EXCLUDE_PRIORITY if r in reasons), "")
            row["exclude_reason"] = reason
            # Soft-labelled (contested) clips train but stay out of primary eval.
            row["eval_exclude"] = bool(reason) or bool(row["soft_label"])
            row["_flags"] = flags
            panel = panels.get(md5, {})
            row["_processor"] = "legacy" if row["cohort"] == "legacy_1350" else panel.get("processor", "")
            row["_scope"] = f"{panel.get('model', '')}:{panel.get('serial', '')}" if panel.get("serial") else ""
            rows.append(row)
    return rows


# --- grouping and chronology -----------------------------------------------------

class _DSU:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def impute_processor(rows: list[dict]) -> None:
    """Clips whose panel OCR failed inherit their patient's majority processor."""
    maj: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["patient_id"] and r["_processor"]:
            maj[r["patient_id"]][r["_processor"]] += 1
    for r in rows:
        if not r["_processor"] and maj.get(r["patient_id"]):
            r["_processor"] = maj[r["patient_id"]].most_common(1)[0][0]
            r["_flags"].append("processor_imputed")


def session_windows(rows: list[dict]) -> dict[tuple[str, object, str], tuple[datetime, datetime]]:
    """(patient, date, processor) -> [first start - pad, last end + pad], from
    firm OSD times only."""
    spans: dict[tuple, list[datetime]] = defaultdict(list)
    for r in rows:
        if r["patient_id"] and r["_t_firm"] and r["_processor"] and r["exclude_reason"] not in NO_GROUP_REASONS:
            t0 = r["_t"]
            spans[(r["patient_id"], t0.date(), r["_processor"])] += [t0, t0 + timedelta(seconds=r["dur_s"])]
    return {k: (min(v) - WINDOW_PAD, max(v) + WINDOW_PAD) for k, v in spans.items()}


def assign_groups(rows: list[dict]) -> list[tuple[str, str]]:
    """group_id per row; returns the (patient, patient) merges that were made.

    A clip of patient P links P to patient Q only if its OSD start lies in
    Q's window for the SAME date and processor. Concurrent procedures in the
    other room never merge.
    """
    wins = session_windows(rows)
    by_day_proc: dict[tuple, list[tuple[str, datetime, datetime]]] = defaultdict(list)
    for (p, d, proc), (a, b) in wins.items():
        by_day_proc[(d, proc)].append((p, a, b))
    dsu = _DSU()
    merges = []
    for r in rows:
        p, t = r["patient_id"], r["_t"]
        if not p:
            continue
        dsu.find(p)
        if not r["_t_firm"] or not r["_processor"] or r["exclude_reason"] in NO_GROUP_REASONS:
            continue
        for q, a, b in by_day_proc.get((t.date(), r["_processor"]), ()):
            if q != p and a <= t <= b:
                dsu.union(p, q)
                merges.append((p, q))
    for r in rows:
        r["group_id"] = dsu.find(r["patient_id"]) if r["patient_id"] else ""
    return sorted(set(tuple(sorted(m)) for m in merges))


def chronology(rows: list[dict]) -> None:
    """rel_t_s / order_in_patient, and cross-label time overlaps per patient.

    overlap_head_s / overlap_tail_s: seconds at the start / end of this clip
    that overlap (OSD clock, 1 s resolution) another clip of the same patient
    with a DIFFERENT label; training drops those frames.
    """
    by_p: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        r.update(rel_t_s="", order_in_patient="", overlap_head_s="", overlap_tail_s="", overlap_with="")
        if r["patient_id"]:
            by_p[r["patient_id"]].append(r)
    for L in by_p.values():
        timed = sorted((r for r in L if r["_t"]), key=lambda r: (r["_t"], r["rel_path"]))
        if not timed:
            continue
        t_first = timed[0]["_t"]
        for k, r in enumerate(timed):
            r["rel_t_s"] = round((r["_t"] - t_first).total_seconds(), 1)
            r["order_in_patient"] = k
        usable = [r for r in timed if r["exclude_reason"] not in NO_GROUP_REASONS]
        for a in usable:
            a0 = a["_t"]
            a1 = a0 + timedelta(seconds=a["dur_s"])
            head = tail = 0.0
            others = []
            for b in usable:
                if b is a or _labels(b) == _labels(a):
                    continue
                b0 = b["_t"]
                b1 = b0 + timedelta(seconds=b["dur_s"])
                ov = (min(a1, b1) - max(a0, b0)).total_seconds()
                if ov <= 0:
                    continue
                others.append(b["clip_uid"])
                if b0 <= a0:
                    head = max(head, (min(a1, b1) - a0).total_seconds())
                if b1 >= a1:
                    tail = max(tail, (a1 - max(a0, b0)).total_seconds())
            if others:
                a["overlap_head_s"] = round(min(head, a["dur_s"]), 2)
                a["overlap_tail_s"] = round(min(tail, a["dur_s"]), 2)
                a["overlap_with"] = "|".join(sorted(others))
                a["_flags"].append("time_overlap_other_label")


def _labels(r: dict) -> frozenset:
    if r["soft_label"]:
        return frozenset(x.split(":")[0] for x in r["soft_label"].split("|"))
    return frozenset([r["label"]])


# --- phase B: measurements from the frame cache ------------------------------------

def apply_cache(rows: list[dict], cache_root: Path = CACHE_ROOT) -> dict:
    """Fill NBI fraction, mode, frozen, cap and near-duplicates from the cache."""
    frames_path = cache_root / "frames.csv.gz"
    if not frames_path.exists():
        return {"cache": "absent"}
    import pandas as pd

    from training.landmarks.derive_layouts import CAP_PATIENT_SHARE, CAP_THRESHOLD, cap_scores
    from training.landmarks.extract_cache import near_duplicate_pairs

    fr = pd.read_csv(frames_path, dtype={"clip_uid": str, "dup_of": "Int64"})
    g = fr.groupby("clip_uid")
    per = pd.DataFrame({
        "n_cached": g.size(),
        "n_unique": g["dup_of"].apply(lambda s: int(s.isna().sum())),
        "nbi_frac": g["mode_frame"].apply(lambda s: float((s != "wl").mean())),
    })
    caps = cap_scores(cache_root, [r for r in rows if r["clip_uid"] in per.index])
    for r in rows:
        uid = r["clip_uid"]
        if uid not in per.index:
            continue
        s = per.loc[uid]
        n, nu, nbi = int(s["n_cached"]), int(s["n_unique"]), float(s["nbi_frac"])
        r["n_cached"], r["n_unique"], r["nbi_frac"] = n, nu, round(nbi, 4)
        r["mode_major"] = "nbi" if nbi >= NBI_MAJOR else ("wl" if nbi <= WL_MAJOR else "mixed")
        r["frozen"] = bool(nu < FROZEN_MIN_UNIQUE or nu / max(n, 1) < FROZEN_UNIQUE_FRAC)
        if uid in caps:
            r["cap_score"] = round(caps[uid], 4)
            r["cap"] = bool(caps[uid] >= CAP_THRESHOLD)
        if r["frozen"]:
            r["_flags"].append("frozen")
    # The cap is attached for a whole procedure; rings are missed in some
    # close-ups, so a patient with most clips above threshold is cap-positive.
    share: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        if r.get("cap") in (True, False) and r["patient_id"]:
            share[r["patient_id"]].append(r["cap"])
    cap_patients = {p for p, v in share.items() if np.mean(v) >= CAP_PATIENT_SHARE}
    for r in rows:
        if r["patient_id"] in cap_patients and r.get("cap") is False:
            r["cap"] = True
            r["_flags"].append("cap_by_patient")
    # Near duplicates: full-content matches become dup_near (keep the
    # canonical name); partial matches only link clips into a near_dup_group.
    pairs = near_duplicate_pairs(cache_root, fr, [r for r in rows if r["clip_uid"] in per.index])
    by_uid = {r["clip_uid"]: r for r in rows}
    dsu = _DSU()
    n_full = n_partial = n_cross = 0
    for a, b, frac_a, frac_b in pairs:
        ra, rb = by_uid[a], by_uid[b]
        if ra["md5"] == rb["md5"]:
            continue
        dsu.union(a, b)
        n_cross += ra["patient_id"] != rb["patient_id"]
        if min(frac_a, frac_b) >= 0.8:
            n_full += 1
            loser = max((ra, rb), key=_row_canonical_key)
            if "dup_near" not in loser["_flags"]:
                loser["_flags"].append("dup_near")
        else:
            n_partial += 1
    members: dict[str, list[str]] = defaultdict(list)
    for uid in list(dsu.p):
        members[dsu.find(uid)].append(uid)
    for root, uids in members.items():
        if len(uids) > 1:
            for u in uids:
                by_uid[u]["near_dup_group"] = f"nd-{root}"
    return {"cache": "present", "clips_measured": int(len(per)), "near_dup_full_pairs": n_full,
            "near_dup_partial_pairs": n_partial, "near_dup_cross_patient_pairs": n_cross}


def finalize(rows: list[dict]) -> None:
    for r in rows:
        if "dup_near" in r["_flags"] and r["exclude_reason"] not in ("dup_exact", "too_short"):
            r["exclude_reason"] = "dup_near"
            r["eval_exclude"] = True
        if r["exclude_reason"] in NO_GROUP_REASONS:
            r["split"] = "excluded"
        elif r["cohort"] == "legacy_1350":
            r["split"] = "shift"
        else:
            r["split"] = ""
        r["fold"] = -1
        r["flags"] = "|".join(dict.fromkeys(r["_flags"]))
        r["room_hash"] = private_hash("processor", r["_processor"])
        r["scope_hash"] = private_hash("scope", r["_scope"])
        t = r["_t"]
        r["day_key"] = private_hash("day", t.date().isoformat()) if t else ""
        r["tod_s"] = t.hour * 3600 + t.minute * 60 + t.second if t else ""


# --- summary ---------------------------------------------------------------------

def summarize(rows: list[dict]) -> dict:
    """Aggregate counts only: no IDs, no dates, no paths."""
    def mode(r):
        return r.get("mode_major") or "unknown"

    cells = Counter((r["label"], r["split"] or "unassigned", mode(r), r["cohort"]) for r in rows
                    if not r["exclude_reason"])
    by_class_split = defaultdict(Counter)
    for (lab, split, _, _), n in cells.items():
        by_class_split[lab][split] += n
    patients = defaultdict(set)
    for r in rows:
        if not r["exclude_reason"] and r["patient_id"]:
            patients[r["split"] or "unassigned"].add(r["patient_id"])
    return {
        "manifest_version": MANIFEST_VERSION,
        "schema": SCHEMA,
        "layouts_version": LAYOUTS_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "n_rows": len(rows),
        "n_usable": sum(1 for r in rows if not r["exclude_reason"]),
        "n_soft_label": sum(1 for r in rows if r["soft_label"]),
        "n_eval_exclude": sum(1 for r in rows if as_bool(r["eval_exclude"])),
        "exclude_reason": dict(sorted(Counter(r["exclude_reason"] or "usable" for r in rows).items())),
        "n_patients_by_split": {k: len(v) for k, v in sorted(patients.items())},
        "usable_by_class_split": {k: dict(sorted(by_class_split[k].items())) for k in STATION_ORDER},
        "usable_by_class_split_mode_cohort": [
            {"label": lab, "split": split, "mode": md, "cohort": co, "n": n}
            for (lab, split, md, co), n in sorted(cells.items(), key=lambda kv: (
                STATION_INDEX[kv[0][0]], kv[0][1], kv[0][2], kv[0][3]))
        ],
        "flags": dict(sorted(Counter(f for r in rows for f in str(r["flags"]).split("|")
                                     if f and not f.startswith("manual:")).items())),
    }


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    return read_csv(path)


def write_manifest(rows: list[dict], path: Path = MANIFEST_PATH, summary_path: Path | None = SUMMARY_PATH) -> None:
    rows = sorted(rows, key=lambda r: r["rel_path"])
    bad = [r["rel_path"] for r in rows if is_full_length(r["rel_path"]) or is_full_length(r.get("alt_rel_path", ""))]
    if bad:
        raise RuntimeError(f"full-length rows in manifest: {bad[:3]}")
    write_csv(path, rows, MANIFEST_FIELDS)
    if summary_path is not None:
        write_json(summary_path, summarize(rows))


# --- main ------------------------------------------------------------------------

def build(workers: int = 8, cache_root: Path = CACHE_ROOT) -> tuple[list[dict], dict]:
    clips = iter_clips()
    info = probe_all(clips, workers)
    panels = osd_panels(clips, info, workers)
    ov = load_overrides()
    id_ov = load_id_overrides(OVERRIDES_DIR / "patient_id_overrides.csv")
    ids = resolve_ids(clips, id_ov)
    times = osd_times(clips, info, {rp: p for rp, (p, _) in ids.items()})
    rows = build_rows(clips, info, panels, ov, id_ov, times)
    impute_processor(rows)
    report = {"clips_on_disk": len(clips), "rows": len(rows)}
    report["phase_b"] = apply_cache(rows, cache_root)
    finalize(rows)  # dup_near decided before grouping
    report["group_merges"] = assign_groups(rows)
    chronology(rows)
    for r in rows:
        r["flags"] = "|".join(dict.fromkeys(r["_flags"]))
    report["audit_md5_mismatches"] = _check_audit(rows)
    report["osd_date_outliers"] = _date_outliers(rows)
    return rows, report


def _check_audit(rows: list[dict]) -> int:
    p = AUDIT_DIR / "clips_audit.csv"
    if not p.exists():
        return -1
    audit = {f"{r['cls']}/{r['file']}": r["md5"] for r in drop_full_length(read_csv(p))}
    return sum(1 for r in rows if r["rel_path"] in audit and audit[r["rel_path"]] != r["md5"])


def _date_outliers(rows: list[dict]) -> int:
    """Clips whose OSD date differs from their patient's majority date
    (new cohort; the legacy clock is unreliable). Expected 0 once the
    override table marks the known ones uncertain."""
    days = defaultdict(Counter)
    for r in rows:
        if r["patient_id"] and r["_t"] and r["cohort"] == "new_1920":
            days[r["patient_id"]][r["_t"].date()] += 1
    return sum(1 for r in rows if r["patient_id"] and r["_t"] and r["cohort"] == "new_1920"
               and r["_t"].date() != days[r["patient_id"]].most_common(1)[0][0])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    imp = sub.add_parser("import-audit", help="copy audit tables (REC rows dropped)")
    imp.add_argument("src", type=Path)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = ap.parse_args(argv)
    if args.cmd == "import-audit":
        for k, (n_in, n_out) in import_audit(args.src).items():
            print(f"{k}: {n_in} rows -> {n_out} kept")
        return 0
    rows, report = build(args.workers)
    write_manifest(rows)
    print(f"manifest: {MANIFEST_PATH} ({len(rows)} rows)")
    for k, v in report.items():
        print(f"  {k}: {v}")
    s = summarize(rows)
    print(f"  usable: {s['n_usable']}  exclude_reason: {s['exclude_reason']}")
    print(f"  summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
