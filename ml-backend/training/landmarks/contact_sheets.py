"""Clinician adjudication material (M1a) and read-back of its decisions.

    python -m training.landmarks.contact_sheets            # sheets + templates
    python -m training.landmarks.contact_sheets collect    # completed CSVs -> label_overrides.csv

Writes to data/landmarks_private/adjudication/ (never committed):

    contested_NN.jpg / adjudication_contested.csv
        byte-identical clips filed under both Incisura and Lesser curvature
    suffix_mismatch_NN.jpg / adjudication_suffix_mismatch.csv
        file-name suffix names a different station than the folder
    combo_NN.jpg / adjudication_combo.csv
        "+" names spanning two stations (or an unknown view "x")
    label_audit_NN.jpg / adjudication_label_audit.csv (+ label_audit_key.csv)
        100 random usable clips (10 per station, seed 0) for a BLIND
        2-reader audit: the sheet and template show no label; the key with
        the current labels stays with the coordinator.

Each sheet row is one clip: its first, middle and last cached canonical
frames. Readers fill ``reader1`` / ``reader2`` with a value from ``choices``
and the adjudicator fills ``final``. ``collect`` turns every non-empty
``final`` into a row of overrides/label_overrides.csv (value, reason,
reviewer), which build_manifest applies as label_source=adjudicated; it also
reports reader agreement (Cohen's kappa) for the label audit.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from app.ml.landmarks.stations import STATION_INDEX, STATION_ORDER
from training.landmarks.common import (
    ADJUDICATION_DIR,
    CACHE_ROOT,
    MANIFEST_PATH,
    OVERRIDES_DIR,
    as_bool,
    read_csv,
    write_csv,
)

TILE = 240
PER_PAGE = 10
AUDIT_PER_CLASS = 10
TEMPLATE_FIELDS = ["item", "clip_uid", "sheet", "folder_class", "suffix", "current_label", "choices",
                   "reader1", "reader2", "final", "reviewer", "notes"]
AUDIT_FIELDS = ["item", "clip_uid", "sheet", "choices", "reader1", "reader2", "final", "reviewer", "notes"]


def _frames(cache_root: Path, uid: str) -> list[np.ndarray]:
    files = sorted((cache_root / "frames" / uid).glob("*.jpg"))
    if not files:
        return [np.zeros((TILE, TILE, 3), np.uint8)] * 3
    pick = [files[0], files[len(files) // 2], files[-1]]
    return [cv2.resize(cv2.imread(str(f)), (TILE, TILE), interpolation=cv2.INTER_AREA) for f in pick]


def _row_image(cache_root: Path, item: int, r: dict, caption: str) -> np.ndarray:
    tiles = np.hstack(_frames(cache_root, r["clip_uid"]))
    lines = f"#{item:03d}  {caption}".split("\n")
    band = np.zeros((12 + 22 * len(lines), tiles.shape[1], 3), np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(band, line, (8, 23 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([band, tiles, np.zeros((6, tiles.shape[1], 3), np.uint8)])


def write_sheets(name: str, items: list[tuple[int, dict, str]], out_dir: Path, cache_root: Path) -> list[str]:
    """Pages of PER_PAGE clips; returns the sheet file name of each item."""
    sheet_of = []
    for p in range(0, len(items), PER_PAGE):
        page = items[p:p + PER_PAGE]
        fname = f"{name}_{p // PER_PAGE + 1:02d}.jpg"
        img = np.vstack([_row_image(cache_root, i, r, cap) for i, r, cap in page])
        cv2.imwrite(str(out_dir / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        sheet_of += [fname] * len(page)
    return sheet_of


def _choices(options: list[str]) -> str:
    return " ; ".join(dict.fromkeys(options + ["exclude"]))


def build_sets(rows: list[dict], seed: int = 0) -> dict[str, list[tuple[dict, str]]]:
    """Clip sets and their answer choices (see module docstring)."""
    live = [r for r in rows if r["exclude_reason"] not in ("dup_exact", "too_short")]
    contested = [(r, _choices(["lesser_curvature_retroflex", "incisura",
                               "lesser_curvature_retroflex:0.5|incisura:0.5"]))
                 for r in live if "contested" in r["flags"].split("|")]
    mismatch = [(r, _choices([r["folder_class"], r["suffix_class"]]))
                for r in live if r["exclude_reason"] == "quarantine_suffix_mismatch"]
    combo = []
    for r in live:
        if r["exclude_reason"] == "combo":
            from training.landmarks.ids import parse_suffix, SUFFIX_TO_STATION

            parts, _, _ = parse_suffix(r["suffix"])
            st = [SUFFIX_TO_STATION.get(p) for p in parts if SUFFIX_TO_STATION.get(p)]
            opts = st + ([f"{st[0]}:0.5|{st[1]}:0.5"] if len(st) == 2 else [])
            combo.append((r, _choices(opts or [r["folder_class"]])))
    rng = np.random.default_rng(seed)
    pool = [r for r in rows if not r["exclude_reason"] and not as_bool(r["eval_exclude"])
            and r["cohort"] == "new_1920"]
    audit = []
    for k in STATION_ORDER:
        cand = sorted((r for r in pool if r["label"] == k), key=lambda r: r["clip_uid"])
        take = rng.choice(len(cand), size=min(AUDIT_PER_CLASS, len(cand)), replace=False)
        audit += [cand[i] for i in sorted(take)]
    audit = [audit[i] for i in rng.permutation(len(audit))]  # blind: shuffle station order
    audit_choices = _choices(list(STATION_ORDER) + ["other_or_unclear"])
    return {"contested": contested, "suffix_mismatch": mismatch, "combo": combo,
            "label_audit": [(r, audit_choices) for r in audit]}


def generate(rows: list[dict], out_dir: Path = ADJUDICATION_DIR, cache_root: Path = CACHE_ROOT) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, items in build_sets(rows).items():
        blind = name == "label_audit"
        numbered = [(i + 1, r, (r["clip_uid"] if blind else
                                f"{r['clip_uid']}  folder={r['folder_class']}  suffix={r['suffix']}\n"
                                f"label={r['soft_label'] or r['label']}"))
                    for i, (r, _) in enumerate(items)]
        sheets = write_sheets(name, numbered, out_dir, cache_root) if items else []
        recs = []
        for (i, r, _), (_, choices), sheet in zip(numbered, items, sheets):
            rec = {"item": i, "clip_uid": r["clip_uid"], "sheet": sheet, "choices": choices}
            if not blind:
                rec.update(folder_class=r["folder_class"], suffix=r["suffix"],
                           current_label=r["soft_label"] or r["label"])
            recs.append(rec)
        path = out_dir / f"adjudication_{name}.csv"
        if path.exists() and any(x.get("reader1") or x.get("final") for x in read_csv(path)):
            print(f"  {path.name}: has reader input, not overwritten")
        else:
            write_csv(path, recs, AUDIT_FIELDS if blind else TEMPLATE_FIELDS)
        if blind:
            write_csv(out_dir / "label_audit_key.csv",
                      [{"item": i, "clip_uid": r["clip_uid"], "current_label": r["label"]} for i, r, _ in numbered])
        counts[name] = len(items)
    return counts


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _valid(value: str) -> bool:
    v = value.strip()
    if v in ("exclude", "keep") or v in STATION_INDEX:
        return True
    parts = [p.split(":") for p in v.split("|")]
    return all(len(p) == 2 and p[0] in STATION_INDEX for p in parts) and \
        abs(sum(float(p[1]) for p in parts) - 1) < 1e-6


def collect(out_dir: Path = ADJUDICATION_DIR, overrides_path: Path = OVERRIDES_DIR / "label_overrides.csv") -> dict:
    """Merge every non-empty ``final`` into label_overrides.csv; kappa for the audit."""
    existing = {r["clip_uid"]: r for r in read_csv(overrides_path)} if overrides_path.exists() else {}
    added, bad = 0, []
    for path in sorted(out_dir.glob("adjudication_*.csv")):
        for r in read_csv(path):
            final = (r.get("final") or "").strip()
            if not final:
                continue
            if not _valid(final):
                bad.append(f"{path.name}#{r['item']}: {final!r}")
                continue
            existing[r["clip_uid"]] = {"clip_uid": r["clip_uid"], "label": final,
                                       "reason": f"adjudication ({path.stem.removeprefix('adjudication_')}); "
                                                 f"readers: {r.get('reader1', '')} / {r.get('reader2', '')}",
                                       "reviewer": r.get("reviewer", "")}
            added += 1
    write_csv(overrides_path, list(existing.values()), ["clip_uid", "label", "reason", "reviewer"])
    res = {"overrides_written": added, "invalid": bad}
    audit = out_dir / "adjudication_label_audit.csv"
    key = out_dir / "label_audit_key.csv"
    if audit.exists() and key.exists():
        rs = [r for r in read_csv(audit) if r.get("reader1") and r.get("reader2")]
        truth = {r["clip_uid"]: r["current_label"] for r in read_csv(key)}
        if rs:
            r1, r2 = [r["reader1"].strip() for r in rs], [r["reader2"].strip() for r in rs]
            res["label_audit"] = {
                "n": len(rs),
                "kappa_reader1_reader2": round(cohen_kappa(r1, r2), 3),
                "agree_reader1_current": round(sum(a == truth[r["clip_uid"]] for a, r in zip(r1, rs)) / len(rs), 3),
                "agree_reader2_current": round(sum(b == truth[r["clip_uid"]] for b, r in zip(r2, rs)) / len(rs), 3),
            }
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["generate", "collect"], default="generate")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        res = collect()
        print(res)
        print("re-run build_manifest (then folds) to apply the overrides")
        return 1 if res["invalid"] else 0
    counts = generate(read_csv(MANIFEST_PATH))
    for k, n in counts.items():
        print(f"  {k}: {n} clips")
    print(f"  -> {ADJUDICATION_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
