"""Deterministic patient-grouped splits: a locked TEST set and 5 DEV folds.

    python -m training.landmarks.folds            # assign split/fold in the manifest
    python -m training.landmarks.folds --verify   # integrity checks only (non-zero on failure)

Unit of splitting = ``fold_group``: the manifest ``group_id`` (patient, merged
across a (date, processor) window) further merged with every clip's
``near_dup_group``, so no near-duplicate content can straddle a split.

1. TEST (~18 new-cohort patients, touched once in M4): StratifiedGroupKFold(5)
   over usable new-cohort clips (y = label, groups = fold_group), seeds
   0..499 x folds 0..4 in order; the first fold that has
   >= 15 primary-eval clips per class, >= 5 patients with all 10 stations,
   and an NBI share within +-7 pp of the rest in every station family.
2. DEV: StratifiedGroupKFold(5) over the remaining patients; seed 0..499
   minimising the largest per-fold, per-class relative deviation from the
   class mean, subject to >= 8 primary-eval validation clips per class.

split: dev | test | shift (legacy cohort) | excluded (no safe patient group);
fold: 0..4 for dev, -1 otherwise. Combo / quarantined clips follow their
patient's split (they have an exclude_reason, so they are never trained on).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict

import numpy as np

from app.ml.landmarks.stations import STATION_ORDER, STATION_REGION
from training.landmarks.build_manifest import NO_GROUP_REASONS, _DSU, write_manifest
from training.landmarks.common import MANIFEST_PATH, SUMMARY_PATH, as_bool, is_full_length, read_csv

N_FOLDS = 5
SEEDS = range(500)
MIN_TEST_PER_CLASS = 15
MIN_TEST_ALL10 = 5
NBI_TOL = 0.07
MIN_VAL_PER_CLASS = 8
WINDOW_PAD_S = 120  # must match build_manifest.WINDOW_PAD


def fold_groups(rows: list[dict]) -> dict[str, str]:
    """clip_uid -> fold_group (group_id merged through near_dup_group)."""
    dsu = _DSU()
    for r in rows:
        if r["group_id"]:
            dsu.find("g:" + r["group_id"])
            if r["near_dup_group"]:
                dsu.union("g:" + r["group_id"], "n:" + r["near_dup_group"])
    return {r["clip_uid"]: dsu.find("g:" + r["group_id"]) for r in rows if r["group_id"]}


def _stations(r: dict) -> set[str]:
    if r["soft_label"]:
        return {x.split(":")[0] for x in r["soft_label"].split("|")}
    return {r["label"]}


def _is_nbi(r: dict) -> bool:
    return r["nbi_frac"] not in ("", None) and float(r["nbi_frac"]) >= 0.5


def _pool(rows: list[dict]) -> list[dict]:
    """Rows that take part in the TEST/DEV assignment: usable new-cohort clips."""
    return [r for r in rows if r["cohort"] == "new_1920" and not r["exclude_reason"] and r["group_id"]]


def _stats(rows: list[dict]) -> dict:
    by_cls = Counter(r["label"] for r in rows if not as_bool(r["eval_exclude"]))
    by_patient = defaultdict(set)
    for r in rows:
        by_patient[r["patient_id"]] |= _stations(r)
    fam = defaultdict(lambda: [0, 0])
    for r in rows:
        f = fam[STATION_REGION[r["label"]]]
        f[0] += _is_nbi(r)
        f[1] += 1
    return {"per_class": by_cls, "all10": sum(len(s) == len(STATION_ORDER) for s in by_patient.values()),
            "nbi_share": {k: v[0] / v[1] for k, v in fam.items() if v[1]},
            "patients": len(by_patient)}


def _test_ok(te: dict, tr: dict) -> bool:
    if any(te["per_class"].get(k, 0) < MIN_TEST_PER_CLASS for k in STATION_ORDER):
        return False
    if te["all10"] < MIN_TEST_ALL10:
        return False
    return all(abs(te["nbi_share"].get(f, 0) - tr["nbi_share"].get(f, 0)) <= NBI_TOL for f in tr["nbi_share"])


def choose_test(pool: list[dict], groups: dict[str, str], seeds=SEEDS) -> tuple[int, int, set[str]]:
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.array([r["label"] for r in pool])
    g = np.array([groups[r["clip_uid"]] for r in pool])
    for seed in seeds:
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        for k, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
            if _test_ok(_stats([pool[i] for i in te]), _stats([pool[i] for i in tr])):
                return seed, k, set(g[te])
    raise RuntimeError("no seed satisfies the TEST constraints")


def choose_dev_folds(dev: list[dict], groups: dict[str, str], seeds=SEEDS) -> tuple[int, dict[str, int], float]:
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.array([r["label"] for r in dev])
    g = np.array([groups[r["clip_uid"]] for r in dev])
    ev = np.array([not as_bool(r["eval_exclude"]) for r in dev])
    best = None
    for seed in seeds:
        sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
        assign = {}
        counts = np.zeros((N_FOLDS, len(STATION_ORDER)))
        for k, (_, va) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
            for i in va:
                assign[g[i]] = k
                if ev[i]:
                    counts[k, STATION_ORDER.index(y[i])] += 1
        if counts.min() < MIN_VAL_PER_CLASS:
            continue
        mean = counts.mean(axis=0)
        dev_score = float((np.abs(counts - mean) / mean).max())
        if best is None or dev_score < best[2]:
            best = (seed, assign, dev_score)
    if best is None:
        raise RuntimeError("no seed satisfies the DEV fold constraints")
    return best


def assign_splits(rows: list[dict], seeds=SEEDS) -> dict:
    """Write split/fold into ``rows`` in place; returns what was chosen."""
    groups = fold_groups(rows)
    pool = _pool(rows)
    t_seed, t_fold, test_groups = choose_test(pool, groups, seeds)
    dev = [r for r in pool if groups[r["clip_uid"]] not in test_groups]
    d_seed, dev_assign, d_score = choose_dev_folds(dev, groups, seeds)
    unplaced = 0
    for r in rows:
        r["fold"] = -1
        if r["exclude_reason"] in NO_GROUP_REASONS or not r["group_id"]:
            r["split"] = "excluded"
        elif r["cohort"] == "legacy_1350":
            r["split"] = "shift"
        elif groups[r["clip_uid"]] in test_groups:
            r["split"] = "test"
        else:
            r["split"] = "dev"
            r["fold"] = dev_assign.get(groups[r["clip_uid"]], -1)
            unplaced += r["fold"] == -1
    return {"test_seed": t_seed, "test_fold": t_fold, "dev_seed": d_seed, "dev_max_rel_dev": round(d_score, 3),
            "test_groups": len(test_groups), "dev_rows_without_fold": unplaced}


# --- verification ---------------------------------------------------------------------

def verify(rows: list[dict]) -> list[str]:
    """Split integrity. Returns a list of violations (empty = OK)."""
    errs: list[str] = []
    for r in rows:
        if is_full_length(r["rel_path"]) or is_full_length(r.get("alt_rel_path", "")):
            errs.append(f"full-length row {r['clip_uid']}")
    uids = Counter(r["clip_uid"] for r in rows)
    errs += [f"duplicate clip_uid {u}" for u, n in uids.items() if n > 1]
    live = [r for r in rows if r["split"] != "excluded"]
    for r in rows:
        if r["split"] not in ("dev", "test", "shift", "excluded"):
            errs.append(f"{r['clip_uid']}: bad split {r['split']!r}")
        if r["split"] == "excluded" and r["exclude_reason"] not in NO_GROUP_REASONS and r["group_id"]:
            errs.append(f"{r['clip_uid']}: split=excluded without a no-group reason")
        if r["split"] in ("dev", "test") and r["cohort"] != "new_1920":
            errs.append(f"{r['clip_uid']}: {r['cohort']} clip in {r['split']}")
        if r["split"] == "shift" and r["cohort"] != "legacy_1350":
            errs.append(f"{r['clip_uid']}: new-cohort clip in shift")
        if r["split"] == "dev" and not (0 <= int(r["fold"]) < N_FOLDS):
            errs.append(f"{r['clip_uid']}: dev clip without fold")
        if r["split"] != "dev" and int(r["fold"]) != -1:
            errs.append(f"{r['clip_uid']}: fold on a non-dev clip")

    def crosses(key: str, rs: list[dict], field: str = "split") -> None:
        seen = defaultdict(set)
        for r in rs:
            if r[key]:
                seen[r[key]].add(r[field])
        errs.extend(f"{key} {k} spans {field}s {sorted(v)}" for k, v in seen.items() if len(v) > 1)

    for key in ("group_id", "patient_id", "md5", "near_dup_group"):
        crosses(key, live)
    dev = [r for r in live if r["split"] == "dev"]
    for key in ("group_id", "patient_id", "near_dup_group"):
        crosses(key, dev, "fold")
    errs += _room_window_violations(live)
    return errs


def _room_window_violations(rows: list[dict]) -> list[str]:
    """A clip must not fall inside another split's patient's (day, room) window."""
    win: dict[tuple, list] = defaultdict(list)
    spans: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["tod_s"] not in ("", None) and r["day_key"] and r["room_hash"] and r["patient_id"]:
            t = float(r["tod_s"])
            spans[(r["day_key"], r["room_hash"], r["patient_id"], r["split"])] += [t, t + float(r["dur_s"])]
    for (day, room, p, split), ts in spans.items():
        win[(day, room)].append((p, split, min(ts) - WINDOW_PAD_S, max(ts) + WINDOW_PAD_S))
    out = []
    for r in rows:
        if r["tod_s"] in ("", None) or not r["day_key"]:
            continue
        t = float(r["tod_s"])
        for q, split, a, b in win.get((r["day_key"], r["room_hash"]), ()):
            if q != r["patient_id"] and split != r["split"] and a <= t <= b:
                out.append(f"{r['clip_uid']} ({r['split']}) inside the room window of a {split} patient")
    return out


def report(rows: list[dict]) -> str:
    lines = []
    splits = ("dev", "test", "shift", "excluded")
    usable = [r for r in rows if not r["exclude_reason"]]
    lines.append(f"{'class':28s}" + "".join(f"{s:>9s}" for s in splits) + "   dev folds 0..4 (eval clips)")
    for k in STATION_ORDER:
        c = Counter(r["split"] for r in usable if r["label"] == k)
        folds = Counter(int(r["fold"]) for r in usable if r["label"] == k and r["split"] == "dev"
                        and not as_bool(r["eval_exclude"]))
        lines.append(f"{k:28s}" + "".join(f"{c.get(s, 0):9d}" for s in splits)
                     + "   " + " ".join(f"{folds.get(f, 0):3d}" for f in range(N_FOLDS)))
    pts = {s: len({r["patient_id"] for r in usable if r["split"] == s}) for s in splits}
    lines.append(f"{'patients':28s}" + "".join(f"{pts[s]:9d}" for s in splits))
    lines.append(f"{'clips (usable)':28s}" + "".join(f"{sum(r['split'] == s for r in usable):9d}" for s in splits))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true", help="check only; exit 1 on any violation")
    args = ap.parse_args(argv)
    rows = read_csv(MANIFEST_PATH)
    if not args.verify:
        if any(r["cohort"] == "new_1920" and not r["exclude_reason"] and r["nbi_frac"] == "" for r in rows):
            print("manifest has no phase-B measurements: run extract_cache, then build_manifest again",
                  file=sys.stderr)
            return 2
        chosen = assign_splits(rows)
        write_manifest(rows, MANIFEST_PATH, SUMMARY_PATH)
        print("chosen:", chosen)
    errs = verify(rows)
    print(report(rows))
    if errs:
        print(f"VERIFY FAILED ({len(errs)}):", *errs[:30], sep="\n  ", file=sys.stderr)
        return 1
    print("verify: OK (no group / patient / md5 / near-dup group / room window crosses a split)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
