"""Landmark data pipeline (training/landmarks): IDs, manifest policy, splits.

All inputs are synthetic: invented HT numbers, generated rows and images.
Nothing here reads data/ (the dataset is not in the repo or in CI).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.ml.landmarks.stations import STATION_ORDER
from training.landmarks.common import ClipFile, drop_full_length, is_full_length, iter_clips
from training.landmarks.ids import (
    SUFFIX_TO_STATION,
    IdOverrides,
    normalize_patient_id,
    parse_clip_name,
    resolve_patient_id,
)

# --- ids -----------------------------------------------------------------------------------

OVR = IdOverrides.from_rows([
    {"match_type": "raw_id", "match": "2420", "patient_id": "HT242", "reason": "typo", "reviewer": "t"},
    {"match_type": "file", "match": "HT27-eso2.mp4", "patient_id": "HT271", "reason": "OSD", "reviewer": "t"},
    {"match_type": "file", "match": "HT-gc+a.mp4", "patient_id": "HT277", "reason": "OSD", "reviewer": "t"},
    {"match_type": "file", "match": "HT-a+gc.mp4", "patient_id": "uncertain", "reason": "?", "reviewer": "t"},
    {"match_type": "file", "match": "HT242-gc (3).mp4", "patient_id": "uncertain", "reason": "date", "reviewer": "t"},
])


@pytest.mark.parametrize("name, pid, how", [
    ("HT242-a.mp4", "HT242", "regex"),
    ("HT2O4-inc.mp4", "HT204", "regex_letter_o"),     # letter O for zero
    ("HT42-bd.mp4", "HT042", "regex_padded"),          # two digits
    ("HT0242-a.mp4", None, "ambiguous"),               # four digits: never guessed
    ("HT2420-eso1.mp4", "HT242", "override"),          # four digits, raw-ID override
    ("HT242eso2.mp4", "HT242", "regex"),               # no separator
    ("HT242a-2.mp4", "HT242", "regex"),                # take after hyphen
    ("HT-242-bd.mp4", "HT242", "regex"),               # separator before digits
    ("HT242- bd.mp4", "HT242", "regex"),               # stray space
    ("HT242-eso1(10.mp4", "HT242", "regex"),           # truncated copy marker
    ("HT27-eso2.mp4", "HT271", "override"),            # file override beats padding (HT027)
    ("HT-gc+a.mp4", "HT277", "override"),             # no ID, resolved by OSD
    ("HT-a+gc.mp4", None, "uncertain"),               # no ID, unresolvable
    ("HT242-gc (3).mp4", None, "uncertain"),           # OSD date contradicts the ID
    ("HT-x.mp4", None, "unparsed"),
])
def test_patient_id_dirty_forms(name, pid, how):
    assert resolve_patient_id(parse_clip_name(name), OVR) == (pid, how)


def test_generic_rule_without_overrides():
    assert normalize_patient_id("2O4") == ("HT204", "regex_letter_o")
    assert normalize_patient_id("7") == ("HT007", "regex_padded")
    assert normalize_patient_id("0750") == (None, "ambiguous")
    assert normalize_patient_id(None) == (None, "unparsed")
    assert resolve_patient_id(parse_clip_name("HT27-eso2.mp4")) == ("HT027", "regex_padded")


def test_bad_override_values_are_rejected():
    with pytest.raises(ValueError):
        IdOverrides.from_rows([{"match_type": "file", "match": "a.mp4", "patient_id": "HT42"}])
    with pytest.raises(ValueError):
        IdOverrides.from_rows([{"match_type": "nope", "match": "a.mp4", "patient_id": "HT242"}])


@pytest.mark.parametrize("name, suffix, cls, take, copies", [
    ("HT242-a.mp4", "a", "antrum", None, ()),
    ("HT242-a1.mp4", "a", "antrum", 1, ()),
    ("HT242a-2.mp4", "a", "antrum", 2, ()),
    ("HT242-cfi.mp4", "cfi", "cardia_fundus_retroflex", None, ()),
    ("HT242-cf1.mp4", "cfi", "cardia_fundus_retroflex", 1, ()),
    ("HT242-dd(1).mp4", "dd", "duodenum_descending", None, (1,)),
    ("HT242-dd (2)(1).mp4", "dd", "duodenum_descending", None, (2, 1)),
    ("HT242-eso1.mp4", "eso1", "esophagus_proximal", None, ()),
    ("HT242eso11.mp4", "eso1", "esophagus_proximal", 1, ()),
    ("HT242-eso1(10.mp4", "eso1", "esophagus_proximal", None, (10,)),
    ("HT242-eso2 (3).mp4", "eso2", "esophagus_distal", None, (3,)),
    ("HT242- bd.mp4", "bd", "duodenal_bulb", None, ()),
    ("HT242-gc3 (1).mp4", "gc", "corpus_greater_curvature", 3, (1,)),
    ("HT242-inc.mp4", "inc", "incisura", None, ()),
    ("HT242-lc.mp4", "lc", "lesser_curvature_retroflex", None, ()),
    ("HT242-zline.mp4", "zline", "z_line", None, ()),
    ("HT242-z-line.mp4", "zline", "z_line", None, ()),
])
def test_suffix_parsing(name, suffix, cls, take, copies):
    p = parse_clip_name(name)
    assert (p.suffix, p.suffix_class, p.take, p.copies) == (suffix, cls, take, copies)
    assert not p.is_combo


@pytest.mark.parametrize("name, parts, stations", [
    ("HT242-lc+inc.mp4", ("lc", "inc"), ("lesser_curvature_retroflex", "incisura")),
    ("HT242-lc+in.mp4", ("lc", "inc"), ("lesser_curvature_retroflex", "incisura")),
    ("HT242-dd+b.mp4", ("dd", "bd"), ("duodenum_descending", "duodenal_bulb")),
    ("HT-gc+a.mp4", ("gc", "a"), ("corpus_greater_curvature", "antrum")),
    ("HT242-cfi+x.mp4", ("cfi", "x"), ("cardia_fundus_retroflex", None)),
    ("HT242-dd-x.mp4", ("dd", "x"), ("duodenum_descending", None)),
])
def test_combo_suffixes(name, parts, stations):
    p = parse_clip_name(name)
    assert p.is_combo and p.suffix_class == ""
    assert p.suffix_parts == parts and p.suffix_stations == stations
    assert p.suffix == "+".join(parts)


def test_unknown_suffix_has_no_class():
    p = parse_clip_name("HT242-eso.mp4")
    assert p.suffix_class == "" and p.unknown_tokens == ("eso",) and not p.is_combo


def test_every_suffix_token_maps_to_a_station():
    assert {v for v in SUFFIX_TO_STATION.values() if v} == set(STATION_ORDER)


# --- full-length videos never enter ---------------------------------------------------

def test_full_length_paths_are_recognised():
    assert is_full_length("Full lenght videos/REC_0001.mp4")
    assert is_full_length("data/landmarks/Full length videos/x.mp4")
    assert is_full_length("REC_0007.mp4")
    assert not is_full_length("Antrum/HT242-a.mp4")
    rows = [{"cls": "Full lenght videos", "file": "REC_0001.mp4"}, {"cls": "Antrum", "file": "HT242-a.mp4"},
            {"rel_path": "Full lenght videos/REC_0002.mp4"}]
    assert drop_full_length(rows) == [{"cls": "Antrum", "file": "HT242-a.mp4"}]


def test_iter_clips_skips_full_length_and_unknown_folders(tmp_path):
    for folder, name in [("Antrum", "HT242-a.mp4"), ("Full lenght videos", "REC_0001.mp4"),
                         ("Full lenght videos", "HT242-a.mp4"), ("Misc", "HT242-a.mp4"), ("Z-line", "notes.txt")]:
        (tmp_path / folder).mkdir(exist_ok=True)
        (tmp_path / folder / name).write_bytes(b"x")
    clips = iter_clips(tmp_path)
    assert [c.rel_path for c in clips] == ["Antrum/HT242-a.mp4"]


def test_manifest_writer_and_cache_selection_reject_full_length(tmp_path):
    from training.landmarks.build_manifest import write_manifest
    from training.landmarks.extract_cache import select_rows

    row = _row("aa", "HT242", "antrum", rel_path="Full lenght videos/REC_0001.mp4")
    with pytest.raises(RuntimeError):
        write_manifest([row], tmp_path / "m.csv", None)
    assert select_rows([row]) == []


# --- manifest policy -------------------------------------------------------------------

def _info(md5: str, n_frames: int = 150, width: int = 1920) -> dict:
    return {"md5": md5 * 32 if len(md5) == 1 else md5.ljust(32, "0"), "width": width,
            "height": 1080, "n_frames": n_frames, "fps": 30.0, "dur_s": n_frames / 30}


def _clip(folder: str, file: str) -> ClipFile:
    return ClipFile(f"{folder}/{file}", folder, file)


def test_build_rows_applies_the_exclusion_policy():
    from training.landmarks.build_manifest import Overrides, build_rows

    clips = [
        _clip("Incisura angularis", "HT242-inc.mp4"), _clip("Lesser curvature", "HT242-inc.mp4"),  # identical
        _clip("Descending duodenum", "HT242-dd.mp4"), _clip("Descending duodenum", "HT242-dd(1).mp4"),  # dup
        _clip("Antrum", "HT242-a (2).mp4"),            # too short
        _clip("Duodenal bulb", "HT242-dd+b.mp4"),      # combo
        _clip("Duodenal bulb", "HT242-dd.mp4"),        # suffix != folder
        _clip("Gastric corpus", "HT242-gc (3).mp4"),   # id uncertain
        _clip("Z-line", "HT243-zline.mp4"),            # fine
        _clip("Antrum", "HT904-a.mp4"),                # legacy cohort
    ]
    info = {c.rel_path: _info(m) for c, m in zip(clips, "11223456789")}
    info["Antrum/HT242-a (2).mp4"]["n_frames"] = 2
    info["Antrum/HT904-a.mp4"].update(width=1350)
    rows = build_rows(clips, info, {}, Overrides({}, {}), OVR)
    by = {r["rel_path"]: r for r in rows}
    assert len(rows) == len(clips) - 1  # identical cross-folder pair -> one row
    c = by["Incisura angularis/HT242-inc.mp4"]
    assert c["alt_rel_path"] == "Lesser curvature/HT242-inc.mp4"
    assert c["soft_label"] == "lesser_curvature_retroflex:0.5|incisura:0.5"
    assert c["exclude_reason"] == "" and c["eval_exclude"] is True
    assert by["Descending duodenum/HT242-dd.mp4"]["exclude_reason"] == ""
    dup = by["Descending duodenum/HT242-dd(1).mp4"]
    assert dup["exclude_reason"] == "dup_exact" and dup["clip_uid"].endswith("-dup1")
    assert by["Antrum/HT242-a (2).mp4"]["exclude_reason"] == "too_short"
    assert by["Duodenal bulb/HT242-dd+b.mp4"]["exclude_reason"] == "combo"
    assert by["Duodenal bulb/HT242-dd.mp4"]["exclude_reason"] == "quarantine_suffix_mismatch"
    assert by["Gastric corpus/HT242-gc (3).mp4"]["exclude_reason"] == "id_uncertain"
    assert by["Z-line/HT243-zline.mp4"]["label"] == "z_line"
    assert by["Antrum/HT904-a.mp4"]["cohort"] == "legacy_1350"
    assert len({r["clip_uid"] for r in rows}) == len(rows)


def test_label_overrides_adjudicate_quarantined_and_contested_clips():
    from training.landmarks.build_manifest import Overrides, build_rows

    clips = [_clip("Duodenal bulb", "HT242-dd.mp4"), _clip("Incisura angularis", "HT242-inc.mp4"),
             _clip("Lesser curvature", "HT242-inc.mp4")]
    info = {clips[0].rel_path: _info("1"), clips[1].rel_path: _info("2"), clips[2].rel_path: _info("2")}
    ov = Overrides({"1" * 12: {"label": "duodenum_descending"}, "2" * 12: {"label": "incisura"}},
                   {"Z-line/none.mp4": {"exclude_reason": "x"}})
    rows = {r["clip_uid"]: r for r in build_rows(clips, info, {}, ov, OVR)}
    q, c = rows["1" * 12], rows["2" * 12]
    assert q["label"] == "duodenum_descending" and q["label_source"] == "adjudicated" and not q["exclude_reason"]
    assert c["label"] == "incisura" and c["soft_label"] == "" and c["eval_exclude"] is False


def _timed(pid: str, t: datetime, proc: str, dur: float = 5.0, label: str = "antrum", uid: str = "") -> dict:
    return {"clip_uid": uid or f"{pid}-{t:%H%M%S}", "patient_id": pid, "_t": t, "_t_firm": True,
            "_processor": proc, "dur_s": dur, "exclude_reason": "", "label": label, "soft_label": "",
            "rel_path": f"x/{pid}-{t:%H%M%S}.mp4", "_flags": []}


def test_groups_merge_only_inside_a_same_processor_window():
    from training.landmarks.build_manifest import assign_groups

    t = datetime(2025, 1, 1, 10, 0, 0)
    rows = [
        _timed("HT301", t, "A"), _timed("HT301", t + timedelta(minutes=10), "A"),
        # Other room, same time: concurrent procedure, never merged.
        _timed("HT302", t + timedelta(minutes=5), "B"),
        # Same room, 25 min later: the next patient, not merged.
        _timed("HT303", t + timedelta(minutes=35), "A"),
        # Same room, inside HT301's window: a mis-filed clip -> merged.
        _timed("HT304", t + timedelta(minutes=6), "A"), _timed("HT304", t + timedelta(minutes=8), "A"),
    ]
    merges = assign_groups(rows)
    g = {r["patient_id"]: r["group_id"] for r in rows}
    assert merges == [("HT301", "HT304")]
    assert g["HT301"] == g["HT304"] == "HT301"
    assert g["HT302"] == "HT302" and g["HT303"] == "HT303"


def test_chronology_records_cross_label_overlaps():
    from training.landmarks.build_manifest import chronology

    t = datetime(2025, 1, 1, 10, 0, 0)
    a = _timed("HT301", t, "A", 5.0, "lesser_curvature_retroflex", "a")
    b = _timed("HT301", t + timedelta(seconds=4), "A", 5.0, "incisura", "b")
    c = _timed("HT301", t + timedelta(seconds=60), "A", 5.0, "incisura", "c")
    chronology([a, b, c])
    assert (a["overlap_head_s"], a["overlap_tail_s"], a["overlap_with"]) == (0.0, 1.0, "b")
    assert (b["overlap_head_s"], b["overlap_tail_s"], b["overlap_with"]) == (1.0, 0.0, "a")
    assert c["overlap_with"] == "" and c["rel_t_s"] == 60.0 and c["order_in_patient"] == 2


def test_osd_panel_fingerprint():
    from training.landmarks.build_manifest import parse_panel

    a = parse_panel("E)] Komentarz\n>. GIF-EZ1500\n[sN) 1234567\n@ Zatrzymaj\n@ Oswietlenie obserwacyjne\n")
    b = parse_panel("[=] Clinic X\nfs. GIF-HQ190\n[sx] 7654321\n@ Zatrzymaj\n@ NBI\n© Pompa pod. wode\n")
    assert (a["processor"], a["model"], a["serial"]) == ("A", "EZ1500", "1234567")
    assert (b["processor"], b["model"], b["serial"]) == ("B", "HQ190", "7654321")
    assert parse_panel("Komentarz\n@ NBI\n")["processor"] == ""  # contradictory cues -> unknown
    assert parse_panel("[=] Clinic X\n")["processor"] == ""  # no decisive cue


# --- summary: counts only ---------------------------------------------------------------

def _row(uid: str, pid: str, label: str, split: str = "", **kw) -> dict:
    r = {"clip_uid": uid, "rel_path": f"Antrum/{pid}-{uid}.mp4", "alt_rel_path": "", "label": label,
         "soft_label": "", "exclude_reason": "", "eval_exclude": False, "flags": "", "patient_id": pid,
         "group_id": pid, "near_dup_group": "", "md5": (uid * 32)[:32], "cohort": "new_1920",
         "layout": "olympus_1920x1080", "split": split, "fold": -1, "mode_major": "wl", "nbi_frac": 0.0,
         "day_key": "d1", "room_hash": "r1", "tod_s": "", "dur_s": 5.0}
    r.update(kw)
    return r


def test_summary_has_counts_only():
    from training.landmarks.build_manifest import summarize

    rows = [_row(f"{i:012x}", f"HT{400 + i % 7:03d}", STATION_ORDER[i % 10], "dev", fold=i % 5,
                 tod_s=36000 + i) for i in range(70)]
    rows[0].update(exclude_reason="combo", flags="combo")
    text = json.dumps(summarize(rows))
    assert not re.search(r"HT\d", text)
    assert not re.search(r"[0-9a-f]{10,}", text)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert ".mp4" not in text and "/" not in text.replace("1/", "")
    s = json.loads(text)
    assert s["n_rows"] == 70 and s["n_usable"] == 69


def test_committed_summary_has_counts_only():
    from training.landmarks.common import SUMMARY_PATH

    if not SUMMARY_PATH.exists():
        pytest.skip("summary not generated")
    text = SUMMARY_PATH.read_text()
    assert not re.search(r"HT\d|[0-9a-f]{10,}|\d{4}-\d{2}-\d{2}|\.mp4|REC_", text)


# --- folds -------------------------------------------------------------------------------

def _synthetic_manifest(n_patients: int = 40, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"HT{p + 301:03d}"
        for k, label in enumerate(STATION_ORDER):
            for j in range(2):
                uid = f"{p:04x}{k:04x}{j:04x}"
                nbi = float(rng.random() < (0.8 if k < 3 else 0.1))
                rows.append(_row(uid, pid, label, nbi_frac=nbi, mode_major="nbi" if nbi else "wl",
                                 day_key=f"d{p}", room_hash="r1", tod_s=36000 + 60 * (10 * k + j)))
    for i in range(3):  # a legacy patient and an excluded duplicate
        rows.append(_row(f"feed{i:08x}", "HT900", STATION_ORDER[i], cohort="legacy_1350"))
    rows.append(_row("dead00000000", "HT301", "antrum", exclude_reason="dup_exact"))
    return rows


def test_folds_assign_and_verify():
    pytest.importorskip("sklearn")
    from training.landmarks.folds import assign_splits, verify

    rows = _synthetic_manifest()
    chosen = assign_splits(rows, seeds=range(5))
    assert verify(rows) == []
    splits = {r["split"] for r in rows}
    assert splits == {"dev", "test", "shift", "excluded"}
    assert chosen["dev_rows_without_fold"] == 0
    test_patients = {r["patient_id"] for r in rows if r["split"] == "test"}
    assert 5 <= len(test_patients) <= 12
    assert {int(r["fold"]) for r in rows if r["split"] == "dev"} == set(range(5))
    # Deterministic.
    again = _synthetic_manifest()
    assign_splits(again, seeds=range(5))
    assert [r["split"] for r in rows] == [r["split"] for r in again]


def test_verify_catches_planted_cross_split_leaks():
    pytest.importorskip("sklearn")
    from training.landmarks.folds import assign_splits, verify

    rows = _synthetic_manifest()
    assign_splits(rows, seeds=range(5))
    test_row = next(r for r in rows if r["split"] == "test")
    dev_row = next(r for r in rows if r["split"] == "dev")

    planted = [dict(r) for r in rows]
    planted.append({**test_row, "clip_uid": "planted00001", "split": "dev", "fold": 0})
    assert any("group_id" in e and "spans splits" in e for e in verify(planted))

    planted = [dict(r) for r in rows] + [{**_row("planted00002", "HT999", "antrum", "dev", fold=0),
                                           "md5": test_row["md5"]}]
    assert any(e.startswith("md5") for e in verify(planted))

    planted = [dict(r) for r in rows]
    for r in planted:
        if r["clip_uid"] in (test_row["clip_uid"], dev_row["clip_uid"]):
            r["near_dup_group"] = "nd-x"
    assert any(e.startswith("near_dup_group") for e in verify(planted))

    planted = [dict(r) for r in rows] + [{**_row("planted00003", "HT998", "antrum", "dev", fold=0),
                                           "day_key": test_row["day_key"], "tod_s": test_row["tod_s"]}]
    assert any("room window" in e for e in verify(planted))

    planted = [dict(r) for r in rows] + [_row("planted00004", "HT997", "antrum", "dev", fold=0,
                                              rel_path="Full lenght videos/REC_0001.mp4")]
    assert any("full-length" in e for e in verify(planted))


# --- cache helpers, layouts, cap detector -------------------------------------------------

def test_frame_dups_and_dhash():
    from training.landmarks.extract_cache import centre_thumb, dhash, frame_dups

    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    other = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    t1, t2 = centre_thumb(canvas), centre_thumb(other)
    assert t1.shape == (32, 32) and len(dhash(t1)) == 16 and dhash(t1) == dhash(t1.copy())
    assert frame_dups(np.stack([t1, t1, t2, t1]), [0, 3, 6, 9]) == [None, 0, None, 0]


def test_near_duplicate_search_skips_textureless_frames(tmp_path):
    pd = pytest.importorskip("pandas")
    from training.landmarks.extract_cache import _popcount, centre_thumb, dhash, near_duplicate_pairs

    assert _popcount(np.array([0b1011, 0], np.uint64)).tolist() == [3, 0]
    rng = np.random.default_rng(0)
    textured = [centre_thumb(rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)) for _ in range(4)]
    flat = np.full((32, 32), 120, np.uint8)
    # a and b share 3 textured frames (a re-encode); c and d share only flat frames.
    clips = {"a": textured[:3], "b": textured[:3], "c": [flat] * 3 + [textured[3]], "d": [flat] * 3}
    recs, thumbs = [], []
    for uid, ths in clips.items():
        for k, t in enumerate(ths):
            recs.append({"clip_uid": uid, "src_idx": 3 * k, "dhash": dhash(t), "dup_of": pd.NA, "sharpness": 900.0})
            thumbs.append(t)
    np.save(tmp_path / "thumbs.npy", np.stack(thumbs))
    frames = pd.DataFrame(recs).astype({"dup_of": "Int64"})
    rows = [{"clip_uid": u} for u in clips]
    assert near_duplicate_pairs(tmp_path, frames, rows) == [("a", "b", 1.0, 1.0)]


def test_fit_octagon_recovers_known_apertures():
    import cv2

    from app.ml.landmarks.layouts import LEGACY_1350, OLYMPUS_1920
    from training.landmarks.derive_layouts import fit_octagon

    for lay in (OLYMPUS_1920, LEGACY_1350):
        lit = np.zeros((lay.ref_height, lay.ref_width), np.uint8)
        cv2.fillPoly(lit, [np.array(lay.polygon, np.int32)], 1)
        lit[100:130, 5:60] = 1  # OSD text outside the aperture must not move the edges
        got = fit_octagon(lit.astype(bool))
        assert np.abs(got - np.array(lay.polygon)).max() <= 1.0


def test_cap_ring_detector_separates_ring_from_texture():
    import cv2

    from app.ml.landmarks.layouts import OLYMPUS_1920, aperture_mask
    from training.landmarks.derive_layouts import CAP_THRESHOLD, cap_ring_score

    rng = np.random.default_rng(0)
    mask = aperture_mask(OLYMPUS_1920, 320)

    def frames(ring: bool) -> list[np.ndarray]:
        out = []
        for _ in range(8):
            noise = cv2.GaussianBlur(rng.integers(60, 200, (320, 320), dtype=np.uint8), (0, 0), 3)
            img = np.dstack([noise] * 3)
            if ring:
                cv2.circle(img, (160, 160), 136, (240, 240, 240), 2)
            img[~mask] = 0
            out.append(img)
        return out

    assert cap_ring_score(frames(True)) >= CAP_THRESHOLD
    assert cap_ring_score(frames(False)) < CAP_THRESHOLD


# --- leak audit and adjudication -----------------------------------------------------------

def test_leak_audit_flags_cross_split_copies():
    from training.landmarks.leak_audit import audit, degrade

    rng = np.random.default_rng(0)
    rows, embs = [], []
    for p in range(12):
        base = rng.normal(size=64)
        for c in range(6):
            e = base + rng.normal(scale=0.9, size=(3, 64))
            rows.append({"clip_uid": f"c{p}-{c}", "patient_id": f"P{p}", "split": "test" if p < 3 else "dev",
                         "label": "antrum"})
            embs.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    emb = np.stack(embs)
    clean = audit(rows, emb)
    assert clean["flags"] == [] and clean["B_above_t_dup"] == 0
    assert clean["w_max"] < clean["t_dup"] < 1.0
    copy = emb[0] + rng.normal(scale=0.02, size=(3, 64))
    rows2 = rows + [{"clip_uid": "copy", "patient_id": "PX", "split": "dev", "label": "antrum"}]
    res = audit(rows2, np.concatenate([emb, (copy / np.linalg.norm(copy, axis=1, keepdims=True))[None]]))
    assert [(f["kind"], {f["clip_a"], f["clip_b"]}) for f in res["flags"]] == [("cross_split", {"c0-0", "copy"})]
    img = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    assert degrade(img).shape == img.shape and degrade(img).dtype == np.uint8


def _smooth_canvas(seed: int) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (320, 320, 3), dtype=np.uint8).astype(np.float32), (0, 0), 12)
    return np.clip((img - img.mean()) * 10 + 128, 0, 255).astype(np.uint8)  # mucosa-like low-frequency texture


def test_dhash_rule_catches_shifted_brightened_reexports_gain_invariantly():
    from training.landmarks.extract_cache import centre_thumb, is_near_duplicate, thumb_mad
    from training.landmarks.leak_audit import degrade, dhash_match

    src = np.stack([centre_thumb(_smooth_canvas(i)) for i in range(12)])
    shifted = np.stack([centre_thumb(degrade(_smooth_canvas(i), 75, 1.03, 2)) for i in range(12)])
    bright = np.stack([centre_thumb(degrade(_smooth_canvas(i), 75, 1.12, 2)) for i in range(12)])
    other = np.stack([centre_thumb(_smooth_canvas(100 + i)) for i in range(12)])
    assert dhash_match(src, shifted)["caught"]
    assert dhash_match(src, bright)["caught"] and not dhash_match(src, bright, gain_invariant=False)["caught"]
    assert not dhash_match(src, other)["caught"]
    a = np.full((32, 32), 100, np.uint8)
    assert thumb_mad(a, a + 20) == 20 and thumb_mad(a, a + 20, gain_invariant=True) == 0
    assert is_near_duplicate(10, 10, 3, 3) and not is_near_duplicate(10, 10, 1, 1)
    assert not is_near_duplicate(30, 30, 2, 2)  # 2/30 < 0.3


def _report_inputs(t_dup: float = 0.95):
    res = {"n_clips": 10, "w_max": 0.9, "t_dup": t_dup, "W": [0.5] * 5, "B": [0.4] * 5, "X": [0.4] * 5, "n_W": 1,
           "n_B": 1, "n_X": 1, "B_above_t_dup": 0, "X_above_t_dup": 0, "flags": []}
    pat = {"n_same": 1, "n_null": 1, "same_p50": 0.5, "null_p50": 0.5, "t_at_90_sensitivity": 0.5,
           "false_alarm_at_90_sensitivity": 0.5}
    ctl = {"planted_copy": 0.99, "planted_copy_nn_is_source": True, "planted_copy_dhash": {"frac": 1.0, "caught": True},
           "planted_stress_shift2_q75": 0.87, "planted_stress_shift2_q75_nn_is_source": False,
           "planted_stress_shift2_q75_dhash": {"frac": 0.8, "caught": True},
           "planted_recut": 0.79, "planted_recut_nn_is_source": False, "known_reencode_x": 1.0, "source_clip": "t0"}
    return res, pat, ctl


def test_leak_audit_requires_the_shifted_control_from_either_detector(tmp_path):
    from training.landmarks.leak_audit import control_verdicts, dhash_flags, write_report

    res, pat, ctl = _report_inputs()
    v = control_verdicts(ctl, res["t_dup"])
    assert v["planted_stress_shift2_q75"] == {"similarity": 0.87, "embedding": False, "dhash": True,
                                             "required": True, "caught": True}
    assert not v["planted_recut"]["required"] and v["known_reencode_x"]["required"]
    md, csv = tmp_path / "r.md", tmp_path / "r.csv"
    assert write_report(res, pat, ctl, {"weights_sha256": "0" * 64}, md, csv)
    assert md.read_text().rstrip().splitlines()[-1].startswith("PASS")
    missed = {**ctl, "planted_stress_shift2_q75_dhash": {"frac": 0.1, "caught": False}}
    assert not write_report(res, pat, missed, {"weights_sha256": "0" * 64}, md, csv)  # missed by both: FAIL
    assert "MISSED" in md.read_text()
    rows = [{"clip_uid": "a", "patient_id": "HT901", "split": "test", "label": "antrum"},
            {"clip_uid": "b", "patient_id": "HT902", "split": "dev", "label": "antrum"},
            {"clip_uid": "c", "patient_id": "HT902", "split": "dev", "label": "z_line"}]
    flags = dhash_flags([("a", "b", 0.6, 0.5), ("b", "c", 0.9, 0.9)], rows)  # same patient: not a flag
    assert [f["kind"] for f in flags] == ["dhash_cross_split"]
    assert not write_report(res, pat, ctl, {"weights_sha256": "0" * 64}, md, csv, flags)


def test_aperture_mask_is_eroded_at_the_canvas_border():
    from app.ml.landmarks.layouts import CANON_SIZE, LAYOUTS, OLYMPUS_1920, aperture_mask

    for size in (CANON_SIZE, 224, 288):
        e = max(1, int(round(OLYMPUS_1920.erode_px * size / CANON_SIZE)))
        for lay in LAYOUTS:
            m = aperture_mask(lay, size)
            # the octagon's vertical sides lie on the canvas border: the rim must be eroded there too
            assert not m[:, :e].any() and not m[:, -e:].any(), (lay.name, size)
            assert not m[:e].any() and not m[-e:].any()
        assert aperture_mask(OLYMPUS_1920, size)[:, e].any()  # ... and only the rim


def test_adjudication_collect_and_kappa(tmp_path):
    from training.landmarks.common import read_csv, write_csv
    from training.landmarks.contact_sheets import TEMPLATE_FIELDS, cohen_kappa, collect

    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert abs(cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])) < 1e-9
    write_csv(tmp_path / "adjudication_contested.csv", [
        {"item": 1, "clip_uid": "u1", "final": "incisura", "reviewer": "dr"},
        {"item": 2, "clip_uid": "u2", "final": ""},
        {"item": 3, "clip_uid": "u3", "final": "lesser_curvature_retroflex:0.5|incisura:0.5"},
        {"item": 4, "clip_uid": "u4", "final": "not_a_station"},
    ], TEMPLATE_FIELDS)
    out = tmp_path / "label_overrides.csv"
    res = collect(tmp_path, out)
    assert res["overrides_written"] == 2 and len(res["invalid"]) == 1
    assert {r["clip_uid"]: r["label"] for r in read_csv(out)} == {
        "u1": "incisura", "u3": "lesser_curvature_retroflex:0.5|incisura:0.5"}
