"""Landmark training tooling: sampling, augmentation, optimisation pieces and a
synthetic end-to-end smoke run (train -> OOF -> export -> serving load).

Everything is synthetic: generated manifests, canvases and frame tables with
invented patient IDs. Nothing here reads data/.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from app.ml.landmarks.layouts import OLYMPUS_1920, aperture_mask  # noqa: E402
from app.ml.landmarks.preprocess import to_model_input  # noqa: E402
from app.ml.landmarks.stations import NUM_STATIONS, STATION_INDEX, STATION_ORDER  # noqa: E402
from training.landmarks import dataset as D  # noqa: E402
from training.landmarks.augment import (  # noqa: E402
    AugmentConfig,
    assert_batch_masked,
    assert_zero_outside,
    augment,
    augment_to_tensor,
    eval_input,
    finish,
)

LC_INC = "lesser_curvature_retroflex:0.5|incisura:0.5"


# -- synthetic tables ---------------------------------------------------------------------------------------

def manifest_rows(n_patients: int = 10, stations=STATION_ORDER, clips_per_station: int = 1) -> pd.DataFrame:
    rows = []
    for p in range(n_patients):
        pid = f"HT9{p:02d}"
        for si, st in enumerate(stations):
            for c in range(clips_per_station):
                uid = f"{p:02d}{si:02d}{c:02d}abcdef"[:12]
                rows.append({
                    "clip_uid": uid, "rel_path": f"X/{pid}-{si}-{c}.mp4", "alt_rel_path": "", "folder_class": st,
                    "suffix": "", "suffix_class": st, "label": st, "label_source": "folder", "soft_label": "",
                    "exclude_reason": "", "eval_exclude": "false", "flags": "", "dup_group": "", "near_dup_group": "",
                    "patient_id": pid, "patient_id_raw": pid[2:], "id_resolution": "regex", "group_id": pid,
                    "room_hash": "r1" if p % 2 else "r2", "scope_hash": "s", "day_key": "d", "tod_s": "0",
                    "cohort": "new_1920", "layout": OLYMPUS_1920.name, "width": "1920", "height": "1080",
                    "n_frames": "36", "fps": "30", "dur_s": "1.2", "md5": uid * 2,
                    "nbi_frac": "1" if si < 3 else "0", "mode_major": "nbi" if si < 3 else "wl",
                    "cap": "false", "cap_score": "0", "frozen": "false", "n_cached": "12", "n_unique": "12",
                    "rel_t_s": str(10.0 * (si * clips_per_station + c)), "order_in_patient": str(si),
                    "overlap_head_s": "", "overlap_tail_s": "", "overlap_with": "",
                    "split": "dev", "fold": str(p % 5),
                })
    return pd.DataFrame(rows)


def frame_rows(m: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    rows = []
    for r in m.itertuples():
        for j in range(n):
            rows.append({"clip_uid": r.clip_uid, "src_idx": 3 * j, "u": j / (n - 1), "t_s": j / 10,
                         "sharpness": 1200.0, "dark_frac": 0.0, "sat_frac": 0.0, "redout": 0.0, "quality": 1.0,
                         "sharp_ratio_to_clip_median": 1.0, "dhash": "0", "dup_of": np.nan,
                         "mode_frame": "nbi" if r.mode_major == "nbi" else "wl", "layout_ok": True})
    return pd.DataFrame(rows)


def loaded(m: pd.DataFrame, tmp_path) -> pd.DataFrame:
    path = tmp_path / "manifest.csv"
    m.to_csv(path, index=False)
    return D.load_manifest(path)


# -- dataset ----------------------------------------------------------------------------------------------

def test_clip_targets_use_soft_labels(tmp_path):
    m = manifest_rows(1)
    m.loc[m["label"] == "incisura", "soft_label"] = LC_INC
    t = D.clip_targets(loaded(m, tmp_path))
    inc = int(np.flatnonzero(m["label"] == "incisura")[0])
    assert t[inc, STATION_INDEX["incisura"]] == t[inc, STATION_INDEX["lesser_curvature_retroflex"]] == 0.5
    assert np.allclose(t.sum(1), 1) and (t.max(1) >= 0.5).all()


def test_training_frames_are_filtered(tmp_path):
    m = manifest_rows(1, stations=["antrum"])
    m.loc[0, ["overlap_head_s", "dur_s"]] = ["0.25", "1.1"]
    ml = loaded(m, tmp_path)
    f = frame_rows(m)
    f.loc[5, "dup_of"] = 4
    f.loc[6, "sharp_ratio_to_clip_median"] = 0.3
    f.loc[7, "redout"] = 0.8
    f.loc[8, "layout_ok"] = False
    t = D.train_frame_table(f, ml, D.DataConfig())
    kept = set(t["src_idx"])
    assert t["u"].between(0.1, 0.9).all()
    assert not {15, 18, 21, 24} & kept  # dup, blurred, red-out, unverified layout
    assert min(kept) >= 3 * 3  # t_s < 0.25 s overlaps a different-label clip
    assert (t["tri"] >= 0.2).all() and (t["tri"] <= 1.0).all()


def test_triangular_centre_weighting():
    np.testing.assert_allclose(D.triangular(np.array([0.0, 0.25, 0.5, 1.0])), [0.2, 0.6, 1.0, 0.2])


def test_hierarchical_sampler_caps_each_patient(tmp_path):
    # one heavy patient with 20 antrum clips, three light ones with 1 each
    m = pd.concat([manifest_rows(1, stations=["antrum"], clips_per_station=20),
                   manifest_rows(4, stations=["antrum"]).iloc[1:]], ignore_index=True)
    m["clip_uid"] = [f"{i:012d}" for i in range(len(m))]
    ml = loaded(m, tmp_path)
    f = frame_rows(m)
    s = D.ClipSampler(ml, D.train_frame_table(f, ml), D.clip_targets(ml), D.DataConfig(patient_cap=4))
    share = s.patient_share()
    assert share["HT900"] == pytest.approx(4 / 7)
    assert all(share[p] == pytest.approx(1 / 7) for p in ("HT901", "HT902", "HT903"))


def test_mode_balance_moves_nbi_share_within_the_weight_cap(tmp_path):
    m = manifest_rows(20, stations=["antrum"])
    m["nbi_frac"] = ["1"] * 19 + ["0"]
    ml = loaded(m, tmp_path)
    f = frame_rows(m)
    f["mode_frame"] = np.where(f["clip_uid"].isin(m["clip_uid"][:19]), "nbi", "wl")
    s = D.ClipSampler(ml, D.train_frame_table(f, ml), D.clip_targets(ml), D.DataConfig())
    r = s.mode_report["antrum"]
    assert r.natural_share == pytest.approx(0.95)
    assert r.w_wl == 3.0 and r.w_nbi == 1.0  # the 3x cap binds
    assert r.achieved_share == pytest.approx(0.95 / (0.95 + 0.05 * 3))
    off = D.ClipSampler(ml, D.train_frame_table(f, ml), D.clip_targets(ml), D.DataConfig(mode_balance=False))
    assert off.mode_report["antrum"].achieved_share == pytest.approx(0.95)


def test_frozen_clips_are_down_weighted(tmp_path):
    m = manifest_rows(1, stations=["antrum"], clips_per_station=2)
    m["clip_uid"] = ["a" * 12, "b" * 12]
    m.loc[0, "frozen"] = "true"
    ml = loaded(m, tmp_path)
    s = D.ClipSampler(ml, D.train_frame_table(frame_rows(m), ml), D.clip_targets(ml), D.DataConfig())
    assert s.clip_prob[0] / s.clip_prob[1] == pytest.approx(0.25)


def test_schedule_is_deterministic(tmp_path):
    m = manifest_rows(3)
    ml = loaded(m, tmp_path)
    ft = D.train_frame_table(frame_rows(m), ml)
    s = D.ClipSampler(ml, ft, D.clip_targets(ml), D.DataConfig(k_frames=4, clips_per_batch=5))
    a, b = s.schedule(7, seed=3), s.schedule(7, seed=3)
    assert a.shape == (7, 20)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, s.schedule(7, seed=4))
    clips = ft["clip_i"].to_numpy()[a[0]].reshape(5, 4)
    assert (clips == clips[:, :1]).all()  # k frames of the same clip per slot


def test_permuted_targets_are_a_permutation(tmp_path):
    ml = loaded(manifest_rows(2), tmp_path)
    t = D.clip_targets(ml)
    p = D.permute_targets(ml, t, seed=0)
    assert sorted(map(tuple, p)) == sorted(map(tuple, t)) and not np.array_equal(p, t)


# -- augmentation ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def canvas():
    from app.ml.landmarks.bench import synthetic_olympus_frame
    from app.ml.landmarks.preprocess import canonicalize

    return canonicalize(synthetic_olympus_frame(1920, seed=2, badge="NBI"), OLYMPUS_1920)


def test_augmentation_is_deterministic_per_rng(canvas):
    cfg = AugmentConfig(rotate_p=1.0, blur_p=1.0, jpeg_p=1.0, gray_p=0.5)
    a = augment(canvas, OLYMPUS_1920, cfg, np.random.default_rng([1, 2]))
    b = augment(canvas, OLYMPUS_1920, cfg, np.random.default_rng([1, 2]))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, augment(canvas, OLYMPUS_1920, cfg, np.random.default_rng([1, 3])))


@pytest.mark.parametrize("size", [224, 288])
@pytest.mark.parametrize("cfg", [
    AugmentConfig(),
    AugmentConfig(rotate_p=1.0, rotate_deg=180, hflip_p=0.5, vflip_p=0.5, pseudo_nbi_p=0.5),
    AugmentConfig(mask="circle", jpeg_p=1.0, blur_p=1.0),
    AugmentConfig(enabled=False),
])
def test_every_model_input_is_exactly_zero_outside_the_mask(canvas, cfg, size):
    rng = np.random.default_rng(0)
    for _ in range(25):
        img = augment(canvas, OLYMPUS_1920, cfg, rng, size)
        assert img.shape == (size, size, 3)
        assert_zero_outside(img, OLYMPUS_1920, cfg.mask)
    batch = torch.stack([augment_to_tensor(canvas, OLYMPUS_1920, cfg, rng, size) for _ in range(4)])
    assert_batch_masked(batch, OLYMPUS_1920, cfg.mask)


def test_mask_assertions_catch_a_leak(canvas):
    img = eval_input(canvas, OLYMPUS_1920)
    img[0, 0] = 1
    with pytest.raises(AssertionError):
        assert_zero_outside(img, OLYMPUS_1920)
    x = augment_to_tensor(canvas, OLYMPUS_1920, AugmentConfig(), np.random.default_rng(0))[None]
    x[0, 1, 0, 0] += 0.5
    with pytest.raises(AssertionError):
        assert_batch_masked(x, OLYMPUS_1920)


def test_eval_input_is_the_serving_model_input(canvas):
    np.testing.assert_array_equal(eval_input(canvas, OLYMPUS_1920, 224), to_model_input(canvas, OLYMPUS_1920, 224))


def test_unmasked_path_keeps_pixels_outside_the_aperture():
    from app.ml.landmarks.bench import synthetic_olympus_frame
    from training.landmarks.sanity import canonicalize_unmasked

    raw = canonicalize_unmasked(synthetic_olympus_frame(1920, seed=2, badge="NBI"), OLYMPUS_1920)
    img = finish(raw, OLYMPUS_1920, 224, masked=False)
    assert np.count_nonzero(img[~aperture_mask(OLYMPUS_1920, 224)]) > 0


# -- optimisation pieces -----------------------------------------------------------------------------------

def test_lr_schedule_warmup_then_cosine():
    from training.landmarks.train import lr_factor

    assert lr_factor(0, 10, 100) == pytest.approx(0.1)
    assert lr_factor(9, 10, 100) == pytest.approx(1.0)
    assert lr_factor(55, 10, 100) == pytest.approx(0.5)
    assert lr_factor(100, 10, 100) == pytest.approx(0.0)
    assert lr_factor(100, 10, 100, 0.1) == pytest.approx(0.1)


def test_ema_warms_up_and_tracks_buffers():
    from app.ml.landmarks.architectures import build
    from training.landmarks.train import Ema

    net = build("tiny_test_cnn")
    ema = Ema(net, decay=0.995)
    assert ema.current_decay() == pytest.approx(1 / 10)
    net.train()
    net(torch.randn(8, 3, 32, 32))  # updates BN running stats
    ema.update(net)
    assert ema.current_decay() == pytest.approx(2 / 11)
    bn_live, bn_ema = net.backbone[1].running_mean, ema.module.backbone[1].running_mean
    assert torch.allclose(bn_ema, bn_live * (1 - 2 / 11))  # EMA of a buffer that started at 0
    for _ in range(3000):
        ema.updates += 1
    assert ema.current_decay() == 0.995


def test_param_groups_split_head_and_no_decay():
    from app.ml.landmarks.architectures import build
    from training.landmarks.train import DEFAULTS, param_groups

    net = build("tiny_test_cnn")
    groups = param_groups(net, DEFAULTS["optim"])
    head = [g for g in groups if g["lr"] == DEFAULTS["optim"]["lr_head"]]
    assert sum(p.numel() for g in head for p in g["params"]) == sum(p.numel() for p in net.head.parameters())
    for g in groups:
        if any(p.ndim <= 1 for p in g["params"]):
            assert g["weight_decay"] == 0.0


def test_soft_cross_entropy_with_smoothing():
    from training.landmarks.train import soft_cross_entropy

    logits = torch.randn(4, NUM_STATIONS)
    t = torch.eye(NUM_STATIONS)[:4]
    want = -(((1 - 0.1) * t + 0.1 / NUM_STATIONS) * torch.log_softmax(logits, -1)).sum(-1).mean()
    assert soft_cross_entropy(logits, t, 0.1) == pytest.approx(float(want))


def test_backbone_allowlist_refuses_unlisted_and_probe_only():
    from training.landmarks.backbones import ALLOWLIST, NotAllowed, load_pretrained

    with pytest.raises(NotAllowed):
        load_pretrained("vit_huge_patch14_whatever")
    with pytest.raises(NotAllowed):
        load_pretrained("dinov2_b14")  # frozen-probe ceiling only
    assert all(len(c.sha256) == 64 for c in ALLOWLIST.values())
    assert all(c.revision for c in ALLOWLIST.values() if c.source == "hf")
    assert {c.license for c in ALLOWLIST.values()} <= {"BSD-3-Clause", "Apache-2.0", "MIT (ours)"}


def test_configs_resolve_with_base_inheritance():
    from training.landmarks.run_utils import CONFIGS_DIR
    from training.landmarks.train import resolve_config

    abl = resolve_config("ablation_flips")
    assert abl["candidate"] == "regy32_in1k" and abl["augment"]["hflip_p"] == 0.5
    assert resolve_config("stageB_regy32_in1k")["augment"]["hflip_p"] == 0.0  # no flips by default
    for path in CONFIGS_DIR.glob("*.json"):
        cfg = resolve_config(path.stem)
        assert cfg["input"]["masked"] or cfg["sanity"], path.name  # unmasked only in sanity runs


def test_gates_are_parseable_and_merge_to_eight_nodes():
    from training.landmarks.evaluate import merged_map
    from training.landmarks.run_utils import GATES_PATH

    gates = json.loads(GATES_PATH.read_text())
    nodes, idx = merged_map(gates)
    assert len(nodes) == 8 and idx[0] == idx[1] and idx[7] == idx[8]
    assert set(gates["G-TEST"]["primary_endpoints"]) == {"clip_macro_f1", "s4_fdr_overall", "recall_mean"}


# -- synthetic end to end -----------------------------------------------------------------------------------

@pytest.fixture()
def synthetic_dataset(tmp_path):
    import cv2

    from training.landmarks.common import write_csv

    m = manifest_rows(10)
    f = frame_rows(m, n=12)
    cache = tmp_path / "cache"
    mask = aperture_mask(OLYMPUS_1920)
    rng = np.random.default_rng(0)
    for r in m.itertuples():
        d = cache / "frames" / r.clip_uid
        d.mkdir(parents=True)
        hue = STATION_INDEX[r.label] * 17
        for j in range(12):
            hsv = np.dstack([np.full((320, 320), hue, np.uint8), np.full((320, 320), 150, np.uint8),
                             rng.integers(80, 220, (320, 320), dtype=np.uint8)])
            c = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            c[~mask] = 0
            cv2.imwrite(str(d / f"{3 * j:05d}.jpg"), cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
    write_csv(cache / "frames.csv.gz", f.to_dict("records"), list(f.columns))
    (cache / "CACHE_INFO.json").write_text("{}")
    mpath = tmp_path / "manifest.csv"
    m.to_csv(mpath, index=False)
    return mpath, cache


@pytest.mark.slow
def test_train_smoke_end_to_end_on_synthetic_data(synthetic_dataset, tmp_path):
    from app.ml.landmarks.bench import synthetic_olympus_frame
    from app.ml.landmarks.bundle import load_bundle
    from app.ml.landmarks.classifier import LandmarkModelManager, RealLandmarkClassifier
    from training.landmarks.export_bundle import export
    from training.landmarks.train import SMOKE, resolve_config, train

    mpath, cache = synthetic_dataset
    cfg = resolve_config("smoke", {**SMOKE, "candidate": "scratch:tiny_test_cnn",
                                   "input": {"cache_root": str(cache)},
                                   "data": {**SMOKE["data"], "max_clips": None}, "loader": {"workers": 0}})
    run = tmp_path / "runs" / "exp" / "fold0"
    met = train(cfg, "fold", run, fold=0, device="cpu", manifest_path=mpath, log=lambda *_: None)

    for name in ("config.json", "provenance.json", "metrics.json", "best_raw.pt", "best_ema.pt",
                 "oof_fold0.npz", "inner_fold0.npz", "oof_fold0_alt.npz"):
        assert (run / name).exists(), name
    assert met["best"]["variant"] in ("raw", "ema") and met["history"]
    z = np.load(run / "oof_fold0.npz")
    assert z["logits"].dtype == np.float32 and np.isfinite(z["logits"]).all()
    assert z["emb"].dtype == np.float16 and len(z["clip_uid"]) == 2 * NUM_STATIONS * 12  # every cached frame
    assert set(np.unique(z["clip_uid"])) == set(pd.read_csv(mpath).query("fold == 0")["clip_uid"])
    prov = json.loads((run / "provenance.json").read_text())
    assert prov["manifest_sha256"] and prov["torch"] == torch.__version__

    m = D.load_manifest(mpath)
    res = export(run, "lm-0.0.1-test", tmp_path / "bundle", smoke=True, manifest=m, sharp_ref=1000.0)
    b = load_bundle(res["bundle_dir"], "cpu")
    assert b.arch == "tiny_test_cnn" and b.quality.sharp_ref == 1000.0
    assert b.meta["metrics"]["WARNING"]
    out = RealLandmarkClassifier(LandmarkModelManager(b)).predict(synthetic_olympus_frame(1920, seed=1))
    assert out.status in {"ok", "uncertain", "low_quality"} and len(out.probs) == NUM_STATIONS
    assert len(res["sha256"]) == 64 and res["lock"]["s3_key"].endswith("landmarks-lm-0.0.1-test.tar.gz")


# -- review fixes -------------------------------------------------------------------------------------------

def test_inner_val_selection_follows_the_runs_station_set(tmp_path):
    from training.landmarks.train import early_stop_table, validation_config

    run = D.DataConfig(exclude_stations=("z_line",), permute_labels=True, u_range=(0.3, 0.7), max_clips=5)
    v = validation_config(run)
    assert v.exclude_stations == ("z_line",) and not v.permute_labels
    assert v.u_range == D.DataConfig().u_range and v.max_clips is None  # same population for every ablation
    m = loaded(manifest_rows(5), tmp_path)
    _, inner, _ = D.fold_patients(m, 0)
    clips = D.training_clips(m, inner, v)
    assert len(clips) and "z_line" not in set(clips["label"])  # LSO: the OOD station never selects checkpoints
    table = early_stop_table(frame_rows(clips), clips, v, 2)
    assert set(table["clip_uid"]) <= set(clips["clip_uid"])


def test_fp32_inference_turns_tf32_off_and_restores_it():
    from training.landmarks.run_utils import fp32_inference, tf32_state

    before = tf32_state()
    with fp32_inference():
        assert tf32_state() == {"cudnn_conv": "ieee", "cuda_matmul": "ieee"}
    assert tf32_state() == before
    with pytest.raises(RuntimeError), fp32_inference():
        raise RuntimeError("restored on errors too")
    assert tf32_state() == before
