"""Held-out accuracy of an exported bundle, on a synthetic manifest and frame cache.

Nothing here reads data/: patients, clips and canvases are generated.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from app.ml.landmarks.stations import NUM_STATIONS, STATION_ORDER  # noqa: E402
from training.landmarks import bundle_eval  # noqa: E402
from training.landmarks.common import sha256_file  # noqa: E402
from training.landmarks.run_utils import write_json  # noqa: E402

from tests.test_landmark_train_smoke import synthetic_dataset  # noqa: E402, F401  (fixture)


def _evaluate(bundle, dataset, split="heldout"):
    manifest, cache = dataset
    return bundle_eval.evaluate_bundle(bundle, split, "cpu", manifest, cache, workers=0,
                                       progress=lambda *a: None)


def test_a_fold_model_is_scored_on_the_fold_it_held_out(make_tiny_bundle, synthetic_dataset, tmp_path):  # noqa: F811
    bundle = make_tiny_bundle("lm-0.1.0", train={"run": "fold1"})
    res = _evaluate(bundle, synthetic_dataset)

    # 10 synthetic patients, fold = patient index % 5: fold 1 holds patients 1 and 6, one clip per station.
    assert res["split"] == {"name": "heldout", "label": "dev fold 1 (held out)", "fold": 1, "n_clips": 20,
                            "n_patients": 2, "n_frames": 240, "skipped_clips": 0}
    assert res["version"] == "lm-0.1.0" and res["device"] == "cpu" and res["stations"] == list(STATION_ORDER)
    assert res["calibration"] == {"temperature": 1.0, "q_min": pytest.approx(0.2), "quality_gate": True}

    clip = res["clip"]
    confusion = np.asarray(clip["confusion"])
    assert confusion.shape == (NUM_STATIONS, NUM_STATIONS) and confusion.sum() == 20
    assert (confusion.sum(1) == 2).all()  # two clips of every station
    assert 0.0 <= clip["macro_f1"]["value"] <= 1.0 and clip["coverage"] == 1.0
    assert set(clip["per_class"]) <= set(STATION_ORDER)

    share = res["acceptance"]["share"]
    assert sum(share.values()) == pytest.approx(1.0) and res["acceptance"]["n_frames"] == 240
    assert res["frame"]["all_frames"]["n"] == 240

    # Aggregates only: no patient ID, clip id or path in what is written.
    out = tmp_path / "result.json"
    write_json(out, res)
    text = out.read_text()
    manifest = pd.read_csv(synthetic_dataset[0], dtype=str)
    assert "HT9" not in text and ".mp4" not in text
    assert not any(uid in text for uid in manifest["clip_uid"])
    json.loads(text)


def test_progress_reaches_every_frame(make_tiny_bundle, synthetic_dataset):  # noqa: F811
    manifest, cache = synthetic_dataset
    seen = []
    bundle_eval.evaluate_bundle(make_tiny_bundle("lm-0.1.0", train={"run": "fold0"}), "heldout", "cpu",
                                manifest, cache, batch=100, workers=0, progress=lambda *a: seen.append(a))
    assert seen[0] == (0, 240, "frames") and seen[-1] == (240, 240, "frames")
    assert [done for done, _, _ in seen] == [0, 100, 200, 240]


def test_an_all_dev_model_has_no_held_out_fold(make_tiny_bundle, synthetic_dataset):  # noqa: F811
    with pytest.raises(SystemExit, match="trained on every dev patient"):
        _evaluate(make_tiny_bundle("lm-1.0.0", train={"run": "all_dev"}), synthetic_dataset)


def test_a_bundle_of_unknown_origin_is_refused(make_tiny_bundle, synthetic_dataset):  # noqa: F811
    with pytest.raises(SystemExit, match="cannot tell which dev fold"):
        _evaluate(make_tiny_bundle("lm-0.1.0"), synthetic_dataset)


def test_the_shift_set_is_scored_with_the_quality_gate_off(make_tiny_bundle, synthetic_dataset):  # noqa: F811
    manifest, cache = synthetic_dataset
    m = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    shift = m["patient_id"] == "HT903"
    m.loc[shift, ["split", "fold"]] = ["shift", "-1"]
    m.to_csv(manifest, index=False)

    res = _evaluate(make_tiny_bundle("lm-1.0.0", train={"run": "all_dev"}), synthetic_dataset, "shift")
    assert res["split"]["name"] == "shift" and res["split"]["n_clips"] == 10 and res["split"]["n_patients"] == 1
    assert res["calibration"]["q_min"] == 0.0 and res["calibration"]["quality_gate"] is False


def test_excluded_and_test_clips_are_never_scored(make_tiny_bundle, synthetic_dataset):  # noqa: F811
    manifest, cache = synthetic_dataset
    m = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    held_out = m.index[(m["fold"] == "0") & (m["split"] == "dev")]
    m.loc[held_out[0], "exclude_reason"] = "manual"
    m.loc[held_out[1], "eval_exclude"] = "true"
    m.loc[m["patient_id"] == "HT905", ["split", "fold"]] = ["test", "-1"]  # the other fold-0 patient
    m.to_csv(manifest, index=False)

    res = _evaluate(make_tiny_bundle("lm-0.1.0", train={"run": "fold0"}), synthetic_dataset)
    assert res["split"]["n_patients"] == 1 and res["split"]["n_clips"] == 8


def test_the_locked_test_set_is_not_a_split():
    assert bundle_eval.SPLITS == ("heldout", "shift")
    with pytest.raises(SystemExit):
        bundle_eval.main(["--bundle", "x", "--split", "test"])


def test_warnings_flag_changed_data_and_uncalibrated_exports(make_tiny_bundle, synthetic_dataset):  # noqa: F811
    manifest, cache = synthetic_dataset
    same = make_tiny_bundle("lm-0.1.0", train={"run": "fold0", "manifest_sha256": sha256_file(manifest)})
    assert _evaluate(same, synthetic_dataset)["warnings"] == []

    stale = make_tiny_bundle("lm-0.2.0", train={"run": "fold0", "manifest_sha256": "0" * 64,
                                                "clip_set_sha256": "1" * 64},
                             metrics={"WARNING": "smoke/uncalibrated export: not for serving"})
    warnings = _evaluate(stale, synthetic_dataset)["warnings"]
    assert len(warnings) == 3
    assert "manifest has changed" in warnings[0] and "different clip set" in warnings[1]
    assert warnings[2].startswith("Uncalibrated export")


def test_export_records_the_mode_and_fold_bundle_eval_relies_on(tmp_path):
    from training.landmarks.export_bundle import build_meta

    run = tmp_path / "exp" / "custom_name"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"mode": "fold", "fold": 3}))
    weights = run / "best_raw.pt"
    weights.write_bytes(b"w")
    manifest = pd.DataFrame({"split": ["test"], "patient_id": ["HT900"]})
    meta = build_meta(run, None, "lm-0.0.1-smoke", {"candidate": "scratch:tiny_test_cnn", "input": {"size": 224}},
                      "raw", weights, smoke=True, manifest=manifest, sharp_ref=1000.0)
    assert meta["train"]["mode"] == "fold" and meta["train"]["fold"] == 3 and meta["train"]["run"] == "custom_name"
