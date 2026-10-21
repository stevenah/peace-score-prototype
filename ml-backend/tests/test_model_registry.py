"""Bundle discovery, the saved selection, and swapping the served model."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.config import settings  # noqa: E402
from app.ml.landmarks import registry  # noqa: E402
from app.ml.landmarks.bundle import META_FILE, MODEL_FILE, LandmarkUnavailable  # noqa: E402
from app.ml.landmarks.classifier import (  # noqa: E402
    LandmarkModelManager,
    RealLandmarkClassifier,
    _bundle_path_from_settings,
)
from app.ml.landmarks.registry import ServingState  # noqa: E402


def _edit_meta(bundle_dir: Path, fn) -> None:
    meta = json.loads((bundle_dir / META_FILE).read_text())
    fn(meta)
    (bundle_dir / META_FILE).write_text(json.dumps(meta))


def _corrupt_weights(bundle_dir: Path) -> None:
    with open(bundle_dir / MODEL_FILE, "ab") as f:
        f.write(b"x")


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setattr(settings, "device", "cpu")
    monkeypatch.setattr(settings, "use_mock_models", False)


@pytest.fixture()
def separate_roots(tmp_path, monkeypatch):
    """BUNDLE.lock in its own folder, so its root does not contain the runs."""
    monkeypatch.setattr(settings, "landmark_bundle_lock", str(tmp_path / "models" / "BUNDLE.lock"))


# -- discovery -------------------------------------------------------------------


def test_discover_finds_bundles_under_the_training_runs(make_tiny_bundle, separate_roots):
    run_bundle = Path(settings.runs_dir) / "stageB_x_s0" / "all_dev" / "bundle" / "lm-0.2.0"
    run_bundle.parent.mkdir(parents=True)
    shutil.move(make_tiny_bundle("lm-0.2.0"), run_bundle)

    (found,) = registry.discover()
    assert found.version == "lm-0.2.0" and found.source == "runs" and found.servable
    assert found.path == run_bundle
    summary = found.summary()
    assert summary["arch"] == "tiny_test_cnn" and summary["input_size"] == 224
    assert summary["size_bytes"] > 0 and summary["modified_at"]
    assert len(summary["tau"]) == 10 and summary["q_min"] == pytest.approx(0.2)
    json.dumps(summary, allow_nan=False)


def test_discover_sorts_newest_version_first_and_releases_above_prereleases(make_tiny_bundle):
    for version in ("lm-0.9.0", "lm-0.10.0", "lm-1.0.0-rc1", "lm-1.0.0"):
        make_tiny_bundle(version)
    assert [b.version for b in registry.discover()] == ["lm-1.0.0", "lm-1.0.0-rc1", "lm-0.10.0", "lm-0.9.0"]


def test_a_bundle_this_code_cannot_serve_is_listed_with_the_reason(make_tiny_bundle):
    old = make_tiny_bundle("lm-0.1.0")
    _edit_meta(old, lambda m: m.update(layouts_version="lay-0"))
    make_tiny_bundle("lm-0.2.0")

    by_version = {b.version: b for b in registry.discover()}
    assert by_version["lm-0.2.0"].servable
    assert not by_version["lm-0.1.0"].servable
    assert by_version["lm-0.1.0"].problem.startswith("layouts_version 'lay-0' != ")
    assert str(old) not in by_version["lm-0.1.0"].problem  # the path prefix is dropped


def test_unreadable_metadata_is_listed_under_the_folder_name(make_tiny_bundle):
    broken = make_tiny_bundle("lm-0.1.0")
    (broken / META_FILE).write_text("{not json")
    (found,) = registry.discover()
    assert found.version == "lm-0.1.0" and not found.servable and "unreadable" in found.problem


def test_a_version_found_twice_keeps_the_higher_priority_root(make_tiny_bundle, separate_roots):
    in_models = Path(settings.landmark_bundle_lock).parent / "lm-0.1.0"
    in_runs = Path(settings.runs_dir) / "exp" / "fold0" / "bundle" / "lm-0.1.0"
    src = make_tiny_bundle("lm-0.1.0")
    shutil.copytree(src, in_models)
    shutil.copytree(src, in_runs)
    shutil.rmtree(src)

    (found,) = registry.discover()
    assert found.source == "models" and found.path == in_models


def test_folders_without_weights_and_eval_output_are_not_bundles(make_tiny_bundle):
    no_weights = make_tiny_bundle("lm-0.1.0")
    (no_weights / MODEL_FILE).unlink()
    evals = Path(settings.runs_dir) / "ui_evals" / "lm-9.9.9"
    evals.parent.mkdir(parents=True)
    shutil.copytree(make_tiny_bundle("lm-9.9.9"), evals)
    assert [b.version for b in registry.discover()] == ["lm-9.9.9"]
    assert registry.discover()[0].source == "models"  # the copy under ui_evals/ is skipped


def test_the_cache_root_follows_resolve_cache_dir_without_creating_anything(tmp_path, monkeypatch):
    wanted = tmp_path / "volume" / "landmarks"
    monkeypatch.setattr(settings, "landmark_cache_dir", str(wanted))
    assert registry._cache_root() == wanted and not wanted.exists()

    blocker = tmp_path / "a-file"
    blocker.write_text("")
    monkeypatch.setattr(settings, "landmark_cache_dir", str(blocker / "landmarks"))
    assert registry._cache_root().name == "peace-landmarks"


def test_find_returns_none_for_an_unknown_version(make_tiny_bundle):
    make_tiny_bundle("lm-0.1.0")
    assert registry.find("lm-0.1.0").version == "lm-0.1.0"
    assert registry.find("lm-9.9.9") is None


# -- which run a bundle came from ------------------------------------------------


@pytest.mark.parametrize(
    "train, want",
    [
        ({"mode": "fold", "fold": 3, "run": "whatever"}, {"mode": "fold", "fold": 3}),
        ({"run": "fold2"}, {"mode": "fold", "fold": 2}),
        ({"run": "all_dev"}, {"mode": "all_dev", "fold": None}),
        ({"run": "smoke"}, {"mode": "smoke", "fold": 0}),
        ({"run": "my-run"}, {"mode": None, "fold": None}),
        ({}, {"mode": None, "fold": None}),
        ({"mode": "fold", "fold": True}, {"mode": "fold", "fold": None}),
    ],
)
def test_training_split_reads_the_bundle_metadata(train, want):
    assert registry.training_split({"train": train}) == want


def test_training_split_falls_back_to_the_run_directory(tmp_path):
    bundle = tmp_path / "exp" / "custom" / "bundle" / "lm-0.1.0"
    bundle.mkdir(parents=True)
    (tmp_path / "exp" / "custom" / "metrics.json").write_text(json.dumps({"mode": "fold", "fold": 4}))
    assert registry.training_split({"train": {"run": "custom"}}, bundle) == {"mode": "fold", "fold": 4}
    assert registry.training_split({"train": {"run": "custom"}}) == {"mode": None, "fold": None}


# -- the saved selection ---------------------------------------------------------


def test_state_round_trip_and_removal():
    assert registry.read_state() == ServingState()
    registry.write_state(ServingState("lm-0.1.0", "/somewhere/lm-0.1.0", "shadow"))
    assert registry.read_state() == ServingState("lm-0.1.0", "/somewhere/lm-0.1.0", "shadow")
    assert registry.selected_bundle_dir() == Path("/somewhere/lm-0.1.0")
    registry.write_state(ServingState(mode="display"))
    assert registry.read_state() == ServingState(mode="display") and registry.selected_bundle_dir() is None
    registry.write_state(ServingState())
    assert not Path(settings.landmark_state_file).exists()


@pytest.mark.parametrize("text", ["{not json", "[1, 2]", '{"version": 3, "path": "/x", "mode": "loud"}'])
def test_a_bad_state_file_is_ignored(text):
    Path(settings.landmark_state_file).write_text(text)
    assert registry.read_state() == ServingState()


def test_state_is_ignored_when_model_admin_is_off(monkeypatch):
    registry.write_state(ServingState("lm-0.1.0", "/somewhere/lm-0.1.0", "display"))
    monkeypatch.setattr(settings, "model_admin", False)
    assert registry.read_state() == ServingState()
    assert registry.selected_bundle_dir() is None
    assert registry.apply_saved_mode() is None and registry.current_mode() == "off"


def test_modes_map_onto_the_two_landmark_flags():
    assert registry.current_mode() == "off"
    registry.apply_mode("shadow")
    assert (settings.landmarks_enabled, settings.landmarks_display) == (True, False)
    assert registry.current_mode() == "shadow"
    registry.apply_mode("display")
    assert (settings.landmarks_enabled, settings.landmarks_display) == (True, True)
    registry.apply_mode("off")
    assert not settings.landmarks_enabled and registry.current_mode() == "off"


def test_a_saved_mode_is_applied_at_startup():
    registry.write_state(ServingState(mode="display"))
    assert registry.apply_saved_mode() == "display"
    assert settings.landmarks_enabled and settings.landmarks_display


# -- serving the selection ---------------------------------------------------------


def test_the_selection_outranks_the_env_bundle_dir(make_tiny_bundle, monkeypatch):
    env, chosen = make_tiny_bundle("lm-0.1.0"), make_tiny_bundle("lm-0.2.0")
    monkeypatch.setattr(settings, "landmark_bundle_dir", str(env))
    assert _bundle_path_from_settings() == env
    registry.write_state(ServingState("lm-0.2.0", str(chosen)))
    assert _bundle_path_from_settings() == chosen
    assert LandmarkModelManager.get_instance().version == "lm-0.2.0"


def test_activate_swaps_the_model_and_leaves_open_sessions_on_the_old_one(make_tiny_bundle):
    first, second = make_tiny_bundle("lm-0.1.0", seed=1), make_tiny_bundle("lm-0.2.0", seed=2)
    old = LandmarkModelManager.activate(first)
    session = RealLandmarkClassifier(old)  # a live session opened before the switch

    new = LandmarkModelManager.activate(second)
    assert LandmarkModelManager.get_instance() is new and new.version == "lm-0.2.0"
    assert session.model_version == "lm-0.1.0" and session._manager is old


def test_a_failed_activation_keeps_the_current_model(make_tiny_bundle):
    good, bad = make_tiny_bundle("lm-0.1.0"), make_tiny_bundle("lm-0.2.0")
    _corrupt_weights(bad)
    current = LandmarkModelManager.activate(good)
    with pytest.raises(LandmarkUnavailable, match="sha256 mismatch"):
        LandmarkModelManager.activate(bad)
    assert LandmarkModelManager.peek() is current and LandmarkModelManager.failure() is None


def test_activation_clears_a_cached_failure(make_tiny_bundle):
    with pytest.raises(LandmarkUnavailable):
        LandmarkModelManager.get_instance()  # no lock, no bundle dir
    assert LandmarkModelManager.failure()
    LandmarkModelManager.activate(make_tiny_bundle("lm-0.1.0"))
    assert LandmarkModelManager.failure() is None
    assert LandmarkModelManager.get_instance().version == "lm-0.1.0"


# -- training runs -----------------------------------------------------------------


def _fold_run(exp: Path, fold: int, f1: float | None) -> None:
    run = exp / f"fold{fold}"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({
        "mode": "fold", "fold": fold, "best": {"variant": "ema", "step": 300},
        "oof": {"clip_macro_f1": f1},
    }))
    (run / "config.json").write_text(json.dumps({"candidate": "effb0_in1k", "input": {"size": 224}}))
    (run / f"oof_fold{fold}.npz").write_bytes(b"")


def test_experiments_summarise_finished_fold_runs():
    runs = Path(settings.runs_dir)
    exp = runs / "stageB_effb0_in1k_s0"
    _fold_run(exp, 0, 0.70)
    _fold_run(exp, 1, 0.80)
    _fold_run(exp, 2, None)  # a NaN metric is written as null
    (exp / "fold3").mkdir()  # still training: no metrics.json
    (exp / "calibration.json").write_text("{}")
    (exp / "gates_result.json").write_text(json.dumps({"outcome": "PARTIAL"}))
    (runs / "smoke_dev" / "smoke").mkdir(parents=True)  # no fold runs: not listed
    (runs / "stageA").mkdir()

    (found,) = registry.discover_experiments()
    assert found["name"] == "stageB_effb0_in1k_s0" and found["candidate"] == "effb0_in1k"
    assert [f["fold"] for f in found["folds"]] == [0, 1, 2] and all(f["has_oof"] for f in found["folds"])
    assert found["mean_clip_macro_f1"] == pytest.approx(0.75)
    assert found["has_calibration"] and not found["has_all_dev"] and not found["test_scored"]
    assert found["outcome"] == "PARTIAL"
    json.dumps(found, allow_nan=False)


def test_experiment_dir_only_resolves_listed_runs():
    exp = Path(settings.runs_dir) / "exp_a"
    _fold_run(exp, 0, 0.7)
    assert registry.experiment_dir("exp_a") == exp
    assert registry.experiment_dir("../exp_a") is None
    assert registry.experiment_dir("missing") is None
