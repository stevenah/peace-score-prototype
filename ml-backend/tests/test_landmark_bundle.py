"""Landmark bundles: export/load round trip, refusal on any mismatch, the
S3 fetch cache, and the LandmarkModelManager singleton."""

from __future__ import annotations

import json
import shutil
import tarfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.config import settings  # noqa: E402
from app.ml.landmarks import bundle as bundle_mod  # noqa: E402
from app.ml.landmarks.architectures import ARCHS, build  # noqa: E402
from app.ml.landmarks.bundle import (  # noqa: E402
    DEFAULT_TAU,
    META_FILE,
    MODEL_FILE,
    SOURCE_FILE,
    BundleLock,
    LandmarkUnavailable,
    fetch_bundle,
    load_bundle,
    pack_bundle,
    read_lock,
    resolve_cache_dir,
    selftest_input,
    sha256_file,
    write_bundle,
    write_lock,
)
from app.ml.landmarks.classifier import LandmarkModelManager  # noqa: E402
from app.ml.landmarks.quality import QualityParams  # noqa: E402
from app.ml.landmarks.stations import NUM_STATIONS, STATION_ORDER  # noqa: E402
from app.ml.landmarks.tracker import TrackerParams  # noqa: E402
from app.ml.real_models import ModelManager  # noqa: E402


def _edit_meta(bundle_dir: Path, fn) -> None:
    meta = json.loads((bundle_dir / META_FILE).read_text())
    fn(meta)
    (bundle_dir / META_FILE).write_text(json.dumps(meta))


# -- architectures -------------------------------------------------------------


_SLOW = pytest.mark.slow  # torchvision/timm model construction takes seconds


@pytest.mark.parametrize(
    "arch",
    ["tiny_test_cnn"] + [pytest.param(a, marks=_SLOW) for a in ("effb0", "r50", "regy32", "cnxt_t", "dinov2_s14")],
)
def test_build_returns_logits_and_embedding(arch):
    if arch == "dinov2_s14":
        pytest.importorskip("timm")
    net = build(arch)
    with torch.inference_mode():
        logits, emb = net(torch.zeros(2, 3, 64, 64).contiguous(memory_format=torch.channels_last))
    assert logits.shape == (2, NUM_STATIONS)
    assert emb.shape == (2, net.embedding_dim)


def test_build_rejects_unknown_arch():
    assert "dinov2_s14" in ARCHS
    with pytest.raises(ValueError):
        build("vgg11")


# -- round trip ----------------------------------------------------------------


def test_round_trip_loads_verified_bundle(make_tiny_bundle):
    tracker = TrackerParams(evidence_needed=2, evidence_window=5)
    auto = [True] * NUM_STATIONS
    auto[8] = False
    path = make_tiny_bundle(
        temperature=1.5,
        tau=[0.6] * NUM_STATIONS,
        tracker=tracker.to_dict(),
        auto_enabled=auto,
        quality={**QualityParams(sharp_ref=900.0).to_dict(), "q_min": 0.3},
    )
    meta = json.loads((path / META_FILE).read_text())
    assert meta["stations"] == list(STATION_ORDER)
    assert meta["files"][MODEL_FILE] == sha256_file(path / MODEL_FILE)
    assert len(meta["selftest"]["logits"]) == NUM_STATIONS

    b = load_bundle(path)
    assert (b.version, b.arch, b.input_size) == ("lm-0.0.1", "tiny_test_cnn", 224)
    assert b.temperature == 1.5 and b.tau == (0.6,) * NUM_STATIONS
    assert b.tracker == tracker and b.auto_enabled == tuple(auto)
    assert b.quality.sharp_ref == 900.0 and b.q_min == 0.3
    assert b.serve_layouts == ("olympus_1920x1080",)
    with torch.inference_mode():
        logits, _ = b.model(selftest_input(meta["selftest"]["seed"], 224))
    assert np.allclose(logits[0].numpy(), meta["selftest"]["logits"], rtol=1e-5, atol=1e-5)


def test_selftest_input_is_deterministic_and_bounded():
    a, b = selftest_input(7, 16), selftest_input(7, 16)
    assert torch.equal(a, b) and a.dtype == torch.float32 and a.shape == (1, 3, 16, 16)
    assert float(a.min()) >= -1.0 and float(a.max()) < 1.0
    assert not torch.equal(a, selftest_input(8, 16))


def test_write_bundle_refuses_derived_fields_and_bad_versions(tmp_path):
    net = build("tiny_test_cnn")
    with pytest.raises(ValueError):
        write_bundle(tmp_path / "a", net, {"version": "lm-1.0.0", "stations": ["x"]})
    with pytest.raises(ValueError):
        write_bundle(tmp_path / "b", net, {"version": "1.0.0"})


# -- refusal on mismatch -------------------------------------------------------


def _set(key, value):
    return lambda m: m.__setitem__(key, value)


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(_set("schema", "esge11.v1"), id="schema"),
        pytest.param(lambda m: m["stations"].reverse(), id="station-order"),
        pytest.param(_set("preprocess_version", "lmpre-0"), id="preprocess-version"),
        pytest.param(_set("layouts_version", "lay-0"), id="layouts-version"),
        pytest.param(_set("version", "v1"), id="version-format"),
        pytest.param(_set("arch", "vgg11"), id="arch"),
        pytest.param(_set("mean", [0.5, 0.5, 0.5]), id="mean"),
        pytest.param(_set("tau", [0.5] * 9), id="tau-length"),
        pytest.param(_set("tau", [0.0] * NUM_STATIONS), id="tau-range"),
        pytest.param(_set("temperature", 0), id="temperature"),
        pytest.param(_set("auto_enabled", [1] * NUM_STATIONS), id="auto-enabled-type"),
        pytest.param(_set("serve_layouts", ["legacy_1350x1080"]), id="unserved-layout"),
        pytest.param(_set("rejection", {"method": "knn"}), id="rejection"),
        pytest.param(lambda m: m["selftest"]["logits"].__setitem__(0, 99.0), id="selftest-logits"),
        pytest.param(lambda m: m["files"].__setitem__(MODEL_FILE, "0" * 64), id="sha256"),
        pytest.param(lambda m: m["files"].__setitem__("../model.pt", "0" * 64), id="file-name"),
        pytest.param(lambda m: m.pop("files"), id="no-files"),
    ],
)
def test_load_refuses_tampered_meta(tiny_bundle, tamper):
    _edit_meta(tiny_bundle, tamper)
    with pytest.raises(LandmarkUnavailable):
        load_bundle(tiny_bundle)


def _tracker(**kw):
    return lambda m: m["tracker"].update(kw)


def _quality(**kw):
    return lambda m: m["quality"].update(kw)


@pytest.mark.parametrize(
    "tamper",
    [
        # JSON floats/bools would build a tracker that raises on every socket.
        pytest.param(_tracker(evidence_window=6.0), id="tracker-float-count"),
        pytest.param(_tracker(evidence_needed=True), id="tracker-bool-count"),
        pytest.param(_tracker(current_release=0), id="tracker-zero-count"),
        pytest.param(_tracker(evidence_window=-1), id="tracker-negative-window"),
        pytest.param(_tracker(evidence_needed=7, evidence_window=6), id="tracker-needed-above-window"),
        pytest.param(_tracker(current_k=4, current_window=3), id="tracker-k-above-window"),
        pytest.param(_tracker(best_margin=0.9), id="tracker-margin-below-1"),
        pytest.param(_tracker(best_margin=float("nan")), id="tracker-margin-nan"),
        pytest.param(_tracker(best_cooldown=-1), id="tracker-negative-cooldown"),
        pytest.param(_tracker(evidence_windw=8), id="tracker-unknown-key"),
        # A float kernel size loads but makes cv2 raise on every frame.
        pytest.param(_quality(edge_erode_px=4.0), id="quality-float-erode"),
        pytest.param(_quality(edge_erode_px=-1), id="quality-negative-erode"),
        pytest.param(_quality(edge_erode_px=10**6), id="quality-huge-erode"),
        pytest.param(_quality(sharp_ref=0), id="quality-zero-sharp-ref"),
        pytest.param(_quality(sharp_ref=float("inf")), id="quality-inf-sharp-ref"),
        pytest.param(_quality(sharp_ref="1100"), id="quality-string"),
        pytest.param(_quality(dark_max=1.5), id="quality-fraction-range"),
        pytest.param(_quality(sat_y=300.0), id="quality-luminance-range"),
        pytest.param(_quality(q_min=float("nan")), id="quality-nan-q-min"),
        pytest.param(_quality(sharp_rf=1100.0), id="quality-unknown-key"),
    ],
)
def test_load_refuses_unservable_tracker_or_quality_params(tiny_bundle, tamper):
    _edit_meta(tiny_bundle, tamper)
    with pytest.raises(LandmarkUnavailable, match="tracker|quality"):
        load_bundle(tiny_bundle)


def test_load_accepts_edge_values_and_fills_missing_params_with_defaults(tiny_bundle):
    def edit(m):
        m["tracker"] = {"evidence_needed": 6, "evidence_window": 6, "best_cooldown": 0, "best_margin": 1}
        m["quality"] = {"sharp_ref": 900, "edge_erode_px": 0, "dark_max": 1.0}

    _edit_meta(tiny_bundle, edit)
    b = load_bundle(tiny_bundle)
    assert b.tracker == TrackerParams(evidence_needed=6, evidence_window=6, best_cooldown=0, best_margin=1)
    assert b.quality == QualityParams(sharp_ref=900, edge_erode_px=0, dark_max=1.0)
    assert b.q_min == bundle_mod.DEFAULT_Q_MIN


def test_load_refuses_modified_weights(tiny_bundle):
    with open(tiny_bundle / MODEL_FILE, "ab") as f:
        f.write(b"\0")
    with pytest.raises(LandmarkUnavailable, match="sha256"):
        load_bundle(tiny_bundle)


def test_load_refuses_weights_of_another_shape_even_with_matching_sha(tiny_bundle):
    # A state_dict that does not fit the arch fails strict loading.
    torch.save({"head.weight": torch.zeros(3, 3)}, tiny_bundle / MODEL_FILE)
    _edit_meta(tiny_bundle, lambda m: m["files"].__setitem__(MODEL_FILE, sha256_file(tiny_bundle / MODEL_FILE)))
    with pytest.raises(LandmarkUnavailable):
        load_bundle(tiny_bundle)


def test_load_refuses_missing_bundle(tmp_path):
    with pytest.raises(LandmarkUnavailable, match="meta.json"):
        load_bundle(tmp_path / "nope")


# -- LandmarkModelManager singleton -----------------------------------------------


def test_manager_is_independent_of_the_peace_singleton(tiny_bundle):
    assert not issubclass(LandmarkModelManager, ModelManager)
    peace_before = ModelManager._instance
    m1 = LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")
    m2 = LandmarkModelManager.get_instance()
    assert m1 is m2 and m1.version == "lm-0.0.1"
    assert ModelManager._instance is peace_before
    assert LandmarkModelManager.peek() is m1
    LandmarkModelManager.reset()
    assert LandmarkModelManager.peek() is None and ModelManager._instance is peace_before


def test_manager_caches_failure_until_reset(tmp_path, tiny_bundle, monkeypatch):
    calls = []
    real_load = bundle_mod.load_bundle

    def counting_load(path, device="cpu"):
        calls.append(path)
        return real_load(path, device)

    monkeypatch.setattr("app.ml.landmarks.classifier.load_bundle", counting_load)
    with pytest.raises(LandmarkUnavailable):
        LandmarkModelManager.get_instance(bundle_dir=tmp_path / "missing", device="cpu")
    # Even a good bundle is not retried: the failure is cached.
    with pytest.raises(LandmarkUnavailable):
        LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")
    assert len(calls) == 1 and LandmarkModelManager.failure()
    LandmarkModelManager.reset()
    assert LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu").version == "lm-0.0.1"
    assert len(calls) == 2


def test_manager_loads_once_under_concurrency(tiny_bundle, monkeypatch):
    calls = []
    real_load = bundle_mod.load_bundle

    def slow_load(path, device="cpu"):
        calls.append(path)
        time.sleep(0.05)
        return real_load(path, device)

    monkeypatch.setattr("app.ml.landmarks.classifier.load_bundle", slow_load)
    got = []
    threads = [threading.Thread(target=lambda: got.append(
        LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu"))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len({id(m) for m in got}) == 1


def test_get_instance_warms_up_the_full_live_path(tiny_bundle, monkeypatch):
    from app.ml.landmarks import classifier as clf_mod

    seen = []
    real = clf_mod.quality_metrics

    def spy(img, layout, params):
        seen.append((img.shape, layout.name, params))
        return real(img, layout, params)

    monkeypatch.setattr(clf_mod, "quality_metrics", spy)
    manager = LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")
    # One synthetic frame per served layout, with the bundle's own quality params.
    assert seen == [((224, 224, 3), "olympus_1920x1080", manager.bundle.quality)]


def test_a_bundle_whose_live_path_fails_is_never_published(tiny_bundle, monkeypatch):
    def broken(*args, **kwargs):
        raise TypeError("Can't parse 'ksize'")

    monkeypatch.setattr("app.ml.landmarks.classifier.quality_metrics", broken)
    with pytest.raises(LandmarkUnavailable, match="ksize"):
        LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")
    # Not loaded (so /health says so), and the failure is cached for every socket.
    assert LandmarkModelManager.peek() is None and "ksize" in LandmarkModelManager.failure()
    monkeypatch.undo()
    with pytest.raises(LandmarkUnavailable):
        LandmarkModelManager.get_instance(bundle_dir=tiny_bundle, device="cpu")


def test_manager_uses_settings(tiny_bundle, monkeypatch):
    # No lock file (conftest points the lock at a missing path): unavailable.
    with pytest.raises(LandmarkUnavailable, match="lock"):
        LandmarkModelManager.get_instance(device="cpu")
    LandmarkModelManager.reset()
    monkeypatch.setattr(settings, "landmark_bundle_dir", str(tiny_bundle))
    manager = LandmarkModelManager.get_instance(device="cpu")
    assert manager.version == "lm-0.0.1"
    logits = manager.forward(torch.zeros(3, 224, 224))
    assert logits.shape == (NUM_STATIONS,) and logits.dtype == np.float64


# -- BUNDLE.lock and the S3 cache ----------------------------------------------------


@pytest.fixture()
def packed(tiny_bundle, tmp_path):
    """(lock_path, tarball) for the tiny bundle, as export would produce."""
    tarball = tmp_path / "landmarks-lm-0.0.1.tar.gz"
    digest = pack_bundle(tiny_bundle, tarball)
    lock_path = tmp_path / "BUNDLE.lock"
    write_lock(lock_path, BundleLock("lm-0.0.1", "models/landmarks/landmarks-lm-0.0.1.tar.gz", digest))
    return lock_path, tarball


def _fake_s3(tarball: Path, calls: list):
    def download(key: str, local_path: str) -> bool:
        calls.append(key)
        shutil.copyfile(tarball, local_path)
        return True

    return download


def test_fetch_downloads_verifies_and_caches(packed, tmp_path):
    lock_path, tarball = packed
    cache, calls = tmp_path / "cache", []
    path = fetch_bundle(lock_path, cache, download=_fake_s3(tarball, calls))
    assert path == cache / "lm-0.0.1" and calls == ["models/landmarks/landmarks-lm-0.0.1.tar.gz"]
    assert load_bundle(path).version == "lm-0.0.1"
    assert sorted(p.name for p in cache.iterdir()) == ["lm-0.0.1"]  # no temp dirs left
    # Second fetch: verified cache hit, no download.
    assert fetch_bundle(lock_path, cache, download=_fake_s3(tarball, calls)) == path
    assert len(calls) == 1


def test_fetch_uses_the_s3_service_by_default(packed, tmp_path, monkeypatch):
    lock_path, tarball = packed
    calls: list = []
    monkeypatch.setattr("app.services.s3.download_from_s3", _fake_s3(tarball, calls))
    assert fetch_bundle(lock_path, tmp_path / "cache").name == "lm-0.0.1"
    assert len(calls) == 1


def test_fetch_redownloads_a_corrupt_cache(packed, tmp_path):
    lock_path, tarball = packed
    cache, calls = tmp_path / "cache", []
    path = fetch_bundle(lock_path, cache, download=_fake_s3(tarball, calls))
    (path / MODEL_FILE).write_bytes(b"corrupt")
    assert fetch_bundle(lock_path, cache, download=_fake_s3(tarball, calls)) == path
    assert len(calls) == 2 and load_bundle(path).version == "lm-0.0.1"


@pytest.mark.parametrize("tamper", ["no-marker", "legacy-marker", "other-lock", "meta-edited"])
def test_fetch_refetches_a_cache_not_bound_to_the_lock(packed, tmp_path, tamper):
    lock_path, tarball = packed
    cache, calls = tmp_path / "cache", []
    path = fetch_bundle(lock_path, cache, download=_fake_s3(tarball, calls))
    marker = path / SOURCE_FILE
    source = json.loads(marker.read_text())
    assert source["meta_sha256"] == sha256_file(path / META_FILE)
    if tamper == "no-marker":  # hand-copied or left-over directory
        marker.unlink()
    elif tamper == "legacy-marker":  # written before meta.json was bound
        marker.write_text(json.dumps({"s3_key": source["s3_key"], "sha256": source["sha256"]}))
    elif tamper == "other-lock":
        marker.write_text(json.dumps({**source, "sha256": "f" * 64}))
    if tamper in ("no-marker", "meta-edited"):
        _edit_meta(path, _set("tau", [0.01] * NUM_STATIONS))  # every frame would be "ok"
    assert fetch_bundle(lock_path, cache, download=_fake_s3(tarball, calls)) == path
    assert len(calls) == 2
    assert load_bundle(path).tau == (DEFAULT_TAU,) * NUM_STATIONS
    assert json.loads(marker.read_text()) == source


def test_fetch_fails_hard_on_sha_mismatch_and_leaves_nothing(packed, tmp_path):
    lock_path, tarball = packed
    lock = read_lock(lock_path)
    write_lock(lock_path, BundleLock(lock.version, lock.s3_key, "f" * 64))
    cache = tmp_path / "cache"
    with pytest.raises(LandmarkUnavailable, match="sha256"):
        fetch_bundle(lock_path, cache, download=_fake_s3(tarball, []))
    assert list(cache.iterdir()) == []


def test_fetch_fails_when_download_fails(packed, tmp_path):
    lock_path, _ = packed
    with pytest.raises(LandmarkUnavailable, match="download"):
        fetch_bundle(lock_path, tmp_path / "cache", download=lambda key, path: False)


def test_fetch_refuses_version_mismatch_between_lock_and_bundle(packed, tmp_path):
    lock_path, tarball = packed
    lock = read_lock(lock_path)
    write_lock(lock_path, BundleLock("lm-9.9.9", lock.s3_key, lock.sha256))
    with pytest.raises(LandmarkUnavailable, match="version"):
        fetch_bundle(lock_path, tmp_path / "cache", download=_fake_s3(tarball, []))


def test_fetch_refuses_path_traversal(tmp_path):
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "x.txt"
    payload.write_text("x")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../../escape.txt")
    lock_path = tmp_path / "BUNDLE.lock"
    write_lock(lock_path, BundleLock("lm-0.0.1", "k", sha256_file(evil)))
    with pytest.raises(LandmarkUnavailable, match="refusing"):
        fetch_bundle(lock_path, tmp_path / "cache", download=_fake_s3(evil, []))
    assert not (tmp_path.parent / "escape.txt").exists()


@pytest.mark.parametrize(
    "content",
    ["not json", json.dumps({"version": "lm-1.0.0"}),
     json.dumps({"version": "1.0", "s3_key": "k", "sha256": "0" * 64}),
     json.dumps({"version": "lm-1.0.0", "s3_key": "k", "sha256": "xyz"})],
)
def test_read_lock_rejects_malformed_locks(tmp_path, content):
    p = tmp_path / "BUNDLE.lock"
    p.write_text(content)
    with pytest.raises(LandmarkUnavailable):
        read_lock(p)


def test_cache_dir_falls_back_when_not_writable(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        got = resolve_cache_dir(ro / "models" / "landmarks")
        assert got != ro / "models" / "landmarks" and got.is_dir()
    finally:
        ro.chmod(0o700)
