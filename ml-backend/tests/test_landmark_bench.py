"""The Stage-0 latency benchmark runs end to end on synthetic frames."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

from app.ml.landmarks.bench import encode_jpeg, main, synthetic_olympus_frame  # noqa: E402
from app.ml.landmarks.layouts import OLYMPUS_1920, detect_layout  # noqa: E402


@pytest.mark.parametrize("width", [1920, 1280])
def test_synthetic_frames_pass_layout_detection_after_jpeg(width):
    frame = synthetic_olympus_frame(width, seed=1)
    decoded = np.asarray(Image.open(BytesIO(encode_jpeg(frame))).convert("RGB"))
    assert detect_layout(decoded) is OLYMPUS_1920
    assert not np.array_equal(frame, synthetic_olympus_frame(width, seed=2))


def test_screen_mode_reports_baseline_then_each_config(tmp_path, capsys):
    out = tmp_path / "bench.json"
    report = main(["--mode", "screen", "--archs", "tiny_test_cnn,no_such_arch", "--sizes", "64,96",
                   "--frames", "3", "--warmup", "1", "--peace", "mock", "--width", "1280",
                   "--out", str(out)])
    configs = [r["config"] for r in report["results"]]
    assert configs == ["peace_only", "tiny_test_cnn@64", "tiny_test_cnn@96"]
    assert report["skipped_archs"] == ["no_such_arch"]
    base, lm = report["results"][0], report["results"][1]
    assert base["frames"] == 3 and base["landmark_ms"]["p50"] is None
    assert lm["frames"] == 3 and lm["landmark_ms"]["p50"] > 0 and lm["rss_peak_mb"] > 0
    assert out.read_text().strip().startswith("{")
    assert "peace_only" in capsys.readouterr().out


def test_sustained_mode_runs_concurrent_streams():
    report = main(["--mode", "sustained", "--archs", "tiny_test_cnn", "--sizes", "64", "--frames", "2",
                   "--warmup", "0", "--rate", "0", "--streams", "2", "--peace", "mock", "--width", "1280"])
    assert all(r["streams"] == 2 and r["frames"] == 4 for r in report["results"])
