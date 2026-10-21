"""The bundle self-test: one check at a time, stopping at the first failure."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from app.ml.landmarks import selfcheck  # noqa: E402
from app.ml.landmarks.bundle import META_FILE, MODEL_FILE  # noqa: E402


def _statuses(result: dict) -> dict[str, str]:
    return {c["name"]: c["status"] for c in result["checks"]}


def test_a_good_bundle_passes_every_check(tiny_bundle):
    result = selfcheck.run(tiny_bundle)
    assert result["ok"] and result["version"] == "lm-0.0.1"
    assert _statuses(result) == {"meta": "passed", "files": "passed", "selftest": "passed", "live": "passed"}
    details = {c["name"]: c["detail"] for c in result["checks"]}
    assert "tiny_test_cnn at 224 px" in details["meta"]
    assert "sha256 match" in details["files"]
    assert details["selftest"].startswith("max |logit difference| ")
    assert details["live"].startswith("4 synthetic frames (") and " ms on CPU" in details["live"]
    json.dumps(result, allow_nan=False)


def test_changed_weights_fail_at_the_file_hashes(tiny_bundle):
    with open(tiny_bundle / MODEL_FILE, "ab") as f:
        f.write(b"x")
    result = selfcheck.run(tiny_bundle)
    assert not result["ok"]
    assert _statuses(result) == {"meta": "passed", "files": "failed", "selftest": "skipped", "live": "skipped"}
    assert result["checks"][1]["detail"] == "sha256 mismatch for model.pt"


def test_metadata_from_other_code_fails_first(tiny_bundle):
    meta = json.loads((tiny_bundle / META_FILE).read_text())
    meta["preprocess_version"] = "lmpre-0"
    (tiny_bundle / META_FILE).write_text(json.dumps(meta))
    result = selfcheck.run(tiny_bundle)
    assert _statuses(result)["meta"] == "failed" and result["version"] == "lm-0.0.1"
    assert result["checks"][0]["detail"].startswith("preprocess_version 'lmpre-0' != ")


def test_wrong_reference_logits_fail_the_selftest(tiny_bundle):
    meta = json.loads((tiny_bundle / META_FILE).read_text())
    meta["selftest"]["logits"][0] += 1.0
    (tiny_bundle / META_FILE).write_text(json.dumps(meta))
    result = selfcheck.run(tiny_bundle)
    assert _statuses(result) == {"meta": "passed", "files": "passed", "selftest": "failed", "live": "skipped"}
    assert "selftest logits differ" in result["checks"][2]["detail"]


def test_a_missing_folder_is_a_failed_check_not_a_crash(tmp_path):
    result = selfcheck.run(tmp_path / "lm-0.0.9")
    assert not result["ok"] and result["version"] == "lm-0.0.9"
    assert result["checks"][0] == {"name": "meta", "label": "Metadata", "status": "failed",
                                   "detail": "missing meta.json"}


def test_cli_writes_the_result_and_exits_zero_either_way(tiny_bundle, tmp_path, capsys):
    out = tmp_path / "result.json"
    assert selfcheck.main(["--bundle", str(tiny_bundle), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["ok"] is True
    assert "lm-0.0.1: OK" in capsys.readouterr().out

    (tiny_bundle / MODEL_FILE).unlink()
    assert selfcheck.main(["--bundle", str(tiny_bundle), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["ok"] is False
    assert "FAILED" in capsys.readouterr().out
