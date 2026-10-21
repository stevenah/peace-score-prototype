"""The eval runner: jobs as subprocesses, their files, refusals and cancelling."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from app.config import settings  # noqa: E402
from app.ml.landmarks.bundle import META_FILE  # noqa: E402
from app.services import evals  # noqa: E402
from app.services.evals import KINDS, EvalError, EvalRunner, Kind  # noqa: E402

STUB = Kind(
    "stub", "Stub", "bundle", "A stand-in eval.", "tests.eval_stub",
    clean=lambda params: {"mode": params.get("mode", "ok")},
    args=lambda target, params, out: ["--out", str(out), "--target", target.name, "--mode", params["mode"]],
    result=lambda target, params, out: json.loads(out.read_text()),
    headline=lambda result: (f"value {result['value']}", "pass"),
)


def _wait(runner: EvalRunner, job_id: str, until=lambda job: job["status"] not in evals.ACTIVE,
          timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if until(job):
            return job
        time.sleep(0.05)
    raise AssertionError(f"eval {job_id} still {runner.get(job_id)['status']} after {timeout} s")


@pytest.fixture()
def runner(tmp_path):
    r = EvalRunner(tmp_path / "evals", {"stub": STUB})
    r.start()
    yield r
    r.stop()


@pytest.fixture()
def real_runner(tmp_path):
    r = EvalRunner(tmp_path / "evals")
    r.start()
    yield r
    r.stop()


@pytest.fixture()
def bundle(make_tiny_bundle):
    return make_tiny_bundle("lm-0.1.0")


# -- the job lifecycle -------------------------------------------------------------


def test_a_job_runs_and_keeps_its_result(runner, bundle):
    queued = runner.submit("stub", "lm-0.1.0")
    assert queued["status"] == "queued" and queued["target"] == {
        "type": "bundle", "id": "lm-0.1.0", "path": str(bundle)}
    assert "target_path" not in queued

    job = _wait(runner, queued["id"])
    assert job["status"] == "succeeded" and job["error"] is None
    assert job["headline"] == "value 42" and job["verdict"] == "pass"
    assert job["result"] == {"value": 42, "target": "lm-0.1.0"}
    assert job["progress"] == {"done": 2, "total": 2, "label": "steps"}
    assert job["started_at"] <= job["finished_at"]
    assert "starting" in job["log"] and "@@progress" not in job["log"]
    assert job["log"].startswith("$ python -m tests.eval_stub ")

    (listed,) = runner.list()
    assert listed["id"] == job["id"] and "result" not in listed and "target_path" not in listed


def test_a_failing_job_reports_its_last_log_line(runner, bundle):
    job = _wait(runner, runner.submit("stub", "lm-0.1.0", {"mode": "fail"})["id"])
    assert job["status"] == "failed" and job["error"] == "ValueError: it broke"
    assert job["result"] is None and job["headline"] is None


def test_a_job_that_writes_no_result_fails(runner, bundle):
    job = _wait(runner, runner.submit("stub", "lm-0.1.0", {"mode": "no_result"})["id"])
    assert job["status"] == "failed" and "result could not be read" in job["error"]


def test_a_running_job_can_be_cancelled(runner, bundle):
    job_id = runner.submit("stub", "lm-0.1.0", {"mode": "hang"})["id"]
    _wait(runner, job_id, lambda job: job["status"] == "running" and "starting" in job["log"])
    runner.cancel(job_id)
    job = _wait(runner, job_id, timeout=20)
    assert job["status"] == "cancelled" and job["finished_at"]


def test_jobs_run_one_at_a_time_and_a_queued_one_can_be_cancelled(runner, bundle):
    first = runner.submit("stub", "lm-0.1.0", {"mode": "hang"})["id"]
    second = runner.submit("stub", "lm-0.1.0")["id"]
    third = runner.submit("stub", "lm-0.1.0")["id"]
    _wait(runner, first, lambda job: job["status"] == "running")
    assert runner.get(second)["status"] == "queued"

    assert runner.cancel(second)["status"] == "cancelled"
    runner.cancel(first)
    assert _wait(runner, third)["status"] == "succeeded"
    assert runner.get(second)["status"] == "cancelled" and runner.get(second)["started_at"] is None
    assert [j["id"] for j in runner.list()] == [third, second, first]


def test_a_restart_fails_what_the_last_process_left_unfinished(tmp_path, bundle):
    root = tmp_path / "evals"
    first = EvalRunner(root, {"stub": STUB})  # never started: its job stays queued
    job_id = first.submit("stub", "lm-0.1.0")["id"]

    second = EvalRunner(root, {"stub": STUB})
    second.start()
    try:
        job = second.get(job_id)
        assert job["status"] == "failed" and job["error"].startswith("Interrupted")
    finally:
        second.stop()


def test_stopping_the_runner_ends_the_running_job(tmp_path, bundle):
    r = EvalRunner(tmp_path / "evals", {"stub": STUB})
    r.start()
    job_id = r.submit("stub", "lm-0.1.0", {"mode": "hang"})["id"]
    _wait(r, job_id, lambda job: job["status"] == "running")
    r.stop()
    job = r.get(job_id)
    assert job["status"] == "failed" and "stopped" in job["error"]


def test_old_finished_jobs_are_pruned(runner, bundle, monkeypatch):
    monkeypatch.setattr(evals, "KEEP_JOBS", 2)
    ids = []
    for _ in range(4):
        ids.append(runner.submit("stub", "lm-0.1.0")["id"])
        _wait(runner, ids[-1])
    kept = {j["id"] for j in runner.list()}
    assert ids[-1] in kept and ids[0] not in kept and len(kept) <= 3


def test_lookups_refuse_anything_that_is_not_a_job_id(runner, tmp_path):
    (tmp_path / "job.json").write_text(json.dumps({"id": "x", "status": "succeeded"}))
    assert runner.get("..") is None and runner.get("../..") is None
    assert runner.cancel("nope") is None


# -- refusals ----------------------------------------------------------------------


def test_submit_refuses_unknown_kinds_and_targets(real_runner, bundle):
    with pytest.raises(EvalError) as unknown_kind:
        real_runner.submit("score_the_test_set", "lm-0.1.0")
    assert unknown_kind.value.status == 404
    with pytest.raises(EvalError) as unknown_bundle:
        real_runner.submit("selftest", "lm-9.9.9")
    assert unknown_bundle.value.status == 404
    with pytest.raises(EvalError) as unknown_run:
        real_runner.submit("replay", "../../etc")
    assert unknown_run.value.status in (404, 409)  # 409 where the training extras are missing
    assert real_runner.list() == []


@pytest.mark.parametrize(
    "kind, params",
    [
        ("selftest", {"anything": 1}),
        ("latency", {"frames": 5}),
        ("latency", {"frames": 60.5}),
        ("latency", {"mode": "sustained", "frames": 60}),
        ("latency", {"mode": "sustained", "minutes": 300}),
        ("latency", {"mode": "sustained", "streams": True}),
        ("latency", {"mode": "burst"}),
        ("accuracy", {"split": "test"}),
        ("accuracy", {"split": "dev"}),
        ("accuracy", {"device": "cuda:0; rm -rf /"}),
        ("replay", {"transit": "model", "test": True}),
        ("oof_metrics", {"test": True}),
        ("gates_report", {"force_reason": "curious"}),
    ],
)
def test_bad_parameters_are_refused(kind, params):
    with pytest.raises(ValueError):
        KINDS[kind].clean(params)


def test_parameter_defaults():
    assert KINDS["latency"].clean({}) == {"mode": "screen", "frames": 60}
    assert KINDS["latency"].clean({"mode": "sustained"}) == {"mode": "sustained", "minutes": 2.0, "streams": 2}
    assert KINDS["accuracy"].clean({}) == {"split": "heldout", "device": "auto"}
    assert KINDS["replay"].clean({}) == {"transit": "placeholder"}


def test_no_eval_can_reach_the_locked_test_set(tmp_path):
    """Every command the runner can build, across every accepted parameter value."""
    choices = {
        "selftest": [{}],
        "latency": [{}, {"mode": "sustained"}],
        "accuracy": [{"split": s, "device": d} for s in ("heldout", "shift") for d in ("auto", "cpu")],
        "oof_metrics": [{}],
        "gates_report": [{}],
        "replay": [{"transit": t} for t in ("placeholder", "model")],
    }
    assert set(choices) == set(KINDS)
    for name, kind in KINDS.items():
        for params in choices[name]:
            args = kind.args(tmp_path / "target", kind.clean(params), tmp_path / "out.json")
            assert not {"--test", "--force-reason", "--predict", "--ci-sim", "--temporal-split"} & set(args), name
            assert "test" not in [a for a in args if not a.startswith(str(tmp_path))], name


def test_a_bundle_that_cannot_load_is_refused_except_for_the_selftest(real_runner, bundle):
    meta = json.loads((bundle / META_FILE).read_text())
    meta["layouts_version"] = "lay-0"
    (bundle / META_FILE).write_text(json.dumps(meta))

    with pytest.raises(EvalError) as refused:
        real_runner.submit("latency", "lm-0.1.0")
    assert refused.value.status == 409 and "layouts_version" in str(refused.value)

    job = _wait(real_runner, real_runner.submit("selftest", "lm-0.1.0")["id"])
    assert job["status"] == "succeeded" and job["verdict"] == "fail"
    assert job["headline"] == "Failed at metadata"
    assert [c["status"] for c in job["result"]["checks"]] == ["failed", "skipped", "skipped", "skipped"]


@pytest.mark.parametrize(
    "train, message",
    [
        ({"run": "all_dev"}, "trained on every dev patient"),
        ({"run": "handmade"}, "Cannot tell which dev fold"),
    ],
)
def test_held_out_accuracy_is_refused_without_a_held_out_fold(make_tiny_bundle, train, message):
    target = make_tiny_bundle("lm-0.3.0", train=train)
    assert message in KINDS["accuracy"].refuse(target, {"split": "heldout", "device": "auto"})
    assert KINDS["accuracy"].refuse(target, {"split": "shift", "device": "auto"}) is None
    fold = make_tiny_bundle("lm-0.4.0", train={"run": "fold1"})
    assert KINDS["accuracy"].refuse(fold, {"split": "heldout", "device": "auto"}) is None


def test_script_evals_need_a_calibrated_run(tmp_path):
    exp = tmp_path / "stageB_x_s0"
    exp.mkdir()
    for name in ("oof_metrics", "gates_report", "replay"):
        reason = KINDS[name].refuse(exp, KINDS[name].clean({}))
        assert "calibration.json" in reason and "training.landmarks.calibrate --exp runs/stageB_x_s0" in reason
    (exp / "calibration.json").write_text("{}")
    assert KINDS["replay"].refuse(exp, {"transit": "model"}) is None


def test_an_unavailable_kind_says_why(tmp_path, bundle):
    off = dataclasses.replace(STUB, unavailable=lambda: "Needs the frame cache.")
    r = EvalRunner(tmp_path / "evals", {"stub": off})
    with pytest.raises(EvalError) as refused:
        r.submit("stub", "lm-0.1.0")
    assert refused.value.status == 409 and str(refused.value) == "Needs the frame cache."
    assert evals.describe_kinds({"stub": off}) == [{
        "kind": "stub", "label": "Stub", "target": "bundle", "summary": "A stand-in eval.",
        "available": False, "reason": "Needs the frame cache."}]


def test_training_evals_are_unavailable_without_the_training_extras(monkeypatch):
    def no_pandas(name, *args, **kwargs):
        return None if name == "pandas" else real(name, *args, **kwargs)

    real = evals.importlib.util.find_spec
    monkeypatch.setattr(evals.importlib.util, "find_spec", no_pandas)
    kinds = {k["kind"]: k for k in evals.describe_kinds()}
    assert kinds["selftest"]["available"] and kinds["latency"]["available"]
    for name in ("accuracy", "oof_metrics", "gates_report", "replay"):
        assert not kinds[name]["available"] and "training" in kinds[name]["reason"]


# -- the real bundle evals, end to end -------------------------------------------------


def test_selftest_through_the_runner(real_runner, bundle):
    job = _wait(real_runner, real_runner.submit("selftest", "lm-0.1.0")["id"])
    assert job["status"] == "succeeded", job["log"]
    assert job["verdict"] == "pass" and job["headline"] == "All 4 checks passed"
    assert [c["name"] for c in job["result"]["checks"]] == ["meta", "files", "selftest", "live"]
    assert all(c["status"] == "passed" for c in job["result"]["checks"])


def test_latency_through_the_runner(real_runner, bundle):
    assert settings.use_mock_models  # conftest: the baseline is the mock PEACE pipeline
    job = _wait(real_runner, real_runner.submit("latency", "lm-0.1.0", {"frames": 10})["id"])
    assert job["status"] == "succeeded", job["log"]
    result = job["result"]
    assert result["baseline"]["config"] == "peace_only" and result["model"]["config"] == "lm-0.1.0"
    assert result["model"]["frames"] == 10 and result["model"]["landmark_ms"]["p50"] > 0
    assert result["baseline"]["landmark_ms"]["p50"] is None and result["peace"] == "mock"
    assert job["progress"] == {"done": 2, "total": 2, "label": "configurations"}
    assert "ms median" in job["headline"] and job["verdict"] is None


def test_results_from_a_run_directory(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    f1 = {"value": 0.81, "lo": 0.75, "hi": 0.86}
    (exp / "eval_oof.json").write_text(json.dumps({
        "exp": "exp", "folds": [0, 1], "clip": {"macro_f1": f1, "n_clips": 400, "per_class": {}, "confusion": []},
        "clip_merged": {"macro_f1": f1, "n_clips": 400, "per_class": {"big": 1}}, "by_mode": {"big": 1}}))
    oof = KINDS["oof_metrics"].result(exp, {}, tmp_path / "unused")
    assert "by_mode" not in oof and "per_class" not in oof["clip_merged"]
    assert KINDS["oof_metrics"].headline(oof) == ("Clip macro-F1 0.810 [0.750, 0.860] on 400 out-of-fold clips", None)

    (exp / "gates_result.json").write_text(json.dumps({
        "outcome": "NO-GO", "auto_enabled": [True] * 10,
        "gates": {"G-MODEL": {"status": "FAIL", "value": {"big": 1}}}}))
    (exp / "gates_report.md").write_text("# Gates report: exp\n")
    gates = KINDS["gates_report"].result(exp, {}, tmp_path / "unused")
    assert gates["gates"] == {"G-MODEL": {"status": "FAIL"}} and gates["report"].startswith("# Gates report")
    assert KINDS["gates_report"].headline(gates) == ("Outcome NO-GO", "fail")

    (exp / "stream_eval_model.json").write_text(json.dumps({
        "S2": {"recall_mean": 0.9}, "S4": {"fdr_overall": 0.05}}))
    replay = KINDS["replay"].result(exp, {"transit": "model"}, tmp_path / "unused")
    assert KINDS["replay"].headline(replay) == ("S2 recall 0.900, S4 false-detection rate 0.050", None)
    with pytest.raises(FileNotFoundError):
        KINDS["replay"].result(exp, {"transit": "placeholder"}, tmp_path / "unused")
