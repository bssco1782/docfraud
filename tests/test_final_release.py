"""Day 10 (Founder A): final model validation, accuracy check, benchmark and sign-off helpers."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_model_validation as fmv  # noqa: E402
import inference_benchmark as ib  # noqa: E402
import model_accuracy_eval as mae  # noqa: E402
import model_stability_test as mst  # noqa: E402

from backend.model_utils import ModelAPI  # noqa: E402

MODEL_PATH = ROOT / "data" / "models" / "fraud_model.pkl"
MODEL_SHA = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()


def metrics(correct: int, n: int) -> dict:
    return {"accuracy": correct / n, "correct": correct, "n": n}


def accuracy_result(test=(13, 15), holdout=(138, 149), same=True) -> dict:
    return {"frozen_split_ok": True, "leakage_ok": True,
            "leakage": [("model records its training files", True, "840"), ("a", True, "0 of 15"), ("b", True, "0 of 149")],
            "sets": {"TEST": {"metrics": metrics(*test)}, "HOLDOUT": {"metrics": metrics(*holdout)}},
            "previous": [{"when": "Day 8 validation", "cm": [8, 2, 0, 5]}], "same_as_recorded": same}


# ── Accuracy check ──────────────────────────────────────────────────────────

def test_requirements_pass_for_the_recorded_v2_results():
    checks = mae.requirement_checks(accuracy_result())
    assert all(c["pass"] for c in checks), checks


@pytest.mark.parametrize("kwargs, failing", [
    ({"test": (11, 15)}, "TEST accuracy above 75%"),              # 73.3%
    ({"holdout": (115, 149)}, "HOLDOUT accuracy meets the 78% sprint target"),  # 77.2%
    ({"same": False}, "Metrics identical to the recorded v2 evaluation"),
    ({"same": None}, "Metrics identical to the recorded v2 evaluation"),
])
def test_requirements_fail_when_accuracy_drops_or_metrics_change(kwargs, failing):
    checks = {c["check"]: c["pass"] for c in mae.requirement_checks(accuracy_result(**kwargs))}
    assert checks[failing] is False


def test_recorded_v2_uses_only_records_of_the_same_model(tmp_path, monkeypatch):
    rec = {"model": {"sha256": "other"}, "generated": "2026-01-01",
           "heldout": {"sets": {"TEST": {"metrics": {"n": 15, "correct": 1, "accuracy": 0.1, "precision": 0,
                                                     "recall": 0, "f1": 0, "tn": 1, "fp": 0, "fn": 0, "tp": 0}}}}}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(rec))
    monkeypatch.setattr(mae, "RECORDED_V2", [("old", path, ("heldout", "sets"))])
    assert mae.recorded_v2(MODEL_SHA) == []
    rec["model"]["sha256"] = MODEL_SHA
    path.write_text(json.dumps(rec))
    monkeypatch.setattr(mae, "ROOT", tmp_path)
    assert [r["cm"] for r in mae.recorded_v2(MODEL_SHA)] == [[1, 0, 0, 0]]


def test_near_threshold_band():
    rows = [{"path": p, "actual": a, "predicted": d, "risk": s} for p, a, d, s in
            [("a", "genuine", "tampered", 0.50), ("b", "tampered", "genuine", 0.46), ("c", "genuine", "genuine", 0.44)]]
    nt = mae.near_threshold(rows, band=0.05)
    assert (nt["n"], nt["wrong"]) == (2, 2)


# ── Benchmark ───────────────────────────────────────────────────────────────

def test_stats_ms():
    s = ib.stats_ms([0.1, 0.2, 0.3, 0.4])
    assert (s["n"], s["min"], s["max"]) == (4, 100.0, 400.0)
    assert s["mean"] == pytest.approx(250.0) and s["median"] == pytest.approx(250.0)


def bench_result(**over) -> dict:
    ip = {"predictions": 30, "unstable": [], "crashes": [], "typical_max_ms": 300.0,
          "overall": {"median": 150.0, "p95": 200.0}, "run_to_run_cv": {"median": 0.03}}
    http = {"available": True, "mismatches": [], "errors": [],
            "concurrent": {"mean": 500.0, "p95": 700.0, "success_rate": 1.0, "succeeded": 50, "requests": 50}}
    r = {"in_process": ip, "http": http, "large": [{"error": None, "stable": True}]}
    for key, value in over.items():
        section, field = key.split("__")
        r[section][field] = value
    return r


def test_benchmark_targets_pass_and_fail():
    assert all(c["pass"] for c in ib.target_checks(bench_result()))
    cases = {
        "stable": bench_result(in_process__unstable=["doc.png"]),
        "no_crash": bench_result(http__errors=["HTTP 500"]),
        "latency": bench_result(in_process__typical_max_ms=2500.0),
        "consistent": bench_result(in_process__overall={"median": 100.0, "p95": 250.0}),
        "http": bench_result(http__available=False),
    }
    for target, r in cases.items():
        assert {c["target"]: c["pass"] for c in ib.target_checks(r)}[target] is False, target


# ── Final validation ────────────────────────────────────────────────────────

def test_dependency_checks_fail_only_for_a_scikit_learn_mismatch():
    meta = {"environment": {"scikit-learn": "0.0.1", "numpy": "0.0.1"}}
    status = {c["check"]: c["status"] for c in fmv.dependency_checks(meta)}
    assert status["scikit-learn matches the model file (0.0.1)"] == "FAIL"
    assert status["numpy matches the model file (0.0.1)"] == "WARN"


def test_installed_dependencies_match_the_model_file():
    meta = pickle_meta()
    checks = fmv.dependency_checks(meta)
    assert not fmv.failed(checks), fmv.failed(checks)


def pickle_meta() -> dict:
    import pickle

    return pickle.loads(MODEL_PATH.read_bytes())["metadata"]


def test_serialization_checks_pass_for_the_deployed_model():
    api = ModelAPI(MODEL_PATH)
    api.load_model()
    vectors = np.random.default_rng(0).random((4, len(api.schema.names)))
    checks = fmv.serialization_checks(api.model, api.metadata, vectors)
    assert not fmv.failed(checks), fmv.failed(checks)


def test_release_review_of_ela_and_thresholds():
    manifest, images = mst.load_demo()
    api = ModelAPI(MODEL_PATH)
    api.load_model()
    ela = fmv.review_ela_pipeline(api, images, manifest)
    thresholds = fmv.review_thresholds(api.metadata, manifest, images[0]["demo_file"],
                                       {"near_threshold": {"band": 0.05, "n": 0, "wrong": 0}})
    assert not fmv.failed(ela + thresholds), fmv.failed(ela + thresholds)


def test_recommendation_requires_every_item():
    items = [{"item": "a", "pass": True}, {"item": "b", "pass": True}]
    assert fmv.recommendation(items)["approved"] is True
    items[1]["pass"] = False
    rec = fmv.recommendation(items)
    assert rec["approved"] is False and rec["decision"].startswith("NOT APPROVED")


def test_demo_failure_text():
    tampered = {"image": "t.png", "risk": 0.15, "expected": "tampered", "edit": "combo:text_replace_amount+local_resample"}
    assert "not in model v2's training data" in fmv.demo_failure(tampered)
    assert "not in" not in fmv.demo_failure({**tampered, "edit": "noise_patch"})
    genuine = {"image": "g.png", "risk": 0.6, "expected": "genuine", "edit": "none"}
    assert "genuine page reported as tampered" in fmv.demo_failure(genuine)


# ── Committed sign-off matches the deployed model ───────────────────────────

def test_signoff_is_for_the_deployed_model():
    """Retraining or replacing the model file requires re-running scripts/final_model_validation.py."""
    results = json.loads(fmv.DEFAULT_JSON.read_text(encoding="utf-8"))
    assert results["model"]["sha256"] == MODEL_SHA
    assert results["recommendation"]["approved"] is True
    assert all(item["pass"] for item in results["signoff"])
    report = fmv.DEFAULT_REPORT.read_text(encoding="utf-8")
    assert all(section in report for section in fmv.SECTIONS)
    assert MODEL_SHA in fmv.LIMITATIONS.read_text(encoding="utf-8")
