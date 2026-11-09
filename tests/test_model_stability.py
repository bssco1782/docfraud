"""Checks for scripts/model_stability_test.py and scripts/model_validation.py (Day 8, Founder A)."""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import model_stability_test as mst  # noqa: E402
import model_validation as mv  # noqa: E402

from backend import config  # noqa: E402
from backend.model_utils import ModelAPI  # noqa: E402


def test_summarize_identical_and_spread():
    same = mst.summarize([0.36] * 10)
    assert same["distinct"] == 1 and same["std"] == 0.0
    assert same["max_abs_dev"] <= 1e-12
    spread = mst.summarize([0.30, 0.34])
    assert math.isclose(spread["mean"], 0.32) and math.isclose(spread["max_abs_dev"], 0.02)
    assert spread["min"] == 0.30 and spread["max"] == 0.34 and spread["distinct"] == 2


def test_label_rule_matches_model_api(monkeypatch):
    monkeypatch.setattr(config, "TAMPER_THRESHOLD", 0.5)
    assert mst.label_for(0.5) == "tampered"
    assert mst.label_for(0.4999) == "genuine"


def test_compare_flags_mismatch_and_missing():
    baseline = {"a.png": (0.36, 0.64), "b.png": (0.59, 0.59)}
    ok = mst._compare("x", {"a.png": [(0.36, 0.64)], "b.png": [(0.59, 0.59)]}, baseline, "")
    assert ok["pass"] and not ok["mismatches"]
    bad = mst._compare("x", {"a.png": [(0.37, 0.63)]}, baseline, "")
    assert not bad["pass"]
    assert any("a.png" in m for m in bad["mismatches"]) and any("b.png: no result" in m for m in bad["mismatches"])


def test_drift_detects_score_change():
    manifest, images = mst.load_demo()
    recorded = {r["demo_file"]: (r["tamper_risk_score"], r["confidence"]) for r in images}
    sha = manifest["model"]["sha256"]
    assert mst.drift(images, recorded, manifest, sha)["pass"]
    moved = dict(recorded)
    first = images[0]["demo_file"]
    moved[first] = (recorded[first][0] + 0.05, recorded[first][1])
    assert not mst.drift(images, moved, manifest, sha)["pass"]
    assert not mst.drift(images, recorded, manifest, "0" * 64)["pass"]


def test_wilson_interval_known_values():
    lo, hi = mv.wilson_interval(9, 15)
    assert math.isclose(lo, 0.3575, abs_tol=1e-4) and math.isclose(hi, 0.8018, abs_tol=1e-4)
    assert mv.wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = mv.wilson_interval(5, 5)
    assert 0.0 <= lo < hi <= 1.0


def test_day6_record_parsed_from_report():
    rec = mv.day6_record()
    assert rec is not None
    assert (rec["correct"], rec["n"]) == (9, 15) and math.isclose(rec["accuracy"], 0.6)
    assert math.isclose(rec["recall"], 0.8) and math.isclose(rec["f1"], 0.5714)
    assert len(rec["per_image_risk"]) == 15


def test_bbox_fraction_from_manifest():
    _, images = mst.load_demo()
    by_name = {r["demo_file"]: r for r in images}
    assert math.isclose(mv.bbox_fraction(by_name["tampered_payslip_01.png"]), 55 * 12 / (850 * 1050))
    assert mv.bbox_fraction(by_name["genuine_payslip_01.png"]) is None


@pytest.mark.parametrize("runs", [3])
def test_repeatability_on_demo_image_is_exact(runs):
    """Real model, real demo image: identical output on every call, equal to the Day-7 record."""
    _, images = mst.load_demo()
    api = ModelAPI(mst.DEFAULT_MODEL_PATH)
    api.load_model()
    [entry] = mst.repeatability(api, images[:1], runs)
    assert entry["pass"] and entry["crashes"] == 0
    assert entry["score"]["distinct"] == 1 and entry["confidence"]["distinct"] == 1
    assert all(r["tamper_risk_score"] == images[0]["tamper_risk_score"] for r in entry["runs"])
