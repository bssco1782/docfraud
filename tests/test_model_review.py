"""Day 9 (Founder A): inference optimisation, feature importance, robustness and review helpers."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import feature_importance as fi  # noqa: E402
import model_review as mr  # noqa: E402
import model_robustness_test as rob  # noqa: E402

from backend.model_utils import ModelAPI  # noqa: E402
from core import tamper_features  # noqa: E402

MODEL_PATH = ROOT / "data" / "models" / "fraud_model.pkl"
PAGE = ROOT / "data" / "demo" / "genuine_payslip_01.png"


# ── Day-9 inference optimisation ────────────────────────────────────────────

def test_api_serves_single_threaded_forest_with_identical_outputs():
    api = ModelAPI(MODEL_PATH)
    api.load_model()
    assert api.model.n_jobs == 1
    saved = pickle.loads(MODEL_PATH.read_bytes())["model"]  # n_jobs as saved
    x = np.array([api.schema.vector(api.schema.extract(PAGE))])
    proba = saved.predict_proba(x)[0]
    out = api.predict(PAGE)
    assert out["tamper_risk_score"] == round(float(proba[1]), 4)
    assert out["confidence"] == round(float(proba[int(saved.predict(x)[0])]), 4)


# ── Robustness variants and summaries ───────────────────────────────────────

def test_variants_change_geometry_as_described(tmp_path):
    img = Image.new("RGB", (200, 100), "white")
    sizes = {}
    for name, (_, build) in rob.VARIANTS.items():
        with Image.open(build(img, tmp_path)) as out:
            sizes[name] = out.size
    assert sizes["original"] == (200, 100) and sizes["rotate_90"] == (100, 200)
    assert sizes["downscale_50"] == (100, 50) and sizes["down_up_50"] == (200, 100)
    assert sizes["crop_2px"] == (196, 96) and sizes["crop_10pct"] == (160, 80)
    assert sizes["skew_3deg"][0] > 200  # canvas expanded, white fill
    assert set(rob.VARIANTS) == set(rob.DESCRIPTION)


def test_summary_counts_flips_and_rates():
    def row(path, label, variant, score):
        return {"set": "T", "path": path, "label": label, "doc_type": "payslip", "method": "-",
                "variant": variant, "outcome": "scored", "score": score, "predicted": rob.label_for(score), "ms": 1.0}

    rows = [row("a", "genuine", "original", 0.1), row("a", "genuine", "crop_2px", 0.7),
            row("b", "tampered", "original", 0.9), row("b", "tampered", "crop_2px", 0.8)]
    s = rob.summarize(rows)
    assert s["original"]["accuracy"] == 1.0 and s["original"]["flips"] == 0
    assert s["crop_2px"]["fpr"] == 1.0 and s["crop_2px"]["fnr"] == 0.0 and s["crop_2px"]["flips"] == 1
    assert s["crop_2px"]["mean_abs_score_change"] == pytest.approx(0.35)
    assert rob.fragile_documents(rows)[0]["flips"] == ["crop_2px"]


# ── Feature importance helpers ──────────────────────────────────────────────

def test_every_model_feature_has_a_family():
    assert {fi.family_of(n) for n in tamper_features.FEATURE_NAMES} == set(fi.FAMILIES)


def test_correlated_pairs_ignores_constant_columns():
    rng = np.random.RandomState(0)
    X = rng.rand(60, 3)
    X[:, 1] = 2 * X[:, 0] + 1
    X[:, 2] = 5.0
    assert fi.correlated_pairs(X, ["a", "b", "c"]) == [("a", "b", pytest.approx(1.0))]


def test_partial_dependence_sign_follows_the_feature():
    rng = np.random.RandomState(1)
    X = rng.rand(300, 2)
    y = (X[:, 0] > 0.5).astype(int)  # only feature 0 matters, positively
    model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
    delta, lo, hi = fi.partial_dependence_delta(model, X)
    assert delta[0] > 0.5 and abs(delta[1]) < 0.2 and (hi > lo).all()


# ── Review helpers ──────────────────────────────────────────────────────────

def test_distribution_and_bar():
    assert mr.distribution([0.05, 0.15, 0.16, 0.95], [i / 10 for i in range(11)]) == [1, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    assert mr._bar(10, 10) == "#" * 40 and mr._bar(0, 10) == "" and mr._bar(3, 0) == ""
