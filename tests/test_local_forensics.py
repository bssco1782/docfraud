"""Tests for local-forensics-v1, the frozen split, augmentation integrity and the local model."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core import local_forensics as lf

ROOT = Path(__file__).parent.parent
SYNTHETIC = ROOT / "data" / "synthetic"
SPLIT_PATH = ROOT / "data" / "splits" / "day5_frozen_split.json"
AUG_MANIFEST = ROOT / "data" / "augmented" / "train_v2_1" / "manifest.csv"
LOCAL_MODEL = ROOT / "data" / "models" / "fraud_model_local_v1.pkl"
FIXTURES = ROOT / "tests" / "fixtures"

SPLIT = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
TRAIN_GENUINE = SYNTHETIC / next(r["path"] for r in SPLIT["train"] if r["label"] == "genuine")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def feats():
    return lf.extract_local_features(TRAIN_GENUINE)


# ── 1-4: extraction ──────────────────────────────────────────────────────────

def test_feature_dimension_and_order(feats):
    assert list(feats) == lf.FEATURE_NAMES
    assert len(lf.FEATURE_NAMES) == 52
    assert len(set(lf.FEATURE_NAMES)) == 52
    assert lf.FEATURE_SCHEMA_VERSION == "local-forensics-v1"


def test_features_finite_floats(feats):
    for k, v in feats.items():
        assert isinstance(v, float) and math.isfinite(v), k


def test_extraction_deterministic(feats):
    assert lf.extract_local_features(TRAIN_GENUINE) == feats


@pytest.mark.parametrize("name", ["clean_rgb.jpg", "grayscale.jpg", "small_image.jpg", "large_image.jpg"])
def test_extraction_on_fixtures(name):
    f = lf.extract_local_features(FIXTURES / name)
    assert len(f) == len(lf.FEATURE_NAMES)
    assert all(math.isfinite(v) for v in f.values())


def test_corrupt_file_raises():
    with pytest.raises(ValueError):
        lf.extract_local_features(FIXTURES / "corrupt.jpg")


def test_exif_orientation_applied(tmp_path):
    img = Image.new("RGB", (120, 60), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    p = tmp_path / "rot.jpg"
    img.save(p, exif=exif)
    assert lf.load_oriented(p).size == (60, 120)


# ── 5: anomaly calculation ──────────────────────────────────────────────────

def test_robust_z_flags_outlier_only():
    v = np.zeros((10, 10))
    v[4, 4] = 10.0
    z = lf.robust_z(v, floor=0.5)
    assert z[4, 4] == lf.Z_CLIP or z[4, 4] >= 19.9
    assert z.sum() == z[4, 4]


def test_robust_z_ignores_nan():
    v = np.full((4, 4), np.nan)
    v[0, :] = [1, 1, 1, 9]
    z = lf.robust_z(v, floor=1.0)
    assert np.isfinite(z).all() and z[1:, :].sum() == 0 and z[0, 3] > 0


NOISE_BOX = (300, 400, 160, 40)


def _noise_edit(tmp_path) -> Path:
    src = Image.open(TRAIN_GENUINE).convert("RGB")
    a = np.asarray(src, dtype=np.float32).copy()
    x, y, w, h = NOISE_BOX
    a[y:y + h, x:x + w] += np.random.default_rng(0).normal(0, 12, size=(h, w, 1))
    edited = tmp_path / "edited.png"
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(edited)
    return edited


def _overlaps(region) -> bool:
    x, y, w, h = NOISE_BOX
    rx, ry, rw, rh = region
    return rx < x + w and x < rx + rw and ry < y + h and y < ry + rh


def test_local_noise_edit_is_detected_and_localised(tmp_path):
    """A noise patch on one field must raise the noise anomaly and be reported at its location."""
    edited = _noise_edit(tmp_path)
    f_src, f_edit = lf.extract_local_features(TRAIN_GENUINE), lf.extract_local_features(edited)
    assert f_edit["lf_noise_max"] > f_src["lf_noise_max"] + 4

    hits = [r for r in lf.localize(edited) if _overlaps(r["region"]) and "noise_anomaly" in r["signals"]]
    assert hits, "edit not reported as a noise-anomaly region"
    assert not any(_overlaps(r["region"]) and "noise_anomaly" in r["signals"] for r in lf.localize(TRAIN_GENUINE))
    for r in hits:
        assert 0.0 <= r["anomaly_score"] <= 1.0
        assert r["severity"] in ("low", "medium", "high")


@pytest.mark.xfail(strict=True, reason="Known limitation (Day 5 CV report): fixed template elements "
                                       "(borders, coloured bands) outrank weak edits in region ranking.")
def test_local_noise_edit_is_top_ranked_region(tmp_path):
    assert _overlaps(lf.localize(_noise_edit(tmp_path))[0]["region"])


def test_uniform_global_change_is_not_local_anomaly(tmp_path):
    """Brightening the whole page must not create a localised brightness anomaly."""
    src = Image.open(TRAIN_GENUINE).convert("RGB")
    a = np.clip(np.asarray(src, dtype=np.int16) - 10, 0, 255).astype(np.uint8)
    p = tmp_path / "darker.png"
    Image.fromarray(a).save(p)
    f_src, f_dark = lf.extract_local_features(TRAIN_GENUINE), lf.extract_local_features(p)
    assert abs(f_dark["lf_intensity_anom_frac"] - f_src["lf_intensity_anom_frac"]) < 0.02


# ── 8-10: data integrity / leakage ──────────────────────────────────────────

def test_frozen_test_set_unchanged():
    assert len(SPLIT["test"]) == 15
    assert {r["label"] for r in SPLIT["test"]} == {"genuine", "tampered"}
    for r in SPLIT["test"]:
        p = SYNTHETIC / r["path"]
        assert p.exists(), r["path"]
        assert _sha(p) == r["sha256"], f"frozen test image changed: {r['path']}"
        assert r["path"].split("/")[0] == r["label"]


def test_split_disjoint_and_complete():
    train = {r["path"] for r in SPLIT["train"]}
    test = {r["path"] for r in SPLIT["test"]}
    assert not train & test
    on_disk = {p.relative_to(SYNTHETIC).as_posix() for d in ("genuine", "tampered") for p in (SYNTHETIC / d).glob("*.png")}
    assert train | test == on_disk


def test_augmentation_uses_train_genuine_sources_only():
    rows = list(csv.DictReader(open(AUG_MANIFEST, newline="", encoding="utf-8")))
    test = {r["path"] for r in SPLIT["test"]}
    train_genuine = {r["path"]: r["sha256"] for r in SPLIT["train"] if r["label"] == "genuine"}
    assert rows
    for r in rows:
        assert r["source_path"] not in test
        assert train_genuine.get(r["source_path"]) == r["source_sha256"]
        assert r["label"] in ("genuine", "tampered")
        assert r["transform"] and r["seed"] and r["sha256"]
        if r["label"] == "tampered":
            assert len(json.loads(r["bbox"])) == 4


def test_no_label_leakage_in_feature_names():
    from core.forensic_model import COMBINED_FEATURE_NAMES
    banned = ("label", "genuine", "tampered", "file", "path", "name", "doc_type", "index")
    for n in COMBINED_FEATURE_NAMES:
        assert not any(b in n.lower() for b in banned), n


def test_features_do_not_depend_on_filename(tmp_path, feats):
    renamed = tmp_path / "tampered_FAKE_copy.png"
    renamed.write_bytes(TRAIN_GENUINE.read_bytes())
    assert lf.extract_local_features(renamed) == feats


# ── 6-7: model save/load + prediction compatibility ─────────────────────────

@pytest.mark.skipif(not LOCAL_MODEL.exists(), reason="fraud_model_local_v1.pkl not trained yet")
def test_local_model_loads_and_predicts():
    from core.forensic_model import ForensicModel
    fm = ForensicModel.load(LOCAL_MODEL)
    md = fm.metadata
    for key in ("feature_names", "feature_schema_version", "training_seed", "training_dataset_sha256",
                "training_image_count", "rf_params", "cv_metrics", "created_at", "model_version"):
        assert key in md, key
    assert fm.model.n_features_in_ == len(md["feature_names"])
    out = fm.analyze(TRAIN_GENUINE)
    assert out["processing_status"] == "ok"
    assert out["tamper_classification"] in ("tampering_evidence", "no_tampering_evidence")
    assert 0.0 <= out["model_score"] <= 1.0
    assert isinstance(out["suspicious_regions"], list)
    assert "probability" not in out["summary"].lower()


@pytest.mark.skipif(not LOCAL_MODEL.exists(), reason="fraud_model_local_v1.pkl not trained yet")
def test_local_model_reports_error_status_on_corrupt_file():
    from core.forensic_model import ForensicModel
    out = ForensicModel.load(LOCAL_MODEL).analyze(FIXTURES / "corrupt.jpg")
    assert out["processing_status"] == "error"


def test_day5_model_preserved():
    """The Day 5 model (v1) must stay available (historical evidence and rollback).

    Since Day 8 the deployed fraud_model.pkl is model v2; v1 is kept byte-for-byte as
    fraud_model_v1.pkl (the SHA-256 recorded when the demo set was first scored).
    """
    import hashlib
    import pickle
    v1 = ROOT / "data" / "models" / "fraud_model_v1.pkl"
    assert hashlib.sha256(v1.read_bytes()).hexdigest() == (
        "bcf4d02f15980f68efc63c5c6089c91118be1078bd12f1df65dd81f25e55c6a3")
    bundle = pickle.loads(v1.read_bytes())
    assert bundle["metadata"]["feature_schema_version"] == "day4-v2-22"
    assert bundle["metadata"]["evaluation"]["test_files"] == [r["path"] for r in SPLIT["test"]]
