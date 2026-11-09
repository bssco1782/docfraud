"""
tests/test_model.py
Founder C — Day 3: Automated model test suite.

Tests document analysis (ELA), feature extraction, and model prediction
using the real synthetic dataset generated on Days 1–2.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.document_analyzer import (
    FEATURE_NAMES,
    ela_analysis,
    extract_features,
    features_to_vector,
    load_image,
)

SYNTHETIC = Path(__file__).parent.parent / "data" / "synthetic"
GENUINE_DIR = SYNTHETIC / "genuine"
TAMPERED_DIR = SYNTHETIC / "tampered"
INDEX_CSV = SYNTHETIC / "index.csv"

GENUINE_IMAGES = sorted(GENUINE_DIR.glob("*.png")) if GENUINE_DIR.exists() else []
TAMPERED_IMAGES = sorted(TAMPERED_DIR.glob("*.png")) if TAMPERED_DIR.exists() else []
ALL_IMAGES = GENUINE_IMAGES + TAMPERED_IMAGES


def _load_index():
    if not INDEX_CSV.exists():
        return []
    with open(INDEX_CSV, newline="") as f:
        return list(csv.DictReader(f))


class TestElaOnSyntheticData:
    """TEST 1 — ELA analysis on real synthetic documents."""

    @pytest.mark.skipif(not ALL_IMAGES, reason="synthetic dataset not found")
    @pytest.mark.parametrize(
        "img_path",
        ALL_IMAGES[:6],
        ids=lambda p: p.name,
    )
    def test_ela_returns_heatmap_and_score(self, img_path):
        heatmap, score = ela_analysis(img_path)
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.dtype == np.uint8
        assert heatmap.ndim == 3 and heatmap.shape[2] == 3
        assert isinstance(score, float)
        assert math.isfinite(score)
        assert 0.0 <= score <= 100.0

    @pytest.mark.skipif(not ALL_IMAGES, reason="synthetic dataset not found")
    def test_ela_heatmap_matches_image_dimensions(self):
        img_path = ALL_IMAGES[0]
        img = load_image(img_path)
        heatmap, _ = ela_analysis(img_path)
        assert heatmap.shape[:2] == (img.size[1], img.size[0])

    @pytest.mark.skipif(not ALL_IMAGES, reason="synthetic dataset not found")
    def test_ela_is_deterministic(self):
        img_path = ALL_IMAGES[0]
        _, score1 = ela_analysis(img_path)
        _, score2 = ela_analysis(img_path)
        assert score1 == score2


class TestFeatureExtractionOnSyntheticData:
    """TEST 2 — Feature extraction on real synthetic documents."""

    @pytest.mark.skipif(not ALL_IMAGES, reason="synthetic dataset not found")
    @pytest.mark.parametrize(
        "img_path",
        ALL_IMAGES[:6],
        ids=lambda p: p.name,
    )
    def test_extract_features_returns_22_finite_floats(self, img_path):
        feats = extract_features(img_path)
        assert len(feats) == 22
        assert list(feats.keys()) == FEATURE_NAMES
        for k, v in feats.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"
            assert math.isfinite(v), f"{k} is not finite: {v}"

    @pytest.mark.skipif(not ALL_IMAGES, reason="synthetic dataset not found")
    def test_features_to_vector_ordering(self):
        feats = extract_features(ALL_IMAGES[0])
        vec = features_to_vector(feats)
        assert isinstance(vec, list)
        assert len(vec) == 22
        for i, name in enumerate(FEATURE_NAMES):
            assert vec[i] == feats[name]

    @pytest.mark.skipif(not ALL_IMAGES, reason="synthetic dataset not found")
    def test_feature_extraction_is_deterministic(self):
        img_path = ALL_IMAGES[0]
        feats1 = extract_features(img_path)
        feats2 = extract_features(img_path)
        for k in FEATURE_NAMES:
            assert feats1[k] == feats2[k], f"{k} not deterministic"

    @pytest.mark.skipif(
        not GENUINE_IMAGES or not TAMPERED_IMAGES,
        reason="need both genuine and tampered images",
    )
    def test_genuine_and_tampered_both_produce_valid_features(self):
        genuine_feats = extract_features(GENUINE_IMAGES[0])
        tampered_feats = extract_features(TAMPERED_IMAGES[0])
        for feats in (genuine_feats, tampered_feats):
            assert len(feats) == 22
            assert all(math.isfinite(v) for v in feats.values())

    @pytest.mark.skipif(not INDEX_CSV.exists(), reason="index.csv not found")
    def test_index_csv_rows_match_dataset(self):
        rows = _load_index()
        on_disk = {p.relative_to(SYNTHETIC).as_posix() for p in ALL_IMAGES}
        indexed = [row["path"] for row in rows]
        assert len(indexed) == len(set(indexed)), "duplicate paths in index.csv"
        assert set(indexed) == on_disk
        assert len(rows) >= 75
        for row in rows:
            assert row["doc_type"] in ("payslip", "bank_statement", "id_proof")
            assert row["label"] in ("genuine", "tampered")


MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "fraud_model.pkl"
_HAVE_MODEL = MODEL_PATH.exists()


class TestModelPrediction:
    """TEST 3 — Model prediction using Founder A's trained model."""

    @pytest.mark.skipif(not _HAVE_MODEL, reason="trained model not found at data/models/fraud_model.pkl")
    @pytest.mark.skipif(not ALL_IMAGES, reason="synthetic dataset not found")
    def test_model_prediction_on_synthetic_image(self):
        from core.model_trainer import DocumentFraudDetector

        detector = DocumentFraudDetector()
        model, metadata = detector.load(MODEL_PATH)
        detector.model = model

        result = detector.predict(ALL_IMAGES[0])
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "features" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0
        assert "genuine" in result["probabilities"]
        assert "tampered" in result["probabilities"]

    @pytest.mark.skipif(not _HAVE_MODEL, reason="trained model not found at data/models/fraud_model.pkl")
    @pytest.mark.skipif(not ALL_IMAGES, reason="synthetic dataset not found")
    def test_prediction_returns_valid_label(self):
        from core.model_trainer import DocumentFraudDetector

        detector = DocumentFraudDetector()
        model, metadata = detector.load(MODEL_PATH)
        detector.model = model

        result = detector.predict(GENUINE_IMAGES[0])
        assert result["prediction"] in ("genuine", "tampered")

        result2 = detector.predict(TAMPERED_IMAGES[0])
        assert result2["prediction"] in ("genuine", "tampered")
