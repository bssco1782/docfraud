"""Tests for core.model_trainer — model training pipeline."""

from pathlib import Path

import numpy as np
import pytest

from core.document_analyzer import FEATURE_NAMES
from core.model_trainer import (
    DocumentFraudDetector,
)


@pytest.fixture
def detector():
    d = DocumentFraudDetector(random_state=42)
    d.load_dataset(use_fixtures_fallback=True)
    return d


class TestLoadDataset:
    def test_loads_from_fixtures(self, detector):
        assert detector.X is not None
        assert detector.y is not None
        assert detector.X.shape[0] == detector.y.shape[0]
        assert detector.X.shape[1] == len(FEATURE_NAMES)

    def test_both_classes_present(self, detector):
        assert 0 in detector.y
        assert 1 in detector.y

    def test_summary_consistent(self, detector):
        s = detector._summary
        assert s.usable == detector.X.shape[0]
        assert s.genuine + s.tampered == s.usable
        assert s.n_features == len(FEATURE_NAMES)

    def test_features_finite(self, detector):
        assert np.all(np.isfinite(detector.X))

    def test_missing_dir_with_fallback(self):
        d = DocumentFraudDetector()
        s = d.load_dataset(dataset_dir=Path("nonexistent_dir"), use_fixtures_fallback=True)
        assert s.usable > 0

    def test_missing_dir_no_fallback_raises(self):
        d = DocumentFraudDetector()
        with pytest.raises(RuntimeError):
            d.load_dataset(dataset_dir=Path("nonexistent_dir"), use_fixtures_fallback=False)


class TestTrain:
    def test_trains_model(self, detector):
        model = detector.train()
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_before_load_raises(self):
        d = DocumentFraudDetector()
        with pytest.raises(RuntimeError):
            d.train()


class TestEvaluate:
    def test_evaluate_returns_metrics(self, detector):
        detector.train()
        ev = detector.evaluate()
        assert 0.0 <= ev.accuracy <= 1.0
        assert 0.0 <= ev.precision <= 1.0
        assert 0.0 <= ev.recall <= 1.0
        assert 0.0 <= ev.f1 <= 1.0
        assert ev.confusion.shape == (2, 2)
        assert ev.n_features == len(FEATURE_NAMES)


class TestSaveLoad:
    def test_save_and_reload(self, detector, tmp_path):
        detector.train()
        path = tmp_path / "test_model.pkl"
        detector.save(path)
        assert path.exists()
        assert path.stat().st_size > 0

        model, metadata = DocumentFraudDetector.load(path)
        assert model is not None
        assert metadata["feature_count"] == len(FEATURE_NAMES)
        assert metadata["model_version"] == "1.0.0"
        assert len(metadata["feature_names"]) == len(FEATURE_NAMES)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            DocumentFraudDetector.load(Path("no_such_model.pkl"))


class TestPredict:
    def test_predict_returns_result(self, detector):
        detector.train()
        fixtures = Path(__file__).parent / "fixtures"
        img = fixtures / "clean_rgb.jpg"
        result = detector.predict(img)
        assert result["prediction"] in ("genuine", "tampered")
        assert 0.0 <= result["confidence"] <= 1.0
        assert "genuine" in result["probabilities"]
        assert "tampered" in result["probabilities"]
        assert len(result["features"]) == len(FEATURE_NAMES)
