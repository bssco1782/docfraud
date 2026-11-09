"""
tests/test_model_utils.py
Unit tests for backend/model_utils.py (ModelAPI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.model_utils import ModelAPI

SYNTHETIC = Path(__file__).parent.parent / "data" / "synthetic"
SAMPLE_IMAGE = SYNTHETIC / "genuine" / "payslip_001.png"
if not SAMPLE_IMAGE.exists():
    SAMPLE_IMAGE = Path(__file__).parent / "fixtures" / "clean_rgb.png"


def test_model_api_init():
    api = ModelAPI()
    assert api.model is None or api.is_loaded is True
    assert isinstance(api.metadata, dict)


def test_model_api_load_model():
    api = ModelAPI()
    success = api.load_model()
    assert success is True
    assert api.is_loaded is True
    assert api.model is not None
    assert "model_version" in api.metadata


def test_model_api_predict_from_path():
    api = ModelAPI()
    result = api.predict(SAMPLE_IMAGE)
    assert isinstance(result, dict)
    assert "tamper_risk_score" in result
    assert "confidence" in result
    assert "flagged_regions" in result
    assert "interpretation" in result
    assert 0.0 <= result["tamper_risk_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["flagged_regions"], list)
    assert isinstance(result["interpretation"], str)


def test_model_api_predict_from_bytes():
    api = ModelAPI()
    content = SAMPLE_IMAGE.read_bytes()
    result = api.predict(content, filename=SAMPLE_IMAGE.name)
    assert isinstance(result, dict)
    assert 0.0 <= result["tamper_risk_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_model_api_extract_flagged_regions():
    api = ModelAPI()
    # Create synthetic heatmap with a hot 30x30 block
    heatmap = np.zeros((100, 100, 3), dtype=np.uint8)
    heatmap[20:50, 20:50, :] = 255  # bright block

    regions = api.extract_flagged_regions(heatmap, threshold=50, min_area=20)
    assert len(regions) >= 1
    assert "x" in regions[0]
    assert "y" in regions[0]
    assert "width" in regions[0]
    assert "height" in regions[0]
    assert "area" in regions[0]
    assert regions[0]["area"] >= 20


def test_model_api_has_no_magic_bytes_shortcut():
    """Arbitrary bytes must go through real decoding, never a canned response."""
    api = ModelAPI()
    with pytest.raises(ValueError):
        api.predict(b"fake image bytes for verification")


def test_model_api_invalid_image_raises_value_error():
    api = ModelAPI()
    with pytest.raises(ValueError):
        api.predict(b"corrupted binary non-image data", filename="corrupt.png")
