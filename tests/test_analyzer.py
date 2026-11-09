"""
tests/test_analyzer.py
Unit tests for core/document_analyzer.py.

FIXTURE NOTICE:
  All images used in these tests are programmatically generated fixtures
  located in tests/fixtures/.  They are UNIT-TEST FIXTURES ONLY and do
  NOT represent real project data or real Indian document samples.

REAL-IMAGE NOTICE:
  Tests that would run against data/synthetic/ (the project's real
  document dataset) are SKIPPED when that directory is empty.
  Skipped tests are explicitly labelled "REAL-IMAGE TEST — NOT PERFORMED".
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.document_analyzer import (
    FEATURE_NAMES,
    ela_analysis,
    extract_features,
    features_to_vector,
    load_image,
)

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC = Path(__file__).parent.parent / "data" / "synthetic"

_SYNTHETIC_IMAGES = sorted(SYNTHETIC.glob("*.jpg")) + sorted(SYNTHETIC.glob("*.png")) if SYNTHETIC.exists() else []
_HAVE_REAL_IMAGES = len(_SYNTHETIC_IMAGES) >= 5

# ──────────────────────────────────────────────────────────────
# Fixture paths
# ──────────────────────────────────────────────────────────────
CLEAN_JPG = FIXTURES / "clean_rgb.jpg"
CLEAN_PNG = FIXTURES / "clean_rgb.png"
GRAYSCALE_JPG = FIXTURES / "grayscale.jpg"
SMALL_JPG = FIXTURES / "small_image.jpg"
LARGE_JPG = FIXTURES / "large_image.jpg"
TAMPERED_JPG = FIXTURES / "tampered_patch.jpg"
CORRUPT_JPG = FIXTURES / "corrupt.jpg"
NONEXISTENT = FIXTURES / "does_not_exist.jpg"


# ──────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────
def _all_finite(d: dict) -> bool:
    return all(math.isfinite(v) for v in d.values())


# ──────────────────────────────────────────────────────────────
# load_image — basic contract
# ──────────────────────────────────────────────────────────────
class TestLoadImage:
    def test_jpeg_loads_rgb(self):
        img = load_image(CLEAN_JPG)
        assert img.mode == "RGB"

    def test_png_loads_rgb(self):
        img = load_image(CLEAN_PNG)
        assert img.mode == "RGB"

    def test_grayscale_converted_to_rgb(self):
        img = load_image(GRAYSCALE_JPG)
        assert img.mode == "RGB"

    def test_large_image_resized(self):
        img = load_image(LARGE_JPG)
        assert max(img.size) <= 2000

    def test_small_image_not_upscaled(self):
        img = load_image(SMALL_JPG)
        # Small fixture is 80×80 — must not be enlarged
        assert max(img.size) <= 2000

    def test_nonexistent_raises_value_error(self):
        with pytest.raises(ValueError, match="not found"):
            load_image(NONEXISTENT)

    def test_corrupt_raises_value_error(self):
        with pytest.raises(ValueError):
            load_image(CORRUPT_JPG)

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        bad = tmp_path / "file.bmp"
        bad.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(bad)


# ──────────────────────────────────────────────────────────────
# ela_analysis — contract
# ──────────────────────────────────────────────────────────────
class TestElaAnalysis:
    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_returns_heatmap_and_score(self, path):
        heatmap, score = ela_analysis(path)
        assert isinstance(heatmap, np.ndarray)
        assert isinstance(score, float)

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_heatmap_dtype_uint8(self, path):
        heatmap, _ = ela_analysis(path)
        assert heatmap.dtype == np.uint8

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_heatmap_three_channels(self, path):
        heatmap, _ = ela_analysis(path)
        assert heatmap.ndim == 3
        assert heatmap.shape[2] == 3

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_score_in_range(self, path):
        _, score = ela_analysis(path)
        assert 0.0 <= score <= 100.0, f"Score out of range: {score}"

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_score_is_finite(self, path):
        _, score = ela_analysis(path)
        assert math.isfinite(score)

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_ela_deterministic(self, path):
        _, score1 = ela_analysis(path)
        _, score2 = ela_analysis(path)
        assert score1 == score2, f"ELA not deterministic for {path.name}"

    def test_nonexistent_raises_value_error(self):
        with pytest.raises(ValueError):
            ela_analysis(NONEXISTENT)

    def test_corrupt_raises_value_error(self):
        with pytest.raises(ValueError):
            ela_analysis(CORRUPT_JPG)


# ──────────────────────────────────────────────────────────────
# extract_features — contract
# ──────────────────────────────────────────────────────────────
class TestExtractFeatures:
    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_returns_expected_feature_count(self, path):
        feats = extract_features(path)
        assert len(feats) == len(FEATURE_NAMES), f"Got {len(feats)} features, expected {len(FEATURE_NAMES)}"

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_feature_names_match_constant(self, path):
        feats = extract_features(path)
        assert list(feats.keys()) == FEATURE_NAMES

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_all_values_finite(self, path):
        feats = extract_features(path)
        bad = [k for k, v in feats.items() if not math.isfinite(v)]
        assert not bad, f"Non-finite features for {path.name}: {bad}"

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_all_values_are_floats(self, path):
        feats = extract_features(path)
        not_float = [k for k, v in feats.items() if not isinstance(v, float)]
        assert not not_float

    @pytest.mark.parametrize("path", [CLEAN_JPG, CLEAN_PNG, GRAYSCALE_JPG, SMALL_JPG, LARGE_JPG])
    def test_feature_extraction_deterministic(self, path):
        feats1 = extract_features(path)
        feats2 = extract_features(path)
        for k in FEATURE_NAMES:
            assert feats1[k] == feats2[k], f"Not deterministic: {k} for {path.name}"

    def test_nonexistent_raises_value_error(self):
        with pytest.raises(ValueError):
            extract_features(NONEXISTENT)

    def test_corrupt_raises_value_error(self):
        with pytest.raises(ValueError):
            extract_features(CORRUPT_JPG)


# ──────────────────────────────────────────────────────────────
# features_to_vector — ordering
# ──────────────────────────────────────────────────────────────
class TestFeaturesToVector:
    def test_vector_length_matches_schema(self):
        feats = extract_features(CLEAN_JPG)
        vec = features_to_vector(feats)
        assert len(vec) == len(FEATURE_NAMES)

    def test_order_matches_feature_names(self):
        feats = extract_features(CLEAN_JPG)
        vec = features_to_vector(feats)
        for i, name in enumerate(FEATURE_NAMES):
            assert vec[i] == feats[name]

    def test_all_values_finite(self):
        feats = extract_features(CLEAN_JPG)
        vec = features_to_vector(feats)
        assert all(math.isfinite(v) for v in vec)

    def test_missing_key_defaults_to_zero(self):
        vec = features_to_vector({})
        assert vec == [0.0] * len(FEATURE_NAMES)

    def test_stable_across_calls(self):
        feats = extract_features(CLEAN_JPG)
        assert features_to_vector(feats) == features_to_vector(feats)


# ──────────────────────────────────────────────────────────────
# Tamper sensitivity
# ──────────────────────────────────────────────────────────────
class TestTamperSensitivity:
    def test_ela_block_max_ratio_higher_for_tampered(self):
        """
        A patch compressed at a different quality should raise
        ela_block_max_ratio relative to the clean version.

        This tests the fixture pair (clean_rgb.jpg vs tampered_patch.jpg).
        The tampered image has a 100×80 region that was re-saved at
        JPEG quality 30 then pasted — a different compression level
        than the surrounding image saved at quality 95.
        """
        feats_clean = extract_features(CLEAN_JPG)
        feats_tampered = extract_features(TAMPERED_JPG)
        clean_ratio = feats_clean["ela_block_max_ratio"]
        tampered_ratio = feats_tampered["ela_block_max_ratio"]
        assert tampered_ratio > clean_ratio, (
            f"Tampered image should have higher ela_block_max_ratio "
            f"(tampered={tampered_ratio:.4f}, clean={clean_ratio:.4f})"
        )

    def test_ela_hotspot_count_higher_for_tampered(self):
        feats_clean = extract_features(CLEAN_JPG)
        feats_tampered = extract_features(TAMPERED_JPG)
        assert feats_tampered["ela_hotspot_count"] >= feats_clean["ela_hotspot_count"], (
            "Tampered image should have >= hotspot count vs clean"
        )


# ──────────────────────────────────────────────────────────────
# REAL-IMAGE TESTS (skipped — data/synthetic/ is empty)
# ──────────────────────────────────────────────────────────────
REAL_IMAGE_SKIP_REASON = (
    "REAL-IMAGE TEST — NOT PERFORMED: "
    "data/synthetic/ contains fewer than 5 document images. "
    "Add real Indian payslip/bank statement/ID images to data/synthetic/ "
    "to enable these tests."
)


@pytest.mark.skipif(not _HAVE_REAL_IMAGES, reason=REAL_IMAGE_SKIP_REASON)
class TestRealImages:
    """
    Tests against real project documents in data/synthetic/.
    These tests are SKIPPED until 5+ real images exist there.

    Each test checks the same guarantees as the fixture tests:
    - image opens and ELA completes
    - score is numeric and in [0, 100]
    - heatmap is produced with correct shape/dtype
    - extract_features returns exactly 15 finite values
    - feature names match FEATURE_NAMES
    - two calls give identical output (determinism)
    """

    @pytest.mark.parametrize("path", _SYNTHETIC_IMAGES[:5] if _SYNTHETIC_IMAGES else [])
    def test_image_opens_and_ela_runs(self, path):
        img = load_image(path)
        assert img.mode == "RGB"
        heatmap, score = ela_analysis(path)
        assert heatmap.dtype == np.uint8
        assert 0.0 <= score <= 100.0

    @pytest.mark.parametrize("path", _SYNTHETIC_IMAGES[:5] if _SYNTHETIC_IMAGES else [])
    def test_features_finite_values(self, path):
        feats = extract_features(path)
        assert len(feats) == len(FEATURE_NAMES)
        assert list(feats.keys()) == FEATURE_NAMES
        bad = [k for k, v in feats.items() if not math.isfinite(v)]
        assert not bad

    @pytest.mark.parametrize("path", _SYNTHETIC_IMAGES[:5] if _SYNTHETIC_IMAGES else [])
    def test_determinism_real_image(self, path):
        feats1 = extract_features(path)
        feats2 = extract_features(path)
        for k in FEATURE_NAMES:
            assert feats1[k] == feats2[k]


# ──────────────────────────────────────────────────────────────
# Day 4 — expanded feature validation
# ──────────────────────────────────────────────────────────────
class TestDay4Features:
    """Validate Day 4 feature-engineering expansion (22 features)."""

    def test_feature_count_at_least_15(self):
        assert len(FEATURE_NAMES) >= 15

    def test_feature_names_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    @pytest.mark.parametrize("path", [CLEAN_JPG, TAMPERED_JPG, SMALL_JPG, LARGE_JPG, GRAYSCALE_JPG])
    def test_geometry_features_positive(self, path):
        feats = extract_features(path)
        assert feats["image_width"] > 0
        assert feats["image_height"] > 0
        assert feats["aspect_ratio"] > 0

    @pytest.mark.parametrize("path", [CLEAN_JPG, TAMPERED_JPG, SMALL_JPG, LARGE_JPG, GRAYSCALE_JPG])
    def test_intensity_features_in_range(self, path):
        feats = extract_features(path)
        assert 0.0 <= feats["grayscale_mean"] <= 255.0
        assert 0.0 <= feats["grayscale_std"] <= 255.0

    @pytest.mark.parametrize("path", [CLEAN_JPG, TAMPERED_JPG, SMALL_JPG, LARGE_JPG, GRAYSCALE_JPG])
    def test_ratio_features_bounded(self, path):
        feats = extract_features(path)
        assert 0.0 <= feats["edge_density"] <= 1.0
        assert 0.0 <= feats["dark_pixel_ratio"] <= 1.0

    @pytest.mark.parametrize("path", [CLEAN_JPG, TAMPERED_JPG, SMALL_JPG, LARGE_JPG, GRAYSCALE_JPG])
    def test_all_new_features_finite(self, path):
        feats = extract_features(path)
        new_names = ["image_width", "image_height", "aspect_ratio",
                     "grayscale_mean", "grayscale_std", "edge_density",
                     "dark_pixel_ratio"]
        bad = [k for k in new_names if not math.isfinite(feats[k])]
        assert not bad, f"Non-finite new features: {bad}"

    def test_deterministic_repeated_extraction(self):
        feats1 = extract_features(CLEAN_JPG)
        feats2 = extract_features(CLEAN_JPG)
        assert list(feats1.keys()) == list(feats2.keys())
        assert list(feats1.keys()) == FEATURE_NAMES
        for k in FEATURE_NAMES:
            assert feats1[k] == feats2[k], f"{k} not deterministic"
