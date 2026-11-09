"""Localisation: flagged_regions from the local-tamper-v2 tile maps (core/tamper_features.py)."""

import json
import random
from pathlib import Path

import pytest
from PIL import Image

from backend import config
from backend.model_utils import ModelAPI
from core import tamper_features

ROOT = Path(__file__).parent.parent
SPLIT = json.loads((ROOT / "data" / "splits" / "day5_frozen_split.json").read_text(encoding="utf-8"))
GENUINE_PAGE = ROOT / "data" / "synthetic" / SPLIT["train"][0]["path"]  # a frozen TRAIN image
MODEL_V1 = ROOT / "data" / "models" / "fraud_model_v1.pkl"
DEMO_TAMPERED = ROOT / "data" / "demo" / "tampered_payslip_01.png"
DEMO_EDIT_BOX = (197, 321, 55, 12)  # bbox recorded in data/demo/demo_manifest.json
REGION_KEYS = {"x", "y", "width", "height", "area", "type"}


def intersects(region: dict, rect) -> bool:
    x, y, w, h = rect
    return (region["x"] < x + w and x < region["x"] + region["width"]
            and region["y"] < y + h and y < region["y"] + region["height"])


def edited_page(tmp_path: Path, method: str, scale: float = 1.0):
    """Apply a dataset-generator edit to a genuine TRAIN page; returns (path, edit rect in saved pixels)."""
    from scripts.generate_tampered_documents import _apply_tampering

    random.seed(3)  # fixes the edit rectangle
    with Image.open(GENUINE_PAGE) as im:
        page = im.convert("RGB")
    w, h = page.size
    state = random.getstate()  # same draw as _apply_tampering (see edit_footprints)
    rect = [random.randint(50, w - 250), random.randint(80, h - 150), random.randint(120, 200), random.randint(30, 60)]
    random.setstate(state)
    edited = _apply_tampering(page, method)
    if scale != 1.0:
        edited = edited.resize((round(w * scale), round(h * scale)), Image.NEAREST)
        rect = [round(v * scale) for v in rect]
    path = tmp_path / f"{method}.png"
    edited.save(path)
    return path, rect


@pytest.fixture(scope="module")
def api():
    return ModelAPI()


@pytest.fixture(scope="module")
def api_v1():
    api = ModelAPI(MODEL_V1)
    assert api.load_model()
    return api


def check_regions(regions, size):
    assert isinstance(regions, list) and len(regions) <= tamper_features.MAX_REGIONS
    for r in regions:
        assert set(r) == REGION_KEYS
        assert all(isinstance(r[k], int) and r[k] >= 0 for k in ("x", "y", "width", "height", "area"))
        assert r["width"] > 0 and r["height"] > 0
        assert r["x"] + r["width"] <= size[0] and r["y"] + r["height"] <= size[1]
        assert set(r["type"].split("+")) <= set(tamper_features.SIGNALS)
    assert [r["area"] for r in regions] == sorted((r["area"] for r in regions), reverse=True)


# ── Locator ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method, signal", [("noise_patch", "noise"), ("color_shift", "colour")])
def test_generator_edit_is_boxed(tmp_path, method, signal):
    path, rect = edited_page(tmp_path, method)
    regions = tamper_features.locate_regions(path)
    check_regions(regions, Image.open(path).size)
    assert regions and intersects(regions[0], rect)
    assert signal in regions[0]["type"].split("+")


def test_genuine_page_has_no_region():
    assert tamper_features.locate_regions(GENUINE_PAGE) == []


def test_extract_with_regions_keeps_the_model_features(tmp_path):
    """The model's inputs must not change: features equal extract_tamper_features on both classes."""
    path, _ = edited_page(tmp_path, "noise_patch")
    for page in (GENUINE_PAGE, path):
        features, _ = tamper_features.extract_with_regions(page)
        assert features == tamper_features.extract_tamper_features(page)


def test_boxes_are_mapped_to_the_uploaded_pixels(tmp_path, monkeypatch):
    """A page above 2000 px is analysed downscaled; boxes must come back in the upload's pixels.

    The analysis is pinned to the native page so the test checks only the coordinate mapping
    (resampled pages also change the tile statistics; see logs/localization_report.txt).
    """
    native, rect = edited_page(tmp_path, "noise_patch")
    big = tmp_path / "big.png"
    with Image.open(native) as im:
        im.resize((round(im.width * 2.5), round(im.height * 2.5)), Image.NEAREST).save(big)
    expected = tamper_features.locate_regions(native)
    real_load = tamper_features.load_image
    monkeypatch.setattr(tamper_features, "load_image", lambda path: real_load(native))
    regions = tamper_features.locate_regions(big)
    assert expected and len(regions) == len(expected)
    for r, e in zip(regions, expected):
        x0, y0 = round(e["x"] * 2.5), round(e["y"] * 2.5)
        assert (r["x"], r["y"]) == (x0, y0)
        assert (r["width"], r["height"]) == (round((e["x"] + e["width"]) * 2.5) - x0, round((e["y"] + e["height"]) * 2.5) - y0)
        assert r["area"] == round(e["area"] * 6.25) and r["type"] == e["type"]
    check_regions(regions, Image.open(big).size)
    assert intersects(regions[0], [round(v * 2.5) for v in rect])


def test_select_regions_scales_merges_and_limits():
    found = [
        ("noise_k2", 80, 40, 40, 16, 640),       # above its minimum (384)
        ("colour_rare_4pct", 120, 40, 80, 16, 5200),  # touches the noise box -> merged
        ("colour_rare_4pct", 400, 400, 16, 16, 256),  # below its minimum -> dropped
        ("ink_run", 10, 300, 60, 10, 12),            # 12 glyphs; area = run box
        ("noise_k4", 500, 500, 64, 64, 4096),        # not a locating mask -> ignored
    ]
    regions = tamper_features._select_regions(found, (1000, 800), (2000, 1600), max_regions=5)
    assert regions == [
        {"x": 160, "y": 80, "width": 240, "height": 32, "area": 20800, "type": "noise+colour"},
        {"x": 20, "y": 600, "width": 120, "height": 20, "area": 2400, "type": "ink"},
    ]
    assert tamper_features._select_regions(found, (1000, 800), (1000, 800), max_regions=1)[0]["area"] == 5200


def test_pdf_original_size_matches_the_rendered_page(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    from core.document_analyzer import _load_pdf_page

    path = tmp_path / "page.pdf"
    with pymupdf.open() as doc:
        doc.new_page(width=595, height=842)  # A4 in points
        doc.save(str(path))
    # load_image renders page 1 at 200 DPI, then caps the longest side at 2000 px
    assert tamper_features.original_size(path) == _load_pdf_page(path).size == (1653, 2339)


def test_demo_text_replacement_is_boxed_as_ink():
    regions = tamper_features.locate_regions(DEMO_TAMPERED)
    assert regions and regions[0]["type"] == "ink" and intersects(regions[0], DEMO_EDIT_BOX)


# ── ModelAPI and /verify ────────────────────────────────────────────────────

def test_v2_flagged_document_returns_its_regions(api, tmp_path):
    path, rect = edited_page(tmp_path, "noise_patch")
    result = api.predict(path)
    assert result["tamper_risk_score"] >= config.TAMPER_THRESHOLD
    assert result["flagged_regions"] == tamper_features.locate_regions(path)
    assert intersects(result["flagged_regions"][0], rect)
    assert "suspicious region(s) located" in result["interpretation"]


def test_v2_regions_are_withheld_below_the_threshold(api, tmp_path, monkeypatch):
    path, _ = edited_page(tmp_path, "noise_patch")
    assert tamper_features.locate_regions(path)  # regions exist ...
    monkeypatch.setattr(config, "TAMPER_THRESHOLD", 1.01)
    result = api.predict(path)  # ... but the document is not flagged
    assert result["flagged_regions"] == []
    assert "region" not in result["interpretation"]


def test_v2_genuine_document_returns_no_region(api):
    result = api.predict(GENUINE_PAGE)
    assert result["tamper_risk_score"] < config.TAMPER_THRESHOLD
    assert result["flagged_regions"] == []


def test_v1_model_never_returns_regions(api_v1, tmp_path, monkeypatch):
    assert api_v1.schema.extract_with_regions is None
    path, _ = edited_page(tmp_path, "noise_patch")
    monkeypatch.setattr(config, "TAMPER_THRESHOLD", 0.0)  # every document is flagged
    result = api_v1.predict(path)
    assert result["interpretation"].startswith("Evidence of digital tampering")
    assert result["flagged_regions"] == []


def test_verify_response_carries_regions(client, auth_headers, tmp_path):
    path, rect = edited_page(tmp_path, "color_shift")
    with path.open("rb") as fh:
        response = client.post("/verify", files={"file": (path.name, fh, "image/png")}, headers=auth_headers)
    assert response.status_code == 200, response.text
    regions = response.json()["flagged_regions"]
    check_regions(regions, Image.open(path).size)
    assert regions and intersects(regions[0], rect)
