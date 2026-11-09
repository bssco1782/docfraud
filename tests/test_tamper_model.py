"""Day 8 (Founder A): model v2 - local-tamper-v2 features, schema registry, data isolation, provenance."""

import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core import document_analyzer, tamper_features  # noqa: E402
from core.feature_schemas import LEGACY_SCHEMA, get_schema, schema_for_model  # noqa: E402
from scripts import generate_extra_documents as gen_extra  # noqa: E402
from scripts import train_tamper_model as ttm  # noqa: E402

MODEL_PATH = ROOT / "data" / "models" / "fraud_model.pkl"
SPLIT = json.loads((ROOT / "data" / "splits" / "day5_frozen_split.json").read_text(encoding="utf-8"))
GENUINE_PAGE = ROOT / "data" / "synthetic" / SPLIT["train"][0]["path"]  # a frozen TRAIN image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def bundle():
    return pickle.loads(MODEL_PATH.read_bytes())


# ── Features ────────────────────────────────────────────────────────────────

def test_features_complete_finite_and_deterministic():
    a = tamper_features.extract_tamper_features(GENUINE_PAGE)
    b = tamper_features.extract_tamper_features(GENUINE_PAGE)
    assert list(a) == tamper_features.FEATURE_NAMES and len(a) == 29
    assert all(isinstance(v, float) and np.isfinite(v) for v in a.values())
    assert a == b


def test_too_small_image_is_a_controlled_error(tmp_path):
    path = tmp_path / "tiny.png"
    Image.new("RGB", (12, 40), "white").save(path)
    with pytest.raises(ValueError, match="too small"):
        tamper_features.extract_tamper_features(path)


@pytest.mark.parametrize("method, feature", [("noise_patch", "noise_blob_k2"), ("color_shift", "colour_rare_blob_4pct")])
def test_generator_edits_raise_their_region_features(tmp_path, method, feature):
    """A known edit from the dataset generator, applied to a genuine TRAIN page, must be visible locally."""
    from scripts.generate_tampered_documents import _apply_tampering

    random.seed(3)  # fixes the edit rectangle
    with Image.open(GENUINE_PAGE) as im:
        edited = _apply_tampering(im.convert("RGB"), method)
    path = tmp_path / "edited.png"
    edited.save(path)
    before = tamper_features.extract_tamper_features(GENUINE_PAGE)[feature]
    after = tamper_features.extract_tamper_features(path)[feature]
    assert after > before + 10, (before, after)


# ── Schema registry ─────────────────────────────────────────────────────────

def test_schema_registry():
    assert get_schema(None).version == LEGACY_SCHEMA  # models saved before the schema was recorded
    assert get_schema(LEGACY_SCHEMA).names == tuple(document_analyzer.FEATURE_NAMES)
    v2 = get_schema(tamper_features.FEATURE_SCHEMA_VERSION)
    assert v2.names == tuple(tamper_features.FEATURE_NAMES)
    with pytest.raises(ValueError, match="unknown feature schema"):
        get_schema("no-such-schema")


def test_schema_for_model_uses_metadata_then_column_count(bundle):
    model = bundle["model"]
    assert schema_for_model(model, bundle["metadata"]).version == tamper_features.FEATURE_SCHEMA_VERSION
    assert schema_for_model(model).version == tamper_features.FEATURE_SCHEMA_VERSION  # 29 columns

    class Five:
        n_features_in_ = 5

    with pytest.raises(ValueError, match="cannot infer"):
        schema_for_model(Five())


def test_api_uses_the_models_schema():
    from backend.model_utils import ModelAPI

    api = ModelAPI(MODEL_PATH)
    api.load_model()
    assert api.schema.version == tamper_features.FEATURE_SCHEMA_VERSION
    out = api.predict(GENUINE_PAGE)
    assert 0.0 <= out["tamper_risk_score"] <= 1.0
    assert "hotspot" not in out["interpretation"]  # the Day-4 hotspot count does not exist in v2


# ── Data isolation and provenance ───────────────────────────────────────────

def test_generated_sets_verified_and_disjoint_from_frozen_data():
    frozen = {r["sha256"] for r in SPLIT["test"] + SPLIT["train"]}
    dev = gen_extra.load_set("dev")          # raises if any file differs from its manifest
    holdout = gen_extra.load_set("holdout")
    dev_h = {sha256(r["file"]) for r in dev}
    hold_h = {sha256(r["file"]) for r in holdout}
    assert len(dev) == 150 and len(holdout) == 149
    assert not dev_h & hold_h and not (dev_h | hold_h) & frozen
    manifest = json.loads((gen_extra.GENERATED / "holdout_v1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] not in gen_extra.USED_SEEDS
    assert [e["path"] for e in manifest["excluded"]] == ["genuine/id_proof_2131.png"]


def test_model_trained_without_test_or_holdout(bundle):
    meta = bundle["metadata"]
    trained = set(meta["training_files"].values())
    assert len(trained) == meta["n_training_samples"] == 210
    assert not trained & {r["sha256"] for r in SPLIT["test"]}
    holdout = json.loads((gen_extra.GENERATED / "holdout_v1" / "manifest.json").read_text(encoding="utf-8"))
    assert not trained & set(holdout["files"].values())
    assert holdout["created_utc"] > meta["trained_at"]  # holdout generated after the model was frozen


def test_training_documents_match_model_record(bundle):
    docs = ttm.training_documents()
    assert {d["path"]: d["sha256"] for d in docs} == bundle["metadata"]["training_files"]


def test_processing_is_deterministic_and_label_independent(tmp_path):
    with Image.open(GENUINE_PAGE) as im:
        im = im.convert("RGB")
        a = ttm.process(im, random.Random(7), tmp_path, "a")
        b = ttm.process(im, random.Random(7), tmp_path, "b")
    assert a[1] == b[1] and sha256(a[0]) == sha256(b[0])


def test_demo_record_belongs_to_deployed_model_and_keeps_v1_history():
    manifest = json.loads((ROOT / "data" / "demo" / "demo_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["sha256"] == sha256(MODEL_PATH)
    history = manifest["scoring_history"]
    assert history[0]["model"]["sha256"] == sha256(ROOT / "data" / "models" / "fraud_model_v1.pkl")
    assert sum(x["correct"] for x in history[0]["images"]) == 2  # v1 record: 2/5 correct
