"""
core/feature_schemas.py
Registry of feature schemas: how an image becomes the vector a saved model expects.

A model file records ``metadata["feature_schema_version"]``; ``get_schema()`` returns the
feature names (in model column order) and the extractor for that version, so the API and the
scripts never assume one particular feature set.

    schema = get_schema(metadata.get("feature_schema_version"))
    features = schema.extract(path)          # dict, name -> float
    vector = schema.vector(features)         # list in model column order

Schemas that can localise edits also provide ``extract_with_regions`` (features plus anomalous
regions from one pass); for the others it is None and no regions are reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from core import document_analyzer, tamper_features

Regions = list[dict]

# Models saved before the schema was recorded in metadata used the Day-4 features.
LEGACY_SCHEMA = "day4-v2-22"


@dataclass(frozen=True)
class FeatureSchema:
    version: str
    names: tuple[str, ...]
    extract: Callable[[str | Path], dict[str, float]]
    description: str
    extract_with_regions: Optional[Callable[[str | Path], tuple[dict[str, float], Regions]]] = None

    def vector(self, features: dict[str, float]) -> list[float]:
        """Feature values in model column order."""
        return [float(features[n]) for n in self.names]


SCHEMAS: dict[str, FeatureSchema] = {
    LEGACY_SCHEMA: FeatureSchema(
        LEGACY_SCHEMA, tuple(document_analyzer.FEATURE_NAMES), document_analyzer.extract_features,
        "22 whole-image features (ELA, noise, compression, geometry, intensity); model v1",
    ),
    tamper_features.FEATURE_SCHEMA_VERSION: FeatureSchema(
        tamper_features.FEATURE_SCHEMA_VERSION, tuple(tamper_features.FEATURE_NAMES),
        tamper_features.extract_tamper_features,
        "29 local tamper features (noise, colour, softness, ink regions + processing fingerprint); model v2",
        tamper_features.extract_with_regions,
    ),
}


def get_schema(version: Optional[str]) -> FeatureSchema:
    """Schema for a model's recorded version (None = legacy Day-4 model). Raises ValueError if unknown."""
    key = version or LEGACY_SCHEMA
    if key not in SCHEMAS:
        raise ValueError(f"unknown feature schema '{key}' (known: {', '.join(SCHEMAS)})")
    return SCHEMAS[key]


def schema_for_model(model, metadata: Optional[dict] = None) -> FeatureSchema:
    """Schema from metadata when available, else the registered schema with the model's column count."""
    if metadata and metadata.get("feature_schema_version"):
        return get_schema(metadata["feature_schema_version"])
    n = getattr(model, "n_features_in_", None)
    matches = [s for s in SCHEMAS.values() if len(s.names) == n]
    if len(matches) != 1:
        raise ValueError(f"cannot infer the feature schema of a model with {n} features")
    return matches[0]
