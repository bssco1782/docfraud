"""
core/forensic_model.py
Prediction interface for the local-forensics model (``fraud_model_local_v1.pkl``).

Founder B can consume this without touching feature code:

    fm = ForensicModel.load()
    result = fm.analyze("upload.png")

Output is written for an NBFC review workflow.  ``model_score`` is the
Random Forest vote fraction — an UNCALIBRATED score, not a probability
that the document (or applicant) is fraudulent.
"""

from __future__ import annotations

import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core import local_forensics as lf
from core.document_analyzer import FEATURE_NAMES as DAY4_NAMES
from core.document_analyzer import extract_features

ROOT = Path(__file__).parent.parent
DEFAULT_LOCAL_MODEL_PATH = ROOT / "data" / "models" / "fraud_model_local_v1.pkl"
MODEL_VERSION = "local-v1.0.0"
COMBINED_FEATURE_NAMES: list[str] = list(DAY4_NAMES) + list(lf.FEATURE_NAMES)
DECISION_THRESHOLD = 0.5  # default RF majority vote; not tuned


def extract_combined(image_path: str | Path, names: list[str]) -> list[float]:
    """Feature vector in exactly ``names`` order (day4 and/or local-forensics-v1)."""
    feats: dict[str, float] = {}
    if any(n in DAY4_NAMES for n in names):
        feats.update(extract_features(image_path))
    if any(n in lf.FEATURE_NAMES for n in names):
        feats.update(lf.extract_local_features(image_path))
    missing = [n for n in names if n not in feats]
    if missing:
        raise ValueError(f"model expects unknown features: {missing}")
    return [float(feats[n]) for n in names]


def _evidence(score: float) -> tuple[str, str]:
    """Map a model score to an evidence level and a human-readable summary."""
    if score >= 0.8:
        return "strong", "Model detected strong tampering evidence."
    if score >= DECISION_THRESHOLD:
        return "moderate", "Model detected moderate tampering evidence; manual review recommended."
    if score >= 0.3:
        return "weak", "Model found weak, inconclusive signals; no tampering evidence above threshold."
    return "none", "Model found no tampering evidence."


class ForensicModel:
    """Experimental local-forensics classifier (not used by the API)."""

    def __init__(self, model, metadata: dict):
        """Wrap a fitted model; raises ValueError if metadata and model feature counts differ."""
        self.model = model
        self.metadata = metadata
        self.feature_names: list[str] = list(metadata["feature_names"])
        if getattr(model, "n_features_in_", len(self.feature_names)) != len(self.feature_names):
            raise ValueError("model / metadata feature count mismatch")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ForensicModel":
        """Load a pickled {model, metadata} bundle (trusted, repo-produced files only)."""
        path = Path(path) if path else DEFAULT_LOCAL_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:  # trusted, repo-produced artifact only
            bundle = pickle.load(f)
        return cls(bundle["model"], bundle["metadata"])

    def score(self, image_path: str | Path) -> float:
        """Return the uncalibrated tampered-class score for one image."""
        vec = np.array([extract_combined(image_path, self.feature_names)], dtype=np.float64)
        return float(self.model.predict_proba(vec)[0][1])

    def analyze(self, image_path: str | Path, max_regions: int = 5) -> dict:
        """Classification + uncalibrated model score + localised forensic evidence."""
        t0 = time.perf_counter()
        base = {
            "model_version": self.metadata.get("model_version"),
            "feature_schema_version": self.metadata.get("feature_schema_version"),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            score = self.score(image_path)
            regions = lf.localize(image_path)[:max_regions]
        except (ValueError, OSError) as exc:
            return {**base, "processing_status": "error", "error": str(exc),
                    "processing_ms": round(1000 * (time.perf_counter() - t0), 1)}
        strength, summary = _evidence(score)
        return {
            **base,
            "processing_status": "ok",
            "tamper_classification": "tampering_evidence" if score >= DECISION_THRESHOLD else "no_tampering_evidence",
            "model_score": round(score, 4),
            "model_score_note": "Uncalibrated Random Forest vote fraction; not a fraud probability.",
            "evidence_strength": strength,
            "summary": summary,
            "suspicious_regions": regions,
            "processing_ms": round(1000 * (time.perf_counter() - t0), 1),
        }
