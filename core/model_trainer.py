"""
core/model_trainer.py
Random Forest fraud-detection model: training, evaluation, persistence.

Uses feature extraction from core.document_analyzer (Day-2).
Independent of FastAPI — no API logic here.
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from core.document_analyzer import FEATURE_NAMES, extract_features, features_to_vector
from core.feature_schemas import schema_for_model

ROOT = Path(__file__).parent.parent
DEFAULT_DATASET_DIR = ROOT / "data" / "synthetic"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
DEFAULT_MODEL_PATH = ROOT / "data" / "models" / "fraud_model.pkl"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Labels used by Founder C's dataset structure
LABEL_GENUINE = "genuine"
LABEL_TAMPERED = "tampered"
VALID_LABELS = {LABEL_GENUINE, LABEL_TAMPERED}

# Label encoding: genuine=0, tampered=1
LABEL_MAP = {LABEL_GENUINE: 0, LABEL_TAMPERED: 1}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# Known fixture labels (fallback when C's dataset missing)
FIXTURE_LABELS = {
    "tampered_patch.jpg": LABEL_TAMPERED,
}


class DatasetSummary(NamedTuple):
    total: int
    genuine: int
    tampered: int
    usable: int
    skipped: int
    skip_reasons: list[str]
    n_features: int


class EvalResult(NamedTuple):
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    method: str
    train_size: int
    test_size: int
    total_size: int
    genuine_count: int
    tampered_count: int
    n_features: int
    test_files: tuple[str, ...] = ()


# Bump when FEATURE_NAMES changes (Day 4: 15 → 22 features).
FEATURE_SCHEMA_VERSION = "day4-v2-22"

# Random Forest hyperparameters (unchanged since Day 3; sklearn defaults otherwise).
RF_PARAMS = {"n_estimators": 100, "class_weight": "balanced"}
TEST_SIZE = 0.2


class DocumentFraudDetector:
    """Train and persist a Random Forest fraud-detection model."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model: RandomForestClassifier | None = None
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.files: list[str] = []
        self.dataset_sha256: str | None = None
        self._summary: DatasetSummary | None = None
        self._eval: EvalResult | None = None

    def load_dataset(
        self,
        dataset_dir: Path | None = None,
        use_fixtures_fallback: bool = True,
    ) -> DatasetSummary:
        """Load labelled images, extract features, build X/y arrays.

        Primary source: dataset_dir/genuine/ and dataset_dir/tampered/.
        Fallback: tests/fixtures/ with hardcoded labels if primary empty.
        """
        dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR

        images: list[tuple[Path, str]] = []

        # Primary: C's dataset structure
        for label in [LABEL_GENUINE, LABEL_TAMPERED]:
            sub = dataset_dir / label
            if sub.is_dir():
                for p in sorted(sub.iterdir()):
                    if p.suffix.lower() in IMAGE_SUFFIXES:
                        images.append((p, label))

        source = "synthetic"
        if not images and use_fixtures_fallback and FIXTURES_DIR.is_dir():
            source = "fixtures"
            for p in sorted(FIXTURES_DIR.iterdir()):
                if p.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if "corrupt" in p.stem.lower():
                    continue
                label = FIXTURE_LABELS.get(p.name, LABEL_GENUINE)
                images.append((p, label))

        if not images:
            raise RuntimeError(
                f"No images found in {dataset_dir} or {FIXTURES_DIR}. "
                "Ensure Founder C's dataset exists."
            )

        # Exact-duplicate guard: identical files must never straddle the
        # train/test split. Fail loudly instead of silently dropping samples.
        digests: dict[str, list[str]] = {}
        for path, label in images:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            digests.setdefault(digest, []).append(f"{label}/{path.name}")
        dupes = [names for names in digests.values() if len(names) > 1]
        if dupes:
            raise RuntimeError(f"Exact duplicate images in dataset: {dupes}")
        # Fingerprint of dataset state (label/name + content hash), for reproducibility.
        self.dataset_sha256 = hashlib.sha256(
            "\n".join(sorted(f"{n[0]}:{d}" for d, n in digests.items())).encode()
        ).hexdigest()

        vectors: list[list[float]] = []
        labels: list[int] = []
        files: list[str] = []
        skip_reasons: list[str] = []

        print(f"\nLoading dataset from {source} ({len(images)} candidates)...")
        for path, label in images:
            if label not in VALID_LABELS:
                skip_reasons.append(f"{path.name}: invalid label '{label}'")
                continue
            try:
                feats = extract_features(str(path))
                vec = features_to_vector(feats)
                if len(vec) != len(FEATURE_NAMES):
                    skip_reasons.append(f"{path.name}: feature count {len(vec)} != {len(FEATURE_NAMES)}")
                    continue
                if not all(np.isfinite(v) for v in vec):
                    skip_reasons.append(f"{path.name}: non-finite feature values")
                    continue
                vectors.append(vec)
                labels.append(LABEL_MAP[label])
                files.append(f"{label}/{path.name}")
                print(f"  [{label:>8s}] {path.name}")
            except Exception as exc:
                skip_reasons.append(f"{path.name}: {exc}")

        if not vectors:
            raise RuntimeError("No usable samples after feature extraction.")

        self.X = np.array(vectors, dtype=np.float64)
        self.y = np.array(labels, dtype=np.int32)
        self.files = files

        genuine_count = int((self.y == 0).sum())
        tampered_count = int((self.y == 1).sum())

        self._summary = DatasetSummary(
            total=len(images),
            genuine=genuine_count,
            tampered=tampered_count,
            usable=len(vectors),
            skipped=len(skip_reasons),
            skip_reasons=skip_reasons,
            n_features=self.X.shape[1],
        )

        print("\nDataset summary:")
        print(f"  Total candidates: {self._summary.total}")
        print(f"  Genuine:  {self._summary.genuine}")
        print(f"  Tampered: {self._summary.tampered}")
        print(f"  Usable:   {self._summary.usable}")
        print(f"  Skipped:  {self._summary.skipped}")
        print(f"  Features: {self._summary.n_features}")
        if skip_reasons:
            print("  Skip reasons:")
            for r in skip_reasons:
                print(f"    - {r}")

        if genuine_count == 0 or tampered_count == 0:
            raise RuntimeError(
                f"Need both classes. Got genuine={genuine_count}, tampered={tampered_count}."
            )

        return self._summary

    def train(self) -> RandomForestClassifier:
        """Train RandomForestClassifier on loaded dataset."""
        if self.X is None or self.y is None:
            raise RuntimeError("Call load_dataset() first.")

        self.model = RandomForestClassifier(
            **RF_PARAMS,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(self.X, self.y)
        print(f"\nModel trained: {self.model.n_estimators} trees, {self.X.shape[1]} features")
        return self.model

    def evaluate(self) -> EvalResult:
        """Evaluate model. Uses cross-validation for small datasets, holdout for larger."""
        if self.model is None or self.X is None or self.y is None:
            raise RuntimeError("Call train() first.")

        n_samples = len(self.y)
        genuine_count = int((self.y == 0).sum())
        tampered_count = int((self.y == 1).sum())
        min_class = min(genuine_count, tampered_count)

        if n_samples >= 20 and min_class >= 4:
            # Holdout split
            # Split indices (not arrays) so the held-out files can be logged.
            idx_train, idx_test = train_test_split(
                np.arange(n_samples),
                test_size=TEST_SIZE,
                random_state=self.random_state,
                stratify=self.y,
            )
            y_train, y_test = self.y[idx_train], self.y[idx_test]
            self.model.fit(self.X[idx_train], y_train)
            y_pred = self.model.predict(self.X[idx_test])
            y_true = y_test
            test_files = tuple(self.files[i] for i in idx_test) if self.files else ()
            method = f"holdout 80/20 (train={len(y_train)}, test={len(y_test)})"
            train_size = len(y_train)
            test_size = len(y_test)
            # Retrain on full data after evaluation
            self.model.fit(self.X, self.y)
        else:
            # Leave-one-out or small-k CV
            n_folds = min(n_samples, min_class, 5)
            if n_folds < 2:
                n_folds = 2
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
            y_pred = cross_val_predict(self.model, self.X, self.y, cv=cv)
            y_true = self.y
            method = f"{n_folds}-fold stratified CV (n={n_samples})"
            train_size = n_samples
            test_size = n_samples
            test_files = ()

        zero_div = "zero_division"
        self._eval = EvalResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, **{zero_div: 0.0})),
            recall=float(recall_score(y_true, y_pred, **{zero_div: 0.0})),
            f1=float(f1_score(y_true, y_pred, **{zero_div: 0.0})),
            confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
            method=method,
            train_size=train_size,
            test_size=test_size,
            total_size=n_samples,
            genuine_count=genuine_count,
            tampered_count=tampered_count,
            n_features=self.X.shape[1],
            test_files=test_files,
        )

        self._print_eval(self._eval, y_true, y_pred)
        return self._eval

    @staticmethod
    def _print_eval(ev: EvalResult, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print evaluation metrics and the confusion matrix to stdout."""
        print(f"\n{'='*50}")
        print("EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"  Method:    {ev.method}")
        print(f"  Samples:   {ev.total_size} (genuine={ev.genuine_count}, tampered={ev.tampered_count})")
        print(f"  Features:  {ev.n_features}")
        print(f"  Accuracy:  {ev.accuracy:.4f}")
        print(f"  Precision: {ev.precision:.4f}")
        print(f"  Recall:    {ev.recall:.4f}")
        print(f"  F1:        {ev.f1:.4f}")
        print("\n  Confusion matrix (rows=actual, cols=predicted):")
        print("               predicted_genuine  predicted_tampered")
        cm = ev.confusion
        if cm.shape == (2, 2):
            print(f"  actual_genuine   {cm[0,0]:>8d}  {cm[0,1]:>8d}")
            print(f"  actual_tampered  {cm[1,0]:>8d}  {cm[1,1]:>8d}")
        else:
            print(f"  {cm}")
        print("\n  Full report:")
        print(classification_report(
            y_true, y_pred,
            target_names=[LABEL_GENUINE, LABEL_TAMPERED],
            zero_division=0.0,
        ))

    def save(self, path: Path | None = None) -> Path:
        """Save model + metadata to pickle."""
        if self.model is None:
            raise RuntimeError("Call train() first.")

        path = Path(path) if path else DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "feature_names": self.feature_names,
            "feature_count": len(self.feature_names),
            "label_map": LABEL_MAP,
            "label_map_inv": {str(k): v for k, v in LABEL_MAP_INV.items()},
            "model_version": "1.0.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "random_state": self.random_state,
            "n_estimators": self.model.n_estimators,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "rf_params": {**RF_PARAMS, "random_state": self.random_state},
            "dataset_sha256": self.dataset_sha256,
            "n_training_samples": int(len(self.y)) if self.y is not None else None,
        }
        if self._eval is not None:
            ev = self._eval
            metadata["evaluation"] = {
                "method": ev.method,
                "test_size_fraction": TEST_SIZE,
                "train_size": ev.train_size,
                "test_size": ev.test_size,
                "accuracy": ev.accuracy,
                "precision": ev.precision,
                "recall": ev.recall,
                "f1": ev.f1,
                "confusion_matrix": ev.confusion.tolist(),
                "test_files": list(ev.test_files),
            }

        bundle = {
            "model": self.model,
            "metadata": metadata,
        }

        with open(path, "wb") as f:
            pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\nModel saved: {path} ({path.stat().st_size:,} bytes)")
        return path

    @staticmethod
    def load(path: Path | None = None) -> tuple[RandomForestClassifier, dict]:
        """Load saved model + metadata. Returns (model, metadata)."""
        path = Path(path) if path else DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        with open(path, "rb") as f:
            bundle = pickle.load(f)

        model = bundle["model"]
        metadata = bundle["metadata"]
        print(f"Model loaded: {path}")
        print(f"  Version: {metadata.get('model_version')}")
        print(f"  Features: {metadata.get('feature_count')}")
        print(f"  Trained: {metadata.get('trained_at')}")
        return model, metadata

    def predict(self, image_path: str | Path) -> dict:
        """Run prediction on single image. Returns label + probabilities."""
        if self.model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")

        # A loaded model may use another feature schema (e.g. local-tamper-v2); pick its extractor.
        schema = schema_for_model(self.model, getattr(self, "metadata", None))
        feats = schema.extract(str(image_path))
        X = np.array([schema.vector(feats)], dtype=np.float64)

        pred_class = int(self.model.predict(X)[0])
        probas = self.model.predict_proba(X)[0]

        return {
            "prediction": LABEL_MAP_INV[pred_class],
            "confidence": float(probas[pred_class]),
            "probabilities": {
                LABEL_GENUINE: float(probas[0]),
                LABEL_TAMPERED: float(probas[1]),
            },
            "features": feats,
        }
