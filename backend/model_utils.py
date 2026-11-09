"""
backend/model_utils.py
ModelAPI wrapper for document fraud detection model inference.

Besides loading the model and running the pipeline, ModelAPI guards the analyzer:
image dimensions are checked from the file header before any pixel is decoded
(decompression bombs, oversized PDF pages, images too small to analyse), and error
messages returned to clients never contain server-side temporary paths.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from backend import config
from core.document_analyzer import (
    ELA_HIGH_THRESHOLD,
    ELA_MIN_COMPONENT_AREA,
    MAX_IMAGE_SIDE,
)
from core.feature_schemas import FeatureSchema, get_schema
from core.model_trainer import DocumentFraudDetector

logger = logging.getLogger(__name__)

# Make Pillow itself refuse images above the configured pixel budget (it raises
# DecompressionBombError above twice this value and only warns in between).
Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS


class ImageValidationError(ValueError):
    """The upload is a readable file but its dimensions cannot be analysed (client error)."""


class ModelUnavailableError(RuntimeError):
    """The model is missing, failed to load, or is incompatible with the feature extractor."""


def check_image_dimensions(path: Path) -> None:
    """
    Validate image or PDF-page dimensions from the header, without decoding pixels.

    Raises:
        ImageValidationError: too many pixels, or too small after downscaling.
        ValueError: the file cannot be parsed as an image / PDF (corrupted).
    """
    if path.suffix.lower() == ".pdf":
        width, height = _pdf_page_pixels(path)
    else:
        try:
            # Pillow warns between 1x and 2x its pixel limit and raises above 2x; both mean
            # "over MAX_IMAGE_PIXELS", so the warning is turned into the same rejection.
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as img:  # reads the header only
                    width, height = img.size
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ImageValidationError(f"Image dimensions exceed the {config.MAX_IMAGE_PIXELS:,} pixel limit") from exc
        except Exception as exc:
            raise ValueError(f"cannot identify image file: {exc.__class__.__name__}") from exc

    if width * height > config.MAX_IMAGE_PIXELS:
        raise ImageValidationError(
            f"Image dimensions {width}x{height} exceed the {config.MAX_IMAGE_PIXELS:,} pixel limit"
        )
    scale = min(1.0, MAX_IMAGE_SIDE / max(width, height))
    if min(width, height) * scale < config.MIN_IMAGE_SIDE:
        raise ImageValidationError(
            f"Image {width}x{height} is too small or too narrow to analyse "
            f"(shortest side must be at least {config.MIN_IMAGE_SIDE} px after scaling to {MAX_IMAGE_SIDE} px)"
        )


def _pdf_page_pixels(path: Path) -> tuple[int, int]:
    """Return the rendered pixel size of PDF page 1; reject empty or encrypted PDFs."""
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - pymupdf is in requirements.txt
        raise ValueError("PDF support requires pymupdf") from exc
    try:
        with pymupdf.open(str(path)) as doc:
            if doc.needs_pass:
                raise ImageValidationError("Password-protected PDFs are not supported")
            if doc.page_count < 1:
                raise ImageValidationError("PDF has no pages")
            rect = doc[0].rect
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"cannot open PDF: {exc.__class__.__name__}") from exc
    zoom = config.PDF_RENDER_DPI / 72
    return int(rect.width * zoom), int(rect.height * zoom)


class ModelAPI:
    """
    API interface for fraud model loading, feature extraction, and prediction.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else Path(config.MODEL_PATH)
        self.model = None
        self.metadata: Dict[str, Any] = {}
        # Feature extractor matching the loaded model (from metadata["feature_schema_version"]).
        self.schema: Optional[FeatureSchema] = None
        self.is_loaded: bool = False
        self.last_error: Optional[str] = None
        self.load_seconds: Optional[float] = None
        self._lock = threading.Lock()

    def status(self) -> Dict[str, Any]:
        """Model readiness summary for /health."""
        return {
            "loaded": self.is_loaded and self.model is not None,
            "version": self.metadata.get("model_version"),
            "features": getattr(self.model, "n_features_in_", None),
            "error": self.last_error,
        }

    def load_model(self, model_path: str | Path | None = None) -> bool:
        """
        Load trained model weights and metadata from disk in a thread-safe manner.
        """
        target_path = Path(model_path) if model_path else self.model_path
        with self._lock:
            # If already loaded from this path, return immediately
            if self.is_loaded and self.model is not None and target_path == self.model_path:
                return True

            if not target_path.exists():
                logger.error("Model file not found at %s", target_path)
                self.is_loaded = False
                self.last_error = "model file not found"
                return False

            start = time.perf_counter()
            try:
                model, metadata = DocumentFraudDetector.load(target_path)
            except Exception as exc:
                logger.exception("Failed to load model from %s: %s", target_path, exc)
                self.is_loaded = False
                self.last_error = f"model file could not be loaded ({exc.__class__.__name__})"
                raise RuntimeError(f"Failed to load model from {target_path}: {exc}") from exc

            # Startup validation: the model's recorded feature schema must be known, and the model must
            # expect exactly the features that schema's extractor produces, in the same order.
            n_features = getattr(model, "n_features_in_", None)
            try:
                schema = get_schema(metadata.get("feature_schema_version"))
            except ValueError as exc:
                schema, self.last_error = None, str(exc)
            else:
                if n_features != len(schema.names):
                    self.last_error = f"model expects {n_features} features, extractor produces {len(schema.names)}"
                elif metadata.get("feature_names") not in (None, list(schema.names)):
                    self.last_error = f"model feature names do not match schema {schema.version}"
                else:
                    self.last_error = None
            if schema is None or self.last_error:
                self.is_loaded = False
                logger.error("Incompatible model at %s: %s", target_path, self.last_error)
                raise RuntimeError(f"Incompatible model at {target_path}: {self.last_error}")

            # One document per call: joblib parallelism inside a single prediction only adds thread
            # start-up cost (measured 184 ms vs 42 ms per document for the 300-tree v2 forest) and
            # competes with the MAX_CONCURRENT_ANALYSES request slots. Outputs are identical.
            if getattr(model, "n_jobs", None) not in (None, 1):
                model.set_params(n_jobs=1)

            self.model, self.metadata, self.schema = model, metadata, schema
            self.model_path = target_path
            self.is_loaded = True
            self.last_error = None
            self.load_seconds = time.perf_counter() - start
            logger.info(
                "Model loaded from %s (version %s, schema %s, %d features, %.2f s)",
                target_path,
                metadata.get("model_version"),
                schema.version,
                n_features,
                self.load_seconds,
            )
            return True

    def extract_flagged_regions(
        self,
        heatmap: np.ndarray,
        threshold: int = ELA_HIGH_THRESHOLD,
        min_area: int = ELA_MIN_COMPONENT_AREA,
    ) -> List[Dict[str, Any]]:
        """
        Extract bounding boxes of anomalous ELA hotspots from an ELA heatmap.

        Not called by predict(). /verify regions come from the model's feature schema
        (core.tamper_features.extract_with_regions for local-tamper-v2); v1 models return none.
        """
        ela_gray = heatmap.mean(axis=2).astype(np.uint8)
        binary = (ela_gray > threshold).astype(np.uint8)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        flagged: List[Dict[str, Any]] = []
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                w = int(stats[label, cv2.CC_STAT_WIDTH])
                h = int(stats[label, cv2.CC_STAT_HEIGHT])
                flagged.append({"x": x, "y": y, "width": w, "height": h, "area": area, "type": "ela_hotspot"})
        return flagged

    def predict(
        self,
        image_input: Union[str, Path, bytes],
        filename: str = "upload.png",
    ) -> Dict[str, Any]:
        """
        Execute full fraud detection pipeline:
        1. Parse input image
        2. Extract the features of the model's schema (local-tamper-v2: 29 local tamper features),
           and the anomalous regions when the schema can localise edits
        3. Model inference to compute tamper risk score and confidence
        4. Return structured VerifyResponse dictionary
        """
        # Lazy-load on first use, and retry after a failed startup load (graceful recovery once
        # the model file is fixed). load_model() acquires self._lock itself and re-checks
        # is_loaded, so it must not be called while holding the lock (threading.Lock is not
        # re-entrant and would deadlock).
        if not self.is_loaded or self.model is None:
            try:
                self.load_model()
            except RuntimeError as exc:
                raise ModelUnavailableError(self.last_error or "model could not be loaded") from exc

        if self.model is None or self.schema is None:
            raise ModelUnavailableError(self.last_error or "model is not loaded")

        cleanup = False
        target_path: Path

        if isinstance(image_input, (str, Path)):
            target_path = Path(image_input)
            if not target_path.exists():
                raise ValueError(f"Image path does not exist: {target_path}")
        elif isinstance(image_input, bytes):
            suffix = Path(filename).suffix.lower()
            if suffix not in config.ALLOWED_EXTENSIONS:
                suffix = ".png"
            temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_file.write(image_input)
            temp_file.flush()
            temp_file.close()
            target_path = Path(temp_file.name)
            cleanup = True
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        try:
            # 0. Check dimensions from the header before decoding anything
            check_image_dimensions(target_path)

            # 1. Extract forensic features. Client-facing messages must not reveal the
            #    server's temporary path, so it is replaced by the uploaded file name.
            try:
                if self.schema.extract_with_regions is not None:
                    feats, regions = self.schema.extract_with_regions(target_path)
                else:  # e.g. v1 (day4-v2-22): no localisation
                    feats, regions = self.schema.extract(target_path), []
            except ValueError as exc:
                raise ValueError(str(exc).replace(str(target_path), Path(filename).name)) from exc
            X = np.array([self.schema.vector(feats)], dtype=np.float64)

            # 2. Model prediction & class probabilities (one forest pass: the predicted class is
            #    the argmax of the probabilities, exactly as RandomForestClassifier.predict does)
            probas = self.model.predict_proba(X)[0]
            pred_class = int(np.argmax(probas))

            tamper_risk_score = round(float(probas[1]), 4)
            confidence = round(float(probas[pred_class]), 4)

            # 3. Flagged regions explain a tampering verdict, so they are returned only when the
            #    score reaches TAMPER_THRESHOLD; a document reported as genuine gets none.
            flagged: bool = tamper_risk_score >= config.TAMPER_THRESHOLD
            flagged_regions: List[Dict[str, Any]] = regions if flagged else []
            if "ela_hotspot_count" in feats:  # Day-4 (v1) models report their ELA hotspot count
                hotspots = f" {int(feats['ela_hotspot_count'])} anomalous hotspot region(s) detected."
            elif flagged_regions:
                hotspots = f" {len(flagged_regions)} suspicious region(s) located (see flagged_regions)."
            else:
                hotspots = ""

            if flagged:
                interpretation = (
                    f"Evidence of digital tampering detected "
                    f"(risk score: {tamper_risk_score:.2f}, confidence: {confidence:.2f}).{hotspots}"
                )
            else:
                interpretation = (
                    f"Document verified successfully. Low risk of digital tampering detected "
                    f"(risk score: {tamper_risk_score:.2f}, confidence: {confidence:.2f})."
                )

            return {
                "tamper_risk_score": tamper_risk_score,
                "confidence": confidence,
                "flagged_regions": flagged_regions,
                "interpretation": interpretation,
            }
        finally:
            if cleanup and target_path.exists():
                try:
                    target_path.unlink()
                except OSError:
                    pass


# Module-level default instance
model_api = ModelAPI()
