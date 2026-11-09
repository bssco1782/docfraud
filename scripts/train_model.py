"""
scripts/train_model.py
Train the fraud detection model. No FastAPI dependency.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --dataset-dir data/synthetic
    python scripts/train_model.py --model-path data/models/fraud_model.pkl
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.model_trainer import (
    DEFAULT_DATASET_DIR,
    DEFAULT_MODEL_PATH,
    FEATURE_SCHEMA_VERSION,
    RF_PARAMS,
    DocumentFraudDetector,
)

DEFAULT_LOG_PATH = ROOT / "logs" / "training_log.txt"


def write_training_log(log_path: Path, detector: DocumentFraudDetector, summary, ev, model_path: Path) -> None:
    """Append one run's measured results to the training log."""
    cm = ev.confusion
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    lines = [
        "=" * 60,
        f"Date (UTC):        {datetime.now(timezone.utc).isoformat()}",
        f"Dataset sha256:    {detector.dataset_sha256}",
        f"Dataset size:      {summary.usable} usable of {summary.total} (skipped {summary.skipped})",
        f"Genuine:           {summary.genuine}",
        f"Tampered:          {summary.tampered}",
        f"Evaluation:        {ev.method}",
        f"Train size:        {ev.train_size}",
        f"Test size:         {ev.test_size}",
        f"Random seed:       {detector.random_state}",
        f"Feature count:     {summary.n_features}",
        f"Feature schema:    {FEATURE_SCHEMA_VERSION}",
        f"Feature names:     {', '.join(detector.feature_names)}",
        f"RF params:         {dict(RF_PARAMS, random_state=detector.random_state)} (other params = sklearn defaults)",
        f"Accuracy:          {ev.accuracy:.4f}",
        f"Precision:         {ev.precision:.4f}  (positive class = tampered)",
        f"Recall:            {ev.recall:.4f}",
        f"F1-score:          {ev.f1:.4f}",
        "Confusion matrix (rows=actual, cols=predicted; [genuine, tampered]):",
        f"                   {cm.tolist()}",
        f"TP={tp} TN={tn} FP={fp} FN={fn}",
        f"Test files:        {', '.join(ev.test_files)}",
        f"Model path:        {model_path}",
        "Saved model is refit on all usable samples after holdout evaluation.",
        "",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nTraining log appended: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--no-fixtures-fallback", action="store_true",
                        help="Fail if primary dataset missing instead of using fixtures")
    args = parser.parse_args()

    detector = DocumentFraudDetector()

    # 1. Load dataset
    print("=" * 50)
    print("STEP 1: Load dataset")
    print("=" * 50)
    try:
        summary = detector.load_dataset(
            dataset_dir=args.dataset_dir,
            use_fixtures_fallback=not args.no_fixtures_fallback,
        )
    except RuntimeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    # 2. Train
    print("\n" + "=" * 50)
    print("STEP 2: Train model")
    print("=" * 50)
    detector.train()

    # 3. Evaluate
    print("\n" + "=" * 50)
    print("STEP 3: Evaluate")
    print("=" * 50)
    ev = detector.evaluate()

    # 4. Save
    print("\n" + "=" * 50)
    print("STEP 4: Save model")
    print("=" * 50)
    saved_path = detector.save(args.model_path)
    write_training_log(args.log_path, detector, summary, ev, saved_path)

    # 5. Verify reload + prediction
    print("\n" + "=" * 50)
    print("STEP 5: Verify saved model")
    print("=" * 50)
    model, metadata = DocumentFraudDetector.load(saved_path)
    print("  Reload: OK")

    # Quick prediction on first available image
    from core.model_trainer import FIXTURES_DIR, IMAGE_SUFFIXES
    test_img = None
    for p in sorted(FIXTURES_DIR.iterdir()):
        if p.suffix.lower() in IMAGE_SUFFIXES and "corrupt" not in p.stem:
            test_img = p
            break

    if test_img:
        det2 = DocumentFraudDetector()
        det2.model = model
        result = det2.predict(test_img)
        print(f"  Test prediction on {test_img.name}:")
        print(f"    Label:      {result['prediction']}")
        print(f"    Confidence: {result['confidence']:.4f}")
        print(f"    P(genuine): {result['probabilities']['genuine']:.4f}")
        print(f"    P(tampered):{result['probabilities']['tampered']:.4f}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Model: {saved_path}")
    print(f"  Accuracy:  {ev.accuracy:.4f}")
    print(f"  Precision: {ev.precision:.4f}")
    print(f"  Recall:    {ev.recall:.4f}")
    print(f"  F1:        {ev.f1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
