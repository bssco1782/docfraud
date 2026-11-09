"""
scripts/train_tamper_model.py
Day 8 (Founder A): train model v2 on the ``local-tamper-v2`` features.

Usage:
    python scripts/train_tamper_model.py                 # grouped CV, fit, save data/models/fraud_model.pkl
    python scripts/train_tamper_model.py --cv-only       # cross-validation report only, nothing saved
    python scripts/train_tamper_model.py --output other.pkl

Training data (the frozen TEST set and the generated HOLDOUT set are never used):
  - the 60 frozen TRAIN images (data/splits/day5_frozen_split.json, SHA-256 verified), and
  - the 150 generated DEV documents (data/generated/dev_v1/, SHA-256 verified against its manifest),
  - plus AUG_COPIES processed copies of every one of those 210 documents, for BOTH classes at the
    same rate: 1-2 of JPEG re-save (q 60-95), resize (0.85-1.15x), brightness and contrast
    (+/-7%). This teaches the model that a re-saved genuine page is not an edited one (Day-5
    lesson: processing applied to one class only becomes a label shortcut). Copies are made
    in a temporary directory with a per-document seed and are not stored.

Model: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42).
The decision threshold stays at TAMPER_THRESHOLD (0.5); nothing is tuned on held-out data.

Cross-validation: repeated stratified 5-fold (3 repeats) over the 210 originals; the copies of a
validation document are never in its training fold. Reported for pristine originals and for
processed copies. Writes logs/training_log_v2.txt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import pickle
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold

from core.model_trainer import DEFAULT_MODEL_PATH, LABEL_MAP, LABEL_MAP_INV
from core.tamper_features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_tamper_features
from scripts.generate_extra_documents import GENERATED, SETS, load_set

SPLIT_PATH = ROOT / "data" / "splits" / "day5_frozen_split.json"
SYNTHETIC = ROOT / "data" / "synthetic"
DEFAULT_LOG = ROOT / "logs" / "training_log_v2.txt"
MODEL_VERSION = "2.0.0"
RF_PARAMS = {"n_estimators": 300, "class_weight": "balanced"}
RANDOM_STATE = 42
AUG_COPIES = 3
AUG_SEED = 12345
CV_SPLITS, CV_REPEATS = 5, 3
LABELS = ("genuine", "tampered")  # 0, 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Data ────────────────────────────────────────────────────────────────────

def training_documents() -> list[dict]:
    """The 60 frozen TRAIN images + the 150 DEV documents, hash-verified, with no TEST / HOLDOUT overlap."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    index = {r["path"]: r for r in csv.DictReader(open(SYNTHETIC / "index.csv", newline="", encoding="utf-8"))}
    docs = []
    for r in split["train"]:
        path = SYNTHETIC / r["path"]
        if sha256(path) != r["sha256"]:
            raise SystemExit(f"STOP: frozen TRAIN image changed: {r['path']}")
        docs.append({"set": "train", "path": f"synthetic/{r['path']}", "file": path, "label": r["label"],
                     "method": index[r["path"]]["tamper_method"] or "-", "doc_type": index[r["path"]]["doc_type"],
                     "sha256": r["sha256"]})
    for r in load_set("dev"):  # verifies every file against the DEV manifest
        docs.append({"set": "dev", "path": f"generated/{SETS['dev']['dir']}/{r['path']}", "file": r["file"],
                     "label": r["label"], "method": r["tamper_method"] or "-", "doc_type": r["doc_type"],
                     "sha256": sha256(r["file"])})

    excluded = {t["sha256"]: f"frozen TEST {t['path']}" for t in split["test"]}
    holdout = GENERATED / SETS["holdout"]["dir"] / "manifest.json"
    if holdout.exists():
        for name, digest in json.loads(holdout.read_text(encoding="utf-8"))["files"].items():
            excluded[digest] = f"HOLDOUT {name}"
    clash = [f"{d['path']} == {excluded[d['sha256']]}" for d in docs if d["sha256"] in excluded]
    if clash:
        raise SystemExit(f"STOP: training data overlaps held-out data: {clash}")
    if len({d["sha256"] for d in docs}) != len(docs):
        raise SystemExit("STOP: duplicate images in the training data")
    return docs


def process(img: Image.Image, rnd: random.Random, workdir: Path, stem: str) -> tuple[Path, str]:
    """One label-independent processing of a page: 1-2 ops, saved as JPEG when 'jpeg' is drawn."""
    ops = rnd.sample(["jpeg", "resize", "brightness", "contrast"], rnd.choice([1, 2]))
    desc = []
    if "resize" in ops:
        s = rnd.uniform(0.85, 1.15)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
        desc.append(f"resize {s:.2f}")
    if "brightness" in ops:
        s = rnd.uniform(0.93, 1.07)
        img = ImageEnhance.Brightness(img).enhance(s)
        desc.append(f"brightness {s:.2f}")
    if "contrast" in ops:
        s = rnd.uniform(0.93, 1.07)
        img = ImageEnhance.Contrast(img).enhance(s)
        desc.append(f"contrast {s:.2f}")
    if "jpeg" in ops:
        q = rnd.randint(60, 95)
        path = workdir / f"{stem}.jpg"
        img.save(path, quality=q)
        desc.append(f"jpeg q{q}")
    else:
        path = workdir / f"{stem}.png"
        img.save(path)
    return path, " + ".join(desc)


def build_matrix(docs: list[dict], copies: int = AUG_COPIES, verbose: bool = True) -> dict:
    """Features of the originals and of `copies` processed copies of each (grouped by document index)."""
    X, y, groups, is_copy, ops = [], [], [], [], []
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for gi, d in enumerate(docs):
            label = LABEL_MAP[d["label"]]
            feats = extract_tamper_features(d["file"])
            X.append([feats[n] for n in FEATURE_NAMES]); y.append(label); groups.append(gi)
            is_copy.append(False); ops.append("original")
            rnd = random.Random(AUG_SEED + gi)
            with Image.open(d["file"]) as im:
                im = im.convert("RGB")
                for k in range(copies):
                    path, desc = process(im, rnd, work, f"{gi}_{k}")
                    feats = extract_tamper_features(path)
                    X.append([feats[n] for n in FEATURE_NAMES]); y.append(label); groups.append(gi)
                    is_copy.append(True); ops.append(desc)
            if verbose and (gi + 1) % 50 == 0:
                print(f"  {gi + 1}/{len(docs)} documents ({time.perf_counter() - t0:.0f} s)")
    return {"X": np.array(X, dtype=np.float64), "y": np.array(y, dtype=np.int32), "groups": np.array(groups),
            "is_copy": np.array(is_copy), "ops": ops}


def fit(X: np.ndarray, y: np.ndarray, seed: int = RANDOM_STATE, n_estimators: int | None = None
        ) -> RandomForestClassifier:
    params = {**RF_PARAMS, **({"n_estimators": n_estimators} if n_estimators else {})}
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1).fit(X, y)


# ── Cross-validation ────────────────────────────────────────────────────────

def grouped_cv(docs: list[dict], data: dict) -> dict:
    X, y, groups, is_copy, ops = data["X"], data["y"], data["groups"], data["is_copy"], data["ops"]
    orig = np.flatnonzero(~is_copy)
    y_orig = y[orig]
    folds, recall, by_op = [], defaultdict(list), defaultdict(lambda: [0, 0])
    pooled = np.zeros(len(orig))
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    for tr_docs, va_docs in cv.split(orig, y_orig):
        train_rows = np.isin(groups, tr_docs)
        assert not np.isin(groups[train_rows], va_docs).any()  # no copy of a validation document is trained on
        model = fit(X[train_rows], y[train_rows])
        va = orig[va_docs]
        p = model.predict(X[va])
        pooled[va_docs] += p
        for i, pi in zip(va_docs, p):
            if y_orig[i] == 1:
                recall[docs[i]["method"]].append(int(pi))
        cp = np.flatnonzero(np.isin(groups, va_docs) & is_copy)
        q = model.predict(X[cp])
        for i, qi in zip(cp, q):
            kind = "jpeg" if "jpeg" in ops[i] else ("resize" if "resize" in ops[i] else "brightness/contrast")
            by_op[(kind, int(y[i]))][0] += int(qi)
            by_op[(kind, int(y[i]))][1] += 1
        folds.append({
            "pristine_acc": float((p == y[va]).mean()),
            "pristine_fpr": float(((p == 1) & (y[va] == 0)).sum() / max((y[va] == 0).sum(), 1)),
            "pristine_fnr": float(((p == 0) & (y[va] == 1)).sum() / max((y[va] == 1).sum(), 1)),
            "processed_acc": float((q == y[cp]).mean()),
            "processed_fpr": float(((q == 1) & (y[cp] == 0)).sum() / max((y[cp] == 0).sum(), 1)),
        })
    mean = {k: statistics.fmean(f[k] for f in folds) for k in folds[0]}
    std = {k: statistics.pstdev(f[k] for f in folds) for k in folds[0]}
    majority = pooled / CV_REPEATS >= 0.5
    sets = {s: float((majority[[d["set"] == s for d in docs]] == y_orig[[d["set"] == s for d in docs]]).mean())
            for s in ("train", "dev")}
    return {
        "method": f"repeated stratified {CV_SPLITS}-fold x {CV_REPEATS}, grouped by document "
                  f"(copies of a validation document never in its training fold)",
        "n_documents": int(len(orig)), "n_folds": len(folds), "mean": mean, "std": std,
        "recall_by_method": {k: [int(sum(v)), len(v)] for k, v in sorted(recall.items())},
        "processed_by_op": {f"{k} {LABELS[c]}": v for (k, c), v in sorted(by_op.items())},
        "accuracy_by_set_majority_vote": sets,
    }


def cv_report(cv: dict) -> list[str]:
    m, s = cv["mean"], cv["std"]
    L = [f"Method: {cv['method']}; {cv['n_documents']} documents, {cv['n_folds']} folds",
         f"Pristine documents:  accuracy {m['pristine_acc']:.3f} +/- {s['pristine_acc']:.3f}, "
         f"genuine flagged (FPR) {m['pristine_fpr']:.3f}, tampered missed (FNR) {m['pristine_fnr']:.3f}",
         f"Processed copies:    accuracy {m['processed_acc']:.3f} +/- {s['processed_acc']:.3f}, "
         f"genuine flagged (FPR) {m['processed_fpr']:.3f}",
         "Tampered recall by method (pristine, all repeats): "
         + ", ".join(f"{k} {a}/{b}" for k, (a, b) in cv["recall_by_method"].items()),
         "Processed copies flagged as tampered, by operation: "
         + ", ".join(f"{k} {a}/{b}" for k, (a, b) in cv["processed_by_op"].items()),
         "Accuracy by source (majority vote over repeats): "
         + ", ".join(f"{k} {v:.3f}" for k, v in cv["accuracy_by_set_majority_vote"].items())]
    return L


# ── Main ────────────────────────────────────────────────────────────────────

def environment() -> dict:
    return {d: importlib.metadata.version(d) for d in ("scikit-learn", "numpy", "opencv-python", "pillow", "scipy")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train model v2 (local-tamper-v2 features)")
    ap.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--cv-only", action="store_true", help="report cross-validation, save nothing")
    args = ap.parse_args()

    docs = training_documents()
    n_t = sum(d["label"] == "tampered" for d in docs)
    print(f"Training documents: {len(docs)} ({len(docs) - n_t} genuine / {n_t} tampered); "
          f"extracting features for originals + {AUG_COPIES} processed copies each...")
    data = build_matrix(docs)
    print("Cross-validating...")
    cv = grouped_cv(docs, data)
    lines = cv_report(cv)
    print("\n".join(lines))
    if args.cv_only:
        return 0

    model = fit(data["X"], data["y"])
    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])
    metadata = {
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_map": LABEL_MAP,
        "label_map_inv": {str(k): v for k, v in LABEL_MAP_INV.items()},
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RANDOM_STATE,
        "n_estimators": RF_PARAMS["n_estimators"],
        "rf_params": {**RF_PARAMS, "random_state": RANDOM_STATE},
        "n_training_samples": len(docs),
        "n_training_rows": int(len(data["y"])),
        "training_sets": {
            "train": "data/splits/day5_frozen_split.json TRAIN list (60 images of data/synthetic/)",
            "dev": f"data/generated/{SETS['dev']['dir']}/ (seed {SETS['dev']['seed']}, 150 documents)",
        },
        "training_files": {d["path"]: d["sha256"] for d in docs},
        "excluded_from_training": ["frozen TEST set (15 images)", f"data/generated/{SETS['holdout']['dir']}/"],
        "augmentation": {"copies_per_document": AUG_COPIES, "seed": AUG_SEED, "classes": "both, same rate",
                         "ops": "1-2 of: JPEG q60-95, resize 0.85-1.15x, brightness 0.93-1.07, contrast 0.93-1.07"},
        "decision_threshold": 0.5,
        "cross_validation": cv,
        "environment": environment(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"model": model, "metadata": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
    digest = sha256(args.output)

    log = [
        "--------------------------------",
        "DOCFRAUD MODEL v2 TRAINING LOG",
        "--------------------------------",
        f"Generated (UTC): {metadata['trained_at']}",
        "Script: scripts/train_tamper_model.py",
        f"Environment: {', '.join(f'{k} {v}' for k, v in metadata['environment'].items())}",
        "",
        f"Model: {args.output.relative_to(ROOT).as_posix() if args.output.is_relative_to(ROOT) else args.output}",
        f"  version {MODEL_VERSION}, feature schema {FEATURE_SCHEMA_VERSION} ({len(FEATURE_NAMES)} features)",
        f"  RandomForest {metadata['rf_params']}",
        f"  SHA-256 {digest}",
        "",
        "Training data (frozen TEST and generated HOLDOUT excluded; checked by SHA-256):",
        f"  TRAIN: 60 frozen-split images; DEV: 150 generated documents (seed {SETS['dev']['seed']})",
        f"  {len(docs)} documents ({len(docs) - n_t} genuine / {n_t} tampered) + {AUG_COPIES} processed copies each "
        f"= {len(data['y'])} rows",
        f"  Processing (both classes, same rate): {metadata['augmentation']['ops']}",
        "",
        "Cross-validation (training data only):",
        *[f"  {line}" for line in lines],
        "",
        "Feature importances (top 10):",
        *[f"  {n:<32} {v:.4f}" for n, v in importances[:10]],
        "",
        "Decision threshold: 0.5 (TAMPER_THRESHOLD, unchanged). Scores are uncalibrated RF vote fractions.",
        "Held-out evaluation: scripts/model_validation.py (frozen TEST + generated HOLDOUT).",
        "",
    ]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text("\n".join(log), encoding="utf-8")
    print(f"\nModel saved: {args.output} (SHA-256 {digest[:16]}...)\nLog written: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
