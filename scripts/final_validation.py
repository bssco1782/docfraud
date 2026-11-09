"""
scripts/final_validation.py
Day 6 final model validation on the frozen 15-image test set - model v1 (day4-v2-22) only.

Since Day 8 the deployed model is v2 (local-tamper-v2), validated by
scripts/model_validation.py. Run against a non-v1 model, this script only prints that and
exits with code 3; its helpers (integrity checks, metrics) are reused by the Day-8 scripts.
logs/final_validation_report.txt is the Day-6 record for model v1.

Usage:
    python scripts/final_validation.py
    python scripts/final_validation.py --report logs/final_validation_report.txt

What it does:
  1. Verifies the frozen split (15 test / 60 train, SHA-256 of every file,
     labels, dataset fingerprint). Any test-hash mismatch STOPS the run.
  2. Checks for training leakage: deployed artifact, augmentation manifests,
     feature cache, CV results.
  3. Scores all 15 test images with
       a) HELD-OUT model: the deployed model's exact config (RF_PARAMS,
          random_state, 22 day4 features) fitted in memory on the 60 frozen
          TRAIN images only. This is the only valid held-out measurement.
       b) DEPLOYED artifact (data/models/fraud_model.pkl) - it was refit on
          all 75 images, i.e. it has seen the test set. Reported for
          transparency only; its numbers are NOT a valid accuracy estimate.
  4. Edge cases (small / large / grayscale / low-quality) through the real
     API inference path (backend.model_utils.ModelAPI). Temporary images are
     made from a TRAIN image in a temp dir; the test set is never touched.
  5. Writes logs/final_validation_report.txt.

Nothing here trains on, tunes on, or modifies the frozen test set.
The decision threshold is the RandomForest default (argmax / 0.5), not tuned.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from backend.model_utils import ModelAPI
from core.document_analyzer import FEATURE_NAMES, extract_features, features_to_vector
from core.model_trainer import DEFAULT_MODEL_PATH, FEATURE_SCHEMA_VERSION, RF_PARAMS, DocumentFraudDetector

DATASET = ROOT / "data" / "synthetic"
SPLIT_PATH = ROOT / "data" / "splits" / "day5_frozen_split.json"
AUG_DIR = ROOT / "data" / "augmented"
CACHE_PATH = ROOT / "data" / "cache" / "features_local_v1.pkl"
CV_RESULTS = ROOT / "logs" / "day5_ml_recovery_results.json"
DEFAULT_REPORT = ROOT / "logs" / "final_validation_report.txt"
LABELS = ("genuine", "tampered")  # 0, 1


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── 1. Frozen test-set integrity ────────────────────────────────────────────

def check_integrity(split: dict) -> list[tuple[str, bool, str]]:
    test, train = split["test"], split["train"]
    checks = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    add("exactly 15 test images", len(test) == 15, f"n={len(test)}")
    add("exactly 60 train images", len(train) == 60, f"n={len(train)}")
    test_paths = {r["path"] for r in test}
    add("test/train paths disjoint", not test_paths & {r["path"] for r in train})
    add("test/train hashes disjoint", not {r["sha256"] for r in test} & {r["sha256"] for r in train})
    add("labels match folder", all(r["path"].split("/")[0] == r["label"] for r in test + train))

    bad = [r["path"] for r in test if not (DATASET / r["path"]).is_file() or sha256(DATASET / r["path"]) != r["sha256"]]
    add("all 15 test hashes match frozen split", not bad, ", ".join(bad) or "15/15 match")
    if bad:
        # Hard stop: do not score, regenerate or replace anything.
        raise SystemExit(f"STOP: frozen test image missing or modified: {bad}")

    bad_train = [r["path"] for r in train if sha256(DATASET / r["path"]) != r["sha256"]]
    add("all 60 train hashes match frozen split", not bad_train, ", ".join(bad_train) or "60/60 match")

    # Same fingerprint as DocumentFraudDetector.load_dataset().
    digests = {sha256(p): f"{lab}/{p.name}" for lab in LABELS for p in sorted((DATASET / lab).iterdir())
               if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    fp = hashlib.sha256("\n".join(sorted(f"{n}:{d}" for d, n in digests.items())).encode()).hexdigest()
    add("dataset fingerprint matches frozen split", fp == split["dataset_sha256"], fp[:16])
    return checks


# ── 2. Training leakage ─────────────────────────────────────────────────────

def check_leakage(split: dict, deployed_meta: dict) -> list[tuple[str, bool, str]]:
    test_paths = {r["path"] for r in split["test"]}
    test_hashes = {r["sha256"] for r in split["test"]}
    checks = []

    n_deployed = deployed_meta.get("n_training_samples")
    ev_files = set(deployed_meta.get("evaluation", {}).get("test_files", []))
    checks.append((
        "deployed artifact not trained on test images",
        n_deployed is not None and n_deployed <= len(split["train"]),
        f"fraud_model.pkl n_training_samples={n_deployed} (dataset = 75 incl. all 15 test images; "
        "refit on full data after Day-5 holdout)",
    ))
    checks.append(("deployed artifact's Day-5 holdout files == frozen test list", ev_files == test_paths,
                   f"{len(ev_files & test_paths)}/15 overlap"))

    for manifest in sorted(AUG_DIR.glob("*/manifest.csv")):
        rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
        hit = [r["file"] for r in rows if r["source_path"] in test_paths
               or r["source_sha256"] in test_hashes or r["sha256"] in test_hashes]
        checks.append((f"no test image in augmentation {manifest.parent.name}", not hit,
                       f"{len(rows)} rows, {len(hit)} hits"))

    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            keys = [k for k in pickle.load(f) if isinstance(k, str)]
        hit = [k for k in keys if any(k.endswith(p) for p in test_paths)]
        checks.append(("no test image in feature cache", not hit, f"{len(keys)} keys, {len(hit)} hits"))

    if CV_RESULTS.exists():
        res = json.loads(CV_RESULTS.read_text(encoding="utf-8"))
        cv_text = json.dumps({k: v for k, v in res.items() if k.startswith("cv")})
        hit = sorted(p for p in test_paths if p in cv_text)
        checks.append(("no test image in CV / model-selection results", not hit, f"{len(hit)} hits"))
        checks.append(("frozen test not yet consumed by ml_recovery final", "final" not in res,
                       "local-forensics model never finalized"))
    return checks


# ── 3. Scoring + metrics ────────────────────────────────────────────────────

def feature_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([features_to_vector(extract_features(DATASET / r["path"])) for r in rows], dtype=np.float64)


def compute_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    div = lambda a, b: a / b if b else 0.0
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    m = {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": len(y), "correct": tn + tp,
        "accuracy": div(tn + tp, len(y)), "precision": precision, "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
    }
    # Cross-check hand-computed numbers against sklearn.
    assert tn + fp + fn + tp == len(y)
    assert confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist() == [tn, fp, fn, tp]
    assert np.isclose(m["accuracy"], accuracy_score(y, pred))
    assert np.isclose(m["precision"], precision_score(y, pred, zero_division=0.0))
    assert np.isclose(m["recall"], recall_score(y, pred, zero_division=0.0))
    assert np.isclose(m["f1"], f1_score(y, pred, zero_division=0.0))
    return m


def ascii_confusion(m: dict) -> str:
    return "\n".join([
        "                   Pred Genuine   Pred Tampered",
        f"Actual Genuine     {'TN = ' + str(m['tn']):<14} {'FP = ' + str(m['fp'])}",
        f"Actual Tampered    {'FN = ' + str(m['fn']):<14} {'TP = ' + str(m['tp'])}",
    ])


def per_image_table(rows, index, y, pred, score) -> str:
    out = [f"{'Filename':<28} | {'Doc type':<14} | {'Method':<12} | {'Actual':<8} | {'Predicted':<9} | {'Risk':>5} | Correct",
           "-" * 100]
    for r, a, p, s in zip(rows, y, pred, score):
        meta = index.get(r["path"], {})
        out.append(f"{r['path']:<28} | {meta.get('doc_type', '?'):<14} | {meta.get('tamper_method') or '-':<12} | "
                   f"{LABELS[a]:<8} | {LABELS[p]:<9} | {s:>5.2f} | {'YES' if a == p else 'NO'}")
    return "\n".join(out)


def score(model, X) -> tuple[np.ndarray, np.ndarray]:
    return model.predict(X).astype(int), model.predict_proba(X)[:, 1]


# ── 4. Edge cases ───────────────────────────────────────────────────────────

def run_edge_cases(api: ModelAPI, source: Path) -> dict[str, dict]:
    """Every case must return a finite risk score in [0, 1] or raise a controlled ValueError."""
    base = Image.open(source).convert("RGB")
    cases = {
        "small": [("80x60 px", lambda d: _save(base.resize((80, 60)), d / "small.png")),
                  ("16x16 px", lambda d: _save(base.resize((16, 16)), d / "tiny.png"))],
        "large": [("2550x3150 px", lambda d: _save(base.resize((2550, 3150), Image.BICUBIC), d / "large.png")),
                  ("4000x5000 px", lambda d: _save(base.resize((4000, 5000), Image.BICUBIC), d / "huge.png"))],
        "grayscale": [("mode L", lambda d: _save(base.convert("L"), d / "gray.png")),
                      ("mode LA", lambda d: _save(base.convert("LA"), d / "gray_alpha.png"))],
        "low_quality": [("0.5x + blur + JPEG q=5", lambda d: _save(
            base.resize((base.width // 2, base.height // 2)).filter(ImageFilter.GaussianBlur(1.5)),
            d / "lowq.jpg", quality=5)),
                        ("JPEG q=1", lambda d: _save(base, d / "q1.jpg", quality=1))],
    }
    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for case, variants in cases.items():
            runs = []
            for desc, make in variants:
                path = make(tmp)
                run = {"variant": desc, "file_bytes": path.stat().st_size}
                tracemalloc.start()
                t0 = time.perf_counter()
                try:
                    vec = features_to_vector(extract_features(path))
                    out = api.predict(path)
                    risk = out["tamper_risk_score"]
                    ok = len(vec) == len(FEATURE_NAMES) and all(np.isfinite(vec)) and 0.0 <= risk <= 1.0
                    run.update(status="PASS" if ok else "FAIL", outcome="processed", risk=risk,
                               detail="" if ok else f"invalid features/score (n={len(vec)}, risk={risk})")
                except ValueError as exc:  # controlled rejection is acceptable
                    run.update(status="PASS", outcome="controlled error", detail=str(exc)[:120])
                except Exception as exc:  # uncontrolled crash
                    run.update(status="FAIL", outcome="crash", detail=f"{type(exc).__name__}: {exc}"[:160])
                run["seconds"] = time.perf_counter() - t0
                run["peak_mb"] = tracemalloc.get_traced_memory()[1] / 2**20
                tracemalloc.stop()
                runs.append(run)
            results[case] = {"status": "PASS" if all(r["status"] == "PASS" for r in runs) else "FAIL", "runs": runs}
    # Reference: the untouched source image through the same path.
    results["_reference_risk"] = api.predict(source)["tamper_risk_score"]
    return results


def _save(img: Image.Image, path: Path, **kw) -> Path:
    img.save(path, **kw)
    return path


# ── 5. Report ───────────────────────────────────────────────────────────────

def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def metric_block(m: dict) -> list[str]:
    return [
        f"Accuracy:            {pct(m['accuracy'])}  ({m['correct']}/{m['n']})",
        f"Precision:           {pct(m['precision'])}  (positive class = tampered)",
        f"Recall:              {pct(m['recall'])}",
        f"F1:                  {m['f1']:.4f}",
        "",
        f"True Negatives:      {m['tn']}",
        f"False Positives:     {m['fp']}",
        f"False Negatives:     {m['fn']}",
        f"True Positives:      {m['tp']}",
        "",
        f"False Positive Rate: {pct(m['fpr'])}  ({m['fp']}/{m['fp'] + m['tn']} genuine flagged)",
        f"False Negative Rate: {pct(m['fnr'])}  ({m['fn']}/{m['fn'] + m['tp']} tampered missed)",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 6 final validation on the frozen test set")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    index = {r["path"]: r for r in csv.DictReader(open(DATASET / "index.csv", newline="", encoding="utf-8"))}

    integrity = check_integrity(split)  # raises SystemExit on any test-hash mismatch
    deployed, deployed_meta = DocumentFraudDetector.load(DEFAULT_MODEL_PATH)
    schema = deployed_meta.get("feature_schema_version") or FEATURE_SCHEMA_VERSION
    if schema != FEATURE_SCHEMA_VERSION:
        print(f"Deployed model {deployed_meta.get('model_version')} uses feature schema {schema}; this Day-6 "
              f"script validates model v1 ({FEATURE_SCHEMA_VERSION}) only. Use scripts/model_validation.py.")
        return 3
    leakage = check_leakage(split, deployed_meta)
    integrity_ok = all(ok for _, ok, _ in integrity)
    deployed_leak_ok = leakage[0][1]
    pipeline_leak_ok = all(ok for name, ok, _ in leakage[1:] if "ml_recovery final" not in name)

    train, test = split["train"], split["test"]
    assert {r["path"] for r in train}.isdisjoint(r["path"] for r in test)
    y_train = np.array([LABELS.index(r["label"]) for r in train])
    y_test = np.array([LABELS.index(r["label"]) for r in test])

    print("Extracting features (60 train + 15 test)...")
    X_train, X_test = feature_matrix(train), feature_matrix(test)
    assert X_test.shape == (15, len(FEATURE_NAMES)) and np.isfinite(X_test).all()
    assert deployed.n_features_in_ == len(FEATURE_NAMES)

    # a) Held-out model: deployed config, fitted on the 60 frozen TRAIN images only.
    holdout = RandomForestClassifier(**RF_PARAMS, random_state=deployed_meta.get("random_state", 42), n_jobs=-1)
    holdout.fit(X_train, y_train)
    h_pred, h_score = score(holdout, X_test)
    h = compute_metrics(y_test, h_pred)
    day5_cm = deployed_meta.get("evaluation", {}).get("confusion_matrix")
    reproduces_day5 = day5_cm == [[h["tn"], h["fp"]], [h["fn"], h["tp"]]]

    # b) Deployed artifact (trained on all 75 incl. test) - contaminated.
    d_pred, d_score = score(deployed, X_test)
    d = compute_metrics(y_test, d_pred)

    # Per-image table must agree with the confusion matrix.
    assert sum(int(a == p) for a, p in zip(y_test, h_pred)) == h["correct"]

    print("Running edge cases through ModelAPI...")
    api = ModelAPI(DEFAULT_MODEL_PATH)
    api.load_model()
    edge_src = DATASET / train[0]["path"]
    edge = run_edge_cases(api, edge_src)
    edge_ok = all(v["status"] == "PASS" for k, v in edge.items() if not k.startswith("_"))

    ready = integrity_ok and edge_ok and deployed_leak_ok and h["accuracy"] >= 0.78
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    errors = [(r["path"], LABELS[a], LABELS[p], s) for r, a, p, s in zip(test, y_test, h_pred, h_score) if a != p]

    L = [
        "--------------------------------",
        "DOCFRAUD FINAL MODEL VALIDATION",
        "--------------------------------",
        f"Generated (UTC): {now}",
        "Script: scripts/final_validation.py",
        "",
        "Model:",
        f"  Deployed artifact (loaded by API): {DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix()}",
        f"  RandomForest {deployed_meta.get('rf_params')} (other params = sklearn defaults)",
        f"  Trained on {deployed_meta.get('n_training_samples')} images at {deployed_meta.get('trained_at')}",
        "  Held-out replica used for metrics: same class / params / seed / features,",
        "  fitted in memory on the 60 frozen TRAIN images only (not saved).",
        "  Not validated: local-forensics-v1 model (fraud_model_local_v1.pkl) - never",
        "  finalized; Day-5 CV stopped before grid/final, awaiting review.",
        "",
        "Model version:",
        f"  {deployed_meta.get('model_version')} / feature schema {deployed_meta.get('feature_schema_version')} "
        f"({len(FEATURE_NAMES)} features; code schema {FEATURE_SCHEMA_VERSION})",
        "",
        "Validation dataset:",
        f"  Frozen 15-image test set ({split['name']}, dataset_sha256 {split['dataset_sha256'][:16]}...)",
        "",
        f"Test images:\n  {len(test)} ({int((y_test == 0).sum())} genuine / {int((y_test == 1).sum())} tampered)",
        "",
        f"Test-set integrity:\n  {'PASS' if integrity_ok else 'FAIL'}",
    ]
    L += [f"    [{'x' if ok else ' '}] {n}{' - ' + dt if dt else ''}" for n, ok, dt in integrity]
    L += [
        "",
        "Training leakage:",
        f"  Held-out replica (metrics below): {'PASS' if pipeline_leak_ok else 'FAIL'} - trained on 60 TRAIN images only",
        f"  Deployed artifact fraud_model.pkl: {'PASS' if deployed_leak_ok else 'FAIL'} - trained on all 75 images,",
        "    including all 15 frozen test images. Its test-set numbers are NOT valid.",
    ]
    L += [f"    [{'x' if ok else ' '}] {n}{' - ' + dt if dt else ''}" for n, ok, dt in leakage]
    L += [
        "",
        "--------------------------------",
        "OVERALL METRICS (held-out replica, valid)",
        "--------------------------------",
        *metric_block(h),
        "",
        f"Reproduces Day-5 holdout confusion matrix {day5_cm}: {'YES' if reproduces_day5 else 'NO'}",
        "Decision threshold: RandomForest default (0.5 vote), not tuned.",
        "Risk score = RF vote fraction for 'tampered' - uncalibrated, not a fraud probability.",
        "",
        "--------------------------------",
        "CONFUSION MATRIX (held-out replica)",
        "--------------------------------",
        ascii_confusion(h),
        "",
        "--------------------------------",
        "PER-IMAGE RESULTS (held-out replica, all 15)",
        "--------------------------------",
        per_image_table(test, index, y_test, h_pred, h_score),
        "",
        f"Incorrect: {len(errors)}",
        *[f"  {p}: actual {a}, predicted {pr} (risk {s:.2f})" for p, a, pr, s in errors],
        "",
        "--------------------------------",
        "DEPLOYED ARTIFACT ON TEST SET (CONTAMINATED - NOT A VALID MEASURE)",
        "--------------------------------",
        "fraud_model.pkl has seen every test image during training; this only shows",
        "how well it memorised them.",
        f"Accuracy {pct(d['accuracy'])} ({d['correct']}/15), precision {pct(d['precision'])}, "
        f"recall {pct(d['recall'])}, F1 {d['f1']:.4f}",
        ascii_confusion(d),
        "",
        "--------------------------------",
        "EDGE CASE RESULTS",
        "--------------------------------",
        "Path: backend.model_utils.ModelAPI.predict (deployed artifact) + extract_features",
        f"Source image: {edge_src.relative_to(DATASET).as_posix()} (TRAIN split; temp copies only, deleted after run)",
        f"Reference risk of untouched source: {edge['_reference_risk']:.2f}",
        "",
    ]
    names = {"small": "Small image", "large": "Large image", "grayscale": "Grayscale", "low_quality": "Low quality"}
    for key, label in names.items():
        L.append(f"{label}:\n  {edge[key]['status']}")
        for r in edge[key]["runs"]:
            risk = f"risk {r['risk']:.2f}" if "risk" in r else ""
            L.append(f"    - {r['variant']:<24} {r['status']:<4} {r['outcome']:<16} {risk:<10} "
                     f"{r['seconds']:.2f}s  peak {r['peak_mb']:.0f} MB  {r['detail']}")
    L += [
        "",
        "--------------------------------",
        "LIMITATIONS",
        "--------------------------------",
        f"- Held-out accuracy {pct(h['accuracy'])} is below the 78% sprint target. The model flags "
        f"{h['fp']}/{h['fp'] + h['tn']} genuine documents (FPR {pct(h['fpr'])}) - for an NBFC this means "
        "many good applicants sent to manual review.",
        f"- It misses {h['fn']}/{h['fn'] + h['tp']} tampered documents (FNR {pct(h['fnr'])}).",
        "- n=15 (10 genuine / 5 tampered): one image moves accuracy by 6.7 points and recall by 20 points."
        " Wide uncertainty; not a stable estimate.",
        "- The deployed artifact was refit on all 75 images including the test set, so there is no clean"
        " held-out number for the exact bytes in production. The replica has identical config and data"
        " minus the 15 test images.",
        "- The 22 features are whole-image statistics; tampering in this dataset is local. Day-5 CV on train"
        " only gave 58.3% for these features, consistent with this result.",
        "- The better-scoring local-forensics-v1 representation (CV 73.3%) has no finalized model and was"
        " not evaluated here - doing so now would be test-set model selection.",
        "- Data is synthetic (3 templates, 4 simple tamper methods). No real NBFC documents; real-world"
        " performance is unknown.",
        "- Risk score is an uncalibrated RF vote fraction; do not present it as a probability.",
        "- ModelAPI.predict returns flagged_regions=[] always; the API gives no localisation evidence.",
        "- Images above 2000 px are downscaled before feature extraction; very small images are processed"
        " but carry little information, so their scores are not meaningful.",
        f"- Large inputs are slow: {edge['large']['runs'][0]['seconds']:.1f}s for the 2550x3150 case"
        " (feature extraction runs twice there: once directly, once via ModelAPI).",
        "",
        "--------------------------------",
        "READINESS",
        "--------------------------------",
        "Validation run completed:        YES",
        f"Model meets 78% target:          {'YES' if h['accuracy'] >= 0.78 else 'NO'}",
        f"READY FOR DAY-6 VALIDATION:      {'YES' if ready else 'NO'}",
        "PRODUCTION READY:                NO",
        "",
    ]
    report = "\n".join(L)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}")
    return 0 if integrity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
