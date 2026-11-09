"""
scripts/model_validation.py
Day 8 (Founder A) model validation: model integrity, demo images, held-out accuracy, checklist.

Usage:
    python scripts/model_validation.py
    python scripts/model_validation.py --runs 10 --report logs/model_validation_report.txt

Steps (all on the deployed data/models/fraud_model.pkl, through the API's own loader):
  1. Model integrity - every model file in data/models/ loads; fraud_model.pkl's SHA-256 equals
     the record in data/demo/demo_manifest.json; the recorded feature schema is known and matches
     the model; the forest is complete; ModelAPI loads it; every training file recorded in the
     metadata still matches its SHA-256.
  2. Demo validation - each demo image through ModelAPI.predict(bytes), the call /verify makes:
     expected label, predicted label, confidence, risk score, PASS/FAIL.
  3. Stability - every demo image scored --runs times (model_stability_test.repeatability); each
     run within +/-0.02 of the image's mean and the label never changes. Full analysis:
     scripts/model_stability_test.py.
  4. Held-out accuracy - the deployed file itself, scored through ModelAPI on
       TEST     the 15 frozen test images (data/splits/day5_frozen_split.json), and
       HOLDOUT  data/generated/holdout_v1/ (generated after the model was saved).
     Leakage checks first: no held-out file is among the model's recorded training files, and
     HOLDOUT was created after the model was trained. Accuracy, precision, recall, F1, confusion
     matrices and 95% intervals, compared with the earlier recorded results.
  5. Inference robustness - test fixtures plus corrupt, truncated, empty and non-image uploads.
     A ValueError is a controlled rejection (HTTP 400); any other exception is a crash.
  6. Error analysis - the generator runs are replayed to measure how many pixels each missed
     edit changed (an edit on blank paper can change nothing at all).
  7. Final model checklist.

Nothing is trained, tuned or selected here; the threshold is TAMPER_THRESHOLD (0.5). Nothing
under data/ is written. Models without recorded training files (v1) are validated with
scripts/final_validation.py (Day-6 held-out replica method) instead.

Exit code: 0 if every checklist item passes, 1 if any item fails, 2 if the frozen test set
changed (the run stops before scoring).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_validation as fv
import model_stability_test as mst
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from backend import config
from backend.model_utils import ModelAPI
from core.feature_schemas import get_schema
from core.model_trainer import DEFAULT_MODEL_PATH, DocumentFraudDetector
from scripts.generate_extra_documents import GENERATED, SETS, edit_footprints, load_set

FIXTURES = ROOT / "tests" / "fixtures"
DAY6_REPORT = ROOT / "logs" / "final_validation_report.txt"
DEFAULT_REPORT = ROOT / "logs" / "model_validation_report.txt"
DEFAULT_JSON = ROOT / "logs" / "model_validation_results.json"
REQUIRED_ACCURACY = 0.75  # Day 8 requirement: held-out accuracy above 75%
SPRINT_TARGET = 0.78      # original sprint target
LABELS = mst.LABELS

# Earlier recorded results (model v1 and its predecessor), for the comparison table.
PREVIOUS = [
    {"when": "Day 3", "model": "v0 (15 features, 25 docs)", "set": "5-image test", "n": 5, "correct": 4,
     "accuracy": 0.8, "precision": 0.5, "recall": 1.0, "f1": 2 / 3,
     "source": "PROGRESS.md Day 3", "note": "1 tampered test image; model no longer exists"},
    {"when": "Day 5", "model": "v1 (day4-v2-22)", "set": "frozen TEST", "n": 15, "correct": 9,
     "accuracy": 0.6, "precision": 4 / 9, "recall": 0.8, "f1": 0.5714,
     "source": "v1 model metadata / PROGRESS.md Day 5", "note": "training-time holdout"},
    {"when": "Day 8", "model": "v1 (day4-v2-22)", "set": "frozen TEST", "n": 15, "correct": 9,
     "accuracy": 0.6, "precision": 4 / 9, "recall": 0.8, "f1": 0.5714,
     "source": "PROGRESS.md Day 8 (v1 revalidation)", "note": "same as Day 5 and Day 6"},
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ── 1. Model integrity ──────────────────────────────────────────────────────

def model_integrity(manifest: dict) -> tuple[list[tuple[str, bool, str]], object, dict, ModelAPI | None]:
    """Integrity checks, plus (deployed model, metadata, loaded ModelAPI) - None when loading fails."""
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    path = DEFAULT_MODEL_PATH
    others = sorted(p for p in path.parent.glob("*.pkl") if p != path)
    if not others:
        add("other model files in data/models/", True, "none - fraud_model.pkl is the only model file")
    for other in others:  # every model file present must at least unpickle
        try:
            with open(other, "rb") as f:
                pickle.load(f)
            add(f"{other.name} loads", True, f"{other.stat().st_size:,} bytes")
        except Exception as exc:
            add(f"{other.name} loads", False, f"{type(exc).__name__}: {exc}"[:120])

    add("fraud_model.pkl present", path.is_file(), f"{path.stat().st_size:,} bytes" if path.is_file() else "missing")
    if not path.is_file():
        return checks, None, {}, None
    sha = mst.sha256_file(path)
    add("SHA-256 matches the demo manifest record", sha == manifest["model"]["sha256"], sha)

    t0 = time.perf_counter()
    try:
        model, meta = DocumentFraudDetector.load(path)
    except Exception as exc:
        add("pickle loads (DocumentFraudDetector.load)", False, f"{type(exc).__name__}: {exc}"[:120])
        return checks, None, {}, None
    add("pickle loads (DocumentFraudDetector.load)", True, f"{(time.perf_counter() - t0) * 1000:.0f} ms")
    try:
        schema = get_schema(meta.get("feature_schema_version"))
        add("feature schema is known", True, f"{schema.version}: {schema.description}")
        add("metadata feature names == schema feature names", meta.get("feature_names") == list(schema.names),
            f"{len(schema.names)} features")
        add(f"model expects {len(schema.names)} features", getattr(model, "n_features_in_", None) == len(schema.names),
            f"n_features_in_={getattr(model, 'n_features_in_', None)}")
    except ValueError as exc:
        add("feature schema is known", False, str(exc))
    trees = getattr(model, "estimators_", [])
    add("forest complete (all trees fitted)",
        len(trees) == model.n_estimators == meta.get("n_estimators") and all(hasattr(t, "tree_") for t in trees),
        f"{len(trees)}/{meta.get('n_estimators')} trees")
    classes = [int(c) for c in model.classes_]
    add("classes are [genuine=0, tampered=1]", classes == [0, 1], str(classes))

    files = meta.get("training_files") or {}
    changed = [p for p, digest in files.items() if not (ROOT / "data" / p).is_file()
               or mst.sha256_file(ROOT / "data" / p) != digest]
    add("recorded training files present and unchanged", bool(files) and not changed,
        f"{len(files) - len(changed)}/{len(files)} match" if files else "no training files recorded (legacy model)")

    api = ModelAPI(path)
    try:
        loaded = api.load_model()
    except RuntimeError as exc:
        loaded = False
        api.last_error = str(exc)
    add("ModelAPI.load_model() (the API's loader)", loaded and api.is_loaded,
        f"{(api.load_seconds or 0) * 1000:.0f} ms" if loaded else str(api.last_error))
    return checks, model, meta, api if loaded else None


# ── 2. Demo validation ──────────────────────────────────────────────────────

def bbox_fraction(rec: dict) -> float | None:
    """Fraction of the page covered by the recorded edit bounding box [x, y, w, h], if any."""
    try:
        box = json.loads(rec.get("bbox") or "null")
    except json.JSONDecodeError:
        return None
    if not box:
        return None
    return box[2] * box[3] / (rec["width"] * rec["height"])


def demo_validation(api: ModelAPI, images: list[dict]) -> list[dict]:
    rows = []
    for rec in images:
        path = mst.DEMO_DIR / rec["demo_file"]
        row = {"image": rec["demo_file"], "role": rec["role"], "expected": rec["actual_label"],
               "recorded_score": rec["tamper_risk_score"], "edit": rec.get("transform", "none"),
               "edit_area_fraction": bbox_fraction(rec)}
        try:
            out = api.predict(path.read_bytes(), filename=path.name)
        except Exception as exc:
            rows.append({**row, "error": f"{type(exc).__name__}: {exc}"[:200], "crash": not isinstance(exc, ValueError),
                         "pass": False})
            continue
        score, conf = out["tamper_risk_score"], out["confidence"]
        predicted = mst.label_for(score)
        rows.append({**row, "actual": predicted, "confidence": conf, "risk": score,
                     "margin": score - config.TAMPER_THRESHOLD,
                     "valid_output": 0.0 <= score <= 1.0 and 0.0 <= conf <= 1.0
                     and regions_valid(out["flagged_regions"], score, rec),
                     "flagged_regions": out["flagged_regions"],
                     "interpretation": out["interpretation"], "crash": False, "pass": predicted == rec["actual_label"]})
    return rows


def regions_valid(regions: list, score: float, rec: dict) -> bool:
    """Regions only for a tampering verdict, as non-negative integer boxes inside the page."""
    if score < config.TAMPER_THRESHOLD and regions:
        return False
    return all(isinstance(r[k], int) and r[k] >= 0 for r in regions for k in ("x", "y", "width", "height", "area"))\
        and all(r["x"] + r["width"] <= rec["width"] and r["y"] + r["height"] <= rec["height"] for r in regions)


# ── 4. Held-out accuracy ────────────────────────────────────────────────────

def day6_record() -> dict | None:
    """Held-out metrics and per-image risks recorded in logs/final_validation_report.txt."""
    if not DAY6_REPORT.exists():
        return None
    text = DAY6_REPORT.read_text(encoding="utf-8")
    acc = re.search(r"Accuracy:\s+([\d.]+)%\s+\((\d+)/(\d+)\)", text)
    prec = re.search(r"Precision:\s+([\d.]+)%", text)
    rec = re.search(r"Recall:\s+([\d.]+)%", text)
    f1 = re.search(r"F1:\s+([\d.]+)", text)
    gen = re.search(r"Generated \(UTC\): (\S+)", text)
    if not (acc and prec and rec and f1):
        return None
    risks = {m.group(1): float(m.group(2))
             for m in re.finditer(r"^((?:genuine|tampered)/\S+)\s*\|.*\|\s*([\d.]+)\s*\|\s*(?:YES|NO)$", text, re.M)}
    return {"when": "Day 6", "model": "v1 (day4-v2-22)", "set": "frozen TEST",
            "source": f"logs/final_validation_report.txt ({gen.group(1) if gen else '?'})",
            "n": int(acc.group(3)), "correct": int(acc.group(2)), "accuracy": float(acc.group(1)) / 100,
            "precision": float(prec.group(1)) / 100, "recall": float(rec.group(1)) / 100,
            "f1": float(f1.group(1)), "per_image_risk": risks, "note": "held-out replica of v1"}


def heldout_documents(split: dict, index: dict) -> dict[str, list[dict]]:
    test = [{"path": r["path"], "file": fv.DATASET / r["path"], "label": r["label"], "sha256": r["sha256"],
             "doc_type": index[r["path"]]["doc_type"], "method": index[r["path"]]["tamper_method"] or "-"}
            for r in split["test"]]
    holdout = [{"path": r["path"], "file": r["file"], "label": r["label"], "sha256": mst.sha256_file(r["file"]),
                "doc_type": r["doc_type"], "method": r["tamper_method"] or "-"} for r in load_set("holdout")]
    return {"TEST": test, "HOLDOUT": holdout}


def leakage_checks(meta: dict, sets: dict[str, list[dict]]) -> list[tuple[str, bool, str]]:
    trained = set((meta.get("training_files") or {}).values())
    checks = [("model records its training files", bool(trained), f"{len(trained)} files")]
    for name, docs in sets.items():
        hits = [d["path"] for d in docs if d["sha256"] in trained]
        checks.append((f"no {name} image among the training files", not hits, f"{len(hits)} of {len(docs)}"))
    created = json.loads((GENERATED / SETS["holdout"]["dir"] / "manifest.json").read_text(encoding="utf-8"))
    after = created["created_utc"] > meta.get("trained_at", "")
    checks.append(("HOLDOUT generated after the model was trained", after,
                   f"model {meta.get('trained_at', '?')[:19]}, holdout {created['created_utc'][:19]}"))
    checks.append(("no processed copy of a held-out image in training",
                   "augmentation" in meta and "frozen TEST set (15 images)" in meta.get("excluded_from_training", []),
                   "copies are made from training documents only"))
    return checks


def score_set(api: ModelAPI, docs: list[dict]) -> dict:
    y = np.array([LABELS.index(d["label"]) for d in docs])
    scores = np.array([api.predict(d["file"])["tamper_risk_score"] for d in docs])
    pred = np.array([LABELS.index(mst.label_for(s)) for s in scores])
    m = fv.compute_metrics(y, pred)  # asserts agreement with sklearn
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0.0)
    by_method, by_type = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for d, yi, pi in zip(docs, y, pred):
        if yi == 1:
            by_method[d["method"]][0] += int(pi == 1)
            by_method[d["method"]][1] += 1
        by_type[d["doc_type"]][0] += int(pi == yi)
        by_type[d["doc_type"]][1] += 1
    return {
        "metrics": m,
        "ci95": {"accuracy": wilson_interval(m["correct"], m["n"]),
                 "precision": wilson_interval(m["tp"], m["tp"] + m["fp"]),
                 "recall": wilson_interval(m["tp"], m["tp"] + m["fn"])},
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                  "support": int(s[i])} for i in range(2)},
        "recall_by_method": dict(sorted(by_method.items())),
        "accuracy_by_doc_type": dict(sorted(by_type.items())),
        "per_image": [{"path": d["path"], "doc_type": d["doc_type"], "method": d["method"], "actual": d["label"],
                       "predicted": LABELS[pi], "risk": float(sc)} for d, pi, sc in zip(docs, pred, scores)],
        "y": y, "pred": pred, "scores": scores,
    }


def heldout_evaluation(api: ModelAPI, meta: dict, split: dict, index: dict) -> dict:
    sets = heldout_documents(split, index)
    leakage = leakage_checks(meta, sets)
    results = {name: score_set(api, docs) for name, docs in sets.items()}
    y = np.concatenate([results[n]["y"] for n in sets])
    pred = np.concatenate([results[n]["pred"] for n in sets])
    combined = fv.compute_metrics(y, pred)
    test = sets["TEST"]
    return {
        "leakage": leakage,
        "leakage_ok": all(ok for _, ok, _ in leakage),
        "sets": {n: {k: v for k, v in r.items() if k not in ("y", "pred", "scores")} for n, r in results.items()},
        "combined": {"metrics": combined, "ci95": {"accuracy": wilson_interval(combined["correct"], combined["n"])}},
        "test_table": fv.per_image_table(test, index, results["TEST"]["y"], results["TEST"]["pred"],
                                         results["TEST"]["scores"]),
        "previous": PREVIOUS[:2] + [x for x in [day6_record()] if x] + PREVIOUS[2:],
    }


# ── 5. Inference robustness ─────────────────────────────────────────────────

def inference_robustness(api: ModelAPI, demo_file: str) -> list[dict]:
    demo = (mst.DEMO_DIR / demo_file).read_bytes()
    cases = [(p.name, p.name, p.read_bytes(), "rejected" if "corrupt" in p.stem else "processed")
             for p in sorted(FIXTURES.iterdir()) if p.suffix.lower() in config.ALLOWED_EXTENSIONS]
    cases += [
        ("truncated demo PNG (first 50% of bytes)", "truncated.png", demo[: len(demo) // 2], "rejected"),
        ("empty file", "empty.png", b"", "rejected"),
        ("text file named .png", "notes.png", b"not an image\n", "rejected"),
    ]
    rows = []
    for name, filename, data, expected in cases:
        t0 = time.perf_counter()
        try:
            out = api.predict(data, filename=filename)
            score = out["tamper_risk_score"]
            valid = 0.0 <= score <= 1.0 and 0.0 <= out["confidence"] <= 1.0
            outcome, detail, crash = "processed", f"risk {score:.2f}" + ("" if valid else " (INVALID)"), not valid
        except ValueError as exc:  # includes ImageValidationError -> HTTP 400
            outcome, detail, crash = "rejected", f"{type(exc).__name__}: {exc}"[:90], False
        except Exception as exc:
            outcome, detail, crash = "crash", f"{type(exc).__name__}: {exc}"[:90], True
        rows.append({"case": name, "bytes": len(data), "expected": expected, "outcome": outcome, "detail": detail,
                     "crash": crash, "seconds": time.perf_counter() - t0,
                     "pass": outcome == expected and not crash})
    return rows


# ── 6. Error analysis ───────────────────────────────────────────────────────

def edit_kind(method: str, changed_px: int) -> str:
    """What a missed edit left on the page (from the replayed generator run)."""
    if changed_px == 0:
        return "no pixel changed (edit on blank paper)"
    if method == "text_overlay" and changed_px < 1000:
        return "only the inserted words (white box on white paper)"
    return "visible local edit"


def error_analysis(rv: dict) -> dict:
    """Changed-pixel footprint of every missed edit in TEST and HOLDOUT (replayed generator runs)."""
    footprints: dict[str, dict] = {}
    try:
        for run in ("synthetic_day2", "synthetic_day4"):
            footprints.update(edit_footprints(run))
        holdout = {f"HOLDOUT:{k}": v for k, v in edit_footprints("holdout").items()}
    except ValueError as exc:  # replay does not reproduce the files (e.g. different fonts)
        return {"available": False, "reason": str(exc)}
    missed = []
    for name, res in rv["sets"].items():
        for row in res["per_image"]:
            if row["actual"] == "tampered" and row["predicted"] == "genuine":
                key = row["path"] if name == "TEST" else f"HOLDOUT:{row['path']}"
                fp = footprints.get(key) if name == "TEST" else holdout.get(key)
                missed.append({"set": name, "path": row["path"], "method": row["method"], "risk": row["risk"],
                               "changed_px": fp["changed_px"] if fp else None,
                               "kind": edit_kind(row["method"], fp["changed_px"]) if fp else "unknown"})
    all_edits = list(holdout.values()) + [v for k, v in footprints.items()
                                          if any(r["path"] == k for r in rv["sets"]["TEST"]["per_image"])]
    return {"available": True, "missed": missed,
            "heldout_edits_without_pixel_change": sum(e["changed_px"] == 0 for e in all_edits),
            "heldout_edits": len(all_edits)}


# ── Report ──────────────────────────────────────────────────────────────────

def _metric_line(name: str, r: dict) -> str:
    m, ci = r["metrics"], r["ci95"]["accuracy"]
    return (f"{name:<9} {m['n']:>4} {pct(m['accuracy']):>9} ({pct(ci[0])}-{pct(ci[1])}) "
            f"{pct(m['precision']) if 'precision' in m else '':>9} {pct(m['recall']) if 'recall' in m else '':>8} "
            f"{m.get('f1', 0):>6.3f} {pct(m['fpr']) if 'fpr' in m else '':>8} {pct(m['fnr']) if 'fnr' in m else '':>8}")


def build_report(r: dict) -> str:
    env, model = r["environment"], r["model"]
    rv, demo, rep, rob, ea = r["heldout"], r["demo"], r["repeatability"], r["robustness"], r["errors"]
    n_correct = sum(x["pass"] for x in demo)
    test, hold = rv["sets"]["TEST"]["metrics"], rv["sets"]["HOLDOUT"]["metrics"]

    L = [
        "--------------------------------",
        "DOCFRAUD MODEL VALIDATION REPORT",
        "--------------------------------",
        f"Generated (UTC): {r['generated']}",
        f"Script: scripts/model_validation.py --runs {r['runs']}",
        f"Environment: Python {env['python']}, {env['platform']}, {env['cpus']} CPUs; "
        + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
        "",
        "Model:",
        f"  {model['path']} (loaded by the API), SHA-256 {model['sha256']}",
        f"  version {model['version']}, feature schema {model['schema']} ({model['n_features']} features)",
        f"  RandomForest {model['rf_params']}, trained {model['trained_at']}",
        f"  Training data: {model['n_training_samples']} documents (frozen TRAIN 60 + generated DEV 150) "
        f"+ processed copies = {model['n_training_rows']} rows",
        f"  Decision rule: tampered if tamper_risk_score >= {config.TAMPER_THRESHOLD} (TAMPER_THRESHOLD, not tuned)",
        "  tamper_risk_score is an uncalibrated RandomForest vote fraction, not a fraud probability.",
        "",
        "--------------------------------",
        "FINAL MODEL CHECKLIST",
        "--------------------------------",
    ]
    for item in r["checklist"]:
        L.append(f"[{_mark(item['pass'])}] {item['item']:<38} {item['detail']}")
    n_pass = sum(i["pass"] for i in r["checklist"])
    L += [f"Result: {n_pass}/{len(r['checklist'])} items pass", ""]
    both = min(test["accuracy"], hold["accuracy"])
    L += [
        "Release readiness (model):",
        f"  {'Deterministic and stable:':<44}{'YES' if r['stable'] else 'NO'}",
        f"  {'Model file intact, training data recorded:':<44}{'YES' if r['integrity_ok'] else 'NO'}",
        f"  {'Held-out accuracy above 75%:':<44}{'YES' if both > REQUIRED_ACCURACY else 'NO'} "
        f"(TEST {pct(test['accuracy'])}, HOLDOUT {pct(hold['accuracy'])})",
        f"  {'Meets the 78% sprint target:':<44}{'YES' if both >= SPRINT_TARGET else 'NO'}",
        f"  {'All demo images predicted correctly:':<44}{'YES' if n_correct == len(demo) else 'NO'} "
        f"({n_correct}/{len(demo)})",
        f"  {'Production ready (real documents):':<44}NO - synthetic data only; see limitations",
        "",
        "--------------------------------",
        "1. MODEL INTEGRITY",
        "--------------------------------",
    ]
    L += [f"[{'x' if ok else ' '}] {n}{' - ' + d if d else ''}" for n, ok, d in r["integrity"]]

    L += ["", "--------------------------------", "2. DEMO IMAGE VALIDATION (ModelAPI.predict(bytes) = the /verify call)",
          "--------------------------------",
          f"{'Image Name':<26} {'Expected Result':<16} {'Actual Result':<14} {'Confidence':>10} {'Risk Score':>10}  "
          "Pass/Fail",
          "-" * 92]
    for x in demo:
        if "error" in x:
            L.append(f"{x['image']:<26} {x['expected']:<16} {'ERROR':<14} {'-':>10} {'-':>10}  FAIL  {x['error']}")
            continue
        L.append(f"{x['image']:<26} {x['expected']:<16} {x['actual']:<14} {x['confidence']:>10.2f} "
                 f"{x['risk']:>10.2f}  {_mark(x['pass'])}")
    L.append(f"Correct: {n_correct}/{len(demo)}. Five demonstration images cannot measure accuracy; see section 4.")
    wrong = [x for x in demo if not x["pass"] and "error" not in x]
    if wrong:
        L.append("Failure analysis:")
    for x in wrong:
        kind = "false positive (genuine flagged)" if x["expected"] == "genuine" else "missed tampering"
        area = (f"; the edit ({x['edit']}) covers {100 * x['edit_area_fraction']:.2f}% of the page"
                if x["edit_area_fraction"] else "")
        L.append(f"  - {x['image']}: {kind}; risk {x['risk']:.2f} is {abs(x['margin']):.2f} "
                 f"{'above' if x['margin'] >= 0 else 'below'} the threshold{area}.")

    L += ["", "--------------------------------", f"3. STABILITY ({r['runs']} runs per demo image)",
          "--------------------------------",
          f"{'Image':<26} {'Label':<9} {'Score mean':>10} {'min':>6} {'max':>6} {'max dev':>8} {'Conf dev':>8}  Result",
          "-" * 88]
    for e in rep:
        if "score" not in e:
            L.append(f"{e['demo_file']:<26} no successful run  FAIL")
            continue
        L.append(f"{e['demo_file']:<26} {'/'.join(e['labels']):<9} {e['score']['mean']:>10.4f} {e['score']['min']:>6.2f} "
                 f"{e['score']['max']:>6.2f} {e['score']['max_abs_dev']:>8.4f} {e['confidence']['max_abs_dev']:>8.4f}  "
                 f"{_mark(e['pass'])}")
    L.append(f"Tolerance +/-{mst.TOLERANCE:.2f}. Full analysis: logs/model_stability_report.txt")

    L += ["", "--------------------------------", "4. HELD-OUT ACCURACY (deployed model file, through ModelAPI)",
          "--------------------------------",
          "TEST    = the 15 frozen test images (data/splits/day5_frozen_split.json).",
          f"HOLDOUT = {rv['sets']['HOLDOUT']['metrics']['n']} documents in data/generated/holdout_v1/ "
          f"(seed {SETS['holdout']['seed']}), generated after the model was saved.",
          "Neither set was used for training, feature design, threshold choice or model selection;",
          "each was scored once with the frozen model.",
          "Leakage checks:"]
    L += [f"  [{'x' if ok else ' '}] {n}{' - ' + d if d else ''}" for n, ok, d in rv["leakage"]]
    L += ["Frozen test-set integrity:"]
    L += [f"  [{'x' if ok else ' '}] {n}{' - ' + d if d else ''}" for n, ok, d in r["test_integrity"]]
    L += ["",
          f"{'Set':<9} {'n':>4} {'Accuracy (95% CI)':>27} {'Precision':>9} {'Recall':>8} {'F1':>6} {'FPR':>8} {'FNR':>8}",
          "-" * 88,
          _metric_line("TEST", rv["sets"]["TEST"]),
          _metric_line("HOLDOUT", rv["sets"]["HOLDOUT"]),
          _metric_line("COMBINED", rv["combined"]),
          "Positive class = tampered. FPR = genuine flagged; FNR = tampered missed.", ""]
    for name in ("TEST", "HOLDOUT"):
        s = rv["sets"][name]
        L += [f"{name} confusion matrix:", fv.ascii_confusion(s["metrics"]),
              "  Tampered caught by method: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in s["recall_by_method"].items()),
              "  Accuracy by document type: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in s["accuracy_by_doc_type"].items()),
              ""]
    L += ["TEST per-image results:", rv["test_table"], ""]
    hold_err = [x for x in rv["sets"]["HOLDOUT"]["per_image"] if x["actual"] != x["predicted"]]
    L.append(f"HOLDOUT errors ({len(hold_err)}):")
    L += [f"  {x['path']:<34} {x['doc_type']:<15} {x['method']:<13} actual {x['actual']:<8} risk {x['risk']:.2f}"
          for x in hold_err]

    L += ["", "Comparison with previous results:",
          f"{'When':<6} {'Model':<26} {'Set':<13} {'n':>4} {'Accuracy':>9} {'Precision':>9} {'Recall':>8} {'F1':>6}  Source",
          "-" * 118]
    for p in rv["previous"]:
        L.append(f"{p['when']:<6} {p['model']:<26} {p['set']:<13} {p['n']:>4} {pct(p['accuracy']):>9} "
                 f"{pct(p['precision']):>9} {pct(p['recall']):>8} {p['f1']:>6.3f}  {p['source']}; {p['note']}")
    for name in ("TEST", "HOLDOUT"):
        m = rv["sets"][name]["metrics"]
        L.append(f"{'Day 8':<6} {'v2 (' + model['schema'] + ')':<26} {name:<13} {m['n']:>4} {pct(m['accuracy']):>9} "
                 f"{pct(m['precision']):>9} {pct(m['recall']):>8} {m['f1']:>6.3f}  this run; deployed file")
    t_prev = PREVIOUS[1]["accuracy"]
    L += [f"On the same 15 frozen TEST images, accuracy moves from {pct(t_prev)} (v1) to {pct(test['accuracy'])} (v2), "
          f"{100 * (test['accuracy'] - t_prev):+.2f} points.",
          "The v1 figures come from a held-out replica (the v1 file had been trained on TEST); the v2 figures",
          "are for the deployed file itself, which never saw TEST or HOLDOUT.", ""]

    L += ["--------------------------------", "5. INFERENCE ROBUSTNESS (ModelAPI.predict, bytes)",
          "--------------------------------",
          f"{'Case':<42} {'Bytes':>9} {'Expected':<9} {'Outcome':<9} {'Time':>6}  Detail", "-" * 118]
    for x in rob:
        L.append(f"{x['case']:<42} {x['bytes']:>9,} {x['expected']:<9} {x['outcome']:<9} {x['seconds']:>5.2f}s  "
                 f"{x['detail']}")
    L.append("'rejected' = controlled ValueError, returned by /verify as HTTP 400.")

    L += ["", "--------------------------------", "6. ERROR ANALYSIS (missed edits, replayed generator runs)",
          "--------------------------------"]
    if ea.get("available"):
        L.append(f"{ea['heldout_edits_without_pixel_change']} of the {ea['heldout_edits']} held-out edits changed no "
                 "pixel at all (the generator blurred blank paper); no detector can find those.")
        L.append(f"{'Set':<8} {'Document':<34} {'Method':<13} {'Risk':>5} {'Changed px':>10}  What the edit left")
        for x in ea["missed"]:
            L.append(f"{x['set']:<8} {x['path']:<34} {x['method']:<13} {x['risk']:>5.2f} "
                     f"{x['changed_px'] if x['changed_px'] is not None else '?':>10}  {x['kind']}")
    else:
        L.append(f"Not available: {ea.get('reason')}")

    L += ["", "--------------------------------", "EDGE CASES AND FAILURE SCENARIOS", "--------------------------------"]
    L += r["failure_scenarios"]
    L += ["", "--------------------------------", "SUGGESTED IMPROVEMENTS", "--------------------------------"]
    L += r["recommendations"]
    L.append("")
    return "\n".join(L)


def failure_scenarios(r: dict) -> list[str]:
    rv, ea = r["heldout"], r["errors"]
    out = []
    wrong = [x for x in r["demo"] if not x["pass"]]
    out.append(f"- Demo: {len(wrong)}/{len(r['demo'])} images wrong"
               + (": " + ", ".join(f"{x['image']} ({x['expected']} -> {x.get('actual', 'error')})" for x in wrong)
                  if wrong else "") + ".")
    for name in ("TEST", "HOLDOUT"):
        m = rv["sets"][name]["metrics"]
        out.append(f"- {name}: {m['fp']}/{m['fp'] + m['tn']} genuine flagged (FPR {pct(m['fpr'])}), "
                   f"{m['fn']}/{m['fn'] + m['tp']} tampered missed (FNR {pct(m['fnr'])}).")
    if ea.get("available") and ea["missed"]:
        kinds: dict[str, int] = defaultdict(int)
        for x in ea["missed"]:
            kinds[x["kind"]] += 1
        out.append("- Missed edits by what they left on the page: "
                   + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + ".")
    near = [x for x in r["demo"] if "margin" in x and abs(x["margin"]) <= 0.10]
    if near:
        out.append("- Near-threshold demo scores (within 0.10 of 0.5): "
                   + ", ".join(f"{x['image']} {x['risk']:.2f}" for x in near) + ".")
    rejected = [x["case"] for x in r["robustness"] if x["outcome"] == "rejected"]
    out.append(f"- Invalid uploads are rejected, not crashed: {', '.join(rejected)}." if rejected else
               "- No invalid upload was rejected.")
    processed = [x for x in r["robustness"] if x["outcome"] == "processed"]
    if processed:
        out.append("- Off-distribution inputs (non-document fixtures, small, grayscale, large) still get a score: "
                   + ", ".join(f"{x['case']} {x['detail'].split()[-1]}" for x in processed)
                   + ". The model has no 'not a document' output; such scores are not meaningful.")
    out.append("- Re-saved, resized or re-exposed copies are harder: cross-validation on processed copies of the "
               "training documents gives lower accuracy than on pristine files (logs/training_log_v2.txt; "
               "logs/model_stability_report.txt section 6).")
    return out


def recommendations(r: dict) -> list[str]:
    rv = r["heldout"]
    hold = rv["sets"]["HOLDOUT"]["metrics"]
    return [
        f"- The accuracy requirement is met on synthetic data (HOLDOUT {pct(hold['accuracy'])}). The next limit is "
        "real documents: collect a labelled set of real (consented or public-sample) payslips, statements and IDs "
        "and measure before any production use.",
        "- Robustness to re-saving: processed copies are classified less accurately than pristine files, and JPEG is "
        "the weakest case. Score tiles relative to the page's own compression level (estimated JPEG quality) "
        "instead of absolute levels, and widen the processing used in training.",
        "- Blank-paper edits: some generated 'tampered' documents change no pixel. Fix the generator to place edits "
        "on content (or record the edit box) so labels match what is on the page; report detection only on edits "
        "that changed pixels.",
        "- Localisation: flagged_regions now boxes the anomalous region for flagged documents "
        "(logs/localization_report.txt); it misses most text overlays on bank statements, where the inserted "
        "words share the page's ink level.",
        "- Keep the evaluation clean: HOLDOUT and TEST have now been used once for v2. A future model needs a new "
        "holdout set (new seed) so it is not selected on these.",
        "- Calibrate the score and add a 'needs review' band around the threshold, chosen on validation data.",
        "- Pin scikit-learn, numpy, OpenCV, Pillow and SciPy, record the model SHA-256 per release, and re-run "
        "scripts/model_stability_test.py and scripts/model_validation.py after every dependency or model change.",
    ]


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Day 8 model validation: demo images, held-out accuracy, checklist")
    ap.add_argument("--runs", type=int, default=mst.DEFAULT_RUNS, help="stability runs per demo image (default 10)")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="machine-readable results")
    args = ap.parse_args()
    if args.runs < 2:
        ap.error("--runs must be at least 2")

    manifest, images = mst.load_demo()
    split = json.loads(fv.SPLIT_PATH.read_text(encoding="utf-8"))
    index = {r["path"]: r for r in csv.DictReader(open(fv.DATASET / "index.csv", newline="", encoding="utf-8"))}
    try:
        test_integrity = fv.check_integrity(split)
    except SystemExit as exc:  # a frozen test image is missing or modified: do not score anything
        print(exc)
        return 2

    print("1/6 Model integrity...")
    integrity, deployed, meta, api = model_integrity(manifest)
    load_ok = deployed is not None and api is not None
    integrity_ok = load_ok and all(ok for _, ok, _ in integrity)
    has_training_record = bool(meta.get("training_files"))

    demo, rep, rob, rv, ea = [], [], [], None, {}
    if load_ok:
        print("2/6 Demo validation...")
        demo = demo_validation(api, images)
        print(f"3/6 Stability ({args.runs} runs x {len(images)} images)...")
        rep = mst.repeatability(api, images, args.runs)
        if has_training_record:
            print("4/6 Held-out accuracy (TEST + HOLDOUT)...")
            rv = heldout_evaluation(api, meta, split, index)
            print("6/6 Error analysis (replaying generator runs)...")
            ea = error_analysis(rv)
        else:
            print("4/6 Legacy model without recorded training files: use scripts/final_validation.py")
        print("5/6 Inference robustness...")
        rob = inference_robustness(api, images[0]["demo_file"])
    else:
        print("Model did not load; skipping inference steps.")

    n_calls = len(demo) + sum(len(e["runs"]) for e in rep) + len(rob)
    if rv:
        n_calls += sum(s["metrics"]["n"] for s in rv["sets"].values())
    crashes = sum(x["crash"] for x in demo) + sum(e["crashes"] for e in rep) + sum(x["crash"] for x in rob)
    stable = bool(rep) and all(e["pass"] for e in rep)
    worst = max((e.get("score", {}).get("max_abs_dev", 0.0) for e in rep), default=0.0)
    n_correct = sum(x["pass"] for x in demo)
    wrong = [x["image"] for x in demo if not x["pass"]]
    test_acc = rv["sets"]["TEST"]["metrics"]["accuracy"] if rv else 0.0
    hold_acc = rv["sets"]["HOLDOUT"]["metrics"]["accuracy"] if rv else 0.0
    checklist = [
        {"item": "Model loads successfully", "pass": load_ok,
         "detail": f"DocumentFraudDetector.load and ModelAPI.load_model; schema {meta.get('feature_schema_version')}"
         if load_ok else "load failed"},
        {"item": "All demo images predict correctly", "pass": bool(demo) and n_correct == len(demo),
         "detail": f"{n_correct}/{len(demo)} correct" + (f"; wrong: {', '.join(wrong)}" if wrong else "")},
        {"item": "Scores stable within +/-2%", "pass": stable,
         "detail": f"{sum(len(e['runs']) for e in rep)} runs ({args.runs} per image); largest deviation {worst:.4f}"},
        {"item": "No corrupted model files", "pass": integrity_ok,
         "detail": "SHA-256 matches record, schema and structure valid, training files unchanged" if integrity_ok
         else "see section 1"},
        {"item": "No inference crashes", "pass": load_ok and crashes == 0,
         "detail": f"{crashes} crashes in {n_calls} ModelAPI calls "
                   f"({sum(x['outcome'] == 'rejected' for x in rob)} controlled rejections of invalid files)"},
        {"item": "Metrics generated", "pass": rv is not None and all(ok for _, ok, _ in test_integrity)
         and rv["leakage_ok"],
         "detail": "accuracy, precision, recall, F1, confusion matrices on TEST and HOLDOUT; leakage checks pass"
         if rv and rv["leakage_ok"] else "not generated or leakage check failed"},
        {"item": "Held-out accuracy above 75%", "pass": rv is not None and min(test_acc, hold_acc) > REQUIRED_ACCURACY,
         "detail": f"TEST {pct(test_acc)}, HOLDOUT {pct(hold_acc)}" if rv else "not measured"},
    ]

    schema = meta.get("feature_schema_version")
    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": args.runs,
        "threshold": config.TAMPER_THRESHOLD,
        "environment": mst.environment(),
        "model": {
            "path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(),
            "sha256": mst.sha256_file(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH.is_file() else None,
            "version": meta.get("model_version"), "schema": schema,
            "n_features": getattr(deployed, "n_features_in_", None),
            "rf_params": meta.get("rf_params"), "trained_at": meta.get("trained_at"),
            "n_training_samples": meta.get("n_training_samples"), "n_training_rows": meta.get("n_training_rows"),
        },
        "integrity": integrity, "integrity_ok": integrity_ok, "test_integrity": test_integrity,
        "demo": demo, "repeatability": rep, "stable": stable, "heldout": rv, "errors": ea, "robustness": rob,
        "checklist": checklist,
    }
    if rv is None:  # nothing to analyse; still write the checklist and integrity results
        result["failure_scenarios"], result["recommendations"] = [], []
        text = "\n".join([f"[{_mark(i['pass'])}] {i['item']}: {i['detail']}" for i in checklist]
                         + [f"[{'x' if ok else ' '}] {n} - {d}" for n, ok, d in integrity])
    else:
        result["failure_scenarios"] = failure_scenarios(result)
        result["recommendations"] = recommendations(result)
        text = build_report(result)

    for path, body in ((args.report, text), (args.json, json.dumps(result, indent=1, default=str) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    print(text)
    print(f"Report written: {args.report}\nResults written: {args.json}")
    return 0 if all(i["pass"] for i in checklist) else 1


if __name__ == "__main__":
    raise SystemExit(main())
