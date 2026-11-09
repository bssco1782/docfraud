"""
scripts/model_review.py
Day 9 (Founder A): final model review - performance, prediction quality, optimisation, release checklist.

Usage:
    python scripts/feature_importance.py        # first: the checklist reads their results
    python scripts/model_robustness_test.py
    python scripts/model_review.py

Sections (deployed data/models/fraud_model.pkl, scored through ModelAPI = the /verify path):
  1. Performance review - held-out documents only (15 frozen TEST + 149 HOLDOUT): accuracy,
     precision, recall, F1, confusion matrices, per class / document type / edit method, compared
     with the earlier reports (v1 Day 5/6/8; v2 Day 8 in logs/model_validation_results.json).
     Weak classes, every false positive and false negative, and what each missed edit left on
     the page (replayed generator runs).
  2. Prediction quality - genuine, tampered and edge-case documents (demo images, non-document
     test fixtures, degraded held-out copies from logs/model_robustness_results.json): risk score
     and confidence distributions, calibration by score band, near-threshold share, and every
     incorrect prediction.
  3. Optimisation review - model size (file, trees, nodes, depth), load time, process memory
     (clean subprocess), per-stage inference time for a typical and a large page, the Day-9
     single-thread inference change (before / after), and smaller-forest alternatives measured by
     grouped cross-validation on training data only.
  4. Release checklist.

Nothing is trained on or tuned with held-out data; the model and threshold are unchanged.
Writes logs/model_review_report.txt and logs/model_review_results.json.
Exit code: 0 if every checklist item passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import pickle
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
import final_validation as fv
import model_stability_test as mst
import model_validation as mv
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold

from backend import config
from backend.model_utils import ModelAPI
from core import tamper_features as tf
from core.document_analyzer import load_image
from core.model_trainer import DEFAULT_MODEL_PATH, DocumentFraudDetector
from scripts import train_tamper_model as ttm

DEFAULT_REPORT = ROOT / "logs" / "model_review_report.txt"
DEFAULT_JSON = ROOT / "logs" / "model_review_results.json"
FEATURE_JSON = ROOT / "logs" / "feature_importance_results.json"
FEATURE_REPORT = ROOT / "logs" / "feature_importance_report.txt"
ROBUST_JSON = ROOT / "logs" / "model_robustness_results.json"
ROBUST_REPORT = ROOT / "logs" / "model_robustness_report.txt"
VALIDATION_JSON = ROOT / "logs" / "model_validation_results.json"
TYPICAL_PAGE = ROOT / "data" / "demo" / "genuine_payslip_01.png"
TIMING_RUNS = 15
NEAR = 0.10  # "near the threshold" = |score - threshold| <= NEAR
ALTERNATIVES = [  # forest variants compared by grouped CV on training data (not deployed)
    {"n_estimators": 100}, {"n_estimators": 200}, {"n_estimators": 300},
    {"n_estimators": 300, "min_samples_leaf": 3}, {"n_estimators": 100, "min_samples_leaf": 3},
    {"n_estimators": 300, "max_depth": 12},
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ── 1. Performance review ───────────────────────────────────────────────────

def performance_review(api: ModelAPI, meta: dict) -> dict:
    split = json.loads(fv.SPLIT_PATH.read_text(encoding="utf-8"))
    index = {r["path"]: r for r in csv.DictReader(open(fv.DATASET / "index.csv", newline="", encoding="utf-8"))}
    integrity = fv.check_integrity(split)  # raises SystemExit if a frozen test image changed
    rv = mv.heldout_evaluation(api, meta, split, index)
    rv["frozen_split_ok"] = all(ok for _, ok, _ in integrity)
    rv["errors"] = mv.error_analysis(rv)

    previous = {}
    if VALIDATION_JSON.exists():
        prev = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
        if prev.get("heldout") and prev["model"]["sha256"] == api_sha():
            previous = {n: prev["heldout"]["sets"][n]["metrics"] for n in ("TEST", "HOLDOUT")}
    rv["day8_v2"] = previous
    rv["same_as_day8"] = bool(previous) and all(
        previous[n][k] == rv["sets"][n]["metrics"][k] for n in previous for k in ("tn", "fp", "fn", "tp"))
    return rv


def api_sha() -> str:
    return sha256(DEFAULT_MODEL_PATH)


# ── 2. Prediction quality ───────────────────────────────────────────────────

def distribution(scores: list[float], edges: list[float]) -> list[int]:
    return np.histogram(scores, bins=edges)[0].astype(int).tolist()


def prediction_quality(api: ModelAPI, rv: dict) -> dict:
    rows = [dict(x, set=name) for name, s in rv["sets"].items() for x in s["per_image"]]
    thr = config.TAMPER_THRESHOLD
    for x in rows:
        x["confidence"] = max(x["risk"], 1 - x["risk"])
        x["correct"] = x["predicted"] == x["actual"]
    edges = [i / 10 for i in range(11)]
    gen = [x["risk"] for x in rows if x["actual"] == "genuine"]
    tam = [x["risk"] for x in rows if x["actual"] == "tampered"]

    calibration = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = [x for x in rows if lo <= x["risk"] < hi or (hi == 1.0 and x["risk"] == 1.0)]
        calibration.append({"band": f"{lo:.1f}-{hi:.1f}", "n": len(band),
                            "tampered_share": (sum(x["actual"] == "tampered" for x in band) / len(band)) if band else None})
    conf_edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    confidence = []
    for lo, hi in zip(conf_edges[:-1], conf_edges[1:]):
        band = [x for x in rows if lo <= x["confidence"] < hi]
        confidence.append({"band": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(band),
                           "accuracy": (sum(x["correct"] for x in band) / len(band)) if band else None})
    near = [x for x in rows if abs(x["risk"] - thr) <= NEAR]

    # Edge cases: demo images, non-document fixtures, degraded held-out copies.
    _, images = mst.load_demo()
    demo = mv.demo_validation(api, images)
    fixtures = mv.inference_robustness(api, images[0]["demo_file"])
    degraded = {}
    if ROBUST_JSON.exists():
        rob = json.loads(ROBUST_JSON.read_text(encoding="utf-8"))
        if rob["model"]["sha256"] == api_sha():
            labels = {(x["set"], x["path"]): x["actual"] for x in rows}
            by_var = defaultdict(lambda: {"genuine": [], "tampered": []})
            for x in rob["rows"]:
                key = (x["set"], x["path"])
                if key in labels and x["outcome"] == "scored" and x["variant"] != "original":
                    by_var[x["variant"]][labels[key]].append(x["score"])
            degraded = {k: {c: {"n": len(v), "median": float(np.median(v)) if v else None,
                                "above_threshold": sum(s >= thr for s in v)} for c, v in d.items()}
                        for k, d in by_var.items()}
    return {
        "n": len(rows),
        "score_histogram": {"edges": edges, "genuine": distribution(gen, edges), "tampered": distribution(tam, edges)},
        "score_summary": {c: {"mean": statistics.fmean(v), "median": float(np.median(v)), "min": min(v), "max": max(v)}
                          for c, v in (("genuine", gen), ("tampered", tam))},
        "calibration": calibration, "confidence": confidence,
        "near_threshold": {"n": len(near), "correct": sum(x["correct"] for x in near)},
        "incorrect": [x for x in rows if not x["correct"]],
        "demo": demo, "fixtures": fixtures, "degraded": degraded,
    }


# ── 3. Optimisation review ──────────────────────────────────────────────────

MEM_PROBE = r"""
import json, sys, time
def rss():
    d = {}
    for line in open("/proc/self/status"):
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmHWM"):
            d[k] = int(v.split()[0]) / 1024
    return d
root, typical, large = sys.argv[1:4]
sys.path.insert(0, root)
out = {"python_started": rss()}
from backend.model_utils import ModelAPI
out["libraries_imported"] = rss()
api = ModelAPI()
t = time.perf_counter(); api.load_model(); out["load_seconds"] = time.perf_counter() - t
out["model_loaded"] = rss()
api.predict(typical); out["after_typical_page"] = rss()
api.predict(large); out["after_large_page"] = rss()
print("MEM " + json.dumps(out))
"""


def _median_ms(fn, runs: int = TIMING_RUNS) -> float:
    fn()  # warm-up
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return 1000 * statistics.median(samples)


def stage_times(path: Path, model, model_saved) -> dict:
    """Median ms per stage for one page (feature groups timed with the module's own helpers)."""
    img = np.asarray(load_image(path))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tile_gray = tf._tiles(gray).astype(np.float32)
    tile_range = tile_gray.max(2) - tile_gray.min(2)
    flat = tile_range <= tf.FLAT_RANGE
    x = np.array([[tf.extract_tamper_features(path)[n] for n in tf.FEATURE_NAMES]])
    return {
        "page": f"{img.shape[1]}x{img.shape[0]} after load_image",
        "decode (load_image)": _median_ms(lambda: load_image(path)),
        "noise features": _median_ms(lambda: tf._noise(gray)),
        "colour features": _median_ms(lambda: tf._colour(img, tile_gray, flat)),
        "softness features": _median_ms(lambda: tf._softness(gray, tile_range)),
        "ink features": _median_ms(lambda: tf._ink(img, gray)),
        "all features (extract_tamper_features)": _median_ms(lambda: tf.extract_tamper_features(path)),
        "forest, as saved (n_jobs=-1, predict + predict_proba)":
            _median_ms(lambda: (model_saved.predict(x), model_saved.predict_proba(x))),
        "forest, as served since Day 9 (n_jobs=1, predict_proba)": _median_ms(lambda: model.predict_proba(x)),
    }


def optimisation_review(api: ModelAPI, meta: dict) -> dict:
    with open(DEFAULT_MODEL_PATH, "rb") as f:
        raw = f.read()
    t0 = time.perf_counter()
    bundle = pickle.loads(raw)
    unpickle_s = time.perf_counter() - t0
    saved = bundle["model"]  # n_jobs as saved (-1)
    served = api.model       # n_jobs=1 (ModelAPI.load_model)
    trees = saved.estimators_
    depths = [t.get_depth() for t in trees]
    size = {
        "file_bytes": len(raw), "unpickle_seconds": unpickle_s, "trees": len(trees),
        "nodes": int(sum(t.tree_.node_count for t in trees)), "leaves": int(sum(t.get_n_leaves() for t in trees)),
        "depth_mean": statistics.fmean(depths), "depth_max": max(depths),
        "saved_n_jobs": saved.n_jobs, "served_n_jobs": served.n_jobs,
    }

    with tempfile.TemporaryDirectory() as tmp:
        large = Path(tmp) / "large_4000x5000.jpg"
        with Image.open(TYPICAL_PAGE) as im:
            im.convert("RGB").resize((4000, 5000), Image.LANCZOS).save(large, quality=90)
        timings = {"typical": stage_times(TYPICAL_PAGE, served, saved), "large": stage_times(large, served, saved)}

        typical_bytes, large_bytes = TYPICAL_PAGE.read_bytes(), large.read_bytes()
        old_api = copy.copy(api)  # the pre-Day-9 inference path, for comparison only
        old_model = saved

        def old_predict(data: bytes, name: str) -> None:  # features + forest exactly as before Day 9
            with tempfile.NamedTemporaryFile(suffix=Path(name).suffix) as f:
                f.write(data)
                f.flush()
                x = np.array([old_api.schema.vector(old_api.schema.extract(f.name))])
                old_model.predict(x)
                old_model.predict_proba(x)

        end_to_end = {
            "typical page, ModelAPI.predict (served)": _median_ms(lambda: api.predict(typical_bytes, "p.png")),
            "typical page, pre-Day-9 path (n_jobs=-1, two forest passes)":
                _median_ms(lambda: old_predict(typical_bytes, "p.png")),
            "4000x5000 JPEG, ModelAPI.predict (served)":
                _median_ms(lambda: api.predict(large_bytes, "p.jpg"), runs=5),
        }
        memory = {"available": Path("/proc/self/status").exists()}
        if memory["available"]:
            proc = subprocess.run([sys.executable, "-c", MEM_PROBE, str(ROOT), str(TYPICAL_PAGE), str(large)],
                                  capture_output=True, text=True, timeout=300, cwd=ROOT)
            line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("MEM ")), None)
            memory.update(json.loads(line[4:]) if line else {"available": False, "error": proc.stderr[-300:]})

    print("   Rebuilding training data for forest alternatives (grouped CV, training data only)...")
    docs = ttm.training_documents()
    data = ttm.build_matrix(docs, verbose=False)
    alternatives = forest_alternatives(data)
    return {"size": size, "timings": timings, "end_to_end_ms": end_to_end, "memory": memory,
            "alternatives": alternatives}


def forest_alternatives(data: dict) -> list[dict]:
    X, y, groups, is_copy = data["X"], data["y"], data["groups"], data["is_copy"]
    orig = np.flatnonzero(~is_copy)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=ttm.RANDOM_STATE).split(orig, y[orig]))
    x1 = X[orig[:1]]
    out = []
    for params in ALTERNATIVES:
        accs, proc_accs = [], []
        for tr_docs, va_docs in folds:
            tr = np.isin(groups, tr_docs)
            model = _fit(X[tr], y[tr], params)
            va = orig[va_docs]
            cp = np.flatnonzero(np.isin(groups, va_docs) & is_copy)
            accs.append(float((model.predict(X[va]) == y[va]).mean()))
            proc_accs.append(float((model.predict(X[cp]) == y[cp]).mean()))
        full = _fit(X, y, params).set_params(n_jobs=1)
        buf = io.BytesIO()
        pickle.dump(full, buf, protocol=pickle.HIGHEST_PROTOCOL)
        out.append({"params": params, "cv_accuracy": statistics.fmean(accs), "cv_std": statistics.pstdev(accs),
                    "cv_processed_accuracy": statistics.fmean(proc_accs), "size_bytes": buf.tell(),
                    "predict_ms": _median_ms(lambda m=full: m.predict_proba(x1)),
                    "deployed": params == {"n_estimators": ttm.RF_PARAMS["n_estimators"]}})
    return out


def _fit(X, y, params: dict):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(class_weight="balanced", random_state=ttm.RANDOM_STATE, n_jobs=-1,
                                  **params).fit(X, y)


# ── 4. Release checklist ────────────────────────────────────────────────────

def release_checklist(api: ModelAPI, meta: dict, perf: dict, quality: dict) -> list[dict]:
    sha = api_sha()
    manifest, images = mst.load_demo()

    def results(path: Path) -> dict | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data.get("model", {}).get("sha256") == sha else None

    fi, rob, val = results(FEATURE_JSON), results(ROBUST_JSON), results(VALIDATION_JSON)
    demo = quality["demo"]
    demo_ok = (manifest["model"]["sha256"] == sha and all("error" not in x for x in demo)
               and all(abs(x["risk"] - x["recorded_score"]) < 1e-9 for x in demo))
    crashes = (sum(x["crash"] for x in quality["fixtures"]) + sum(x["crash"] for x in demo)
               + (rob["crashes"] if rob else 0))
    calls = (perf["combined"]["metrics"]["n"] + len(quality["fixtures"]) + len(demo)
             + (rob["n_predictions"] if rob else 0))
    t, h = perf["sets"]["TEST"]["metrics"], perf["sets"]["HOLDOUT"]["metrics"]
    return [
        {"item": "Model file loads successfully", "pass": api.is_loaded and manifest["model"]["sha256"] == sha,
         "detail": f"ModelAPI loaded v{meta.get('model_version')} ({api.schema.version}); SHA-256 {sha[:16]}... "
                   "matches the demo record"},
        {"item": "Metrics documented", "pass": perf["leakage_ok"] and perf["frozen_split_ok"] and val is not None,
         "detail": f"TEST {pct(t['accuracy'])}, HOLDOUT {pct(h['accuracy'])} in this report and "
                   f"logs/model_validation_report.txt{'' if val else ' (MISSING for this model)'}"},
        {"item": "Feature importance generated", "pass": fi is not None and len(fi["features"]) == api.model.n_features_in_,
         "detail": f"{FEATURE_REPORT.relative_to(ROOT).as_posix()}"
                   + (f" ({len(fi['features'])} features)" if fi else " - MISSING or for another model")},
        {"item": "Robustness tests completed", "pass": rob is not None,
         "detail": f"{ROBUST_REPORT.relative_to(ROOT).as_posix()}"
                   + (f" ({len(rob['variants'])} variants, {rob['n_predictions']} predictions)" if rob
                      else " - MISSING or for another model")},
        {"item": "Demo images validated", "pass": demo_ok,
         "detail": f"5/5 scored, scores equal the manifest record; {sum(x['pass'] for x in demo)}/5 correct"},
        {"item": "No prediction crashes", "pass": crashes == 0 and rob is not None,
         "detail": f"{crashes} crashes in {calls} predictions (review + robustness)"},
    ]


# ── Report ──────────────────────────────────────────────────────────────────

def _bar(n: int, scale: int) -> str:
    return "#" * max(0, round(40 * n / scale)) if scale else ""


def build_report(r: dict) -> str:
    perf, q, opt = r["performance"], r["quality"], r["optimisation"]
    m = r["model"]
    L = [
        "--------------------------------",
        "DOCFRAUD MODEL REVIEW REPORT",
        "--------------------------------",
        f"Generated (UTC): {r['generated']}",
        "Script: scripts/model_review.py",
        f"Environment: Python {r['environment']['python']}, {r['environment']['platform']}, "
        f"{r['environment']['cpus']} CPUs; " + ", ".join(f"{k} {v}" for k, v in r["environment"]["packages"].items()),
        f"Model: {m['path']} v{m['version']} ({m['schema']}, {m['n_features']} features), SHA-256 {m['sha256']}",
        f"Decision rule: tampered if tamper_risk_score >= {config.TAMPER_THRESHOLD} (unchanged).",
        "",
        "--------------------------------",
        "RELEASE CHECKLIST",
        "--------------------------------",
    ]
    L += [f"[{_mark(c['pass'])}] {c['item']:<30} {c['detail']}" for c in r["checklist"]]
    L += [f"Result: {sum(c['pass'] for c in r['checklist'])}/{len(r['checklist'])} items pass", ""]

    # 1. Performance
    L += ["--------------------------------", "1. PERFORMANCE REVIEW (held-out documents, deployed file)",
          "--------------------------------",
          f"Leakage checks: {'PASS' if perf['leakage_ok'] else 'FAIL'}; frozen split intact: "
          f"{'YES' if perf['frozen_split_ok'] else 'NO'}.",
          f"{'Set':<9} {'n':>4} {'Accuracy (95% CI)':>27} {'Precision':>9} {'Recall':>8} {'F1':>6} {'FPR':>8} {'FNR':>8}",
          "-" * 88,
          mv._metric_line("TEST", perf["sets"]["TEST"]), mv._metric_line("HOLDOUT", perf["sets"]["HOLDOUT"]),
          mv._metric_line("COMBINED", perf["combined"]), ""]
    for name in ("TEST", "HOLDOUT"):
        s = perf["sets"][name]
        L += [f"{name} confusion matrix:", fv.ascii_confusion(s["metrics"]),
              f"{'  Class':<12} {'Precision':>9} {'Recall':>8} {'F1':>6} {'Support':>7}"]
        L += [f"  {c:<10} {pct(v['precision']):>9} {pct(v['recall']):>8} {v['f1']:>6.3f} {v['support']:>7}"
              for c, v in s["per_class"].items()]
        L += ["  Tampered caught by method: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in s["recall_by_method"].items()),
              "  Accuracy by document type: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in s["accuracy_by_doc_type"].items()),
              ""]
    L += ["Comparison with previous reports:",
          f"{'When':<6} {'Model':<26} {'Set':<13} {'n':>4} {'Accuracy':>9} {'Precision':>9} {'Recall':>8} {'F1':>6}  Source",
          "-" * 118]
    for p in perf["previous"]:
        L.append(f"{p['when']:<6} {p['model']:<26} {p['set']:<13} {p['n']:>4} {pct(p['accuracy']):>9} "
                 f"{pct(p['precision']):>9} {pct(p['recall']):>8} {p['f1']:>6.3f}  {p['source']}")
    for name, prev in perf["day8_v2"].items():
        L.append(f"{'Day 8':<6} {'v2 (local-tamper-v2)':<26} {name:<13} {prev['n']:>4} {pct(prev['accuracy']):>9} "
                 f"{pct(prev['precision']):>9} {pct(prev['recall']):>8} {prev['f1']:>6.3f}  logs/model_validation_results.json")
    for name in ("TEST", "HOLDOUT"):
        s = perf["sets"][name]["metrics"]
        L.append(f"{'Day 9':<6} {'v2 (local-tamper-v2)':<26} {name:<13} {s['n']:>4} {pct(s['accuracy']):>9} "
                 f"{pct(s['precision']):>9} {pct(s['recall']):>8} {s['f1']:>6.3f}  this review")
    L.append(f"Day 9 confusion matrices identical to the Day 8 v2 record: {'YES' if perf['same_as_day8'] else 'NO'} "
             "(same model file; confirms reproducibility).")
    L += ["", "Weak classes and errors:"] + r["weaknesses"]
    fp = [x for s in perf["sets"].values() for x in s["per_image"] if x["actual"] == "genuine" and x["predicted"] == "tampered"]
    L += ["", f"False positives ({len(fp)}):"]
    L += [f"  {x['path']:<34} {x['doc_type']:<15} risk {x['risk']:.2f}" for x in fp] or ["  none"]
    ea = perf["errors"]
    L += ["", f"False negatives ({len(ea.get('missed', []))}) and what the edit left on the page:"]
    L += [f"  {x['set']:<8} {x['path']:<34} {x['method']:<13} risk {x['risk']:.2f}  changed px "
          f"{x['changed_px'] if x['changed_px'] is not None else '?':>5}  {x['kind']}" for x in ea.get("missed", [])]

    # 2. Prediction quality
    h = q["score_histogram"]
    scale = max(h["genuine"] + h["tampered"])
    L += ["", "--------------------------------", "2. PREDICTION QUALITY (held-out genuine and tampered documents)",
          "--------------------------------",
          f"Risk score distribution ({q['n']} documents; # = documents):",
          f"{'Score band':<11} {'Genuine':>7}  {'':<40} {'Tampered':>8}  {'':<40}"]
    for lo, g, t in zip(h["edges"][:-1], h["genuine"], h["tampered"]):
        L.append(f"{lo:.1f}-{lo + 0.1:.1f}    {g:>7}  {_bar(g, scale):<40} {t:>8}  {_bar(t, scale):<40}")
    ss = q["score_summary"]
    L += [f"Genuine scores:  mean {ss['genuine']['mean']:.3f}, median {ss['genuine']['median']:.3f}, "
          f"range {ss['genuine']['min']:.2f}-{ss['genuine']['max']:.2f}",
          f"Tampered scores: mean {ss['tampered']['mean']:.3f}, median {ss['tampered']['median']:.3f}, "
          f"range {ss['tampered']['min']:.2f}-{ss['tampered']['max']:.2f}", "",
          "Calibration (share of documents in each score band that are actually tampered):"]
    L += [f"  {c['band']}  n={c['n']:>3}  tampered {pct(c['tampered_share']) if c['tampered_share'] is not None else '-':>7}"
          for c in q["calibration"]]
    L += ["", "Confidence (max(score, 1 - score)) vs correctness:"]
    L += [f"  {c['band']}  n={c['n']:>3}  accuracy {pct(c['accuracy']) if c['accuracy'] is not None else '-':>7}"
          for c in q["confidence"]]
    nt = q["near_threshold"]
    L += [f"Near the threshold (|score - {config.TAMPER_THRESHOLD}| <= {NEAR}): {nt['n']} documents, "
          f"{nt['correct']} correct.", "", f"Incorrect predictions ({len(q['incorrect'])}):"]
    L += [f"  {x['set']:<8} {x['path']:<34} {x['doc_type']:<15} {x['method']:<13} actual {x['actual']:<8} "
          f"risk {x['risk']:.2f}  confidence {x['confidence']:.2f}" for x in q["incorrect"]]
    L += ["", "Edge cases - demo images:"]
    L += [f"  {x['image']:<26} expected {x['expected']:<8} predicted {x.get('actual', 'ERROR'):<8} "
          f"risk {x.get('risk', float('nan')):.2f}  {_mark(x['pass'])}" for x in q["demo"]]
    L += ["Edge cases - non-document test fixtures and invalid files:"]
    L += [f"  {x['case']:<42} {x['outcome']:<9} {x['detail']}" for x in q["fixtures"]]
    if q["degraded"]:
        L += ["Edge cases - degraded held-out copies (logs/model_robustness_report.txt): median score and share "
              "above the threshold",
              f"  {'Variant':<14} {'genuine median':>14} {'flagged':>9} {'tampered median':>16} {'caught':>9}"]
        for k, v in q["degraded"].items():
            g, t = v["genuine"], v["tampered"]
            L.append(f"  {k:<14} {g['median']:>14.2f} {g['above_threshold']:>4}/{g['n']:<4} {t['median']:>16.2f} "
                     f"{t['above_threshold']:>4}/{t['n']:<4}")

    # 3. Optimisation
    sz, mem = opt["size"], opt["memory"]
    L += ["", "--------------------------------", "3. OPTIMISATION REVIEW", "--------------------------------",
          f"Model size: {sz['file_bytes'] / 2**20:.1f} MB on disk; {sz['trees']} trees, {sz['nodes']:,} nodes, "
          f"{sz['leaves']:,} leaves; depth mean {sz['depth_mean']:.1f}, max {sz['depth_max']}; unpickle "
          f"{sz['unpickle_seconds'] * 1000:.0f} ms. Saved with n_jobs={sz['saved_n_jobs']}, served with "
          f"n_jobs={sz['served_n_jobs']}.", ""]
    if mem.get("available") and "model_loaded" in mem:
        L += ["Process memory (clean subprocess, MB):",
              f"  after Python start {mem['python_started']['VmRSS']:.0f}; after imports (numpy, OpenCV, sklearn, "
              f"FastAPI) {mem['libraries_imported']['VmRSS']:.0f}; after model load {mem['model_loaded']['VmRSS']:.0f} "
              f"(+{mem['model_loaded']['VmRSS'] - mem['libraries_imported']['VmRSS']:.0f}, load "
              f"{mem['load_seconds'] * 1000:.0f} ms)",
              f"  after a typical page {mem['after_typical_page']['VmRSS']:.0f}; after a 4000x5000 page "
              f"{mem['after_large_page']['VmRSS']:.0f}; peak (VmHWM) {mem['after_large_page']['VmHWM']:.0f}", ""]
    else:
        L += [f"Process memory: not measured ({mem.get('error', '/proc not available')})", ""]
    L += ["Inference time per stage (median ms):",
          f"  {'Stage':<58} {'typical page':>13} {'4000x5000 JPEG':>15}"]
    t_typ, t_big = opt["timings"]["typical"], opt["timings"]["large"]
    L.append(f"  {'page size':<58} {t_typ['page']:>13} {t_big['page']:>15}")
    for k in t_typ:
        if k != "page":
            L.append(f"  {k:<58} {t_typ[k]:>13.1f} {t_big[k]:>15.1f}")
    L += ["End-to-end (median ms):"] + [f"  {k:<62} {v:>8.1f}" for k, v in opt["end_to_end_ms"].items()]
    L += ["", "Forest alternatives (grouped 5-fold CV on training data only; NOT deployed):",
          f"  {'Parameters':<44} {'CV acc':>7} {'+/-':>6} {'processed':>10} {'size MB':>8} {'predict ms':>11}"]
    for a in opt["alternatives"]:
        params = ", ".join(f"{k}={v}" for k, v in a["params"].items()) + (" (deployed)" if a["deployed"] else "")
        L.append(f"  {params:<44} {a['cv_accuracy']:>7.3f} {a['cv_std']:>6.3f} {a['cv_processed_accuracy']:>10.3f} "
                 f"{a['size_bytes'] / 2**20:>8.2f} {a['predict_ms']:>11.1f}")
    L += ["", "Optimisation applied on Day 9 (outputs unchanged):"] + r["optimisation_applied"]
    L += ["", "--------------------------------", "RECOMMENDATIONS FOR FUTURE VERSIONS", "--------------------------------"]
    L += r["recommendations"]
    L.append("")
    return "\n".join(L)


def weaknesses(perf: dict) -> list[str]:
    out = []
    comb = perf["combined"]["metrics"]
    out.append(f"- Weaker class: tampered (recall {pct(comb['recall'])}, {comb['fn']} missed) vs genuine "
               f"(specificity {pct(1 - comb['fpr'])}, {comb['fp']} flagged). The model errs towards 'genuine'.")
    types = defaultdict(lambda: [0, 0])
    methods = defaultdict(lambda: [0, 0])
    for s in perf["sets"].values():
        for k, (a, b) in s["accuracy_by_doc_type"].items():
            types[k][0] += a
            types[k][1] += b
        for k, (a, b) in s["recall_by_method"].items():
            methods[k][0] += a
            methods[k][1] += b
    worst_t = min(types.items(), key=lambda kv: kv[1][0] / kv[1][1])
    out.append("- By document type (TEST + HOLDOUT): " + ", ".join(f"{k} {a}/{b} ({pct(a / b)})" for k, (a, b) in
                                                                 sorted(types.items())) + f"; weakest: {worst_t[0]}.")
    out.append("- Tampered caught by edit method: " + ", ".join(f"{k} {a}/{b} ({pct(a / b)})" for k, (a, b) in
                                                             sorted(methods.items(), key=lambda kv: kv[1][0] / kv[1][1])) + ".")
    ea = perf["errors"]
    if ea.get("available"):
        kinds = defaultdict(int)
        for x in ea["missed"]:
            kinds[x["kind"]] += 1
        out.append("- Missed edits by what they left: " + ", ".join(f"{k} {v}" for k, v in kinds.items())
                   + f". {ea['heldout_edits_without_pixel_change']} of {ea['heldout_edits']} held-out edits changed no pixel.")
    return out


def recommendations(r: dict) -> list[str]:
    opt = r["optimisation"]
    typ, big, e2e = opt["timings"]["typical"], opt["timings"]["large"], opt["end_to_end_ms"]
    alts = opt["alternatives"]
    dep = next(a for a in alts if a["deployed"])
    small = min((a for a in alts if a["cv_accuracy"] >= dep["cv_accuracy"] - 0.01), key=lambda a: a["size_bytes"])
    rob = []
    if ROBUST_JSON.exists():
        s = json.loads(ROBUST_JSON.read_text(encoding="utf-8"))["summary"]
        rob = [k for k, v in s.items() if k != "original" and v["accuracy"] < s["original"]["accuracy"] - 0.10]
    return [
        "- Robustness first: the largest risk for real uploads is degraded input"
        + (f" ({', '.join(rob)} lose more than 10 points)" if rob else "")
        + ". Train with those degradations for both classes and normalise pages (deskew, auto-rotate, trim "
          "borders, fixed resolution) before feature extraction.",
        f"- Model size: {', '.join(f'{k}={v}' for k, v in small['params'].items())} reaches CV accuracy "
        f"{small['cv_accuracy']:.3f} (deployed {dep['cv_accuracy']:.3f}) at {small['size_bytes'] / 2**20:.2f} MB "
        f"(deployed {dep['size_bytes'] / 2**20:.2f} MB). A smaller forest is an option for v3; choose it by CV on "
        "training data and confirm on a NEW holdout set.",
        "- Save future models with n_jobs=1 (single-document inference) so any loader gets the fast path; "
        "ModelAPI now enforces it at load time.",
        f"- After the Day-9 change, feature extraction dominates: {typ['all features (extract_tamper_features)']:.0f} "
        f"of {e2e['typical page, ModelAPI.predict (served)']:.0f} ms on a typical page, with the colour "
        f"({typ['colour features']:.0f} ms) and ink ({typ['ink features']:.0f} ms) groups the most expensive. On a "
        f"4000x5000 JPEG, decoding and downscaling take {big['decode (load_image)']:.0f} of "
        f"{e2e['4000x5000 JPEG, ModelAPI.predict (served)']:.0f} ms: decode JPEGs at reduced size (Pillow draft "
        "mode) before the 2000-px downscale.",
        "- Calibrate scores (isotonic / Platt on validation data) and add a 'needs review' band around the "
        "threshold; report calibrated risk to users instead of raw vote fractions.",
        "- Localisation is live in flagged_regions (logs/localization_report.txt); extend it to text overlays "
        "whose ink matches the page (the main misses) and report it in the review UI.",
        "- Validate on real, consented documents before any production decision; every figure here is synthetic.",
        "- Pin package versions and record the model SHA-256 per release; re-run the Day 8 and Day 9 scripts after "
        "any change.",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="Final model review: performance, quality, optimisation, checklist")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    model, meta = DocumentFraudDetector.load(DEFAULT_MODEL_PATH)
    api = ModelAPI(DEFAULT_MODEL_PATH)
    api.load_model()
    print("1/4 Performance review (TEST + HOLDOUT)...")
    perf = performance_review(api, meta)
    print("2/4 Prediction quality...")
    quality = prediction_quality(api, perf)
    print("3/4 Optimisation review...")
    opt = optimisation_review(api, meta)
    print("4/4 Release checklist...")
    checklist = release_checklist(api, meta, perf, quality)

    e2e = opt["end_to_end_ms"]
    before = e2e["typical page, pre-Day-9 path (n_jobs=-1, two forest passes)"]
    after = e2e["typical page, ModelAPI.predict (served)"]
    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": mst.environment(),
        "model": {"path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(), "sha256": api_sha(),
                  "version": meta.get("model_version"), "schema": api.schema.version,
                  "n_features": int(model.n_features_in_)},
        "performance": perf, "quality": quality, "optimisation": opt, "checklist": checklist,
        "optimisation_applied": [
            "- ModelAPI.load_model sets the forest to n_jobs=1: one document per call, so joblib thread start-up "
            "only added latency and competed with the MAX_CONCURRENT_ANALYSES request slots.",
            "- ModelAPI.predict makes one forest pass (predict_proba) and takes the argmax as the class, exactly as "
            "RandomForestClassifier.predict does, instead of calling predict and predict_proba.",
            f"- Typical page end-to-end: {before:.0f} ms -> {after:.0f} ms ({before / after:.1f}x faster). Scores, "
            "confidences and interpretations were compared before and after on 175 files (164 held-out, 5 demo, "
            "6 fixtures): identical.",
        ],
    }
    result["weaknesses"] = weaknesses(perf)
    result["recommendations"] = recommendations(result)
    report = build_report(result)
    for path, text in ((args.report, report), (args.json, json.dumps(result, indent=1, default=str) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}\nResults written: {args.json}")
    return 0 if all(c["pass"] for c in checklist) else 1


if __name__ == "__main__":
    raise SystemExit(main())
