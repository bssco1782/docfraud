"""
scripts/final_model_validation.py
Founder A (Day 10): final model validation, release review and sign-off.

Usage:
    python scripts/final_model_validation.py            # reuses current accuracy / benchmark results
    python scripts/final_model_validation.py --rerun    # re-runs model_accuracy_eval.py and inference_benchmark.py
    python scripts/final_model_validation.py --runs 10  # repeats per demo image

Steps:
  1. Model validation: model file and SHA-256; dependencies (runtime versions against those recorded in
     the model file, requirements pin, pip check); serialization (unpickle warnings, bundle structure,
     re-pickle round trip, rollback model); runtime (every accepted input type is processed, invalid
     inputs are rejected with a controlled error and never crash).
  2. Demo validation: each demo image through ModelAPI.predict (the /verify call), giving prediction,
     confidence, risk score and pass/fail. Repeated --runs times and compared with the scores recorded in
     data/demo/demo_manifest.json.
  3. Accuracy: logs/final_accuracy_results.json (scripts/model_accuracy_eval.py).
  4. Performance: logs/inference_benchmark_results.json (scripts/inference_benchmark.py).
  5. Release review: feature extraction pipeline, ELA pipeline, prediction logic and thresholds.
  6. Sign-off checklist and release recommendation.
A results file from step 3 or 4 is used only if it was produced with the current model file (same
SHA-256). Otherwise, or with --rerun, that script is run again here.

Writes logs/final_model_release_report.txt, logs/model_signoff_checklist.txt and
logs/final_model_release_results.json. Exit code 0 when every sign-off item passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import logging
import pickle
import re
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_validation as fv  # noqa: E402
import inference_benchmark as ib  # noqa: E402
import model_accuracy_eval as mae  # noqa: E402
import model_stability_test as mst  # noqa: E402
import model_validation as mv  # noqa: E402

from backend import config  # noqa: E402
from backend.model_utils import ModelAPI  # noqa: E402
from core import document_analyzer, tamper_features  # noqa: E402
from core.model_trainer import DEFAULT_MODEL_PATH  # noqa: E402

DEFAULT_REPORT = ROOT / "logs" / "final_model_release_report.txt"
DEFAULT_CHECKLIST = ROOT / "logs" / "model_signoff_checklist.txt"
DEFAULT_JSON = ROOT / "logs" / "final_model_release_results.json"
LIMITATIONS = ROOT / "logs" / "model_limitations.txt"
RELEASE_NOTES = ROOT / "logs" / "model_release_notes.txt"
ROBUSTNESS_JSON = ROOT / "logs" / "model_robustness_results.json"
LOCALIZATION_JSON = ROOT / "logs" / "localization_results.json"
FOUNDER_B_REPORT = ROOT / "logs" / "final_release_report.txt"
ROLLBACK_MODEL = DEFAULT_MODEL_PATH.parent / "fraud_model_v1.pkl"
REQUIREMENTS = ROOT / "requirements.txt"
RUNS = 10
LOW_CONFIDENCE = 0.60  # confidence below this = the score is within 0.1 of the threshold
REQUIRED_METADATA = ("model_version", "feature_schema_version", "feature_names", "decision_threshold",
                     "training_files", "environment", "rf_params")
TAMPERED_TEXT = "Evidence of digital tampering"
GENERATOR_EDITS = {"blur_region", "color_shift", "noise_patch", "text_overlay"}  # v2's training edits
SECTIONS = ("SECTION 1: FINAL METRICS", "SECTION 2: DEMO VALIDATION RESULTS", "SECTION 3: PERFORMANCE RESULTS",
            "SECTION 4: KNOWN LIMITATIONS", "SECTION 5: RELEASE RECOMMENDATION")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, status: bool | str, detail: str = "") -> dict:
    """status: True/False or 'PASS' / 'WARN' / 'FAIL' / 'INFO'. Only FAIL blocks sign-off."""
    if isinstance(status, bool):
        status = "PASS" if status else "FAIL"
    return {"check": name, "status": status, "detail": detail}


def failed(checks: list[dict]) -> list[dict]:
    return [c for c in checks if c["status"] == "FAIL"]


# ── 1. Model validation ─────────────────────────────────────────────────────

def dependency_checks(meta: dict) -> list[dict]:
    out = []
    recorded = meta.get("environment") or {}
    for pkg, version in recorded.items():
        try:
            now = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            now = "not installed"
        if now == version:
            out.append(check(f"{pkg} {now} == version the model was saved with", True))
        elif pkg == "scikit-learn":  # the pickle contract
            out.append(check(f"scikit-learn matches the model file ({version})", False, f"installed {now}"))
        else:
            out.append(check(f"{pkg} matches the model file ({version})", "WARN",
                             f"installed {now}; features may differ slightly - re-run the validation scripts"))
    pin = re.search(r"^scikit-learn==(\S+)", REQUIREMENTS.read_text(encoding="utf-8"), re.M)
    out.append(check("requirements.txt pins scikit-learn to the model's version",
                     bool(pin) and pin.group(1) == recorded.get("scikit-learn"),
                     f"scikit-learn=={pin.group(1)}" if pin else "not pinned"))
    for module, why in (("pymupdf", "PDF uploads"), ("cv2", "image features"), ("scipy", "region labelling"),
                        ("fastapi", "API"), ("uvicorn", "server")):
        try:
            __import__(module)
            out.append(check(f"import {module} ({why})", True))
        except ImportError as exc:
            out.append(check(f"import {module} ({why})", False, str(exc)))
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    out.append(check("pip check (installed packages consistent)", "PASS" if pip.returncode == 0 else "WARN",
                     (pip.stdout or pip.stderr).strip().splitlines()[0][:100] if (pip.stdout or pip.stderr) else ""))
    return out


def serialization_checks(model, meta: dict, demo_vectors: np.ndarray) -> list[dict]:
    out = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(DEFAULT_MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
    out.append(check("unpickling raises no warnings (e.g. InconsistentVersionWarning)", not caught,
                     "; ".join(f"{w.category.__name__}: {w.message}"[:90] for w in caught) or "0 warnings"))
    out.append(check("bundle is {'model', 'metadata'}", isinstance(bundle, dict) and set(bundle) == {"model", "metadata"}))
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    out.append(check("metadata complete", not missing, f"missing: {missing}" if missing else
                     f"{len(REQUIRED_METADATA)} required fields present"))
    clone = pickle.loads(pickle.dumps(model))
    same = np.array_equal(clone.predict_proba(demo_vectors), model.predict_proba(demo_vectors))
    out.append(check("re-pickled model gives identical probabilities", same, f"{len(demo_vectors)} demo vectors"))
    notes = RELEASE_NOTES.read_text(encoding="utf-8") if RELEASE_NOTES.exists() else ""
    rec = re.search(r"^SHA-256\s+:\s+([0-9a-f]{64})", notes, re.M)
    out.append(check("SHA-256 matches logs/model_release_notes.txt", bool(rec) and rec.group(1) == sha256(DEFAULT_MODEL_PATH),
                     rec.group(1)[:16] + "..." if rec else "no SHA-256 in the release notes"))
    v1 = re.search(r"fraud_model_v1\.pkl \(SHA-256 ([0-9a-f]+)", notes)
    if ROLLBACK_MODEL.exists():
        api_v1 = ModelAPI(ROLLBACK_MODEL)
        try:
            ok = api_v1.load_model()
        except RuntimeError as exc:
            ok, api_v1.last_error = False, str(exc)
        out.append(check("rollback model v1 loads through ModelAPI", ok,
                         f"schema {api_v1.schema.version}" if ok else str(api_v1.last_error)))
        out.append(check("rollback model SHA-256 matches the release notes",
                         bool(v1) and sha256(ROLLBACK_MODEL).startswith(v1.group(1)),
                         v1.group(1) if v1 else "not recorded"))
    else:
        out.append(check("rollback model v1 present", "WARN", f"{ROLLBACK_MODEL.name} missing - no rollback"))
    return out


def runtime_checks(api: ModelAPI, demo_file: str, tmp: Path) -> tuple[list[dict], list[dict]]:
    rows = mv.inference_robustness(api, demo_file)  # fixtures, truncated / empty / text files
    demo = mst.DEMO_DIR / demo_file
    with Image.open(demo) as im:
        page = im.convert("RGB")
    extra = []
    rgba = io.BytesIO()
    page.convert("RGBA").save(rgba, format="PNG")
    extra.append(("demo page as RGBA PNG", "rgba.png", rgba.getvalue()))
    jpg = io.BytesIO()
    page.save(jpg, format="JPEG", quality=90)
    extra.append(("demo page as JPEG q90", "page.jpeg", jpg.getvalue()))
    try:
        import pymupdf

        pdf = tmp / "page.pdf"
        with pymupdf.open() as doc:
            p = doc.new_page(width=595, height=842)
            p.insert_image(p.rect, filename=str(demo))
            doc.save(str(pdf))
        extra.append(("demo page in an A4 PDF", "page.pdf", pdf.read_bytes()))
    except ImportError:
        pass
    for name, filename, data in extra:
        try:
            out = api.predict(data, filename=filename)
            ok = 0 <= out["tamper_risk_score"] <= 1
            rows.append({"case": name, "bytes": len(data), "expected": "processed", "outcome": "processed",
                         "detail": f"risk {out['tamper_risk_score']:.2f}", "crash": not ok, "pass": ok})
        except Exception as exc:
            rows.append({"case": name, "bytes": len(data), "expected": "processed", "outcome": "error",
                         "detail": f"{type(exc).__name__}: {exc}"[:90], "crash": not isinstance(exc, ValueError),
                         "pass": False})
    processed = [r for r in rows if r["expected"] == "processed"]
    rejected = [r for r in rows if r["expected"] == "rejected"]
    checks = [
        check("every valid input processed (PNG, JPEG, grayscale, RGBA, small, large, PDF)",
              all(r["pass"] for r in processed), f"{sum(r['pass'] for r in processed)}/{len(processed)}"),
        check("invalid inputs rejected with a controlled error (HTTP 400), not a crash",
              all(r["pass"] for r in rejected), f"{sum(r['pass'] for r in rejected)}/{len(rejected)}"),
        check("no runtime crashes", not any(r["crash"] for r in rows), f"{len(rows)} inputs"),
    ]
    return checks, rows


# ── 2. Demo validation ──────────────────────────────────────────────────────

def demo_validation(api: ModelAPI, images: list[dict], runs: int) -> dict:
    rows = mv.demo_validation(api, images)
    rep = {r["demo_file"]: r for r in mst.repeatability(api, images, runs)}
    for row, rec in zip(rows, images):
        rp = rep[rec["demo_file"]]
        row["stable"] = rp["pass"]
        row["max_score_deviation"] = rp.get("score", {}).get("max_abs_dev")
        row["matches_record"] = row.get("risk") == rec["tamper_risk_score"]
        if "risk" in row:
            row["confidence_consistent"] = (row["confidence"] >= 0.5
                                            and abs(row["confidence"] - max(row["risk"], 1 - row["risk"])) < 1e-3)
    ok = [r for r in rows if "risk" in r]
    return {
        "runs": runs, "rows": rows,
        "correct": sum(r["pass"] for r in rows), "n": len(rows),
        "valid_outputs": sum(r.get("valid_output", False) for r in rows),
        "stable": all(r["stable"] for r in rows),
        "match_record": all(r["matches_record"] for r in rows),
        "confidence_consistent": all(r.get("confidence_consistent", False) for r in rows),
        "low_confidence": [r["image"] for r in ok if r["confidence"] < LOW_CONFIDENCE],
        "crashes": sum(r.get("crash", False) for r in rows) + sum(v["crashes"] for v in rep.values()),
        "wrong": [r["image"] for r in rows if not r["pass"]],
    }


# ── 5. Release review ───────────────────────────────────────────────────────

def review_feature_pipeline(api: ModelAPI, meta: dict, demo_paths: list[Path]) -> list[dict]:
    out = [check("extractor chosen from the model's recorded schema",
                 api.schema.version == meta.get("feature_schema_version") == tamper_features.FEATURE_SCHEMA_VERSION,
                 f"{api.schema.version}: {len(api.schema.names)} features")]
    finite = deterministic = parity = True
    for p in demo_paths:
        runs = [tamper_features.extract_tamper_features(p) for _ in range(3)]
        finite &= list(runs[0]) == list(api.schema.names) and all(np.isfinite(v) for v in runs[0].values())
        deterministic &= all(r == runs[0] for r in runs[1:])
        parity &= tamper_features.extract_with_regions(p)[0] == runs[0]
    out += [check("29 finite features in model column order", finite, f"{len(demo_paths)} demo images"),
            check("features deterministic (3 extractions per image)", deterministic),
            check("served extractor (extract_with_regions) == training extractor (extract_tamper_features)", parity)]
    return out


def review_ela_pipeline(api: ModelAPI, images: list[dict], manifest: dict) -> list[dict]:
    out = []
    ela_named = [n for n in api.schema.names if n.startswith("ela_")]
    calls = {"n": 0}
    original = document_analyzer.ela_analysis

    def counting(*a, **k):
        calls["n"] += 1
        return original(*a, **k)

    document_analyzer.ela_analysis = counting
    try:
        for rec in images:
            api.predict(mst.DEMO_DIR / rec["demo_file"])
        v2_calls = calls["n"]
        v1_scores, v1_regions = {}, []
        if ROLLBACK_MODEL.exists():
            api_v1 = ModelAPI(ROLLBACK_MODEL)
            api_v1.load_model()
            for rec in images:
                out_v1 = api_v1.predict(mst.DEMO_DIR / rec["demo_file"])
                v1_scores[rec["demo_file"]] = out_v1["tamper_risk_score"]
                v1_regions += out_v1["flagged_regions"]
    finally:
        document_analyzer.ela_analysis = original
    out.append(check("deployed model v2 does not use ELA", not ela_named and v2_calls == 0,
                     f"{len(ela_named)} ELA features; ela_analysis called {v2_calls} times in {len(images)} v2 predictions"))
    page = mst.DEMO_DIR / images[0]["demo_file"]
    h1, s1 = document_analyzer.ela_analysis(page)
    h2, s2 = document_analyzer.ela_analysis(page)
    size = document_analyzer.load_image(page).size
    out.append(check("ELA pipeline (v1 rollback path): heatmap matches the page, score in 0-100, deterministic",
                     h1.shape[:2] == (size[1], size[0]) and 0 <= s1 <= 100 and s1 == s2 and np.array_equal(h1, h2),
                     f"heatmap {h1.shape[1]}x{h1.shape[0]}, score {s1:.1f}"))
    history = next((h for h in manifest.get("scoring_history", [])
                    if ROLLBACK_MODEL.exists() and h["model"].get("sha256") == sha256(ROLLBACK_MODEL)), None)
    if history and v1_scores:
        recorded = {i["demo_file"]: i["tamper_risk_score"] for i in history["images"]}
        same = all(abs(v1_scores[k] - recorded[k]) < 1e-9 for k in recorded)
        out.append(check("rollback model v1 reproduces its recorded demo scores (uses ELA; no regions)",
                         same and calls["n"] > v2_calls and not v1_regions,
                         ", ".join(f"{v1_scores[k]:.2f}" for k in recorded)))
    else:
        out.append(check("rollback model v1 reproduces its recorded demo scores", "WARN", "no v1 record"))
    return out


def review_prediction_logic(api: ModelAPI, model, docs: list[Path]) -> list[dict]:
    exact = rule = regions = True
    for p in docs:
        out = api.predict(p)
        vec = np.array([api.schema.vector(tamper_features.extract_tamper_features(p))])
        proba = model.predict_proba(vec)[0]
        exact &= (out["tamper_risk_score"] == round(float(proba[1]), 4)
                  and out["confidence"] == round(float(proba.max()), 4))
        flagged = out["tamper_risk_score"] >= config.TAMPER_THRESHOLD
        rule &= out["interpretation"].startswith(TAMPERED_TEXT) == flagged
        regions &= flagged or not out["flagged_regions"]
    return [
        check("class order [genuine=0, tampered=1]; score = tampered column", [int(c) for c in model.classes_] == [0, 1]),
        check("API score / confidence == forest probabilities on independently extracted features", exact,
              f"{len(docs)} documents (demo + TEST)"),
        check("verdict text follows score >= TAMPER_THRESHOLD", rule, f"{len(docs)} documents"),
        check("flagged_regions only for a tampering verdict", regions, f"{len(docs)} documents"),
    ]


def review_thresholds(meta: dict, manifest: dict, demo_file: str, accuracy: dict) -> list[dict]:
    from fastapi.testclient import TestClient

    from backend import main as backend_main

    thr, rec, dem = config.TAMPER_THRESHOLD, meta.get("decision_threshold"), manifest.get("risk_threshold")
    out = [check("TAMPER_THRESHOLD == model metadata == demo manifest", thr == rec == dem,
                 f"config {thr}, model {rec}, demo manifest {dem}"),
           check("threshold within [0, 1]", 0.0 <= thr <= 1.0, str(thr))]
    saved = config.TAMPER_THRESHOLD
    try:
        config.TAMPER_THRESHOLD = 1.5
        warned = any("TAMPER_THRESHOLD" in w for w in backend_main.validate_startup_configuration())
    finally:
        config.TAMPER_THRESHOLD = saved
    out.append(check("startup validation warns about an out-of-range TAMPER_THRESHOLD", warned))
    key, saved_key = "signoff-key", config.API_KEY
    config.API_KEY = key
    try:
        with TestClient(backend_main.app) as client:
            content = (mst.DEMO_DIR / demo_file).read_bytes()
            bodies = [client.post("/verify", params=params, headers={"X-API-Key": key},
                                  files={"file": (demo_file, content, "image/png")}).json()
                      for params in (None, {"threshold": 0.01}, {"threshold": 0.99})]
    finally:
        config.API_KEY = saved_key
    out.append(check("'threshold' query parameter does not change the result (documented as ignored)",
                     bodies[0] == bodies[1] == bodies[2] and "tamper_risk_score" in bodies[0],
                     f"risk {bodies[0].get('tamper_risk_score')} with threshold unset, 0.01 and 0.99"))
    nt = accuracy.get("near_threshold", {})
    out.append(check(f"held-out documents within {nt.get('band')} of the threshold", "INFO",
                     f"{nt.get('n')} of 164, {nt.get('wrong')} wrong - the score is least reliable there"))
    return out


# ── Supporting evidence ─────────────────────────────────────────────────────

def current_results(path: Path, model_sha: str) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if data.get("model", {}).get("sha256") == model_sha else None


def issues_found(r: dict) -> list[dict]:
    """Non-blocking issues found during the review, with their status."""
    issues = []
    pinned = any(c["check"].startswith("requirements.txt pins scikit-learn") and c["status"] == "PASS"
                 for c in r["model_validation"]["dependencies"])
    issues.append({"issue": "scikit-learn was unpinned in requirements.txt although the model is a pickle; an "
                            "environment with another version would fail or warn at load",
                   "severity": "high for deployment", "status": "fixed: pinned to 1.9.1" if pinned else "OPEN"})
    fixtures = [(x["case"], float(x["detail"].split()[1])) for x in r["model_validation"]["runtime_rows"]
                if x["expected"] == "processed" and x["case"].endswith((".jpg", ".png")) and x["detail"].startswith("risk")]
    flagged = [f"{name} {risk:.2f}" for name, risk in fixtures if risk >= config.TAMPER_THRESHOLD]
    issues.append({"issue": f"non-document images get a score: {len(flagged)} of {len(fixtures)} synthetic test "
                            f"fixtures (not documents) are reported as tampered ({', '.join(flagged)})",
                   "severity": "medium (misleading output)", "status": "known limitation, documented"})
    if FOUNDER_B_REPORT.exists() and "v1.0.0" in FOUNDER_B_REPORT.read_text(encoding="utf-8"):
        issues.append({"issue": "logs/final_release_report.txt and logs/final_load_report.txt (Founder B, Day 10) "
                                "describe model v1 (60%); their model figures are superseded by this report",
                       "severity": "low (documentation)", "status": "noted; Founder B's dated reports left unchanged"})
    fps = [p for n in ("TEST", "HOLDOUT") for p in r["accuracy"]["sets"][n]["per_image"]
           if p["actual"] == "genuine" and p["predicted"] == "tampered"]
    types = sorted({p["doc_type"] for p in fps})
    issues.append({"issue": f"all {len(fps)} held-out false positives are {', '.join(types)} pages",
                   "severity": "medium (weakest document type)", "status": "known limitation, documented"})
    issues.append({"issue": "the 'threshold' query parameter of /verify is accepted but ignored",
                   "severity": "low", "status": "documented behaviour (API_DOCS.md)"})
    return issues


# ── Sign-off ────────────────────────────────────────────────────────────────

def signoff(r: dict) -> list[dict]:
    mvr, demo, acc, perf = r["model_validation"], r["demo"], r["accuracy"], r["performance"]
    model_checks = mvr["integrity"] + mvr["dependencies"] + mvr["serialization"] + mvr["runtime"]
    review = [c for group in r["review"].values() for c in group]
    crashes = demo["crashes"] + sum(1 for x in mvr["runtime_rows"] if x["crash"]) \
        + len(perf["in_process"]["crashes"]) + len(perf["http"].get("errors", []))
    items = [
        {"item": "Model file loads correctly", "pass": not failed(model_checks),
         "evidence": f"{len(model_checks) - len(failed(model_checks))}/{len(model_checks)} load, dependency, "
                     f"serialization and runtime checks without failure "
                     f"({sum(c['status'] == 'WARN' for c in model_checks)} warnings)"},
        {"item": "Demo images validated",
         "pass": demo["valid_outputs"] == demo["n"] and demo["stable"] and demo["match_record"]
         and demo["confidence_consistent"] and not demo["crashes"],
         "evidence": f"{demo['valid_outputs']}/{demo['n']} valid outputs, identical over {demo['runs']} runs, equal to "
                     f"the recorded scores; {demo['correct']}/{demo['n']} correct "
                     f"(known miss: {', '.join(demo['wrong']) or 'none'})"},
        {"item": "Accuracy report generated", "pass": acc["requirements_met"] and mae.DEFAULT_REPORT.exists(),
         "evidence": f"TEST {mv.pct(acc['sets']['TEST']['metrics']['accuracy'])}, HOLDOUT "
                     f"{mv.pct(acc['sets']['HOLDOUT']['metrics']['accuracy'])} (> 75% required); "
                     f"{mae.DEFAULT_REPORT.relative_to(ROOT).as_posix()}"},
        {"item": "Performance validated", "pass": perf["targets_met"] and ib.DEFAULT_REPORT.exists(),
         "evidence": f"{sum(c['pass'] for c in perf['targets'])}/{len(perf['targets'])} targets met; "
                     f"{ib.DEFAULT_REPORT.relative_to(ROOT).as_posix()}"},
        {"item": "No critical bugs found", "pass": not failed(review) and crashes == 0,
         "evidence": f"{len(review) - len(failed(review))}/{len(review)} review checks without failure; {crashes} crashes; "
                     f"{sum(i['status'] == 'OPEN' for i in r['issues'])} open issues (non-blocking list in Appendix C)"},
        {"item": "Release report completed", "pass": None, "evidence": ""},  # set once the report is built
    ]
    return items


def recommendation(items: list[dict]) -> dict:
    approved = all(i["pass"] for i in items)
    return {
        "approved": approved,
        "decision": ("APPROVED for release (model v2.0.0) for the live demo, the investor presentation and an "
                     "internal pilot on synthetic or non-decision documents. NOT APPROVED for automated fraud "
                     "decisions on real documents.") if approved else
                    "NOT APPROVED - fix the failing sign-off items and re-run this script.",
        "conditions": [
            "Install with the pinned scikit-learn (requirements.txt) and keep MODEL_PATH at the default "
            "data/models/fraud_model.pkl; the file's SHA-256 must match this report.",
            "Present tamper_risk_score as a risk indicator for human review, never as a probability of fraud; "
            "quote accuracy as measured on synthetic documents.",
            "Keep a person in the loop for every flagged document, and for pages that were re-saved, scanned, "
            "photographed, rotated or cropped (accuracy 50-87% on such pages).",
            "Operator conditions from DEPLOYMENT.md still apply (API_KEY, CORS_ORIGINS, HTTPS).",
            "Rollback: MODEL_PATH=data/models/fraud_model_v1.pkl and restart (verified in Appendix A).",
        ],
        "before_production": [
            "Evaluate on a labelled, consented set of real documents and meet the accuracy target there.",
            "Train and test with degraded pages (re-save, scan, rotation, crop, resolution) or normalise pages first.",
            "Calibrate scores and add a 'needs review' band around the threshold.",
            "Reject or flag non-document images.",
            "Build a new holdout set: TEST and HOLDOUT have been used for v2's evaluation.",
        ],
    }


# ── Run ─────────────────────────────────────────────────────────────────────

def run(runs: int = RUNS, rerun: bool = False) -> dict:
    manifest, images = mst.load_demo()
    model_sha = sha256(DEFAULT_MODEL_PATH)
    integrity, model, meta, api = mv.model_integrity(manifest)
    if api is None:
        raise SystemExit("STOP: the model does not load; see the integrity checks: "
                         + "; ".join(f"{n}: {d}" for n, ok, d in integrity if not ok))
    demo_paths = [mst.DEMO_DIR / r["demo_file"] for r in images]
    demo_vectors = np.array([api.schema.vector(tamper_features.extract_tamper_features(p)) for p in demo_paths])
    with tempfile.TemporaryDirectory() as tmp:
        runtime, runtime_rows = runtime_checks(api, images[0]["demo_file"], Path(tmp))
    model_validation = {
        "integrity": [check(n, ok, d) for n, ok, d in integrity],
        "dependencies": dependency_checks(meta),
        "serialization": serialization_checks(model, meta, demo_vectors),
        "runtime": runtime, "runtime_rows": runtime_rows,
    }
    demo = demo_validation(api, images, runs)

    accuracy = None if rerun else current_results(mae.DEFAULT_JSON, model_sha)
    if accuracy is None:
        print("Running scripts/model_accuracy_eval.py ...")
        accuracy = mae.evaluate(api)
        mae.DEFAULT_REPORT.write_text(mae.build_report(accuracy), encoding="utf-8")
        mae.DEFAULT_JSON.write_text(json.dumps(accuracy, indent=1, default=str) + "\n", encoding="utf-8")
        accuracy = json.loads(mae.DEFAULT_JSON.read_text(encoding="utf-8"))
    performance = None if rerun else current_results(ib.DEFAULT_JSON, model_sha)
    if performance is None:
        print("Running scripts/inference_benchmark.py ...")
        performance = ib.run()
        ib.DEFAULT_REPORT.write_text(ib.build_report(performance), encoding="utf-8")
        ib.DEFAULT_JSON.write_text(json.dumps(performance, indent=1) + "\n", encoding="utf-8")

    split = json.loads(fv.SPLIT_PATH.read_text(encoding="utf-8"))
    test_paths = [fv.DATASET / t["path"] for t in split["test"]]
    review = {
        "Feature extraction pipeline": review_feature_pipeline(api, meta, demo_paths),
        "ELA analysis pipeline": review_ela_pipeline(api, images, manifest),
        "Model prediction logic": review_prediction_logic(api, model, demo_paths + test_paths),
        "Threshold values": review_thresholds(meta, manifest, images[0]["demo_file"], accuracy),
    }
    r = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": mst.environment(),
        "model": {"path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(), "sha256": model_sha,
                  "version": meta.get("model_version"), "schema": meta.get("feature_schema_version"),
                  "trained_at": meta.get("trained_at"), "rf_params": meta.get("rf_params"),
                  "size_bytes": DEFAULT_MODEL_PATH.stat().st_size},
        "threshold": config.TAMPER_THRESHOLD,
        "model_validation": model_validation, "demo": demo, "accuracy": accuracy, "performance": performance,
        "review": review,
        "robustness": current_results(ROBUSTNESS_JSON, model_sha),
        "localization": current_results(LOCALIZATION_JSON, model_sha),
    }
    r["issues"] = issues_found(r)
    r["signoff"] = signoff(r)
    return r


# ── Report ──────────────────────────────────────────────────────────────────

def _checks(checks: list[dict]) -> list[str]:
    return [f"  [{c['status']}] {c['check']}" + (f"\n         {c['detail']}" if c["detail"] else "") for c in checks]


def limitation_lines(r: dict) -> list[str]:
    acc = r["accuracy"]
    by_type: dict[str, list[int]] = {}
    for n in ("TEST", "HOLDOUT"):
        for t, (a, b) in acc["sets"][n]["accuracy_by_doc_type"].items():
            by_type.setdefault(t, [0, 0])
            by_type[t][0] += a
            by_type[t][1] += b
    L = ["  - Synthetic data only: one generator, three templates (payslip, bank statement, ID proof), four",
         "    edit types. Accuracy on real documents has never been measured.",
         "  - Held-out accuracy by document type: " + ", ".join(f"{t} {mv.pct(a / b)} ({a}/{b})" for t, (a, b)
                                                              in sorted(by_type.items())) + "."]
    rb = r.get("robustness")
    if rb:
        s = rb["summary"]
        L += [f"  - Degraded pages (164 held-out documents, {mv.pct(s['original']['accuracy'])} unmodified): "
              f"JPEG q85 {mv.pct(s['jpeg_q85']['accuracy'])},",
              f"    50% downscale {mv.pct(s['downscale_50']['accuracy'])}, 180 deg rotation "
              f"{mv.pct(s['rotate_180']['accuracy'])}, 3 deg skew {mv.pct(s['skew_3deg']['accuracy'])}, "
              f"2 px crop {mv.pct(s['crop_2px']['accuracy'])} (logs/model_robustness_report.txt)."]
    L += ["  - Non-document images get a score and are often reported as tampered (Appendix C).",
          "  - Scores are uncalibrated vote fractions; "
          f"{acc['near_threshold']['n']} held-out documents lie within {acc['near_threshold']['band']} of the threshold.",
          "  - Missed edits: blur on blank paper changes no pixel (undetectable); text overlays whose ink matches",
          "    the page; some colour shifts. Text-replacement edits (demo tampered_payslip_02) were never trained."]
    lo = r.get("localization")
    if lo:
        a = lo["heldout"]["api"]
        L.append(f"  - flagged_regions: a box lands on {a['hit']}/{a['edits']} held-out edits; boxes are anomalies, not "
                 "proof, and unreliable on resampled pages (logs/localization_report.txt).")
    h = r["performance"]["http"]
    if h.get("available"):
        L.append(f"  - Only page 1 of a PDF is analysed. One uvicorn process served {h['concurrent']['throughput_rps']:.1f} "
                 f"requests/s on {r['performance']['environment']['cpus']} CPUs; multi-worker set-ups are unmeasured.")
    L += [
          f"  Full list, edge cases, failure scenarios and future work: {LIMITATIONS.relative_to(ROOT).as_posix()}"
          + ("" if LIMITATIONS.exists() else " (MISSING)")]
    return L


def demo_failure(row: dict) -> str:
    if row["expected"] == "genuine":
        return f"  Failure: {row['image']} ({row['risk']:.2f}) - a genuine page reported as tampered."
    untrained = not set(re.split(r"[:+]", row["edit"])) & GENERATOR_EDITS
    return (f"  Failure: {row['image']} ({row['risk']:.2f}) - edit '{row['edit']}' missed"
            + ("; this edit type is not in model v2's training data." if untrained else "."))


def build_report(r: dict) -> str:
    m, acc, demo, perf, rec = r["model"], r["accuracy"], r["demo"], r["performance"], r["recommendation"]
    env = r["environment"]
    L = [
        "=" * 78,
        "DOCFRAUD - FINAL MODEL RELEASE REPORT",
        f"Model v{m['version']} ({m['schema']}) - Day 10 final QA and sign-off, Founder A",
        "=" * 78,
        f"Generated (UTC): {r['generated']}",
        "Script         : scripts/final_model_validation.py",
        f"Model file     : {m['path']} ({m['size_bytes'] / 2**20:.1f} MB)",
        f"SHA-256        : {m['sha256']}",
        "Model          : RandomForestClassifier, " + ", ".join(
            f"{k}={v}" for k, v in (m["rf_params"] or {}).items()) + f"; trained {m['trained_at'][:19]} UTC",
        f"Decision rule  : tampered if tamper_risk_score >= {r['threshold']} (TAMPER_THRESHOLD)",
        f"Environment    : {env['platform']}, {env['cpus']} CPUs, Python {env['python']}; "
        + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
        "",
        f"DECISION: {'APPROVED' if rec['approved'] else 'NOT APPROVED'} - sign-off "
        f"{sum(bool(i['pass']) for i in r['signoff'])}/{len(r['signoff'])}",
        "  " + rec["decision"],
        "",
        SECTIONS[0],
        "-" * 78,
        "  Held-out evaluation through ModelAPI.predict (positive class = tampered); nothing tuned on these sets.",
        f"  {'Set':<10} {'n':>4} {'Accuracy':>9} {'95% CI':>17} {'Precision':>10} {'Recall':>8} {'F1':>7}  TN/FP/FN/TP",
    ]
    for name, res in (("TEST", acc["sets"]["TEST"]), ("HOLDOUT", acc["sets"]["HOLDOUT"]), ("Combined", acc["combined"])):
        x, ci = res["metrics"], res["ci95"]["accuracy"]
        L.append(f"  {name:<10} {x['n']:>4} {mv.pct(x['accuracy']):>9} {mv.pct(ci[0]) + '-' + mv.pct(ci[1]):>17} "
                 f"{mv.pct(x['precision']):>10} {mv.pct(x['recall']):>8} {x['f1']:>7.4f}  "
                 f"{x['tn']}/{x['fp']}/{x['fn']}/{x['tp']}")
    x = acc["combined"]["metrics"]
    L += ["", "  Combined confusion matrix:", *("    " + line for line in fv.ascii_confusion(x).splitlines()), "",
          "  Release requirements:"]
    L += [f"    [{'PASS' if c['pass'] else 'FAIL'}] {c['check']} - {c['detail']}" for c in acc["requirements"]]
    L += ["", "  Comparison with previous metrics (frozen TEST, 15 images):",
          "    model v1 (Days 5, 6, 8): 60.00% (9/15)  ->  model v2 (Day 8, Day 9, Day 10): "
          f"{mv.pct(acc['sets']['TEST']['metrics']['accuracy'])} ({acc['sets']['TEST']['metrics']['correct']}/15).",
          "    v2 metrics are identical in every evaluation since Day 8 (same confusion matrices)."
          if acc.get("same_as_recorded") else "    v2 metrics were not compared with an earlier record.",
          "  Details: logs/final_accuracy_report.txt"]

    L += ["", SECTIONS[1], "-" * 78,
          f"  {'File Name':<26} {'Actual':<9} {'Prediction':<11} {'Confidence':>10} {'Risk Score':>11}  {'Pass/Fail':<9} Regions"]
    for row in demo["rows"]:
        if "risk" not in row:
            L.append(f"  {row['image']:<26} ERROR {row.get('error')}")
            continue
        regs = ", ".join(f"{g['type']} {g['width']}x{g['height']} at ({g['x']},{g['y']})" for g in row["flagged_regions"])
        L.append(f"  {row['image']:<26} {row['expected']:<9} {row['actual']:<11} {row['confidence']:>10.4f} "
                 f"{row['risk']:>11.4f}  {'PASS' if row['pass'] else 'FAIL':<9} {regs or '-'}")
    h = perf["http"]
    L += ["",
          f"  Correct: {demo['correct']}/{demo['n']}. Five demonstration images do not measure accuracy (Section 1 does).",
          f"  Consistent: {'yes' if demo['match_record'] else 'NO'} - every score equals the score recorded for this "
          "model in data/demo/demo_manifest.json;",
          f"    over HTTP (real server, {ib.HTTP_REPEATS} requests per image) every response "
          f"{'equals' if h.get('available') and not h.get('mismatches') else 'does NOT equal'} the in-process output.",
          f"  Stable: {'yes' if demo['stable'] else 'NO'} - {demo['runs']} runs per image, largest score deviation "
          f"{max(row['max_score_deviation'] or 0 for row in demo['rows']):.4f}.",
          f"  Confidence: {'reasonable' if demo['confidence_consistent'] else 'INCONSISTENT'} - always max(score, 1 - score), "
          f"between 0.5 and 1; below {LOW_CONFIDENCE}: {', '.join(demo['low_confidence']) or 'none'}.",
          *(demo_failure(row) for row in demo["rows"] if "risk" in row and not row["pass"])]

    ip, lo = perf["in_process"], perf["model_load"]
    L += ["", SECTIONS[2], "-" * 78,
          f"  Measured {perf['generated'][:19]} UTC on {perf['environment']['platform']}, "
          f"{perf['environment']['cpus']} CPUs (logs/inference_benchmark_report.txt).",
          f"  Model load (ModelAPI.load_model): average {lo['mean']:.0f} ms (min {lo['min']:.0f}, max {lo['max']:.0f}).",
          f"  In-process inference, {ip['predictions']} predictions ({ip['documents']} documents x {ip['passes']}):",
          f"    average {ip['overall']['mean']:.0f} ms, minimum {ip['overall']['min']:.0f} ms, maximum "
          f"{ip['overall']['max']:.0f} ms (median {ip['overall']['median']:.0f}, p95 {ip['overall']['p95']:.0f}); "
          f"first call after load {ip['cold_first_call_ms']:.0f} ms."]
    for x in perf["large"]:
        if x["stats"]:
            L.append(f"    {x['case']}: average {x['stats']['mean']:.0f} ms (max {x['stats']['max']:.0f})")
    if h.get("available"):
        c = h["concurrent"]
        L += [f"  HTTP /verify, sequential: average {h['sequential_mean_ms']:.0f} ms, minimum {h['sequential_min_ms']:.0f} ms, "
              f"maximum {h['sequential_max_ms']:.0f} ms.",
              f"  HTTP /verify, {c['users']} concurrent users: average {c['mean']:.0f} ms, p95 {c['p95']:.0f} ms, max "
              f"{c['max']:.0f} ms; success {c['success_rate']:.0%}; {c['throughput_rps']:.1f} requests/s."]
    L += ["  Targets:"] + [f"    [{'PASS' if t['pass'] else 'FAIL'}] {t['description']}\n           {t['detail']}"
                           for t in perf["targets"]]

    L += ["", SECTIONS[3], "-" * 78, *limitation_lines(r)]

    L += ["", SECTIONS[4], "-" * 78, "  Sign-off checklist:"]
    L += [f"    [{'x' if i['pass'] else ' '}] {i['item']:<28} {i['evidence']}" for i in r["signoff"]]
    L += ["", f"  Decision: {rec['decision']}", "", "  Conditions of release:"]
    L += [f"    {n}. {c}" for n, c in enumerate(rec["conditions"], 1)]
    L += ["", "  Required before any production use on real documents:"]
    L += [f"    - {c}" for c in rec["before_production"]]

    mvr = r["model_validation"]
    L += ["", "APPENDIX A: MODEL VALIDATION (load, dependencies, serialization, runtime)", "-" * 78,
          "  Model file and loading:", *_checks(mvr["integrity"]), "  Dependencies:", *_checks(mvr["dependencies"]),
          "  Serialization:", *_checks(mvr["serialization"]), "  Runtime:", *_checks(mvr["runtime"]),
          f"  {'Input':<42} {'Bytes':>9}  {'Expected':<9} {'Outcome':<9} Detail"]
    for x in mvr["runtime_rows"]:
        L.append(f"  {x['case'][:42]:<42} {x['bytes']:>9,}  {x['expected']:<9} {x['outcome']:<9} {x['detail']}")
    L += ["", "APPENDIX B: RELEASE REVIEW", "-" * 78]
    for group, checks in r["review"].items():
        L += [f"  {group}:", *_checks(checks)]
    L += ["", "APPENDIX C: ISSUES FOUND DURING THE REVIEW (none blocks the release)", "-" * 78]
    L += [f"  - {i['issue']}\n    severity: {i['severity']}; status: {i['status']}" for i in r["issues"]]
    L.append("")
    return "\n".join(line for line in L if line is not None)


def build_checklist(r: dict) -> str:
    rec = r["recommendation"]
    L = ["=" * 78, "DOCFRAUD - MODEL DEPLOYMENT APPROVAL CHECKLIST (Day 10, Founder A)", "=" * 78,
         f"Generated (UTC): {r['generated']}   Script: scripts/final_model_validation.py",
         f"Model: {r['model']['path']} v{r['model']['version']}, SHA-256 {r['model']['sha256']}", ""]
    L += [f"[{'x' if i['pass'] else ' '}] {i['item']}\n      {i['evidence']}" for i in r["signoff"]]
    L += ["", f"Decision: {'APPROVED' if rec['approved'] else 'NOT APPROVED'}", f"  {rec['decision']}",
          "Conditions:"] + [f"  {n}. {c}" for n, c in enumerate(rec["conditions"], 1)]
    L += ["", "Full report: logs/final_model_release_report.txt", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Final model validation, release review and sign-off")
    ap.add_argument("--runs", type=int, default=RUNS, help=f"repeats per demo image (default {RUNS})")
    ap.add_argument("--rerun", action="store_true", help="re-run the accuracy and benchmark scripts")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--checklist", type=Path, default=DEFAULT_CHECKLIST)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()
    logging.getLogger("backend").setLevel(logging.WARNING)
    r = run(args.runs, args.rerun)
    r["recommendation"] = recommendation([i for i in r["signoff"] if i["pass"] is not None])
    draft = build_report(r)
    item = r["signoff"][-1]
    item["pass"] = all(s in draft for s in SECTIONS)
    item["evidence"] = f"{sum(s in draft for s in SECTIONS)}/{len(SECTIONS)} sections; " \
                       f"{args.report.name}, {args.checklist.name}"
    r["recommendation"] = recommendation(r["signoff"])
    report, checklist = build_report(r), build_checklist(r)
    slim = {k: v for k, v in r.items() if k not in ("accuracy", "performance", "robustness", "localization")}
    slim["accuracy"] = {k: r["accuracy"][k] for k in ("model", "requirements", "requirements_met", "combined")}
    slim["accuracy"]["sets"] = {n: {"metrics": s["metrics"]} for n, s in r["accuracy"]["sets"].items()}
    slim["performance"] = {k: r["performance"][k] for k in ("model", "generated", "targets", "targets_met", "model_load")}
    for path, text in ((args.report, report), (args.checklist, checklist),
                       (args.json, json.dumps(slim, indent=1, default=str) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}\nChecklist written: {args.checklist}\nResults written: {args.json}")
    return 0 if r["recommendation"]["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
