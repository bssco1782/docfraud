"""
scripts/model_stability_test.py
Day 8 (Founder A) model stability and consistency test on the demo images.

Usage:
    python scripts/model_stability_test.py
    python scripts/model_stability_test.py --runs 10 --report logs/model_stability_report.txt

Uses the deployed model (data/models/fraud_model.pkl) and the 5 demo images in
data/demo/. Nothing is trained on, saved to or modified in data/.

Gating checks (decide PASS/FAIL and the exit code):
  1. Repeatability - every demo image is scored --runs times through
     ModelAPI.predict(bytes, filename), the exact call /verify makes. The label
     must never change, and every run's tamper_risk_score and confidence must
     stay within +/-TOLERANCE (0.02 on the 0-1 scale) of that image's mean.
  2. Execution paths - the same (score, confidence) from a file path instead of
     bytes, a fresh ModelAPI (model reloaded from disk), the /verify HTTP route
     (in-process TestClient), concurrent threads on one shared ModelAPI, and a
     separate Python process.
  3. Feature extraction - the model's feature vector (the extractor of its recorded
     feature schema) is bitwise identical on every run, and after a lossless PNG
     re-encode (same pixels, different file bytes).
  4. ELA (core.document_analyzer) - heatmap (SHA-256) and ELA score identical on every
     run; the Day-4 ELA features agree with statistics recomputed from the heatmap.
     Model v2 does not use ELA; the check covers the analyzer module.
  5. Drift - model SHA-256 equals the demo manifest record, and each current score
     is within TOLERANCE of the score recorded there.

Informational analyses (reported, never gating):
  6. Input sensitivity - small edits that keep the document content (JPEG
     re-save, 2-px crop, 98% resize, +/-3% brightness): score change, label flips.
  7. Retraining-seed sensitivity - the deployed configuration refit in memory on
     the model's own training data (v2: rebuilt by scripts/train_tamper_model.py;
     v1: the 75 indexed images) with other random seeds: how much each demo score
     depends on the forest's seed. Also checks that the recorded seed reproduces
     the deployed artifact's outputs exactly.

Exit code: 0 if all gating checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageEnhance
from sklearn.ensemble import RandomForestClassifier

from backend import config
from backend.model_utils import ModelAPI
from core.document_analyzer import (
    ELA_HIGH_THRESHOLD,
    ela_analysis,
    extract_features,
    features_to_vector,
)
from core.feature_schemas import FeatureSchema, get_schema
from core.model_trainer import DEFAULT_MODEL_PATH, IMAGE_SUFFIXES, RF_PARAMS, DocumentFraudDetector

DEMO_DIR = ROOT / "data" / "demo"
MANIFEST_PATH = DEMO_DIR / "demo_manifest.json"
DATASET = ROOT / "data" / "synthetic"
DEFAULT_REPORT = ROOT / "logs" / "model_stability_report.txt"
DEFAULT_JSON = ROOT / "logs" / "model_stability_results.json"

DEFAULT_RUNS = 10
TOLERANCE = 0.02  # +/-2 points on the 0-1 score scale ("within +/-2%")
RETRAIN_SEEDS = tuple(range(10))
WIDE_FOREST_FACTOR = 3  # evidence for a recommendation only: refit with 3x the deployed tree count
LABELS = ("genuine", "tampered")  # 0, 1
CHILD_MARKER = "STABILITY_CHILD_RESULT "
ELA_FEATURES = ("ela_mean", "ela_std", "ela_max", "ela_p95", "ela_high_ratio")


# ── Helpers ─────────────────────────────────────────────────────────────────

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def label_for(score: float) -> str:
    """ModelAPI's rule: tampered if tamper_risk_score >= TAMPER_THRESHOLD (default 0.5)."""
    return "tampered" if score >= config.TAMPER_THRESHOLD else "genuine"


def summarize(values: list[float]) -> dict:
    """Mean, population std, range and largest absolute deviation from the mean."""
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "max_abs_dev": max(abs(v - mean) for v in values),
        "distinct": len(set(values)),
    }


def load_demo() -> tuple[dict, list[dict]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest, manifest["images"]


def environment() -> dict:
    packages = {}
    for dist in ("scikit-learn", "numpy", "opencv-python", "pillow", "fastapi"):
        try:
            packages[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            packages[dist] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "cpus": os.cpu_count(),
        "packages": packages,
    }


def score_pair(out: dict) -> tuple[float, float]:
    return out["tamper_risk_score"], out["confidence"]


# ── 1. Repeatability ────────────────────────────────────────────────────────

def repeatability(api: ModelAPI, images: list[dict], runs: int) -> list[dict]:
    """Score each demo image `runs` times through ModelAPI.predict(bytes), as /verify does."""
    results = []
    for rec in images:
        path = DEMO_DIR / rec["demo_file"]
        data = path.read_bytes()
        rows: list[dict] = []
        for i in range(runs):
            t0 = time.perf_counter()
            try:
                out = api.predict(data, filename=path.name)
            except Exception as exc:  # recorded as a crash, never hidden
                rows.append({"run": i + 1, "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            rows.append({
                "run": i + 1,
                "tamper_risk_score": out["tamper_risk_score"],
                "confidence": out["confidence"],
                "label": label_for(out["tamper_risk_score"]),
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            })
        ok = [r for r in rows if "error" not in r]
        entry = {"demo_file": rec["demo_file"], "expected": rec["actual_label"], "runs": rows,
                 "crashes": len(rows) - len(ok), "pass": False}
        if ok:
            entry["score"] = summarize([r["tamper_risk_score"] for r in ok])
            entry["confidence"] = summarize([r["confidence"] for r in ok])
            entry["labels"] = sorted({r["label"] for r in ok})
            entry["pass"] = (
                entry["crashes"] == 0
                and len(entry["labels"]) == 1
                and entry["score"]["max_abs_dev"] <= TOLERANCE
                and entry["confidence"]["max_abs_dev"] <= TOLERANCE
            )
        results.append(entry)
    return results


# ── 2. Execution paths ──────────────────────────────────────────────────────

def _compare(name: str, got: dict[str, list[tuple[float, float]]], baseline: dict[str, tuple[float, float]],
             detail: str) -> dict:
    mismatches = [f"{f}: {g} != {baseline.get(f)}" for f, outs in got.items() for g in outs if g != baseline.get(f)]
    missing = sorted(set(baseline) - set(got))
    n = sum(len(v) for v in got.values())
    return {"check": name, "calls": n, "pass": not mismatches and not missing and n > 0,
            "detail": detail, "mismatches": mismatches + [f"{f}: no result" for f in missing]}


def _failed(name: str, exc: Exception) -> dict:
    return {"check": name, "calls": 0, "pass": False, "detail": "could not run",
            "mismatches": [f"{type(exc).__name__}: {exc}"[:200]]}


def execution_paths(api: ModelAPI, images: list[dict], baseline: dict[str, tuple[float, float]]) -> list[dict]:
    files = [r["demo_file"] for r in images]
    checks = []

    name = "file path input (ModelAPI.predict(path))"
    try:
        got = {f: [score_pair(api.predict(DEMO_DIR / f))] for f in files}
        checks.append(_compare(name, got, baseline, "1 call per image"))
    except Exception as exc:
        checks.append(_failed(name, exc))

    name = "fresh ModelAPI (model reloaded from disk)"
    try:
        fresh = ModelAPI(DEFAULT_MODEL_PATH)
        fresh.load_model()
        got = {f: [score_pair(fresh.predict((DEMO_DIR / f).read_bytes(), filename=f))] for f in files}
        checks.append(_compare(name, got, baseline, "1 call per image"))
    except Exception as exc:
        checks.append(_failed(name, exc))

    name = "POST /verify route (in-process TestClient)"
    try:
        from fastapi.testclient import TestClient

        from backend.main import app
        from backend.model_utils import model_api as route_model

        got = {}
        # The in-process server logs every request; its output is summarised in the report instead.
        logging.getLogger("backend").setLevel(logging.ERROR)
        with TestClient(app) as client:
            for f in files:
                for _ in range(2):
                    resp = client.post("/verify", headers={"X-API-Key": config.API_KEY},
                                       files={"file": (f, (DEMO_DIR / f).read_bytes(), "image/png")})
                    resp.raise_for_status()
                    got.setdefault(f, []).append(score_pair(resp.json()))
        same_model = Path(route_model.model_path).resolve() == DEFAULT_MODEL_PATH.resolve()
        check = _compare(name, got, baseline, f"2 requests per image, HTTP 200; route model = "
                                              f"{'deployed artifact' if same_model else route_model.model_path}")
        check["pass"] = check["pass"] and same_model
        checks.append(check)
    except Exception as exc:
        checks.append(_failed(name, exc))

    name = "4 concurrent threads, one shared ModelAPI"
    try:
        jobs = [f for f in files for _ in range(2)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outs = list(pool.map(lambda f: score_pair(api.predict((DEMO_DIR / f).read_bytes(), filename=f)), jobs))
        got = {}
        for f, pair in zip(jobs, outs):
            got.setdefault(f, []).append(pair)
        checks.append(_compare(name, got, baseline, f"{len(jobs)} calls, 2 per image"))
    except Exception as exc:
        checks.append(_failed(name, exc))

    name = "separate Python process"
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child-scores"],
                              capture_output=True, text=True, timeout=300, cwd=ROOT)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(CHILD_MARKER)), None)
        if proc.returncode != 0 or line is None:
            raise RuntimeError(f"child exit {proc.returncode}: {proc.stderr.strip()[-200:]}")
        got = {f: [tuple(v)] for f, v in json.loads(line[len(CHILD_MARKER):]).items()}
        checks.append(_compare(name, got, baseline, "new interpreter, model loaded from disk, 1 call per image"))
    except Exception as exc:
        checks.append(_failed(name, exc))
    return checks


def _child_scores() -> int:
    """Entry point for the separate-process check: print (score, confidence) per demo image."""
    api = ModelAPI(DEFAULT_MODEL_PATH)
    api.load_model()
    _, images = load_demo()
    out = {}
    for r in images:  # an image that fails here is already reported by the repeatability check
        try:
            out[r["demo_file"]] = score_pair(api.predict((DEMO_DIR / r["demo_file"]).read_bytes(),
                                                         filename=r["demo_file"]))
        except Exception:
            continue
    print(CHILD_MARKER + json.dumps(out))
    return 0


# ── 3. Feature extraction consistency ───────────────────────────────────────

def feature_consistency(images: list[dict], runs: int, schema: FeatureSchema) -> list[dict]:
    results = []
    for rec in images:
        path = DEMO_DIR / rec["demo_file"]
        vecs = [np.array(schema.vector(schema.extract(path)), dtype=np.float64) for _ in range(runs)]
        identical = all(np.array_equal(vecs[0], v) for v in vecs[1:])
        valid = all(v.shape == (len(schema.names),) and np.isfinite(v).all() for v in vecs)

        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / path.name
            with Image.open(path) as im:
                im.save(copy, format="PNG", compress_level=9)
            with Image.open(path) as a, Image.open(copy) as b:
                same_pixels = np.array_equal(np.asarray(a), np.asarray(b))
            other_bytes = sha256_file(copy) != sha256_file(path)
            re_vec = np.array(schema.vector(schema.extract(copy)), dtype=np.float64)
        changed = [schema.names[i] for i in np.flatnonzero(re_vec != vecs[0])]
        results.append({
            "demo_file": rec["demo_file"],
            "runs": runs,
            "n_features": int(vecs[0].shape[0]),
            "all_finite": valid,
            "identical_across_runs": identical,
            "reencode_same_pixels": same_pixels,
            "reencode_different_bytes": other_bytes,
            "reencode_identical_features": not changed,
            "reencode_changed_features": changed,
            "features": dict(zip(schema.names, vecs[0].tolist())),
            "pass": valid and identical and same_pixels and not changed,
        })
    return results


# ── 4. ELA consistency ──────────────────────────────────────────────────────

def ela_consistency(images: list[dict], runs: int) -> list[dict]:
    """ELA of core.document_analyzer: identical on every run and consistent with the Day-4 ELA features."""
    results = []
    for rec in images:
        path = DEMO_DIR / rec["demo_file"]
        outs = [ela_analysis(path) for _ in range(runs)]
        hashes = {sha256_bytes(h.tobytes()) for h, _ in outs}
        scores = {s for _, s in outs}
        heat = outs[0][0]
        gray = heat.mean(axis=2).astype(np.float32)
        recomputed = {
            "ela_mean": float(gray.mean()),
            "ela_std": float(gray.std()),
            "ela_max": float(gray.max()),
            "ela_p95": float(np.percentile(gray, 95)),
            "ela_high_ratio": float(np.mean(gray > ELA_HIGH_THRESHOLD)),
        }
        feats = extract_features(path)
        disagree = [k for k in ELA_FEATURES if not math.isclose(feats[k], recomputed[k], rel_tol=1e-9, abs_tol=1e-12)]
        with Image.open(path) as im:
            expected_shape = (im.height, im.width, 3)
        results.append({
            "demo_file": rec["demo_file"],
            "runs": runs,
            "heatmap_sha256": sorted(hashes),
            "ela_score": sorted(scores),
            "heatmap_shape_ok": heat.shape == expected_shape and heat.dtype == np.uint8,
            "features_agree_with_heatmap": not disagree,
            "disagreeing_features": disagree,
            "ela_stats": recomputed,
            "pass": len(hashes) == 1 and len(scores) == 1 and not disagree and heat.shape == expected_shape
            and heat.dtype == np.uint8,
        })
    return results


# ── 5. Drift vs Day-7 record ────────────────────────────────────────────────

def drift(images: list[dict], baseline: dict[str, tuple[float, float]], manifest: dict, model_sha: str) -> dict:
    rows = []
    for rec in images:
        current = baseline[rec["demo_file"]][0]
        delta = current - rec["tamper_risk_score"]
        rows.append({
            "demo_file": rec["demo_file"],
            "recorded_score": rec["tamper_risk_score"],
            "recorded_label": rec["predicted_label"],
            "current_score": current,
            "current_label": label_for(current),
            "delta": delta,
            "pass": abs(delta) <= TOLERANCE and label_for(current) == rec["predicted_label"],
        })
    recorded_sha = manifest["model"]["sha256"]
    return {
        "recorded_at": manifest["model"].get("scored_at"),
        "model_sha256_recorded": recorded_sha,
        "model_sha256_current": model_sha,
        "model_unchanged": model_sha == recorded_sha,
        "images": rows,
        "pass": model_sha == recorded_sha and all(r["pass"] for r in rows),
    }


# ── 6. Input sensitivity (informational) ────────────────────────────────────

def _save(img: Image.Image, path: Path, **kw) -> Path:
    img.save(path, **kw)
    return path


PERTURBATIONS = {
    "JPEG re-save q=95": lambda im, d: _save(im, d / "q95.jpg", quality=95),
    "JPEG re-save q=75": lambda im, d: _save(im, d / "q75.jpg", quality=75),
    "2-px border crop": lambda im, d: _save(im.crop((2, 2, im.width - 2, im.height - 2)), d / "crop.png"),
    "resize to 98%": lambda im, d: _save(im.resize((round(im.width * 0.98), round(im.height * 0.98)),
                                                   Image.BICUBIC), d / "resize.png"),
    "brightness +3%": lambda im, d: _save(ImageEnhance.Brightness(im).enhance(1.03), d / "bright.png"),
    "brightness -3%": lambda im, d: _save(ImageEnhance.Brightness(im).enhance(0.97), d / "dark.png"),
}


def largest_shifts(before: np.ndarray, after: np.ndarray, scale: np.ndarray, names: tuple[str, ...],
                   top: int = 3) -> list[list]:
    """Features that moved most, in units of their std over the training rows (descriptive, not causal)."""
    z = np.abs(after - before) / np.where(scale > 0, scale, np.inf)
    return [[names[i], round(float(z[i]), 2)] for i in np.argsort(-z)[:top] if z[i] > 0]


def input_sensitivity(api: ModelAPI, images: list[dict], baseline: dict[str, tuple[float, float]],
                      demo_vectors: dict[str, dict], train_std: np.ndarray, schema: FeatureSchema) -> list[dict]:
    """Score content-preserving variants of each demo image (temp files only)."""
    results = []
    for rec in images:
        base_score = baseline[rec["demo_file"]][0]
        base_label = label_for(base_score)
        base_vec = np.array([demo_vectors[rec["demo_file"]][k] for k in schema.names])
        variants = []
        with tempfile.TemporaryDirectory() as tmp, Image.open(DEMO_DIR / rec["demo_file"]) as src:
            src = src.convert("RGB")
            for name, make in PERTURBATIONS.items():
                path = make(src, Path(tmp))
                try:
                    score = api.predict(path)["tamper_risk_score"]
                    vec = np.array(schema.vector(schema.extract(path)), dtype=np.float64)
                except Exception as exc:
                    variants.append({"variant": name, "error": f"{type(exc).__name__}: {exc}"[:200]})
                    continue
                variants.append({"variant": name, "score": score, "delta": score - base_score,
                                 "label": label_for(score), "flip": label_for(score) != base_label,
                                 "largest_feature_shifts": largest_shifts(base_vec, vec, train_std, schema.names)})
        scored = [v for v in variants if "score" in v]
        results.append({
            "demo_file": rec["demo_file"],
            "expected": rec["actual_label"],
            "baseline_score": base_score,
            "baseline_label": base_label,
            "variants": variants,
            "max_abs_delta": max((abs(v["delta"]) for v in scored), default=0.0),
            "flips": [v["variant"] for v in scored if v["flip"]],
            "errors": [v["variant"] for v in variants if "error" in v],
        })
    return results


# ── 7. Retraining-seed sensitivity (informational) ──────────────────────────

def dataset_features() -> dict[str, np.ndarray]:
    """22-feature vectors of the 75 indexed documents, in DocumentFraudDetector.load_dataset() order."""
    out: dict[str, np.ndarray] = {}
    for label in LABELS:
        for p in sorted((DATASET / label).iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                out[f"{label}/{p.name}"] = np.array(features_to_vector(extract_features(p)), dtype=np.float64)
    return out


def dataset_fingerprint() -> str:
    """Same fingerprint as DocumentFraudDetector.load_dataset() (stored as metadata['dataset_sha256'])."""
    digests = {sha256_file(p): f"{lab}/{p.name}" for lab in LABELS for p in sorted((DATASET / lab).iterdir())
               if p.suffix.lower() in IMAGE_SUFFIXES}
    return hashlib.sha256("\n".join(sorted(f"{n}:{d}" for d, n in digests.items())).encode()).hexdigest()


def fit_forest(X: np.ndarray, y: np.ndarray, seed: int, n_estimators: int | None = None) -> RandomForestClassifier:
    """Model v1 configuration (core.model_trainer.RF_PARAMS, n_jobs=-1 as in DocumentFraudDetector.train)."""
    params = {**RF_PARAMS, **({"n_estimators": n_estimators} if n_estimators else {})}
    return RandomForestClassifier(**params, random_state=seed, n_jobs=-1).fit(X, y)


def training_matrix(feats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack(list(feats.values()))
    y = np.array([LABELS.index(k.split("/")[0]) for k in feats], dtype=np.int32)
    return X, y


def _seed_spread(models: dict[int, RandomForestClassifier], demo_X: np.ndarray, images: list[dict],
                 deployed_scores: np.ndarray) -> list[dict]:
    scores = {seed: m.predict_proba(demo_X)[:, 1] for seed, m in models.items()}
    rows = []
    for i, rec in enumerate(images):
        per_seed = [float(s[i]) for s in scores.values()]
        dep_label = label_for(float(deployed_scores[i]))
        rows.append({
            "demo_file": rec["demo_file"],
            "expected": rec["actual_label"],
            "deployed_score": float(deployed_scores[i]),
            "per_seed": dict(zip([str(k) for k in scores], per_seed)),
            **{k: v for k, v in summarize(per_seed).items() if k != "distinct"},
            "label_flips_vs_deployed": sum(label_for(s) != dep_label for s in per_seed),
            "correct_seeds": sum(label_for(s) == rec["actual_label"] for s in per_seed),
        })
    return rows


def training_data(meta: dict) -> dict:
    """The deployed model's training matrix, a fit function with its parameters, and a data check.

    v2 (metadata records training_files): rebuilt exactly as scripts/train_tamper_model.py builds it.
    v1: the 75 indexed images with the Day-4 features.
    """
    if meta.get("training_files"):
        from scripts import train_tamper_model as ttm

        docs = ttm.training_documents()
        data = ttm.build_matrix(docs, verbose=False)
        return {"X": data["X"], "y": data["y"], "fit": ttm.fit, "rows": len(data["y"]), "documents": len(docs),
                "description": f"{len(docs)} training documents (60 TRAIN + 150 DEV) + processed copies = "
                               f"{len(data['y'])} rows, rebuilt by scripts/train_tamper_model.py",
                "data_matches_metadata": {d["sha256"] for d in docs} == set(meta["training_files"].values())}
    X, y = training_matrix(dataset_features())
    return {"X": X, "y": y, "fit": fit_forest, "rows": len(y), "documents": len(y),
            "description": f"the {len(y)} indexed images (v1 training data)",
            "data_matches_metadata": dataset_fingerprint() == meta.get("dataset_sha256")}


def retrain_seed_sensitivity(deployed: RandomForestClassifier, meta: dict, images: list[dict],
                             demo_vectors: dict[str, dict], train: dict, schema: FeatureSchema) -> dict:
    X, y, fit = train["X"], train["y"], train["fit"]
    demo_X = np.array([[demo_vectors[r["demo_file"]][k] for k in schema.names] for r in images])
    deployed_demo = deployed.predict_proba(demo_X)
    recorded_seed = int(meta.get("random_state", 42))
    trees = int(meta.get("n_estimators") or deployed.n_estimators)

    replica = fit(X, y, recorded_seed)
    reproduces = (np.array_equal(replica.predict_proba(X), deployed.predict_proba(X))
                  and np.array_equal(replica.predict_proba(demo_X), deployed_demo))

    base = _seed_spread({s: fit(X, y, s) for s in RETRAIN_SEEDS}, demo_X, images, deployed_demo[:, 1])
    wide = _seed_spread({s: fit(X, y, s, WIDE_FOREST_FACTOR * trees) for s in RETRAIN_SEEDS}, demo_X, images,
                        deployed_demo[:, 1])
    return {
        "training_data": train["description"],
        "n_training_images": train["documents"],
        "n_training_rows": train["rows"],
        "fingerprint_matches_model_metadata": bool(train["data_matches_metadata"]),
        "recorded_seed": recorded_seed,
        "recorded_seed_reproduces_deployed": bool(reproduces),
        "seeds": list(RETRAIN_SEEDS),
        "deployed_trees": trees,
        "per_image": base,
        "wide_forest_trees": WIDE_FOREST_FACTOR * trees,
        "wide_forest_per_image": wide,
    }


# ── Report ──────────────────────────────────────────────────────────────────

def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def build_report(r: dict) -> str:
    env, model = r["environment"], r["model"]
    runs = r["runs"]
    rep, paths, feats, ela, dr = r["repeatability"], r["execution_paths"], r["features"], r["ela"], r["drift"]
    sens, seeds = r["input_sensitivity"], r["retrain_seeds"]
    n_runs = sum(len(e["runs"]) for e in rep)
    n_identical = sum(e.get("score", {}).get("distinct") == 1 and e.get("confidence", {}).get("distinct") == 1
                      for e in rep)
    worst_dev = max((e.get("score", {}).get("max_abs_dev", 0.0) for e in rep), default=0.0)
    flips = [(s["demo_file"], v) for s in sens for v in s["flips"]]
    n_variants = sum(len(s["variants"]) for s in sens)

    L = [
        "--------------------------------",
        "DOCFRAUD MODEL STABILITY REPORT",
        "--------------------------------",
        f"Generated (UTC): {r['generated']}",
        f"Script: scripts/model_stability_test.py --runs {runs}",
        f"Environment: Python {env['python']}, {env['platform']}, {env['cpus']} CPUs; "
        + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
        "",
        "Model:",
        f"  {model['path']} (loaded by the API)",
        f"  SHA-256 {model['sha256']}",
        f"  version {model['version']}, feature schema {model['schema']}, {model['n_features']} features",
        f"  RandomForest {model['rf_params']}, trained on {model['n_training_samples']} documents",
        f"  Decision rule: tampered if tamper_risk_score >= {config.TAMPER_THRESHOLD} (TAMPER_THRESHOLD)",
        "",
        f"Images: {len(rep)} demo images in data/demo/ (demo_manifest.json). Demonstration images only,",
        "  not a validation set.",
        f"Tolerance: every run within +/-{TOLERANCE:.2f} of the image's mean (2 points on the 0-1 score",
        "  scale), for both tamper_risk_score and confidence; the label must never change.",
        "",
        "--------------------------------",
        "SUMMARY",
        "--------------------------------",
        f"[{_mark(all(e['pass'] for e in rep))}] 1. Repeatability: {n_runs} calls ({runs} per image) through "
        f"ModelAPI.predict(bytes);",
        f"         {n_identical}/{len(rep)} images returned the identical score and confidence on every run; "
        f"largest deviation {worst_dev:.4f}",
        f"[{_mark(all(c['pass'] for c in paths))}] 2. Execution paths: {sum(c['calls'] for c in paths)} extra calls "
        f"over {len(paths)} paths; {sum(c['pass'] for c in paths)}/{len(paths)} paths equal to run 1",
        f"[{_mark(all(f['pass'] for f in feats))}] 3. Feature extraction: {model['n_features']} features "
        f"({model['schema']}) bitwise identical on {runs} runs and after lossless re-encode",
        f"[{_mark(all(e['pass'] for e in ela))}] 4. ELA (analyzer): heatmap and score identical on {runs} runs; "
        "Day-4 ELA features agree with heatmap",
        f"[{_mark(dr['pass'])}] 5. Drift: model SHA-256 {'matches' if dr['model_unchanged'] else 'DIFFERS FROM'} the "
        f"demo manifest record; max score change {max((abs(x['delta']) for x in dr['images']), default=0.0):.4f}"
        + (f"; not scored: {', '.join(dr['unscored'])}" if dr.get("unscored") else ""),
        "",
        f"STABILITY (gating checks 1-5): {_mark(r['pass'])}",
        "",
        "Informational (not gating):",
        f"  6. Input sensitivity: {len(flips)} label flip(s) in {n_variants} content-preserving variants"
        + (f" ({', '.join(sorted({f for f, _ in flips}))})" if flips else ""),
        f"  7. Retraining seeds: recorded seed {seeds['recorded_seed']} reproduces the deployed artifact: "
        f"{'YES' if seeds['recorded_seed_reproduces_deployed'] else 'NO'}; demo labels that change with the seed: "
        f"{sum(1 for p in seeds['per_image'] if p['label_flips_vs_deployed'])}/{len(seeds['per_image'])}",
        "",
        "--------------------------------",
        f"1. REPEATABILITY ({runs} runs per image, ModelAPI.predict(bytes, filename) = the /verify call)",
        "--------------------------------",
        f"{'Image':<26} {'Label (all runs)':<17} {'Score mean':>10} {'std':>7} {'min':>6} {'max':>6} "
        f"{'max dev':>8} {'Conf mean':>9} {'std':>7} {'ms avg':>7}  Result",
        "-" * 122,
    ]
    for e in rep:
        if "score" not in e:
            L.append(f"{e['demo_file']:<26} no successful run ({e['crashes']} crashes)  FAIL")
            continue
        s, c = e["score"], e["confidence"]
        ms = statistics.fmean(x["ms"] for x in e["runs"] if "ms" in x)
        L.append(f"{e['demo_file']:<26} {'/'.join(e['labels']):<17} {s['mean']:>10.4f} {s['std']:>7.4f} "
                 f"{s['min']:>6.2f} {s['max']:>6.2f} {s['max_abs_dev']:>8.4f} {c['mean']:>9.4f} {c['std']:>7.4f} "
                 f"{ms:>7.0f}  {_mark(e['pass'])}")
    L += ["", "Per-run tamper_risk_score / confidence:"]
    for e in rep:
        vals = " ".join(f"{x['tamper_risk_score']:.2f}/{x['confidence']:.2f}" if "error" not in x else "ERROR"
                        for x in e["runs"])
        L.append(f"  {e['demo_file']:<26} {vals}")
    L.append(f"Crashes: {sum(e['crashes'] for e in rep)}")

    L += ["", "--------------------------------", "2. EXECUTION PATHS (each result compared with run 1)",
          "--------------------------------"]
    for c in paths:
        L.append(f"[{_mark(c['pass'])}] {c['check']:<44} {c['calls']:>3} calls - {c['detail']}")
        L += [f"         mismatch: {m}" for m in c["mismatches"]]

    L += ["", "--------------------------------", "3. FEATURE EXTRACTION CONSISTENCY",
          "--------------------------------",
          f"{'Image':<26} {'Features':>8} {'Finite':>7} {'Identical x' + str(runs):>13} {'Re-encode':>10}  Result",
          "-" * 80]
    for f in feats:
        reenc = "same" if f["reencode_identical_features"] else "CHANGED"
        L.append(f"{f['demo_file']:<26} {f['n_features']:>8} {'yes' if f['all_finite'] else 'NO':>7} "
                 f"{'yes' if f['identical_across_runs'] else 'NO':>13} {reenc:>10}  {_mark(f['pass'])}")
        if f["reencode_changed_features"]:
            L.append(f"    changed after re-encode: {', '.join(f['reencode_changed_features'])}")
    L.append("Re-encode = the image saved again as PNG (compress_level=9): identical pixels, "
             f"different file bytes ({sum(f['reencode_different_bytes'] for f in feats)}/{len(feats)}).")

    L += ["", "--------------------------------", "4. ELA CONSISTENCY", "--------------------------------",
          f"{'Image':<26} {'ELA score':>9} {'ela_mean':>9} {'ela_p95':>8} {'high ratio':>10} {'Heatmap SHA-256':<18} "
          "Runs identical  Result",
          "-" * 112]
    for e in ela:
        st = e["ela_stats"]
        L.append(f"{e['demo_file']:<26} {e['ela_score'][0]:>9.3f} {st['ela_mean']:>9.3f} {st['ela_p95']:>8.1f} "
                 f"{st['ela_high_ratio']:>10.4f} {e['heatmap_sha256'][0][:16]:<18} "
                 f"{'yes' if len(e['heatmap_sha256']) == 1 and len(e['ela_score']) == 1 else 'NO':<15} "
                 f"{_mark(e['pass'])}")
    L.append("ELA features in the 22-feature vector recomputed from the heatmap: "
             f"{'agree' if all(e['features_agree_with_heatmap'] for e in ela) else 'DISAGREE'} for all images.")

    L += ["", "--------------------------------", "5. PREDICTION DRIFT (vs demo manifest record)",
          "--------------------------------",
          f"Recorded: {dr['recorded_at']}; model SHA-256 recorded {dr['model_sha256_recorded'][:16]}..., "
          f"current {dr['model_sha256_current'][:16]}... -> {'unchanged' if dr['model_unchanged'] else 'CHANGED'}",
          f"{'Image':<26} {'Recorded':>8} {'Current':>8} {'Delta':>8} {'Label recorded -> now':<24} Result",
          "-" * 88]
    for x in dr["images"]:
        L.append(f"{x['demo_file']:<26} {x['recorded_score']:>8.2f} {x['current_score']:>8.4f} {x['delta']:>+8.4f} "
                 f"{x['recorded_label'] + ' -> ' + x['current_label']:<24} {_mark(x['pass'])}")

    L += ["", "--------------------------------", "6. INPUT SENSITIVITY (informational)",
          "--------------------------------",
          "Content-preserving variants of each demo image, scored through ModelAPI.predict (temp files,",
          "deleted after the run). 'Largest feature shifts' = features that moved most, in units of their",
          "std over the model's training rows (descriptive; the forest combines many features).",
          f"{'Image / variant':<28} {'Score':>5} {'Change':>7}  {'Label':<9} {'Flip':<4}  Largest feature shifts",
          "-" * 118]
    for s in sens:
        L.append(f"{s['demo_file']:<28} {s['baseline_score']:>5.2f} {'':>7}  {s['baseline_label']:<9} "
                 f"{'':<4}  (unmodified; expected {s['expected']})")
        for v in s["variants"]:
            if "error" in v:
                L.append(f"  {v['variant']:<26} ERROR {v['error']}")
                continue
            shifts = ", ".join(f"{n} {z:.1f}" for n, z in v["largest_feature_shifts"])
            L.append(f"  {v['variant']:<26} {v['score']:>5.2f} {v['delta']:>+7.2f}  {v['label']:<9} "
                     f"{'YES' if v['flip'] else '':<4}  {shifts}")
    per_variant = {k: sum(1 for s in sens for v in s["variants"] if v["variant"] == k and v.get("flip"))
                   for k in PERTURBATIONS}
    L += ["", "Label flips per variant (out of " + f"{len(sens)} images): "
          + ", ".join(f"{k} {n}" for k, n in per_variant.items())]

    L += ["", "--------------------------------", "7. RETRAINING-SEED SENSITIVITY (informational)",
          "--------------------------------",
          f"The deployed configuration refit in memory on {seeds['training_data']}; training data "
          f"{'matches' if seeds['fingerprint_matches_model_metadata'] else 'DIFFERS FROM'} the model metadata; "
          "nothing saved.",
          f"Recorded seed {seeds['recorded_seed']} reproduces the deployed artifact's probabilities exactly "
          f"({seeds['n_training_rows']} training rows + 5 demo images): "
          f"{'YES' if seeds['recorded_seed_reproduces_deployed'] else 'NO'}",
          f"Seeds {seeds['seeds'][0]}-{seeds['seeds'][-1]}, {seeds['deployed_trees']} trees (deployed size):",
          f"{'Image':<26} {'Expected':<9} {'Deployed':>8} {'Seed min':>8} {'max':>6} {'std':>6} "
          f"{'Label changes':>13} {'Correct seeds':>13}",
          "-" * 98]
    for p in seeds["per_image"]:
        L.append(f"{p['demo_file']:<26} {p['expected']:<9} {p['deployed_score']:>8.2f} {p['min']:>8.2f} "
                 f"{p['max']:>6.2f} {p['std']:>6.3f} {p['label_flips_vs_deployed']:>9}/{len(seeds['seeds']):<3} "
                 f"{p['correct_seeds']:>9}/{len(seeds['seeds'])}")
    L += ["", f"Same seeds with {seeds['wide_forest_trees']} trees (NOT deployed; shows how much a larger "
              "forest reduces seed noise):",
          f"{'Image':<26} {'Seed min':>8} {'max':>6} {'std':>6} {'Label changes':>13}", "-" * 64]
    for p in seeds["wide_forest_per_image"]:
        L.append(f"{p['demo_file']:<26} {p['min']:>8.2f} {p['max']:>6.2f} {p['std']:>6.3f} "
                 f"{p['label_flips_vs_deployed']:>9}/{len(seeds['seeds'])}")

    L += ["", "--------------------------------", "INCONSISTENT OR UNSTABLE OUTPUTS", "--------------------------------"]
    L += r["findings"] or ["None."]
    L += ["", "--------------------------------", "SUGGESTED IMPROVEMENTS", "--------------------------------"]
    L += r["recommendations"]
    L.append("")
    return "\n".join(L)


def findings_and_recommendations(r: dict) -> tuple[list[str], list[str]]:
    """Derive the findings text from the measured results (nothing here is hard-coded)."""
    findings: list[str] = []
    for e in r["repeatability"]:
        if not e["pass"]:
            findings.append(f"- Repeatability FAIL for {e['demo_file']}: labels {e.get('labels')}, "
                            f"score dev {e.get('score', {}).get('max_abs_dev')}, crashes {e['crashes']}.")
    for c in r["execution_paths"]:
        if not c["pass"]:
            findings.append(f"- Execution path '{c['check']}' differs: {'; '.join(c['mismatches'][:3])}")
    for f in r["features"]:
        if not f["pass"]:
            findings.append(f"- Feature extraction inconsistent for {f['demo_file']}: "
                            f"changed {f['reencode_changed_features']}")
    for e in r["ela"]:
        if not e["pass"]:
            findings.append(f"- ELA inconsistent for {e['demo_file']}")
    if not r["drift"]["pass"]:
        findings.append("- Drift vs the demo manifest record: " + ", ".join(
            f"{x['demo_file']} {x['delta']:+.4f}" for x in r["drift"]["images"] if not x["pass"]))

    thr = config.TAMPER_THRESHOLD
    near = [(e["demo_file"], e["score"]["mean"]) for e in r["repeatability"]
            if "score" in e and abs(e["score"]["mean"] - thr) <= 0.10]
    if near:
        findings.append("- Deterministic but close to the threshold (|score - " + f"{thr}| <= 0.10): "
                        + ", ".join(f"{f} {s:.2f}" for f, s in near)
                        + ". Repeated calls agree, but small input or retraining changes can move them across it.")
    sens = r["input_sensitivity"]
    for s in sens:
        if s["flips"]:
            findings.append(f"- {s['demo_file']}: label flips under {', '.join(s['flips'])} "
                            f"(max score change {s['max_abs_delta']:.2f}).")
        if s["errors"]:
            findings.append(f"- {s['demo_file']}: variants failed to score: {', '.join(s['errors'])}.")
    flipped = [v for s in sens for v in s["variants"] if v.get("flip")]
    top: dict[str, int] = {}
    for v in flipped:
        for name, _ in v["largest_feature_shifts"]:
            top[name] = top.get(name, 0) + 1
    common = sorted(top.items(), key=lambda kv: -kv[1])[:4]
    if flipped:
        n_variants = sum(len(s["variants"]) for s in sens)
        findings.append(f"- Overall {len(flipped)}/{n_variants} content-preserving variants change the label. "
                        "Features most often among the largest shifts in those cases: "
                        + ", ".join(f"{n} ({c}x)" for n, c in common) + ".")
    seeds = r["retrain_seeds"]
    for p in seeds["per_image"]:
        if p["label_flips_vs_deployed"]:
            findings.append(f"- {p['demo_file']}: a refit with another seed changes its label in "
                            f"{p['label_flips_vs_deployed']}/{len(seeds['seeds'])} seeds "
                            f"(scores {p['min']:.2f}-{p['max']:.2f}; deployed {p['deployed_score']:.2f}).")
    if not seeds["recorded_seed_reproduces_deployed"]:
        findings.append("- The recorded configuration does NOT reproduce the deployed artifact from the current "
                        "data and code.")

    wide = {p["demo_file"]: p for p in seeds["wide_forest_per_image"]}
    base_std = statistics.fmean(p["std"] for p in seeds["per_image"])
    wide_std = statistics.fmean(p["std"] for p in wide.values())
    recs = [
        "- Keep the inference path as it is: it is deterministic (identical output on every call, path, thread "
        "and process). Record the SHA-256 of fraud_model.pkl in every release and re-run this script after any "
        "dependency upgrade; scikit-learn / numpy / OpenCV / Pillow versions are part of the model.",
        "- Pin the package versions listed above (requirements.txt is unpinned). A different library version is "
        "the most likely source of future score drift, and pickle compatibility is not guaranteed across "
        "scikit-learn versions.",
        f"- Retraining noise: with {seeds['deployed_trees']} trees the demo scores move by up to "
        f"{max(p['max'] - p['min'] for p in seeds['per_image']):.2f} between seeds (mean std {base_std:.3f}); "
        f"{seeds['wide_forest_trees']} trees gives mean std {wide_std:.3f}. When the model is next retrained, "
        f"consider n_estimators={seeds['wide_forest_trees']} (choose it by train-only cross-validation, not on "
        "the frozen test set or the demo images).",
        "- Report scores near the threshold as 'needs review' instead of a hard genuine/tampered verdict "
        "(e.g. a band around TAMPER_THRESHOLD, set from validation data), so near-threshold images are not "
        "presented as confident decisions.",
        "- Input sensitivity is a model-quality problem, not a determinism problem. "
        + (f"The label flips above come mostly with large shifts in {', '.join(n for n, _ in common[:3])}, "
           "which change when a page is re-saved, resized or re-exposed, so the model partly measures the file's "
           "processing history rather than the edit. " if common else "")
        + "Measure local anomalies relative to the page's own compression level (estimated JPEG quality, noise "
        "floor) and widen the processing applied to training copies.",
        "- Stability says nothing about accuracy: see logs/model_validation_report.txt (held-out accuracy and "
        "demo results).",
    ]
    return findings, recs


# ── Main ────────────────────────────────────────────────────────────────────

def run(runs: int) -> dict:
    manifest, images = load_demo()
    model_sha = sha256_file(DEFAULT_MODEL_PATH)
    deployed, meta = DocumentFraudDetector.load(DEFAULT_MODEL_PATH)
    schema = get_schema(meta.get("feature_schema_version"))

    api = ModelAPI(DEFAULT_MODEL_PATH)
    api.load_model()

    print(f"1/7 Repeatability ({runs} runs x {len(images)} images)...")
    rep = repeatability(api, images, runs)
    baseline = {e["demo_file"]: (e["runs"][0]["tamper_risk_score"], e["runs"][0]["confidence"])
                for e in rep if e["runs"] and "error" not in e["runs"][0]}
    # An image whose first run failed is already a repeatability FAIL; later stages use the scored ones.
    scored = [r for r in images if r["demo_file"] in baseline]
    print("2/7 Execution paths...")
    paths = execution_paths(api, scored, baseline)
    print("3/7 Feature extraction consistency...")
    feats = feature_consistency(scored, runs, schema)
    print("4/7 ELA consistency...")
    ela = ela_consistency(scored, runs)
    print("5/7 Drift...")
    dr = drift(scored, baseline, manifest, model_sha)
    dr["unscored"] = [r["demo_file"] for r in images if r["demo_file"] not in baseline]
    dr["pass"] = dr["pass"] and not dr["unscored"]
    print("Rebuilding the model's training data (used by 6 and 7)...")
    train = training_data(meta)
    demo_vectors = {f["demo_file"]: f["features"] for f in feats}
    print("6/7 Input sensitivity...")
    sens = input_sensitivity(api, scored, baseline, demo_vectors, train["X"].std(axis=0), schema)
    print("7/7 Retraining-seed sensitivity...")
    seeds = retrain_seed_sensitivity(deployed, meta, scored, demo_vectors, train, schema)

    ok = (all(e["pass"] for e in rep) and all(c["pass"] for c in paths) and all(f["pass"] for f in feats)
          and all(e["pass"] for e in ela) and dr["pass"])
    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
        "tolerance": TOLERANCE,
        "threshold": config.TAMPER_THRESHOLD,
        "environment": environment(),
        "model": {
            "path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(),
            "sha256": model_sha,
            "version": meta.get("model_version"),
            "schema": meta.get("feature_schema_version"),
            "n_features": int(deployed.n_features_in_),
            "rf_params": meta.get("rf_params"),
            "n_training_samples": meta.get("n_training_samples"),
        },
        "repeatability": rep,
        "execution_paths": paths,
        "features": feats,
        "ela": ela,
        "drift": dr,
        "input_sensitivity": sens,
        "retrain_seeds": seeds,
        "pass": ok,
    }
    result["findings"], result["recommendations"] = findings_and_recommendations(result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="Day 8 model stability test on the demo images")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="predictions per image (default 10)")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON, help="machine-readable results")
    ap.add_argument("--child-scores", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child_scores:
        return _child_scores()
    if args.runs < 2:
        ap.error("--runs must be at least 2")

    result = run(args.runs)
    report = build_report(result)
    for path, text in ((args.report, report), (args.json, json.dumps(result, indent=1, default=str) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}\nResults written: {args.json}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
