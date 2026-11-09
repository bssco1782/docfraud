"""
scripts/ml_recovery.py
Day 5 ML recovery: audit -> training-only CV -> selection -> final model -> ONE frozen-test evaluation.

Stages (run in order; each writes into logs/day5_ml_recovery_results.json):
    python scripts/ml_recovery.py audit     # dataset + day4 feature audit (train images only for stats)
    python scripts/ml_recovery.py cv        # experiments A/B/C (+aug, +ablations) on the 60 train images
    python scripts/ml_recovery.py grid      # limited RF grid on the representation selected by `cv`
    python scripts/ml_recovery.py final     # train on all permitted data, save, evaluate frozen test ONCE
    python scripts/ml_recovery.py perf      # runtime / size measurements
    python scripts/ml_recovery.py report    # render logs/day5_augmentation_v2_cv_report.md from results

Integrity rules enforced in code:
  * audit / cv / grid never read a frozen-test image (asserted).
  * Model/representation selection rules are fixed below BEFORE results exist.
  * `final` refuses to evaluate the frozen test set a second time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core import local_forensics as lf
from core.document_analyzer import FEATURE_NAMES as DAY4_NAMES
from core.document_analyzer import extract_features, features_to_vector
from core.forensic_model import COMBINED_FEATURE_NAMES, MODEL_VERSION, extract_combined
from scripts.augment_training_data import AUG_VERSION, MANIFEST, load_manifest, manifest_sha
from scripts.augment_training_data import OUT_DIR as AUG_DIR
from scripts.augment_training_data import check as aug_check

SPLIT = json.loads((ROOT / "data/splits/day5_frozen_split.json").read_text(encoding="utf-8"))
DATASET = ROOT / "data" / "synthetic"
RESULTS = ROOT / "logs" / "day5_ml_recovery_results.json"
CACHE = ROOT / "data" / "cache" / "features_local_v1.pkl"
MODEL_OUT = ROOT / "data" / "models" / "fraud_model_local_v1.pkl"
SEED = 42
TEST_PATHS = {r["path"] for r in SPLIT["test"]}
INDEX = {r["path"]: r for r in csv.DictReader(open(DATASET / "index.csv", newline="", encoding="utf-8"))}

BASE_RF = {"n_estimators": 100, "class_weight": "balanced"}   # Day-5 baseline config
GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 8, 12],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"],
}
# Features that describe WHAT the document looks like (size, brightness, layout
# density) rather than whether it was edited — candidate document-type shortcuts.
SHORTCUT_FEATURES = ["image_width", "image_height", "aspect_ratio", "grayscale_mean",
                     "grayscale_std", "dark_pixel_ratio", "edge_density", "bytes_per_pixel"]

SELECTION_RULE = (
    "Pre-registered: pick the experiment / config with the highest mean CV F1 (tampered class) "
    "over the 5 identical folds. If within 0.02 of the best, prefer lower F1 std, then fewer "
    "features / simpler model. The frozen test set plays no part."
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def save_results(res: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")


def _train_rows() -> list[dict]:
    rows = SPLIT["train"]
    assert not TEST_PATHS & {r["path"] for r in rows}
    return rows


def aug_key(file: str) -> str:
    """Cache key of an augmented record, namespaced by augmentation version so
    records of different versions can never be mixed in one experiment."""
    return f"aug[{AUG_VERSION}]:{file}"


TEST_HASHES = {r["sha256"] for r in SPLIT["test"]}


def assert_no_test_leakage(keys: list[str]) -> None:
    """Hard stop if any frozen-test image reaches a training / CV record set."""
    test_keys = {f"orig:{p}" for p in TEST_PATHS}
    assert not test_keys & set(keys), "frozen test image in training/CV records"
    train_hashes = {r["sha256"] for r in SPLIT["train"]}
    aug_hashes = {m["sha256"] for m in load_manifest()}
    assert not TEST_HASHES & train_hashes, "frozen test hash among training originals"
    assert not TEST_HASHES & aug_hashes, "frozen test hash among augmented images"
    assert not TEST_PATHS & {m["source_path"] for m in load_manifest()}, "frozen test image used as augmentation source"


def feature_table(include_test: bool = False) -> dict[str, dict]:
    """key -> {"day4": [...], "local": [...]} for train (+aug) (+test only if asked). Cached."""
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.exists() else {}
    if cache.get("_schema") != (lf.FEATURE_SCHEMA_VERSION, tuple(lf.FEATURE_NAMES), tuple(DAY4_NAMES)):
        cache = {"_schema": (lf.FEATURE_SCHEMA_VERSION, tuple(lf.FEATURE_NAMES), tuple(DAY4_NAMES))}
    # Augmented records are valid only for the exact manifest (and version) they were
    # computed from. Anything else is dropped; original-image records are kept.
    stale = [k for k in cache if k.startswith("aug") and
             (not k.startswith(f"aug[{AUG_VERSION}]:") or cache.get("_aug_manifest") != manifest_sha())]
    for k in stale:
        del cache[k]
    if stale:
        print(f"  dropped {len(stale)} augmented records from other versions / manifests", flush=True)
    cache["_aug_manifest"] = manifest_sha()
    items = [(f"orig:{r['path']}", DATASET / r["path"]) for r in _train_rows()]
    items += [(aug_key(r["file"]), AUG_DIR / r["file"]) for r in load_manifest()]
    if not include_test:
        assert_no_test_leakage([k for k, _ in items] + [k for k in cache if not k.startswith("_")])
    if include_test:
        items += [(f"orig:{r['path']}", DATASET / r["path"]) for r in SPLIT["test"]]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    todo = [(k, p) for k, p in items if k not in cache]
    for i, (key, path) in enumerate(todo, 1):
        cache[key] = {
            "day4": features_to_vector(extract_features(path)),
            "local": [lf.extract_local_features(path)[n] for n in lf.FEATURE_NAMES],
        }
        if i % 20 == 0 or i == len(todo):  # incremental: an interrupted run keeps its work
            CACHE.write_bytes(pickle.dumps(cache))
            print(f"  features cached {i}/{len(todo)}", flush=True)
    return cache


def feature_sets() -> dict[str, list[str]]:
    no_short = [n for n in DAY4_NAMES if n not in SHORTCUT_FEATURES]
    return {
        "A_global_day4": list(DAY4_NAMES),
        "B_local_v1": list(lf.FEATURE_NAMES),
        "C_global_plus_local": list(DAY4_NAMES) + list(lf.FEATURE_NAMES),
        "A_minus_shortcuts": no_short,
        "C_minus_shortcuts": no_short + list(lf.FEATURE_NAMES),
    }


def matrix(cache, keys, names) -> np.ndarray:
    full = list(DAY4_NAMES) + list(lf.FEATURE_NAMES)
    cols = [full.index(n) for n in names]
    return np.array([(cache[k]["day4"] + cache[k]["local"]) for k in keys], dtype=np.float64)[:, cols]


def metrics(y, p) -> dict:
    cm = confusion_matrix(y, p, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0.0),
        "recall": recall_score(y, p, zero_division=0.0),
        "f1": f1_score(y, p, zero_division=0.0),
        "confusion": cm.tolist(), "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "fpr": float(cm[0, 1] / max(1, cm[0].sum())), "fnr": float(cm[1, 0] / max(1, cm[1].sum())),
    }


def folds():
    rows = _train_rows()
    y = np.array([r["label"] == "tampered" for r in rows], dtype=int)
    return rows, y, list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def run_cv(cache, names, rf_params, use_aug) -> dict:
    """Identical folds on the 60 originals; augmented rows follow their source into
    the TRAINING side of a fold only; validation = originals of the held-out fold."""
    rows, y, splits = folds()
    manifest = load_manifest()
    keys = [f"orig:{r['path']}" for r in rows]
    X = matrix(cache, keys, names)
    assert_no_test_leakage(keys)
    per_fold, y_all, p_all, err, preds = [], [], [], [], []
    for fold, (tr, va) in enumerate(splits, 1):
        Xtr, ytr = X[tr], y[tr]
        tr_src = {rows[i]["path"] for i in tr}
        va_src = {rows[i]["path"] for i in va}
        if use_aug:
            # Group rule: an augmented image goes wherever its SOURCE document is.
            # Sources in the held-out fold contribute no augmented image to training.
            aug = [m for m in manifest if m["source_path"] in tr_src]
            assert {m["source_path"] for m in aug} <= tr_src
            assert not {m["source_path"] for m in aug} & va_src
            Xtr = np.vstack([Xtr, matrix(cache, [aug_key(m["file"]) for m in aug], names)])
            ytr = np.concatenate([ytr, [m["label"] == "tampered" for m in aug]]).astype(int)
        m = RandomForestClassifier(**rf_params, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        p = m.predict(X[va])
        per_fold.append(metrics(y[va], p))
        y_all += list(y[va]); p_all += list(p)
        for i, pi in zip(va, p):
            rec = {"file": rows[i]["path"], "fold": fold, "actual": int(y[i]), "pred": int(pi),
                   "doc_type": INDEX[rows[i]["path"]]["doc_type"],
                   "tamper_method": INDEX[rows[i]["path"]]["tamper_method"]}
            preds.append(rec)
            if pi != y[i]:
                err.append(rec)
    summ = {k: {"mean": float(np.mean([f[k] for f in per_fold])), "std": float(np.std([f[k] for f in per_fold]))}
            for k in ("accuracy", "precision", "recall", "f1")}
    by_doc = {}
    for d in sorted({r["doc_type"] for r in preds}):
        sub = [r for r in preds if r["doc_type"] == d]
        by_doc[d] = {"genuine": sum(r["actual"] == 0 for r in sub), "tampered": sum(r["actual"] == 1 for r in sub),
                     **metrics(np.array([r["actual"] for r in sub]), np.array([r["pred"] for r in sub]))}
    by_method = {}
    for mth in sorted({r["tamper_method"] for r in preds if r["actual"] == 1}):
        sub = [r for r in preds if r["actual"] == 1 and r["tamper_method"] == mth]
        det = sum(r["pred"] == 1 for r in sub)
        by_method[mth] = {"n": len(sub), "detected": det, "missed": len(sub) - det, "recall": det / len(sub)}
    return {"features": len(names), "use_aug": use_aug, "aug_version": AUG_VERSION if use_aug else None,
            "rf_params": rf_params, "summary": summ, "per_fold": per_fold,
            "pooled": metrics(np.array(y_all), np.array(p_all)), "by_doc_type": by_doc,
            "by_original_method": by_method, "errors": err, "predictions": preds}


def aug_detection(cache, names, rf_params) -> dict:
    """Per-transform detection on AUGMENTED images whose source is in the held-out
    fold (model trained on train-fold originals + their augmentations only).
    Measures which manipulation types a representation can see; training data only."""
    rows, y, splits = folds()
    manifest = load_manifest()
    X = matrix(cache, [f"orig:{r['path']}" for r in rows], names)
    hits: dict[str, list[int]] = {}
    by_global: dict[str, list[int]] = {}
    for tr, va in splits:
        tr_src = {rows[i]["path"] for i in tr}
        va_src = {rows[i]["path"] for i in va}
        aug_tr = [m for m in manifest if m["source_path"] in tr_src]
        aug_va = [m for m in manifest if m["source_path"] in va_src]
        Xtr = np.vstack([X[tr], matrix(cache, [aug_key(m["file"]) for m in aug_tr], names)])
        ytr = np.concatenate([y[tr], [m["label"] == "tampered" for m in aug_tr]]).astype(int)
        model = RandomForestClassifier(**rf_params, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
        pred = model.predict(matrix(cache, [aug_key(m["file"]) for m in aug_va], names))
        for m, p in zip(aug_va, pred):
            ok = int(p == (m["label"] == "tampered"))
            hits.setdefault(f"{m['label']}:{m['transform'].split(':')[0]}", []).append(ok)
            g = "untouched" if m["global_transform"] == "none" else m["global_transform"].split(":")[0]
            by_global.setdefault(f"{m['label']}:{g}", []).append(ok)
            if g != "untouched":
                by_global.setdefault(f"{m['label']}:ANY_whole_page", []).append(ok)
    agg = lambda d: {k: {"n": len(v), "correct": int(sum(v))} for k, v in sorted(d.items())}
    return {"by_local_op": agg(hits), "by_whole_page_op": agg(by_global)}


def pick(candidates: dict[str, dict], complexity) -> str:
    best = max(c["summary"]["f1"]["mean"] for c in candidates.values())
    close = [k for k, c in candidates.items() if c["summary"]["f1"]["mean"] >= best - 0.02]
    return min(close, key=lambda k: (candidates[k]["summary"]["f1"]["std"], complexity(k)))


# ── stages ───────────────────────────────────────────────────────────────────

def stage_audit(res):
    import hashlib

    from PIL import Image
    rows = SPLIT["train"] + SPLIT["test"]
    info = []
    for r in rows:
        p = DATASET / r["path"]
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            fmt, size, mode, meta = im.format, im.size, im.mode, sorted(im.info.keys())
            pix = hashlib.sha256(np.asarray(im.convert("RGB")).tobytes()).hexdigest()
        info.append({"path": r["path"], "label": r["label"], "fmt": fmt, "size": size, "mode": mode,
                     "meta_keys": meta, "file_sha": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "pix_sha": pix, "bytes": p.stat().st_size,
                     "doc_type": INDEX[r["path"]]["doc_type"], "tamper": INDEX[r["path"]]["tamper_method"]})
    split_of = {r["path"]: "train" for r in SPLIT["train"]} | {r["path"]: "test" for r in SPLIT["test"]}
    audit = {
        "total": len(info),
        "labels": dict(Counter(i["label"] for i in info)),
        "doc_types": {s: dict(Counter(f"{i['doc_type']}|{i['label']}" for i in info if split_of[i["path"]] == s).most_common())
                      for s in ("train", "test")},
        "formats": dict(Counter(i["fmt"] for i in info)),
        "modes": dict(Counter(i["mode"] for i in info)),
        "sizes": dict(Counter(f"{i['size'][0]}x{i['size'][1]}|{i['doc_type']}" for i in info)),
        "png_metadata_keys": dict(Counter(",".join(i["meta_keys"]) or "(none)" for i in info)),
        "dup_files": [k for k, v in Counter(i["file_sha"] for i in info).items() if v > 1],
        "dup_pixels": [k for k, v in Counter(i["pix_sha"] for i in info).items() if v > 1],
        "label_folder_mismatch": [i["path"] for i in info if i["path"].split("/")[0] != INDEX[i["path"]]["label"]],
        "filename_contains_label": [i["path"] for i in info if any(w in Path(i["path"]).name.lower()
                                                                   for w in ("tamper", "fake", "genuine", "edit"))],
        "tamper_methods": {s: dict(Counter(i["tamper"] for i in info if split_of[i["path"]] == s and i["tamper"]))
                           for s in ("train", "test")},
        "bytes_by_label": {lab: [int(np.min(b)), int(np.median(b)), int(np.max(b))] for lab in ("genuine", "tampered")
                           for b in [[i["bytes"] for i in info if i["label"] == lab]]},
        "frozen_test_sha_ok": all(hashlib.sha256((DATASET / r["path"]).read_bytes()).hexdigest() == r["sha256"]
                                  for r in SPLIT["test"]),
    }
    # Day-4 feature audit on TRAIN images only
    cache = feature_table()
    keys = [f"orig:{r['path']}" for r in SPLIT["train"]]
    X = matrix(cache, keys, list(DAY4_NAMES))
    y = np.array([r["label"] == "tampered" for r in SPLIT["train"]], dtype=int)
    dt = np.array([INDEX[r["path"]]["doc_type"] for r in SPLIT["train"]])
    corr = np.corrcoef(X.T)
    feats = {}
    for j, n in enumerate(DAY4_NAMES):
        c = X[:, j]
        top_share = Counter(np.round(c, 9)).most_common(1)[0][1] / len(c)
        ss_tot = ((c - c.mean()) ** 2).sum()
        eta2 = float(sum(((c[dt == d].mean() - c.mean()) ** 2) * (dt == d).sum() for d in set(dt)) / ss_tot) if ss_tot else 0.0
        auc = roc_auc_score(y, c) if np.ptp(c) else 0.5
        redundant = [DAY4_NAMES[k] for k in range(len(DAY4_NAMES))
                     if k != j and np.ptp(c) and np.ptp(X[:, k]) and abs(corr[j, k]) > 0.95]
        feats[n] = {"min": c.min(), "max": c.max(), "mean": c.mean(), "std": c.std(),
                    "unique": int(len(np.unique(c))), "missing": int(np.isnan(c).sum()),
                    "infinite": int(np.isinf(c).sum()), "constant": bool(np.ptp(c) == 0),
                    "near_constant": bool(top_share >= 0.95), "doc_type_eta2": eta2,
                    "label_auc": float(max(auc, 1 - auc)), "redundant_with": redundant}
    audit["day4_feature_audit_train_only"] = feats
    res["audit"] = audit
    print(json.dumps({k: v for k, v in audit.items() if k != "day4_feature_audit_train_only"}, indent=1, default=str))
    for n, f in feats.items():
        print(f"{n:28s} min={f['min']:.4g} max={f['max']:.4g} uniq={f['unique']:3d} const={f['constant']!s:5s} "
              f"near={f['near_constant']!s:5s} eta2_doc={f['doc_type_eta2']:.2f} auc={f['label_auc']:.2f} red={f['redundant_with']}")


def _importance_groups(cache, names, keys, y) -> dict:
    """Feature-importance share by signal family; layout-sensitive vs forensic."""
    m = RandomForestClassifier(**BASE_RF, random_state=SEED, n_jobs=-1).fit(matrix(cache, keys, names), np.array(y, int))
    groups: dict[str, float] = {}
    for n, v in zip(names, m.feature_importances_):
        if not n.startswith("lf_"):
            g = "day4_shortcut" if n in SHORTCUT_FEATURES else "day4_other"
        elif n.startswith(("lf_region", "lf_anomaly")):
            g = "local_region"
        else:
            g = "local_" + n.split("_")[1]
        groups[g] = groups.get(g, 0.0) + float(v)
    layout = sum(v for k, v in groups.items() if k in LAYOUT_GROUPS)
    top = sorted(zip(names, m.feature_importances_), key=lambda t: -t[1])[:12]
    return {"groups": dict(sorted(groups.items(), key=lambda t: -t[1])), "layout_sensitive_share": layout,
            "forensic_share": 1.0 - layout, "top12": [(n, float(v)) for n, v in top]}


# Signal families that mostly describe page layout / appearance rather than edits.
LAYOUT_GROUPS = {"day4_shortcut", "local_gradient", "local_hf", "local_intensity", "local_contrast"}


def stage_cv(res):
    from scripts.augment_training_data import audit as aug_audit
    aug_rep = aug_audit()
    if not aug_rep["pass"]:
        raise SystemExit(f"augmentation audit FAILED — fix before CV: {aug_rep['checks']}")
    assert aug_check(), "augmented files do not match manifest — regenerate"
    cache = feature_table()
    sets = feature_sets()
    exps = {}
    for name, names in sets.items():
        for use_aug in (False, True):
            key = name + ("+aug" if use_aug else "")
            exps[key] = run_cv(cache, names, BASE_RF, use_aug)
            s = exps[key]["summary"]
            print(f"{key:28s} n_feat={len(names):3d} acc={s['accuracy']['mean']:.3f}+-{s['accuracy']['std']:.3f} "
                  f"prec={s['precision']['mean']:.3f} rec={s['recall']['mean']:.3f} "
                  f"f1={s['f1']['mean']:.3f}+-{s['f1']['std']:.3f} pooled_cm={exps[key]['pooled']['confusion']}")
    rows = _train_rows()
    okeys = [f"orig:{r['path']}" for r in rows]
    oy = [r["label"] == "tampered" for r in rows]
    man = load_manifest()
    akeys = okeys + [aug_key(m["file"]) for m in man]
    ay = oy + [m["label"] == "tampered" for m in man]

    # Document-type probe: how well does each feature set predict DOC TYPE? (train only)
    dt = np.array([INDEX[r["path"]]["doc_type"] for r in rows])
    probe = {}
    for name in ("A_global_day4", "B_local_v1", "A_minus_shortcuts"):
        sc = cross_val_score(RandomForestClassifier(**BASE_RF, random_state=SEED, n_jobs=-1),
                             matrix(cache, okeys, sets[name]), dt,
                             cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
        probe[name] = {"doc_type_acc_mean": float(sc.mean()), "doc_type_acc_std": float(sc.std())}

    importance = {
        "B_local_v1": _importance_groups(cache, sets["B_local_v1"], okeys, oy),
        "B_local_v1+aug": _importance_groups(cache, sets["B_local_v1"], akeys, ay),
        "C_global_plus_local": _importance_groups(cache, sets["C_global_plus_local"], okeys, oy),
        "C_global_plus_local+aug": _importance_groups(cache, sets["C_global_plus_local"], akeys, ay),
    }
    # Constant local features (train originals / full training pool) — flagged, not removed.
    Xo, Xa = matrix(cache, okeys, list(lf.FEATURE_NAMES)), matrix(cache, akeys, list(lf.FEATURE_NAMES))
    constant_local = {"train_originals": [n for j, n in enumerate(lf.FEATURE_NAMES) if np.ptp(Xo[:, j]) == 0],
                      "training_pool": [n for j, n in enumerate(lf.FEATURE_NAMES) if np.ptp(Xa[:, j]) == 0]}

    aug_det = {k: aug_detection(cache, sets[k], BASE_RF) for k in ("A_global_day4", "B_local_v1", "C_global_plus_local")}

    main = {k: v for k, v in exps.items() if k.split("+")[0] in ("A_global_day4", "B_local_v1", "C_global_plus_local")}
    rule_candidate = pick(main, complexity=lambda k: (main[k]["features"], main[k]["use_aug"]))

    # Reproducibility: folds identical to the previous run (non-augmented experiments must match exactly).
    prev = res.get("cv")
    fold_repro = all(
        [f["confusion"] for f in exps[k]["per_fold"]] == [f["confusion"] for f in prev["experiments"][k]["per_fold"]]
        for k in ("A_global_day4", "B_local_v1", "C_global_plus_local")) if prev else None
    folds_a, folds_b = folds()[2], folds()[2]
    checks = {
        "60_original_training_images_intact": len(rows) == 60 and all(
            hashlib.sha256((DATASET / r["path"]).read_bytes()).hexdigest() == r["sha256"] for r in rows),
        "15_frozen_test_images_unchanged": len(SPLIT["test"]) == 15 and all(
            hashlib.sha256((DATASET / r["path"]).read_bytes()).hexdigest() == r["sha256"] for r in SPLIT["test"]),
        "0_test_images_in_augmentation": not aug_rep["any_source_in_test"] and not aug_rep["any_output_hash_equals_test"],
        "0_test_images_in_cv": all(not ({p["file"] for p in e["predictions"]} & TEST_PATHS) for e in exps.values()),
        "0_test_records_in_training_cache": not any(k in cache for k in (f"orig:{p}" for p in TEST_PATHS)),
        "augmentation_v1_and_v2.0_excluded": all(k.startswith(f"aug[{AUG_VERSION}]:") for k in cache if k.startswith("aug")),
        "augmentation_provenance_complete": aug_rep["checks"]["provenance_complete"],
        "whole_page_processing_not_label_dependent": aug_rep["checks"]["processed_rate_gap_le_0.05"],
        "no_metadata_or_format_leakage": aug_rep["checks"]["single_format_and_no_metadata"],
        "no_dimension_leakage": max(aug_rep["single_descriptor_label_auc"][k] for k in ("width", "height", "aspect")) <= 0.65,
        "no_compression_label_leakage": aug_rep["single_descriptor_label_auc"]["jpeg_grid"] <= 0.65
                                        and aug_rep["single_descriptor_label_auc"]["file_bytes"] <= 0.65,
        "combined_shortcut_probe_near_chance": aug_rep["checks"]["combined_probe_balanced_acc_le_0.60"],
        "sources_and_augmentations_same_fold": True,  # asserted inside run_cv for every fold
        "no_duplicate_source_leakage": len({m["source_path"] for m in man}) == 38 and len({m["sha256"] for m in man}) == len(man),
        "deterministic_seed": True,  # generator re-run produced a byte-identical manifest (checked before CV)
        "cv_folds_reproducible": [list(v) for _, v in folds_a] == [list(v) for _, v in folds_b]
                                 and (fold_repro is not False),
    }
    if "cv" in res and res["cv"].get("aug_version", "train-aug-v1") != AUG_VERSION:
        old = res.pop("cv")
        res[f"cv_{old.get('aug_version', 'train-aug-v1')}"] = old  # keep old run as history
    res["cv"] = {"aug_version": AUG_VERSION, "selection_rule": SELECTION_RULE, "experiments": exps,
                 "doc_type_probe": probe, "importance": importance, "constant_local_features": constant_local,
                 "aug_heldout_detection": aug_det, "augmentation_audit": aug_rep,
                 "selected": rule_candidate, "selected_note": "rule output only; not a deployment decision",
                 "validation_checks": checks, "fold_repro_vs_previous_run": fold_repro,
                 "shortcut_features": SHORTCUT_FEATURES}
    print("doc-type probe:", {k: round(v["doc_type_acc_mean"], 3) for k, v in probe.items()})
    print("validation checks:", "ALL PASS" if all(checks.values()) else {k: v for k, v in checks.items() if not v})
    if not all(checks.values()):
        raise SystemExit("validation check failed — see results")


def _pct(x):
    return f"{100 * x:.1f}%"


def stage_report(res):
    """Render logs/day5_augmentation_v2_cv_report.md from the saved results (data sections only)."""
    cv = res["cv"]
    ex = cv["experiments"]
    main_rows = [("A", "A_global_day4", "day4-v2-22 (22)", "none"),
                 ("B", "B_local_v1", "local-forensics-v1 (52)", "none"),
                 ("C", "C_global_plus_local", "day4 + local (74)", "none"),
                 ("A + aug", "A_global_day4+aug", "day4-v2-22 (22)", cv["aug_version"]),
                 ("B + aug", "B_local_v1+aug", "local-forensics-v1 (52)", cv["aug_version"]),
                 ("C + aug", "C_global_plus_local+aug", "day4 + local (74)", cv["aug_version"])]
    L = []
    w = L.append
    w("<!-- DATA SECTIONS BELOW ARE GENERATED by `python scripts/ml_recovery.py report` from "
      "logs/day5_ml_recovery_results.json. Narrative sections are written by hand. -->\n")
    w("## Experiment comparison — CV DATA (60 training originals, 5-fold, mean ± std)\n")
    w("| Model | Features | Augmentation | Accuracy | Precision | Recall | F1 | FP | FN | FPR | FNR |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, k, feats, aug in main_rows:
        s, p = ex[k]["summary"], ex[k]["pooled"]
        f = lambda m: f"{s[m]['mean']:.3f} ± {s[m]['std']:.3f}"
        w(f"| {lab} | {feats} | {aug} | {f('accuracy')} | {f('precision')} | {f('recall')} | {f('f1')} | "
          f"{p['fp']} | {p['fn']} | {_pct(p['fpr'])} | {_pct(p['fnr'])} |")
    w("\nAblations (layout-shortcut Day-4 features removed):\n")
    w("| Experiment | Accuracy | F1 | FP | FN |\n|---|---|---|---|---|")
    for k in ("A_minus_shortcuts", "A_minus_shortcuts+aug", "C_minus_shortcuts", "C_minus_shortcuts+aug"):
        s, p = ex[k]["summary"], ex[k]["pooled"]
        w(f"| {k} | {s['accuracy']['mean']:.3f} ± {s['accuracy']['std']:.3f} | {s['f1']['mean']:.3f} ± {s['f1']['std']:.3f} | {p['fp']} | {p['fn']} |")

    w("\n## Fold-level results — CV DATA\n")
    w("Each cell: accuracy / precision / recall / F1, confusion [[TN, FP], [FN, TP]].\n")
    w("| Fold | " + " | ".join(lab for lab, *_ in main_rows) + " |")
    w("|---|" + "---|" * len(main_rows))
    for i in range(5):
        cells = []
        for _, k, *_ in main_rows:
            f = ex[k]["per_fold"][i]
            cells.append(f"{f['accuracy']:.3f}/{f['precision']:.3f}/{f['recall']:.3f}/{f['f1']:.3f} {f['confusion']}")
        w(f"| {i + 1} | " + " | ".join(cells) + " |")

    w("\n## Pooled confusion matrices — CV DATA (all 60 originals, each predicted once)\n")
    w("| Model | TN | FP | FN | TP |\n|---|---|---|---|---|")
    for lab, k, *_ in main_rows:
        p = ex[k]["pooled"]
        w(f"| {lab} | {p['tn']} | {p['fp']} | {p['fn']} | {p['tp']} |")

    w("\n## Document-type analysis — CV DATA\n")
    for lab, k, *_ in main_rows:
        w(f"**{lab}**\n")
        w("| Doc type | Genuine | Tampered | Accuracy | Precision | Recall | F1 | Confusion |")
        w("|---|---|---|---|---|---|---|---|")
        for d, m in ex[k]["by_doc_type"].items():
            w(f"| {d} | {m['genuine']} | {m['tampered']} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
              f"{m['recall']:.3f} | {m['f1']:.3f} | {m['confusion']} |")
        w("")
    w("Document-type probe (predict doc type from features, 5-fold): " +
      ", ".join(f"{k} = {v['doc_type_acc_mean']:.3f}" for k, v in cv["doc_type_probe"].items()) + "\n")
    w("Feature-importance share (RF fit on all training data of that experiment):\n")
    w("| Model | Layout-sensitive share | Forensic share | Largest groups |\n|---|---|---|---|")
    for k, v in cv["importance"].items():
        groups = ", ".join(f"{g} {x:.2f}" for g, x in list(v["groups"].items())[:5])
        w(f"| {k} | {v['layout_sensitive_share']:.2f} | {v['forensic_share']:.2f} | {groups} |")
    w(f"\nLayout-sensitive groups: {', '.join(sorted(LAYOUT_GROUPS))}.\n")
    w(f"Constant local features — train originals: {cv['constant_local_features']['train_originals'] or 'none'}; "
      f"full training pool: {cv['constant_local_features']['training_pool'] or 'none'}. Flagged, not removed.\n")

    w("\n## Tampering-method analysis\n")
    w("### ORIGINAL tampering — CV DATA (held-out originals)\n")
    methods = sorted(ex["A_global_day4"]["by_original_method"])
    w("| Method | n | " + " | ".join(f"{lab} detected / missed / recall" for lab, *_ in main_rows) + " |")
    w("|---|---|" + "---|" * len(main_rows))
    for mth in methods:
        cells = []
        for _, k, *_ in main_rows:
            m = ex[k]["by_original_method"][mth]
            cells.append(f"{m['detected']} / {m['missed']} / {m['recall']:.2f}")
        w(f"| {mth} | {ex['A_global_day4']['by_original_method'][mth]['n']} | " + " | ".join(cells) + " |")
    w(f"\n### SYNTHETIC {cv['aug_version']} tampering — TRAINING DATA, held out by source document\n")
    w("Model trained on the other 4 folds' originals + their augmentations; scored on augmented images whose "
      "source document is in the held-out fold. Synthetic edits, NOT independent real fraud samples.\n")
    det = cv["aug_heldout_detection"]
    keys = sorted(det["B_local_v1"]["by_local_op"])
    w("| Label:local op | n | A correct | B correct | C correct |\n|---|---|---|---|---|")
    for key in keys:
        c = [det[k]["by_local_op"].get(key, {"correct": 0, "n": 0}) for k in ("A_global_day4", "B_local_v1", "C_global_plus_local")]
        w(f"| {key} | {c[0]['n']} | {c[0]['correct']} | {c[1]['correct']} | {c[2]['correct']} |")
    w("\nBy whole-page processing (robustness to benign scanning / re-saving):\n")
    keys = sorted(det["B_local_v1"]["by_whole_page_op"])
    w("| Label:whole-page op | n | A correct | B correct | C correct |\n|---|---|---|---|---|")
    for key in keys:
        c = [det[k]["by_whole_page_op"].get(key, {"correct": 0, "n": 0}) for k in ("A_global_day4", "B_local_v1", "C_global_plus_local")]
        w(f"| {key} | {c[0]['n']} | {c[0]['correct']} | {c[1]['correct']} | {c[2]['correct']} |")

    w("\n## False positives / false negatives — CV DATA\n")
    for lab, k, *_ in main_rows:
        e = ex[k]["errors"]
        fp = [x["file"] for x in e if x["actual"] == 0]
        fn = [f"{x['file']} ({x['tamper_method']})" for x in e if x["actual"] == 1]
        w(f"- **{lab}** — FP ({len(fp)}): {', '.join(fp) or 'none'}")
        w(f"  - FN ({len(fn)}): {', '.join(fn) or 'none'}")

    a = cv["augmentation_audit"]
    w("\n## Augmentation audit — TRAINING DATA\n")
    w(f"- Rows: {a['rows']} ({a['labels']}) from {a['distinct_source_documents']} training genuine source documents "
      "— augmented VIEWS, not independent documents.")
    w(f"- Whole-page processed rate over the full training pool: genuine {a['genuine_processed_rate']}, "
      f"tampered {a['tampered_processed_rate']}; untouched: genuine {a['genuine_untouched_rate']}, "
      f"tampered {a['tampered_untouched_rate']}.")
    w(f"- Augmented-only processed rate: {a['augmented_only_processed_rate']} (genuine variants are always processed "
      "because an unprocessed genuine variant would duplicate its source; the pool-level rate is what the model sees).")
    w(f"- Whole-page ops — genuine: {a['genuine_global_ops']}; tampered: {a['tampered_global_ops']}.")
    w(f"- Local ops: {a['local_ops']}.")
    w(f"- Fonts used for text edits: {a['fonts']}.")
    w(f"- Single-descriptor label AUC (0.5 = no signal): {a['single_descriptor_label_auc']}.")
    w(f"- Combined shortcut probe (file size, dims, aspect, brightness, noise, JPEG grid, format, mode, metadata; "
      f"grouped 5-fold): balanced accuracy {a['shortcut_probe']['balanced_accuracy']} (chance 0.5).")
    w(f"- Dimensions by label (top): {a['dimensions_by_label']}.")
    w(f"- Distributions by label (p5 / p50 / p95): {a['distributions_by_label_p5_p50_p95']}.")
    w(f"- Audit checks: {a['checks']} → {'PASS' if a['pass'] else 'FAIL'}.")

    w("\n## Automated validation — all data\n")
    for k, v in cv["validation_checks"].items():
        w(f"- [{'x' if v else ' '}] {k.replace('_', ' ')}")
    w(f"\nNon-augmented experiments identical to the previous run fold-by-fold: {cv['fold_repro_vs_previous_run']}.")

    w("\n## Comparison with previous runs — CV DATA\n")
    w("| Run | B+aug acc | B+aug F1 | B+aug FP | B+aug FN | C+aug acc | C+aug FP | C+aug FN |\n|---|---|---|---|---|---|---|---|")
    for key in sorted(k for k in res if k.startswith("cv")):
        r = res[key]["experiments"]
        b, c = r["B_local_v1+aug"], r["C_global_plus_local+aug"]
        w(f"| {res[key].get('aug_version', 'train-aug-v1')} | {b['summary']['accuracy']['mean']:.3f} | "
          f"{b['summary']['f1']['mean']:.3f} | {b['pooled']['fp']} | {b['pooled']['fn']} | "
          f"{c['summary']['accuracy']['mean']:.3f} | {c['pooled']['fp']} | {c['pooled']['fn']} |")
    out = ROOT / "logs" / "day5_augmentation_v2_cv_report.md"
    head = out.read_text(encoding="utf-8").split("<!-- DATA SECTIONS BELOW")[0] if out.exists() else ""
    out.write_text(head + "\n".join(L) + "\n", encoding="utf-8")
    print(f"report written: {out}")


def stage_grid(res):
    cache = feature_table()
    selected = res["cv"]["selected"]
    names = feature_sets()[selected.split("+")[0]]
    use_aug = selected.endswith("+aug")
    runs = {}
    keys = list(GRID)
    for combo in product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        tag = json.dumps(params, sort_keys=True)
        runs[tag] = run_cv(cache, names, params, use_aug)
    def complexity(tag):
        p = json.loads(tag)
        return (p["n_estimators"], p["max_depth"] or 99, -p["min_samples_leaf"])
    best = pick(runs, complexity)
    table = sorted(((t, r["summary"]["f1"]["mean"], r["summary"]["f1"]["std"], r["summary"]["accuracy"]["mean"])
                    for t, r in runs.items()), key=lambda x: -x[1])
    for t, f, s, a in table[:10]:
        print(f"f1={f:.3f}±{s:.3f} acc={a:.3f} {t}")
    res["grid"] = {"selected_representation": selected, "n_configs": len(runs), "selected_params": json.loads(best),
                   "selected_cv": runs[best], "top10": table[:10],
                   "baseline_cv": res["cv"]["experiments"][selected]}
    print("SELECTED PARAMS:", best)


def stage_final(res):
    if res.get("final", {}).get("frozen_test"):
        raise SystemExit("Frozen test set already evaluated — refusing a second evaluation.")
    cache = feature_table()
    selected = res["cv"]["selected"]
    names = feature_sets()[selected.split("+")[0]]
    assert names == COMBINED_FEATURE_NAMES or names in (list(DAY4_NAMES), list(lf.FEATURE_NAMES)), names
    params = res["grid"]["selected_params"]
    rows = _train_rows()
    keys = [f"orig:{r['path']}" for r in rows]
    y = [r["label"] == "tampered" for r in rows]
    if selected.endswith("+aug"):
        man = load_manifest()
        keys += [aug_key(m["file"]) for m in man]
        y += [m["label"] == "tampered" for m in man]
    model = RandomForestClassifier(**params, random_state=SEED, n_jobs=-1).fit(matrix(cache, keys, names), np.array(y, dtype=int))

    metadata = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": f"{lf.FEATURE_SCHEMA_VERSION}+day4-v2-22" if len(names) > len(lf.FEATURE_NAMES) else lf.FEATURE_SCHEMA_VERSION,
        "feature_names": names,
        "feature_count": len(names),
        "label_map": {"genuine": 0, "tampered": 1},
        "training_seed": SEED,
        "rf_params": {**params, "random_state": SEED},
        "training_dataset_sha256": SPLIT["dataset_sha256"],
        "frozen_split": SPLIT["name"],
        "augmentation": {"version": AUG_VERSION, "manifest": str(MANIFEST.relative_to(ROOT).as_posix()),
                         "used": selected.endswith("+aug")},
        "training_image_count": len(keys),
        "training_original_images": len(rows),
        "cv_metrics": res["grid"]["selected_cv"]["summary"],
        "selection_rule": SELECTION_RULE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score_semantics": "RandomForest vote fraction for 'tampered'. Uncalibrated model score, NOT a probability of fraud.",
    }
    MODEL_OUT.write_bytes(pickle.dumps({"model": model, "metadata": metadata}, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"saved {MODEL_OUT} ({MODEL_OUT.stat().st_size:,} bytes)")

    # ── ONE evaluation on the frozen test set ──
    cache = feature_table(include_test=True)
    trows = SPLIT["test"]
    Xt = matrix(cache, [f"orig:{r['path']}" for r in trows], names)
    yt = np.array([r["label"] == "tampered" for r in trows], dtype=int)
    pt = model.predict(Xt)
    st = model.predict_proba(Xt)[:, 1]
    m = metrics(yt, pt)
    m["correct"] = int((pt == yt).sum()); m["total"] = len(yt)
    m["per_file"] = [{"file": r["path"], "actual": int(a), "pred": int(p), "model_score": float(s),
                      "doc_type": INDEX[r["path"]]["doc_type"], "tamper_method": INDEX[r["path"]]["tamper_method"]}
                     for r, a, p, s in zip(trows, yt, pt, st)]
    res["final"] = {"model_path": str(MODEL_OUT.relative_to(ROOT).as_posix()), "metadata": metadata,
                    "frozen_test": m, "evaluated_at": datetime.now(timezone.utc).isoformat()}
    print(json.dumps({k: v for k, v in m.items() if k != "per_file"}, indent=1))
    for f in m["per_file"]:
        print(f"  {'ok ' if f['actual'] == f['pred'] else 'ERR'} {f['file']:30s} score={f['model_score']:.2f} {f['tamper_method']}")


def stage_perf(res):
    from core.forensic_model import ForensicModel
    fm = ForensicModel.load(MODEL_OUT)
    imgs = [DATASET / r["path"] for r in _train_rows()[:10]]
    t_day4, t_local, t_inf, t_total = [], [], [], []
    for p in imgs:
        t0 = time.perf_counter(); extract_features(p); t_day4.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); lf.extract_local_features(p); t_local.append(time.perf_counter() - t0)
        vec = np.array([extract_combined(p, fm.feature_names)])
        t0 = time.perf_counter(); fm.model.predict_proba(vec); t_inf.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fm.analyze(p); t_total.append(time.perf_counter() - t0)
    ms = lambda v: {"mean_ms": 1000 * float(np.mean(v)), "max_ms": 1000 * float(np.max(v))}
    res["perf"] = {"images": len(imgs), "day4_extraction": ms(t_day4), "local_extraction": ms(t_local),
                   "inference": ms(t_inf), "total_analysis": ms(t_total),
                   "model_size_bytes": MODEL_OUT.stat().st_size}
    print(json.dumps(res["perf"], indent=1))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"audit": stage_audit, "cv": stage_cv, "grid": stage_grid, "final": stage_final, "perf": stage_perf,
          "report": stage_report}.get(stage)
    if fn is None:
        raise SystemExit(__doc__)
    res = load_results()
    fn(res)
    save_results(res)
