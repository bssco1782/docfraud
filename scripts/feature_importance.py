"""
scripts/feature_importance.py
Day 9 (Founder A): feature importance of the deployed model.

Usage:
    python scripts/feature_importance.py
    python scripts/feature_importance.py --report logs/feature_importance_report.txt --json logs/feature_importance_results.json

For every feature of the deployed model (data/models/fraud_model.pkl):
  MDI          mean decrease in impurity from the deployed forest (feature_importances_), with the
               spread across its trees.
  Permutation  drop in accuracy (and ROC AUC) when the feature is shuffled, measured out-of-fold:
               grouped 5-fold CV on the model's own training data (60 frozen TRAIN + 150 DEV
               documents + processed copies, rebuilt by scripts/train_tamper_model.py); copies of a
               validation document are never in its training fold; scored on pristine originals.
  Impact       partial dependence of the deployed model: mean tamper score with the feature set to
               its 90th percentile minus with it set to its 10th percentile (over the training rows;
               for sparse features where both are equal, the 99th percentile / maximum is used).
               Positive = higher values push towards "tampered".
  Split share  share of the forest's split nodes that use the feature.
Plus family totals, a retrain-without-each-family check (correlated features share credit, so
single-feature permutation understates them), a CV check without the least useful features, and
strongly correlated feature pairs.

Only training data is used: the frozen TEST set and data/generated/holdout_v1/ are not read, so
this analysis can guide future feature choices without leaking held-out information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import rankdata
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from core.feature_schemas import get_schema
from core.model_trainer import DEFAULT_MODEL_PATH, DocumentFraudDetector
from scripts import train_tamper_model as ttm

DEFAULT_REPORT = ROOT / "logs" / "feature_importance_report.txt"
DEFAULT_JSON = ROOT / "logs" / "feature_importance_results.json"
CV_SPLITS = 5
PERM_REPEATS = 10
SEED = 42
FAMILIES = {"noise": "noise_", "colour": "colour_", "softness": "soft_", "ink": "ink_rare",
            "fingerprint": ("jpeg_grid", "flat_frac", "paper_colours", "ink_levels")}
FAMILY_MEANING = {
    "noise": "a region whose pixel noise is above the page's own floor (noise patches)",
    "colour": "a region of rare or novel paper colour (colour casts, pasted boxes)",
    "softness": "content tiles with weak edges for their contrast (blurred regions)",
    "ink": "a run of glyphs with a rare darkness level (inserted text)",
    "fingerprint": "how the file was saved (JPEG grid, flatness, colour / ink variety)",
}
USEFUL_MIN_PERM = 0.002   # accuracy drop below this ...
USEFUL_MIN_MDI = 0.02     # ... and MDI below this = "least useful"
CORRELATED = 0.9          # |Spearman rho| at or above this = redundant pair


def family_of(name: str) -> str:
    for fam, prefix in FAMILIES.items():
        if (isinstance(prefix, tuple) and name in prefix) or (isinstance(prefix, str) and name.startswith(prefix)):
            return fam
    return "other"


def training_matrix() -> dict:
    docs = ttm.training_documents()
    data = ttm.build_matrix(docs, verbose=False)
    data["docs"] = docs
    return data


def grouped_folds(data: dict):
    """(train rows, validation original rows, validation doc indices) per fold; copies stay with their document."""
    orig = np.flatnonzero(~data["is_copy"])
    for tr_docs, va_docs in StratifiedKFold(CV_SPLITS, shuffle=True, random_state=SEED).split(orig, data["y"][orig]):
        train_rows = np.flatnonzero(np.isin(data["groups"], tr_docs))
        yield train_rows, orig[va_docs]


def cv_accuracy(data: dict, columns: list[int], processed: bool = False) -> float | tuple[float, float]:
    """Grouped-CV accuracy on pristine originals (and, with processed=True, on processed copies too)."""
    X, y = data["X"][:, columns], data["y"]
    accs, proc = [], []
    for tr, va in grouped_folds(data):
        model = ttm.fit(X[tr], y[tr])
        accs.append(float((model.predict(X[va]) == y[va]).mean()))
        if processed:
            cp = np.flatnonzero(np.isin(data["groups"], data["groups"][va]) & data["is_copy"])
            proc.append(float((model.predict(X[cp]) == y[cp]).mean()))
    return (statistics.fmean(accs), statistics.fmean(proc)) if processed else statistics.fmean(accs)


def permutation_oof(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Out-of-fold permutation importance (accuracy and AUC drops) on pristine validation originals."""
    X, y = data["X"], data["y"]
    acc_drops, auc_drops, base = [], [], []
    for tr, va in grouped_folds(data):
        model = ttm.fit(X[tr], y[tr]).set_params(n_jobs=1)
        base.append(float((model.predict(X[va]) == y[va]).mean()))
        for scoring, store in (("accuracy", acc_drops), ("roc_auc", auc_drops)):
            r = permutation_importance(model, X[va], y[va], scoring=scoring, n_repeats=PERM_REPEATS,
                                       random_state=SEED, n_jobs=-1)
            store.append(r.importances_mean)
    return np.mean(acc_drops, axis=0), np.std(acc_drops, axis=0), np.mean(auc_drops, axis=0), statistics.fmean(base)


def partial_dependence_delta(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean tamper score with each feature at its p90 minus at its p10 (other features as observed)."""
    lo, hi = np.percentile(X, 10, axis=0), np.percentile(X, 90, axis=0)
    wide = np.where(np.percentile(X, 99, axis=0) > lo, np.percentile(X, 99, axis=0), X.max(axis=0))
    hi = np.where(hi > lo, hi, wide)  # sparse features: p10 == p90
    deltas = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xl, Xh = X.copy(), X.copy()
        Xl[:, j], Xh[:, j] = lo[j], hi[j]
        deltas[j] = model.predict_proba(Xh)[:, 1].mean() - model.predict_proba(Xl)[:, 1].mean()
    return deltas, lo, hi


def split_share(model, n_features: int) -> np.ndarray:
    counts = np.zeros(n_features)
    for tree in model.estimators_:
        f = tree.tree_.feature
        counts += np.bincount(f[f >= 0], minlength=n_features)
    return counts / counts.sum()


def correlated_pairs(X: np.ndarray, names: list[str]) -> list[tuple[str, str, float]]:
    ranks = rankdata(X, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):  # constant columns have no correlation
        rho = np.nan_to_num(np.corrcoef(ranks, rowvar=False))
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(rho[i, j]) >= CORRELATED:
                pairs.append((names[i], names[j], float(rho[i, j])))
    return sorted(pairs, key=lambda t: -abs(t[2]))


def analyse() -> dict:
    model, meta = DocumentFraudDetector.load(DEFAULT_MODEL_PATH)
    model.set_params(n_jobs=1)
    schema = get_schema(meta.get("feature_schema_version"))
    if not meta.get("training_files"):
        raise SystemExit("The deployed model records no training files (legacy v1); this analysis needs model v2+.")
    names = list(schema.names)

    print("Rebuilding the model's training data (TRAIN + DEV + processed copies)...")
    data = training_matrix()
    X, y = data["X"], data["y"]
    if {d["sha256"] for d in data["docs"]} != set(meta["training_files"].values()):
        raise SystemExit("Training data differs from the model's recorded training files.")

    print("MDI, split share, partial dependence...")
    mdi = model.feature_importances_
    mdi_std = np.std([t.feature_importances_ for t in model.estimators_], axis=0)
    share = split_share(model, len(names))
    impact, lo, hi = partial_dependence_delta(model, X)
    auc_train = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

    print("Out-of-fold permutation importance (grouped CV)...")
    perm, perm_std, perm_auc, base_acc = permutation_oof(data)

    print("Retraining without each feature family...")
    all_cols = list(range(len(names)))
    fam_rows = []
    for fam in FAMILIES:
        cols = [i for i in all_cols if family_of(names[i]) != fam]
        fam_idx = [i for i in all_cols if family_of(names[i]) == fam]
        acc, proc = cv_accuracy(data, cols, processed=True)
        fam_rows.append({"family": fam, "n_features": len(fam_idx), "mdi": float(mdi[fam_idx].sum()),
                         "cv_accuracy_without": acc, "cv_processed_accuracy_without": proc})
    least = [i for i in all_cols if perm[i] < USEFUL_MIN_PERM and mdi[i] < USEFUL_MIN_MDI]
    reduced = cv_accuracy(data, [i for i in all_cols if i not in least]) if least else None

    features = []
    for i, n in enumerate(names):
        features.append({"feature": n, "family": family_of(n), "mdi": float(mdi[i]), "mdi_std_trees": float(mdi_std[i]),
                         "perm_acc_drop": float(perm[i]), "perm_acc_drop_std": float(perm_std[i]),
                         "perm_auc_drop": float(perm_auc[i]), "split_share": float(share[i]),
                         "impact_p10_to_p90": float(impact[i]), "p10": float(lo[i]), "p90": float(hi[i])})
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(),
                  "sha256": hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest(),
                  "version": meta.get("model_version"), "schema": schema.version, "n_features": len(names),
                  "trees": len(model.estimators_)},
        "training_rows": int(len(y)), "training_documents": len(data["docs"]),
        "cv_baseline_accuracy": base_acc,
        **dict(zip(("cv_accuracy_all_features", "cv_processed_accuracy_all_features"),
                   cv_accuracy(data, all_cols, processed=True))),
        "train_auc_deployed": auc_train,
        "features": features, "families": fam_rows,
        "least_useful": [names[i] for i in least], "cv_accuracy_without_least_useful": reduced,
        "correlated_pairs": correlated_pairs(X, names),
    }


def build_report(r: dict) -> str:
    m = r["model"]
    feats = sorted(r["features"], key=lambda f: -f["perm_acc_drop"] - f["mdi"])
    by_perm = sorted(r["features"], key=lambda f: -f["perm_acc_drop"])
    by_mdi = sorted(r["features"], key=lambda f: -f["mdi"])
    L = [
        "--------------------------------",
        "DOCFRAUD FEATURE IMPORTANCE REPORT",
        "--------------------------------",
        f"Generated (UTC): {r['generated']}",
        "Script: scripts/feature_importance.py",
        f"Model: {m['path']} v{m['version']}, schema {m['schema']} ({m['n_features']} features, {m['trees']} trees)",
        f"  SHA-256 {m['sha256']}",
        f"Data: the model's training data only - {r['training_documents']} documents + processed copies = "
        f"{r['training_rows']} rows. Frozen TEST and HOLDOUT are not used.",
        "",
        "Columns:",
        "  MDI      mean decrease in impurity (deployed forest); sums to 1. +/- = spread across trees.",
        f"  Perm     accuracy drop when the feature is shuffled, out-of-fold (grouped {CV_SPLITS}-fold CV, "
        f"{PERM_REPEATS} shuffles, pristine originals). CV baseline accuracy {r['cv_baseline_accuracy']:.3f}.",
        "  AUC      ROC AUC drop for the same shuffles (finer-grained than accuracy).",
        "  Splits   share of the forest's split nodes on the feature.",
        "  Impact   mean tamper score at the feature's p90 minus at its p10 (partial dependence, deployed model);",
        "           + pushes towards tampered, - towards genuine.",
        "",
        "--------------------------------",
        "PER-FEATURE IMPORTANCE (sorted by Perm + MDI)",
        "--------------------------------",
        f"{'#':>2} {'Feature':<32} {'Family':<11} {'MDI':>6} {'+/-':>6} {'Perm':>7} {'AUC':>7} {'Splits':>7} "
        f"{'Impact':>7}  p10 -> p90",
        "-" * 118,
    ]
    for k, f in enumerate(feats, 1):
        L.append(f"{k:>2} {f['feature']:<32} {f['family']:<11} {f['mdi']:>6.3f} {f['mdi_std_trees']:>6.3f} "
                 f"{f['perm_acc_drop']:>+7.3f} {f['perm_auc_drop']:>+7.3f} {f['split_share']:>7.3f} "
                 f"{f['impact_p10_to_p90']:>+7.3f}  {f['p10']:.3g} -> {f['p90']:.3g}")

    L += ["", "--------------------------------", "FEATURE FAMILIES", "--------------------------------",
          f"All features, grouped CV accuracy: pristine {r['cv_accuracy_all_features']:.3f}, processed copies "
          f"{r['cv_processed_accuracy_all_features']:.3f}",
          f"{'Family':<12} {'Features':>8} {'MDI total':>9} {'Pristine w/o':>13} {'Change':>7} {'Processed w/o':>14} "
          f"{'Change':>7}  Detects", "-" * 118]
    for fam in sorted(r["families"], key=lambda x: x["cv_accuracy_without"]):
        change = fam["cv_accuracy_without"] - r["cv_accuracy_all_features"]
        pchange = fam["cv_processed_accuracy_without"] - r["cv_processed_accuracy_all_features"]
        L.append(f"{fam['family']:<12} {fam['n_features']:>8} {fam['mdi']:>9.3f} {fam['cv_accuracy_without']:>13.3f} "
                 f"{100 * change:>+6.1f}p {fam['cv_processed_accuracy_without']:>14.3f} {100 * pchange:>+6.1f}p  "
                 f"{FAMILY_MEANING[fam['family']]}")
    L.append("'w/o' = the model retrained without that family (same folds), scored on pristine originals and on the "
             "processed copies of the validation documents. A large drop means the family carries information the "
             "others cannot replace.")

    L += ["", "--------------------------------", "MOST USEFUL FEATURES", "--------------------------------"]
    L += [f"  {f['feature']:<32} perm {f['perm_acc_drop']:+.3f}, MDI {f['mdi']:.3f}, impact {f['impact_p10_to_p90']:+.3f}"
          for f in by_perm[:5]]
    L.append("  Highest MDI: " + ", ".join(f"{f['feature']} {f['mdi']:.3f}" for f in by_mdi[:5]))
    L += ["", "--------------------------------", "LEAST USEFUL FEATURES", "--------------------------------",
          f"Rule: permutation accuracy drop < {USEFUL_MIN_PERM} and MDI < {USEFUL_MIN_MDI}."]
    least = [f for f in r["features"] if f["feature"] in r["least_useful"]]
    L += [f"  {f['feature']:<32} perm {f['perm_acc_drop']:+.3f}, MDI {f['mdi']:.3f}, splits {f['split_share']:.3f}"
          for f in sorted(least, key=lambda f: f["mdi"])] or ["  none"]
    if r["cv_accuracy_without_least_useful"] is not None:
        L.append(f"CV accuracy without these {len(least)}: {r['cv_accuracy_without_least_useful']:.3f} "
                 f"(all features: {r['cv_accuracy_all_features']:.3f}).")
    L += ["Low single-feature importance can mean the information is shared with a correlated feature, not that "
          "it is absent; see the family check above and the pairs below."]

    L += ["", "--------------------------------", f"STRONGLY CORRELATED PAIRS (|Spearman| >= {CORRELATED})",
          "--------------------------------"]
    L += [f"  {a:<32} {b:<32} {rho:+.3f}" for a, b, rho in r["correlated_pairs"]] or ["  none"]

    L += ["", "--------------------------------", "INTERPRETATION AND RECOMMENDATIONS", "--------------------------------"]
    L += recommendations(r)
    L.append("")
    return "\n".join(L)


def recommendations(r: dict) -> list[str]:
    fams = sorted(r["families"], key=lambda x: x["cv_accuracy_without"])
    base = r["cv_accuracy_all_features"]
    out = [f"- The {fams[0]['family']} family matters most: without it CV accuracy falls "
           f"{100 * (base - fams[0]['cv_accuracy_without']):.1f} points. "
           + ", ".join(f"{f['family']} {100 * (base - f['cv_accuracy_without']):+.1f}" for f in fams[1:])
           + " points for the others (positive = accuracy lost)."]
    pbase = r["cv_processed_accuracy_all_features"]
    for fam in r["families"]:
        d_pr = 100 * (fam["cv_accuracy_without"] - base)
        d_pc = 100 * (fam["cv_processed_accuracy_without"] - pbase)
        if abs(d_pr) < 1 and abs(d_pc) < 1:
            out.append(f"- The {fam['family']} family adds nothing measurable (without it: pristine {d_pr:+.1f}, "
                       f"processed {d_pc:+.1f} points) despite an MDI total of {fam['mdi']:.3f}; other features take "
                       "over its splits. A removal candidate for the next version, decided by CV on training data.")
        elif d_pr <= -1 and d_pc >= 0:
            out.append(f"- The {fam['family']} family helps on pristine files ({d_pr:+.1f} points without it) but not "
                       f"on processed copies ({d_pc:+.1f}): its signal does not survive re-saving / resizing.")
    neg = [f["feature"] for f in r["features"] if f["impact_p10_to_p90"] < -0.02]
    if neg:
        out.append("- Features whose high values push towards 'genuine': " + ", ".join(neg)
                   + ". Most are page-wide measures (share of anomalous tiles, fingerprint) or region fill ratios "
                     "(1.0 for tiny layout specks). Partial dependence shows the direction of the model's response, "
                     "not its cause.")
    if r["least_useful"]:
        red = r["cv_accuracy_without_least_useful"]
        verdict = "no measurable loss" if red is not None and red >= base - 0.01 else "a measurable loss"
        out.append(f"- {len(r['least_useful'])} feature(s) add little on their own ({', '.join(r['least_useful'])}); "
                   f"removing them gives {verdict} in CV ({red:.3f} vs {base:.3f}). Candidates to drop in the next "
                   "version to simplify the model, decided by CV on training data, never on TEST / HOLDOUT.")
    if r["correlated_pairs"]:
        pairs = "; ".join(f"{a} ~ {b}" for a, b, _ in r["correlated_pairs"][:3])
        out.append(f"- {len(r['correlated_pairs'])} strongly correlated pair(s), e.g. {pairs}: threshold / summary "
                   "variants of the same region measure overlap. Keep one variant per signal if the model is simplified.")
    out.append("- Importance here explains this model on this synthetic data. It is not evidence that the same "
               "features matter on real documents.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Feature importance of the deployed model (training data only)")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()
    result = analyse()
    report = build_report(result)
    for path, text in ((args.report, report), (args.json, json.dumps(result, indent=1) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}\nResults written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
