"""
scripts/model_accuracy_eval.py
Founder A (Day 10): final accuracy check of the deployed model on the held-out sets.

Usage:
    python scripts/model_accuracy_eval.py
    python scripts/model_accuracy_eval.py --report <path> --json <path>

Nothing is trained, tuned or selected here. The decision rule is the server's:
tampered if tamper_risk_score >= TAMPER_THRESHOLD (0.5).

Steps:
  1. Check that the frozen split is intact (15 TEST images, SHA-256) and that no held-out document
     is among the model's recorded training files.
  2. Score the frozen TEST set (15 images) and the generated HOLDOUT set (149 documents) through
     ModelAPI.predict, the /verify pipeline. Compute accuracy, precision, recall and F1 (tampered is
     the positive class), FPR, FNR, confusion matrices and 95% Wilson intervals per set and combined.
  3. Compare with every earlier recorded result: model v1 (Day 5, 6 and 8) and model v2 (the Day 8
     validation JSON and the Day 9 review JSON, when they were made with this model file).
  4. Check the release requirements: held-out accuracy above 75% on both sets (Day 8 requirement),
     the 78% sprint target, and metrics identical to the recorded v2 evaluation (a deterministic
     model scoring unchanged files must reproduce them).

Writes logs/final_accuracy_report.txt and logs/final_accuracy_results.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_validation as fv  # noqa: E402
import model_stability_test as mst  # noqa: E402
import model_validation as mv  # noqa: E402

from backend import config  # noqa: E402
from backend.model_utils import ModelAPI  # noqa: E402
from core.model_trainer import DEFAULT_MODEL_PATH, DocumentFraudDetector  # noqa: E402

DEFAULT_REPORT = ROOT / "logs" / "final_accuracy_report.txt"
DEFAULT_JSON = ROOT / "logs" / "final_accuracy_results.json"
RECORDED_V2 = [  # earlier v2 evaluations, used only when made with the same model file
    ("Day 8 validation", ROOT / "logs" / "model_validation_results.json", ("heldout", "sets")),
    ("Day 9 review", ROOT / "logs" / "model_review_results.json", ("performance", "sets")),
]
REQUIRED_ACCURACY = mv.REQUIRED_ACCURACY  # 0.75, Day 8 requirement
SPRINT_TARGET = mv.SPRINT_TARGET  # 0.78
NEAR = 0.05  # "near the threshold" = |score - threshold| <= NEAR
SETS = ("TEST", "HOLDOUT")
CM_KEYS = ("tn", "fp", "fn", "tp")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── Comparison with earlier records ─────────────────────────────────────────

def recorded_v2(model_sha: str) -> list[dict]:
    """Held-out metrics recorded by earlier v2 evaluations of this exact model file."""
    out = []
    for label, path, keys in RECORDED_V2:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model", {}).get("sha256") != model_sha:
            continue
        sets = data
        for k in keys:
            sets = sets.get(k) or {}
        for name in SETS:
            if name in sets:
                m = sets[name]["metrics"]
                out.append({"when": label, "model": "v2 (local-tamper-v2)", "set": name, "n": m["n"],
                            "correct": m["correct"], "accuracy": m["accuracy"], "precision": m["precision"],
                            "recall": m["recall"], "f1": m["f1"], "cm": [m[k] for k in CM_KEYS],
                            "source": f"{path.relative_to(ROOT).as_posix()} (generated {data.get('generated', '?')[:10]})"})
    return out


def previous_records(model_sha: str) -> list[dict]:
    v1 = mv.PREVIOUS[:2] + [x for x in [mv.day6_record()] if x] + mv.PREVIOUS[2:]
    return [{k: v for k, v in r.items() if k != "per_image_risk"} for r in v1] + recorded_v2(model_sha)


# ── Requirements ────────────────────────────────────────────────────────────

def requirement_checks(r: dict) -> list[dict]:
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    add("Frozen TEST split intact", r["frozen_split_ok"], "15/15 SHA-256 and dataset fingerprint match")
    add("No held-out document used in training", r["leakage_ok"],
        "; ".join(f"{name}: {detail}" for name, _, detail in r["leakage"][1:3]))
    for name in SETS:
        m = r["sets"][name]["metrics"]
        add(f"{name} accuracy above {REQUIRED_ACCURACY:.0%}", m["accuracy"] > REQUIRED_ACCURACY,
            f"{mv.pct(m['accuracy'])} ({m['correct']}/{m['n']})")
    for name in SETS:
        m = r["sets"][name]["metrics"]
        add(f"{name} accuracy meets the {SPRINT_TARGET:.0%} sprint target", m["accuracy"] >= SPRINT_TARGET,
            mv.pct(m["accuracy"]))
    same = r["same_as_recorded"]
    add("Metrics identical to the recorded v2 evaluation", same is True,
        "confusion matrices equal to " + ", ".join(sorted({p["when"] for p in r["previous"] if "cm" in p}))
        if same else ("no earlier v2 record for this model file" if same is None else "DIFFERENT - investigate"))
    return checks


def near_threshold(per_image: list[dict], band: float = NEAR) -> dict:
    near = [p for p in per_image if abs(p["risk"] - config.TAMPER_THRESHOLD) <= band]
    return {"band": band, "n": len(near), "wrong": sum(p["actual"] != p["predicted"] for p in near),
            "documents": [{"path": p["path"], "actual": p["actual"], "risk": p["risk"]} for p in near]}


# ── Run ─────────────────────────────────────────────────────────────────────

def evaluate(api: ModelAPI | None = None) -> dict:
    split = json.loads(fv.SPLIT_PATH.read_text(encoding="utf-8"))
    with open(fv.DATASET / "index.csv", newline="", encoding="utf-8") as f:
        index = {r["path"]: r for r in csv.DictReader(f)}
    integrity = fv.check_integrity(split)  # stops if a frozen TEST image changed
    model_sha = sha256(DEFAULT_MODEL_PATH)
    _, meta = DocumentFraudDetector.load(DEFAULT_MODEL_PATH)
    if api is None:
        api = ModelAPI(DEFAULT_MODEL_PATH)
        api.load_model()
    rv = mv.heldout_evaluation(api, meta, split, index)
    previous = previous_records(model_sha)
    recorded = [p for p in previous if "cm" in p]
    same = None
    if recorded:
        same = all(p["cm"] == [rv["sets"][p["set"]]["metrics"][k] for k in CM_KEYS] for p in recorded)
    per_image = [p for name in SETS for p in rv["sets"][name]["per_image"]]
    r = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": mst.environment(),
        "model": {"path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(), "sha256": model_sha,
                  "version": meta.get("model_version"), "schema": meta.get("feature_schema_version")},
        "threshold": config.TAMPER_THRESHOLD,
        "frozen_split": [{"check": n, "pass": ok, "detail": d} for n, ok, d in integrity],
        "frozen_split_ok": all(ok for _, ok, _ in integrity),
        "leakage": rv["leakage"],
        "leakage_ok": rv["leakage_ok"],
        "sets": rv["sets"],
        "combined": rv["combined"],
        "test_table": rv["test_table"],
        "errors": mv.error_analysis(rv),
        "near_threshold": near_threshold(per_image),
        "previous": previous,
        "same_as_recorded": same,
    }
    r["requirements"] = requirement_checks(r)
    r["requirements_met"] = all(c["pass"] for c in r["requirements"])
    return r


# ── Report ──────────────────────────────────────────────────────────────────

def _metrics_block(name: str, res: dict) -> list[str]:
    m, ci = res["metrics"], res["ci95"]
    L = [f"{name} ({m['n']} documents: {m['tn'] + m['fp']} genuine, {m['tp'] + m['fn']} tampered)",
         f"  Accuracy   {mv.pct(m['accuracy']):>8}  ({m['correct']}/{m['n']}; 95% CI "
         f"{mv.pct(ci['accuracy'][0])} - {mv.pct(ci['accuracy'][1])})"]
    if "precision" in ci:
        L += [f"  Precision  {mv.pct(m['precision']):>8}  (95% CI {mv.pct(ci['precision'][0])} - {mv.pct(ci['precision'][1])})",
              f"  Recall     {mv.pct(m['recall']):>8}  (95% CI {mv.pct(ci['recall'][0])} - {mv.pct(ci['recall'][1])})"]
    else:
        L += [f"  Precision  {mv.pct(m['precision']):>8}", f"  Recall     {mv.pct(m['recall']):>8}"]
    L += [f"  F1 score   {m['f1']:>8.4f}",
          f"  FPR        {mv.pct(m['fpr']):>8}  (genuine flagged)    FNR {mv.pct(m['fnr'])} (tampered missed)",
          "  Confusion matrix:", *("    " + line for line in fv.ascii_confusion(m).splitlines())]
    return L


def build_report(r: dict) -> str:
    L = [
        "=" * 78,
        "DOCFRAUD - FINAL ACCURACY CHECK (Day 10, Founder A)",
        "=" * 78,
        f"Generated (UTC): {r['generated']}",
        "Script         : scripts/model_accuracy_eval.py",
        f"Model          : {r['model']['path']} v{r['model']['version']} ({r['model']['schema']})",
        f"SHA-256        : {r['model']['sha256']}",
        f"Decision rule  : tampered if tamper_risk_score >= {r['threshold']} (TAMPER_THRESHOLD; not tuned)",
        "Scored through : ModelAPI.predict (the /verify pipeline), one pass per document",
        "Positive class : tampered",
        f"Result         : release requirements {'MET' if r['requirements_met'] else 'NOT MET'}",
        "",
        "1. RELEASE REQUIREMENTS",
        "-" * 78,
    ]
    L += [f"  [{mv._mark(c['pass'])}] {c['check']:<52} {c['detail']}" for c in r["requirements"]]
    L += ["", "2. METRICS", "-" * 78]
    for name in SETS:
        L += _metrics_block(name, r["sets"][name]) + [""]
    L += _metrics_block("COMBINED (TEST + HOLDOUT)", r["combined"])
    L += ["", "  Per class:"]
    for name in SETS:
        for label, pc in r["sets"][name]["per_class"].items():
            L.append(f"    {name:<8} {label:<9} precision {mv.pct(pc['precision']):>8}  recall "
                     f"{mv.pct(pc['recall']):>8}  F1 {pc['f1']:.4f}  support {pc['support']}")
    L += ["", "  Accuracy by document type:"]
    for name in SETS:
        L.append(f"    {name:<8} " + ", ".join(f"{t} {a}/{n}" for t, (a, n) in r["sets"][name]["accuracy_by_doc_type"].items()))
    L += ["  Tampered documents caught, by edit method:"]
    for name in SETS:
        L.append(f"    {name:<8} " + ", ".join(f"{t} {a}/{n}" for t, (a, n) in r["sets"][name]["recall_by_method"].items()))

    L += ["", "3. COMPARISON WITH PREVIOUS METRICS", "-" * 78,
          f"  {'When':<17} {'Model':<27} {'Set':<12} {'n':>4} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7}"]
    rows = r["previous"] + [{"when": "Day 10 (this run)", "model": "v2 (local-tamper-v2)", "set": n,
                             **r["sets"][n]["metrics"]} for n in SETS]
    for p in rows:
        L.append(f"  {p['when']:<17} {p['model']:<27} {p['set']:<12} {p['n']:>4} {mv.pct(p['accuracy']):>9} "
                 f"{mv.pct(p['precision']):>10} {mv.pct(p['recall']):>8} {p['f1']:>7.4f}")
    L += ["  Sources:"] + [f"    {p['when']} {p['set']}: {p['source']}" for p in r["previous"]]
    same = r["same_as_recorded"]
    L.append("  Change since the recorded v2 evaluation: " + (
        "none - identical confusion matrices (deterministic model, unchanged files)." if same
        else "no earlier v2 record for this model file." if same is None else "DIFFERENT - investigate before release."))
    t = r["sets"]["TEST"]["metrics"]
    L.append(f"  Against model v1 on the same 15 TEST images: 60.00% (9/15) -> {mv.pct(t['accuracy'])} "
             f"({t['correct']}/{t['n']}).")

    L += ["", "4. ERRORS", "-" * 78]
    fps = [p for n in SETS for p in r["sets"][n]["per_image"] if p["actual"] == "genuine" and p["predicted"] == "tampered"]
    L.append(f"  False positives ({len(fps)}): " + ", ".join(f"{p['path']} {p['risk']:.2f}" for p in fps))
    e = r["errors"]
    if e.get("available"):
        kinds: dict[str, list[str]] = {}
        for x in e["missed"]:
            kinds.setdefault(x["kind"], []).append(f"{x['method']} {x['risk']:.2f}")
        L.append(f"  False negatives ({len(e['missed'])}), by what the edit left on the page:")
        L += [f"    {k}: {len(v)} ({', '.join(v)})" for k, v in kinds.items()]
    nt = r["near_threshold"]
    L.append(f"  Near the threshold (|score - {r['threshold']}| <= {nt['band']}): {nt['n']} documents, "
             f"{nt['wrong']} of them wrong.")
    L += ["", "  Frozen TEST set, per image:", *("    " + line for line in r["test_table"].splitlines())]

    L += ["", "5. NOTES", "-" * 78,
          "  - TEST has only 15 images: its 95% interval runs well below 75%. HOLDOUT (149 documents, generated",
          "    after the model was saved) is the main evidence.",
          "  - Both sets come from the same synthetic generator as the training data. Accuracy on real",
          "    documents is unknown, and degraded files score much lower (logs/model_robustness_report.txt).",
          "  - TEST and HOLDOUT have been scored several times since Day 8 but never used to train, tune or",
          "    choose anything. A new model needs a new holdout set.",
          ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Final accuracy check of the deployed model on the held-out sets")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()
    logging.getLogger("backend").setLevel(logging.WARNING)
    r = evaluate()
    report = build_report(r)
    for path, text in ((args.report, report), (args.json, json.dumps(r, indent=1, default=str) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}\nResults written: {args.json}")
    return 0 if r["requirements_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
