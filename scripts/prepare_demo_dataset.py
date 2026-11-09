"""
scripts/prepare_demo_dataset.py
Day 7: build the 5-image demo set in data/demo/ and score it ONCE with the
deployed model through ModelAPI (the same path the API uses).

Usage:
    python scripts/prepare_demo_dataset.py --dry-run   # selection + integrity only, no scoring
    python scripts/prepare_demo_dataset.py             # copy, verify, score once, write docs
    python scripts/prepare_demo_dataset.py --doc-only  # rewrite DEMO_IMAGES.md from the saved record
    python scripts/prepare_demo_dataset.py --rescore   # score the same 5 files once with a NEW model

Selection rules are fixed in code BEFORE scoring and never look at predictions:
  genuine_payslip_01/02 : first 2 Day-1 payslips (data/synthetic/payslip_*.png)
                          that are not a near-copy of any of the 75 dataset images.
                          The Day-2 generator reused the Faker seed, so some Day-1
                          records reappear in the dataset (payslip_001 is a near-copy
                          of training image genuine/payslip_001). Genuine by
                          construction: the Day-1 generator (commit bf18a4c) has no
                          tampering code.
  tampered_payslip_01/02: first 2 train-aug-v2.1 tampered payslip views whose only
                          edit is a local text-field replacement (no whole-page
                          processing). Label + edit from the augmentation manifest.
  edge_case_01          : first train-aug-v2.1 genuine payslip view whose only
                          change is whole-page JPEG compression.

Files are copied byte-for-byte (SHA-256 verified). The frozen test set is only
read for hash / pixel overlap checks. Re-running after scoring is refused so the
demo record is scored exactly once per model.

--rescore (Day 8, model v2): when the deployed model changes, the same 5 files (SHA-256
checked against the record) are scored once with the new model. The previous model block
and scores move to "scoring_history" in demo_manifest.json; selection and integrity records
are kept. Refused if the record already belongs to the deployed model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from backend.model_utils import ModelAPI
from core.model_trainer import DEFAULT_MODEL_PATH

SYNTH = ROOT / "data" / "synthetic"
AUG = ROOT / "data" / "augmented" / "train_v2_1"
SPLIT = json.loads((ROOT / "data" / "splits" / "day5_frozen_split.json").read_text(encoding="utf-8"))
DEMO = ROOT / "data" / "demo"
MANIFEST_OUT = DEMO / "demo_manifest.json"
DOC_OUT = DEMO / "DEMO_IMAGES.md"
# Near-copy = fewer than 0.4% of grayscale pixels differ by > 40 levels.
# Measured: same record / augmented view of it 0.0000-0.0025; different documents >= 0.0056.
NEAR_COPY_FRAC = 0.004
# Visually verified record relations for Day-1 payslips (header, employee, figures).
DAY1_RELATION = {
    "payslip_002.png": "Sibling record: training image genuine/payslip_002.png has the same company, employee, "
                       "employee ID and bank account (different month and salary figures).",
    "payslip_003.png": "No matching record in the 75-image dataset (nearest by pixels: genuine/payslip_006.png, "
                       "a different company and employee).",
}
# Visually verified description of each demo image (original vs demo copy compared by eye).
VISIBLE_CHANGE = {
    "genuine_payslip_01.png": "None - untouched generator output.",
    "genuine_payslip_02.png": "None - untouched generator output.",
    "tampered_payslip_01.png": "Earnings table: the Medical Allowance value '₹ 1,568' is overwritten with the "
                               "14-digit string '73437992415073' in a different font. Visible on close inspection; "
                               "not a plausible salary edit.",
    "tampered_payslip_02.png": "Employee details: the 'Employee Name:' label is overwritten with a blurred "
                               "'2,290,395' (text replacement + local resampling). Visible; not a plausible "
                               "salary edit.",
    "edge_case_01.png": "Whole page JPEG-compressed (quality 78); no content change. File is ~4.5x larger "
                        "than the other PNGs because JPEG noise compresses poorly as PNG.",
}
RISK_THRESHOLD = 0.5  # ModelAPI's own cut-off (tamper_risk_score >= 0.5 => tampering); not tuned


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.int16)


def diff_frac(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a - b) > 40).mean()) if a.shape == b.shape else 1.0


def select() -> list[dict]:
    dataset = [gray(p) for lab in ("genuine", "tampered") for p in sorted((SYNTH / lab).glob("*.png"))]
    day1 = [p for p in sorted(SYNTH.glob("payslip_*.png"))
            if min(diff_frac(gray(p), d) for d in dataset) >= NEAR_COPY_FRAC][:2]
    assert [p.name for p in day1] == list(DAY1_RELATION), [p.name for p in day1]
    genuine = [{
        "demo_file": f"genuine_payslip_{i:02d}.png", "role": "genuine payslip", "source": p,
        "actual_label": "genuine",
        "label_source": "Day-1 generator output (scripts/generate_tampered_documents.py @ bf18a4c, "
                        "generate_payslip_batch: render only, no tampering code); PROGRESS.md Day 1",
        "provenance": f"data/synthetic/{p.name} - Founder C Day-1 synthetic payslip, commit bf18a4c",
        "training_overlap": "Image not in any training set. " + DAY1_RELATION[p.name],
    } for i, p in enumerate(day1, 1)]

    rows = sorted(csv.DictReader(open(AUG / "manifest.csv", newline="", encoding="utf-8")), key=lambda r: r["file"])
    payslip = [r for r in rows if r["source_path"].startswith("genuine/payslip_")]
    tampered_rows, seen = [], set()
    for r in payslip:
        if (r["label"] == "tampered" and "text_replace" in r["transform"] and r["global_transform"] == "none"
                and r["source_path"] not in seen):
            tampered_rows.append(r)
            seen.add(r["source_path"])
    edge_rows = [r for r in payslip if r["label"] == "genuine" and r["global_transform"] == "global_jpeg"]

    def aug_entry(r, demo_file, role, extra=""):
        return {
            "demo_file": demo_file, "role": role, "source": AUG / r["file"], "actual_label": r["label"],
            "label_source": f"data/augmented/train_v2_1/manifest.csv (aug_version {r['aug_version']})",
            "provenance": (f"data/augmented/train_v2_1/{r['file']} - view of training document {r['source_path']} "
                           f"(source sha256 {r['source_sha256'][:12]}...); local edit: {r['transform']} {r['params']} "
                           f"bbox {r['bbox']}; whole-page: {r['global_transform']} {r['global_params']}{extra}"),
            "manifest_sha256": r["sha256"], "source_path": r["source_path"], "source_sha256": r["source_sha256"],
            "transform": r["transform"], "global_transform": r["global_transform"],
            "global_params": r["global_params"], "bbox": r["bbox"],
            "training_overlap": (f"Image not in any training set of the deployed model, but its source document "
                                 f"{r['source_path']} is a training image of it; only the "
                                 f"{'edit' if r['label'] == 'tampered' else 'compression'} is new."),
        }

    tampered = [aug_entry(r, f"tampered_payslip_{i:02d}.png", "tampered payslip")
                for i, r in enumerate(tampered_rows[:2], 1)]
    edge = [aug_entry(edge_rows[0], "edge_case_01.png", "edge case")]
    edge[0]["edge_case_type"] = f"JPEG-compressed page (whole-page {edge_rows[0]['global_params']}), stored as PNG"

    sel = genuine + tampered + edge
    assert [s["actual_label"] for s in sel] == ["genuine", "genuine", "tampered", "tampered", "genuine"], sel
    return sel


def integrity(sel: list[dict]) -> dict[str, tuple[bool, str]]:
    test_paths = {r["path"] for r in SPLIT["test"]}
    test_hashes = {r["sha256"] for r in SPLIT["test"]}
    dataset_hashes = {sha256(p) for lab in ("genuine", "tampered") for p in (SYNTH / lab).glob("*.png")}
    checks: dict[str, tuple[bool, str]] = {}

    bad_test = [r["path"] for r in SPLIT["test"] if sha256(SYNTH / r["path"]) != r["sha256"]]
    checks["frozen test set unchanged (15/15 SHA-256)"] = (len(SPLIT["test"]) == 15 and not bad_test, ", ".join(bad_test))

    src_hashes = {s["demo_file"]: sha256(s["source"]) for s in sel}
    overlap = [f for f, h in src_hashes.items() if h in test_hashes]
    overlap += [s["demo_file"] for s in sel if s.get("source_path") in test_paths or s.get("source_sha256") in test_hashes]
    checks["no demo file or source document in frozen test set"] = (not overlap, ", ".join(overlap))

    # Near-copy guard: decoded pixels vs every frozen test image.
    test_px = [(r["path"], gray(SYNTH / r["path"])) for r in SPLIT["test"]]
    near, closest = [], []
    for s in sel:
        a = gray(s["source"])
        fr = sorted((diff_frac(a, b), path) for path, b in test_px)
        closest.append(f"{s['demo_file']}:{fr[0][0]:.4f}")
        near += [f"{s['demo_file']}~{path}" for f, path in fr if f < NEAR_COPY_FRAC]
    checks[f"no pixel near-copy of a frozen test image (diff fraction < {NEAR_COPY_FRAC})"] = (
        not near, ", ".join(near) or "closest " + ", ".join(closest))

    mismatch = [s["demo_file"] for s in sel if "manifest_sha256" in s and src_hashes[s["demo_file"]] != s["manifest_sha256"]]
    checks["augmented sources match manifest SHA-256"] = (not mismatch, ", ".join(mismatch))
    seen = [s["demo_file"] for s in sel if src_hashes[s["demo_file"]] in dataset_hashes]
    checks["no demo image is one of the 75 training/test originals"] = (not seen, ", ".join(seen))
    checks["label provenance recorded for every image"] = (all(s["label_source"] for s in sel), "")
    counts = [sum(s["role"] == r for s in sel) for r in ("genuine payslip", "tampered payslip", "edge case")]
    checks["5 images: 2 genuine / 2 tampered payslips / 1 edge case"] = (counts == [2, 2, 1], str(counts))
    return checks


VALIDATION_JSON = ROOT / "logs" / "model_validation_results.json"


def heldout_summary(model: dict) -> str:
    """Held-out accuracy of the recorded model, from logs/model_validation_results.json when it matches."""
    if model.get("model_version", "").startswith("1."):
        return "the Day-6 held-out result (60%, 9/15)"
    try:
        res = json.loads(VALIDATION_JSON.read_text(encoding="utf-8"))
        if res["model"]["sha256"] == model["sha256"] and res.get("heldout"):
            t, h = res["heldout"]["sets"]["TEST"]["metrics"], res["heldout"]["sets"]["HOLDOUT"]["metrics"]
            return (f"the model v2 held-out results: {100 * t['accuracy']:.2f}% ({t['correct']}/{t['n']}) on the "
                    f"frozen test set and {100 * h['accuracy']:.2f}% ({h['correct']}/{h['n']}) on the generated "
                    "holdout set (`logs/model_validation_report.txt`)")
    except (OSError, KeyError, ValueError):
        pass
    return "the held-out results in `logs/model_validation_report.txt`"


def write_doc(records: list[dict], checks: dict, model: dict, history: list[dict] | None = None) -> None:
    ok = lambda name: "PASS" if checks[name][0] else "FAIL"
    v1 = model.get("model_version", "").startswith("1.")
    L = [
        "# DocFraud Demo Dataset", "",
        "## Purpose", "",
        "Five images for API / frontend demonstration only. They show what the current "
        "pipeline returns for a clean payslip, an edited payslip and a compressed page. "
        "**This is not a validation set.** Five images cannot measure accuracy; the valid "
        f"measurement is {heldout_summary(model)}. Predictions below are "
        "recorded as returned, including wrong ones.", "",
        "All documents are synthetic and watermarked \"SYNTHETIC / TEST DATA\". No real PII.", "",
        "## Demo Images", "",
    ]
    titles = ["Genuine Payslip 01", "Genuine Payslip 02", "Tampered Payslip 01", "Tampered Payslip 02", "Edge Case"]
    for i, (t, r) in enumerate(zip(titles, records), 1):
        L += [
            f"### {i}. {t}",
            f"- File: `{r['demo_file']}`",
            f"- Actual label: **{r['actual_label']}** (from {r['label_source']})",
            f"- Prediction: **{r['predicted_label']}**",
            f"- Risk score: {r['tamper_risk_score']:.4f} (confidence {r['confidence']:.4f})",
            f"- Correct: {'YES' if r['correct'] else 'NO'}",
            f"- Dimensions: {r['width']}x{r['height']} {r['mode']}, {r['file_bytes']:,} bytes",
            f"- Provenance: {r['provenance']}",
            f"- SHA-256: `{r['sha256']}`",
            f"- Visible change: {VISIBLE_CHANGE[r['demo_file']]}",
            f"- Relation to training data: {r['training_overlap']}",
        ]
        if "edge_case_type" in r:
            L.append(f"- Edge-case type: {r['edge_case_type']}")
        L.append("")
    n_ok = sum(r["correct"] for r in records)
    L += [
        f"Demo predictions correct: {n_ok}/5 (not an accuracy estimate).", "",
        "## Selection Rules", "",
        "Fixed in `scripts/prepare_demo_dataset.py` before scoring; predictions played no part.", "",
        "- Genuine: first two Day-1 payslips (`data/synthetic/payslip_*.png`, outside `index.csv`) that are not a "
        f"near-copy of any of the 75 dataset images (< {NEAR_COPY_FRAC:.1%} of pixels differing). `payslip_001.png` "
        "is excluded: the Day-2 generator reused the Faker seed and it is a near-copy of training image "
        "`genuine/payslip_001.png`. Selected: `payslip_002.png`, `payslip_003.png`.",
        "- Tampered: first two `train-aug-v2.1` tampered payslip views whose only change is a local text "
        "replacement (`text_replace_*`, no whole-page processing). The augmenter writes the new value over a "
        "detected text token, which is not always the field the edit type names - see \"Visible change\".",
        "- Edge case: first `train-aug-v2.1` genuine payslip view whose only change is whole-page JPEG "
        "compression (typical of forwarded or re-saved scans).", "",
        "## Integrity Checks", "",
        f"- Frozen test overlap: {'PASS' if checks['no demo file or source document in frozen test set'][0] and checks[f'no pixel near-copy of a frozen test image (diff fraction < {NEAR_COPY_FRAC})'][0] else 'FAIL'} "
        "(file hashes, source documents, pixel near-copies)",
        f"- Hash verification: {'PASS' if checks['augmented sources match manifest SHA-256'][0] and checks['frozen test set unchanged (15/15 SHA-256)'][0] else 'FAIL'} "
        "(demo copies byte-identical to sources; augmented sources match manifest; 15/15 frozen test hashes unchanged)",
        f"- Label provenance: {ok('label provenance recorded for every image')}",
        "- Demo count: 5",
        "- Genuine payslips: 2",
        "- Tampered payslips: 2",
        "- Edge cases: 1", "",
        "Full check list: `demo_manifest.json`.", "",
        "## Model Information", "",
        f"- Artifact: `{model['path']}` (SHA-256 `{model['sha256'][:16]}...`)",
        f"- Version: {model['model_version']}, feature schema {model['feature_schema_version']} "
        f"({model['feature_count']} features, "
        f"`{'core.document_analyzer.extract_features' if v1 else 'core.tamper_features.extract_tamper_features'}`)",
        f"- RandomForest {model['rf_params']}; trained {model['trained_at']} on {model['n_training_samples']} "
        + ("images" if v1 else "documents (60 frozen TRAIN + 150 generated DEV, plus processed copies)"),
        "- Scoring path: `backend.model_utils.ModelAPI.predict` (same as `/verify`)",
        f"- Predicted label: tampered if `tamper_risk_score >= {RISK_THRESHOLD}` (ModelAPI's rule, not tuned)",
        f"- Scored once at {model['scored_at']} (UTC)", "",
    ]
    if history:
        L += ["## Scoring History", "",
              "Earlier models scored the same five files once each (kept in `demo_manifest.json`):", ""]
        for h in history:
            m, n_ok = h["model"], sum(x["correct"] for x in h["images"])
            L.append(f"- Model {m['model_version']} ({m['feature_schema_version']}, SHA-256 `{m['sha256'][:16]}...`), "
                     f"scored {m['scored_at']}: {n_ok}/5 correct - "
                     + ", ".join(f"{x['demo_file']} {x['tamper_risk_score']:.2f}" for x in h["images"]))
        L.append("")
    L += ["## Limitations", ""]
    if v1:
        L += [
            "- The deployed `fraud_model.pkl` was trained on all 75 dataset images, including the 15 frozen "
            "test images; its 15/15 on the test set is leakage, not performance. Held-out accuracy is **60%**.",
            "- None of the 5 files was a training image, but the deployed model has seen related data: images 3-5 "
            "are edited / compressed views of its training documents, and genuine payslip 01 is a sibling of "
            "training image `genuine/payslip_002.png` (same employee record, different month and figures). "
            "Only genuine payslip 02 has no related training record.",
        ]
    else:
        L += [
            "- None of the 5 files is a training image of model v2, but images 3-5 are edited / compressed views "
            "of its training documents (frozen TRAIN images), and genuine payslip 01 is a sibling of training "
            "image `genuine/payslip_002.png`. Only genuine payslip 02 has no related training record.",
            "- Model v2 was trained on the generator's four edit types (blur, colour shift, noise patch, text "
            "overlay). The two tampered demo payslips use a different edit (text replacement from the Day-5 "
            "augmenter), so they test generalisation to an edit type the model has not seen.",
        ]
    L += [
        "- The two tampered payslips are Founder A's Day-5 training augmentations, not Founder C's dataset "
        "tampering methods. Their edits land on the wrong fields (an account-style number in an allowance "
        "cell, an amount over a field label), so they are not realistic fraud scenarios.",
        "- Risk score is an uncalibrated RandomForest vote fraction, not a fraud probability.",
        "- `/verify` returns no flagged regions, so the demo cannot show where an edit is.",
        "- Synthetic documents only; real NBFC documents may behave differently.",
        "- Not production-ready.", "",
    ]
    DOC_OUT.write_text("\n".join(L), encoding="utf-8")


def rescore() -> int:
    """Score the recorded demo files once with the deployed model; keep the previous record as history."""
    saved = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
    current_sha = sha256(DEFAULT_MODEL_PATH)
    if saved["model"]["sha256"] == current_sha:
        raise SystemExit("The demo record already belongs to the deployed model. Refusing to score again.")
    for r in saved["images"]:
        if sha256(DEMO / r["demo_file"]) != r["sha256"]:
            raise SystemExit(f"STOP: {r['demo_file']} differs from its record; nothing scored.")

    api = ModelAPI(DEFAULT_MODEL_PATH)
    api.load_model()
    md = api.metadata
    model = {
        "path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(), "sha256": current_sha,
        "model_version": md.get("model_version"), "feature_schema_version": md.get("feature_schema_version"),
        "feature_count": md.get("feature_count"), "rf_params": md.get("rf_params"),
        "trained_at": md.get("trained_at"), "n_training_samples": md.get("n_training_samples"),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    keys = ("tamper_risk_score", "confidence", "predicted_label", "correct", "interpretation")
    history = saved.get("scoring_history", []) + [{
        "model": saved["model"],
        "images": [{"demo_file": r["demo_file"], **{k: r[k] for k in keys}} for r in saved["images"]],
    }]
    for r in saved["images"]:
        out = api.predict(DEMO / r["demo_file"])  # scored exactly once with this model
        risk = out["tamper_risk_score"]
        pred = "tampered" if risk >= RISK_THRESHOLD else "genuine"
        r.update(tamper_risk_score=risk, confidence=out["confidence"], predicted_label=pred,
                 correct=pred == r["actual_label"], interpretation=out["interpretation"])
        print(f"{r['demo_file']:<24} actual={r['actual_label']:<8} pred={pred:<8} risk={risk:.4f} "
              f"{'OK' if r['correct'] else 'WRONG'}")
    saved["description"] = ("Day-7 demo set, re-scored with each new deployed model (Day 8: model v2). "
                            "Demonstration only - not a validation set.")
    saved["model"], saved["scoring_history"] = model, history
    MANIFEST_OUT.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    checks = {k: (v["pass"], v["detail"]) for k, v in saved["integrity_checks"].items()}
    write_doc(saved["images"], checks, model, history)
    print(f"Wrote {DOC_OUT} and {MANIFEST_OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and score the Day-7 demo set")
    ap.add_argument("--dry-run", action="store_true", help="selection + integrity only; no copy, no scoring")
    ap.add_argument("--doc-only", action="store_true", help="rewrite DEMO_IMAGES.md from demo_manifest.json")
    ap.add_argument("--rescore", action="store_true", help="score the recorded files once with a new deployed model")
    args = ap.parse_args()

    if args.rescore:
        return rescore()
    if args.doc_only:
        saved = json.loads(MANIFEST_OUT.read_text(encoding="utf-8"))
        checks = {k: (v["pass"], v["detail"]) for k, v in saved["integrity_checks"].items()}
        write_doc(saved["images"], checks, saved["model"], saved.get("scoring_history"))
        return 0
    if MANIFEST_OUT.exists():
        raise SystemExit(f"{MANIFEST_OUT} exists: demo set already scored. Refusing to score again.")

    sel = select()
    checks = integrity(sel)
    for name, (ok, detail) in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{' - ' + detail if detail else ''}")
    for s in sel:
        print(f"  {s['demo_file']:<24} <- {s['source'].relative_to(ROOT).as_posix()} ({s['actual_label']})")
    if not all(ok for ok, _ in checks.values()):
        raise SystemExit("Integrity check failed - nothing copied or scored.")
    if args.dry_run:
        return 0

    DEMO.mkdir(parents=True, exist_ok=True)
    api = ModelAPI(DEFAULT_MODEL_PATH)
    api.load_model()
    md = api.metadata
    model = {
        "path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(), "sha256": sha256(DEFAULT_MODEL_PATH),
        "model_version": md.get("model_version"), "feature_schema_version": md.get("feature_schema_version"),
        "feature_count": md.get("feature_count"), "rf_params": md.get("rf_params"),
        "trained_at": md.get("trained_at"), "n_training_samples": md.get("n_training_samples"),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    records = []
    for s in sel:
        dst = DEMO / s["demo_file"]
        shutil.copy2(s["source"], dst)
        assert sha256(dst) == sha256(s["source"]), dst
        with Image.open(dst) as im:
            w, h, mode = im.width, im.height, im.mode
        out = api.predict(dst)  # scored exactly once
        risk = out["tamper_risk_score"]
        pred = "tampered" if risk >= RISK_THRESHOLD else "genuine"
        rec = {k: (v.relative_to(ROOT).as_posix() if isinstance(v, Path) else v) for k, v in s.items()}
        rec.update(sha256=sha256(dst), width=w, height=h, mode=mode, file_bytes=dst.stat().st_size,
                   tamper_risk_score=risk, confidence=out["confidence"], predicted_label=pred,
                   correct=pred == s["actual_label"], interpretation=out["interpretation"])
        records.append(rec)

    MANIFEST_OUT.write_text(json.dumps({
        "description": "Day-7 demo set. Demonstration only - not a validation set.",
        "model": model, "risk_threshold": RISK_THRESHOLD,
        "integrity_checks": {k: {"pass": ok, "detail": d} for k, (ok, d) in checks.items()},
        "images": records,
    }, indent=2), encoding="utf-8")
    write_doc(records, checks, model)

    for r in records:
        print(f"{r['demo_file']:<24} actual={r['actual_label']:<8} pred={r['predicted_label']:<8} "
              f"risk={r['tamper_risk_score']:.4f} {'OK' if r['correct'] else 'WRONG'}")
    print(f"Wrote {DOC_OUT} and {MANIFEST_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
