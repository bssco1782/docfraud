"""
scripts/localization_eval.py
Founder A (Day 9 follow-up): how well /verify's flagged_regions locate the actual edit.

Usage:
    python scripts/localization_eval.py
    python scripts/localization_eval.py --dev-only
    python scripts/localization_eval.py --no-rescale

Ground truth: the generator runs are replayed (scripts/generate_extra_documents.edit_footprints)
to recover every edit's true rectangle and how many pixels it changed. Edits that changed no
pixel (a blur on blank paper) are excluded: there is nothing to find.

Sets:
  DEV       data/generated/dev_v1/ - the design set. The LOCATE_MASKS thresholds in
            core/tamper_features.py were chosen here, so DEV figures are optimistic. DEV is also
            model-v2 training data, so its model scores are in-sample.
  HELD-OUT  data/generated/holdout_v1/ + the 15 frozen TEST images (skipped with --dev-only).
            Measured after the design was frozen; nothing was chosen on them.

Two views per set:
  Locator   core.tamper_features.locate_regions on every document, whatever the score.
  API       ModelAPI.predict (the /verify path): regions only when the score reaches
            TAMPER_THRESHOLD.

Resolution: the DEV pages are also resampled (Lanczos, 0.75x, 1.5x, 2.5x) and measured the same
way, to show how the tile thresholds behave away from the generator's page size (skipped with
--no-rescale). Measurement only; nothing was chosen on it.

Metrics: hit = a returned box intersects the true edit rectangle; top-1 hit = the largest box
does; box precision = share of returned boxes that intersect the edit; best IoU = IoU of the
best box with the edit rectangle (mean over edits); genuine documents that receive boxes.
Writes logs/localization_report.txt and logs/localization_results.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config
from backend.model_utils import ModelAPI
from core import tamper_features as tf
from core.model_trainer import DEFAULT_MODEL_PATH
from scripts.generate_extra_documents import edit_footprints, load_set

SPLIT_PATH = ROOT / "data" / "splits" / "day5_frozen_split.json"
SYNTHETIC = ROOT / "data" / "synthetic"
DEFAULT_REPORT = ROOT / "logs" / "localization_report.txt"
DEFAULT_JSON = ROOT / "logs" / "localization_results.json"
RESCALES = (0.75, 1.5, 2.5)


def intersection(a: dict, rect: list[int]) -> int:
    x, y, w, h = rect
    ix = max(0, min(a["x"] + a["width"], x + w) - max(a["x"], x))
    iy = max(0, min(a["y"] + a["height"], y + h) - max(a["y"], y))
    return ix * iy


def iou(a: dict, rect: list[int]) -> float:
    inter = intersection(a, rect)
    union = a["width"] * a["height"] + rect[2] * rect[3] - inter
    return inter / union if inter else 0.0


# ── Documents ───────────────────────────────────────────────────────────────

def dev_documents() -> list[dict]:
    fp = edit_footprints("dev")
    return [{"set": "DEV", "path": r["path"], "file": r["file"], "label": r["label"],
             "method": r["tamper_method"] or "-", "doc_type": r["doc_type"], **_truth(fp.get(r["path"]))}
            for r in load_set("dev")]


def heldout_documents() -> list[dict]:
    fp_hold = edit_footprints("holdout")
    docs = [{"set": "HOLDOUT", "path": r["path"], "file": r["file"], "label": r["label"],
             "method": r["tamper_method"] or "-", "doc_type": r["doc_type"], **_truth(fp_hold.get(r["path"]))}
            for r in load_set("holdout")]
    fp_syn = {**edit_footprints("synthetic_day2"), **edit_footprints("synthetic_day4")}
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    index = {r["path"]: r for r in csv.DictReader(open(SYNTHETIC / "index.csv", newline="", encoding="utf-8"))}
    docs += [{"set": "TEST", "path": t["path"], "file": SYNTHETIC / t["path"], "label": t["label"],
              "method": index[t["path"]]["tamper_method"] or "-", "doc_type": index[t["path"]]["doc_type"],
              **_truth(fp_syn.get(t["path"]))} for t in split["test"]]
    return docs


def _truth(fp: dict | None) -> dict:
    return {"rect": fp["rect"], "changed_px": fp["changed_px"]} if fp else {"rect": None, "changed_px": None}


# ── Scoring ─────────────────────────────────────────────────────────────────

def run(api: ModelAPI, docs: list[dict]) -> list[dict]:
    for d in docs:
        d["locator"] = tf.locate_regions(d["file"])
        out = api.predict(d["file"])
        d["score"] = out["tamper_risk_score"]
        d["flagged"] = out["tamper_risk_score"] >= config.TAMPER_THRESHOLD
        d["api"] = out["flagged_regions"]
    return docs


def metrics(docs: list[dict], view: str) -> dict:
    """Localisation metrics for one set and one view ('locator' or 'api')."""
    edits = [d for d in docs if d["label"] == "tampered" and d["changed_px"]]
    hit = top1 = boxes = good = 0
    ious, by_method, misses = [], defaultdict(lambda: [0, 0]), defaultdict(int)
    for d in edits:
        regions = d[view]
        ok = [intersection(r, d["rect"]) > 0 for r in regions]
        hit += any(ok)
        top1 += bool(ok) and ok[0]
        boxes += len(regions)
        good += sum(ok)
        ious.append(max((iou(r, d["rect"]) for r in regions), default=0.0))
        by_method[d["method"]][0] += any(ok)
        by_method[d["method"]][1] += 1
        if not any(ok):
            misses[f"{d['method']} on {d['doc_type']}"] += 1
    genuine = [d for d in docs if d["label"] == "genuine"]
    blank = [d for d in docs if d["label"] == "tampered" and d["changed_px"] == 0]
    out = {
        "edits": len(edits), "hit": hit, "top1": top1, "boxes": boxes, "boxes_on_edit": good,
        "precision": good / boxes if boxes else None, "mean_best_iou": statistics.fmean(ious) if ious else 0.0,
        "by_method": dict(sorted(by_method.items())),
        "misses": dict(sorted(misses.items(), key=lambda kv: -kv[1])),
        "genuine": len(genuine), "genuine_with_boxes": sum(bool(d[view]) for d in genuine),
        "no_pixel_edits": len(blank), "no_pixel_edits_with_boxes": sum(bool(d[view]) for d in blank),
    }
    if view == "api":
        out["edits_flagged"] = sum(d["flagged"] for d in edits)
        out["flagged_without_box"] = sum(d["flagged"] and not d["api"] for d in edits)
    return out


def rescaled(api: ModelAPI, docs: list[dict], scales: tuple[float, ...] = RESCALES) -> list[dict]:
    """Both views on every document resampled by each factor (Lanczos); edit rectangles scaled alike."""
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "page.png"
        for scale in scales:
            copies = []
            for d in docs:
                with Image.open(d["file"]) as im:
                    page = im.convert("RGB")
                page.resize((round(page.width * scale), round(page.height * scale)), Image.LANCZOS).save(target)
                copy = {**d, "file": target,
                        "rect": [round(v * scale) for v in d["rect"]] if d["rect"] else None}
                copies.append(run(api, [copy])[0])
            out.append({"scale": scale, "locator": metrics(copies, "locator"), "api": metrics(copies, "api"),
                        "genuine_flagged": sum(d["flagged"] for d in copies if d["label"] == "genuine")})
    return out


def overhead_ms(files: list[Path], runs: int = 5) -> dict:
    def median_ms(fn) -> float:
        fn()
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
        return 1000 * statistics.median(samples)

    feats = statistics.fmean(median_ms(lambda f=f: tf.extract_tamper_features(f)) for f in files)
    both = statistics.fmean(median_ms(lambda f=f: tf.extract_with_regions(f)) for f in files)
    return {"features_ms": feats, "features_and_regions_ms": both, "pages": len(files)}


# ── Report ──────────────────────────────────────────────────────────────────

def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "-"


def section(name: str, docs: list[dict], note: str) -> list[str]:
    L = [f"{name}", "-" * 60, note]
    for view, label in (("locator", "Locator (every document)"), ("api", "API (/verify: regions only when flagged)")):
        m = metrics(docs, view)
        L += [f"  {label}:",
              f"    edits with changed pixels: {m['edits']}"
              + (f"; flagged by the model: {m['edits_flagged']}" if view == "api" else ""),
              f"    hit (a box on the edit):   {_pct(m['hit'], m['edits'])}",
              f"    top-1 hit (largest box):   {_pct(m['top1'], m['edits'])}",
              f"    box precision:             {_pct(m['boxes_on_edit'], m['boxes'])}",
              f"    mean best IoU:             {m['mean_best_iou']:.2f}",
              "    hit by edit method:        " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in m["by_method"].items()),
              "    misses:                    " + (", ".join(f"{k} {v}" for k, v in m["misses"].items()) or "none"),
              f"    genuine documents with boxes: {m['genuine_with_boxes']}/{m['genuine']}; "
              f"no-pixel edits with boxes: {m['no_pixel_edits_with_boxes']}/{m['no_pixel_edits']}"]
        if view == "api":
            L.append(f"    flagged edits without a box: {m['flagged_without_box']}")
    return L


def build_report(r: dict) -> str:
    L = [
        "--------------------------------",
        "DOCFRAUD LOCALISATION REPORT (flagged_regions)",
        "--------------------------------",
        f"Generated (UTC): {r['generated']}",
        "Script: scripts/localization_eval.py",
        f"Model: {r['model']['path']} v{r['model']['version']} ({r['model']['schema']}), SHA-256 {r['model']['sha256']}",
        f"Rule: regions in core/tamper_features.LOCATE_MASKS; at most {tf.MAX_REGIONS}, largest first; the API "
        f"returns them only when tamper_risk_score >= {config.TAMPER_THRESHOLD}.",
        "Ground truth: replayed generator runs (true edit rectangle and changed pixels per edit).",
        "",
        *section("DEV (design set - thresholds chosen here; optimistic)", r["dev_docs"],
                 "  150 generated documents, model-v2 training data (scores in-sample)."),
    ]
    if r.get("heldout_docs"):
        L += ["", *section("HELD-OUT (HOLDOUT 149 + frozen TEST 15; measured once after the design was frozen)",
                           r["heldout_docs"], "  Never used to choose thresholds; the model never trained on them.")]
    if r.get("rescaled"):
        L += ["", "RESOLUTION (DEV pages resampled with Lanczos; measurement only)", "-" * 60,
              "  The tile thresholds were chosen at the generator's page size (850 x 1100 px). A resampled page gets",
              "  new anti-aliased colours and different noise, so the colour and noise masks fire on genuine pages too;",
              "  the model's score, built from the same tile maps, is unreliable there as well",
              "  (logs/model_robustness_report.txt).",
              "  scale  locator hit  locator top-1  genuine w/ locator boxes  genuine flagged  API hit  genuine w/ API boxes"]
        for x in [{"scale": 1.0, "locator": r["dev"]["locator"], "api": r["dev"]["api"],
                   "genuine_flagged": sum(d["flagged"] for d in r["dev_docs"] if d["label"] == "genuine")},
                  *r["rescaled"]]:
            lo, a = x["locator"], x["api"]
            L.append(f"  {x['scale']:>4.2f}x  {lo['hit']:>3}/{lo['edits']:<7} {lo['top1']:>3}/{lo['edits']:<9} "
                     f"{lo['genuine_with_boxes']:>3}/{lo['genuine']:<21} {x['genuine_flagged']:>3}/{lo['genuine']:<12} "
                     f"{a['hit']:>3}/{a['edits']:<4} {a['genuine_with_boxes']:>3}/{a['genuine']}")
        L.append("  Boxes (and scores) are only meaningful for pages at the resolution the model was trained on;")
        L.append("  normalising page resolution before analysis is v3 work (logs/model_release_notes.txt).")
    o = r["overhead"]
    L += ["", "Cost", "-" * 60,
          f"  extract_tamper_features {o['features_ms']:.0f} ms vs extract_with_regions {o['features_and_regions_ms']:.0f} ms "
          f"per page (median over {o['pages']} pages); the model sees identical features.",
          "", "Notes", "-" * 60,
          "  - A 'hit' needs only an intersection with the edit rectangle; IoU shows how tight the box is. Ink "
          "boxes cover the inserted words, which are smaller than the edit's white box, so their IoU is low "
          "even when correct.",
          "  - A text overlay is found through its ink (a run of glyphs with a darkness level rare on the page) or, "
          "on tinted paper, through its white box; where the page's own text shares the overlay's darkness "
          "level, neither signal fires.",
          "  - Regions describe where the page is anomalous, not proof of an edit; synthetic data only.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure flagged_regions against the true edit rectangles")
    ap.add_argument("--dev-only", action="store_true", help="skip the held-out measurement")
    ap.add_argument("--no-rescale", action="store_true", help="skip the resampled-DEV measurement")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    logging.getLogger("backend").setLevel(logging.WARNING)
    api = ModelAPI(DEFAULT_MODEL_PATH)
    api.load_model()
    if api.schema.extract_with_regions is None:
        raise SystemExit(f"The deployed model's schema ({api.schema.version}) does not localise edits.")
    print("DEV...")
    dev = run(api, dev_documents())
    heldout = []
    if not args.dev_only:
        print("Held-out (HOLDOUT + TEST)...")
        heldout = run(api, heldout_documents())
    scaled = []
    if not args.no_rescale:
        print("DEV resampled " + ", ".join(f"{x}x" for x in RESCALES) + "...")
        scaled = rescaled(api, dev)
    overhead = overhead_ms([d["file"] for d in dev[:10]])

    def slim(docs: list[dict]) -> list[dict]:
        return [{k: (str(v) if k == "file" else v) for k, v in d.items() if k != "file"} for d in docs]

    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(),
                  "sha256": hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest(),
                  "version": api.metadata.get("model_version"), "schema": api.schema.version},
        "locate_masks": {k: list(v) for k, v in tf.LOCATE_MASKS.items()},
        "dev": {v: metrics(dev, v) for v in ("locator", "api")},
        "heldout": {v: metrics(heldout, v) for v in ("locator", "api")} if heldout else None,
        "rescaled": scaled,
        "overhead": overhead,
        "dev_docs": slim(dev), "heldout_docs": slim(heldout),
    }
    report = build_report(result)
    for path, text in ((args.report, report), (args.json, json.dumps(result, indent=1) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}\nResults written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
