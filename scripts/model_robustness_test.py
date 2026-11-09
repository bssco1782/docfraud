"""
scripts/model_robustness_test.py
Day 9 (Founder A): robustness of the deployed model to common image degradations.

Usage:
    python scripts/model_robustness_test.py
    python scripts/model_robustness_test.py --workers 4 --report logs/model_robustness_report.txt

Documents: held-out documents only - the 15 frozen TEST images and the 149 documents of
data/generated/holdout_v1/, which the deployed model never saw - plus the 5 demo images,
reported separately.

Every document is degraded in each way below (temporary files, deleted afterwards) and scored
through ModelAPI.predict, the path /verify uses:
  reference       unmodified
  re-save         JPEG re-save at quality 85 (a common upload path)
  low resolution  downscale to 75% and 50% of the pixel size; 50% down and back up to full
                  size (detail lost, size kept)
  rotation        90 and 180 degrees; 1 and 3 degree skew (white fill, canvas expanded)
  cropping        2-px border; 5% and 10% border on every side (a 10% crop can cut into an
                  edit; labels are kept)
  blur            Gaussian blur, radius 0.5 and 1.0 px

Per variant: accuracy, genuine flagged (FPR), tampered missed (FNR), label flips against the
unmodified document, mean |score change|, controlled rejections (ValueError -> HTTP 400) and
crashes (any other exception). Nothing is trained or tuned: this documents the frozen model.

Exit code: 0 if no crash occurred (robustness levels are reported, not gated), 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageFilter

from backend import config
from backend.model_utils import ModelAPI
from core.model_trainer import DEFAULT_MODEL_PATH
from scripts.generate_extra_documents import load_set

SPLIT_PATH = ROOT / "data" / "splits" / "day5_frozen_split.json"
SYNTHETIC = ROOT / "data" / "synthetic"
DEMO_MANIFEST = ROOT / "data" / "demo" / "demo_manifest.json"
DEFAULT_REPORT = ROOT / "logs" / "model_robustness_report.txt"
DEFAULT_JSON = ROOT / "logs" / "model_robustness_results.json"
ORIGINAL = "original"


def _save(img: Image.Image, path: Path, **kw) -> Path:
    img.save(path, **kw)
    return path


def _scale(img: Image.Image, f: float) -> Image.Image:
    return img.resize((max(1, round(img.width * f)), max(1, round(img.height * f))), Image.LANCZOS)


def _crop(img: Image.Image, frac: float = 0.0, px: int = 0) -> Image.Image:
    dx, dy = px or round(img.width * frac), px or round(img.height * frac)
    return img.crop((dx, dy, img.width - dx, img.height - dy))


def _rotate(img: Image.Image, degrees: float) -> Image.Image:
    return img.rotate(degrees, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))


# name -> (category, builder(img, workdir) -> path)
VARIANTS = {
    ORIGINAL: ("reference", lambda im, d: _save(im, d / "orig.png")),
    "jpeg_q85": ("re-save", lambda im, d: _save(im, d / "q85.jpg", quality=85)),
    "downscale_75": ("low resolution", lambda im, d: _save(_scale(im, 0.75), d / "s75.png")),
    "downscale_50": ("low resolution", lambda im, d: _save(_scale(im, 0.5), d / "s50.png")),
    "down_up_50": ("low resolution", lambda im, d: _save(_scale(_scale(im, 0.5), 2.0), d / "du50.png")),
    "rotate_90": ("rotation", lambda im, d: _save(im.transpose(Image.Transpose.ROTATE_90), d / "r90.png")),
    "rotate_180": ("rotation", lambda im, d: _save(im.transpose(Image.Transpose.ROTATE_180), d / "r180.png")),
    "skew_1deg": ("rotation", lambda im, d: _save(_rotate(im, 1), d / "k1.png")),
    "skew_3deg": ("rotation", lambda im, d: _save(_rotate(im, 3), d / "k3.png")),
    "crop_2px": ("cropping", lambda im, d: _save(_crop(im, px=2), d / "c2.png")),
    "crop_5pct": ("cropping", lambda im, d: _save(_crop(im, frac=0.05), d / "c5.png")),
    "crop_10pct": ("cropping", lambda im, d: _save(_crop(im, frac=0.10), d / "c10.png")),
    "blur_r0_5": ("blur", lambda im, d: _save(im.filter(ImageFilter.GaussianBlur(0.5)), d / "b05.png")),
    "blur_r1_0": ("blur", lambda im, d: _save(im.filter(ImageFilter.GaussianBlur(1.0)), d / "b10.png")),
}
DESCRIPTION = {
    ORIGINAL: "unmodified file", "jpeg_q85": "JPEG re-save, quality 85",
    "downscale_75": "75% of width and height", "downscale_50": "50% of width and height",
    "down_up_50": "50% then back to full size", "rotate_90": "rotated 90 degrees",
    "rotate_180": "rotated 180 degrees", "skew_1deg": "rotated 1 degree (white fill)",
    "skew_3deg": "rotated 3 degrees (white fill)", "crop_2px": "2 px off every side",
    "crop_5pct": "5% off every side", "crop_10pct": "10% off every side",
    "blur_r0_5": "Gaussian blur, radius 0.5 px", "blur_r1_0": "Gaussian blur, radius 1.0 px",
}


def label_for(score: float) -> str:
    return "tampered" if score >= config.TAMPER_THRESHOLD else "genuine"


# ── Documents ───────────────────────────────────────────────────────────────

def heldout_documents() -> list[dict]:
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    index = {r["path"]: r for r in csv.DictReader(open(SYNTHETIC / "index.csv", newline="", encoding="utf-8"))}
    docs = [{"set": "TEST", "path": r["path"], "file": str(SYNTHETIC / r["path"]), "label": r["label"],
             "doc_type": index[r["path"]]["doc_type"], "method": index[r["path"]]["tamper_method"] or "-"}
            for r in split["test"]]
    docs += [{"set": "HOLDOUT", "path": r["path"], "file": str(r["file"]), "label": r["label"],
              "doc_type": r["doc_type"], "method": r["tamper_method"] or "-"} for r in load_set("holdout")]
    return docs


def demo_documents() -> list[dict]:
    manifest = json.loads(DEMO_MANIFEST.read_text(encoding="utf-8"))
    return [{"set": "DEMO", "path": r["demo_file"], "file": str(ROOT / "data" / "demo" / r["demo_file"]),
             "label": r["actual_label"], "doc_type": "payslip", "method": r.get("transform", "-")}
            for r in manifest["images"]]


# ── Scoring (one process per worker, model loaded once) ─────────────────────

_API: ModelAPI | None = None


def _init_worker() -> None:
    global _API
    logging.getLogger("backend").setLevel(logging.WARNING)
    _API = ModelAPI(DEFAULT_MODEL_PATH)
    _API.load_model()


def score_document(doc: dict) -> list[dict]:
    """All variants of one document through ModelAPI.predict."""
    if _API is None:
        _init_worker()
    rows = []
    with tempfile.TemporaryDirectory() as tmp, Image.open(doc["file"]) as src:
        img = src.convert("RGB")
        for name, (category, build) in VARIANTS.items():
            path = build(img, Path(tmp))
            with Image.open(path) as im:
                size = im.size
            row = {"set": doc["set"], "path": doc["path"], "label": doc["label"], "doc_type": doc["doc_type"],
                   "method": doc["method"], "variant": name, "category": category, "size": list(size)}
            t0 = time.perf_counter()
            try:
                out = _API.predict(path)
                row.update(outcome="scored", score=out["tamper_risk_score"], predicted=label_for(out["tamper_risk_score"]))
            except ValueError as exc:  # controlled rejection (HTTP 400 in the API)
                row.update(outcome="rejected", detail=str(exc)[:120])
            except Exception as exc:  # crash
                row.update(outcome="crash", detail=f"{type(exc).__name__}: {exc}"[:160])
            row["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            rows.append(row)
    return rows


def run_all(docs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [r for d in docs for r in score_document(d)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        return [r for rows in pool.map(score_document, docs, chunksize=4) for r in rows]


# ── Summaries ───────────────────────────────────────────────────────────────

def summarize(rows: list[dict]) -> dict:
    base = {(r["set"], r["path"]): r for r in rows if r["variant"] == ORIGINAL}
    out = {}
    for name in VARIANTS:
        vr = [r for r in rows if r["variant"] == name]
        scored = [r for r in vr if r["outcome"] == "scored"]
        gen = [r for r in scored if r["label"] == "genuine"]
        tam = [r for r in scored if r["label"] == "tampered"]
        flips = [r for r in scored if base[(r["set"], r["path"])].get("predicted") not in (None, r["predicted"])]
        deltas = [abs(r["score"] - base[(r["set"], r["path"])]["score"]) for r in scored
                  if base[(r["set"], r["path"])].get("score") is not None]
        by_method, by_type = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for r in tam:
            by_method[r["method"]][0] += r["predicted"] == "tampered"
            by_method[r["method"]][1] += 1
        for r in scored:
            by_type[r["doc_type"]][0] += r["predicted"] == r["label"]
            by_type[r["doc_type"]][1] += 1
        out[name] = {
            "category": VARIANTS[name][0], "description": DESCRIPTION[name], "n": len(vr), "scored": len(scored),
            "accuracy": sum(r["predicted"] == r["label"] for r in scored) / max(len(scored), 1),
            "fpr": sum(r["predicted"] == "tampered" for r in gen) / max(len(gen), 1),
            "fnr": sum(r["predicted"] == "genuine" for r in tam) / max(len(tam), 1),
            "flips": len(flips), "flip_rate": len(flips) / max(len(scored), 1),
            "mean_abs_score_change": statistics.fmean(deltas) if deltas else 0.0,
            "rejected": sum(r["outcome"] == "rejected" for r in vr),
            "crashes": sum(r["outcome"] == "crash" for r in vr),
            "mean_ms": statistics.fmean(r["ms"] for r in scored) if scored else 0.0,
            "recall_by_method": dict(sorted(by_method.items())),
            "accuracy_by_doc_type": dict(sorted(by_type.items())),
        }
    return out


def fragile_documents(rows: list[dict], top: int = 10) -> list[dict]:
    base = {(r["set"], r["path"]): r for r in rows if r["variant"] == ORIGINAL}
    flips = defaultdict(list)
    for r in rows:
        b = base[(r["set"], r["path"])]
        if r["variant"] != ORIGINAL and r["outcome"] == "scored" and b.get("predicted") and r["predicted"] != b["predicted"]:
            flips[(r["set"], r["path"])].append(r["variant"])
    ranked = sorted(flips.items(), key=lambda kv: -len(kv[1]))[:top]
    return [{"set": k[0], "path": k[1], "label": base[k]["label"], "doc_type": base[k]["doc_type"],
             "method": base[k]["method"], "original_score": base[k]["score"], "flips": v} for k, v in ranked]


# ── Report ──────────────────────────────────────────────────────────────────

def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def findings(r: dict) -> list[str]:
    s = r["summary"]
    ref = s[ORIGINAL]
    out = []
    worst = sorted((k for k in s if k != ORIGINAL), key=lambda k: s[k]["accuracy"])
    for k in worst[:4]:
        v = s[k]
        more_fp, more_fn = v["fpr"] > ref["fpr"] + 0.05, v["fnr"] > ref["fnr"] + 0.05
        cause = ("more genuine pages flagged and more edits missed" if more_fp and more_fn
                 else "more genuine pages flagged" if more_fp else "more edits missed")
        out.append(f"- {k} ({v['description']}): accuracy {pct(v['accuracy'])} (unmodified {pct(ref['accuracy'])}); "
                   f"FPR {pct(v['fpr'])}, FNR {pct(v['fnr'])} - {cause}.")
    stable = [k for k in s if k != ORIGINAL and s[k]["accuracy"] >= ref["accuracy"] - 0.02]
    if stable:
        out.append("- Within 2 points of the unmodified accuracy: " + ", ".join(stable) + ".")
    by_cat = defaultdict(list)
    for k, v in s.items():
        if k != ORIGINAL:
            by_cat[v["category"]].append(v["accuracy"])
    out.append("- Mean accuracy by category: " + ", ".join(f"{c} {pct(statistics.fmean(a))}" for c, a in by_cat.items())
               + ".")
    crashes = sum(v["crashes"] for v in s.values())
    rejected = sum(v["rejected"] for v in s.values())
    out.append(f"- Crashes: {crashes}. Controlled rejections: {rejected}.")
    return out


def recommendations(r: dict) -> list[str]:
    s = r["summary"]
    weak = [k for k in s if k != ORIGINAL and s[k]["accuracy"] < s[ORIGINAL]["accuracy"] - 0.10]
    return [
        "- Treat the reported accuracy as valid for clean, upright, full-page PNG/PDF renders only. Variants below "
        f"lose more than 10 points: {', '.join(weak) or 'none'}.",
        "- Add these degradations to the training copies (rotation by 90/180 and small skew, border crops, "
        "downscaling, light blur) for both classes, as was done for JPEG / resize / brightness in v2, and "
        "measure the effect by grouped cross-validation on training data before touching a new holdout set.",
        "- Normalise before analysis: deskew and auto-rotate (text-line orientation), trim scanner borders, and "
        "resample to a fixed resolution, so the features see a canonical page.",
        "- Make the features scale-aware: tile size and region thresholds are in pixels, so the same edit covers "
        "fewer tiles on a downscaled page. Express them relative to page size or text height.",
        "- Reject or flag inputs outside the validated range (very small pages, rotated pages) instead of scoring "
        "them silently; the API currently returns a score for every decodable page.",
        "- Re-run this script on every model change; the frozen TEST and HOLDOUT documents are reused here for "
        "measurement only, never for training or selection.",
    ]


def build_report(r: dict) -> str:
    s = r["summary"]
    ref = s[ORIGINAL]
    L = [
        "--------------------------------",
        "DOCFRAUD MODEL ROBUSTNESS REPORT",
        "--------------------------------",
        f"Generated (UTC): {r['generated']}",
        f"Script: scripts/model_robustness_test.py (workers={r['workers']})",
        f"Model: {r['model']['path']} v{r['model']['version']}, SHA-256 {r['model']['sha256']}",
        f"Documents: {r['n_heldout']} held-out (15 frozen TEST + {r['n_heldout'] - 15} HOLDOUT; "
        f"{r['n_genuine']} genuine / {r['n_tampered']} tampered), never seen in training.",
        f"Path: ModelAPI.predict (the /verify pipeline); tampered if score >= {config.TAMPER_THRESHOLD}.",
        f"Total predictions: {r['n_predictions']}, time {r['seconds']:.0f} s.",
        "",
        "--------------------------------",
        "SUMMARY BY VARIANT (held-out documents)",
        "--------------------------------",
        f"{'Variant':<14} {'Category':<15} {'Description':<31} {'Acc':>6} {'Change':>7} {'FPR':>6} {'FNR':>6} "
        f"{'Flips':>6} {'|dScore|':>8} {'Rej':>4} {'Crash':>5}",
        "-" * 118,
    ]
    for k, v in s.items():
        L.append(f"{k:<14} {v['category']:<15} {v['description']:<31} {pct(v['accuracy']):>6} "
                 f"{100 * (v['accuracy'] - ref['accuracy']):>+6.1f}p {pct(v['fpr']):>6} {pct(v['fnr']):>6} "
                 f"{v['flips']:>6} {v['mean_abs_score_change']:>8.3f} {v['rejected']:>4} {v['crashes']:>5}")
    L += ["Acc = accuracy; FPR = genuine flagged; FNR = tampered missed; Flips = label differs from the unmodified",
          "document; |dScore| = mean absolute change of tamper_risk_score; Rej = controlled rejections (HTTP 400).", ""]

    methods = sorted({m for v in s.values() for m in v["recall_by_method"]})
    L += ["--------------------------------", "TAMPERED DOCUMENTS CAUGHT, BY EDIT METHOD", "--------------------------------",
          f"{'Variant':<14} " + " ".join(f"{m:>14}" for m in methods), "-" * (15 + 15 * len(methods))]
    for k, v in s.items():
        cells = [f"{v['recall_by_method'].get(m, [0, 0])[0]:>7}/{v['recall_by_method'].get(m, [0, 0])[1]:<6}"
                 for m in methods]
        L.append(f"{k:<14} " + " ".join(f"{c:>14}" for c in cells))

    types = sorted({t for v in s.values() for t in v["accuracy_by_doc_type"]})
    L += ["", "--------------------------------", "ACCURACY BY DOCUMENT TYPE", "--------------------------------",
          f"{'Variant':<14} " + " ".join(f"{t:>16}" for t in types), "-" * (15 + 17 * len(types))]
    for k, v in s.items():
        L.append(f"{k:<14} " + " ".join(
            f"{pct(v['accuracy_by_doc_type'][t][0] / max(v['accuracy_by_doc_type'][t][1], 1)):>16}" for t in types))

    L += ["", "--------------------------------", "DEMO IMAGES (score per variant; * = label differs from expected)",
          "--------------------------------",
          f"{'Variant':<14} " + " ".join(f"{d['path'][:18]:>18}" for d in r["demo_docs"]), "-" * 110]
    demo = {(x["path"], x["variant"]): x for x in r["demo_rows"]}
    for k in VARIANTS:
        cells = []
        for d in r["demo_docs"]:
            x = demo[(d["path"], k)]
            cells.append(f"{x['score']:.2f}{'*' if x['predicted'] != x['label'] else ' '}" if x["outcome"] == "scored"
                         else x["outcome"])
        L.append(f"{k:<14} " + " ".join(f"{c:>18}" for c in cells))
    L.append("Expected: " + ", ".join(f"{d['path']} {d['label']}" for d in r["demo_docs"]))

    L += ["", "--------------------------------", "MOST FRAGILE HELD-OUT DOCUMENTS (label flips across variants)",
          "--------------------------------"]
    for f in r["fragile"]:
        L.append(f"  {f['set']:<8} {f['path']:<34} {f['label']:<8} {f['doc_type']:<15} score {f['original_score']:.2f}"
                 f"  flips under {len(f['flips'])}: {', '.join(f['flips'])}")

    problems = [x for x in r["rows"] if x["outcome"] != "scored"]
    L += ["", "--------------------------------", "REJECTIONS AND CRASHES", "--------------------------------"]
    L += [f"  {x['outcome']:<9} {x['set']:<8} {x['path']:<34} {x['variant']:<14} {x.get('detail', '')}"
          for x in problems[:20]] or ["  none - every variant of every document was scored"]

    L += ["", "--------------------------------", "FINDINGS", "--------------------------------"]
    L += r["findings"]
    L += ["", "--------------------------------", "RECOMMENDATIONS", "--------------------------------"]
    L += r["recommendations"]
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Robustness of the deployed model to image degradations")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()

    docs, demo_docs = heldout_documents(), demo_documents()
    probe = ModelAPI(DEFAULT_MODEL_PATH)
    probe.load_model()
    print(f"Scoring {len(docs) + len(demo_docs)} documents x {len(VARIANTS)} variants with {args.workers} worker(s)...")
    t0 = time.perf_counter()
    rows = run_all(docs + demo_docs, args.workers)
    seconds = time.perf_counter() - t0
    held = [x for x in rows if x["set"] != "DEMO"]
    result = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workers": args.workers,
        "model": {"path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(),
                  "sha256": hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest(),
                  "version": probe.metadata.get("model_version"), "schema": probe.schema.version},
        "variants": {k: {"category": v[0], "description": DESCRIPTION[k]} for k, v in VARIANTS.items()},
        "n_heldout": len(docs), "n_genuine": sum(d["label"] == "genuine" for d in docs),
        "n_tampered": sum(d["label"] == "tampered" for d in docs),
        "n_predictions": len(rows), "seconds": seconds,
        "summary": summarize(held),
        "fragile": fragile_documents(held),
        "demo_docs": [{k: v for k, v in d.items() if k != "file"} for d in demo_docs],  # no local paths
        "demo_rows": [x for x in rows if x["set"] == "DEMO"],
        "rows": [{k: x[k] for k in ("set", "path", "variant", "outcome", "score", "detail") if k in x} for x in rows],
        "crashes": sum(x["outcome"] == "crash" for x in rows),
    }
    result["findings"] = findings(result)
    result["recommendations"] = recommendations(result)
    report = build_report(result)
    for path, text in ((args.report, report), (args.json, json.dumps(result, indent=1) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}\nResults written: {args.json}")
    return 0 if result["crashes"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
