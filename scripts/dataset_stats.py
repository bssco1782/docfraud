"""
scripts/dataset_stats.py
Founder C — Day 7: final dataset index statistics.

Read-only. Recomputes statistics for the indexed synthetic dataset from
data/synthetic/index.csv and the images it references, and prints them to
stdout. Never writes, moves or modifies any file.

Usage:
    python scripts/dataset_stats.py                    # plain-text report
    python scripts/dataset_stats.py --format markdown  # markdown tables
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = ROOT / "data" / "synthetic"
INDEX_CSV = SYNTHETIC / "index.csv"

# Preferred display order; any value not listed is appended alphabetically.
LABEL_ORDER = ["genuine", "tampered"]
DOC_TYPE_ORDER = ["payslip", "bank_statement", "id_proof"]
TAMPER_ORDER = ["blur_region", "color_shift", "noise_patch", "text_overlay"]
DAY1_GLOB = "payslip_[0-9][0-9][0-9].png"


def ordered(values, preferred):
    present = set(values)
    return [v for v in preferred if v in present] + sorted(present - set(preferred))


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "n/a"


def load_rows() -> list[dict]:
    with open(INDEX_CSV, newline="") as f:
        return list(csv.DictReader(f))


def inspect_images(rows: list[dict]) -> dict:
    missing, unreadable, mismatched = [], [], []
    for r in rows:
        path = SYNTHETIC / r["path"]
        if not path.is_file():
            missing.append(r["path"])
            continue
        try:
            with Image.open(path) as img:
                img.load()
                size = img.size
        except Exception:
            unreadable.append(r["path"])
            continue
        if size != (int(r["width"]), int(r["height"])) or path.stat().st_size != int(r["file_size_bytes"]):
            mismatched.append(r["path"])
    return {"missing": missing, "unreadable": unreadable, "mismatched": mismatched}


def compute(rows: list[dict]) -> dict:
    total = len(rows)
    labels = Counter(r["label"] for r in rows)
    doc_types = Counter(r["doc_type"] for r in rows)
    tampered = [r for r in rows if r["tamper_method"]]
    methods = Counter(r["tamper_method"] for r in tampered)
    sizes = [int(r["file_size_bytes"]) for r in rows]
    label_keys = ordered(labels, LABEL_ORDER)
    type_keys = ordered(doc_types, DOC_TYPE_ORDER)
    method_keys = ordered(methods, TAMPER_ORDER)

    by_type = {}
    for t in type_keys:
        sub = [r for r in rows if r["doc_type"] == t]
        s = [int(r["file_size_bytes"]) for r in sub]
        by_type[t] = {
            "total": len(sub),
            "labels": {lb: sum(r["label"] == lb for r in sub) for lb in label_keys},
            "methods": {m: sum(r["tamper_method"] == m for r in sub) for m in method_keys},
            "dims": sorted(Counter(f"{r['width']}x{r['height']}" for r in sub).items()),
            "size_mean": round(statistics.mean(s)),
            "size_min": min(s),
            "size_max": max(s),
        }

    return {
        "total": total,
        "labels": {k: labels[k] for k in label_keys},
        "doc_types": {k: doc_types[k] for k in type_keys},
        "tampered_total": len(tampered),
        "methods": {k: methods[k] for k in method_keys},
        "by_type": by_type,
        "size": {
            "total": sum(sizes),
            "mean": round(statistics.mean(sizes)),
            "median": round(statistics.median(sizes)),
            "min": min(sizes),
            "max": max(sizes),
        },
        "day1_outside_index": len([
            p for p in SYNTHETIC.glob(DAY1_GLOB)
            if p.name not in {r["path"] for r in rows}
        ]),
        "integrity": inspect_images(rows),
    }


def render_text(s: dict) -> str:
    t = s["total"]
    out = ["DocFraud synthetic dataset — final index statistics",
           f"Source: {INDEX_CSV.relative_to(ROOT).as_posix()}", ""]
    out.append(f"Total indexed documents: {t}")
    out.append(f"Day 1 payslips outside the index: {s['day1_outside_index']}")
    out += ["", "Labels:"]
    out += [f"  {k:<15} {v:>4}  {pct(v, t):>6}" for k, v in s["labels"].items()]
    out += ["", "Document types:"]
    out += [f"  {k:<15} {v:>4}  {pct(v, t):>6}" for k, v in s["doc_types"].items()]
    labels = list(s["labels"])
    out += ["", "Labels by document type:",
            "  " + f"{'doc_type':<15}" + "".join(f"{lb:>10}" for lb in labels) + f"{'total':>8}{'% tampered':>12}"]
    for k, b in s["by_type"].items():
        out.append("  " + f"{k:<15}" + "".join(f"{b['labels'][lb]:>10}" for lb in labels)
                   + f"{b['total']:>8}{pct(b['labels'].get('tampered', 0), b['total']):>12}")
    tt = s["tampered_total"]
    out += ["", f"Tamper methods (of {tt} tampered):"]
    out += [f"  {k:<15} {v:>4}  {pct(v, tt):>6}" for k, v in s["methods"].items()]
    methods = list(s["methods"])
    out += ["", "Tamper methods by document type:",
            "  " + f"{'doc_type':<15}" + "".join(f"{m:>14}" for m in methods)]
    for k, b in s["by_type"].items():
        out.append("  " + f"{k:<15}" + "".join(f"{b['methods'][m]:>14}" for m in methods))
    out += ["", "Image dimensions (width x height px) by document type:"]
    for k, b in s["by_type"].items():
        out.append(f"  {k:<15} " + ", ".join(f"{d} ({n})" for d, n in b["dims"]))
    z = s["size"]
    out += ["", "File sizes (bytes):",
            f"  total  {z['total']:>10,}", f"  mean   {z['mean']:>10,}", f"  median {z['median']:>10,}",
            f"  min    {z['min']:>10,}", f"  max    {z['max']:>10,}", "",
            "File sizes by document type (bytes):"]
    for k, b in s["by_type"].items():
        out.append(f"  {k:<15} mean {b['size_mean']:>7,}  min {b['size_min']:>7,}  max {b['size_max']:>7,}")
    i = s["integrity"]
    out += ["", "Integrity (read-only check of referenced images):",
            f"  missing files:         {len(i['missing'])}",
            f"  unreadable images:     {len(i['unreadable'])}",
            f"  size/bytes mismatches: {len(i['mismatched'])}"]
    return "\n".join(out)


def render_markdown(s: dict) -> str:
    t, tt = s["total"], s["tampered_total"]
    labels, methods = list(s["labels"]), list(s["methods"])
    out = ["| Metric | Value |", "|---|---|",
           f"| Total indexed documents | {t} |",
           f"| Day 1 payslips outside the index | {s['day1_outside_index']} |", "",
           "| Label | Count | Share |", "|---|---:|---:|"]
    out += [f"| {k} | {v} | {pct(v, t)} |" for k, v in s["labels"].items()]
    out += ["", "| Document type | Count | Share |", "|---|---:|---:|"]
    out += [f"| {k} | {v} | {pct(v, t)} |" for k, v in s["doc_types"].items()]
    out += ["", "| Document type | " + " | ".join(labels) + " | Total | % tampered |",
            "|---|" + "---:|" * (len(labels) + 2)]
    for k, b in s["by_type"].items():
        out.append(f"| {k} | " + " | ".join(str(b["labels"][lb]) for lb in labels)
                   + f" | {b['total']} | {pct(b['labels'].get('tampered', 0), b['total'])} |")
    out += ["", f"| Tamper method | Count | Share of {tt} tampered |", "|---|---:|---:|"]
    out += [f"| {k} | {v} | {pct(v, tt)} |" for k, v in s["methods"].items()]
    out += ["", "| Document type | " + " | ".join(methods) + " |", "|---|" + "---:|" * len(methods)]
    for k, b in s["by_type"].items():
        out.append(f"| {k} | " + " | ".join(str(b["methods"][m]) for m in methods) + " |")
    out += ["", "| Document type | Dimensions (px) | Mean bytes | Min bytes | Max bytes |",
            "|---|---|---:|---:|---:|"]
    for k, b in s["by_type"].items():
        dims = ", ".join(f"{d} ({n})" for d, n in b["dims"])
        out.append(f"| {k} | {dims} | {b['size_mean']:,} | {b['size_min']:,} | {b['size_max']:,} |")
    z = s["size"]
    out += ["", "| File size (all indexed) | Bytes |", "|---|---:|",
            f"| Total | {z['total']:,} |", f"| Mean | {z['mean']:,} |", f"| Median | {z['median']:,} |",
            f"| Min | {z['min']:,} |", f"| Max | {z['max']:,} |"]
    i = s["integrity"]
    out += ["", "| Integrity check | Count |", "|---|---:|",
            f"| Missing files | {len(i['missing'])} |",
            f"| Unreadable images | {len(i['unreadable'])} |",
            f"| Size/bytes mismatches vs index.csv | {len(i['mismatched'])} |"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=["text", "markdown"], default="text")
    args = parser.parse_args(argv)
    if not INDEX_CSV.is_file():
        print(f"index.csv not found at {INDEX_CSV}", file=sys.stderr)
        return 1
    stats = compute(load_rows())
    print(render_markdown(stats) if args.format == "markdown" else render_text(stats))
    i = stats["integrity"]
    return 1 if (i["missing"] or i["unreadable"] or i["mismatched"]) else 0


if __name__ == "__main__":
    sys.exit(main())
