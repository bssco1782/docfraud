"""
scripts/inspect_features.py
Read-only inspection tool: saves colormapped ELA heatmaps and prints
a feature table for all images in data/synthetic/ (and optionally
tests/fixtures/).

Usage:
    python scripts/inspect_features.py [--fixtures]

Flags:
    --fixtures   Also process tests/fixtures/ images in addition to
                 data/synthetic/.

Output:
    - Colormapped heatmap PNGs saved to logs/heatmaps/<filename>_heatmap.png
    - Feature table printed to stdout
    - Per-class mean comparison if both genuine/ and tampered/ sub-folders exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make project root importable when run from any directory
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.document_analyzer import FEATURE_NAMES, ela_analysis, extract_features

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
FIXTURES_DIR = ROOT / "tests" / "fixtures"
HEATMAP_DIR = ROOT / "logs" / "heatmaps"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _collect_images(include_fixtures: bool) -> list[tuple[Path, str]]:
    """Return list of (path, label) tuples."""
    images: list[tuple[Path, str]] = []

    # data/synthetic/ — label by sub-folder if available
    for sub in ["genuine", "tampered"]:
        sub_dir = SYNTHETIC_DIR / sub
        if sub_dir.exists():
            for p in sorted(sub_dir.iterdir()):
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    images.append((p, sub))
    if not images and SYNTHETIC_DIR.exists():
        for p in sorted(SYNTHETIC_DIR.iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES:
                images.append((p, "unknown"))

    if include_fixtures and FIXTURES_DIR.exists():
        for p in sorted(FIXTURES_DIR.iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES and "corrupt" not in p.stem:
                images.append((p, "fixture"))

    return images


def _save_heatmap(heatmap: np.ndarray, name: str) -> Path:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    colormapped = cv2.applyColorMap(heatmap.mean(axis=2).astype(np.uint8), cv2.COLORMAP_JET)
    out_path = HEATMAP_DIR / f"{name}_heatmap.png"
    cv2.imwrite(str(out_path), colormapped)
    return out_path


def _print_table(rows: list[dict], label_col: bool = True) -> None:
    col_w = 28
    fmt_h = f"{'filename':30s}{'label':10s}{'ela_score':10s}" if label_col else f"{'filename':30s}{'ela_score':10s}"
    for name in FEATURE_NAMES:
        fmt_h += f"{name[:col_w]:>{col_w}s}"
    print(fmt_h)
    print("-" * len(fmt_h))
    for row in rows:
        line = f"{row['filename'][:30]:30s}"
        if label_col:
            line += f"{row['label']:10s}"
        line += f"{row['ela_score']:10.2f}"
        for name in FEATURE_NAMES:
            line += f"{row['features'][name]:>{col_w}.4f}"
        print(line)


def _print_class_means(rows: list[dict]) -> None:
    classes = {r["label"] for r in rows if r["label"] in ("genuine", "tampered")}
    if len(classes) < 2:
        return

    print("\n" + "=" * 60)
    print("Per-class feature means")
    print("=" * 60)
    header = f"{'feature':30s}{'genuine':>14s}{'tampered':>14s}{'flag':>8s}"
    print(header)
    print("-" * len(header))

    for feat in FEATURE_NAMES:
        gen_vals = [r["features"][feat] for r in rows if r["label"] == "genuine"]
        tam_vals = [r["features"][feat] for r in rows if r["label"] == "tampered"]
        gen_mean = np.mean(gen_vals) if gen_vals else float("nan")
        tam_mean = np.mean(tam_vals) if tam_vals else float("nan")
        # Flag features where both classes have nearly identical means
        diff = abs(gen_mean - tam_mean)
        mean_mag = (abs(gen_mean) + abs(tam_mean)) / 2 + 1e-9
        flag = "WEAK" if diff / mean_mag < 0.05 else ""
        print(f"{feat:30s}{gen_mean:>14.4f}{tam_mean:>14.4f}{flag:>8s}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", action="store_true",
                        help="Include tests/fixtures/ images")
    args = parser.parse_args()

    images = _collect_images(include_fixtures=args.fixtures)

    if not images:
        print("No images found.")
        print(f"  Looked in: {SYNTHETIC_DIR}")
        if args.fixtures:
            print(f"             {FIXTURES_DIR}")
        print("Add images to data/synthetic/ to use this script with real data.")
        return

    rows = []
    for path, label in images:
        print(f"Processing {path.name} ...", end=" ", flush=True)
        try:
            heatmap, score = ela_analysis(path)
            feats = extract_features(path)
            saved = _save_heatmap(heatmap, path.stem)
            print(f"score={score:.2f}  heatmap->{saved.name}")
            rows.append({
                "filename": path.name,
                "label": label,
                "ela_score": score,
                "features": feats,
            })
        except ValueError as exc:
            print(f"SKIPPED ({exc})")

    if not rows:
        print("No images could be processed.")
        return

    print("\n" + "=" * 60)
    print("Feature table")
    print("=" * 60)
    _print_table(rows, label_col=True)
    _print_class_means(rows)


if __name__ == "__main__":
    main()
