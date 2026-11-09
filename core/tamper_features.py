"""
core/tamper_features.py
Local tamper features — schema ``local-tamper-v2`` (Day 8, Founder A).

Why this exists
---------------
The Day-4 schema (``day4-v2-22``) summarises the whole page, while the edits in this
dataset are local rectangles (a noise patch, a colour cast, a blurred field, an overlaid
value). Page-wide averages barely move, so the Day-4 model scored 60% held-out.

This module asks, for each kind of edit, whether one part of the page behaves differently
from the rest of the SAME page, and summarises the answer as the size of the largest
anomalous region (in 8 x 8 px tiles). Page-level "processing fingerprint" features describe
how the file was saved (JPEG grid, flatness, colour and ink variety) so the classifier can
tell a re-saved genuine page from an edited one.

Feature groups (FEATURE_NAMES order)
------------------------------------
Noise (7)       Tile median of |gray - median3x3(gray)|. Text edges touch few pixels of a
                tile, so the median stays ~0 on clean renders; added noise raises every
                pixel. Anomalous = above the page's own floor + spread.
Colour (11)     "Paper colour" of each flat tile (median RGB of its brightest 25%). Regions
                of rare paper colours, and colours far from every common page colour, mark
                a local colour cast or a pasted box on a tinted background.
Softness (5)    Content tiles whose strongest gradient is low for their contrast: a blurred
                region inside otherwise sharp rendered text.
Ink (2)         Darkness of neutral glyphs; a line of glyphs with a darkness level that is
                rare on the page marks inserted text.
Fingerprint (4) JPEG 8-px grid strength, flat-tile fraction, number of paper colours and
                ink levels.

Region rules: regions touching the page edge or wider than 60% of the page are layout
(headers, footers, borders), not local edits, and are ignored.

Localisation: ``extract_with_regions`` / ``locate_regions`` also return the anomalous regions as
bounding boxes in original-image pixels (see LOCATE_MASKS). ``extract_tamper_features`` output
is unchanged by this, so the trained model is unaffected.

Everything is deterministic and returns finite floats. These are anomaly measurements,
NOT fraud probabilities.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from core.document_analyzer import load_image

FEATURE_SCHEMA_VERSION = "local-tamper-v2"

TILE = 8                     # px; anomaly maps are built on TILE x TILE tiles
FLAT_RANGE = 12              # tile max - min gray at or below this = flat (no text / edges)
CONTENT_RANGE = 8            # tile max - min gray at or above this = has content
LAYOUT_WIDTH_FRAC = 0.6      # regions wider than this fraction of the page are layout bands
PAPER_QUANT = 5              # RGB quantisation step for paper colours
COMMON_COLOUR_FRAC = 0.04    # a paper colour on >= 4% of flat tiles is "common"
INK_GRAY_MAX = 170           # glyph pixels: gray below this ...
INK_NEUTRAL_MAX = 20         # ... and |R - B| below this (neutral ink; coloured template text excluded)
INK_MIN_AREA = 8             # px; smaller components are specks, not glyphs
INK_QUANT = 8                # gray-level quantisation for glyph darkness
INK_RARE_FRAC = 0.05         # a darkness level on < 5% of glyphs is rare
INK_LINE_DY, INK_LINE_DX = 6, 120  # px; glyphs within this box of each other form a text run
MIN_TILES = 2                # the page must be at least 2 x 2 tiles

NOISE_STEPS = (2, 4)                 # gray levels above floor + spread
RARE_COLOUR_PCTS = (0.5, 1, 2, 4)    # % of flat tiles
NOVEL_DISTANCES = (10, 20)           # RGB distance to the nearest common colour
SOFT_SHARPNESS = (1.0, 1.6)          # max Sobel gradient / (tile range + 1)

# Localisation. A candidate region is reported only if it is larger than every region of the same
# kind on the 87 genuine documents of data/generated/dev_v1 (the DEV set; chosen there only, never
# on the frozen test set or the holdout set). Sizes are px^2 of the analysed page (tiles x 64),
# except ink runs, which count glyphs. scripts/localization_eval.py measures the result.
LOCATE_MASKS: dict[str, tuple[str, float]] = {
    "noise_k2": ("noise", 384),
    "colour_rare_1pct": ("colour", 256),
    "colour_rare_2pct": ("colour", 4672),
    "colour_rare_4pct": ("colour", 5120),
    "colour_novel_d20": ("colour", 5120),
    "soft_s1_6": ("softness", 64),       # genuine DEV pages have none; require at least 2 tiles
    "ink_run": ("ink", 9),               # at least 10 glyphs with the same rare darkness level
}
SIGNALS = ("noise", "colour", "softness", "ink")
MERGE_GAP = 8                        # px; regions closer than this are reported as one
MAX_REGIONS = 5
PDF_DPI = 200                        # document_analyzer renders PDF page 1 at this resolution


def _pct_name(p: float) -> str:
    return str(p).replace(".", "_") + "pct"


FEATURE_NAMES: list[str] = [
    "noise_floor",
    "noise_excess_p999",
    *[name for k in NOISE_STEPS for name in (f"noise_blob_k{k}", f"noise_blob_k{k}_fill")],
    "noise_anomalous_frac",
    *[name for p in RARE_COLOUR_PCTS for name in (f"colour_rare_blob_{_pct_name(p)}",
                                                   f"colour_rare_blob_{_pct_name(p)}_fill")],
    *[f"colour_novel_blob_d{d}" for d in NOVEL_DISTANCES],
    "colour_novel_p999",
    *[name for s in SOFT_SHARPNESS for name in (f"soft_blob_s{str(s).replace('.', '_')}",
                                                 f"soft_total_s{str(s).replace('.', '_')}")],
    "soft_frac",
    "ink_rare_line",
    "ink_rare_frac",
    "jpeg_grid",
    "flat_frac",
    "paper_colours",
    "ink_levels",
]


def _tiles(arr: np.ndarray, t: int = TILE) -> np.ndarray:
    """(rows, cols, t*t[, channels]) view of the image cut into t x t tiles (ragged edge dropped)."""
    h, w = arr.shape[:2]
    a = arr[: h // t * t, : w // t * t]
    shape = (h // t, t, w // t, t) + a.shape[2:]
    return a.reshape(shape).swapaxes(1, 2).reshape(h // t, w // t, t * t, *a.shape[2:])


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Interior connected regions of a tile mask as (x, y, w, h, area), in tiles, in label order."""
    if not mask.any():
        return []
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rows, cols = mask.shape
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if x < 1 or y < 1 or x + w > cols - 1 or y + h > rows - 1:
            continue  # touches the page edge: header / footer / border
        if w > LAYOUT_WIDTH_FRAC * cols:
            continue  # full-width band: layout element
        out.append((x, y, w, h, area))
    return out


def _regions(mask: np.ndarray, found: list | None = None, name: str = "") -> tuple[int, float, int]:
    """Largest interior region of a tile mask: (area in tiles, bbox fill ratio, total interior area).

    With ``found``, every interior region is also appended as (name, x, y, w, h, area) in pixels
    of the analysed (loaded) image, for localisation.
    """
    best, fill, total = 0, 0.0, 0
    for x, y, w, h, area in _components(mask):
        total += area
        if area > best:
            best, fill = area, area / float(w * h)
        if found is not None:
            found.append((name, x * TILE, y * TILE, w * TILE, h * TILE, area * TILE * TILE))
    return best, fill, total


def _noise(gray: np.ndarray, found: list | None = None) -> list[float]:
    g16 = gray.astype(np.int16)
    resid = np.abs(g16 - cv2.medianBlur(gray, 3).astype(np.int16)).astype(np.float32)
    tile_resid = np.median(_tiles(resid), axis=2)
    floor = float(np.median(tile_resid))
    spread = float(np.percentile(tile_resid, 90) - floor)
    out = [floor, float(np.percentile(tile_resid, 99.9) - floor)]
    for k in NOISE_STEPS:
        area, fill, _ = _regions(tile_resid >= floor + k + spread, found, f"noise_k{k}")
        out += [area, fill]
    out.append(float((tile_resid >= floor + 3 + spread).mean()))
    return out


def _colour(img: np.ndarray, tile_gray: np.ndarray, flat: np.ndarray,
            found: list | None = None) -> tuple[list[float], int]:
    tile_rgb = _tiles(img).astype(np.float32)
    bright = (tile_gray >= np.percentile(tile_gray, 75, axis=2, keepdims=True))[..., None]
    paper = (tile_rgb * bright).sum(2) / np.maximum(bright.sum(2), 1)
    q = np.round(paper / PAPER_QUANT).astype(np.int64)
    key = (q[..., 0] * 64 + q[..., 1]) * 64 + q[..., 2]
    n_flat = max(int(flat.sum()), 1)
    vals, counts = np.unique(key[flat], return_counts=True)
    pos = np.clip(np.searchsorted(vals, key), 0, max(len(vals) - 1, 0))
    known = (vals[pos] == key) if len(vals) else np.zeros(key.shape, bool)
    freq = np.where(known, counts[pos] if len(vals) else 0, 0) / n_flat

    out: list[float] = []
    for p in RARE_COLOUR_PCTS:
        area, fill, _ = _regions(flat & (freq < p / 100), found, f"colour_rare_{_pct_name(p)}")
        out += [area, fill]
    common = [np.array([k // 4096, (k // 64) % 64, k % 64]) * PAPER_QUANT
              for k, c in zip(vals.tolist(), counts.tolist()) if c >= COMMON_COLOUR_FRAC * n_flat]
    if common:
        palette = np.stack(common).astype(np.float32)
        dist = np.min(np.linalg.norm(paper[..., None, :] - palette[None, None], axis=-1), axis=-1)
        dist[~flat] = 0
        out += [_regions(dist > d, found, f"colour_novel_d{d}")[0] for d in NOVEL_DISTANCES]
        out.append(float(np.percentile(dist[flat], 99.9)) if flat.any() else 0.0)
    else:
        out += [0] * len(NOVEL_DISTANCES) + [0.0]
    return out, len(vals)


def _softness(gray: np.ndarray, tile_range: np.ndarray, found: list | None = None) -> list[float]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sharpness = _tiles(np.hypot(gx, gy)).max(2) / (tile_range + 1)
    content = tile_range >= CONTENT_RANGE
    out: list[float] = []
    for s in SOFT_SHARPNESS:
        area, _, total = _regions(content & (sharpness < s), found, f"soft_s{str(s).replace('.', '_')}")
        out += [area, total]
    out.append(float((content & (sharpness < SOFT_SHARPNESS[-1])).sum() / max(int(content.sum()), 1)))
    return out


def _ink(img: np.ndarray, gray: np.ndarray, found: list | None = None) -> tuple[list[float], np.ndarray]:
    neutral = np.abs(img[..., 0].astype(np.int16) - img[..., 2].astype(np.int16)) < INK_NEUTRAL_MAX
    dark = ((gray < INK_GRAY_MAX) & neutral).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    darkest = np.asarray(ndimage.minimum(gray, labels=labels, index=np.arange(n)), dtype=np.int64)
    glyph = stats[:, cv2.CC_STAT_AREA] >= INK_MIN_AREA
    glyph[0] = False  # background label
    levels = darkest[glyph] // INK_QUANT
    if len(levels) < 10:
        return [0, 0.0], levels
    vals, counts = np.unique(levels, return_counts=True)
    share = dict(zip(vals.tolist(), (counts / len(levels)).tolist()))
    rare = np.array([share[v] < INK_RARE_FRAC for v in levels.tolist()])
    cx = stats[glyph, cv2.CC_STAT_LEFT] + stats[glyph, cv2.CC_STAT_WIDTH] / 2
    cy = stats[glyph, cv2.CC_STAT_TOP] + stats[glyph, cv2.CC_STAT_HEIGHT] / 2
    best = 0
    for i in np.flatnonzero(rare):
        run = rare & (levels == levels[i]) & (np.abs(cy - cy[i]) < INK_LINE_DY) & (np.abs(cx - cx[i]) < INK_LINE_DX)
        best = max(best, int(run.sum()))
    if found is not None:
        _ink_runs(rare, levels, cx, cy, stats[glyph], found)
    return [best, float(rare.mean())], levels


def _ink_runs(rare: np.ndarray, levels: np.ndarray, cx: np.ndarray, cy: np.ndarray, boxes: np.ndarray,
              found: list) -> None:
    """Group rare-darkness glyphs of the same level into text runs (chained within the run window).

    Appends ("ink_run", x, y, w, h, n_glyphs) per run, in pixels of the analysed image; the last
    field is the number of glyphs (runs have no tile area).
    """
    idx = np.flatnonzero(rare)
    parent = {int(i): int(i) for i in idx}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in idx:
        near = idx[(levels[idx] == levels[a]) & (np.abs(cy[idx] - cy[a]) < INK_LINE_DY)
                   & (np.abs(cx[idx] - cx[a]) < INK_LINE_DX)]
        for b in near:
            parent[root(int(b))] = root(int(a))
    runs: dict[int, list[int]] = {}
    for i in idx:
        runs.setdefault(root(int(i)), []).append(int(i))
    for members in runs.values():
        b = boxes[members]
        x0, y0 = int(b[:, cv2.CC_STAT_LEFT].min()), int(b[:, cv2.CC_STAT_TOP].min())
        x1 = int((b[:, cv2.CC_STAT_LEFT] + b[:, cv2.CC_STAT_WIDTH]).max())
        y1 = int((b[:, cv2.CC_STAT_TOP] + b[:, cv2.CC_STAT_HEIGHT]).max())
        found.append(("ink_run", x0, y0, x1 - x0, y1 - y0, len(members)))


def extract_tamper_features(image_path: str | Path) -> dict[str, float]:
    """Extract the ``local-tamper-v2`` features of one image (JPEG, PNG or PDF page 1).

    Returns:
        dict with exactly FEATURE_NAMES keys, all finite floats.

    Raises:
        ValueError: unreadable file (from load_image) or an image too small to tile.
    """
    return _analyse(np.asarray(load_image(image_path)))


def _analyse(img: np.ndarray, found: list | None = None) -> dict[str, float]:
    """Features of a loaded RGB page; with ``found``, also collect candidate regions (see _regions)."""
    h, w = img.shape[:2]
    if h < MIN_TILES * TILE or w < MIN_TILES * TILE:
        raise ValueError(f"Image {w}x{h} is too small for local analysis (min {MIN_TILES * TILE} px per side)")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tile_gray = _tiles(gray).astype(np.float32)
    tile_range = tile_gray.max(2) - tile_gray.min(2)
    flat = tile_range <= FLAT_RANGE

    colour, n_paper_colours = _colour(img, tile_gray, flat, found)
    ink, levels = _ink(img, gray, found)
    diffs = np.abs(np.diff(gray.astype(np.int16), axis=1)).astype(np.float32)
    on_grid = (np.arange(diffs.shape[1]) % 8) == 7
    fingerprint = [
        float(diffs[:, on_grid].mean() / (diffs[:, ~on_grid].mean() + 1e-6)),
        float(flat.mean()),
        float(n_paper_colours),
        float(len(np.unique(levels))) if len(levels) else 0.0,
    ]
    values = _noise(gray, found) + colour + _softness(gray, tile_range, found) + ink + fingerprint
    assert len(values) == len(FEATURE_NAMES)
    out = {name: float(v) for name, v in zip(FEATURE_NAMES, values)}
    for name, v in out.items():
        if not np.isfinite(v):
            out[name] = 0.0
    return out


# ── Localisation ─────────────────────────────────────────────────────────────

def original_size(image_path: str | Path) -> tuple[int, int]:
    """(width, height) of the page before load_image's downscale: image header, or PDF page 1 at PDF_DPI."""
    path = Path(image_path)
    if path.suffix.lower() == ".pdf":
        import pymupdf  # noqa: PLC0415 - optional dependency, as in document_analyzer

        with pymupdf.open(str(path)) as doc:
            box = (doc[0].rect * pymupdf.Matrix(PDF_DPI / 72, PDF_DPI / 72)).irect
            return int(box.width), int(box.height)
    with Image.open(path) as im:
        return im.size


def _select_regions(found: list, analysed: tuple[int, int], original: tuple[int, int],
                    max_regions: int) -> list[dict]:
    """Keep candidates above their LOCATE_MASKS size, merge nearby ones, scale to original pixels."""
    boxes = []
    for name, x, y, w, h, size in found:
        if name in LOCATE_MASKS and size > LOCATE_MASKS[name][1]:
            area = w * h if name == "ink_run" else size  # ink runs: the run's box
            boxes.append({"x0": x, "y0": y, "x1": x + w, "y1": y + h, "area": area,
                          "signals": {LOCATE_MASKS[name][0]: area}})
    merged = True
    while merged:  # union boxes that overlap or lie within MERGE_GAP of each other
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (a["x0"] - MERGE_GAP < b["x1"] and b["x0"] < a["x1"] + MERGE_GAP
                        and a["y0"] - MERGE_GAP < b["y1"] and b["y0"] < a["y1"] + MERGE_GAP):
                    signals = dict(a["signals"])
                    for k, v in b["signals"].items():
                        signals[k] = max(signals.get(k, 0), v)
                    boxes[i] = {"x0": min(a["x0"], b["x0"]), "y0": min(a["y0"], b["y0"]),
                                "x1": max(a["x1"], b["x1"]), "y1": max(a["y1"], b["y1"]),
                                "area": max(signals.values()), "signals": signals}
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break

    sx, sy = original[0] / analysed[0], original[1] / analysed[1]
    out = []
    for b in sorted(boxes, key=lambda b: -b["area"])[:max_regions]:
        x0, y0 = min(round(b["x0"] * sx), original[0]), min(round(b["y0"] * sy), original[1])
        x1, y1 = min(round(b["x1"] * sx), original[0]), min(round(b["y1"] * sy), original[1])
        out.append({"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                    "area": int(round(b["area"] * sx * sy)),
                    "type": "+".join(s for s in SIGNALS if s in b["signals"])})
    return out


def extract_with_regions(image_path: str | Path, max_regions: int = MAX_REGIONS) -> tuple[dict[str, float], list[dict]]:
    """Features (identical to extract_tamper_features) plus anomalous regions, from one analysis pass.

    Each region: x, y, width, height (original-image pixels; PDF page 1 at PDF_DPI), area (pixel
    area of the anomalous tiles, or of the text-run box for ink) and type ("noise", "colour",
    "softness", "ink", or several joined by "+" when regions of different kinds overlap).
    Largest first, at most ``max_regions``.
    """
    img = np.asarray(load_image(image_path))
    found: list = []
    features = _analyse(img, found)
    regions = _select_regions(found, (img.shape[1], img.shape[0]), original_size(image_path), max_regions)
    return features, regions


def locate_regions(image_path: str | Path, max_regions: int = MAX_REGIONS) -> list[dict]:
    """Anomalous regions of one page (see extract_with_regions)."""
    return extract_with_regions(image_path, max_regions)[1]
