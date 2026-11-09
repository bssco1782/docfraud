"""
core/local_forensics.py
Local forensic analysis — schema ``local-forensics-v1``.

Why this exists
---------------
The Day-4 schema (``day4-v2-22``) summarises the WHOLE image.  Document
tampering is usually LOCAL (an edited amount, a pasted block, a blurred
field), so a whole-image average barely moves and the classifier ends up
learning document type/brightness instead of manipulation.

This module asks a different question:

    "Does one part of this document behave differently from the rest of
     the SAME document?"

Pipeline (deterministic, no randomness anywhere)
------------------------------------------------
1. Load safely (reuses ``document_analyzer.load_image``), apply EXIF
   orientation, keep resolution (capped at MAX_IMAGE_SIDE by the loader).
2. Build per-pixel forensic maps: ELA, noise residual, gradient,
   Laplacian (high-frequency energy), chroma, gray level.
3. Reduce each map to fine tiles (TILE px).  Noise-type signals use the
   tile MEDIAN: re-compression / noise touch most pixels of a tile, while
   text edges touch few — so text does not look like noise.
4. Score every tile against the SAME document with robust statistics
   (median / MAD), comparing a tile only with tiles of the same content
   class (flat / light-text / dense-text x dark / mid / light), where the
   class is computed on a denoised copy so an edit cannot move a tile out
   of its own class.
5. Summarise tile anomalies into a fixed-length feature vector.
6. Group tiles into a GRID x GRID region grid for localisation evidence.

All scores are forensic anomaly measurements, NOT fraud probabilities.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.document_analyzer import load_image

FEATURE_SCHEMA_VERSION = "local-forensics-v1"

TILE = 16                 # px; fine tile for anomaly scoring
GRID = 4                  # GRID x GRID regions for localisation
ELA_QUALITY = 90          # JPEG quality for ELA reference re-save
Z_CLIP = 20.0             # anomaly scores are clipped to [0, Z_CLIP]
ANOMALY_Z = 4.0           # |z| above this = anomalous tile
FLAT_CONTRAST = 2.0       # denoised gray std below this = flat tile
INK_RANGE = 50.0          # p95 - p5 gray above this = tile contains ink (sharpness defined)
INTENSITY_EDGES = (85.0, 170.0)  # dark / mid / light tiles
MIN_BIN_TILES = 8         # classes smaller than this fall back to document stats
SPREAD_FLOOR = 0.1        # scale >= SPREAD_FLOOR * (p99 - p1) of the document

# Signal -> (absolute scale floor, compared within content class?).
# Floors stop divide-by-~0 on perfectly flat synthetic backgrounds.
SIGNALS: dict[str, tuple[float, bool]] = {
    "ela": (0.5, True),         # ELA: tile median |gray - JPEG90(gray)|
    "noise": (0.5, True),       # noise: tile median |gray - median3x3(gray)|
    "sharpness": (0.05, False),  # edge: p95 gradient / (ink range + 1), ink tiles only
    "gradient": (1.0, True),    # edge: tile mean Sobel magnitude
    "hf_energy": (1.0, True),   # texture: tile mean |Laplacian|
    "chroma": (1.0, True),      # colour: chroma of the tile's lightest 25% ("paper colour")
    "intensity": (2.0, True),   # intensity: tile mean gray (local brightness)
    "contrast": (1.0, False),   # intensity: tile gray std (local contrast)
}
# Signals that carry manipulation evidence rather than layout; used for localisation.
FORENSIC_SIGNALS = ("ela", "noise", "sharpness", "chroma")

_SUMMARIES = ["max", "p99", "p95", "mean", "anom_frac", "neighbor_incons"]
_REGION_FEATURES = [
    "region_top_score",
    "region_top_median_ratio",
    "region_anomalous_count",
    "anomaly_concentration",
]

FEATURE_NAMES: list[str] = [
    f"lf_{sig}_{summ}" for sig in SIGNALS for summ in _SUMMARIES
] + [f"lf_{name}" for name in _REGION_FEATURES]

_EXIF_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


# ──────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────

def load_oriented(path: str | Path) -> Image.Image:
    """Load via the canonical loader, then apply EXIF orientation."""
    img = load_image(path)
    try:
        with Image.open(path) as raw:
            orientation = raw.getexif().get(0x0112, 1)
    except Exception:
        orientation = 1
    op = _EXIF_TRANSPOSE.get(orientation)
    return img.transpose(op) if op is not None else img


# ──────────────────────────────────────────────────────────────
# Tiling
# ──────────────────────────────────────────────────────────────

def _tile_size(h: int, w: int) -> int:
    """Tile edge length in pixels; keeps at least 2 tiles per region per side on small images."""
    return int(min(TILE, max(2, min(h, w) // (GRID * 2))))


def _tiles(arr: np.ndarray, t: int) -> np.ndarray:
    """(H, W) -> (rows, cols, t*t) over complete tiles."""
    rows, cols = arr.shape[0] // t, arr.shape[1] // t
    a = arr[: rows * t, : cols * t]
    return a.reshape(rows, t, cols, t).transpose(0, 2, 1, 3).reshape(rows, cols, t * t)


def tile_signals(img: Image.Image) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    """Per-tile signal grids for every SIGNALS name, the content class grid, and tile size."""
    rgb = np.asarray(img, dtype=np.uint8)
    gray_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32)
    # Luminance-only ELA: colour ELA fires on saturated design colours
    # (JPEG chroma subsampling), which is layout, not tampering.
    buf = io.BytesIO()
    Image.fromarray(gray_u8).save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    with Image.open(buf) as resaved:
        ela = np.abs(gray - np.asarray(resaved, dtype=np.float32))
    noise = np.abs(gray - cv2.medianBlur(gray_u8, 3).astype(np.float32))
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    chroma = np.hypot(ycc[..., 1] - 128.0, ycc[..., 2] - 128.0)
    denoised = cv2.medianBlur(gray_u8, 5).astype(np.float32)

    t = _tile_size(*gray.shape)
    g = _tiles(gray, t)
    p5, p75, p95 = np.percentile(g, [5, 75, 95], axis=2)
    ink_range = p95 - p5
    grad_t = _tiles(grad, t)
    sharp = np.percentile(grad_t, 95, axis=2) / (ink_range + 1.0)
    light = g >= p75[..., None]
    chroma_t = _tiles(chroma, t)

    signals = {
        "ela": np.median(_tiles(ela, t), axis=2),
        "noise": np.median(_tiles(noise, t), axis=2),
        "sharpness": np.where(ink_range > INK_RANGE, sharp, np.nan),
        "gradient": grad_t.mean(axis=2),
        "hf_energy": _tiles(lap, t).mean(axis=2),
        "chroma": (chroma_t * light).sum(axis=2) / light.sum(axis=2),
        "intensity": g.mean(axis=2),
        "contrast": g.std(axis=2),
    }
    classes = content_classes(_tiles(denoised, t))
    return {k: v.astype(np.float64) for k, v in signals.items()}, classes, t


def content_classes(denoised_tiles: np.ndarray) -> np.ndarray:
    """{flat, light text, dense text} x {dark, mid, light} per tile."""
    contrast = denoised_tiles.std(axis=2)
    intensity = denoised_tiles.mean(axis=2)
    textured = contrast[contrast >= FLAT_CONTRAST]
    split = float(np.median(textured)) if textured.size else FLAT_CONTRAST
    c_bin = np.where(contrast < FLAT_CONTRAST, 0, np.where(contrast < split, 1, 2))
    i_bin = np.searchsorted(np.asarray(INTENSITY_EDGES), intensity, side="right")
    return c_bin * 3 + i_bin


# ──────────────────────────────────────────────────────────────
# Anomaly scoring (robust, within the same document)
# ──────────────────────────────────────────────────────────────

def robust_z(values: np.ndarray, floor: float) -> np.ndarray:
    """|x - median| / max(1.4826 * MAD, floor), clipped to [0, Z_CLIP].  NaN in -> 0 out."""
    out = np.zeros(values.shape, dtype=np.float64)
    valid = np.isfinite(values)
    if not valid.any():
        return out
    v = values[valid]
    med = np.median(v)
    scale = max(1.4826 * np.median(np.abs(v - med)), floor)
    out[valid] = np.clip(np.abs(v - med) / scale, 0.0, Z_CLIP)
    return out


def _scale_floor(values: np.ndarray, abs_floor: float) -> float:
    """Absolute floor, raised for documents whose signal legitimately varies
    a lot (coloured headers, tables) so layout is not scored as tampering."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return abs_floor
    p1, p99 = np.percentile(v, [1, 99])
    return max(abs_floor, SPREAD_FLOOR * float(p99 - p1))


def anomaly_maps(signals: dict[str, np.ndarray], classes: np.ndarray) -> dict[str, np.ndarray]:
    """Per-tile |z| of each signal vs. the same document (within content class)."""
    out: dict[str, np.ndarray] = {}
    for name, (abs_floor, conditioned) in SIGNALS.items():
        v = signals[name]
        floor = _scale_floor(v, abs_floor)
        z = robust_z(v, floor)  # document-level (and fallback for small classes)
        if conditioned:
            for c in np.unique(classes):
                mask = classes == c
                if np.isfinite(v[mask]).sum() >= MIN_BIN_TILES:
                    z[mask] = robust_z(v[mask], floor)
        out[name] = z
    return out


def _smooth(z: np.ndarray) -> np.ndarray:
    """Horizontal 1x3 mean: suppresses single-tile spikes (isolated glyph
    edges) without diluting one-line-high edits, which run horizontally."""
    return cv2.blur(z.astype(np.float32), (3, 1), borderType=cv2.BORDER_REFLECT).astype(np.float64)


def _neighbor_inconsistency(v: np.ndarray, abs_floor: float) -> float:
    """Max |tile - median of its 8 neighbours|, in robust-scale units."""
    v = np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 0.0)
    padded = np.pad(v, 1, mode="edge")
    rows, cols = v.shape
    neigh = np.stack([
        padded[1 + dy: 1 + dy + rows, 1 + dx: 1 + dx + cols]
        for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)
    ])
    diff = np.abs(v - np.median(neigh, axis=0))
    scale = max(1.4826 * np.median(np.abs(v - np.median(v))), _scale_floor(v, abs_floor))
    return float(min(diff.max() / scale, Z_CLIP))


def _region_slices(rows: int, cols: int) -> list[tuple[int, int, slice, slice]]:
    """Split a rows x cols tile grid into GRID x GRID regions as (i, j, row_slice, col_slice)."""
    r_edges = np.linspace(0, rows, GRID + 1).astype(int)
    c_edges = np.linspace(0, cols, GRID + 1).astype(int)
    return [
        (i, j, slice(r_edges[i], r_edges[i + 1]), slice(c_edges[j], c_edges[j + 1]))
        for i in range(GRID) for j in range(GRID)
    ]


def forensic_map(zmaps: dict[str, np.ndarray]) -> np.ndarray:
    """Per-tile combined evidence: strongest smoothed FORENSIC signal."""
    return np.max(np.stack([_smooth(zmaps[s]) for s in FORENSIC_SIGNALS]), axis=0)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def extract_local_features(image_path: str | Path) -> dict[str, float]:
    """Return ``local-forensics-v1`` features (FEATURE_NAMES order, finite floats)."""
    signals, classes, _ = tile_signals(load_oriented(image_path))
    return features_from_signals(signals, classes)


def features_from_signals(signals: dict[str, np.ndarray], classes: np.ndarray) -> dict[str, float]:
    """Summarise per-tile anomaly maps into the local-forensics feature dictionary."""
    zmaps = anomaly_maps(signals, classes)
    feats: dict[str, float] = {}
    for name, (abs_floor, _) in SIGNALS.items():
        z = zmaps[name]
        zs = _smooth(z)
        feats[f"lf_{name}_max"] = float(zs.max())
        feats[f"lf_{name}_p99"] = float(np.percentile(zs, 99))
        feats[f"lf_{name}_p95"] = float(np.percentile(zs, 95))
        feats[f"lf_{name}_mean"] = float(z.mean())
        feats[f"lf_{name}_anom_frac"] = float((z > ANOMALY_Z).mean())
        feats[f"lf_{name}_neighbor_incons"] = _neighbor_inconsistency(signals[name], abs_floor)

    combined = forensic_map(zmaps)
    slices = _region_slices(*combined.shape)
    region_scores = np.array([combined[rs, cs].max() for _, _, rs, cs in slices])
    anomalous = combined > ANOMALY_Z
    _, _, rs, cs = slices[int(region_scores.argmax())]
    feats["lf_region_top_score"] = float(region_scores.max())
    feats["lf_region_top_median_ratio"] = float(region_scores.max() / (np.median(region_scores) + 1.0))
    feats["lf_region_anomalous_count"] = float((region_scores > ANOMALY_Z).sum())
    feats["lf_anomaly_concentration"] = float(
        anomalous[rs, cs].sum() / anomalous.sum() if anomalous.any() else 0.0
    )

    assert list(feats) == FEATURE_NAMES
    return {k: float(np.nan_to_num(v, nan=0.0, posinf=Z_CLIP, neginf=0.0)) for k, v in feats.items()}


def localize(image_path: str | Path, min_score: float = ANOMALY_Z) -> list[dict]:
    """Suspicious regions of the document, strongest first.

    Each entry::

        {"region": [x, y, width, height],     # pixels, oriented image
         "grid_cell": [row, col],             # GRID x GRID cell
         "anomaly_score": 0.82,               # 0-1, uncalibrated
         "raw_score": 13.1,                   # robust-z units
         "severity": "low" | "medium" | "high",
         "signals": ["noise_anomaly", ...]}

    ``region`` is the bounding box of the anomalous tiles inside the cell.
    """
    signals, classes, t = tile_signals(load_oriented(image_path))
    zmaps = anomaly_maps(signals, classes)
    smoothed = {s: _smooth(zmaps[s]) for s in FORENSIC_SIGNALS}
    combined = forensic_map(zmaps)

    regions = []
    for i, j, rs, cs in _region_slices(*combined.shape):
        cell = combined[rs, cs]
        score = float(cell.max())
        if score < min_score:
            continue
        hot_r, hot_c = np.nonzero(cell >= min_score)
        r0, r1 = rs.start + hot_r.min(), rs.start + hot_r.max() + 1
        c0, c1 = cs.start + hot_c.min(), cs.start + hot_c.max() + 1
        responsible = sorted(
            (s for s in FORENSIC_SIGNALS if smoothed[s][rs, cs].max() >= min_score),
            key=lambda s: -smoothed[s][rs, cs].max(),
        )
        regions.append({
            "region": [int(c0 * t), int(r0 * t), int((c1 - c0) * t), int((r1 - r0) * t)],
            "grid_cell": [i, j],
            "anomaly_score": round(float(1.0 - np.exp(-score / 10.0)), 4),
            "raw_score": round(score, 3),
            "severity": "high" if score >= 12 else "medium" if score >= 8 else "low",
            "signals": [f"{s}_anomaly" for s in responsible],
        })
    return sorted(regions, key=lambda r: -r["raw_score"])
