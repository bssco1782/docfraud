"""
core/document_analyzer.py
Document fraud detection: ELA analysis and feature extraction.

This module provides image-analysis signals for detecting potential
document tampering. The computed scores and features are statistical
measurements of image properties — they are NOT fraud probabilities,
confidence percentages, or final fraud verdicts.  A high score means
the image has properties associated with JPEG re-compression artefacts
or editing, not that the document is fraudulent.

Supported formats: JPEG, PNG.  PDF (page 1 only) via optional pymupdf.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

# ──────────────────────────────────────────────────────────────
# Constants — change these to tune analysis globally
# ──────────────────────────────────────────────────────────────
MAX_IMAGE_SIDE: int = 2000        # px; longest side is capped here
MIN_IMAGE_SIDE: int = 64          # px; images smaller than this are still processed
ELA_QUALITY: int = 90             # JPEG quality for ELA reference re-save
ELA_SCALE: int = 15               # amplification factor applied to ELA difference
ELA_HIGH_THRESHOLD: int = 50      # uint8 ELA value above which a pixel is "hot"
ELA_BLOCK_SIZE: int = 32          # px; grid cell size for block-level statistics
ELA_HOT_BLOCK_THRESHOLD: float = 0.75  # fraction of ELA_HIGH_THRESHOLD that flags a block
ELA_MIN_COMPONENT_AREA: int = 20  # px²; minimum connected-component area for hotspot count

NOISE_BLOCK_SIZE: int = 16        # px; cell size for per-block noise inconsistency
BLOCKINESS_GRID: int = 8          # px; standard JPEG block boundary size
ORB_N_KEYPOINTS: int = 500        # maximum ORB keypoints for copy-move detection
ORB_MATCH_RATIO: float = 0.75     # Lowe's ratio test threshold for ORB matching
ORB_MIN_SPATIAL_DIST: int = 15    # px; minimum spatial separation to reject trivial matches

# Software tags that indicate post-capture editing
EDITING_SOFTWARE_KEYWORDS = frozenset(
    {"photoshop", "canva", "gimp", "editor", "paint", "affinity",
     "lightroom", "capture one", "snapseed"}
)

# Fixed ordering used by features_to_vector.
# Day 4: expanded from 15 → 22 features.  Existing model (fraud_model.pkl)
# was trained on the first 15; Day 5 retraining required.
FEATURE_NAMES: list[str] = [
    # — ELA statistics (5) —
    "ela_mean",
    "ela_std",
    "ela_max",
    "ela_p95",
    "ela_high_ratio",
    # — Localisation (2) —
    "ela_block_max_ratio",
    "ela_hotspot_count",
    # — Noise / sharpness (3) —
    "noise_std",
    "noise_block_inconsistency",
    "laplacian_var",
    # — Compression (2) —
    "blockiness",
    "bytes_per_pixel",
    # — Fraud-specific (3) —
    "copy_move_matches",
    "exif_edit_flag",
    "text_height_std",
    # — Image geometry (3) — Day 4
    "image_width",
    "image_height",
    "aspect_ratio",
    # — Intensity / contrast (2) — Day 4
    "grayscale_mean",
    "grayscale_std",
    # — Edge / texture (2) — Day 4
    "edge_density",
    "dark_pixel_ratio",
]


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> Image.Image:
    """Load an image file and return it as an RGB PIL Image.

    Supports JPEG, PNG, and PDF (page 1 rendered at 200 DPI).
    Converts grayscale and RGBA to RGB.  Resizes so that the longest
    side does not exceed MAX_IMAGE_SIDE while preserving aspect ratio.
    Images whose longest side is below MIN_IMAGE_SIDE are accepted
    without upscaling.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        A PIL Image in RGB mode, with longest side ≤ MAX_IMAGE_SIDE.

    Raises:
        ValueError: File does not exist, is corrupt, or has an
            unsupported extension.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        img = _load_pdf_page(path)
    elif suffix in {".jpg", ".jpeg", ".png"}:
        try:
            img = Image.open(path)
            img.load()  # force decode — catches corrupt files
        except Exception as exc:
            raise ValueError(f"Cannot open image '{path}': {exc}") from exc
    else:
        raise ValueError(
            f"Unsupported file type '{suffix}' for '{path}'. "
            "Supported: .jpg, .jpeg, .png, .pdf"
        )

    # Normalise colour mode
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Downsize if needed
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def ela_analysis(
    image_path: str | Path,
    quality: int = ELA_QUALITY,
    scale: int = ELA_SCALE,
) -> tuple[np.ndarray, float]:
    """Perform Error Level Analysis (ELA) on a document image.

    ELA works by re-saving the image as JPEG at a known quality level
    and measuring the absolute pixel-by-pixel difference from the
    original.  Regions that were already compressed at a similar
    quality (authentic areas) show little error; regions that were
    composited from a different source, or re-saved multiple times,
    tend to show higher error because their original compression grid
    is disrupted.

    Processing steps:
      1. Load and resize via load_image().
      2. Re-save to an in-memory JPEG buffer at ``quality``.
      3. Reload the re-saved JPEG.
      4. Compute absolute channel-wise difference, amplify by ``scale``.
      5. Convert to uint8 and derive the composite score.

    Note for PNG inputs: PNGs are losslessly encoded.  On first ELA
    the difference measures the JPEG quantisation artefacts rather
    than re-encoding error, so scores tend to be systematically higher.
    This is expected and documented here to avoid misinterpretation.

    Score formula (documented for Day-3 training):
        ela_mean_norm   = clip(mean(diff) * scale, 0, 255) / 255 * 40
        high_ratio_norm = clip(high_ratio * 200, 0, 100) * 0.35
        block_max_norm  = clip(block_max_mean / max(global_mean, 1),
                               0, 5) / 5 * 25
        score           = ela_mean_norm + high_ratio_norm + block_max_norm
        score           = clip(score, 0, 100)

    Weights (40 / 35 / 25) reflect approximate empirical sensitivity:
    mean error is the most stable signal; the hotspot ratio adds
    sensitivity to localised edits; block concentration catches
    copy-paste regions not visible in the global mean.

    LIMITATION — this score is an image-analysis signal.  It is NOT:
      • a probability of fraud
      • a confidence percentage
      • a final fraud verdict
      • proof that a document is fraudulent

    High scores on scanned documents, photos of photos, and heavily
    compressed originals are common false positives.

    Args:
        image_path: Path to the image file (JPEG or PNG).
        quality: JPEG quality for the reference re-save (default 90).
        scale: Amplification factor applied to the difference image
            (default 15).  Higher values increase contrast on the
            heatmap but do not change the underlying error.

    Returns:
        A tuple ``(heatmap, score)`` where:
        - ``heatmap`` is a uint8 numpy array of shape (H, W, 3),
          representing amplified per-pixel error in RGB space.
        - ``score`` is a float in [0, 100] summarising the error level.
    """
    img = load_image(image_path)
    img_arr = np.array(img, dtype=np.float32)

    # Re-save as JPEG in memory
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    recompressed.load()
    recomp_arr = np.array(recompressed, dtype=np.float32)

    # Absolute difference, amplified
    diff = np.abs(img_arr - recomp_arr) * scale
    diff_clipped = np.clip(diff, 0, 255).astype(np.uint8)

    # Scalar statistics for score
    diff_mean = float(np.mean(diff))
    high_ratio = float(np.mean(diff > ELA_HIGH_THRESHOLD))

    # Block-level max mean
    gray_diff = diff_clipped.mean(axis=2)  # (H, W) float
    global_mean = float(gray_diff.mean()) if gray_diff.mean() > 0 else 1.0
    block_max_mean = _block_max_mean(gray_diff, ELA_BLOCK_SIZE)

    # Score formula
    ela_mean_norm = np.clip(diff_mean * scale, 0, 255) / 255.0 * 40.0
    high_ratio_norm = np.clip(high_ratio * 200, 0, 100) * 0.35
    block_max_norm = np.clip(block_max_mean / max(global_mean, 1.0), 0, 5.0) / 5.0 * 25.0
    score = float(np.clip(ela_mean_norm + high_ratio_norm + block_max_norm, 0.0, 100.0))

    return diff_clipped, score


def extract_features(image_path: str | Path) -> dict[str, float]:
    """Extract deterministic image-analysis features (22 as of Day 4).

    Every value is a finite Python float (no NaN, no infinity).
    When a feature cannot be computed (e.g. no text components found),
    a documented fallback value of 0.0 is returned rather than raising.

    Features (in FEATURE_NAMES order):

    ELA statistics (from ela_analysis):
      ela_mean          — Mean amplified ELA pixel value across all
                          channels.  Reflects average re-encoding error.
      ela_std           — Standard deviation of amplified ELA values.
                          High std suggests localised rather than uniform
                          artefacts.
      ela_max           — Maximum per-pixel ELA value (scalar).
      ela_p95           — 95th percentile ELA value; more robust to
                          isolated bright spots than ela_max.
      ela_high_ratio    — Fraction of pixels with ELA value above
                          ELA_HIGH_THRESHOLD.  Proxy for tampered area.

    Localisation:
      ela_block_max_ratio — Ratio of the hottest ELA_BLOCK_SIZE×ELA_BLOCK_SIZE
                            block mean to the global mean.  Values >> 1
                            indicate a localised high-error region, which
                            is consistent with a composited patch.
      ela_hotspot_count   — Number of connected components in the
                            thresholded ELA grayscale map whose area
                            exceeds ELA_MIN_COMPONENT_AREA.  Counts
                            distinct spatially-separated anomalous regions.

    Noise and sharpness:
      noise_std               — Standard deviation of the residual signal
                                after applying a 3×3 median blur.  Measures
                                high-frequency noise not attributable to
                                image structure.
      noise_block_inconsistency — Standard deviation of per-block noise
                                  estimates (cells of NOISE_BLOCK_SIZE).
                                  Uniform documents have consistent noise;
                                  composited regions break this uniformity.
      laplacian_var           — Variance of the Laplacian of the grayscale
                                image.  High values = sharp image; low
                                values = blurry.  An unusually blurry region
                                in a sharp document may indicate airbrushing.

    Compression:
      blockiness      — Mean absolute difference across 8×8 JPEG block
                        boundaries relative to within-block differences.
                        High blockiness indicates heavy JPEG compression;
                        inconsistent blockiness can indicate a splice.
      bytes_per_pixel — Image JPEG encoding size (at quality 85) divided
                        by total pixel count.  Proxy for image complexity
                        and compression history; re-saved images tend to
                        compress more efficiently (lower value).

    Fraud-specific:
      copy_move_matches — Count of ORB keypoint matches that survive
                          Lowe's ratio test AND have spatial separation
                          ≥ ORB_MIN_SPATIAL_DIST pixels.  Detects
                          copy-pasted regions within the same document.
                          EXPERIMENTAL: text documents contain many
                          repeated glyphs that produce false positives.
                          See _compute_copy_move_matches() for details.
      exif_edit_flag    — 1.0 if EXIF metadata contains a 'Software' tag
                          whose value (case-insensitive) contains any of
                          the keywords in EDITING_SOFTWARE_KEYWORDS
                          (photoshop, canva, gimp, editor, paint, …).
                          0.0 otherwise, including when EXIF is absent or
                          the Software tag is present but does not match.
                          LIMITATION: absence of EXIF is NOT treated as
                          evidence of tampering; many cameras and scanners
                          omit EXIF.  A 0.0 means "no editing software
                          detected", not "definitely unedited".
      text_height_std   — Standard deviation of connected-component
                          heights after Otsu binarisation, restricted to
                          components whose height is in the range expected
                          for text (5–100 px after noise filtering by
                          ELA_MIN_COMPONENT_AREA).  Documents with
                          consistent typography have low std; documents
                          with text at wildly different scales may
                          indicate cut-and-paste.  Returns 0.0 when fewer
                          than 2 valid components are detected.
                          LIMITATION: handwriting, low-DPI scans, and
                          images with little text produce unreliable values.

    Args:
        image_path: Path to the image file.

    Returns:
        dict with exactly len(FEATURE_NAMES) keys, all finite floats.

    Raises:
        ValueError: Propagated from load_image() for invalid inputs.
    """
    heatmap, ela_score = ela_analysis(image_path)
    img = load_image(image_path)
    img_arr = np.array(img, dtype=np.uint8)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    ela_gray = heatmap.mean(axis=2).astype(np.float32)

    features: dict[str, float] = {}

    # ── ELA statistics ─────────────────────────────────────────
    features["ela_mean"] = _safe_float(ela_gray.mean())
    features["ela_std"] = _safe_float(ela_gray.std())
    features["ela_max"] = _safe_float(float(ela_gray.max()))
    features["ela_p95"] = _safe_float(float(np.percentile(ela_gray, 95)))
    features["ela_high_ratio"] = _safe_float(
        float(np.mean(ela_gray > ELA_HIGH_THRESHOLD))
    )

    # ── Localisation ───────────────────────────────────────────
    global_mean = float(ela_gray.mean()) or 1.0
    block_max = _block_max_mean(ela_gray, ELA_BLOCK_SIZE)
    features["ela_block_max_ratio"] = _safe_float(block_max / global_mean)
    features["ela_hotspot_count"] = _safe_float(
        _count_hotspots(ela_gray, ELA_HIGH_THRESHOLD, ELA_MIN_COMPONENT_AREA)
    )

    # ── Noise and sharpness ────────────────────────────────────
    gray_f = gray.astype(np.float32)
    blurred = cv2.medianBlur(gray, 3).astype(np.float32)
    residual = gray_f - blurred
    features["noise_std"] = _safe_float(float(residual.std()))
    features["noise_block_inconsistency"] = _safe_float(
        _block_noise_inconsistency(gray, NOISE_BLOCK_SIZE)
    )
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    features["laplacian_var"] = _safe_float(float(lap.var()))

    # ── Compression ────────────────────────────────────────────
    features["blockiness"] = _safe_float(_compute_blockiness(gray))
    features["bytes_per_pixel"] = _safe_float(
        _compute_bytes_per_pixel(img)
    )

    # ── Fraud-specific ─────────────────────────────────────────
    features["copy_move_matches"] = _safe_float(
        _compute_copy_move_matches(gray)
    )
    features["exif_edit_flag"] = _safe_float(
        _compute_exif_edit_flag(image_path)
    )
    features["text_height_std"] = _safe_float(
        _compute_text_height_std(gray)
    )

    # ── Image geometry (Day 4) ────────────────────────────────
    h, w = img_arr.shape[:2]
    features["image_width"] = _safe_float(float(w))
    features["image_height"] = _safe_float(float(h))
    features["aspect_ratio"] = _safe_float(float(w) / max(float(h), 1.0))

    # ── Intensity / contrast (Day 4) ──────────────────────────
    features["grayscale_mean"] = _safe_float(float(gray.mean()))
    features["grayscale_std"] = _safe_float(float(gray.std()))

    # ── Edge / texture (Day 4) ────────────────────────────────
    edges = cv2.Canny(gray, 100, 200)
    features["edge_density"] = _safe_float(
        float(np.count_nonzero(edges)) / max(float(edges.size), 1.0)
    )
    features["dark_pixel_ratio"] = _safe_float(
        float(np.mean(gray < 50))
    )

    # Guarantee exactly FEATURE_NAMES keys, all finite
    assert set(features.keys()) == set(FEATURE_NAMES), (
        f"Feature key mismatch: {set(features.keys()) ^ set(FEATURE_NAMES)}"
    )
    return {k: features[k] for k in FEATURE_NAMES}


def features_to_vector(features_dict: dict[str, float]) -> list[float]:
    """Convert a features dict to a fixed-order list for ML consumption.

    The order is determined by FEATURE_NAMES.  Missing keys default to
    0.0.  This function is the contract between this module and
    training — the order must never change without versioning FEATURE_NAMES.

    Args:
        features_dict: Dict as returned by extract_features().

    Returns:
        List of floats in FEATURE_NAMES order.
    """
    return [float(features_dict.get(k, 0.0)) for k in FEATURE_NAMES]


# ──────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────

def _load_pdf_page(path: Path) -> Image.Image:
    """Render page 0 of a PDF at 200 DPI using pymupdf.

    pymupdf is imported lazily so the module works without it; a clear
    error is raised only when a PDF path is actually passed.
    """
    try:
        import pymupdf  # noqa: PLC0415 — intentional lazy import
    except ImportError as exc:
        raise ValueError(
            "PDF support requires pymupdf. "
            "Install it with: pip install pymupdf"
        ) from exc
    try:
        doc = pymupdf.open(str(path))
        page = doc[0]
        mat = pymupdf.Matrix(200 / 72, 200 / 72)  # 200 DPI
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)
        doc.close()
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    except Exception as exc:
        raise ValueError(f"Cannot render PDF '{path}': {exc}") from exc


def _block_max_mean(arr: np.ndarray, block_size: int) -> float:
    """Return the mean of the block with the highest average value.

    Tiles ``arr`` into non-overlapping ``block_size``×``block_size``
    cells and returns the mean of the cell with the maximum mean.
    """
    h, w = arr.shape
    max_mean = 0.0
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            block = arr[r: r + block_size, c: c + block_size]
            bm = float(block.mean())
            if bm > max_mean:
                max_mean = bm
    return max_mean


def _count_hotspots(
    arr: np.ndarray,
    threshold: int,
    min_area: int,
) -> int:
    """Count connected components in thresholded ``arr`` above ``threshold``.

    Components smaller than ``min_area`` pixels are excluded to suppress
    noise from isolated hot pixels.
    """
    binary = (arr > threshold).astype(np.uint8)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    count = 0
    for label in range(1, n_labels):  # label 0 is background
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            count += 1
    return count


def _block_noise_inconsistency(gray: np.ndarray, block_size: int) -> float:
    """Std of per-block noise estimates.

    For each non-overlapping cell of ``block_size``×``block_size``,
    compute the residual std (gray minus median-blurred).  Returns the
    std of these per-block values — 0.0 means uniform noise across the
    image; high values indicate regions with very different noise levels.
    """
    h, w = gray.shape
    per_block: list[float] = []
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            block = gray[r: r + block_size, c: c + block_size]
            if block.size < 4:
                continue
            blurred = cv2.medianBlur(block, 3) if min(block.shape) >= 3 else block
            residual = block.astype(np.float32) - blurred.astype(np.float32)
            per_block.append(float(residual.std()))
    if len(per_block) < 2:
        return 0.0
    return float(np.std(per_block))


def _compute_blockiness(gray: np.ndarray) -> float:
    """Measure JPEG block boundary artefacts on an 8×8 grid.

    Computes the mean absolute difference across horizontal and vertical
    JPEG block boundaries (multiples of BLOCKINESS_GRID) versus the mean
    absolute difference of non-boundary pixel pairs, then returns the
    boundary-to-interior ratio.  A ratio near 1 means uniform; >> 1
    indicates heavy JPEG compression artefacts.
    """
    h, w = gray.shape
    gray_f = gray.astype(np.float32)

    boundary_diffs: list[float] = []
    interior_diffs: list[float] = []

    # Horizontal boundaries
    for r in range(BLOCKINESS_GRID, h, BLOCKINESS_GRID):
        if r >= h:
            break
        diff_row = float(np.mean(np.abs(
            gray_f[r, :] - gray_f[r - 1, :]
        )))
        boundary_diffs.append(diff_row)

    # Vertical boundaries
    for c in range(BLOCKINESS_GRID, w, BLOCKINESS_GRID):
        if c >= w:
            break
        diff_col = float(np.mean(np.abs(
            gray_f[:, c] - gray_f[:, c - 1]
        )))
        boundary_diffs.append(diff_col)

    # Interior (non-boundary) horizontal pairs
    for r in range(1, h):
        if r % BLOCKINESS_GRID != 0:
            interior_diffs.append(float(np.mean(np.abs(
                gray_f[r, :] - gray_f[r - 1, :]
            ))))
        if len(interior_diffs) >= 200:
            break

    boundary_mean = float(np.mean(boundary_diffs)) if boundary_diffs else 0.0
    interior_mean = float(np.mean(interior_diffs)) if interior_diffs else 1.0
    return boundary_mean / max(interior_mean, 1e-6)


def _compute_bytes_per_pixel(img: Image.Image) -> float:
    """Encode ``img`` as JPEG at quality 85 and return bytes-per-pixel.

    This is a proxy for image complexity and compression history.
    Images that have already been JPEG-compressed tend to compress more
    efficiently (lower bytes-per-pixel) because high-frequency detail
    has been discarded.
    """
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    n_bytes = buf.tell()
    n_pixels = img.size[0] * img.size[1]
    return n_bytes / max(n_pixels, 1)


def _compute_copy_move_matches(gray: np.ndarray) -> float:
    """Count suspicious intra-image ORB keypoint matches.

    EXPERIMENTAL — text documents contain many repeated glyphs that
    produce false positive matches.  The count is useful as a relative
    feature but should not be interpreted as a direct indicator of
    copy-move forgery without calibration on document data.

    Algorithm:
      1. Detect up to ORB_N_KEYPOINTS ORB keypoints.
      2. Match all keypoints against themselves using a BFMatcher with
         Hamming distance (appropriate for binary ORB descriptors).
      3. Apply Lowe's ratio test (ORB_MATCH_RATIO) to keep only
         distinctive matches.
      4. Discard matches where both keypoints are spatially closer than
         ORB_MIN_SPATIAL_DIST pixels (same-character matches in text).
      5. Return the count of surviving matches.

    A returned value of 0 is legitimate when no suitable keypoints are
    found or no matches survive spatial filtering.  It does NOT mean
    the image is unedited.

    Args:
        gray: Grayscale uint8 numpy array.

    Returns:
        Float count of surviving matches (cast to float for consistency
        with the feature vector).
    """
    # ORB is deterministic when nfeatures and fastThreshold are fixed
    orb = cv2.ORB_create(
        nfeatures=ORB_N_KEYPOINTS,
        fastThreshold=10,
        edgeThreshold=10,
        patchSize=31,
        scaleFactor=1.2,
        nlevels=8,
        scoreType=cv2.ORB_HARRIS_SCORE,
        firstLevel=0,
        WTA_K=2,
    )
    kps, descs = orb.detectAndCompute(gray, None)

    if descs is None or len(descs) < 2:
        return 0.0

    # BFMatcher with Hamming — deterministic, no GPU
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # k=2 for ratio test
    matches = bf.knnMatch(descs, descs, k=2)

    count = 0
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        # Ratio test
        if m.distance >= ORB_MATCH_RATIO * n.distance:
            continue
        # Skip trivial self-match (same keypoint)
        if m.queryIdx == m.trainIdx:
            continue
        # Skip nearby keypoints (likely same character/glyph)
        kp1 = kps[m.queryIdx]
        kp2 = kps[m.trainIdx]
        dist = math.hypot(kp1.pt[0] - kp2.pt[0], kp1.pt[1] - kp2.pt[1])
        if dist < ORB_MIN_SPATIAL_DIST:
            continue
        count += 1

    return float(count)


def _compute_exif_edit_flag(image_path: str | Path) -> float:
    """Return 1.0 if EXIF Software tag matches known editing software.

    Rule: flag = 1.0 iff the 'Software' EXIF tag exists AND its value
    (case-insensitive) contains one of the keywords in
    EDITING_SOFTWARE_KEYWORDS.

    Return 0.0 when:
      - EXIF data is absent (not uncommon for scanned documents)
      - The Software tag is absent
      - The Software tag does not contain an editing software keyword
      - The image format does not support EXIF (e.g. some PNGs)

    LIMITATION:
      - EXIF can be stripped by any software, including non-editing
        tools; 0.0 does NOT mean the image is unedited.
      - EXIF can be added or forged; 1.0 does NOT prove editing.
      - This feature has limited utility on scanned documents which
        rarely embed meaningful EXIF.

    Args:
        image_path: Path to the image file.

    Returns:
        1.0 or 0.0.
    """
    try:
        img = Image.open(Path(image_path))
        exif_data = img._getexif()  # Returns None if absent
    except Exception:
        return 0.0

    if not exif_data:
        return 0.0

    # Build a tag-name→value dict
    tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}
    software = tag_map.get("Software", "")
    if not isinstance(software, str):
        try:
            software = software.decode("utf-8", errors="ignore")
        except Exception:
            return 0.0

    software_lower = software.lower()
    for keyword in EDITING_SOFTWARE_KEYWORDS:
        if keyword in software_lower:
            return 1.0
    return 0.0


def _compute_text_height_std(gray: np.ndarray) -> float:
    """Estimate variability of text-component heights via binarisation.

    How text-like regions are detected:
      1. Apply Otsu's threshold to binarise the grayscale image.
      2. Invert so dark text on light background becomes foreground.
      3. Find connected components (8-connectivity).
      4. Keep components whose bounding-box height falls in [5, 100] px
         AND whose area exceeds ELA_MIN_COMPONENT_AREA (noise filter).
      5. Return the std of the retained heights.

    What the std represents:
      Documents with consistent typography (e.g. a clean payslip) have
      small std — text glyphs are roughly the same height.  A document
      where text has been pasted from different sources or at different
      zoom levels will show a larger spread.

    Documented fallback:
      Returns 0.0 when fewer than 2 valid components are detected.
      This happens for blank pages, images with no text, very low-DPI
      scans, or images that are entirely text-free.

    LIMITATION:
      - Handwriting produces components of widely varying size even in
        genuine documents — high std is expected.
      - Low-DPI scans (<150 DPI) blur character boundaries, inflating
        component sizes.
      - Heavy JPEG compression merges adjacent characters into larger
        components.

    Args:
        gray: Grayscale uint8 numpy array.

    Returns:
        Std of text-component heights, or 0.0 if < 2 components found.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(binary)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    heights: list[int] = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area >= ELA_MIN_COMPONENT_AREA and 5 <= height <= 100:
            heights.append(height)

    if len(heights) < 2:
        return 0.0
    return float(np.std(heights))


def _safe_float(value: float) -> float:
    """Return ``value`` as a finite float, falling back to 0.0."""
    try:
        f = float(value)
        if math.isfinite(f):
            return f
        return 0.0
    except (TypeError, ValueError):
        return 0.0
