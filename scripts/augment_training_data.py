"""
scripts/augment_training_data.py
Deterministic, TRAINING-ONLY tampering augmentation (``train-aug-v2.1``).

Sources are the GENUINE images of the frozen split's TRAIN list
(data/splits/day5_frozen_split.json).  Frozen test images are never read.

For every source image:
  * N_TAMPERED local manipulations of real text fields -> label "tampered"
  * 1 genuine variant -> label "genuine"

Whole-page processing (JPEG re-save, brightness, noise, resize) is
LABEL-INDEPENDENT.  v1 gave it only to genuine variants, which let a model
learn "processed = genuine" (CV: 23/38 genuine originals flagged).  v2 sets the
processed fraction of EACH CLASS in the full training pool (originals + augmented)
to the same value, drawing the whole-page op from the same distribution for both.

Local edits act on real ink tokens (words / numbers found by connected
components), match font size / ink colour / local background, and blend
through feathered masks — no hard rectangles, no fixed text, no fixed place.

Every row records: label, source + sha256, local and whole-page transforms,
parameters, bbox, seed, output sha256.  Same seed => identical files
(fonts come from the host; the manifest hashes detect any drift).

Usage:
    python scripts/augment_training_data.py            # generate
    python scripts/augment_training_data.py --check    # verify files vs manifest
    python scripts/augment_training_data.py --audit    # provenance / shortcut audit
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SPLIT_PATH = ROOT / "data" / "splits" / "day5_frozen_split.json"
DATASET_DIR = ROOT / "data" / "synthetic"
OUT_DIR = ROOT / "data" / "augmented" / "train_v2_1"
MANIFEST = OUT_DIR / "manifest.csv"
AUG_VERSION = "train-aug-v2.1"   # v1 invalid (label shortcut); v2.0 kept in data/augmented/train_v2 as history
BASE_SEED = 9000
N_TAMPERED = 2
MIN_CHANGE = 1.5  # mean |pixel change| inside the edit box; below = no-op edit, retry

LOCAL_OPS = [
    "text_replace_number", "text_replace_date", "text_replace_amount", "text_replace_account",
    "text_replace_id", "text_replace_name", "copy_move", "region_duplicate",
    "local_blur", "local_sharpen", "local_brightness", "local_contrast",
    "local_color", "noise_injection", "local_jpeg", "local_resample", "combo",
]
GLOBAL_OPS = ["global_jpeg", "global_brightness", "global_contrast", "global_noise",
              "global_resize", "global_combo"]
_BASE_GLOBAL = GLOBAL_OPS[:-1]
FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Meera", "Arjun", "Kavya", "Rahul", "Sneha",
               "Amit", "Pooja", "Karan", "Neha", "Suresh", "Divya", "Manoj", "Isha", "Nikhil", "Ritu"]
LAST_NAMES = ["Sharma", "Patel", "Iyer", "Reddy", "Gupta", "Nair", "Singh", "Das", "Mehta", "Rao",
              "Joshi", "Kulkarni", "Menon", "Bose", "Verma", "Pillai", "Shah", "Chopra", "Kapoor", "Jain"]
FONT_FILES = ["verdana.ttf", "tahoma.ttf", "arial.ttf", "segoeui.ttf"]  # + Pillow default
FIELDS = ["file", "label", "source_path", "source_sha256", "source_type", "transform", "params", "bbox",
          "global_transform", "global_params", "seed", "sha256", "aug_version"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _font(rng, size: int) -> tuple[ImageFont.ImageFont, str]:
    name = str(rng.choice(FONT_FILES + ["pillow_default"]))
    if name != "pillow_default":
        try:
            return ImageFont.truetype(name, size), name
        except OSError:
            pass
    return ImageFont.load_default(size=size), "pillow_default"


# ── text tokens (words / numbers) ───────────────────────────────────────────

def ink_tokens(img: Image.Image) -> list[tuple[int, int, int, int]]:
    """Word-level ink boxes (x, y, w, h): dark pixels merged horizontally."""
    gray = np.asarray(img.convert("L"))
    H = gray.shape[0]
    ink = (gray < 140).astype(np.uint8)
    merged = cv2.dilate(ink, np.ones((3, 9), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    boxes = []
    for x, y, w, h, area in stats[1:]:
        if 8 <= h <= 40 and 12 <= w <= 320 and y + h < H - 40 and area > 0.2 * w * h:
            boxes.append((int(x), int(y), int(w), int(h)))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _ring_background(a: np.ndarray, box) -> tuple[int, int, int]:
    """Median colour of a 3-px ring around the box: what an eraser/clone tool samples."""
    x, y, w, h = box
    H, W = a.shape[:2]
    x0, y0, x1, y1 = max(0, x - 3), max(0, y - 3), min(W, x + w + 3), min(H, y + h + 3)
    ring = np.concatenate([a[y0:y, x0:x1].reshape(-1, 3), a[y + h:y1, x0:x1].reshape(-1, 3),
                           a[y0:y1, x0:x].reshape(-1, 3), a[y0:y1, x + w:x1].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(ring, axis=0)) if len(ring) else (255, 255, 255)


def _ink_colour(a: np.ndarray, box) -> tuple[int, int, int]:
    x, y, w, h = box
    crop = a[y:y + h, x:x + w].reshape(-1, 3)
    dark = crop[crop.mean(axis=1) < 140]
    return tuple(int(v) for v in np.median(dark, axis=0)) if len(dark) else (30, 30, 30)


def _content(rng, kind: str) -> str:
    if kind == "number":
        return "".join(str(int(d)) for d in rng.integers(0, 10, size=int(rng.integers(3, 9))))
    if kind == "date":
        sep = str(rng.choice(["/", "-", " "]))
        return f"{int(rng.integers(1, 29)):02d}{sep}{int(rng.integers(1, 13)):02d}{sep}20{int(rng.integers(18, 27))}"
    if kind == "amount":
        v = int(rng.integers(500, 2_500_000))
        return f"{v:,}" + (".00" if rng.random() < 0.5 else "")
    if kind == "id":
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if rng.random() < 0.5:  # PAN-style
            return "".join(letters[int(i)] for i in rng.integers(0, 26, 5)) +                 "".join(str(int(d)) for d in rng.integers(0, 10, 4)) + letters[int(rng.integers(26))]
        return " ".join("".join(str(int(d)) for d in rng.integers(0, 10, 4)) for _ in range(3))  # Aadhaar-style
    if kind == "name":
        return f"{FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]} {LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]}"
    return f"XXXX-XXXX-{int(rng.integers(0, 10**6)):06d}" if rng.random() < 0.5 else \
        "".join(str(int(d)) for d in rng.integers(0, 10, size=int(rng.integers(9, 15))))


def _text_replace(img: Image.Image, box, rng, kind: str) -> tuple[Image.Image, dict]:
    a = np.asarray(img)
    bg, ink = _ring_background(a, box), _ink_colour(a, box)
    x, y, w, h = box
    text = _content(rng, kind)
    size = max(8, int(round(h * rng.uniform(0.95, 1.25))))
    font, font_name = _font(rng, size)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x - 1, y - 1, x + w, y + h], fill=bg)
    top = font.getbbox(text)[1]
    draw.text((x, y - top + int(rng.integers(-1, 2))), text, font=font, fill=ink)
    return img, {"text": text, "font": font_name, "font_size": size, "ink": list(ink), "bg": list(bg)}


def _copy_move(img: Image.Image, box, rng, tokens) -> tuple[Image.Image, dict] | None:
    x, y, w, h = box
    cands = [t for t in tokens if t != box and abs(t[3] - h) <= 3 and abs(t[1] - y) > h]
    if not cands:
        return None
    sx, sy, sw, sh = cands[int(rng.integers(len(cands)))]
    a = np.asarray(img)
    ImageDraw.Draw(img).rectangle([x - 1, y - 1, x + w, y + h], fill=_ring_background(a, box))
    img.paste(img.crop((sx, sy, sx + sw, sy + sh)), (x, y))
    return img, {"source_token": [sx, sy, sw, sh]}


def _region_duplicate(img: Image.Image, token, rng, tokens):
    """Clone a whole field region (text + its background) from another line onto this one."""
    fx, fy, fw, fh = _field_box(img, token, rng)
    x, y, w, h = token
    cands = [t for t in tokens if t != token and abs(t[3] - h) <= 3 and abs(t[1] - y) > h]
    if not cands:
        return None
    sx, sy, _, _ = cands[int(rng.integers(len(cands)))]
    W, H = img.size
    ox = int(np.clip(sx - (x - fx), 0, W - fw))
    oy = int(np.clip(sy - (y - fy), 0, H - fh))
    img.paste(img.crop((ox, oy, ox + fw, oy + fh)), (fx, fy))
    return img, {"source_region": [ox, oy, fw, fh]}, [fx, fy, fw, fh]


def _feathered(img: Image.Image, edited: Image.Image, box, rng) -> tuple[Image.Image, float]:
    """Blend an edited crop back through a Gaussian-feathered mask (no hard edge)."""
    x, y, w, h = box
    sigma = float(rng.uniform(1.0, 3.0))
    mask = np.zeros((h, w), np.float32)
    m = max(1, int(round(sigma)))
    mask[m:h - m, m:w - m] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)[..., None]
    base = np.asarray(img.crop((x, y, x + w, y + h)), np.float32)
    out = base * (1 - mask) + np.asarray(edited, np.float32) * mask
    img.paste(Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8)), (x, y))
    return img, sigma


def _field_box(img, token, rng) -> tuple[int, int, int, int]:
    W, H = img.size
    x, y, w, h = token
    mx, my = int(rng.integers(2, 21)), int(rng.integers(2, 9))
    x0, y0 = max(0, x - mx), max(0, y - my)
    return x0, y0, min(W, x + w + mx) - x0, min(H, y + h + my) - y0


def _pixel_op(crop: Image.Image, op: str, rng) -> tuple[Image.Image, dict]:
    w, h = crop.size
    if op == "local_blur":
        r = float(rng.uniform(0.6, 3.0)); return crop.filter(ImageFilter.GaussianBlur(r)), {"radius": r}
    if op == "local_sharpen":
        pct = int(rng.integers(60, 250)); return crop.filter(ImageFilter.UnsharpMask(2, pct, 2)), {"percent": pct}
    if op == "local_brightness":
        f = float(rng.choice([rng.uniform(0.7, 0.95), rng.uniform(1.05, 1.3)]))
        return ImageEnhance.Brightness(crop).enhance(f), {"factor": f}
    if op == "local_contrast":
        f = float(rng.choice([rng.uniform(0.5, 0.85), rng.uniform(1.15, 1.7)]))
        return ImageEnhance.Contrast(crop).enhance(f), {"factor": f}
    if op == "local_color":
        shift = rng.integers(-30, 31, size=3)
        a = np.clip(np.asarray(crop, np.int16) + shift, 0, 255).astype(np.uint8)
        return Image.fromarray(a), {"rgb_shift": [int(v) for v in shift]}
    if op == "noise_injection":
        sigma = float(rng.uniform(2, 15))
        a = np.asarray(crop, np.float32) + rng.normal(0, sigma, size=(h, w, 1))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), {"sigma": sigma}
    if op == "local_jpeg":
        q = int(rng.integers(20, 76)); buf = io.BytesIO(); crop.save(buf, "JPEG", quality=q)
        buf.seek(0); return Image.open(buf).convert("RGB"), {"quality": q}
    if op == "local_resample":
        f = float(rng.uniform(0.35, 0.75))
        small = crop.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
        up = Image.NEAREST if rng.random() < 0.5 else Image.BILINEAR
        return small.resize((w, h), up), {"factor": f, "upsample": "nearest" if up == Image.NEAREST else "bilinear"}
    raise ValueError(op)


def _apply_local(img: Image.Image, op: str, token, tokens, rng):
    if op.startswith("text_replace_"):
        img, p = _text_replace(img, token, rng, op.rsplit("_", 1)[1])
        return img, p, list(token)
    if op == "copy_move":
        out = _copy_move(img, token, rng, tokens)
        if out is None:
            return None
        return out[0], out[1], list(token)
    if op == "region_duplicate":
        return _region_duplicate(img, token, rng, tokens)
    box = _field_box(img, token, rng)
    x, y, w, h = box
    edited, p = _pixel_op(img.crop((x, y, x + w, y + h)), op, rng)
    img, sigma = _feathered(img, edited, box, rng)
    p["feather_sigma"] = sigma
    return img, p, list(box)


def _changed(a: Image.Image, b: Image.Image, box) -> float:
    x, y, w, h = box
    ca = np.asarray(a.crop((x, y, x + w, y + h)), np.int16)
    cb = np.asarray(b.crop((x, y, x + w, y + h)), np.int16)
    return float(np.abs(ca - cb).mean())


def make_tampered(src: Image.Image, rng, op: str) -> tuple[Image.Image, str, dict, list[int]]:
    """One realistic local edit of a real text field; retried until pixels visibly change."""
    tokens = ink_tokens(src)
    for _ in range(30):
        img = src.copy()
        token = tokens[int(rng.integers(len(tokens)))]
        if op == "combo":
            first = str(rng.choice([o for o in LOCAL_OPS if o.startswith("text_replace_")] + ["copy_move", "region_duplicate"]))
            after = str(rng.choice(["local_jpeg", "local_resample", "local_blur", "noise_injection"]))
            out = _apply_local(img, first, token, tokens, rng)
            if out is None:
                continue
            img, p1, bbox = out
            img, p2, bbox2 = _apply_local(img, after, token, tokens, rng)
            name, params = f"combo:{first}+{after}", {first: p1, after: p2}
            bbox = bbox2
        else:
            out = _apply_local(img, op, token, tokens, rng)
            if out is None:
                continue
            img, params, bbox = out
            name = op
        change = _changed(src, img, bbox)
        if change >= MIN_CHANGE:
            params["mean_abs_change"] = round(change, 3)
            return img, name, params, bbox
    raise RuntimeError(f"could not produce a visible '{op}' edit")


def whole_page(img: Image.Image, rng, op: str) -> tuple[Image.Image, str, dict]:
    """Label-independent, mild whole-page processing (scan / re-save / resize)."""
    if op == "global_combo":
        ops = [str(o) for o in rng.choice(_BASE_GLOBAL, size=2, replace=False)]
        params = {}
        for o in ops:
            img, _, params[o] = whole_page(img, rng, o)
        return img, "global_combo:" + "+".join(ops), params
    if op == "global_contrast":
        f = float(rng.uniform(0.9, 1.1)); return ImageEnhance.Contrast(img).enhance(f), op, {"factor": f}
    if op == "global_jpeg":
        q = int(rng.integers(70, 96)); buf = io.BytesIO(); img.save(buf, "JPEG", quality=q)
        buf.seek(0); return Image.open(buf).convert("RGB"), op, {"quality": q}
    if op == "global_brightness":
        f = float(rng.uniform(0.9, 1.1)); return ImageEnhance.Brightness(img).enhance(f), op, {"factor": f}
    if op == "global_noise":
        sigma = float(rng.uniform(0.5, 3.0))
        a = np.asarray(img, np.float32) + rng.normal(0, sigma, size=(img.size[1], img.size[0], 1))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), op, {"sigma": sigma}
    f = float(rng.uniform(0.8, 1.2))
    W, H = img.size
    return img.resize((int(W * f), int(H * f)), Image.LANCZOS), op, {"factor": f}


# ── driver ───────────────────────────────────────────────────────────────────

def _split() -> dict:
    return json.loads(SPLIT_PATH.read_text(encoding="utf-8"))


def train_genuine_sources() -> list[dict]:
    split = _split()
    test = {r["path"] for r in split["test"]}
    test_hashes = {r["sha256"] for r in split["test"]}
    sources = [r for r in split["train"] if r["label"] == "genuine"]
    assert not test & {r["path"] for r in sources}, "frozen test image used as source"
    for r in sources:
        h = sha256(DATASET_DIR / r["path"])
        if h != r["sha256"] or h in test_hashes:
            raise RuntimeError(f"source changed or equals a test image: {r['path']}")
    return sources


def _balanced(options: list[str], n: int, rng) -> list[str]:
    """n draws covering every option as evenly as possible, in seeded random order."""
    return [str(o) for o in rng.permutation((options * (n // len(options) + 1))[:n])]


def global_plan(n_sources: int) -> tuple[list[str], list[str], list[str], dict]:
    """Seeded plan so that neither whole-page processing nor its TYPE predicts the label.

    * processed fraction is equal for both classes over the whole training pool;
    * whole-page op types are spread evenly within each class;
    * local edit types are spread evenly over the tampered variants.

    Genuine variants are always processed (an unprocessed genuine variant would
    be an exact duplicate of its source, which is already in the pool)."""
    split = _split()
    n_gen_orig = sum(r["label"] == "genuine" for r in split["train"])
    n_tam_orig = sum(r["label"] == "tampered" for r in split["train"])
    n_gen_aug, n_tam_aug = n_sources, n_sources * N_TAMPERED
    rate = n_gen_aug / (n_gen_orig + n_gen_aug)
    k = int(round(rate * (n_tam_orig + n_tam_aug)))
    rng = np.random.default_rng(BASE_SEED - 1)
    tam_global = _balanced(GLOBAL_OPS, k, rng) + ["none"] * (n_tam_aug - k)
    tam_global = [str(o) for o in rng.permutation(tam_global)]
    gen_global = _balanced(GLOBAL_OPS, n_gen_aug, rng)
    tam_local = _balanced(LOCAL_OPS, n_tam_aug, rng)
    plan = {"target_rate": rate,
            "genuine_processed": [n_gen_aug, n_gen_orig + n_gen_aug],
            "tampered_processed": [k, n_tam_orig + n_tam_aug],
            "genuine_global_ops": dict(Counter(gen_global)),
            "tampered_global_ops": dict(Counter(tam_global)),
            "tampered_local_ops": dict(Counter(tam_local))}
    return tam_global, gen_global, tam_local, plan


def generate() -> list[dict]:
    sources = train_genuine_sources()
    tam_global, gen_global, tam_local, plan = global_plan(len(sources))
    print("whole-page plan:", plan)
    rows, t_idx = [], 0
    for i, src_row in enumerate(sources):
        seed = BASE_SEED + i
        rng = np.random.default_rng(seed)
        with Image.open(DATASET_DIR / src_row["path"]) as im:
            src = im.convert("RGB")
        stem = Path(src_row["path"]).stem
        variants = []
        for _ in range(N_TAMPERED):
            img, op, params, bbox = make_tampered(src, rng, tam_local[t_idx])
            g = whole_page(img, rng, tam_global[t_idx]) if tam_global[t_idx] != "none" else (img, "none", {})
            t_idx += 1
            variants.append(("tampered", g[0], op, params, bbox, g[1], g[2]))
        g = whole_page(src, rng, gen_global[i])
        variants.append(("genuine", g[0], "none", {}, None, g[1], g[2]))
        for k, (label, img, op, params, bbox, gop, gparams) in enumerate(variants):
            rel = f"{label}/{stem}_aug{k}.png"
            out = OUT_DIR / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, format="PNG")
            rows.append({
                "file": rel, "label": label, "source_path": src_row["path"],
                "source_sha256": src_row["sha256"], "source_type": "train_original_genuine", "transform": op,
                "params": json.dumps(params, sort_keys=True), "bbox": json.dumps(bbox),
                "global_transform": gop, "global_params": json.dumps(gparams, sort_keys=True),
                "seed": seed, "sha256": sha256(out), "aug_version": AUG_VERSION,
            })
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    return rows


def load_manifest() -> list[dict]:
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_sha() -> str:
    return sha256(MANIFEST)


def check() -> bool:
    """True if every manifest file exists with its recorded sha256."""
    return all((OUT_DIR / r["file"]).exists() and sha256(OUT_DIR / r["file"]) == r["sha256"]
               for r in load_manifest())


def audit() -> dict:
    """Provenance + label-shortcut audit over the full training pool."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    split = _split()
    test_paths = {r["path"] for r in split["test"]}
    test_hashes = {r["sha256"] for r in split["test"]}
    train_gen = {r["path"]: r["sha256"] for r in split["train"] if r["label"] == "genuine"}
    man = load_manifest()
    rep = {
        "rows": len(man), "labels": dict(Counter(r["label"] for r in man)),
        "distinct_source_documents": len({r["source_path"] for r in man}),
        "sources_all_train_genuine": all(train_gen.get(r["source_path"]) == r["source_sha256"] for r in man),
        "any_source_in_test": any(r["source_path"] in test_paths for r in man),
        "any_output_hash_equals_test": any(r["sha256"] in test_hashes for r in man),
        "files_match_manifest": check(),
        "local_ops": dict(Counter(r["transform"].split(":")[0] if not r["transform"].startswith("combo")
                                  else r["transform"] for r in man if r["label"] == "tampered")),
        "fonts": dict(Counter(json.loads(r["params"]).get("font") or
                              next((v.get("font") for v in json.loads(r["params"]).values() if isinstance(v, dict) and v.get("font")), None)
                              for r in man if r["label"] == "tampered")),
    }
    # Whole-page processing rate per class over the FULL training pool.
    pool = [{"path": DATASET_DIR / r["path"], "label": r["label"], "group": r["path"], "global": "none"}
            for r in split["train"]]
    pool += [{"path": OUT_DIR / r["file"], "label": r["label"], "group": r["source_path"],
              "global": r["global_transform"]} for r in man]
    for lab in ("genuine", "tampered"):
        rows = [p for p in pool if p["label"] == lab]
        proc = [p for p in rows if p["global"] != "none"]
        rep[f"{lab}_processed_rate"] = round(len(proc) / len(rows), 4)
        rep[f"{lab}_untouched_rate"] = round(1 - len(proc) / len(rows), 4)
        rep[f"{lab}_global_ops"] = dict(Counter(p["global"].split(":")[0] for p in rows))
    aug_only = {lab: [r for r in man if r["label"] == lab] for lab in ("genuine", "tampered")}
    rep["augmented_only_processed_rate"] = {lab: round(sum(r["global_transform"] != "none" for r in rows) / len(rows), 4)
                                            for lab, rows in aug_only.items()}
    # Shortcut probe: can the LABEL be predicted from processing / format descriptors alone?
    desc, y, groups = [], [], []
    for p in pool:
        with Image.open(p["path"]) as im:
            fmt, mode, meta = im.format, im.mode, len(im.info)
            g = np.asarray(im.convert("L"), np.float32)
        W, H = g.shape[1], g.shape[0]
        noise = float(np.median(np.abs(g - cv2.medianBlur(g.astype(np.uint8), 3))))
        col = np.abs(np.diff(g, axis=1))
        block = float(col[:, 7::8].mean() / (col.mean() + 1e-6))  # JPEG 8x8 grid strength
        desc.append([p["path"].stat().st_size, W, H, W / H, float(g.mean()), noise, block,
                     fmt == "PNG", mode == "RGB", meta])
        y.append(p["label"] == "tampered"); groups.append(p["group"])
    X, y = np.array(desc, float), np.array(y, int)
    from sklearn.metrics import roc_auc_score
    names = ["file_bytes", "width", "height", "aspect", "global_brightness", "global_noise", "jpeg_grid"]
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [5, 50, 95])]
    rep["distributions_by_label_p5_p50_p95"] = {
        n: {lab: q(X[y == (lab == "tampered"), j]) for lab in ("genuine", "tampered")} for j, n in enumerate(names)}
    rep["dimensions_by_label"] = {lab: dict(Counter(f"{int(d[1])}x{int(d[2])}" for d, t in zip(desc, y)
                                                    if t == (lab == "tampered")).most_common(6))
                                  for lab in ("genuine", "tampered")}
    rep["single_descriptor_label_auc"] = {n: round(float(max(a, 1 - a)), 3) for j, n in enumerate(names)
                                          for a in [roc_auc_score(y, X[:, j]) if np.ptp(X[:, j]) else 0.5]}
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    pred = cross_val_predict(RandomForestClassifier(200, class_weight="balanced", random_state=42),
                             X, y, cv=cv, groups=groups)
    rep["shortcut_probe"] = {
        "descriptors": ["file_bytes", "width", "height", "aspect", "global_brightness",
                        "global_noise", "jpeg_grid", "is_png", "is_rgb", "n_metadata_keys"],
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "chance": 0.5, "pool_size": len(y),
        "formats": dict(Counter(bool(d[7]) for d in desc)),
        "metadata_keys": dict(Counter(int(d[9]) for d in desc)),
    }
    # Pass criteria (fixed before looking at v2.1 results).
    rep["checks"] = {
        "processed_rate_gap_le_0.05": abs(rep["genuine_processed_rate"] - rep["tampered_processed_rate"]) <= 0.05,
        "combined_probe_balanced_acc_le_0.60": rep["shortcut_probe"]["balanced_accuracy"] <= 0.60,
        "every_single_descriptor_auc_le_0.65": max(rep["single_descriptor_label_auc"].values()) <= 0.65,
        "single_format_and_no_metadata": len(rep["shortcut_probe"]["formats"]) == 1 and
                                         set(rep["shortcut_probe"]["metadata_keys"]) == {0},
        "no_test_source_or_hash": not rep["any_source_in_test"] and not rep["any_output_hash_equals_test"],
        "provenance_complete": all(r[f] not in ("", None) for r in man for f in FIELDS if f != "bbox") and
                               all(json.loads(r["bbox"]) for r in man if r["label"] == "tampered"),
        "files_match_manifest": rep["files_match_manifest"],
    }
    rep["pass"] = all(rep["checks"].values())
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    if args.check:
        ok = check(); print("manifest check:", "OK" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
    if args.audit:
        print(json.dumps(audit(), indent=1)); sys.exit(0)
    rows = generate()
    print(f"{len(rows)} images -> {OUT_DIR}")
    print(Counter(r["label"] for r in rows))
    print(Counter((r["label"], r["global_transform"]) for r in rows))
