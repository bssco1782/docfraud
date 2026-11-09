"""
scripts/generate_extra_documents.py
Day 8 (Founder A): fresh synthetic documents for model development and held-out evaluation.

Usage:
    python scripts/generate_extra_documents.py dev        # -> data/generated/dev_v1/     (seed 7001, 1001-1150)
    python scripts/generate_extra_documents.py holdout    # -> data/generated/holdout_v1/ (seed 7002, 2001-2150)

Uses Founder C's generator (scripts/generate_tampered_documents.py, unmodified) through the
safe workflow documented in DATA_FORMAT_NOTES.md section 6: a separate output directory and a
seed that has never been used (not 42 or 4). data/synthetic/ is never written.

Roles (fixed before any model was trained on the new data):
  dev      training and cross-validation data for feature development, used together with the
           60 frozen TRAIN images.
  holdout  generated only after the model is frozen; scored once, never used for training,
           feature design, threshold choice or model selection.

The set is generated into a temporary directory first:
  - the target directory must not exist (nothing is overwritten),
  - a generated image that is byte-identical to, or a pixel near-copy of (< 0.4% of pixels
    differing, the Day-7 rule), any PNG in data/synthetic/ or in the other generated set is
    dropped before anything is kept, and listed under "excluded" in the manifest (a fixed
    rule applied before any model sees the set; small ID cards can fall under 0.4% by chance).
Writes <set>/genuine/, <set>/tampered/, <set>/index.csv (generator format) and
<set>/manifest.json (seed, counts, exclusions, generator SHA-256, SHA-256 of every file).

Rendering depends on the installed fonts (DejaVu). On the reference machine the generator
reproduces data/synthetic/ documents 026-075 byte-for-byte with seed 4, so new sets follow the
dataset's distribution; the manifest hashes show whether a regenerated set matches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from faker import Faker
from PIL import Image

from scripts.generate_tampered_documents import generate_full_dataset, write_index_csv

GENERATOR = ROOT / "scripts" / "generate_tampered_documents.py"
SYNTHETIC = ROOT / "data" / "synthetic"
GENERATED = ROOT / "data" / "generated"
# start_index keeps file names distinct from data/synthetic/ (001-075) and from each other.
SETS = {
    "dev": {"dir": "dev_v1", "seed": 7001, "start_index": 1001},
    "holdout": {"dir": "holdout_v1", "seed": 7002, "start_index": 2001},
}
COUNTS = {"payslip_count": 50, "bank_stmt_count": 50, "id_proof_count": 50}
USED_SEEDS = {42, 4}  # data/synthetic/ (Day 2 and Day 4)
NEAR_COPY_FRAC = 0.004  # same rule as scripts/prepare_demo_dataset.py


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.int16)


def diff_frac(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a - b) > 40).mean()) if a.shape == b.shape else 1.0


def reference_images(exclude: Path) -> list[Path]:
    """Every PNG the new set must not duplicate: data/synthetic/ and the other generated set."""
    refs = sorted(SYNTHETIC.rglob("*.png"))
    for spec in SETS.values():
        other = GENERATED / spec["dir"]
        if other != exclude and other.is_dir():
            refs += sorted(other.rglob("*.png"))
    return refs


def check_duplicates(new_files: list[Path], refs: list[Path]) -> dict[Path, str]:
    """New files that duplicate a reference image, with the reason."""
    ref_hashes = {sha256(p): p for p in refs}
    ref_gray = [(p, gray(p)) for p in refs]
    problems: dict[Path, str] = {}
    for f in new_files:
        if sha256(f) in ref_hashes:
            problems[f] = f"byte-identical to {ref_hashes[sha256(f)].relative_to(ROOT).as_posix()}"
            continue
        g = gray(f)
        frac, path = min(((diff_frac(g, rg), p) for p, rg in ref_gray), key=lambda t: t[0])
        if frac < NEAR_COPY_FRAC:
            problems[f] = f"near-copy of {path.relative_to(ROOT).as_posix()} ({frac:.4f} of pixels differ)"
    return problems


def generate(name: str) -> int:
    spec = SETS[name]
    target = GENERATED / spec["dir"]
    if spec["seed"] in USED_SEEDS:
        raise SystemExit(f"seed {spec['seed']} was already used for data/synthetic/")
    if target.exists():
        raise SystemExit(f"{target.relative_to(ROOT)} already exists; refusing to overwrite")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / spec["dir"]
        Faker.seed(spec["seed"])
        random.seed(spec["seed"])
        rows = generate_full_dataset(output_dir=work, start_index=spec["start_index"], **COUNTS)
        problems = check_duplicates([work / r["path"] for r in rows], reference_images(target))
        excluded = []
        for r in list(rows):
            reason = problems.get(work / r["path"])
            if reason:
                (work / r["path"]).unlink()
                rows.remove(r)
                excluded.append({"path": Path(r["path"]).as_posix(), "label": r["label"], "reason": reason})
                print(f"  excluded {r['path']}: {reason}")
        write_index_csv(rows, work)

        manifest = {
            "set": name,
            "role": "held-out evaluation only" if name == "holdout" else "training / cross-validation",
            "seed": spec["seed"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "generator": GENERATOR.relative_to(ROOT).as_posix(),
            "generator_sha256": sha256(GENERATOR),
            "counts": {**COUNTS, "total": len(rows),
                       "genuine": sum(r["label"] == "genuine" for r in rows),
                       "tampered": sum(r["label"] == "tampered" for r in rows)},
            "near_copy_rule": f"< {NEAR_COPY_FRAC:.1%} of grayscale pixels differ by > 40 levels",
            "excluded": excluded,
            "checked_against": "all PNGs in data/synthetic/ and the other generated set",
            "files": {Path(r["path"]).as_posix(): sha256(work / r["path"]) for r in rows},
        }
        (work / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(work, target)

    c = manifest["counts"]
    print(f"\n{target.relative_to(ROOT)}: {c['total']} documents ({c['genuine']} genuine, {c['tampered']} tampered), "
          f"seed {spec['seed']}; {len(excluded)} duplicate / near-copy document(s) excluded")
    return 0


def load_set(name: str) -> list[dict]:
    """Rows of a generated set's index.csv, with 'file' = absolute path and hashes verified."""
    base = GENERATED / SETS[name]["dir"]
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(base / "index.csv", newline="", encoding="utf-8")))
    for r in rows:
        r["path"] = Path(r["path"]).as_posix()
        r["file"] = base / r["path"]
        if sha256(r["file"]) != manifest["files"][r["path"]]:
            raise ValueError(f"{r['file']} does not match {base.name}/manifest.json")
    return rows


# Generator runs that produced existing documents: (seed, counts, start_index).
ORIGINAL_RUNS = {
    "synthetic_day2": (42, {"payslip_count": 10, "bank_stmt_count": 8, "id_proof_count": 7}, 1),
    "synthetic_day4": (4, {"payslip_count": 20, "bank_stmt_count": 15, "id_proof_count": 15}, 26),
}


def edit_footprints(run: str) -> dict[str, dict]:
    """Replay a generator run and measure each edit: {path: {method, rect, changed_px}}.

    ``run`` is "dev", "holdout", "synthetic_day2" or "synthetic_day4". The replay wraps the
    generator's _apply_tampering, re-draws the same random rectangle and counts pixels whose
    RGB value changed by more than 3 levels. Raises ValueError if the replay does not reproduce
    the files on disk byte-for-byte (for example, different fonts). Used for error analysis only.
    """
    import scripts.generate_tampered_documents as gen

    if run in SETS:
        seed, counts, start = SETS[run]["seed"], COUNTS, SETS[run]["start_index"]
        base = GENERATED / SETS[run]["dir"]
    else:
        seed, counts, start = ORIGINAL_RUNS[run]
        base = SYNTHETIC
    original = gen._apply_tampering
    edits: list[dict] = []

    def recording(img, method):
        state = random.getstate()
        w, h = img.size
        rect = [random.randint(50, w - 250), random.randint(80, h - 150),
                random.randint(120, 200), random.randint(30, 60)]
        random.setstate(state)
        out = original(img, method)
        changed = np.abs(np.asarray(img, dtype=np.int16) - np.asarray(out, dtype=np.int16)).max(axis=2) > 3
        edits.append({"method": method, "rect": rect, "changed_px": int(changed.sum())})
        return out

    gen._apply_tampering = recording
    try:
        with tempfile.TemporaryDirectory() as tmp:
            Faker.seed(seed)
            random.seed(seed)
            rows = gen.generate_full_dataset(output_dir=Path(tmp), start_index=start, **counts)
            out: dict[str, dict] = {}
            for r in rows:
                on_disk = base / r["path"]
                if not on_disk.exists():  # e.g. a document excluded as a near-copy
                    continue
                if sha256(Path(tmp) / r["path"]) != sha256(on_disk):
                    raise ValueError(f"replay of {run} does not reproduce {on_disk}")
                if r["label"] == "tampered":
                    out[Path(r["path"]).as_posix()] = edits[len([x for x in rows[: rows.index(r)]
                                                                 if x["label"] == "tampered"])]
    finally:
        gen._apply_tampering = original
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a fresh synthetic document set (dev or holdout)")
    ap.add_argument("set", choices=sorted(SETS))
    return generate(ap.parse_args().set)


if __name__ == "__main__":
    raise SystemExit(main())
