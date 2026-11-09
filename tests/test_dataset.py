"""
tests/test_dataset.py
Founder C — Day 5: dataset validation.

Read-only integrity checks for the indexed synthetic dataset
(data/synthetic/genuine/, data/synthetic/tampered/, data/synthetic/index.csv).
The 5 Day 1 payslips stored directly in data/synthetic/ are intentionally
outside the index and are checked separately.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

SYNTHETIC = Path(__file__).parent.parent / "data" / "synthetic"
INDEX_CSV = SYNTHETIC / "index.csv"
LABELS = ("genuine", "tampered")

EXPECTED_TOTAL = 75
EXPECTED_LABELS = {"genuine": 48, "tampered": 27}
EXPECTED_DOC_TYPES = {"payslip": 30, "bank_statement": 23, "id_proof": 22}
DOC_TYPE_PREFIX = {"payslip": "payslip_", "bank_statement": "bank_stmt_", "id_proof": "id_proof_"}
TAMPER_METHODS = {"blur_region", "color_shift", "noise_patch", "text_overlay"}
CSV_COLUMNS = [
    "filename", "path", "doc_type", "label",
    "tamper_method", "width", "height", "file_size_bytes",
]
DAY1_PAYSLIPS = [f"payslip_{i:03d}.png" for i in range(1, 6)]


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    assert INDEX_CSV.is_file(), f"index.csv not found at {INDEX_CSV}"
    with open(INDEX_CSV, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == CSV_COLUMNS, (
            f"index.csv header {reader.fieldnames} != expected {CSV_COLUMNS}"
        )
        return list(reader)


@pytest.fixture(scope="module")
def dataset_files() -> dict[str, list[Path]]:
    return {label: sorted((SYNTHETIC / label).iterdir()) for label in LABELS}


def _fmt(items, limit=10):
    items = sorted(items)
    more = f" … (+{len(items) - limit} more)" if len(items) > limit else ""
    return ", ".join(map(str, items[:limit])) + more


# 1–3: counts
def test_total_indexed_documents(rows):
    assert len(rows) == EXPECTED_TOTAL, (
        f"index.csv has {len(rows)} records, expected {EXPECTED_TOTAL}"
    )


@pytest.mark.parametrize("label", LABELS)
def test_label_counts(rows, dataset_files, label):
    in_csv = sum(1 for r in rows if r["label"] == label)
    on_disk = len(dataset_files[label])
    expected = EXPECTED_LABELS[label]
    assert in_csv == expected, f"index.csv has {in_csv} '{label}' records, expected {expected}"
    assert on_disk == expected, f"{label}/ has {on_disk} files, expected {expected}"


# 4: every record points to an existing file
def test_every_record_points_to_existing_file(rows):
    missing = [r["path"] for r in rows if not (SYNTHETIC / r["path"]).is_file()]
    assert not missing, f"{len(missing)} index.csv paths do not exist: {_fmt(missing)}"


# 5: no duplicate filenames in the CSV
def test_no_duplicate_filenames_in_csv(rows):
    dupes = {k: v for k, v in Counter(r["filename"] for r in rows).items() if v > 1}
    assert not dupes, f"Duplicate filenames in index.csv: {dupes}"
    dupes = {k: v for k, v in Counter(r["path"] for r in rows).items() if v > 1}
    assert not dupes, f"Duplicate paths in index.csv: {dupes}"


# 6: no filename present in both folders
def test_no_duplicate_filenames_across_folders(dataset_files):
    names = Counter(p.name for files in dataset_files.values() for p in files)
    dupes = sorted(k for k, v in names.items() if v > 1)
    assert not dupes, f"Filenames present in both genuine/ and tampered/: {_fmt(dupes)}"


# 7: required fields present
@pytest.mark.parametrize("field", ["filename", "path", "doc_type", "label"])
def test_no_missing_required_fields(rows, field):
    blank = [i for i, r in enumerate(rows, start=2) if not (r.get(field) or "").strip()]
    assert not blank, f"index.csv lines with empty '{field}': {_fmt(blank)}"


# 8: all images readable, and metadata matches
def test_all_images_readable_and_match_metadata(rows):
    broken, mismatched = [], []
    for r in rows:
        path = SYNTHETIC / r["path"]
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
                fmt, size = img.format, img.size
        except Exception as exc:
            broken.append(f"{r['path']} ({exc})")
            continue
        if fmt != "PNG":
            mismatched.append(f"{r['path']}: format {fmt}, expected PNG")
        if size != (int(r["width"]), int(r["height"])):
            mismatched.append(f"{r['path']}: size {size}, index says {r['width']}x{r['height']}")
        if path.stat().st_size != int(r["file_size_bytes"]):
            mismatched.append(
                f"{r['path']}: {path.stat().st_size} bytes, index says {r['file_size_bytes']}"
            )
    assert not broken, f"{len(broken)} unreadable images: {_fmt(broken)}"
    assert not mismatched, f"{len(mismatched)} metadata mismatches: {_fmt(mismatched)}"


# 9: label matches folder
def test_labels_match_folder(rows):
    invalid = [r["filename"] for r in rows if r["label"] not in LABELS]
    assert not invalid, f"Invalid labels (expected {LABELS}): {_fmt(invalid)}"
    wrong = [
        f"{r['path']} labelled '{r['label']}'"
        for r in rows
        if r["path"] != f"{r['label']}/{r['filename']}"
    ]
    assert not wrong, f"{len(wrong)} records whose path is not <label>/<filename>: {_fmt(wrong)}"


# 10: tamper_method only on tampered rows
def test_tamper_method_consistency(rows):
    missing = [r["filename"] for r in rows if r["label"] == "tampered" and not r["tamper_method"]]
    extra = [r["filename"] for r in rows if r["label"] == "genuine" and r["tamper_method"]]
    unknown = [
        f"{r['filename']}={r['tamper_method']}"
        for r in rows
        if r["tamper_method"] and r["tamper_method"] not in TAMPER_METHODS
    ]
    assert not missing, f"Tampered records without tamper_method: {_fmt(missing)}"
    assert not extra, f"Genuine records with a tamper_method: {_fmt(extra)}"
    assert not unknown, f"Unknown tamper_method values (expected {sorted(TAMPER_METHODS)}): {_fmt(unknown)}"


# 11: doc types valid and consistent with filenames
def test_doc_types_valid_and_match_filenames(rows):
    invalid = [f"{r['filename']}={r['doc_type']}" for r in rows if r["doc_type"] not in DOC_TYPE_PREFIX]
    assert not invalid, f"Invalid doc_type values (expected {sorted(DOC_TYPE_PREFIX)}): {_fmt(invalid)}"
    mismatch = [
        f"{r['filename']} ({r['doc_type']})"
        for r in rows
        if not r["filename"].startswith(DOC_TYPE_PREFIX[r["doc_type"]])
    ]
    assert not mismatch, f"Filename prefix does not match doc_type: {_fmt(mismatch)}"


# 12: doc type distribution
def test_doc_type_distribution(rows):
    actual = dict(Counter(r["doc_type"] for r in rows))
    assert actual == EXPECTED_DOC_TYPES, f"doc_type distribution {actual} != expected {EXPECTED_DOC_TYPES}"


# 13: no unexpected or unindexed files
def test_no_unexpected_or_unindexed_files(rows, dataset_files):
    indexed = {r["path"] for r in rows}
    unexpected, unindexed = [], []
    for label, files in dataset_files.items():
        for p in files:
            rel = f"{label}/{p.name}"
            if not p.is_file() or p.name.startswith(".") or p.suffix.lower() != ".png":
                unexpected.append(rel)
            elif rel not in indexed:
                unindexed.append(rel)
    assert not unexpected, f"Unexpected entries in dataset folders: {_fmt(unexpected)}"
    assert not unindexed, f"Files on disk missing from index.csv: {_fmt(unindexed)}"

    allowed_root = {"genuine", "tampered", "index.csv", *DAY1_PAYSLIPS}
    extra_root = [p.name for p in SYNTHETIC.iterdir() if p.name not in allowed_root]
    assert not extra_root, f"Unexpected entries in data/synthetic/: {_fmt(extra_root)}"


def test_day1_payslips_preserved_outside_index(rows):
    missing = [n for n in DAY1_PAYSLIPS if not (SYNTHETIC / n).is_file()]
    assert not missing, f"Day 1 payslips missing from data/synthetic/: {missing}"
    for name in DAY1_PAYSLIPS:
        with Image.open(SYNTHETIC / name) as img:
            img.load()
            assert img.size == (850, 1050), f"{name} is {img.size}, expected (850, 1050)"
    indexed_root = [r["path"] for r in rows if "/" not in r["path"]]
    assert not indexed_root, f"Day 1 root files must not be indexed: {indexed_root}"
