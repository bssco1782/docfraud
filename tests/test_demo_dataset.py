"""Day-7 demo set: files, counts, hashes, frozen-test separation, docs match record."""

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "data" / "demo"
RECORD = json.loads((DEMO / "demo_manifest.json").read_text(encoding="utf-8"))
SPLIT = json.loads((ROOT / "data" / "splits" / "day5_frozen_split.json").read_text(encoding="utf-8"))
EXPECTED = ["genuine_payslip_01.png", "genuine_payslip_02.png", "tampered_payslip_01.png",
            "tampered_payslip_02.png", "edge_case_01.png"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_five_readable_images_with_expected_roles():
    assert sorted(p.name for p in DEMO.glob("*.png")) == sorted(EXPECTED)
    imgs = RECORD["images"]
    assert [r["demo_file"] for r in imgs] == EXPECTED
    assert [r["role"] for r in imgs].count("genuine payslip") == 2
    assert [r["role"] for r in imgs].count("tampered payslip") == 2
    assert [r["role"] for r in imgs].count("edge case") == 1
    for r in imgs:
        with Image.open(DEMO / r["demo_file"]) as im:
            im.verify()


def test_hashes_match_record_and_source():
    for r in RECORD["images"]:
        assert sha256(DEMO / r["demo_file"]) == r["sha256"]
        # Some sources live under git-ignored data/augmented/ folders and are absent from a
        # fresh clone; compare against the source only where it exists locally.
        source = ROOT / r["source"]
        if source.exists():
            assert sha256(source) == r["sha256"]


def test_no_frozen_test_overlap():
    test_paths = {t["path"] for t in SPLIT["test"]}
    test_hashes = {t["sha256"] for t in SPLIT["test"]}
    for r in RECORD["images"]:
        assert r["sha256"] not in test_hashes
        assert r.get("source_path") not in test_paths
        assert r.get("source_sha256") not in test_hashes
    assert all(v["pass"] for v in RECORD["integrity_checks"].values())


def test_predictions_consistent_and_documented():
    doc = (DEMO / "DEMO_IMAGES.md").read_text(encoding="utf-8")
    for r in RECORD["images"]:
        expected = "tampered" if r["tamper_risk_score"] >= RECORD["risk_threshold"] else "genuine"
        assert r["predicted_label"] == expected
        assert r["correct"] == (r["predicted_label"] == r["actual_label"])
        block = doc.split(f"- File: `{r['demo_file']}`")[1].split("###")[0]
        assert f"Actual label: **{r['actual_label']}**" in block
        assert f"Prediction: **{r['predicted_label']}**" in block
        assert f"Risk score: {r['tamper_risk_score']:.4f}" in block
