"""Checks for scripts/final_validation.py: metric maths, ASCII matrix, frozen-split integrity."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import final_validation as fv  # noqa: E402


def test_metrics_match_hand_counts():
    y = np.array([0] * 10 + [1] * 5)
    p = np.array([0] * 5 + [1] * 5 + [0] + [1] * 4)
    m = fv.compute_metrics(y, p)  # also asserts agreement with sklearn
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (5, 5, 1, 4)
    assert m["accuracy"] == 9 / 15
    assert m["precision"] == 4 / 9 and m["recall"] == 4 / 5
    assert m["fpr"] == 0.5 and m["fnr"] == 0.2


def test_ascii_confusion_layout():
    text = fv.ascii_confusion({"tn": 5, "fp": 5, "fn": 1, "tp": 4})
    assert "Actual Genuine     TN = 5" in text and "FP = 5" in text
    assert "Actual Tampered    FN = 1" in text and "TP = 4" in text


def test_frozen_split_integrity():
    split = json.loads(fv.SPLIT_PATH.read_text(encoding="utf-8"))
    checks = fv.check_integrity(split)
    failed = [name for name, ok, _ in checks if not ok]
    assert not failed, failed
