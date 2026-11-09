"""
scripts/generate_api_test_report.py
Run the comprehensive API test suite and write logs/api_test_report.txt.

The report contains: total / passed / failed / skipped / error counts, per-category
results, failed test details, a coverage summary for the backend package, and
execution time. All numbers come from the pytest run itself (JUnit XML + the
pytest-cov terminal report); nothing is hand-entered.

Usage:
    python scripts/generate_api_test_report.py
    python scripts/generate_api_test_report.py --tests tests/test_api_comprehensive.py --output logs/api_test_report.txt

Requires pytest-cov for the coverage section (falls back to "not available").
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTS = "tests/test_api_comprehensive.py"
DEFAULT_OUTPUT = ROOT / "logs" / "api_test_report.txt"
COVERAGE_TARGET = "backend"

# Test class -> human-readable category shown in the report.
CATEGORY_LABELS = {
    "TestHealthEndpoint": "Health endpoint",
    "TestAuthentication": "Authentication",
    "TestFileValidation": "File validation",
    "TestVerifyEndpoint": "Verify endpoint",
    "TestErrorHandling": "Error handling",
}


def run_pytest(tests: str, junit_path: Path) -> tuple[int, str, float]:
    """Run pytest on `tests`, returning (exit code, combined output, wall time in seconds)."""
    cmd = [sys.executable, "-m", "pytest", tests, "-v", "-p", "no:cacheprovider", f"--junitxml={junit_path}"]
    has_cov = importlib.util.find_spec("pytest_cov") is not None
    if has_cov:
        cmd += [f"--cov={COVERAGE_TARGET}", "--cov-report=term-missing"]

    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout + proc.stderr, elapsed


def parse_junit(junit_path: Path) -> dict:
    """Extract counts, per-category results and failures from a JUnit XML file."""
    root = ET.parse(junit_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")

    categories: "OrderedDict[str, dict]" = OrderedDict()
    failures = []
    for case in suite.iter("testcase"):
        cls = case.get("classname", "").rsplit(".", 1)[-1]
        label = CATEGORY_LABELS.get(cls, cls or "Other")
        bucket = categories.setdefault(label, {"total": 0, "passed": 0, "failed": 0, "skipped": 0})
        bucket["total"] += 1

        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            bucket["failed"] += 1
            failures.append((f"{cls}::{case.get('name')}", (problem.get("message") or "").strip()))
        elif case.find("skipped") is not None:
            bucket["skipped"] += 1
        else:
            bucket["passed"] += 1

    total = int(suite.get("tests", 0))
    failed = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    return {
        "total": total,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "passed": total - failed - errors - skipped,
        "pytest_time": float(suite.get("time", 0.0)),
        "categories": categories,
        "failures": failures,
    }


def extract_coverage(output: str) -> str:
    """Return the pytest-cov coverage table from pytest's terminal output."""
    match = re.search(r"(Name\s+Stmts\s+Miss.*?\nTOTAL[^\n]*)", output, re.S)
    return match.group(1).rstrip() if match else "Coverage not available (install pytest-cov)."


def build_report(tests: str, results: dict, coverage: str, wall_time: float, exit_code: int) -> str:
    """Render the plain-text report."""
    rule = "=" * 72
    thin = "-" * 72
    status = "PASS" if exit_code == 0 else "FAIL"
    lines = [
        rule,
        "DOCUMENT FRAUD DETECTION API - COMPREHENSIVE TEST REPORT (DAY 6)",
        rule,
        f"Generated      : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Test module    : {tests}",
        f"Command        : pytest {tests} -v",
        f"Python         : {platform.python_version()} ({platform.system()})",
        f"Overall status : {status} (pytest exit code {exit_code})",
        "",
        "SUMMARY",
        thin,
        f"Total tests executed : {results['total']}",
        f"Passed               : {results['passed']}",
        f"Failed               : {results['failed']}",
        f"Errors               : {results['errors']}",
        f"Skipped              : {results['skipped']}",
        f"Pass rate            : {results['passed'] / results['total'] * 100:.1f}%" if results["total"] else "Pass rate            : n/a",
        "",
        "RESULTS BY CATEGORY",
        thin,
        f"{'Category':<22}{'Total':>7}{'Passed':>8}{'Failed':>8}{'Skipped':>9}",
    ]
    for label, c in results["categories"].items():
        lines.append(f"{label:<22}{c['total']:>7}{c['passed']:>8}{c['failed']:>8}{c['skipped']:>9}")

    lines += ["", "FAILED TESTS", thin]
    if results["failures"]:
        for name, message in results["failures"]:
            lines.append(f"- {name}")
            if message:
                lines.append(f"    {message.splitlines()[0][:200]}")
    else:
        lines.append("None")

    lines += [
        "",
        f"COVERAGE SUMMARY ({COVERAGE_TARGET}/ package, measured on this test module only)",
        thin,
        coverage,
        "",
        "EXECUTION TIME",
        thin,
        f"pytest test time     : {results['pytest_time']:.2f} s",
        f"Total wall time      : {wall_time:.2f} s (includes interpreter start-up and coverage)",
        "",
        "NOTES",
        thin,
        "- Most /verify tests patch backend.main.analyze_document (mock_model fixture) so",
        "  they exercise auth, validation and error handling without model inference.",
        "- test_valid_image_returns_prediction and test_corrupted_file_rejected run the",
        "  real trained model from data/models/fraud_model.pkl.",
        "- Auth tests patch backend.config.API_KEY with a fixed test key (test_api_key).",
        "- Oversized upload test lowers MAX_FILE_SIZE to 1 KB instead of sending 10 MB.",
        "",
        "HOW TO RUN",
        thin,
        f"pytest {tests} -v",
        "pytest -v",
        f"pytest {tests} --cov={COVERAGE_TARGET} --cov-report=term-missing",
        "python scripts/generate_api_test_report.py",
        rule,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", default=DEFAULT_TESTS, help="Test path passed to pytest")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Report file to write")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        junit_path = Path(tmp) / "junit.xml"
        exit_code, output, wall_time = run_pytest(args.tests, junit_path)
        if not junit_path.exists():
            print(output)
            print("pytest did not produce a JUnit report; aborting.", file=sys.stderr)
            return exit_code or 1
        results = parse_junit(junit_path)

    report = build_report(args.tests, results, extract_coverage(output), wall_time, exit_code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to {args.output}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
