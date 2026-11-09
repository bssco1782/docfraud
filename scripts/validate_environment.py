"""
scripts/validate_environment.py
Check that a machine can run the Document Fraud Detection API, optionally starting from a
fresh clone and a clean virtual environment.

Checks (current interpreter, run from the repository root):
  - Python version (3.10+)
  - every package in requirements.txt is installed and importable
  - required repository files exist (backend, core, model, demo set, docs)
  - the model file loads, its recorded feature schema is known, its feature count matches that
    schema's extractor, and it matches the demo manifest's SHA-256
  - backend.main imports and the configuration is sane (warns on the default API key)
  - a uvicorn server starts and answers /health and one real /verify

With --fresh-clone the script first clones the repository into an empty temporary
directory, creates a new virtual environment, installs requirements.txt from scratch,
runs the checks above with that environment, and runs the full test suite there.

Writes logs/environment_validation.txt by default. Exit code 0 means every check passed.

Usage (from the repository root):
    python scripts/validate_environment.py
    python scripts/validate_environment.py --fresh-clone
    python scripts/validate_environment.py --fresh-clone --repo https://github.com/<owner>/docfraud-backend.git --branch main
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "logs" / "environment_validation.txt"
DEFAULT_REPO = "https://github.com/bssco178-a11y/docfraud-backend.git"
MIN_PYTHON = (3, 11)  # oldest version the project has been verified on

# requirements.txt distribution name -> module to import
IMPORT_NAMES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "python-dotenv": "dotenv",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "faker": "faker",
    "pytest": "pytest",
    "pymupdf": "fitz",
    "httpx": "httpx",
    "python-multipart": "multipart",
    "pytest-cov": "pytest_cov",
}

REQUIRED_FILES = [
    "backend/main.py",
    "backend/config.py",
    "backend/model_utils.py",
    "core/document_analyzer.py",
    "core/model_trainer.py",
    "data/models/fraud_model.pkl",
    "data/demo/demo_manifest.json",
    "requirements.txt",
    ".env.example",
    "README.md",
    "DEPLOYMENT.md",
]

CheckResult = Tuple[str, bool, str]  # (name, passed, detail)


# -------------------------------------------------------------------------
# Local checks (run inside the environment being validated)
# -------------------------------------------------------------------------


def requirement_names(requirements: Path) -> List[str]:
    """Return distribution names listed in requirements.txt (extras and versions stripped)."""
    names = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(re.split(r"[\[<>=!~ ]", line, maxsplit=1)[0].lower())
    return names


def check_python() -> CheckResult:
    """Python must be at least MIN_PYTHON."""
    ok = sys.version_info[:2] >= MIN_PYTHON
    return ("Python >= %d.%d" % MIN_PYTHON, ok, f"{platform.python_version()} at {sys.executable}")


def check_packages() -> List[CheckResult]:
    """Every requirement is installed (has metadata) and its module imports."""
    results = []
    for name in requirement_names(ROOT / "requirements.txt"):
        module = IMPORT_NAMES.get(name, name.replace("-", "_"))
        try:
            version = metadata.version(name)
            importlib.import_module(module)
            results.append((f"package {name}", True, f"{version} (import {module})"))
        except Exception as exc:
            results.append((f"package {name}", False, f"{type(exc).__name__}: {exc}"))
    return results


def check_files() -> List[CheckResult]:
    """Required repository files exist."""
    return [(f"file {rel}", (ROOT / rel).exists(), "") for rel in REQUIRED_FILES]


def check_model() -> List[CheckResult]:
    """The model loads, matches its feature schema's extractor and the demo manifest hash."""
    from backend import config
    from core.feature_schemas import get_schema
    from core.model_trainer import DocumentFraudDetector

    path = Path(config.MODEL_PATH)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = json.loads((ROOT / "data/demo/demo_manifest.json").read_text(encoding="utf-8"))["model"]["sha256"]
    try:
        model, meta = DocumentFraudDetector.load(path)
    except Exception as exc:
        return [("model loads", False, repr(exc))]
    n_features = getattr(model, "n_features_in_", None)
    try:
        names = get_schema(meta.get("feature_schema_version")).names
    except ValueError as exc:
        return [("model feature schema known", False, str(exc))]
    return [
        ("model loads", True, f"{path.relative_to(ROOT).as_posix()} v{meta.get('model_version')}"),
        (
            "model feature count matches extractor",
            n_features == len(names),
            f"{n_features} vs {len(names)} ({meta.get('feature_schema_version')})",
        ),
        ("model SHA-256 matches demo manifest", digest == expected, digest[:16] + "..."),
    ]


def check_app_and_config() -> List[CheckResult]:
    """backend.main imports; configuration values are usable."""
    results = []
    try:
        from backend import config
        from backend.main import app

        results.append(("backend.main imports", True, f"{app.title} v{app.version}"))
        results.append(("MAX_FILE_SIZE positive", config.MAX_FILE_SIZE > 0, str(config.MAX_FILE_SIZE)))
        results.append(
            ("TAMPER_THRESHOLD in [0, 1]", 0.0 <= config.TAMPER_THRESHOLD <= 1.0, str(config.TAMPER_THRESHOLD))
        )
        key_detail = (
            "default key in use (fine for local testing; set API_KEY before sharing)"
            if config.API_KEY == "default-api-key"
            else "custom key set"
        )
        results.append(("API_KEY configured", True, key_detail))
        results.append(
            (".env file", True, "present" if (ROOT / ".env").exists() else "absent (optional; defaults used)")
        )
    except Exception as exc:
        results.append(("backend.main imports", False, repr(exc)))
    return results


def free_port() -> int:
    """Return a free TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def check_server() -> List[CheckResult]:
    """Start uvicorn, then call /health and /verify with a demo image."""
    import httpx

    port, key = free_port(), "env-check-key"
    env = dict(os.environ, API_KEY=key)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    results: List[CheckResult] = []
    try:
        start, healthy = time.perf_counter(), False
        while time.perf_counter() - start < 90 and proc.poll() is None:
            try:
                healthy = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200
            except httpx.HTTPError:
                healthy = False
            if healthy:
                break
            time.sleep(0.25)
        results.append(("uvicorn starts and /health returns 200", healthy, f"{time.perf_counter() - start:.1f} s"))
        if healthy:
            demo = ROOT / "data/demo/genuine_payslip_01.png"
            t0 = time.perf_counter()
            r = httpx.post(
                f"http://127.0.0.1:{port}/verify",
                headers={"X-API-Key": key},
                files={"file": (demo.name, demo.read_bytes(), "image/png")},
                timeout=120,
            )
            detail = f"HTTP {r.status_code} in {(time.perf_counter() - t0) * 1000:.0f} ms"
            if r.status_code == 200:
                detail += f", tamper_risk_score={r.json()['tamper_risk_score']}"
            results.append(("POST /verify with a demo image returns 200", r.status_code == 200, detail))
    finally:
        proc.terminate()
        try:
            output, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            output, _ = proc.communicate()
    errors = [ln for ln in (output or "").splitlines() if "Traceback" in ln or "ERROR" in ln]
    results.append(("server log free of errors/tracebacks", not errors, errors[0] if errors else "clean"))
    return results


def run_local_checks() -> List[Tuple[str, List[CheckResult]]]:
    """Run every local check group, in order."""
    return [
        ("Python", [check_python()]),
        ("Packages (requirements.txt)", check_packages()),
        ("Repository files", check_files()),
        ("Model", check_model()),
        ("Application & configuration", check_app_and_config()),
        ("Server smoke test", check_server()),
    ]


# -------------------------------------------------------------------------
# Fresh clone orchestration
# -------------------------------------------------------------------------


def run(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 1800) -> Tuple[int, str, float]:
    """Run a command; return (exit code, combined output, seconds)."""
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout + proc.stderr, time.perf_counter() - start


def venv_python(venv: Path) -> Path:
    """Path of the interpreter inside a virtual environment (Windows or POSIX layout)."""
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def fresh_clone(repo: str, branch: str, keep: bool) -> Tuple[List[str], bool]:
    """Clone, create a venv, install requirements and validate inside it. Returns (report lines, ok)."""
    work = Path(tempfile.mkdtemp(prefix="docfraud_fresh_"))
    clone, venv = work / "docfraud-backend", work / "venv"
    lines, ok = [], True

    def step(name: str, cmd: List[str], cwd: Optional[Path] = None, show_tail: int = 0) -> Tuple[bool, str]:
        nonlocal ok
        code, out, secs = run(cmd, cwd)
        passed = code == 0
        ok &= passed
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}  -- {secs:.1f} s")
        if show_tail or not passed:
            tail = [ln for ln in out.strip().splitlines() if ln.strip()][-(show_tail or 15) :]
            lines.extend("        " + ln for ln in tail)
        return passed, out

    lines.append(f"  Work directory : {work}")
    lines.append(f"  Source         : {repo} (branch {branch})")
    try:
        passed, _ = step(
            "git clone into empty directory", ["git", "clone", "--depth", "1", "--branch", branch, repo, str(clone)]
        )
        if passed:
            _, head = step("checked-out commit", ["git", "log", "-1", "--format=%h %s"], cwd=clone, show_tail=1)
            step("python -m venv (clean environment)", [sys.executable, "-m", "venv", str(venv)])
            py = str(venv_python(venv))
            step("pip install --upgrade pip", [py, "-m", "pip", "install", "-q", "--upgrade", "pip"])
            step(
                "pip install --no-cache-dir -r requirements.txt (from scratch)",
                [py, "-m", "pip", "install", "-q", "--no-cache-dir", "-r", "requirements.txt"],
                cwd=clone,
            )
            _, frozen = step("pip freeze", [py, "-m", "pip", "freeze"])
            lines.append("")
            lines.append("  Installed packages in the fresh environment:")
            lines.extend("    " + ln for ln in frozen.strip().splitlines())
            lines.append("")
            inner = work / "inner_validation.txt"
            passed, _ = step(
                "validate_environment.py inside the fresh clone + venv",
                [py, "scripts/validate_environment.py", "--output", str(inner)],
                cwd=clone,
            )
            if inner.exists():
                lines.append("")
                lines.append("  --- checks run inside the fresh clone ---")
                lines.extend(
                    "  " + ln
                    for ln in inner.read_text(encoding="utf-8").splitlines()
                    if ln.startswith(("  [", "Overall"))
                )
                lines.append("")
            step(
                "python -m pytest -q (full suite, in fresh clone)",
                [py, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                cwd=clone,
                show_tail=1,
            )
    finally:
        if keep:
            lines.append(f"  Kept for inspection: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return lines, ok


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------


def build_report(
    groups: List[Tuple[str, List[CheckResult]]], fresh: Optional[Tuple[List[str], bool]]
) -> Tuple[str, bool]:
    """Render the report; return (text, overall pass)."""
    rule, thin = "=" * 78, "-" * 78
    checks = [c for _, group in groups for c in group]
    ok = all(passed for _, passed, _ in checks) and (fresh is None or fresh[1])
    lines = [
        rule,
        "DOCUMENT FRAUD DETECTION API - ENVIRONMENT VALIDATION",
        rule,
        f"Generated       : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Machine         : {platform.system()} {platform.release()} ({platform.machine()}), {os.cpu_count()} CPUs",
        f"Python          : {platform.python_version()} ({sys.executable})",
        f"Repository      : {ROOT}",
        f"Overall         : {'PASS' if ok else 'FAIL'} ({sum(p for _, p, _ in checks)}/{len(checks)} local checks passed"
        + ("" if fresh is None else f"; fresh clone {'PASS' if fresh[1] else 'FAIL'}")
        + ")",
        "",
    ]
    if fresh is not None:
        lines += ["FRESH ENVIRONMENT TEST (clone -> new venv -> pip install -> validate -> tests)", thin, *fresh[0], ""]
    for i, (title, group) in enumerate(groups, 1):
        lines += [f"{i}. {title.upper()}", thin]
        lines += [f"  [{'PASS' if p else 'FAIL'}] {name}" + (f"  -- {d}" if d else "") for name, p, d in group]
        lines.append("")
    lines += [
        "HOW TO REPRODUCE",
        thin,
        "python scripts/validate_environment.py                 # this machine, current environment",
        "python scripts/validate_environment.py --fresh-clone   # clean clone + new venv",
        "Full cross-machine instructions: DEPLOYMENT.md section 11.",
        rule,
    ]
    return "\n".join(lines) + "\n", ok


def main() -> int:
    """Parse arguments, run checks, write the report and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fresh-clone", action="store_true", help="Also test a fresh clone in a new virtualenv")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Repository URL or path for --fresh-clone")
    parser.add_argument("--branch", default="main", help="Branch to clone for --fresh-clone")
    parser.add_argument("--keep", action="store_true", help="Keep the fresh-clone directory afterwards")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Report path")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    fresh = fresh_clone(args.repo, args.branch, args.keep) if args.fresh_clone else None
    report, ok = build_report(run_local_checks(), fresh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
