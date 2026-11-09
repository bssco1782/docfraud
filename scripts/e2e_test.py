"""
scripts/e2e_test.py
End-to-end test of the Document Fraud Detection API, run the way a real client uses it.

The script starts a real uvicorn server (or targets one you already started), talks to it
over HTTP only, and checks every stage of the /verify workflow:

    upload -> API authentication -> input validation -> image decoding
           -> feature extraction (the model's schema) -> model inference -> JSON response

Sections of the report (logs/e2e_test_report.txt by default):
  1. Server start-up        time to first healthy /health
  2. Authentication         missing / wrong / valid X-API-Key
  3. Input validation       empty, unsupported, corrupted, oversized, bad query, unknown route
  4. Pipeline stages        decoding, the model's features and the model score recomputed
                            in-process for each demo image and compared with the API's answer
  5. Demo images            all 5 images in data/demo/: fields, ranges, latency, and agreement
                            with the scores recorded in data/demo/demo_manifest.json
  6. Reliability            N consecutive /verify requests: crashes, errors, server memory
                            (RSS) growth and prediction consistency

Exit code is 0 only when every check passes.

Usage (from the repository root):
    python scripts/e2e_test.py
    python scripts/e2e_test.py --iterations 20 --port 8765
    python scripts/e2e_test.py --url http://127.0.0.1:8000 --api-key <key>   # existing server
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "data" / "demo"
MANIFEST_PATH = DEMO_DIR / "demo_manifest.json"
DEFAULT_OUTPUT = ROOT / "logs" / "e2e_test_report.txt"

VERIFY_FIELDS = {"tamper_risk_score", "confidence", "flagged_regions", "interpretation"}
REGION_FIELDS = {"x", "y", "width", "height", "area", "type"}
ERROR_FIELDS = {"success", "error", "code", "detail"}

# Reliability thresholds
MEMORY_WARMUP_REQUESTS = 5  # RSS growth is measured after this many requests
MEMORY_GROWTH_LIMIT_MB = 25.0  # max allowed post-warmup RSS growth for a PASS
SCORE_TOLERANCE = 1e-9  # identical input must give an identical score

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -------------------------------------------------------------------------
# Result bookkeeping
# -------------------------------------------------------------------------


@dataclass
class Check:
    """One pass/fail check with a short detail string."""

    section: str
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Results:
    """All checks plus free-form tables collected for the report."""

    checks: List[Check] = field(default_factory=list)
    tables: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, section: str, name: str, passed: bool, detail: str = "") -> bool:
        """Record a check and return its outcome."""
        self.checks.append(Check(section, name, bool(passed), detail))
        return bool(passed)

    def section_checks(self, section: str) -> List[Check]:
        """Return the checks recorded under one section."""
        return [c for c in self.checks if c.section == section]


# -------------------------------------------------------------------------
# Server management
# -------------------------------------------------------------------------


def rss_mb(pid: int) -> Optional[float]:
    """Resident memory of a process in MB, or None if it cannot be measured on this OS."""
    try:
        import psutil  # optional dependency

        return psutil.Process(pid).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    except Exception:
        return None
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return None


def start_server(python: str, repo_dir: Path, port: int, api_key: str, log_path: Path) -> subprocess.Popen:
    """Start uvicorn for backend.main:app in repo_dir with the given API key."""
    env = dict(os.environ, API_KEY=api_key)
    log = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [python, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=repo_dir,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def wait_for_health(base_url: str, timeout_s: float, proc: Optional[subprocess.Popen]) -> Optional[float]:
    """Poll /health until it answers 200; return seconds waited, or None on timeout/crash."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        if proc is not None and proc.poll() is not None:
            return None
        try:
            if httpx.get(f"{base_url}/health", timeout=2).status_code == 200:
                return time.perf_counter() - start
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    return None


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate the uvicorn process and wait for it to exit."""
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


# -------------------------------------------------------------------------
# HTTP helpers
# -------------------------------------------------------------------------


def post_verify(
    client: httpx.Client,
    name: str,
    content: bytes,
    api_key: Optional[str],
    content_type: str = "image/png",
    params: Optional[Dict[str, Any]] = None,
) -> tuple[httpx.Response, float]:
    """POST one file to /verify; return the response and its latency in seconds."""
    headers = {"X-API-Key": api_key} if api_key is not None else {}
    start = time.perf_counter()
    response = client.post("/verify", files={"file": (name, content, content_type)}, headers=headers, params=params)
    return response, time.perf_counter() - start


def is_error_envelope(response: httpx.Response, status: int) -> bool:
    """True if the response has the given status and the standard error JSON body."""
    try:
        body = response.json()
    except ValueError:
        return False
    return (
        response.status_code == status
        and ERROR_FIELDS <= body.keys()
        and body["success"] is False
        and body["code"] == status
    )


def valid_verify_body(body: Dict[str, Any]) -> tuple[bool, str]:
    """Check a /verify success body; return (ok, reason)."""
    if set(body) != VERIFY_FIELDS:
        return False, f"fields {sorted(body)}"
    for key in ("tamper_risk_score", "confidence"):
        value = body[key]
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            return False, f"{key}={value!r} out of range"
    if not isinstance(body["interpretation"], str) or not body["interpretation"].strip():
        return False, "empty interpretation"
    regions = body["flagged_regions"]
    if not isinstance(regions, list):
        return False, "flagged_regions not a list"
    for r in regions:
        if set(r) != REGION_FIELDS or not isinstance(r["type"], str):
            return False, f"flagged region fields {sorted(r)}"
        if any(not isinstance(r[k], int) or r[k] < 0 for k in REGION_FIELDS - {"type"}):
            return False, f"flagged region values {r}"
    if regions and not body["interpretation"].startswith("Evidence of digital tampering"):
        return False, "flagged_regions returned for a document reported as genuine"
    return True, ""


# -------------------------------------------------------------------------
# Test sections
# -------------------------------------------------------------------------


def test_authentication(client: httpx.Client, api_key: str, image: bytes, res: Results) -> None:
    """Section 2: API-key enforcement."""
    sec = "Authentication"
    r, _ = post_verify(client, "doc.png", image, None)
    res.add(sec, "Missing X-API-Key -> 401", is_error_envelope(r, 401), f"got {r.status_code}")
    r, _ = post_verify(client, "doc.png", image, "wrong-key")
    res.add(sec, "Wrong X-API-Key -> 403", is_error_envelope(r, 403), f"got {r.status_code}")
    r, _ = post_verify(client, "doc.png", image, api_key)
    res.add(sec, "Valid X-API-Key -> 200", r.status_code == 200, f"got {r.status_code}")
    r = client.get("/health")
    res.add(sec, "/health is public (no key) -> 200", r.status_code == 200, f"got {r.status_code}")


def test_validation(client: httpx.Client, api_key: str, image: bytes, max_file_size: int, res: Results) -> None:
    """Section 3: input validation and error handling."""
    sec = "Input validation"
    cases = [
        ("Empty file -> 400", "empty.png", b"", "image/png", None, 400),
        ("Unsupported type (.txt) -> 400", "notes.txt", b"plain text", "text/plain", None, 400),
        ("No file extension -> 400", "document", image, "image/png", None, 400),
        ("Corrupted image bytes -> 400", "broken.jpg", b"\xff\xd8\xff\xe0 not a jpeg", "image/jpeg", None, 400),
        ("Oversized file -> 413", "big.png", b"\x00" * (max_file_size + 1), "image/png", None, 413),
        ("threshold=5 (out of range) -> 422", "doc.png", image, "image/png", {"threshold": 5}, 422),
    ]
    for label, name, content, ctype, params, status in cases:
        r, _ = post_verify(client, name, content, api_key, ctype, params)
        res.add(sec, label, is_error_envelope(r, status), f"got {r.status_code}")
    r = client.post("/verify", headers={"X-API-Key": api_key})
    res.add(sec, "Missing file field -> 400", is_error_envelope(r, 400), f"got {r.status_code}")
    r = client.get("/does-not-exist")
    res.add(sec, "Unknown route -> 404", is_error_envelope(r, 404), f"got {r.status_code}")
    r = client.get("/verify", headers={"X-API-Key": api_key})
    res.add(sec, "Wrong method (GET /verify) -> 405", is_error_envelope(r, 405), f"got {r.status_code}")


def pipeline_reference(image_path: Path) -> Dict[str, Any]:
    """Decode, extract the model's features and run inference in-process for one image."""
    import numpy as np

    from backend import config
    from core.document_analyzer import load_image
    from core.feature_schemas import get_schema
    from core.model_trainer import DocumentFraudDetector

    if not hasattr(pipeline_reference, "_model"):
        pipeline_reference._model = DocumentFraudDetector.load(Path(config.MODEL_PATH))  # type: ignore[attr-defined]
    model, metadata = pipeline_reference._model  # type: ignore[attr-defined]
    schema = get_schema(metadata.get("feature_schema_version"))

    image = load_image(image_path)
    vector = schema.vector(schema.extract(image_path))
    proba = model.predict_proba(np.array([vector], dtype=np.float64))[0]
    return {
        "image_size": image.size,
        "schema": schema.version,
        "n_features": len(vector),
        "expected_features": len(schema.names),
        "features_finite": all(math.isfinite(v) for v in vector),
        "score": round(float(proba[1]), 4),
    }


def test_demo_images(client: httpx.Client, api_key: str, manifest: Dict[str, Any], res: Results) -> Dict[str, float]:
    """Sections 4 and 5: pipeline stages and all demo images. Returns API scores by file."""
    stage, demo = "Pipeline stages", "Demo images"
    table = [f"{'File':<26}{'Actual':<10}{'Predicted':<11}{'Risk':>6}{'Conf':>7}{'Recorded':>10}{'Latency':>10}"]
    api_scores: Dict[str, float] = {}

    for record in manifest["images"]:
        name = record["demo_file"]
        path = DEMO_DIR / name
        if not res.add(demo, f"{name}: file present", path.exists(), path.relative_to(ROOT).as_posix()):
            continue

        r, latency = post_verify(client, name, path.read_bytes(), api_key)
        ok = r.status_code == 200
        body = r.json() if ok else {}
        valid, why = valid_verify_body(body) if ok else (False, f"HTTP {r.status_code}")
        if ok and valid:
            regions = body["flagged_regions"]
            why = f"flagged_regions: {len(regions)}" + (
                " (" + ", ".join(f"{g['type']} at {g['x']},{g['y']} {g['width']}x{g['height']}" for g in regions) + ")"
                if regions else "")
        res.add(demo, f"{name}: HTTP 200 + valid JSON schema", ok and valid, why)
        if not (ok and valid):
            continue

        score, conf = body["tamper_risk_score"], body["confidence"]
        api_scores[name] = score
        predicted = "tampered" if score >= manifest["risk_threshold"] else "genuine"
        res.add(demo, f"{name}: risk score generated", isinstance(score, float), f"{score:.4f}")
        res.add(demo, f"{name}: confidence generated", isinstance(conf, float), f"{conf:.4f}")
        res.add(demo, f"{name}: interpretation generated", bool(body["interpretation"].strip()), "")
        res.add(
            demo,
            f"{name}: matches manifest (score {record['tamper_risk_score']:.4f})",
            abs(score - record["tamper_risk_score"]) <= SCORE_TOLERANCE and predicted == record["predicted_label"],
            f"api={score:.4f} {predicted}",
        )
        table.append(
            f"{name:<26}{record['actual_label']:<10}{predicted:<11}{score:>6.2f}{conf:>7.2f}"
            f"{record['tamper_risk_score']:>10.2f}{latency * 1000:>8.0f}ms"
        )

        ref = pipeline_reference(path)
        res.add(
            stage,
            f"{name}: image decoded",
            min(ref["image_size"]) > 0,
            f"{ref['image_size'][0]}x{ref['image_size'][1]} RGB",
        )
        res.add(
            stage,
            f"{name}: {ref['expected_features']} finite features extracted ({ref['schema']})",
            ref["n_features"] == ref["expected_features"] and ref["features_finite"],
            f"{ref['n_features']} features",
        )
        res.add(
            stage,
            f"{name}: in-process model score == API score",
            abs(ref["score"] - score) <= SCORE_TOLERANCE,
            f"in-process {ref['score']:.4f} vs API {score:.4f}",
        )

    res.tables["Demo images"] = table
    return api_scores


def test_reliability(
    client: httpx.Client,
    api_key: str,
    manifest: Dict[str, Any],
    api_scores: Dict[str, float],
    iterations: int,
    server_pid: Optional[int],
    proc: Optional[subprocess.Popen],
    res: Results,
) -> None:
    """Section 6: consecutive requests, crashes, memory growth and prediction consistency."""
    sec = "Reliability"
    names = [r["demo_file"] for r in manifest["images"] if (DEMO_DIR / r["demo_file"]).exists()]
    payloads = {n: (DEMO_DIR / n).read_bytes() for n in names}

    latencies: List[float] = []
    rss: List[Optional[float]] = [rss_mb(server_pid) if server_pid else None]
    failures, mismatches = [], []
    for i in range(iterations):
        name = names[i % len(names)]
        try:
            r, latency = post_verify(client, name, payloads[name], api_key)
        except httpx.HTTPError as exc:
            failures.append(f"#{i + 1} {name}: {exc!r}")
            continue
        latencies.append(latency)
        rss.append(rss_mb(server_pid) if server_pid else None)
        if r.status_code != 200:
            failures.append(f"#{i + 1} {name}: HTTP {r.status_code}")
            continue
        score = r.json()["tamper_risk_score"]
        if name in api_scores and abs(score - api_scores[name]) > SCORE_TOLERANCE:
            mismatches.append(f"#{i + 1} {name}: {score} != {api_scores[name]}")

    res.add(
        sec, f"{iterations} consecutive requests all HTTP 200", not failures, "; ".join(failures[:3]) or "0 failures"
    )
    alive = proc is None or proc.poll() is None
    healthy = client.get("/health").status_code == 200 if alive else False
    res.add(
        sec,
        "Server process still running after the run (no crash)",
        alive and healthy,
        "health 200" if healthy else "server down",
    )
    res.add(
        sec,
        "Predictions consistent (same image -> same score)",
        not mismatches,
        "; ".join(mismatches[:3]) or f"{len(names)} images x {iterations // max(len(names), 1)} repeats, identical",
    )

    measured = [m for m in rss if m is not None]
    if len(measured) > MEMORY_WARMUP_REQUESTS + 1:
        warm, final = measured[MEMORY_WARMUP_REQUESTS], measured[-1]
        growth = final - warm
        res.add(
            sec,
            f"No memory leak (RSS growth after warm-up < {MEMORY_GROWTH_LIMIT_MB:.0f} MB)",
            growth < MEMORY_GROWTH_LIMIT_MB,
            f"start {measured[0]:.1f} MB, after {MEMORY_WARMUP_REQUESTS} req {warm:.1f} MB, end {final:.1f} MB "
            f"({growth:+.1f} MB post-warm-up)",
        )
    else:
        res.add(sec, "Memory measured", False, "RSS unavailable (external server or unsupported OS; install psutil)")

    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
        res.tables["Reliability"] = [
            f"Requests          : {iterations} (demo images cycled: {', '.join(names)})",
            f"Latency avg       : {statistics.mean(latencies) * 1000:.0f} ms",
            f"Latency median    : {statistics.median(latencies) * 1000:.0f} ms",
            f"Latency p95       : {p95 * 1000:.0f} ms",
            f"Latency min / max : {ordered[0] * 1000:.0f} / {ordered[-1] * 1000:.0f} ms",
            "Server RSS (MB)   : " + (", ".join(f"{m:.0f}" for m in measured) if measured else "not measured"),
        ]


# -------------------------------------------------------------------------
# Report
# -------------------------------------------------------------------------


def build_report(res: Results, meta: Dict[str, str]) -> str:
    """Render the plain-text E2E report."""
    rule, thin = "=" * 78, "-" * 78
    passed = sum(c.passed for c in res.checks)
    total = len(res.checks)
    lines = [
        rule,
        "DOCUMENT FRAUD DETECTION API - END-TO-END TEST REPORT",
        rule,
        *(f"{k:<16}: {v}" for k, v in meta.items()),
        f"{'Overall result':<16}: {'PASS' if passed == total else 'FAIL'} ({passed}/{total} checks passed)",
        "",
        "WORKFLOW UNDER TEST",
        thin,
        "Upload image -> API authentication -> input validation -> image decoding",
        "  -> feature extraction (the model's schema) -> model inference -> JSON response",
        "",
    ]
    sections = [
        "Server start-up",
        "Authentication",
        "Input validation",
        "Pipeline stages",
        "Demo images",
        "Reliability",
    ]
    for i, sec in enumerate(sections, 1):
        checks = res.section_checks(sec)
        ok = sum(c.passed for c in checks)
        lines += [f"{i}. {sec.upper()}  ({ok}/{len(checks)} passed)", thin]
        for c in checks:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}" + (f"  -- {c.detail}" if c.detail else ""))
        if sec in res.tables:
            lines += ["", *("  " + row for row in res.tables[sec])]
        lines.append("")
    lines += [
        "NOTES",
        thin,
        "- 'Actual' labels and 'Recorded' scores come from data/demo/demo_manifest.json",
        "  (Founder A; re-scored for each deployed model). The demo set is a demonstration",
        "  record, not an accuracy estimate: held-out accuracy is in",
        "  logs/model_validation_report.txt.",
        "- Pipeline-stage checks recompute the model's features and score in this",
        "  process and require the API's score to match exactly.",
        "- Latency depends on the machine; compare runs on the same hardware only.",
        "",
        "HOW TO RUN",
        thin,
        "python scripts/e2e_test.py",
        "python scripts/e2e_test.py --url http://127.0.0.1:8000 --api-key <key>",
        rule,
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Parse arguments, run all sections, write the report and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="Test an already running server instead of starting one")
    parser.add_argument("--api-key", default="e2e-test-key", help="API key (set on the started server)")
    parser.add_argument("--port", type=int, default=8765, help="Port for the started server")
    parser.add_argument("--python", default=sys.executable, help="Interpreter used to start uvicorn")
    parser.add_argument("--iterations", type=int, default=20, help="Consecutive requests in the reliability test")
    parser.add_argument("--startup-timeout", type=float, default=90.0, help="Seconds to wait for /health")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Report path")
    args = parser.parse_args()

    from backend import config

    res = Results()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    sample = (DEMO_DIR / manifest["images"][0]["demo_file"]).read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    server_log = args.output.with_name("e2e_server.log")

    proc: Optional[subprocess.Popen] = None
    if args.url:
        base_url = args.url.rstrip("/")
        mode = f"existing server at {base_url}"
    else:
        base_url = f"http://127.0.0.1:{args.port}"
        shown_log = server_log.relative_to(ROOT).as_posix() if server_log.is_relative_to(ROOT) else server_log
        mode = f"uvicorn started by this script on {base_url} (log: {shown_log})"
        proc = start_server(args.python, ROOT, args.port, args.api_key, server_log)

    run_start = time.perf_counter()
    try:
        startup = wait_for_health(base_url, args.startup_timeout, proc)
        res.add(
            "Server start-up",
            "Server answers /health with 200",
            startup is not None,
            f"healthy after {startup:.1f} s (includes model load)" if startup is not None else "did not become healthy",
        )
        if startup is None:
            raise RuntimeError("server did not start")
        with httpx.Client(base_url=base_url, timeout=120) as client:
            body = client.get("/health").json()
            res.add("Server start-up", "/health body", body.get("status") == "ok", json.dumps(body))
            res.add("Server start-up", "/docs (Swagger UI) served", client.get("/docs").status_code == 200, "")
            test_authentication(client, args.api_key, sample, res)
            test_validation(client, args.api_key, sample, config.MAX_FILE_SIZE, res)
            api_scores = test_demo_images(client, args.api_key, manifest, res)
            test_reliability(
                client, args.api_key, manifest, api_scores, args.iterations, proc.pid if proc else None, proc, res
            )
    except Exception as exc:  # report whatever was collected, then fail
        res.add("Server start-up", "Run completed without an exception", False, repr(exc))
    finally:
        if proc is not None:
            stop_server(proc)

    meta = {
        "Generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Server": mode,
        "Python": f"{platform.python_version()} ({platform.system()} {platform.machine()})",
        "Model": f"{Path(config.MODEL_PATH).name} (version {manifest['model'].get('model_version', '?')})",
        "Iterations": str(args.iterations),
        "Wall time": f"{time.perf_counter() - run_start:.1f} s",
    }
    report = build_report(res, meta)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to {args.output}")
    return 0 if all(c.passed for c in res.checks) else 1


if __name__ == "__main__":
    sys.exit(main())
