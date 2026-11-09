"""
scripts/inference_benchmark.py
Founder A (Day 10): inference performance of the deployed model.

Usage:
    python scripts/inference_benchmark.py              # in-process and over HTTP (starts its own server)
    python scripts/inference_benchmark.py --no-http    # in-process only
    python scripts/inference_benchmark.py --passes 3 --report <path> --json <path>

Measures, on the machine it runs on:
  1. Model load: a fresh ModelAPI.load_model() (unpickle + checks), repeated.
  2. In-process inference: ModelAPI.predict(bytes, filename), as /verify calls it (temporary file,
     features, forest, regions). Covers the 5 demo images, the 15 frozen TEST images and the 149 HOLDOUT
     documents, repeated --passes times. Records min / mean / median / p95 / max, run-to-run variation
     per document, whether every repeat returns the identical output, and crashes.
  3. Large inputs: a 2400x1800 JPEG fixture, a 4000x5000 JPEG page and a one-page A4 PDF.
  4. HTTP: a real uvicorn server. Each demo image is sent 10 times in sequence, then 5 concurrent users
     send 10 requests each. This is the sprint-target set-up: /verify average under 2 s at 5 users and
     a success rate above 95%. Every response must equal the in-process output for the same file.

Targets were fixed before measuring (TARGETS below). Timings depend on the hardware; compare runs made
on the same machine only.

Writes logs/inference_benchmark_report.txt and logs/inference_benchmark_results.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import socket
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import final_validation as fv  # noqa: E402
import model_stability_test as mst  # noqa: E402

from backend.model_utils import ModelAPI  # noqa: E402
from core.model_trainer import DEFAULT_MODEL_PATH  # noqa: E402
from core.tamper_features import original_size  # noqa: E402
from scripts.e2e_test import start_server, stop_server, wait_for_health  # noqa: E402
from scripts.generate_extra_documents import load_set  # noqa: E402

DEFAULT_REPORT = ROOT / "logs" / "inference_benchmark_report.txt"
DEFAULT_JSON = ROOT / "logs" / "inference_benchmark_results.json"
REVIEW_JSON = ROOT / "logs" / "model_review_results.json"
LOAD_REPORT = ROOT / "logs" / "final_load_report.txt"
LARGE_FIXTURE = ROOT / "tests" / "fixtures" / "large_image.jpg"
TYPICAL_DEMO = "genuine_payslip_01.png"  # the file used by Founder B's load tests
PASSES = 3
LOAD_RUNS = 5
LARGE_RUNS = 3
HTTP_REPEATS = 10
USERS, REQUESTS_PER_USER = 5, 10
API_KEY = "benchmark-key"
OUTPUT_KEYS = ("tamper_risk_score", "confidence", "flagged_regions", "interpretation")

# Fixed before measuring.
MAX_SECONDS_TYPICAL = 2.0   # any single typical page, in-process
MAX_P95_OVER_MEDIAN = 2.0   # no long tail: p95 at most twice the median
MAX_MEDIAN_CV = 0.10        # run-to-run variation per document (coefficient of variation), median over documents
HTTP_AVG_SECONDS = 2.0      # sprint target at 5 concurrent users
HTTP_SUCCESS_RATE = 0.95    # sprint target
TARGETS = {
    "stable": "every repeated prediction identical (score, confidence, regions, interpretation), "
              "in-process and over HTTP",
    "no_crash": "0 crashes; every HTTP request answered 200",
    "latency": f"every typical page (longest side <= 2000 px) under {MAX_SECONDS_TYPICAL:.0f} s in-process",
    "consistent": f"p95 <= {MAX_P95_OVER_MEDIAN:.0f}x median; median run-to-run variation per document "
                  f"<= {MAX_MEDIAN_CV:.0%}",
    "http": f"/verify average < {HTTP_AVG_SECONDS:.0f} s at {USERS} concurrent users and success rate > "
            f"{HTTP_SUCCESS_RATE:.0%} (sprint targets)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats_ms(seconds: list[float]) -> dict:
    ms = [1000 * s for s in seconds]
    return {"n": len(ms), "min": min(ms), "mean": statistics.fmean(ms), "median": statistics.median(ms),
            "p95": float(np.percentile(ms, 95)), "max": max(ms), "stdev": statistics.pstdev(ms)}


def output_of(out: dict) -> dict:
    return {k: out[k] for k in OUTPUT_KEYS}


def corpus() -> list[dict]:
    """Demo images, frozen TEST images and HOLDOUT documents, with their bytes."""
    _, images = mst.load_demo()
    split = json.loads(fv.SPLIT_PATH.read_text(encoding="utf-8"))
    with open(fv.DATASET / "index.csv", newline="", encoding="utf-8") as f:
        index = {r["path"]: r for r in csv.DictReader(f)}
    docs = [{"set": "DEMO", "name": r["demo_file"], "file": mst.DEMO_DIR / r["demo_file"], "doc_type": "payslip"}
            for r in images]
    docs += [{"set": "TEST", "name": r["path"], "file": fv.DATASET / r["path"],
              "doc_type": index[r["path"]]["doc_type"]} for r in split["test"]]
    docs += [{"set": "HOLDOUT", "name": r["path"], "file": Path(r["file"]), "doc_type": r["doc_type"]}
             for r in load_set("holdout")]
    for d in docs:
        d["bytes"] = d["file"].read_bytes()
        d["size"] = original_size(d["file"])
    return docs


# ── 1. Model load ───────────────────────────────────────────────────────────

def model_load(runs: int = LOAD_RUNS) -> dict:
    seconds = []
    for _ in range(runs):
        api = ModelAPI(DEFAULT_MODEL_PATH)
        t0 = time.perf_counter()
        api.load_model()
        seconds.append(time.perf_counter() - t0)
    return stats_ms(seconds)


# ── 2. In-process inference ─────────────────────────────────────────────────

def in_process(api: ModelAPI, docs: list[dict], passes: int) -> dict:
    t0 = time.perf_counter()
    first = api.predict(docs[0]["bytes"], filename=Path(docs[0]["name"]).name)
    cold_first_ms = 1000 * (time.perf_counter() - t0)
    crashes, times, outputs = [], {d["name"]: [] for d in docs}, {d["name"]: [] for d in docs}
    for _ in range(passes):
        for d in docs:
            t0 = time.perf_counter()
            try:
                out = api.predict(d["bytes"], filename=Path(d["name"]).name)
            except Exception as exc:  # any failure on these valid documents counts as a crash
                crashes.append({"document": d["name"], "error": f"{type(exc).__name__}: {exc}"[:160]})
                continue
            times[d["name"]].append(time.perf_counter() - t0)
            outputs[d["name"]].append(output_of(out))
    unstable = [n for n, outs in outputs.items() if any(o != outs[0] for o in outs[1:])]
    cv = [statistics.pstdev(t) / statistics.fmean(t) for t in times.values() if len(t) > 1]
    all_times = [s for t in times.values() for s in t]
    by_group: dict[str, list[float]] = {}
    for d in docs:
        by_group.setdefault(d["set"], []).extend(times[d["name"]])
        by_group.setdefault(f"type:{d['doc_type']}", []).extend(times[d["name"]])
    per_doc = sorted(({"document": d["name"], "set": d["set"], "size": list(d["size"]),
                       "median_ms": 1000 * statistics.median(times[d["name"]])} for d in docs if times[d["name"]]),
                     key=lambda x: -x["median_ms"])
    typical = [s for d in docs if max(d["size"]) <= 2000 for s in times[d["name"]]]
    return {
        "documents": len(docs), "passes": passes, "predictions": len(all_times) + len(crashes),
        "cold_first_call_ms": cold_first_ms, "first_output": output_of(first),
        "overall": stats_ms(all_times), "groups": {k: stats_ms(v) for k, v in sorted(by_group.items())},
        "typical_max_ms": 1000 * max(typical) if typical else None,
        "run_to_run_cv": {"median": statistics.median(cv), "max": max(cv)} if cv else None,
        "slowest": per_doc[:5], "fastest": per_doc[-3:],
        "unstable": unstable, "crashes": crashes,
        "outputs": {n: o[0] for n, o in outputs.items() if o},
    }


# ── 3. Large inputs ─────────────────────────────────────────────────────────

def large_inputs(api: ModelAPI, tmp: Path, runs: int = LARGE_RUNS) -> list[dict]:
    with Image.open(mst.DEMO_DIR / TYPICAL_DEMO) as im:
        page = im.convert("RGB")
    big = tmp / "page_4000x5000.jpg"
    page.resize((4000, 5000), Image.LANCZOS).save(big, quality=90)
    cases = [("2400x1800 JPEG (test fixture)", LARGE_FIXTURE), ("4000x5000 JPEG page (20 MP)", big)]
    try:
        import pymupdf

        pdf = tmp / "page.pdf"
        with pymupdf.open() as doc:
            p = doc.new_page(width=595, height=842)
            p.insert_image(p.rect, filename=str(mst.DEMO_DIR / TYPICAL_DEMO))
            doc.save(str(pdf))
        cases.append(("A4 PDF, one page (rendered at 200 DPI)", pdf))
    except ImportError:
        pass
    rows = []
    for name, path in cases:
        data, seconds, outs, error = path.read_bytes(), [], [], None
        for _ in range(runs):
            t0 = time.perf_counter()
            try:
                outs.append(output_of(api.predict(data, filename=path.name)))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:160]
                break
            seconds.append(time.perf_counter() - t0)
        rows.append({"case": name, "bytes": len(data), "size": list(original_size(path)), "error": error,
                     "stats": stats_ms(seconds) if seconds else None,
                     "stable": bool(outs) and all(o == outs[0] for o in outs),
                     "score": outs[0]["tamper_risk_score"] if outs else None})
    return rows


# ── 4. HTTP ─────────────────────────────────────────────────────────────────

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _peak_rss_mb(pid: int) -> dict | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    vals = {k: int(re.search(rf"^{k}:\s+(\d+) kB", text, re.M).group(1)) / 1024 for k in ("VmHWM", "VmRSS")}
    return {"peak_mb": vals["VmHWM"], "current_mb": vals["VmRSS"]}


def http_benchmark(expected: dict[str, dict], tmp: Path) -> dict:
    """Sequential and concurrent /verify against a real uvicorn server (started here)."""
    _, images = mst.load_demo()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = start_server(sys.executable, ROOT, port, API_KEY, tmp / "server.log")
    try:
        started = wait_for_health(base, 120, proc)
        if started is None:
            return {"available": False, "reason": "server did not become healthy",
                    "log": (tmp / "server.log").read_text(encoding="utf-8")[-600:]}
        headers = {"X-API-Key": API_KEY}
        mismatches, errors, sequential = [], [], {}
        with httpx.Client(base_url=base, timeout=60) as client:
            for rec in images:
                name = rec["demo_file"]
                content = (mst.DEMO_DIR / name).read_bytes()
                seconds = []
                for _ in range(HTTP_REPEATS):
                    t0 = time.perf_counter()
                    r = client.post("/verify", files={"file": (name, content, "image/png")}, headers=headers)
                    seconds.append(time.perf_counter() - t0)
                    if r.status_code != 200:
                        errors.append(f"{name}: HTTP {r.status_code}")
                    elif output_of(r.json()) != expected[name]:
                        mismatches.append(name)
                sequential[name] = stats_ms(seconds)

        content = (mst.DEMO_DIR / TYPICAL_DEMO).read_bytes()

        def user(_: int) -> list[tuple[int, float, dict | None]]:
            results = []
            with httpx.Client(base_url=base, timeout=120) as c:
                for _ in range(REQUESTS_PER_USER):
                    t0 = time.perf_counter()
                    r = c.post("/verify", files={"file": (TYPICAL_DEMO, content, "image/png")}, headers=headers)
                    results.append((r.status_code, time.perf_counter() - t0,
                                    output_of(r.json()) if r.status_code == 200 else None))
            return results

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=USERS) as pool:
            results = [x for batch in pool.map(user, range(USERS)) for x in batch]
        wall = time.perf_counter() - t0
        ok = [x for x in results if x[0] == 200]
        errors += [f"concurrent: HTTP {code}" for code, _, _ in results if code != 200]
        mismatches += [f"concurrent {TYPICAL_DEMO}" for code, _, out in ok if out != expected[TYPICAL_DEMO]]
        memory = _peak_rss_mb(proc.pid)
    finally:
        stop_server(proc)
    return {
        "available": True, "startup_seconds": started,
        "sequential": sequential,
        "sequential_mean_ms": statistics.fmean(v["mean"] for v in sequential.values()),
        "sequential_max_ms": max(v["max"] for v in sequential.values()),
        "sequential_min_ms": min(v["min"] for v in sequential.values()),
        "concurrent": {"users": USERS, "requests": len(results), "succeeded": len(ok),
                       "success_rate": len(ok) / len(results), "throughput_rps": len(results) / wall,
                       **stats_ms([s for _, s, _ in results])},
        "mismatches": mismatches, "errors": errors, "server_memory": memory,
    }


# ── Targets and earlier measurements ────────────────────────────────────────

def target_checks(r: dict) -> list[dict]:
    ip, http = r["in_process"], r.get("http") or {}
    large_ok = all(x["error"] is None and x["stable"] for x in r["large"])
    checks = [
        {"target": "stable", "pass": not ip["unstable"] and large_ok and not (http.get("mismatches")),
         "detail": f"{ip['predictions']} in-process predictions, {len(ip['unstable'])} documents with differing "
                   f"repeats; HTTP responses differing from in-process: {len(http.get('mismatches', []))}"},
        {"target": "no_crash", "pass": not ip["crashes"] and large_ok and not http.get("errors"),
         "detail": f"{len(ip['crashes'])} in-process crashes; HTTP errors: {len(http.get('errors', []))}"},
        {"target": "latency", "pass": ip["typical_max_ms"] is not None and ip["typical_max_ms"] < 1000 * MAX_SECONDS_TYPICAL,
         "detail": f"slowest typical page {ip['typical_max_ms']:.0f} ms"},
        {"target": "consistent",
         "pass": ip["overall"]["p95"] <= MAX_P95_OVER_MEDIAN * ip["overall"]["median"]
         and ip["run_to_run_cv"]["median"] <= MAX_MEDIAN_CV,
         "detail": f"p95 {ip['overall']['p95']:.0f} ms = {ip['overall']['p95'] / ip['overall']['median']:.2f}x median; "
                   f"run-to-run variation median {ip['run_to_run_cv']['median']:.1%}"},
    ]
    if http.get("available"):
        c = http["concurrent"]
        checks.append({"target": "http",
                       "pass": c["mean"] < 1000 * HTTP_AVG_SECONDS and c["success_rate"] > HTTP_SUCCESS_RATE,
                       "detail": f"{USERS} users: average {c['mean']:.0f} ms, p95 {c['p95']:.0f} ms, "
                                 f"success {c['success_rate']:.0%} ({c['succeeded']}/{c['requests']})"})
    else:
        checks.append({"target": "http", "pass": False, "detail": "not measured (" + http.get("reason", "--no-http") + ")"})
    for c in checks:
        c["description"] = TARGETS[c["target"]]
    return checks


def earlier_measurements() -> list[dict]:
    rows = []
    if REVIEW_JSON.exists():
        e2e = json.loads(REVIEW_JSON.read_text(encoding="utf-8")).get("optimisation", {}).get("end_to_end_ms", {})
        for key, label in (("typical page, ModelAPI.predict (served)", "typical page, in-process (median)"),
                           ("4000x5000 JPEG, ModelAPI.predict (served)", "4000x5000 JPEG, in-process (median)")):
            if key in e2e:
                rows.append({"when": "Day 9 review, model v2", "what": label, "ms": e2e[key],
                             "source": REVIEW_JSON.relative_to(ROOT).as_posix()})
    if LOAD_REPORT.exists():
        m = re.search(r"5 users: avg ([\d.]+)s, p95 ([\d.]+)s", LOAD_REPORT.read_text(encoding="utf-8"))
        if m:
            rows.append({"when": "Day 10 load test, model v1", "what": "/verify at 5 users (average)",
                         "ms": 1000 * float(m.group(1)), "source": LOAD_REPORT.relative_to(ROOT).as_posix()})
    return rows


# ── Run ─────────────────────────────────────────────────────────────────────

def run(passes: int = PASSES, http: bool = True) -> dict:
    load = model_load()
    api = ModelAPI(DEFAULT_MODEL_PATH)
    api.load_model()
    docs = corpus()
    ip = in_process(api, docs, passes)
    with tempfile.TemporaryDirectory() as tmp:
        large = large_inputs(api, Path(tmp))
        net = http_benchmark(ip["outputs"], Path(tmp)) if http else {"available": False, "reason": "--no-http"}
    r = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": mst.environment(),
        "model": {"path": DEFAULT_MODEL_PATH.relative_to(ROOT).as_posix(), "sha256": sha256(DEFAULT_MODEL_PATH),
                  "version": api.metadata.get("model_version"), "schema": api.schema.version},
        "model_load": load, "in_process": {k: v for k, v in ip.items() if k != "outputs"},
        "large": large, "http": net, "earlier": earlier_measurements(),
    }
    r["targets"] = target_checks(r)
    r["targets_met"] = all(c["pass"] for c in r["targets"])
    return r


# ── Report ──────────────────────────────────────────────────────────────────

def _row(label: str, s: dict) -> str:
    return (f"  {label:<40} {s['n']:>5} {s['min']:>8.0f} {s['mean']:>8.0f} {s['median']:>8.0f} "
            f"{s['p95']:>8.0f} {s['max']:>8.0f}")


HEADER = f"  {'':<40} {'n':>5} {'min':>8} {'mean':>8} {'median':>8} {'p95':>8} {'max':>8}   (ms)"


def build_report(r: dict) -> str:
    env, ip = r["environment"], r["in_process"]
    L = [
        "=" * 78,
        "DOCFRAUD - INFERENCE PERFORMANCE BENCHMARK (Day 10, Founder A)",
        "=" * 78,
        f"Generated (UTC): {r['generated']}",
        "Script         : scripts/inference_benchmark.py",
        f"Model          : {r['model']['path']} v{r['model']['version']} ({r['model']['schema']})",
        f"SHA-256        : {r['model']['sha256']}",
        f"Machine        : {env['platform']}, {env['cpus']} CPUs, Python {env['python']}; "
        + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
        f"Result         : targets {'MET' if r['targets_met'] else 'NOT MET'}",
        "",
        "1. TARGETS (fixed before measuring)",
        "-" * 78,
    ]
    for c in r["targets"]:
        L += [f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['description']}", f"         {c['detail']}"]

    lo = r["model_load"]
    L += ["", "2. MODEL LOAD (fresh ModelAPI.load_model: unpickle, schema checks, n_jobs=1)", "-" * 78,
          f"  {lo['n']} loads: min {lo['min']:.0f} ms, average {lo['mean']:.0f} ms, max {lo['max']:.0f} ms"]

    L += ["", "3. IN-PROCESS INFERENCE (ModelAPI.predict(bytes): temp file, features, forest, regions)", "-" * 78,
          f"  {ip['documents']} documents x {ip['passes']} passes = {ip['predictions']} predictions. "
          f"First call after load: {ip['cold_first_call_ms']:.0f} ms.", "", HEADER,
          _row("ALL", ip["overall"])]
    L += [_row(k.replace("type:", "doc type: "), s) for k, s in ip["groups"].items()]
    cv = ip["run_to_run_cv"]
    L += ["", f"  Run-to-run variation per document (coefficient of variation): median {cv['median']:.1%}, "
              f"max {cv['max']:.1%}.",
          f"  Identical output on every repeat: {ip['documents'] - len(ip['unstable'])}/{ip['documents']} documents. "
          f"Crashes: {len(ip['crashes'])}.",
          "  Slowest documents (median): " + ", ".join(
              f"{x['document']} {x['median_ms']:.0f} ms ({x['size'][0]}x{x['size'][1]})" for x in ip["slowest"][:3])]

    L += ["", "4. LARGE INPUTS (downscaled to 2000 px before analysis)", "-" * 78, HEADER]
    for x in r["large"]:
        if x["stats"]:
            L.append(_row(x["case"], x["stats"]) + ("" if x["stable"] else "  (outputs differ)"))
        else:
            L.append(f"  {x['case']:<40} ERROR {x['error']}")

    h = r["http"]
    L += ["", "5. HTTP /verify (real uvicorn server, 1 process, client on the same machine)", "-" * 78]
    if h.get("available"):
        c = h["concurrent"]
        L += [f"  Server healthy after {h['startup_seconds']:.1f} s (includes model load).",
              f"  Sequential, {HTTP_REPEATS} requests per demo image:", HEADER]
        L += [_row(name, s) for name, s in h["sequential"].items()]
        L += ["", f"  Concurrent: {c['users']} users x {REQUESTS_PER_USER} requests ({TYPICAL_DEMO}):", HEADER,
              _row(f"{c['users']} users", c),
              f"  Success {c['succeeded']}/{c['requests']} ({c['success_rate']:.0%}); throughput "
              f"{c['throughput_rps']:.1f} requests/s.",
              f"  Responses equal to the in-process output: "
              f"{'all' if not h['mismatches'] else 'NO - ' + ', '.join(h['mismatches'][:5])}."]
        if h.get("server_memory"):
            L.append(f"  Server memory: peak {h['server_memory']['peak_mb']:.0f} MB, "
                     f"{h['server_memory']['current_mb']:.0f} MB at the end.")
    else:
        L.append(f"  Not measured: {h.get('reason')}.")

    L += ["", "6. EARLIER MEASUREMENTS", "-" * 78]
    for e in r["earlier"]:
        L.append(f"  {e['when']:<28} {e['what']:<36} {e['ms']:>7.0f} ms   ({e['source']})")
    L += ["  Earlier figures come from other runs and possibly other hardware, so compare them as indications",
          "  only. Differences in method: Day 9 timed ModelAPI.predict(path) on one page; this run sends bytes (as",
          "  /verify does, via a temporary file) and includes flagged_regions (about 1-3 ms). The v1 figure is",
          "  Founder B's load test of model v1 on the same type of 4-CPU container.",
          "",
          "7. NOTES", "-" * 78,
          "  - Timings are for this machine. The HTTP client runs on the same machine, so concurrent timings",
          "    include client overhead.",
          "  - One uvicorn process with MAX_CONCURRENT_ANALYSES slots; more users queue. Multi-worker deployments",
          "    were not measured.",
          ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Inference performance benchmark of the deployed model")
    ap.add_argument("--passes", type=int, default=PASSES, help=f"repeats over the corpus (default {PASSES})")
    ap.add_argument("--no-http", action="store_true", help="skip the real-server HTTP measurements")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()
    logging.getLogger("backend").setLevel(logging.WARNING)
    r = run(args.passes, http=not args.no_http)
    report = build_report(r)
    for path, text in ((args.report, report), (args.json, json.dumps(r, indent=1) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(report)
    print(f"Report written: {args.report}\nResults written: {args.json}")
    return 0 if r["targets_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
