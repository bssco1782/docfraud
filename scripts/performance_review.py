"""
scripts/performance_review.py
Measure start-up time, prediction time, memory and CPU usage of the API.

Everything is measured on this machine, against a real uvicorn process started by the
script (the same way scripts/e2e_test.py does it):

  1. Start-up       python import time of backend.main, model load time, and time from
                    launching uvicorn until /health returns 200
  2. Prediction     per-stage timing in-process (image decode, ELA, feature extraction,
                    model inference) for the 5 demo images, plus end-to-end HTTP latency
  3. Large inputs   a 4000x5000 image and an A4 PDF through the API
  4. Memory         server RSS at idle, after the workload, and peak (VmHWM)
  5. CPU            server CPU seconds per request, and CPU utilisation / throughput at
                    1, 2 and 4 concurrent clients

Writes logs/performance_review.txt by default. Linux reads /proc for memory and CPU;
elsewhere install psutil for those numbers.

Usage (from the repository root):
    python scripts/performance_review.py
    python scripts/performance_review.py --repeats 5 --output logs/performance_review.txt
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e_test import rss_mb, start_server, stop_server, wait_for_health  # noqa: E402

DEMO_DIR = ROOT / "data" / "demo"
DEFAULT_OUTPUT = ROOT / "logs" / "performance_review.txt"
API_KEY = "perf-review-key"


# -------------------------------------------------------------------------
# Process metrics
# -------------------------------------------------------------------------


def cpu_seconds(pid: int) -> Optional[float]:
    """User + system CPU seconds consumed by a process so far."""
    try:
        import psutil

        times = psutil.Process(pid).cpu_times()
        return times.user + times.system
    except ImportError:
        pass
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        fields = stat.read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    return None


def peak_rss_mb(pid: int) -> Optional[float]:
    """Peak resident memory (VmHWM) of a process in MB, Linux only."""
    status = Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return None


def fmt(value: Optional[float], unit: str, digits: int = 1) -> str:
    """Format an optional number with a unit."""
    return "n/a" if value is None else f"{value:.{digits}f} {unit}"


def stats_ms(samples: List[float]) -> str:
    """mean / median / max of second-valued samples, in ms."""
    ms = [s * 1000 for s in samples]
    return f"mean {statistics.mean(ms):6.0f} ms   median {statistics.median(ms):6.0f} ms   max {max(ms):6.0f} ms"


# -------------------------------------------------------------------------
# Measurements
# -------------------------------------------------------------------------


def measure_import_time() -> float:
    """Seconds for a fresh interpreter to import backend.main (no model load)."""
    code = "import time; t=time.perf_counter(); import backend.main; print(time.perf_counter()-t)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return float(out.stdout.strip().splitlines()[-1])


def measure_stages(repeats: int) -> Tuple[List[str], Dict[str, List[float]]]:
    """Time each pipeline stage in-process on the demo images (the deployed model's feature schema)."""
    import numpy as np

    from backend.model_utils import ModelAPI
    from core.document_analyzer import load_image

    api = ModelAPI()
    t0 = time.perf_counter()
    api.load_model()
    load_time = time.perf_counter() - t0
    schema = api.schema
    extract_stage = f"feature extraction ({schema.version}, incl. decode)"

    stages: Dict[str, List[float]] = {
        "load_image (decode + resize)": [],
        extract_stage: [],
        "model predict_proba": [],
    }
    for path in sorted(DEMO_DIR.glob("*.png")):
        for _ in range(repeats):
            t = time.perf_counter()
            load_image(path)
            stages["load_image (decode + resize)"].append(time.perf_counter() - t)
            t = time.perf_counter()
            feats = schema.extract(path)
            stages[extract_stage].append(time.perf_counter() - t)
            vec = np.array([schema.vector(feats)])
            t = time.perf_counter()
            api.model.predict_proba(vec)
            stages["model predict_proba"].append(time.perf_counter() - t)
    lines = [f"Model load (unpickle + validation) : {load_time * 1000:.0f} ms"]
    lines += [f"{name:<48}: {stats_ms(samples)}" for name, samples in stages.items()]
    decode = statistics.mean(stages["load_image (decode + resize)"])
    total = statistics.mean(stages[extract_stage])
    lines.append(f"Decoding is ~{decode * 1000:.0f} ms of ~{total * 1000:.0f} ms feature time ({100 * decode / total:.0f}%)")
    return lines, stages


def large_inputs() -> Dict[str, Tuple[str, bytes, str]]:
    """A 4000x5000 JPEG built from a demo image, and an A4 PDF."""
    from PIL import Image

    base = Image.open(DEMO_DIR / "genuine_payslip_01.png").convert("RGB").resize((4000, 5000))
    jpg = io.BytesIO()
    base.save(jpg, format="JPEG", quality=90)

    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=str(DEMO_DIR / "genuine_payslip_01.png"))
    return {
        "4000x5000 JPEG": ("large.jpg", jpg.getvalue(), "image/jpeg"),
        "A4 PDF (page rendered at 200 DPI)": ("payslip.pdf", doc.tobytes(), "application/pdf"),
    }


def post(client: httpx.Client, name: str, content: bytes, ctype: str) -> float:
    """POST one file to /verify and return latency; raises on non-200."""
    t = time.perf_counter()
    response = client.post("/verify", files={"file": (name, content, ctype)}, headers={"X-API-Key": API_KEY})
    response.raise_for_status()
    return time.perf_counter() - t


def concurrency_run(base_url: str, pid: int, clients: int, per_client: int, payload: bytes) -> Dict[str, float]:
    """Throughput, latency and server CPU utilisation with N concurrent clients."""
    cpu0, t0 = cpu_seconds(pid), time.perf_counter()

    def worker(_: int) -> List[float]:
        with httpx.Client(base_url=base_url, timeout=300) as c:
            return [post(c, "doc.png", payload, "image/png") for _ in range(per_client)]

    with ThreadPoolExecutor(max_workers=clients) as pool:
        latencies = [lat for batch in pool.map(worker, range(clients)) for lat in batch]
    wall = time.perf_counter() - t0
    cpu1 = cpu_seconds(pid)
    used = None if cpu0 is None or cpu1 is None else cpu1 - cpu0
    return {
        "requests": len(latencies),
        "throughput": len(latencies) / wall,
        "latency_mean": statistics.mean(latencies),
        "latency_max": max(latencies),
        "cpu_per_request": None if used is None else used / len(latencies),
        "cpu_util": None if used is None else used / wall,
    }


def memory_soak(port: int, rounds: int, extra_env: Dict[str, str], log_path: Path) -> List[Tuple[float, float]]:
    """Start a fresh server with extra_env; per round run 4 clients x 5 demo requests plus one
    4000x5000 image and record (server RSS MB, throughput req/s)."""
    payload = (DEMO_DIR / "genuine_payslip_01.png").read_bytes()
    large_name, large_bytes, large_type = large_inputs()["4000x5000 JPEG"]
    saved = {k: os.environ.get(k) for k in extra_env}
    os.environ.update(extra_env)
    try:
        proc = start_server(sys.executable, ROOT, port, API_KEY, log_path)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    samples: List[Tuple[float, float]] = []
    try:
        base_url = f"http://127.0.0.1:{port}"
        if wait_for_health(base_url, 120, proc) is None:
            return samples
        samples.append((rss_mb(proc.pid) or 0.0, 0.0))
        for _ in range(rounds):
            run = concurrency_run(base_url, proc.pid, 4, 5, payload)
            with httpx.Client(base_url=base_url, timeout=300) as client:
                post(client, large_name, large_bytes, large_type)
            samples.append((rss_mb(proc.pid) or 0.0, run["throughput"]))
    finally:
        stop_server(proc)
    return samples


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main() -> int:
    """Run all measurements and write the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8766, help="Port for the server started by this script")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per demo image")
    parser.add_argument("--soak-rounds", type=int, default=8, help="Rounds in the memory soak (0 to skip)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Report path")
    args = parser.parse_args()

    from backend import config

    rule, thin = "=" * 78, "-" * 78
    out: List[str] = [
        rule,
        "DOCUMENT FRAUD DETECTION API - PERFORMANCE REVIEW",
        rule,
        f"Generated   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Machine     : {platform.system()} {platform.machine()}, {os.cpu_count()} CPUs, Python {platform.python_version()}",
        f"Server      : uvicorn, 1 worker, MAX_CONCURRENT_ANALYSES={config.MAX_CONCURRENT_ANALYSES}",
        "",
    ]

    import_time = measure_import_time()
    stage_lines, _ = measure_stages(args.repeats)

    base_url = f"http://127.0.0.1:{args.port}"
    log_path = ROOT / "logs" / "performance_server.log"
    t_launch = time.perf_counter()
    proc = start_server(sys.executable, ROOT, args.port, API_KEY, log_path)
    try:
        ready = wait_for_health(base_url, 120, proc)
        if ready is None:
            print("server did not start; see", log_path)
            return 1
        pid = proc.pid
        idle_rss = rss_mb(pid)

        out += [
            "1. START-UP",
            thin,
            f"  Import backend.main (fresh interpreter)  : {import_time:.2f} s",
            f"  uvicorn launch -> /health 200            : {time.perf_counter() - t_launch:.2f} s "
            "(includes imports, startup validation and model load)",
            f"  Server RSS at idle after start-up        : {fmt(idle_rss, 'MB')}",
            "",
            "2. PREDICTION TIME",
            thin,
            f"  In-process, per stage ({args.repeats} repeats x 5 demo images, 850x1050 PNG):",
            *("    " + line for line in stage_lines),
            "",
        ]

        demo = {p.name: p.read_bytes() for p in sorted(DEMO_DIR.glob("*.png"))}
        with httpx.Client(base_url=base_url, timeout=300) as client:
            post(client, "warmup.png", next(iter(demo.values())), "image/png")
            cpu_before = cpu_seconds(pid)
            http_lat = [post(client, n, b, "image/png") for n, b in demo.items() for _ in range(args.repeats)]
            cpu_after = cpu_seconds(pid)
            out += [
                f"  End-to-end HTTP /verify, sequential ({len(http_lat)} requests): {stats_ms(http_lat)}",
                "",
                "3. LARGE INPUTS (through the API)",
                thin,
            ]
            for label, (name, content, ctype) in large_inputs().items():
                rss0 = rss_mb(pid)
                latency = post(client, name, content, ctype)
                out.append(
                    f"  {label:<36}: {latency:6.2f} s   upload {len(content) / 1e6:5.1f} MB   "
                    f"RSS {fmt(rss0, 'MB')} -> {fmt(rss_mb(pid), 'MB')}"
                )
            out.append("")

        single_cpu = None if cpu_before is None or cpu_after is None else (cpu_after - cpu_before) / len(http_lat)
        runs = {n: concurrency_run(base_url, pid, n, 4, demo["genuine_payslip_01.png"]) for n in (1, 2, 4)}
        final_rss, peak = rss_mb(pid), peak_rss_mb(pid)

        out += [
            "4. MEMORY (server process)",
            thin,
            f"  Idle after start-up : {fmt(idle_rss, 'MB')}",
            f"  After all workloads : {fmt(final_rss, 'MB')}",
            f"  Peak (VmHWM)        : {fmt(peak, 'MB')}",
            "",
            "5. CPU (server process)",
            thin,
            f"  CPU time per /verify (sequential demo images): {fmt(single_cpu and single_cpu * 1000, 'ms', 0)}",
            f"  {'Clients':<8}{'Requests':>9}{'Throughput':>13}{'Mean latency':>15}{'Max latency':>14}{'CPU util':>11}",
        ]
        for n, r in runs.items():
            util = "n/a" if r["cpu_util"] is None else f"{r['cpu_util'] * 100:.0f}%"
            out.append(
                f"  {n:<8}{r['requests']:>9}{r['throughput']:>10.2f}/s{r['latency_mean'] * 1000:>12.0f} ms"
                f"{r['latency_max'] * 1000:>11.0f} ms{util:>11}"
            )
        out += ["  (CPU util is server CPU seconds / wall seconds; 100% = one core fully busy.)", ""]
    finally:
        stop_server(proc)

    soak: Dict[str, List[Tuple[float, float]]] = {}
    if args.soak_rounds > 0:
        soak_log = ROOT / "logs" / "performance_server.log"
        soak["default allocator"] = memory_soak(args.port + 1, args.soak_rounds, {}, soak_log)
        soak["MALLOC_ARENA_MAX=2"] = memory_soak(args.port + 1, args.soak_rounds, {"MALLOC_ARENA_MAX": "2"}, soak_log)
        out += [
            "6. MEMORY SOAK (fresh server per configuration)",
            thin,
            "  Each round: 4 concurrent clients x 5 demo requests, then one 4000x5000 image. RSS after each round (MB):",
        ]
        for label, samples in soak.items():
            rss_values = " ".join(f"{r:.0f}" for r, _ in samples)
            tput = [t for _, t in samples[1:]]
            mean_tput = f"{statistics.mean(tput):.2f}/s" if tput else "n/a"
            out.append(f"  {label:<20}: idle->rounds  {rss_values}   (mean throughput {mean_tput})")
        out += [
            "  Interpretation: RSS that rises under concurrency and then levels off, and stays lower when",
            "  glibc arenas are capped (MALLOC_ARENA_MAX=2), is memory the allocator keeps per thread,",
            "  not an application leak. Capping arenas trades some throughput for a smaller footprint.",
            "",
        ]

    out += [
        "MACHINE-READABLE SUMMARY",
        thin,
        json.dumps(
            {
                "import_s": round(import_time, 2),
                "idle_rss_mb": idle_rss and round(idle_rss, 1),
                "peak_rss_mb": peak and round(peak, 1),
                "http_mean_ms": round(statistics.mean(http_lat) * 1000),
                "cpu_ms_per_request": single_cpu and round(single_cpu * 1000),
                "throughput_rps": {n: round(r["throughput"], 2) for n, r in runs.items()},
                "soak_rss_mb": {k: [round(r) for r, _ in v] for k, v in soak.items()},
            }
        ),
        rule,
    ]

    report = "\n".join(out) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
