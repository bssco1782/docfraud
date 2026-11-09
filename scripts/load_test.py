#!/usr/bin/env python3
"""
scripts/load_test.py
Standalone Load & Stress Testing Tool for Document Fraud Detection Backend.

Features:
- Concurrent simulated user workloads using asyncio and httpx
- In-process ASGI transport or live HTTP server targeting (http://127.0.0.1:8000)
- Measurement of Min, Max, Avg, p50, p95, p99 latencies, and throughput (req/s)
- Stress testing sweeps across 5, 10, and 20 concurrent users
- Live CPU and RAM memory utilization tracking via psutil
- Automated report generation to logs/api_load_report.txt
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config
from backend.main import app
from backend.model_utils import model_api

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class RunMetrics:
    endpoint: str
    concurrent_users: int
    requests_per_user: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies: List[float] = field(default_factory=list)
    total_duration_sec: float = 0.0
    status_codes: Dict[int, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    cpu_percent_start: float = 0.0
    cpu_percent_end: float = 0.0
    mem_rss_mb_start: float = 0.0
    mem_rss_mb_end: float = 0.0

    @property
    def min_latency(self) -> float:
        return min(self.latencies) if self.latencies else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[int(len(s) * 0.50)]

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = min(int(len(s) * 0.95), len(s) - 1)
        return s[idx]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = min(int(len(s) * 0.99), len(s) - 1)
        return s[idx]

    @property
    def throughput(self) -> float:
        return (self.total_requests / self.total_duration_sec) if self.total_duration_sec > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100.0) if self.total_requests > 0 else 0.0

    @property
    def failure_rate(self) -> float:
        return (self.failed_requests / self.total_requests * 100.0) if self.total_requests > 0 else 0.0


def get_system_telemetry() -> Tuple[float, float]:
    """Return current (cpu_percent, memory_rss_mb)."""
    if not HAS_PSUTIL:
        return 0.0, 0.0
    try:
        proc = psutil.Process()
        cpu = psutil.cpu_percent(interval=None)
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        return cpu, mem_mb
    except Exception:
        return 0.0, 0.0


def resolve_test_file(custom_path: Optional[str] = None) -> Tuple[str, bytes]:
    """Locate and read document file bytes for verification testing."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p.name, p.read_bytes()
        print(f"[Warning] Custom file '{custom_path}' not found, falling back to default fixtures.")

    candidates = [
        PROJECT_ROOT / "tests" / "fixtures" / "clean_rgb.png",
        PROJECT_ROOT / "data" / "synthetic" / "genuine" / "payslip_001.png",
        PROJECT_ROOT / "tests" / "fixtures" / "clean_rgb.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c.name, c.read_bytes()

    # Fallback 1x1 PNG bytes
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return "sample_document.png", minimal_png


async def _worker_loop(
    client: httpx.AsyncClient,
    endpoint: str,
    requests_to_make: int,
    filename: str,
    file_bytes: bytes,
    headers: Dict[str, str],
) -> List[Tuple[int, float, Optional[str]]]:
    """Simulates a user executing sequential requests."""
    results: List[Tuple[int, float, Optional[str]]] = []

    for _ in range(requests_to_make):
        t0 = time.perf_counter()
        status_code = 0
        err = None
        try:
            if endpoint == "/health":
                resp = await client.get("/health")
                status_code = resp.status_code
                if status_code != 200:
                    err = f"Status {status_code}: {resp.text}"
            elif endpoint == "/verify":
                content_type = "image/png" if filename.lower().endswith(".png") else "image/jpeg"
                files = {"file": (filename, io.BytesIO(file_bytes), content_type)}
                resp = await client.post("/verify", files=files, headers=headers)
                status_code = resp.status_code
                if status_code != 200:
                    err = f"Status {status_code}: {resp.text}"
                else:
                    payload = resp.json()
                    if "tamper_risk_score" not in payload or "confidence" not in payload:
                        err = f"Malformed payload schema: {payload}"
            else:
                raise ValueError(f"Unsupported endpoint: {endpoint}")
        except Exception as exc:
            status_code = 599
            err = str(exc)
        finally:
            elapsed = time.perf_counter() - t0
            results.append((status_code, elapsed, err))

    return results


async def run_load_benchmark(
    base_url: str,
    endpoint: str,
    concurrent_users: int = 5,
    requests_per_user: int = 10,
    test_filename: str = "doc.png",
    test_file_bytes: bytes = b"",
    api_key: str = "",
    use_asgi: bool = False,
) -> RunMetrics:
    """Execute concurrent load test run."""
    headers = {"X-API-Key": api_key or config.API_KEY}

    # Preload model in memory
    if not model_api.is_loaded:
        model_api.load_model()

    cpu_start, mem_start = get_system_telemetry()

    transport = httpx.ASGITransport(app=app) if use_asgi else None
    client_kwargs: Dict[str, Any] = {"base_url": base_url, "timeout": 120.0}
    if transport:
        client_kwargs["transport"] = transport

    async with httpx.AsyncClient(**client_kwargs) as client:
        start_time = time.perf_counter()
        tasks = [
            _worker_loop(
                client=client,
                endpoint=endpoint,
                requests_to_make=requests_per_user,
                filename=test_filename,
                file_bytes=test_file_bytes,
                headers=headers,
            )
            for _ in range(concurrent_users)
        ]
        user_batches = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    cpu_end, mem_end = get_system_telemetry()

    flat_results = [r for batch in user_batches for r in batch]
    total_reqs = len(flat_results)
    latencies = [r[1] for r in flat_results]
    status_codes: Dict[int, int] = {}
    errors: List[str] = []
    success_count = 0
    failure_count = 0

    for code, _, err in flat_results:
        status_codes[code] = status_codes.get(code, 0) + 1
        if code == 200:
            success_count += 1
        else:
            failure_count += 1
            if err:
                errors.append(err)

    return RunMetrics(
        endpoint=endpoint,
        concurrent_users=concurrent_users,
        requests_per_user=requests_per_user,
        total_requests=total_reqs,
        successful_requests=success_count,
        failed_requests=failure_count,
        latencies=latencies,
        total_duration_sec=total_duration,
        status_codes=status_codes,
        errors=errors,
        cpu_percent_start=cpu_start,
        cpu_percent_end=cpu_end,
        mem_rss_mb_start=mem_start,
        mem_rss_mb_end=mem_end,
    )


def format_cli_summary(metrics: RunMetrics) -> str:
    """Format single run metrics as terminal text."""
    lines = [
        "--------------------------------------------------------------------------------",
        f"ENDPOINT: {metrics.endpoint} | CONCURRENT USERS: {metrics.concurrent_users} | REQS/USER: {metrics.requests_per_user}",
        "--------------------------------------------------------------------------------",
        f"  Total Requests:       {metrics.total_requests}",
        f"  Successful Requests:  {metrics.successful_requests} ({metrics.success_rate:.2f}%)",
        f"  Failed Requests:      {metrics.failed_requests} ({metrics.failure_rate:.2f}%)",
        f"  Total Duration:       {metrics.total_duration_sec:.2f} s",
        f"  Throughput:           {metrics.throughput:.2f} req/s",
        f"  Average Latency:      {metrics.avg_latency * 1000:.2f} ms ({metrics.avg_latency:.4f} s)",
        f"  Min Latency:          {metrics.min_latency * 1000:.2f} ms ({metrics.min_latency:.4f} s)",
        f"  p50 (Median) Latency: {metrics.p50_latency * 1000:.2f} ms ({metrics.p50_latency:.4f} s)",
        f"  p95 Latency:          {metrics.p95_latency * 1000:.2f} ms ({metrics.p95_latency:.4f} s)",
        f"  p99 Latency:          {metrics.p99_latency * 1000:.2f} ms ({metrics.p99_latency:.4f} s)",
        f"  Max Latency:          {metrics.max_latency * 1000:.2f} ms ({metrics.max_latency:.4f} s)",
        f"  Status Code Breakdown:{metrics.status_codes}",
        f"  Memory RSS:           {metrics.mem_rss_mb_start:.1f} MB -> {metrics.mem_rss_mb_end:.1f} MB (Delta: +{metrics.mem_rss_mb_end - metrics.mem_rss_mb_start:.2f} MB)",
    ]
    return "\n".join(lines)


# Acceptance targets from the Day 5 sprint plan
TARGET_SUCCESS_RATE = 95.0  # percent
TARGET_VERIFY_AVG_S = 2.0
TARGET_HEALTH_AVG_S = 0.050


def verdict(metrics: RunMetrics, target_avg_s: float) -> str:
    """PASS/FAIL against the success-rate and average-latency targets, with the reason."""
    ok_rate = metrics.success_rate > TARGET_SUCCESS_RATE
    ok_latency = metrics.avg_latency < target_avg_s
    if ok_rate and ok_latency:
        return "PASS (meets success-rate and average-latency targets)"
    reasons = []
    if not ok_rate:
        reasons.append(f"success {metrics.success_rate:.1f}% <= {TARGET_SUCCESS_RATE:.0f}%")
    if not ok_latency:
        reasons.append(f"avg {metrics.avg_latency:.3f}s >= {target_avg_s:.3f}s")
    return "FAIL (" + "; ".join(reasons) + ")"


def generate_full_report_text(
    health_metrics: Optional[RunMetrics],
    verify_metrics: Optional[RunMetrics],
    stress_metrics: List[RunMetrics],
    target_info: str,
) -> str:
    """Generate comprehensive textual performance report matching all Day 5 requirements."""
    report_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    text = f"""================================================================================
           DOCUMENT FRAUD DETECTION API - PERFORMANCE & LOAD TEST REPORT
================================================================================
Generated At:        {report_timestamp}
Environment:         {target_info}
Platform:            Python {sys.version.split()[0]} on {sys.platform}
Model Architecture:  RandomForestClassifier (100 estimators, 22 forensic features)
Event Loop Model:    FastAPI + asyncio + threadpool offloading (asyncio.to_thread)
================================================================================

1. TEST CONFIGURATION
--------------------------------------------------------------------------------
- Standard Concurrency Test:
  * Simulated Users:       5 concurrent users
  * Requests per User:     10 sequential requests per user
  * Total Requests:        50 requests per endpoint
  * Endpoints Evaluated:   GET /health, POST /verify

- Stress Testing Matrix:
  * Tiers Evaluated:       5, 10, and 20 concurrent users
  * Workload Profile:      High-throughput simultaneous forensic image uploads
  * Payload Type:          Authentic RGB Document Image (PNG/JPEG)
  * Target Success Rate:   > 95%
  * Target Average Latency:< 2.000 seconds for /verify, < 50ms for /health

================================================================================
2. PERFORMANCE METRICS SUMMARY (5 CONCURRENT USERS, 50 REQUESTS)
================================================================================
"""

    if health_metrics:
        text += f"""[GET /health - System Readiness Check]
- Status:                {verdict(health_metrics, TARGET_HEALTH_AVG_S)}
- Total Requests:        {health_metrics.total_requests}
- Success Rate:          {health_metrics.success_rate:.2f}% ({health_metrics.successful_requests}/{health_metrics.total_requests})
- Failure Rate:          {health_metrics.failure_rate:.2f}% ({health_metrics.failed_requests} failed)
- Total Test Duration:   {health_metrics.total_duration_sec:.3f} s
- Throughput:            {health_metrics.throughput:.2f} req/sec
- Average Response Time: {health_metrics.avg_latency * 1000:.2f} ms ({health_metrics.avg_latency:.4f} s)
- Min Response Time:     {health_metrics.min_latency * 1000:.2f} ms ({health_metrics.min_latency:.4f} s)
- Median (p50):          {health_metrics.p50_latency * 1000:.2f} ms ({health_metrics.p50_latency:.4f} s)
- 95th Percentile (p95): {health_metrics.p95_latency * 1000:.2f} ms ({health_metrics.p95_latency:.4f} s)
- 99th Percentile (p99): {health_metrics.p99_latency * 1000:.2f} ms ({health_metrics.p99_latency:.4f} s)
- Max Response Time:     {health_metrics.max_latency * 1000:.2f} ms ({health_metrics.max_latency:.4f} s)
- Tester process RSS:    {health_metrics.mem_rss_mb_start:.1f} MB -> {health_metrics.mem_rss_mb_end:.1f} MB (server memory only when --in-process; 0.0 = psutil not installed)

"""

    if verify_metrics:
        text += f"""[POST /verify - Forensic Document Verification]
- Status:                {verdict(verify_metrics, TARGET_VERIFY_AVG_S)}
- Total Requests:        {verify_metrics.total_requests}
- Success Rate:          {verify_metrics.success_rate:.2f}% ({verify_metrics.successful_requests}/{verify_metrics.total_requests})
- Failure Rate:          {verify_metrics.failure_rate:.2f}% ({verify_metrics.failed_requests} failed)
- Total Test Duration:   {verify_metrics.total_duration_sec:.3f} s
- Throughput:            {verify_metrics.throughput:.2f} req/sec
- Average Response Time: {verify_metrics.avg_latency:.3f} s ({verify_metrics.avg_latency * 1000:.2f} ms)
- Min Response Time:     {verify_metrics.min_latency:.3f} s ({verify_metrics.min_latency * 1000:.2f} ms)
- Median (p50):          {verify_metrics.p50_latency:.3f} s ({verify_metrics.p50_latency * 1000:.2f} ms)
- 95th Percentile (p95): {verify_metrics.p95_latency:.3f} s ({verify_metrics.p95_latency * 1000:.2f} ms)
- 99th Percentile (p99): {verify_metrics.p99_latency:.3f} s ({verify_metrics.p99_latency * 1000:.2f} ms)
- Max Response Time:     {verify_metrics.max_latency:.3f} s ({verify_metrics.max_latency * 1000:.2f} ms)
- Tester process RSS:    {verify_metrics.mem_rss_mb_start:.1f} MB -> {verify_metrics.mem_rss_mb_end:.1f} MB (server memory only when --in-process; 0.0 = psutil not installed)
- Concurrent uploads:    {verify_metrics.successful_requests} succeeded, {verify_metrics.failed_requests} failed, status codes {dict(sorted(verify_metrics.status_codes.items()))}

"""

    text += """================================================================================
3. STRESS TESTING RESULTS (SCALABILITY ACROSS CONCURRENCY TIERS)
================================================================================
| Concurrent Users | Total Reqs | Success Rate | Throughput (req/s) | Avg Latency | p95 Latency | Max Latency |
|------------------|------------|--------------|---------------------|-------------|-------------|-------------|
"""
    for sm in stress_metrics:
        text += (
            f"| {sm.concurrent_users:<16} | {sm.total_requests:<10} | {sm.success_rate:>11.1f}% | "
            f"{sm.throughput:>17.2f}  | {sm.avg_latency:>10.3f}s | {sm.p95_latency:>10.3f}s | {sm.max_latency:>10.3f}s |\n"
        )

    if stress_metrics:
        text += "\nObservations (computed from the table above):\n"
        for sm in stress_metrics:
            meets = "meets" if sm.avg_latency < TARGET_VERIFY_AVG_S and sm.success_rate > TARGET_SUCCESS_RATE else "misses"
            text += (
                f"- {sm.concurrent_users} users: avg {sm.avg_latency:.2f}s, p95 {sm.p95_latency:.2f}s, "
                f"{sm.throughput:.2f} req/s, {sm.success_rate:.1f}% success -> {meets} the "
                f"< {TARGET_VERIFY_AVG_S:.1f}s / > {TARGET_SUCCESS_RATE:.0f}% target\n"
            )
        peak = max(sm.throughput for sm in stress_metrics)
        text += (
            f"- Throughput stays near {peak:.1f} req/s at every tier: analysis is CPU-bound, so extra users\n"
            "  queue and latency grows roughly in proportion to the number of concurrent users.\n"
        )

    text += """
================================================================================
4. NOTES
================================================================================
- Feature extraction (ELA, edge and noise statistics) is CPU-bound; one uvicorn process
  runs at most MAX_CONCURRENT_ANALYSES analyses at once (default 2) and queues the rest.
- /health does not wait for analyses, so it stays fast while /verify is busy.
- For server memory and CPU per request use scripts/performance_review.py.
- More throughput needs more processes (uvicorn --workers N) or machines; not measured.

================================================================================
5. ACCEPTANCE CRITERIA (5 CONCURRENT USERS)
================================================================================
"""
    if verify_metrics:
        server_errors = sum(n for code, n in verify_metrics.status_codes.items() if code >= 500)
        checks = [
            ("No failed /verify requests", verify_metrics.failed_requests == 0,
             f"{verify_metrics.failed_requests} failed"),
            ("No 5xx server errors", server_errors == 0, f"{server_errors} server errors"),
            (f"Success rate > {TARGET_SUCCESS_RATE:.0f}%", verify_metrics.success_rate > TARGET_SUCCESS_RATE,
             f"{verify_metrics.success_rate:.1f}%"),
            (f"/verify average < {TARGET_VERIFY_AVG_S:.1f} s", verify_metrics.avg_latency < TARGET_VERIFY_AVG_S,
             f"{verify_metrics.avg_latency:.3f} s"),
        ]
        if health_metrics:
            checks.append((f"/health average < {TARGET_HEALTH_AVG_S * 1000:.0f} ms",
                           health_metrics.avg_latency < TARGET_HEALTH_AVG_S,
                           f"{health_metrics.avg_latency * 1000:.1f} ms"))
        for label, ok, value in checks:
            text += f"[{'x' if ok else ' '}] {label:<32} {'YES' if ok else 'NO '} ({value})\n"
    text += "================================================================================\n"
    return text


async def main_async():
    parser = argparse.ArgumentParser(description="Document Fraud Detection API Load & Stress Tester")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL of API (default: http://127.0.0.1:8000)")
    parser.add_argument("-u", "--users", type=int, default=5, help="Number of concurrent users (default: 5)")
    parser.add_argument("-r", "--requests", type=int, default=10, help="Requests per user (default: 10)")
    parser.add_argument("-e", "--endpoint", choices=["health", "verify", "both"], default="both", help="Endpoint to test")
    parser.add_argument("-f", "--file", default=None, help="Custom document image path for /verify")
    parser.add_argument("-k", "--api-key", default="", help="Custom API key (default: from config)")
    parser.add_argument("--stress", action="store_true", help="Run multi-tier stress tests (5, 10, 20 users)")
    parser.add_argument("-o", "--output", default="logs/api_load_report.txt", help="Report output path")
    parser.add_argument("--in-process", action="store_true", help="Force in-process ASGI testing instead of network")

    args = parser.parse_args()

    use_asgi = args.in_process
    target_url = args.url

    # Test network connectivity if not explicitly in-process
    if not use_asgi:
        try:
            async with httpx.AsyncClient(timeout=2.0) as check_client:
                r = await check_client.get(f"{target_url.rstrip('/')}/health")
                if r.status_code == 200:
                    print(f"Connected to live server at {target_url}")
                else:
                    print(f"Server at {target_url} returned {r.status_code}. Using in-process ASGI transport.")
                    use_asgi = True
        except Exception:
            print(f"No running server detected at {target_url}. Running in-process ASGI test mode.")
            use_asgi = True

    target_info = "In-Process ASGI (httpx.ASGITransport)" if use_asgi else f"Live HTTP Server ({target_url})"
    test_fname, test_fbytes = resolve_test_file(args.file)
    print(f"Test File: {test_fname} ({len(test_fbytes):,} bytes)")
    print(f"Target:    {target_info}")

    health_metrics: Optional[RunMetrics] = None
    verify_metrics: Optional[RunMetrics] = None
    stress_metrics: List[RunMetrics] = []

    # 1. Standard Concurrency Test (5 users x 10 reqs = 50 reqs)
    if args.endpoint in ("health", "both"):
        print(f"\n[Running Concurrency Test: GET /health ({args.users} users, {args.requests} reqs/user)]...")
        health_metrics = await run_load_benchmark(
            base_url=target_url,
            endpoint="/health",
            concurrent_users=args.users,
            requests_per_user=args.requests,
            use_asgi=use_asgi,
        )
        print(format_cli_summary(health_metrics))

    if args.endpoint in ("verify", "both"):
        print(f"\n[Running Concurrency Test: POST /verify ({args.users} users, {args.requests} reqs/user)]...")
        verify_metrics = await run_load_benchmark(
            base_url=target_url,
            endpoint="/verify",
            concurrent_users=args.users,
            requests_per_user=args.requests,
            test_filename=test_fname,
            test_file_bytes=test_fbytes,
            api_key=args.api_key,
            use_asgi=use_asgi,
        )
        print(format_cli_summary(verify_metrics))

    # 2. Stress Testing Tiers (5, 10, 20 users)
    if args.stress or args.endpoint == "both":
        print("\n================================================================================")
        print("RUNNING STRESS TESTING TIERS (5, 10, 20 CONCURRENT USERS)")
        print("================================================================================")
        tiers = [5, 10, 20]
        for tier_users in tiers:
            print(f"--> Executing Stress Tier: {tier_users} concurrent users (2 requests each = {tier_users * 2} reqs)...")
            tier_m = await run_load_benchmark(
                base_url=target_url,
                endpoint="/verify",
                concurrent_users=tier_users,
                requests_per_user=2,
                test_filename=test_fname,
                test_file_bytes=test_fbytes,
                api_key=args.api_key,
                use_asgi=use_asgi,
            )
            stress_metrics.append(tier_m)
            print(f"    Avg: {tier_m.avg_latency:.3f}s | p95: {tier_m.p95_latency:.3f}s | "
                  f"Throughput: {tier_m.throughput:.2f} req/s | Success: {tier_m.success_rate:.1f}%")

    # 3. Generate Report File
    report_text = generate_full_report_text(
        health_metrics=health_metrics,
        verify_metrics=verify_metrics,
        stress_metrics=stress_metrics,
        target_info=target_info,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\n[Success] Comprehensive performance report saved to: {out_path.resolve()}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
