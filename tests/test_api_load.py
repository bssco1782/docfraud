"""
tests/test_api_load.py
Load and concurrency testing suite for Document Fraud Detection Backend.

Verifies API stability, response latencies, throughput, and error rates under
concurrent user workloads using asyncio and httpx.ASGITransport.
"""

from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import httpx

from backend import config
from backend.main import app
from backend.model_utils import model_api


@dataclass
class LoadMetrics:
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
        sorted_l = sorted(self.latencies)
        return sorted_l[int(len(sorted_l) * 0.50)]

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = min(int(len(sorted_l) * 0.95), len(sorted_l) - 1)
        return sorted_l[idx]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = min(int(len(sorted_l) * 0.99), len(sorted_l) - 1)
        return sorted_l[idx]

    @property
    def throughput(self) -> float:
        if self.total_duration_sec > 0:
            return self.total_requests / self.total_duration_sec
        return 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests > 0:
            return (self.successful_requests / self.total_requests) * 100.0
        return 0.0

    @property
    def failure_rate(self) -> float:
        if self.total_requests > 0:
            return (self.failed_requests / self.total_requests) * 100.0
        return 0.0


def _get_test_image_bytes() -> tuple[str, bytes]:
    """Retrieve sample document image bytes for load testing."""
    import os
    env_img = os.environ.get("LOAD_TEST_IMAGE")
    if env_img and Path(env_img).exists():
        p = Path(env_img)
        return p.name, p.read_bytes()

    fixture_path = Path("tests/fixtures/clean_rgb.png")
    if fixture_path.exists():
        return fixture_path.name, fixture_path.read_bytes()

    synthetic_path = Path("data/synthetic/genuine/payslip_001.png")
    if synthetic_path.exists():
        return synthetic_path.name, synthetic_path.read_bytes()

    # Fallback minimal PNG byte sequence
    minimal_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return "minimal_test.png", minimal_png


async def _simulate_user_requests(
    client: httpx.AsyncClient,
    endpoint: str,
    requests_count: int,
    filename: str,
    file_bytes: bytes,
    headers: Dict[str, str],
) -> List[tuple[int, float, Optional[str]]]:
    """Simulate a single user issuing a sequence of requests."""
    user_results: List[tuple[int, float, Optional[str]]] = []

    for _ in range(requests_count):
        start_time = time.perf_counter()
        error_msg = None
        status_code = 0
        try:
            if endpoint == "/health":
                response = await client.get("/health")
                status_code = response.status_code
                if status_code != 200:
                    error_msg = f"Unexpected status {status_code}: {response.text}"
            elif endpoint == "/verify":
                files = {"file": (filename, io.BytesIO(file_bytes), "image/png")}
                response = await client.post("/verify", files=files, headers=headers)
                status_code = response.status_code
                if status_code != 200:
                    error_msg = f"Unexpected status {status_code}: {response.text}"
                else:
                    data = response.json()
                    if "tamper_risk_score" not in data or "confidence" not in data:
                        error_msg = f"Invalid schema returned: {data}"
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")
        except Exception as exc:
            error_msg = str(exc)
            status_code = 599  # Client-side synthetic failure
        finally:
            elapsed = time.perf_counter() - start_time
            user_results.append((status_code, elapsed, error_msg))

    return user_results


async def execute_concurrent_load_test(
    endpoint: str,
    concurrent_users: int = 5,
    requests_per_user: int = 10,
) -> LoadMetrics:
    """
    Run concurrent load test against FastAPI app using in-process ASGI transport.
    """
    # Ensure model is loaded in memory prior to starting test
    if not model_api.is_loaded:
        model_api.load_model()

    filename, file_bytes = _get_test_image_bytes()
    headers = {"X-API-Key": config.API_KEY}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=60.0) as client:
        start_total = time.perf_counter()

        tasks = [
            _simulate_user_requests(
                client=client,
                endpoint=endpoint,
                requests_count=requests_per_user,
                filename=filename,
                file_bytes=file_bytes,
                headers=headers,
            )
            for _ in range(concurrent_users)
        ]

        nested_results = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_total

    # Aggregate metrics
    all_results = [r for user_res in nested_results for r in user_res]
    total_reqs = len(all_results)
    latencies = [r[1] for r in all_results]
    status_codes: Dict[int, int] = {}
    errors: List[str] = []
    success_count = 0
    failure_count = 0

    for status_code, _, err in all_results:
        status_codes[status_code] = status_codes.get(status_code, 0) + 1
        if status_code == 200:
            success_count += 1
        else:
            failure_count += 1
            if err:
                errors.append(err)

    return LoadMetrics(
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
    )


# =========================================================================
# Unit & Integration Load Tests
# =========================================================================

def test_health_concurrent_load_5_users():
    """
    Test /health under 5 concurrent users × 10 requests = 50 requests.
    Validates stability, response latency, and 100% success rate.
    """
    metrics = asyncio.run(
        execute_concurrent_load_test(
            endpoint="/health",
            concurrent_users=5,
            requests_per_user=10,
        )
    )

    print(f"\n[Load Test: /health] Users: {metrics.concurrent_users}, Total: {metrics.total_requests}")
    print(f"  Success Rate: {metrics.success_rate:.1f}%")
    print(f"  Avg Latency:  {metrics.avg_latency * 1000:.2f} ms")
    print(f"  Min Latency:  {metrics.min_latency * 1000:.2f} ms")
    print(f"  Max Latency:  {metrics.max_latency * 1000:.2f} ms")
    print(f"  Throughput:   {metrics.throughput:.2f} req/s")

    assert metrics.total_requests == 50
    assert metrics.failed_requests == 0
    assert metrics.success_rate == 100.0
    assert metrics.avg_latency < 0.1, f"Avg latency too high: {metrics.avg_latency:.3f}s"
    assert metrics.max_latency < 0.5, f"Max latency too high: {metrics.max_latency:.3f}s"


def test_verify_concurrent_load_5_users_acceptance_criteria():
    """
    Day 5 Acceptance Criteria Test:
    - 5 concurrent users
    - 10 requests each = 50 requests total
    - Target:
        * API remains stable under 5 concurrent users
        * No crashes (failed_requests == 0)
        * Success rate > 95%
        * Average response time < 2 seconds
        * Proper handling of simultaneous uploads
    """
    metrics = asyncio.run(
        execute_concurrent_load_test(
            endpoint="/verify",
            concurrent_users=5,
            requests_per_user=10,
        )
    )

    print(f"\n[Load Test: /verify] Users: {metrics.concurrent_users}, Total: {metrics.total_requests}")
    print(f"  Success Rate: {metrics.success_rate:.1f}% ({metrics.successful_requests}/{metrics.total_requests})")
    print(f"  Failure Rate: {metrics.failure_rate:.1f}%")
    print(f"  Avg Latency:  {metrics.avg_latency:.3f} s")
    print(f"  Min Latency:  {metrics.min_latency:.3f} s")
    print(f"  Max Latency:  {metrics.max_latency:.3f} s")
    print(f"  p50 Latency:  {metrics.p50_latency:.3f} s")
    print(f"  p95 Latency:  {metrics.p95_latency:.3f} s")
    print(f"  Throughput:   {metrics.throughput:.2f} req/s")
    print(f"  Total Time:   {metrics.total_duration_sec:.2f} s")

    # Day 5 Acceptance Criteria Assertions
    assert metrics.total_requests == 50
    assert metrics.failed_requests == 0, f"Encountered request failures: {metrics.errors[:3]}"
    assert metrics.success_rate > 95.0, f"Success rate {metrics.success_rate}% is <= 95%"
    assert metrics.avg_latency < 2.0, f"Average response time {metrics.avg_latency:.3f}s exceeds 2.0s limit"


def test_simultaneous_distinct_uploads_handling():
    """
    Verify simultaneous uploads of distinct document files produce correct,
    isolated analysis results without cross-talk or race conditions.
    """
    async def run_distinct_uploads():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30.0) as client:
            headers = {"X-API-Key": config.API_KEY}
            fixtures_dir = Path("tests/fixtures")
            test_files = [
                fixtures_dir / "clean_rgb.png",
                fixtures_dir / "clean_rgb.jpg",
                fixtures_dir / "grayscale.jpg",
                fixtures_dir / "small_image.jpg",
                fixtures_dir / "tampered_patch.jpg",
            ]

            async def upload_one(p: Path):
                data = p.read_bytes()
                files = {"file": (p.name, io.BytesIO(data), "image/jpeg" if p.suffix == ".jpg" else "image/png")}
                return await client.post("/verify", files=files, headers=headers)

            responses = await asyncio.gather(*[upload_one(p) for p in test_files])
            return responses

    responses = asyncio.run(run_distinct_uploads())
    assert len(responses) == 5
    for resp in responses:
        assert resp.status_code == 200
        body = resp.json()
        assert "tamper_risk_score" in body
        assert "confidence" in body
        assert "interpretation" in body
        assert 0.0 <= body["tamper_risk_score"] <= 1.0


def test_stress_concurrency_tiers():
    """
    Stress testing evaluation across 5, 10, and 20 concurrent users.
    Identifies scalability trends, throughput inflection points, and verifies stability.
    """
    tier_results = []
    concurrency_tiers = [5, 10, 20]
    requests_per_user = 2  # Keep total fast for pytest regression while testing scaling

    for users in concurrency_tiers:
        metrics = asyncio.run(
            execute_concurrent_load_test(
                endpoint="/verify",
                concurrent_users=users,
                requests_per_user=requests_per_user,
            )
        )
        tier_results.append(metrics)
        print(f"\n[Stress Tier: {users} users, {metrics.total_requests} reqs]: "
              f"Avg: {metrics.avg_latency:.3f}s, Throughput: {metrics.throughput:.2f} req/s, "
              f"Success: {metrics.success_rate:.1f}%")

        assert metrics.success_rate >= 95.0
        assert metrics.failed_requests == 0
