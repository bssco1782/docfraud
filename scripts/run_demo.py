"""
scripts/run_demo.py
One-command live demo of the Document Fraud Detection API for presentations.

Starts the API (or uses one you already started), then shows, step by step:
  1. the health check
  2. that requests without a valid API key are refused
  3. that an unsupported file is rejected with a clear message
  4. all 5 demo documents analysed, with score, verdict, correctness, latency and the
     regions located on flagged documents

Output is plain text sized for a projector. Every number is live from the API.

Usage (from the repository root):
    python scripts/run_demo.py                          # starts its own server
    python scripts/run_demo.py --url http://127.0.0.1:8000 --api-key <key>
    python scripts/run_demo.py --pause                  # wait for Enter between steps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.e2e_test import start_server, stop_server, wait_for_health  # noqa: E402

DEMO_DIR = ROOT / "data" / "demo"
DEMO_KEY = "demo-presentation-key"


def step(title: str, pause: bool) -> None:
    """Print a step heading, optionally waiting for Enter first."""
    if pause:
        input("\n[Enter] ")
    print(f"\n=== {title} " + "=" * max(0, 66 - len(title)))


def main() -> int:
    """Run the demo sequence and return 0 if every step behaved as expected."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="Use an already running server")
    parser.add_argument("--api-key", default=DEMO_KEY, help="API key (set on the server this script starts)")
    parser.add_argument("--port", type=int, default=8000, help="Port for the server this script starts")
    parser.add_argument("--pause", action="store_true", help="Wait for Enter between steps")
    args = parser.parse_args()

    manifest = json.loads((DEMO_DIR / "demo_manifest.json").read_text(encoding="utf-8"))
    proc: Optional[object] = None
    base_url = args.url.rstrip("/") if args.url else f"http://127.0.0.1:{args.port}"
    if not args.url:
        print(f"Starting the API on {base_url} ...")
        proc = start_server(sys.executable, ROOT, args.port, args.api_key, ROOT / "logs" / "demo_server.log")
        if wait_for_health(base_url, 90, proc) is None:
            print("The server did not start. See logs/demo_server.log")
            stop_server(proc)
            return 1

    ok = True
    try:
        with httpx.Client(base_url=base_url, timeout=120) as client:
            step("1. Health check (public)", args.pause)
            health = client.get("/health")
            print(f"GET /health -> {health.status_code}")
            print(json.dumps(health.json(), indent=2))
            ok &= health.status_code == 200

            sample = DEMO_DIR / manifest["images"][0]["demo_file"]
            step("2. Security: requests without a valid key are refused", args.pause)
            for label, headers in [("no key", {}), ("wrong key", {"X-API-Key": "guess"})]:
                r = client.post(
                    "/verify", files={"file": (sample.name, sample.read_bytes(), "image/png")}, headers=headers
                )
                print(f"POST /verify ({label:<9}) -> {r.status_code}  {r.json()['error']}")
                ok &= r.status_code in (401, 403)

            step("3. Validation: unsupported files are rejected", args.pause)
            r = client.post(
                "/verify",
                files={"file": ("notes.txt", b"not a document", "text/plain")},
                headers={"X-API-Key": args.api_key},
            )
            print(f"POST /verify (notes.txt) -> {r.status_code}  {r.json()['error']}")
            ok &= r.status_code == 400

            step("4. Analysing the 5 demo documents", args.pause)
            print(f"{'Document':<25}{'Actual':<10}{'Risk':>6}  {'Verdict':<10}{'Correct':<9}{'Time':>7}")
            print("-" * 67)
            correct = 0
            for record in manifest["images"]:
                path = DEMO_DIR / record["demo_file"]
                start = time.perf_counter()
                r = client.post(
                    "/verify",
                    files={"file": (path.name, path.read_bytes(), "image/png")},
                    headers={"X-API-Key": args.api_key},
                )
                elapsed = time.perf_counter() - start
                if r.status_code != 200:
                    print(f"{path.name:<25}HTTP {r.status_code}: {r.json().get('error')}")
                    ok = False
                    continue
                body = r.json()
                verdict = "tampered" if body["tamper_risk_score"] >= manifest["risk_threshold"] else "genuine"
                hit = verdict == record["actual_label"]
                correct += hit
                print(
                    f"{path.name:<25}{record['actual_label']:<10}{body['tamper_risk_score']:>6.2f}  "
                    f"{verdict:<10}{'yes' if hit else 'NO':<9}{elapsed * 1000:>5.0f}ms"
                )
                for region in body["flagged_regions"]:
                    print(f"    -> {region['type']} region at x={region['x']}, y={region['y']} "
                          f"({region['width']} x {region['height']} px)")
            print("-" * 67)
            print(f"{correct}/5 correct on the demo set. Scores are uncalibrated; the model's measured")
            print("held-out accuracy (model v2) is 86.7% on 15 frozen test and 92.6% on 149 generated")
            print("holdout synthetic documents. See README Limitations.")
            print(f"\nExample response:\n{json.dumps(body, indent=2)}")
    finally:
        if proc is not None:
            stop_server(proc)
    print("\nDemo complete." if ok else "\nDemo finished with unexpected results (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
