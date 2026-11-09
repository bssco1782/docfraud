# DocFraud — Document Fraud Detection Backend

A FastAPI service that checks document images (payslips, bank statements and ID proofs)
for signs of digital tampering. Upload an image or PDF; the service measures whether any
part of the page behaves differently from the rest of the same page (noise, colour, sharpness,
ink), feeds 29 local tamper features to a scikit-learn Random Forest (model v2), and returns
a tamper-risk score with a plain-language interpretation.

> **Prototype — not production-ready.** Model v2 reaches **86.7% (13/15)** on the frozen test
> set and **92.6% (138/149)** on a fresh held-out set, but it has only been trained and
> evaluated on **synthetic** documents; accuracy on real documents is unknown. See
> [Current status](#current-status) and [Limitations](#limitations) before using any result
> for a real decision.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Environment setup](#environment-setup)
- [Running the API](#running-the-api)
- [API endpoints](#api-endpoints)
- [Example requests](#example-requests)
- [Example responses](#example-responses)
- [Testing](#testing)
- [Load testing](#load-testing)
- [Project structure](#project-structure)
- [Documentation](#documentation)
- [Current status](#current-status)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)

---

## Overview

Edited documents often carry traces of the edit: regions that were re-saved at a different
JPEG quality, pasted text with different noise, or blurred and re-coloured patches. DocFraud
measures these traces over the whole image and asks a trained classifier whether the
document looks tampered.

The repository contains:

- the HTTP API (`backend/`), protected by an API key;
- the analysis engine (`core/`) that turns an image into 29 local tamper features;
- a 75-document **synthetic** labelled dataset with its generator and validation tests, plus
  two generated sets (150 development documents, 149 held-out documents);
- the trained model (`data/models/fraud_model.pkl`, v2; v1 kept as `fraud_model_v1.pkl`);
- offline tools for training, validation, load testing and report generation (`scripts/`);
- 250+ automated tests (`tests/`).

---

## Features

- **Document fraud detection** — `POST /verify` returns a tamper-risk score, a confidence
  value and a human-readable interpretation for one uploaded document.
- **Local tamper analysis** — sizes of the largest regions where noise, paper colour,
  sharpness or ink darkness differ from the rest of the page, plus a page-level processing
  fingerprint (29 features, schema `local-tamper-v2`, listed in [ARCHITECTURE.md](ARCHITECTURE.md)).
- **Random Forest classification** — 300-tree scikit-learn classifier with balanced class
  weights, trained on 210 synthetic documents plus re-saved / resized copies of each.
- **FastAPI backend** — async endpoints, model inference offloaded to a worker thread,
  interactive OpenAPI docs at `/docs` and `/redoc`.
- **API key authentication** — `X-API-Key` header required on `/verify`; `/health` and the
  docs are public.
- **Input validation and uniform errors** — extension, empty-file and size checks;
  every error uses the same JSON envelope (`success`, `error`, `code`, `detail`).
- **Load testing support** — `scripts/load_test.py` for concurrent-user and stress runs
  against a live server or in-process.
- **Supported inputs** — `.jpg`, `.jpeg`, `.png`, `.pdf` (first page only), up to 10 MB and
  40 megapixels.
- **Hardened for deployment** — uploads limited before parsing, decompression-bomb and
  tiny-image guards, readiness-aware `/health` (503 when the model is missing),
  request IDs in every log line, no internal details in error responses.

---

## Architecture

```
 Client ──HTTP──▶ API-key middleware ──▶ POST /verify ──▶ file validation ──▶ ModelAPI.predict()
                  (backend/main.py)      (backend/main.py) (error_handlers.py)  (model_utils.py)
                                                                                     │
                   core/tamper_features.py (schema from the model file)             ▼
                   load image → 29 local features (+ anomalous regions) ◀── temporary file
                                                          │
                                                          ▼
                              data/models/fraud_model.pkl (RandomForestClassifier)
                                                          │
                                                          ▼
                     { tamper_risk_score, confidence, flagged_regions, interpretation }
```

| Layer | Location | Responsibility |
|---|---|---|
| HTTP service | `backend/` | Routing, authentication, validation, error handling, configuration |
| Analysis engine | `core/` | Image loading, feature extraction (per model schema), model loading |
| Offline tooling | `scripts/` | Dataset generation, training, validation, load tests, reports |

Full detail — request flow, the 29 features, model loading and response fields — is in
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick start

```bash
git clone https://github.com/bssco178-a11y/docfraud-backend.git
cd docfraud-backend
pip install -r requirements.txt

export API_KEY="my-local-secret"          # Windows PowerShell: $env:API_KEY="my-local-secret"
uvicorn backend.main:app --reload

curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/verify \
  -H "X-API-Key: my-local-secret" \
  -F "file=@data/demo/genuine_payslip_01.png"
```

Then open <http://127.0.0.1:8000/docs> for the interactive API explorer, or run the
whole live demo with `python scripts/run_demo.py` (see [DEMO_GUIDE.md](DEMO_GUIDE.md)).

---

## Installation

**Requirements:** Python 3.11 or newer (verified on 3.11 Linux and 3.13 Windows), `pip`, and about
500 MB of disk for dependencies. Run every command from the **repository root**.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs FastAPI, uvicorn, scikit-learn, OpenCV, Pillow, PyMuPDF (PDF
support), pytest and pytest-cov. The trained model `data/models/fraud_model.pkl` is
committed to the repository; no download step is needed.

---

## Environment setup

All settings live in `backend/config.py` and are read from environment variables. Copy the
template and edit it:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `API_KEY` | `default-api-key` | Value clients must send in `X-API-Key`. **Change it** — the default is public. |
| `MODEL_PATH` | `data/models/fraud_model.pkl` | Model file; falls back to the default if the path does not exist |
| `MAX_FILE_SIZE` | `10485760` (10 MB) | Upload limit in bytes; larger uploads get `413` |
| `TAMPER_THRESHOLD` | `0.5` | Risk score at or above which the interpretation reports tampering |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins; restrict before exposing the API |
| `MAX_IMAGE_PIXELS` | `40000000` | Largest image (or PDF page at 200 DPI) accepted, in pixels |
| `MAX_CONCURRENT_ANALYSES` | `min(2, CPUs)` | Documents analysed at once per process; others wait |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

`.env` is git-ignored — never commit real keys. Variables already set in the shell take
precedence over `.env`.

---

## Running the API

```bash
# Development (auto-reload on code changes)
uvicorn backend.main:app --reload

# Normal run
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

A successful start prints:

```
INFO:     Waiting for application startup.
Model loaded: .../data/models/fraud_model.pkl
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

| URL | What it is |
|---|---|
| <http://127.0.0.1:8000/docs> | Swagger UI (try requests in the browser) |
| <http://127.0.0.1:8000/redoc> | ReDoc reference |
| <http://127.0.0.1:8000/openapi.json> | OpenAPI schema |

Binding to `127.0.0.1` accepts local connections only. Deployment options and their
limitations are in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## API endpoints

Full reference: [API_DOCS.md](API_DOCS.md).

### GET /health

Public readiness check. No API key needed.

| Status | Body |
|---|---|
| `200` | Model loaded: `{"status": "ok", "version": "0.1.0", "model_loaded": true, "model_version": "2.0.0", "uptime_seconds": …, "detail": null}` |
| `503` | Model missing or failed to load: `"status": "degraded"`, `"model_loaded": false`, `"detail"` gives the reason |

Every response carries an `X-Request-ID` header (the client's own value if it sends one),
which also appears in the server log lines for that request.

### POST /verify

Analyses one uploaded document. Requires `X-API-Key`.

| Part | Name | Required | Description |
|---|---|---|---|
| Header | `X-API-Key` | Yes | Must equal the server's `API_KEY` |
| Form data | `file` | Yes | `.jpg`, `.jpeg`, `.png` or `.pdf`, 1 byte – 10 MB, at most 40 megapixels, shortest side ≥ 16 px |
| Query | `threshold` | No | Reserved: validated to `0.0–1.0` but currently ignored |

Response fields:

| Field | Type | Meaning |
|---|---|---|
| `tamper_risk_score` | float 0–1 | Share of the forest's trees voting "tampered". **Uncalibrated** — not a probability of fraud. |
| `confidence` | float 0–1 | Vote share of the predicted class |
| `flagged_regions` | list | Where the page is anomalous: boxes `x`, `y`, `width`, `height` (pixels of the uploaded image; PDF: page 1 at 200 DPI), `area`, `type` (`noise`, `colour`, `softness`, `ink`, or several joined by `+`). Largest first, at most 5. Returned only when the score reaches `TAMPER_THRESHOLD` and the model is v2; otherwise `[]` |
| `interpretation` | string | Plain-language summary of the score |

Status codes: `200` success · `400` missing/empty/unsupported/corrupted file, image too
large in pixels or too small, password-protected PDF · `401` missing API key ·
`403` wrong API key · `404` unknown route · `405` wrong method · `413` file too large ·
`422` invalid query parameter · `500` unexpected server error (generic message with
request ID) · `503` model not loaded.

---

## Example requests

**curl (macOS / Linux / Git Bash):**

```bash
# Health check — no key
curl http://127.0.0.1:8000/health

# Verify a document
curl -X POST http://127.0.0.1:8000/verify \
  -H "X-API-Key: my-local-secret" \
  -F "file=@data/demo/genuine_payslip_01.png"

# Missing key → 401
curl -X POST http://127.0.0.1:8000/verify -F "file=@data/demo/genuine_payslip_01.png"

# Wrong key → 403
curl -X POST http://127.0.0.1:8000/verify \
  -H "X-API-Key: wrong-key" -F "file=@data/demo/genuine_payslip_01.png"
```

**PowerShell (Windows):**

```powershell
curl.exe -X POST http://127.0.0.1:8000/verify `
  -H "X-API-Key: my-local-secret" `
  -F "file=@data/demo/genuine_payslip_01.png"
```

**Python (`requests`):**

```python
import requests

with open("data/demo/genuine_payslip_01.png", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/verify",
        headers={"X-API-Key": "my-local-secret"},
        files={"file": ("genuine_payslip_01.png", f, "image/png")},
        timeout=30,
    )
print(response.status_code, response.json())
```

The five images in `data/demo/` are described in
[data/demo/DEMO_IMAGES.md](data/demo/DEMO_IMAGES.md), including which ones the model gets
wrong.

---

## Example responses

Captured from a running server on this commit:

**`GET /health` → 200**

```json
{"status":"ok","version":"0.1.0","model_loaded":true,"model_version":"2.0.0","uptime_seconds":0.2,"detail":null}
```

**`POST /verify` with `data/demo/genuine_payslip_01.png` → 200**

```json
{
  "tamper_risk_score": 0.0567,
  "confidence": 0.9433,
  "flagged_regions": [],
  "interpretation": "Document verified successfully. Low risk of digital tampering detected (risk score: 0.06, confidence: 0.94)."
}
```

**`POST /verify` with `data/demo/tampered_payslip_01.png` → 200**

```json
{
  "tamper_risk_score": 0.7933,
  "confidence": 0.7933,
  "flagged_regions": [
    {"x": 198, "y": 321, "width": 96, "height": 9, "area": 864, "type": "ink"}
  ],
  "interpretation": "Evidence of digital tampering detected (risk score: 0.79, confidence: 0.79). 1 suspicious region(s) located (see flagged_regions)."
}
```

The box covers the replaced account number (the edit recorded for this demo image is
`[197, 321, 55, 12]`): its ink darkness is rare on the page.

When the score is at or above `TAMPER_THRESHOLD`, the interpretation reads
`Evidence of digital tampering detected (risk score: …, confidence: …).`, followed by the
number of regions located when there are any.

**Errors** all share one envelope:

```json
{"success": false, "error": "Unauthorized: Missing X-API-Key header", "code": 401, "detail": "Unauthorized: Missing X-API-Key header"}
```

```json
{"success": false, "error": "Invalid file format: '.md' is not supported. Allowed formats: jpg, jpeg, png, pdf", "code": 400, "detail": "Invalid file format: '.md' is not supported. Allowed formats: jpg, jpeg, png, pdf"}
```

---

## Testing

Use `python -m pytest` so the project's installed dependencies are used.

```bash
# Full suite (~2–3 minutes; includes load and model tests)
python -m pytest -v

# API behaviour: auth, validation, errors, response schema (< 1 second)
python -m pytest tests/test_api_comprehensive.py -v

# Dataset integrity (75 documents, index.csv consistency)
python -m pytest tests/test_dataset.py -v

# Coverage for the API package
python -m pytest tests/test_api_comprehensive.py --cov=backend --cov-report=term-missing

# Regenerate logs/api_test_report.txt from a real run
python scripts/generate_api_test_report.py

# Lint (config in ruff.toml; install with: pip install ruff)
ruff check .
```

Latest full run on this commit: **338 passed, 2 skipped, 1 xfailed, 0 failed, 0 warnings.**
The 2 skips are tests for an experimental local-forensics model that was never finalized;
the expected failure (xfail) documents a known detection weakness in that same experiment.

---

## Load testing

`scripts/load_test.py` sends concurrent requests to `/health` and `/verify` and writes a
report to `logs/api_load_report.txt`.

```bash
# Against a running server (start uvicorn first)
python scripts/load_test.py --url http://127.0.0.1:8000 --api-key my-local-secret

# Without a server (in-process ASGI transport)
python scripts/load_test.py --in-process

# Stress tiers: 5, 10 and 20 concurrent users
python scripts/load_test.py --in-process --stress

# Automated load tests (part of the full suite)
python -m pytest tests/test_api_load.py -v
```

Day 10 results — live uvicorn server over HTTP, 4-CPU Linux, demo payslip
(`logs/final_load_report.txt`). Sprint targets: `/verify` average < 2 s and success > 95%
at 5 concurrent users, `/health` < 50 ms.

| Scenario | Avg latency | p95 | Max | Throughput | Success | Target |
|---|---|---|---|---|---|---|
| `/health`, 5 users × 10 requests | 7.5 ms | 17 ms | 17 ms | 634 req/s | 100% | met |
| `/verify`, 5 users × 10 requests | 1.19 s | 1.62 s | 1.74 s | 4.0 req/s | 100% | met |
| `/verify` stress, 10 users | 1.67 s | 3.46 s | 3.46 s | 4.1 req/s | 100% | met |
| `/verify` stress, 20 users | 3.23 s | 8.09 s | 8.25 s | 4.2 req/s | 100% | **missed** (target is for 5 users) |

Throughput levels off at about 4 documents per second per process, so latency grows with
concurrency; more capacity needs more worker processes (not yet measured). Day 5's
in-process run on a different machine measured 0.94 s at 5 users (`logs/api_load_report.txt`).
Server memory and CPU: `logs/performance_review.txt`.

---

## Project structure

```
docfraud-backend/
├── backend/                 FastAPI service
│   ├── main.py              App, API-key middleware, /health and /verify, response schemas
│   ├── config.py            All settings (env vars with defaults)
│   ├── error_handlers.py    File validation and uniform JSON error responses
│   ├── exceptions.py        Typed API exceptions (400/401/403/404/413/422/500/503)
│   ├── middleware.py        Request IDs + access log; upload-size limit before parsing
│   ├── logging_config.py    Log format with request ID
│   └── model_utils.py       ModelAPI: thread-safe model loading and prediction
├── core/                    Analysis engine
│   ├── document_analyzer.py Image loading (shared), ELA and the v1 22-feature extraction
│   ├── tamper_features.py   Model v2 features: 29 local tamper features (local-tamper-v2)
│   ├── feature_schemas.py   Maps a model's recorded feature schema to its extractor
│   ├── model_trainer.py     v1 training, model save/load
│   ├── local_forensics.py   Experimental region-level features (not used by the API)
│   └── forensic_model.py    Experimental local-forensics model (not used by the API)
├── data/
│   ├── models/              fraud_model.pkl — the model the API loads (v2); fraud_model_v1.pkl
│   ├── generated/           dev_v1/ (training) and holdout_v1/ (evaluation only) documents
│   ├── synthetic/           75-document labelled dataset + index.csv
│   ├── demo/                5 demo images + manifest (demonstration only)
│   ├── splits/              Frozen train/test split
│   └── augmented/           Augmentation manifests (images git-ignored)
├── scripts/                 Offline tools: generation, training, validation, load tests, reports
├── tests/                   pytest suites (API, analyzer, model, dataset, demo, validation)
│   ├── conftest.py          Shared fixtures (client, API key, sample images, model mock)
│   └── fixtures/            Small test images
├── logs/                    Generated reports and release notes
├── README.md                This file
├── DEMO_GUIDE.md            Live demo script, API usage, troubleshooting
├── API_DOCS.md              Endpoint reference
├── ARCHITECTURE.md          System design
├── DEPLOYMENT.md            Install, configure, run, production gaps
├── DATASET.md               Dataset statistics
├── DATA_FORMAT_NOTES.md     Dataset layout and safe generator usage
├── PROGRESS.md              Day-by-day sprint log
├── .env.example             Configuration template
├── ruff.toml                Lint and format settings
├── pytest.ini               pytest settings
└── requirements.txt         Python dependencies
```

---

## Documentation

| Document | Covers |
|---|---|
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | Running a live demo, API usage for newcomers, demo-day troubleshooting |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, request flow, feature pipeline, model loading, response fields, limitations |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Install, configuration, running the server, smoke tests, troubleshooting, production gaps |
| [API_DOCS.md](API_DOCS.md) | Endpoint reference, authentication, error codes, client examples |
| [DATASET.md](DATASET.md) | Final dataset index and statistics (generated by `scripts/dataset_stats.py`) |
| [DATA_FORMAT_NOTES.md](DATA_FORMAT_NOTES.md) | Dataset layout, `index.csv` columns, safe generator usage |
| [data/demo/DEMO_IMAGES.md](data/demo/DEMO_IMAGES.md) | The 5 demo images, their sources and recorded predictions |
| [logs/model_validation_report.txt](logs/model_validation_report.txt) | Day 8 model checklist, demo validation, held-out accuracy of model v2 (Founder A) |
| [logs/training_log_v2.txt](logs/training_log_v2.txt) | Model v2 training data, cross-validation and feature importances |
| [logs/model_release_notes.txt](logs/model_release_notes.txt) | Model v2.0.0 release notes: metrics, limitations, failure cases, rollback |
| [logs/model_review_report.txt](logs/model_review_report.txt) | Day 9 model review: performance, prediction quality, speed / memory / size, release checklist |
| [logs/model_robustness_report.txt](logs/model_robustness_report.txt) | Day 9 held-out accuracy under low resolution, rotation, cropping and blur |
| [logs/feature_importance_report.txt](logs/feature_importance_report.txt) | Day 9 feature importance of model v2 |
| [logs/localization_report.txt](logs/localization_report.txt) | `flagged_regions` accuracy against the true edit rectangles (DEV, held-out, resampled pages) |
| [logs/model_stability_report.txt](logs/model_stability_report.txt) | Day 8 model stability: repeatability, feature/ELA consistency, drift, input and retraining sensitivity |
| [logs/final_model_release_report.txt](logs/final_model_release_report.txt) | Day 10 final model release report and sign-off (Founder A): metrics, demo validation, performance, limitations, recommendation |
| [logs/model_signoff_checklist.txt](logs/model_signoff_checklist.txt) | Day 10 model deployment approval checklist |
| [logs/model_limitations.txt](logs/model_limitations.txt) | Known limitations of model v2: dataset, edge cases, failure scenarios, future improvements |
| [logs/final_accuracy_report.txt](logs/final_accuracy_report.txt) | Day 10 held-out accuracy check and comparison with earlier metrics |
| [logs/inference_benchmark_report.txt](logs/inference_benchmark_report.txt) | Day 10 inference benchmark: load time, min/avg/max latency, HTTP at 5 users, stability |
| [logs/final_release_report.txt](logs/final_release_report.txt) | Day 10 release validation of the API with model v1 (Founder B): tests, performance vs targets, readiness checklist |
| [logs/release_notes_v1.txt](logs/release_notes_v1.txt) | v0.1.0 release notes: features, metrics, limitations, next steps |
| [PROGRESS.md](PROGRESS.md) | Day-by-day sprint log for all three founders |

---

## Current status

- **API:** `GET /health` and `POST /verify` run under uvicorn; verified with a real server
  and curl on this commit.
- **Model (v2, Day 8):** Random Forest, 300 trees, 29 local tamper features (schema
  `local-tamper-v2`). The deployed file never saw its evaluation data:
  - frozen 15-document test set: **86.7% (13/15)**, 2 of 10 genuine flagged, 0 of 5 tampered missed
    (v1 scored 60%, 9/15, on the same images);
  - fresh held-out set of 149 generated documents: **92.6% (138/149)**, 95% CI 87–96%,
    1 of 83 genuine flagged, 10 of 66 tampered missed (3 of those edits changed no pixel).
- **Dataset:** 75 synthetic documents (48 genuine, 27 tampered; 30 payslips, 23 bank
  statements, 22 ID proofs).
- **Demo set:** 4 of 5 demo images classified correctly with v2 (v1: 2 of 5); a
  demonstration record, not an accuracy estimate.
- **Tests:** 338 passed, 2 skipped, 1 xfailed, 0 failed, 0 warnings.
- **Localisation:** `flagged_regions` returns boxes for flagged documents (model v2). On
  held-out documents a box lands on 55 of 67 visible edits (82.1%; top-1 77.6%), and 2 of 93
  genuine documents get boxes; DEV (where the thresholds were chosen) 49/59 (83.1%)
  (`logs/localization_report.txt`).
- **Model sign-off (Day 10):** 6/6 checklist items. Model v2 is approved for the demo, the investor
  presentation and an internal pilot, and not for automated decisions on real documents.
  `/verify` averages 0.5 s at 5 concurrent users with 100% success
  (`logs/final_model_release_report.txt`, `logs/model_limitations.txt`).
- **Model review (Day 9):** release checklist 6/6; `/verify` inference 2.2x faster (single-threaded
  forest, identical outputs); robustness and feature importance documented
  (`logs/model_review_report.txt`, `logs/model_release_notes.txt`).
- **Model stability (Day 8):** deterministic - every demo image returns the identical score on
  repeated calls; retraining with the recorded seed reproduces the file exactly
  (`logs/model_stability_report.txt`, `logs/model_validation_report.txt`).

Details for each day and founder: [PROGRESS.md](PROGRESS.md).

---

## Limitations

- Trained and evaluated on **synthetic** documents only; accuracy on real documents is unknown.
- Held-out accuracy (86.7% / 92.6%) is measured on documents from the same synthetic
  generator as the training data. Edits the generator places on blank paper change no pixel
  and cannot be detected.
- Much less accurate on degraded files than on clean renders. On 164 held-out documents
  (92.1% unmodified): JPEG q85 re-save 73.2%, 50% downscale 50.0%, 180° rotation 64.0%,
  3° skew 70.1%, 2-px crop 79.9%, light blur ~85% (`logs/model_robustness_report.txt`).
  Phone photos and scans are therefore not supported yet.
- Non-document images (photos, blank pages) still get a score, often above 0.5.
- `tamper_risk_score` is an **uncalibrated** Random Forest vote fraction, not a fraud probability.
- `flagged_regions` boxes are anomalies, not proof of an edit. On held-out documents a box
  lands on 82.1% of the edits (top-1 77.6%); text overlays on bank statements are mostly
  missed, and 2 of 93 genuine pages got boxes (`logs/localization_report.txt`). Boxes are
  only meaningful at the page size the model was trained on: on resampled pages the colour
  and noise maps light up everywhere (and the score is unreliable too).
- The `threshold` query parameter is validated but ignored; the server uses `TAMPER_THRESHOLD`.
- Only the first page of a PDF is analysed.
- About 8-9 `/verify` requests per second per process with model v2 on 4 CPUs (0.5 s average
  at 5 concurrent users, `logs/inference_benchmark_report.txt`). Higher concurrency has only been
  load-tested with model v1; multiple workers / horizontal scaling have not been tested.
- Not production-ready: default API key, open CORS by default, no HTTPS, no rate limiting,
  dependencies other than scikit-learn unpinned — see [DEPLOYMENT.md §10](DEPLOYMENT.md#10-production-readiness-limitations).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'backend'` | Server or tests started outside the repository root | `cd` to the repository root first |
| `ModuleNotFoundError: No module named 'fastapi'` when running `pytest` | A global `pytest` without the project's packages | Use `python -m pytest` inside the environment where you ran `pip install` |
| `/verify` returns `401` | No `X-API-Key` header | Add `-H "X-API-Key: <your API_KEY>"` |
| `/verify` returns `403` | Key does not match the server's `API_KEY` | Check the value the server was started with (`.env` or shell) |
| `/verify` returns `400 Invalid file format` | Extension not `.jpg`, `.jpeg`, `.png` or `.pdf` | Convert or rename the file correctly |
| `/verify` returns `400 … corrupted file` | File has an image extension but cannot be decoded | Re-export the image; check it opens in an image viewer |
| `/verify` returns `413` | File larger than `MAX_FILE_SIZE` | Send a smaller file or raise `MAX_FILE_SIZE` |
| `/health` and `/verify` return `503` (`degraded`) | Model missing, unreadable or incompatible; the startup log says which | Restore `data/models/fraud_model.pkl` or fix `MODEL_PATH`; the next request reloads it, no restart needed |
| `/verify` returns `400 … pixel limit` or `too small` | Image over 40 MP, or under 16 px on its short side | Resize the image; raise `MAX_IMAGE_PIXELS` only if you have the memory |
| `/verify` returns `500 … request id X` | Unexpected server error | Search the server log for `req=X` to see the full traceback |
| Startup fails with `Failed to load model` | Incompatible scikit-learn/numpy version for the pickle | Reinstall from `requirements.txt`; see DEPLOYMENT.md §9 |
| `RuntimeError: Form data requires "python-multipart"` | Dependency missing | `pip install -r requirements.txt` |

More cases: [DEPLOYMENT.md §9](DEPLOYMENT.md#9-troubleshooting).

---

## Contributors

| Role | Area |
|---|---|
| **Founder A** — Model | Feature engineering, model training and validation, local-forensics research, demo dataset |
| **Founder B** — API | FastAPI service, authentication, validation and error handling, API docs, load and API testing |
| **Founder C** — Data & QA | Synthetic dataset generation and validation, dataset statistics, architecture and deployment docs, final QA |
