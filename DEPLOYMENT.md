# Deployment — Document Fraud Detection Backend

> Describes how to set up and run the service as it exists on `main` at commit `c2f3754`
> (after Founder B's Day 9 hardening and Day 10 release QA, which changed runtime
> behaviour, configuration settings and `requirements.txt`). The model file is unchanged.
> **Verified** = run successfully in the project's Linux development environment
> (Python 3.11.15). A real production deployment has **not** been performed or
> verified. This service is **not production-ready** — see §10.

For system design see `ARCHITECTURE.md`; for endpoint details see `API_DOCS.md`.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python | **3.11.15 verified.** Other 3.11+ versions are expected to work but were not tested here |
| pip + venv | Standard Python tooling |
| git | To clone the repository (the model file is committed) |
| Writable temp directory | Each upload is written to a temporary file during analysis |

---

## 2. Install

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` contains both runtime and development packages:

| Needed at runtime | Development / testing only |
|---|---|
| `fastapi`, `uvicorn[standard]`, `pydantic`, `python-dotenv`, `python-multipart`, `pillow`, `opencv-python`, `numpy`, `scikit-learn`, `pymupdf` | `pytest`, `pytest-cov`, `httpx`, `faker` |

`.venv/` is already git-ignored.

---

## 3. Required model file

The API serves predictions from **`data/models/fraud_model.pkl`** (model v2, ≈5.7 MB,
committed to git and explicitly excluded from the `*.pkl` ignore rule). The previous model
(v1) is kept as `data/models/fraud_model_v1.pkl` for rollback; point `MODEL_PATH` at it to
serve v1.

- A scikit-learn `RandomForestClassifier` (300 trees) expecting **29 features** (schema
  `local-tamper-v2`). The schema is recorded in the file; `ModelAPI` picks the matching
  feature extractor (`core/feature_schemas.py`) and refuses a model whose schema or
  feature count it does not know.
- It is a pickle, so it should be loaded with a scikit-learn version compatible with
  the one that created it. `requirements.txt` pins `scikit-learn==1.9.1`, the version recorded
  in the model file's metadata (`environment`).
- Only load model files from trusted sources — unpickling executes code.

---

## 4. Configuration

Settings are read in `backend/config.py` from environment variables.

| Variable | Default | Behaviour |
|---|---|---|
| `API_KEY` | `default-api-key` | Value clients must send in the `X-API-Key` header |
| `MODEL_PATH` | `data/models/fraud_model.pkl` | Used **only if the path exists**; otherwise the default path is used silently |
| `MAX_FILE_SIZE` | `10485760` (10 MB) | Upload limit in bytes; larger uploads get `413` |
| `TAMPER_THRESHOLD` | `0.5` | Risk score at or above which the `interpretation` text reports tampering |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins, e.g. `https://app.example.com,https://admin.example.com` |
| `MAX_IMAGE_PIXELS` | `40000000` | Largest accepted image (PDF page 1 measured at 200 DPI); blocks decompression bombs |
| `MAX_CONCURRENT_ANALYSES` | `min(2, CPUs)` | Analyses running at once per process; 2 measured fastest on 4 CPUs, 4 was ~35% slower |
| `LOG_LEVEL` | `INFO` | Level for `backend.*` and `core.*` loggers |

**Process environment (not read by `config.py`):** set `MALLOC_ARENA_MAX=2` for the server
process on Linux. Under concurrent load it kept memory at ~270–310 MB instead of
~450–500 MB, at about 9% lower throughput (`logs/performance_review.txt` §6).

> ⚠️ **The default API key `default-api-key` is insecure and publicly known.**
> Always set your own `API_KEY` before exposing the service to anyone.

### Using a `.env` file

`backend/config.py` calls `python-dotenv`'s `load_dotenv()` at import time, so values
in a `.env` file are loaded automatically. Put `.env` in the **repository root** and
start the server from the repository root. (`load_dotenv()` searches upward from
either the current directory or `backend/`, depending on how Python is launched; with
`.env` and the working directory both at the repository root it is found either way.
Loading from a `.env` under a running uvicorn server was not separately tested; plain
environment variables were.) Environment variables that are already set take
precedence over `.env`. `.env` is git-ignored — never commit real keys.

```ini
# .env
API_KEY=replace-with-a-long-random-secret
MODEL_PATH=data/models/fraud_model.pkl
MAX_FILE_SIZE=10485760
```

`.env.example` in the repository root contains these keys and can be copied to `.env`.

---

## 5. Start the application

Always start from the **repository root** — imports assume `backend.` and `core.`
are importable from the current directory.

**Development (auto-reload on code changes):**

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Normal run (no reload)** — verified with a real uvicorn process:

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`python -m backend.main` also works; it starts uvicorn on `127.0.0.1:8000` **with**
reload enabled (development only).

A successful start logs:

```
INFO:     Waiting for application startup.
Model loaded: .../data/models/fraud_model.pkl
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

If `Model loaded:` does not appear, see §9.

Binding to `127.0.0.1` accepts local connections only. Exposing the service on a
network interface, running multiple workers, or running behind a reverse proxy has
**not** been verified.

---

## 6. Smoke test

**Health (public, no key):**

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","version":"0.1.0","model_loaded":true,"model_version":"2.0.0","uptime_seconds":0.6,"detail":null}
```

`/health` returns **200** only when the model is loaded; it returns **503** with
`"status":"degraded"` and a `detail` reason when the model is missing or failed to load,
so it can be used directly as a load-balancer readiness check.

**Verify (requires `X-API-Key`):**

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -H "X-API-Key: $API_KEY" \
  -F "file=@data/synthetic/genuine/payslip_001.png"
```

Example response (verified; values depend on the input):

```json
{"tamper_risk_score":0.1,"confidence":0.9,"flagged_regions":[],
 "interpretation":"Document verified successfully. Low risk of digital tampering detected (risk score: 0.10, confidence: 0.90)."}
```

Authentication results:

| Request | Status | Verified with |
|---|---|---|
| No `X-API-Key` header | `401` | running uvicorn server |
| Wrong key | `403` | running uvicorn server |
| Correct key + valid PNG | `200` | running uvicorn server |
| `/docs` | `200`, public | running uvicorn server |
| `/redoc`, `/openapi.json` | `200`, public | in-process test client |
| Unknown path | `404` (checked before authentication) | in-process test client |

Accepted uploads: `.jpg`, `.jpeg`, `.png`, `.pdf` (by file extension; PDFs are
analysed on page 1 only). Response fields and their limitations
(`tamper_risk_score` is uncalibrated, `flagged_regions` is filled only for flagged
documents and only with model v2, `threshold` is ignored) are described in `ARCHITECTURE.md` §8.

---

## 7. Files required at runtime

| Required to serve requests | Not required to serve requests |
|---|---|
| `backend/` | `data/synthetic/`, `data/generated/` (training / evaluation data) |
| `core/document_analyzer.py`, `core/tamper_features.py`, `core/feature_schemas.py` | `data/augmented/`, `data/splits/` |
| `core/model_trainer.py` | `scripts/` |
| `data/models/fraud_model.pkl` | `tests/`, `logs/`, `data/models/fraud_model_v1.pkl` (rollback) |
| `requirements.txt` (for install) | `core/forensic_model.py`, `core/local_forensics.py` |
| `.env` or environment variables | Markdown documentation |
| | `data/demo/` (demo images and manifest) |

**Offline and demo material.** `data/demo/` (Founder A's Day 7 demo images) is
repository/demo material: the API does not load it, and it is not needed to serve
requests. Likewise `scripts/dataset_stats.py`, `scripts/final_validation.py`,
`scripts/generate_extra_documents.py`, `scripts/train_tamper_model.py`,
`scripts/model_stability_test.py`, `scripts/model_validation.py`,
`scripts/feature_importance.py`, `scripts/model_robustness_test.py`, `scripts/model_review.py` and
`scripts/prepare_demo_dataset.py` are repository tooling for statistics, data generation,
training, model validation and demo preparation — none is imported by `backend/` or
required at runtime. The demo
images can be sent to a running server as `/verify` examples. See `ARCHITECTURE.md` §10.

---

## 8. Testing

```bash
# Dataset integrity (75 documents, index.csv consistency) — fast
pytest tests/test_dataset.py -v

# API behaviour (auth, validation, errors, response schema)
pytest tests/test_api_comprehensive.py -v

# Full suite (slow: includes load tests and model tests)
pytest -v
```

Verified during Day 6: `tests/test_dataset.py` + `tests/test_api_comprehensive.py`
→ **48 passed, 0 failed**. Full suite on the Day 7 cleanup commit
(`python -m pytest -q`): **253 passed, 2 skipped, 1 xfailed, 0 failed** (~2.5 minutes).
After the Day 9 hardening: **281 passed, 2 skipped, 1 xfailed, 0 failed**.
Use `python -m pytest` so the project's installed dependencies are used.

Do **not** re-run `scripts/generate_tampered_documents.py` in a deployment — depending on
the mode it overwrites the Day 1 payslips or appends duplicate documents to
`data/synthetic/`, changing the validated dataset (see `DATA_FORMAT_NOTES.md` §6).
`scripts/train_tamper_model.py` (model v2) and `scripts/train_model.py` (v1) overwrite
`data/models/fraud_model.pkl` by default; after retraining, re-score the demo set
(`scripts/prepare_demo_dataset.py --rescore`) and re-run `scripts/model_validation.py`.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Startup log shows `Starting in DEGRADED mode: model file not found`; `/health` and `/verify` return **503** | No model at the configured or default path | Ensure `data/models/fraud_model.pkl` exists, or set `MODEL_PATH` to an existing file; the next request reloads it |
| App ignores your `MODEL_PATH` | The path does not exist, so `config.py` falls back to the default | Check the path; relative paths resolve from the directory you start in |
| Startup log shows `Failed to load model` or `Incompatible model … expects N features`; `/health` **503** | Model file cannot be unpickled (corrupt, or incompatible scikit-learn/numpy), or was trained on a different feature set | Restore the committed file; use scikit-learn/numpy versions compatible with the model |
| `/verify` returns `500 … (request id X)` | Unexpected server error | Find `req=X` in the server log for the full traceback |
| Memory grows to ~500 MB under concurrent load | glibc keeps per-thread memory arenas | Set `MALLOC_ARENA_MAX=2` for the server process |
| `RuntimeError: Form data requires "python-multipart" to be installed` | Dependency missing | `pip install -r requirements.txt` (it is listed there) |
| `ModuleNotFoundError: No module named 'backend'` | Server started outside the repository root | `cd` to the repository root first |
| `/verify` returns `401` / `403` | Missing / wrong `X-API-Key` | Send the header with the value of `API_KEY` |
| `/verify` returns `413` | Upload larger than `MAX_FILE_SIZE` | Send a smaller file or raise `MAX_FILE_SIZE` |
| `/verify` returns `400` | Missing file, unsupported extension, empty or unreadable file | Send a valid `.jpg`, `.jpeg`, `.png` or `.pdf` |

---

## 10. Production-readiness limitations

The following are **not** done. Treat the current service as a development prototype.

- **API key:** the default `default-api-key` must be replaced (the server logs a warning
  at startup while it is in use). It is a single shared secret, compared in constant
  time, with no rotation or per-client keys.
- **CORS:** the default `CORS_ORIGINS=*` allows any origin (without credentials). Set
  `CORS_ORIGINS` to the known frontend origins before any real use; the server logs a
  warning at startup while `*` is in use.
- **HTTPS:** not configured. No reverse proxy, TLS termination or rate limiting exists.
- **Dependency pinning:** `scikit-learn` is pinned to the model's version (1.9.1).
  `numpy`, `opencv-python`, `pillow`, `scipy` and `pymupdf` are unpinned; pin the versions
  recorded in `fraud_model.pkl`'s metadata before deployment (they can shift features slightly).
  `python scripts/final_model_validation.py` compares the installed versions with that record.
- **OpenCV on a fresh server:** unverified. `opencv-python` may need extra system
  libraries on minimal Linux images (`opencv-python-headless` is a common
  alternative); this has not been tested.
- **Real deployment:** no production server, container, process manager, multiple
  workers or monitoring has been set up or verified. There is no Dockerfile.
- **Model quality:** model v2 reaches 86.7% (13/15) on the frozen synthetic test set and
  92.6% (138/149) on a fresh synthetic holdout set, but scores are uncalibrated, degraded
  files (JPEG re-save, downscaling, rotation, cropping) drop accuracy to 50–80%
  (`logs/model_robustness_report.txt`), and it has never been evaluated on real documents.
  Results must not be used for real decisions.
- **Rate limiting:** none. Uploads are bounded (size, pixels, 2 concurrent analyses per
  process), but a client can still queue many requests.
- **Performance:** load-test numbers in `logs/api_load_report.txt` were measured
  in-process by Founder B and have not been reproduced on a server.

---

## 11. Cross-machine validation

Use this checklist to confirm the backend works on a new machine (a teammate's laptop,
a demo PC or a server). Everything runs from the repository root. Allow about
10 minutes, mostly for `pip install`.

### 11.1 Prerequisites

| Requirement | Check with |
|---|---|
| Python 3.11+ | `python --version` (Windows: `py --version`) |
| git | `git --version` |
| ~1 GB free disk | dependencies and a virtual environment |
| Internet access to PyPI and GitHub | for `git clone` and `pip install` |

### 11.2 Fresh setup

**macOS / Linux**

```bash
git clone https://github.com/bssco178-a11y/docfraud-backend.git
cd docfraud-backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/bssco178-a11y/docfraud-backend.git
cd docfraud-backend
py -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 11.3 Automated validation

```bash
# 1. Environment: Python, every package, required files, model, config, live server smoke test
python scripts/validate_environment.py            # -> logs/environment_validation.txt

# 2. End-to-end: auth, validation, pipeline stages, 5 demo images, 20-request stability run
python scripts/e2e_test.py                        # -> logs/e2e_test_report.txt

# 3. Test suite (~2-3 minutes)
python -m pytest -q
```

Each script exits with code `0` only if every check passes, and prints the report.
`python scripts/validate_environment.py --fresh-clone` additionally clones the
repository into a temporary directory, builds a brand-new virtual environment and
repeats the checks there — use it to rule out anything specific to your working copy.

### 11.4 Manual check

```bash
export API_KEY="my-local-secret"                   # PowerShell: $env:API_KEY="my-local-secret"
uvicorn backend.main:app --host 127.0.0.1 --port 8000
# second terminal:
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/verify -H "X-API-Key: my-local-secret" \
     -F "file=@data/demo/genuine_payslip_01.png"
```

(On Windows use `curl.exe` and a backtick `` ` `` for line continuation.)

### 11.5 What "working" looks like

| Check | Expected on any machine |
|---|---|
| `validate_environment.py` | `Overall: PASS` |
| `e2e_test.py` | `Overall result: PASS`; all 5 demo scores equal the manifest values below |
| `python -m pytest -q` | 0 failed (2 skipped and 1 xfailed are expected) |
| Demo scores (model v2) | `genuine_payslip_01` 0.06 · `genuine_payslip_02` 0.06 · `tampered_payslip_01` 0.79 · `tampered_payslip_02` 0.15 · `edge_case_01` 0.22 |

The demo scores come from the committed model and are deterministic. A **different score
on another machine** usually means a different scikit-learn, numpy, Pillow or OpenCV
version changed feature extraction or model loading — compare the package list in
`logs/environment_validation.txt` between the two machines. Latency numbers differ by
hardware and are not expected to match.

### 11.6 Machine-specific problems

| Symptom | Fix |
|---|---|
| `python` not found (Windows) | Use `py`, or reinstall Python with "Add to PATH" ticked |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process RemoteSigned`, then activate again |
| `ImportError: libGL.so.1` (minimal Linux) | `sudo apt-get install -y libgl1 libglib2.0-0`, or replace `opencv-python` with `opencv-python-headless` |
| `Address already in use` on port 8000 | Start with `--port 8001`, or stop the other process |
| `InconsistentVersionWarning` when the model loads | scikit-learn differs from the one that saved the model; install the version shown in a passing `environment_validation.txt` |
| Line-ending warnings from git on Windows | Harmless; images and the model are binary and unaffected |
