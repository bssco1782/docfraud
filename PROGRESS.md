# Project Progress

## Day 1 (Sept 21) — Project Setup

### Founder A (Model)

- [x] Created project structure
- [x] Git repo initialized
- [x] requirements.txt created
- [x] .gitignore set up

### Founder B (API)

- [x] FastAPI skeleton created
- [x] /health endpoint working
- [x] API runs on localhost:8000

### Founder C (Data Pipeline & Testing)

- [x] Created `scripts/generate_tampered_documents.py` with reusable `generate_fake_payslip()`
- [x] Synthetic payslip renderer using Faker + Pillow (all fields: company, employee, salary breakdown)
- [x] Every generated image watermarked "SYNTHETIC / TEST DATA" (diagonal + footer banner)
- [x] Generated 5 test payslip images in `data/synthetic/` (payslip_001.png – payslip_005.png)
- [x] Created `DATA_FORMAT_NOTES.md` documenting fields, formats, labelling, and usage
- [x] Verification passed: script runs, all 5 images open as 850×1050 RGB PNGs, no real PII used
- [x] No modifications to Founder A or Founder B files

## Day 2 (Sept 22) — Core Model Engine & API Implementation

### Founder A (Model)

- [x] ELA function working (tested on 6 fixture images; real-image validation NOT PERFORMED — data/synthetic/ empty)
- [x] Feature extraction: 15 features extracted, all finite, deterministic
- [x] Unit tests pass: 74 passed, 3 skipped (real-image tests skipped — data/synthetic/ has no real documents)
- [x] Real-image validation: NOW PASSING (Founder C dataset delivered)

### Founder B (API)

- [x] POST /verify endpoint created
- [x] File upload handling working
- [x] Error handling: 400, 403, 413, 500
- [x] Tested with curl

### Founder C (Data)

- [x] Bank statement generator created
- [x] ID proof generator created (Aadhaar Card / PAN Card)
- [x] 25+ synthetic documents generated (10 payslips, 8 bank statements, 7 ID proofs)
- [x] Dataset organised into `data/synthetic/genuine/` (18) and `data/synthetic/tampered/` (7)
- [x] Tampering methods: blur_region, color_shift, noise_patch, text_overlay
- [x] CSV index created (`data/synthetic/index.csv` — 25 rows with metadata)
- [x] Day 1 payslip data preserved
- [x] No modifications to Founder A or Founder B files

## Day 3 (Sept 23) — Model Training & Test Suite

### Founder A (Model)

- [x] Created `core/model_trainer.py` — `DocumentFraudDetector` class
- [x] Created `scripts/train_model.py` — standalone training script (no FastAPI dependency)
- [x] Created `tests/test_model_trainer.py` — 12 tests
- [x] Fixed holdout evaluation bug (y_true mismatch when using train_test_split)
- [x] Model saved to `data/models/fraud_model.pkl` (134,358 bytes)

**Previous smoke test (before C's data):**
- 6 fixture samples (5 genuine + 1 tampered) — INVALID AS FINAL EVALUATION
- Accuracy: 0.8333, Precision: 0.0000, Recall: 0.0000, F1: 0.0000

**Final Day-3 run (Founder C's dataset):**
- Dataset: 25 samples (18 genuine, 7 tampered)
- Doc types: payslip (10), bank_statement (8), id_proof (7)
- Features: 15 (from Day-2 `document_analyzer.py` FEATURE_NAMES)
- Skipped: 0
- Evaluation method: holdout 80/20 (train=20, test=5)

| Metric | Value |
|---|---|
| Accuracy | 0.8000 |
| Precision (tampered) | 0.5000 |
| Recall (tampered) | 1.0000 |
| F1 (tampered) | 0.6667 |

Confusion matrix (test set, n=5):
```
                   predicted_genuine  predicted_tampered
actual_genuine          3                1
actual_tampered         0                1
```

- Training: RandomForestClassifier, 100 trees, class_weight="balanced", random_state=42
- Model retrained on all 25 samples after evaluation, saved for production use
- Model path: `data/models/fraud_model.pkl`
- Tests: 101 passed, 0 skipped, 0 failures

**Data leakage check:**
- No overlapping IDs between genuine and tampered sets
- Tampered docs use independent numbering (not modified copies of genuine docs)
- Caveat: all docs from same synthetic generator — model may learn generator artifacts rather than real tampering signals

**Limitations:**
- Small dataset (25 samples, only 1 tampered in test set) — metrics have high variance
- Test set contains only 5 samples — single misclassification shifts accuracy by 20%
- Synthetic data only — no real-world documents tested
- Model needs retraining on larger/real dataset for production use
- 1 false positive in test set (genuine predicted as tampered)

### Founder C (Data)

- [x] Test suite created (`tests/test_model.py`)
- [x] Tests pass: 20 passed, 0 skipped
- [x] Test 1 — ELA analysis on synthetic data: 8 tests passed (heatmap structure, score range, dimensions, determinism)
- [x] Test 2 — Feature extraction on synthetic data: 10 tests passed (15 features, finite floats, vector ordering, determinism, genuine+tampered validation, index.csv integrity)
- [x] Test 3 — Model prediction: 2 tests passed (prediction structure validation, valid label on genuine+tampered images) — runs against Founder A's trained model
- [x] No modifications to Founder A or Founder B files

### Founder B (API)

- [x] Model loader integrated into API
- [x] /verify endpoint wired to model
- [x] Tested on 5 synthetic images
- [x] Response time: 0.8-0.9s per image

## Day 4 (Sept 24) — Feature Engineering, Dataset Expansion & API Production Readiness

### Founder A
- [x] Feature engineering completed
- [x] 22 numeric features implemented (was 15)
- [x] Feature schema documented in single `FEATURE_NAMES` constant
- [x] Feature validation completed — 0 NaN, 0 Inf across 25 images
- [x] Feature extraction tested — 124 tests passed, 0 failed
- [x] Determinism verified — same image produces identical features
- [x] Full test suite verified

**New features added (7):**

| Feature | Category | Description |
|---|---|---|
| image_width | Geometry | Image width in pixels after load/resize |
| image_height | Geometry | Image height in pixels after load/resize |
| aspect_ratio | Geometry | width / height |
| grayscale_mean | Intensity | Mean grayscale pixel intensity [0–255] |
| grayscale_std | Intensity | Std of grayscale intensity (contrast proxy) |
| edge_density | Edge/Texture | Fraction of Canny edge pixels |
| dark_pixel_ratio | Texture | Fraction of pixels with intensity < 50 |

**Feature statistics audit (25 synthetic images):**
- All 22 features: 0 NaN, 0 Inf
- Constant features in synthetic data: `image_width` (850 — all synthetic docs same width), `ela_max` (255), `copy_move_matches` (0), `exif_edit_flag` (0)
- `image_width` constant is expected: all synthetic images are 850px wide; will vary with real documents
- `aspect_ratio` and `image_height` correlated (-0.999) because width constant — artifact of synthetic uniformity, not true redundancy

**Model compatibility:**
- Old model (`fraud_model.pkl`): trained on 15 features
- New feature schema: 22 features
- **Day 5 retraining required** — model cannot consume new feature vector

**Performance:**
- Mean extraction time: 434.4 ms/image
- Range: 255–667 ms/image
- Total for 25 images: 10.86 s

### Founder B (API) - Day 4
- [x] Centralized error handling added
- [x] File validation implemented
- [x] Swagger documentation improved
- [x] API_DOCS.md created
- [x] All error scenarios tested

### Founder C (Data)

- [x] Dataset expanded from 25 to 75 documents (50 new, indices 026–075)
- [x] Final dataset: 48 genuine, 27 tampered
- [x] Doc types: payslip (30), bank_statement (23), id_proof (22)
- [x] Tamper methods used: noise_patch (9), color_shift (8), text_overlay (6), blur_region (4)
- [x] `data/synthetic/index.csv` updated in append mode — 75 rows, 0 missing, 0 duplicates, 0 broken images
- [x] Original 25 documents + Day 1 payslips preserved (sha256 verified unchanged)
- [x] Generator: added `--day4` mode (`start_index`, append-only index, refuses to overwrite existing files, seed 4)
- [x] `tests/test_model.py` index test now checks index.csv against files on disk (was hard-coded to 25)
- [ ] Note: 13 tests fail on main before and after this change (22-feature extractor vs 15-feature tests/model) — needs Day 5 retraining; not caused by dataset expansion

## Day 5 (Sept 25) — Model Retraining & Load Testing

### Founder A
- [x] Model retrained using Day 4 feature set (22 features, schema `day4-v2-22`)
- [x] Expanded dataset verified — 75 usable (48 genuine, 27 tampered), 0 corrupt, 0 exact duplicates (file bytes + decoded pixels), labels match folders
- [x] Reproducible train/test split created — stratified holdout 80/20, seed 42 (train=60, test=15: 10 genuine / 5 tampered)
- [x] Accuracy measured: 60.0% (9/15) — **TARGET 78% NOT REACHED**
- [x] Precision measured: 44.4% (tampered class)
- [x] Recall measured: 80.0% (tampered class)
- [x] F1-score measured: 57.1% (tampered class)
- [x] Model saved: data/models/fraud_model.pkl
- [x] Model reload verified (fresh process, 22 features, inference on 4 test documents)
- [x] Full test suite executed — 176 passed, 0 failed, 0 skipped; 2 API test files not collected (see below)

**Confusion matrix (held-out test set, n=15; rows=actual, cols=predicted):**
```
                   predicted_genuine  predicted_tampered
actual_genuine          5                5
actual_tampered         1                4
```
TP=4, TN=5, FP=5, FN=1

**Day 3 vs Day 5 (NOT directly comparable):**

| Metric | Day 3 (sprint baseline) | Day 5 (measured) |
|---|---|---|
| Accuracy | 72% | 60.0% |
| Precision | 70% | 44.4% |
| Recall | 75% | 80.0% |
| F1 | — | 57.1% |

What changed: dataset 25 → 75 docs; features 15 → 22; test set 5 → 15 docs (different split). Model config unchanged (RandomForest, n_estimators=100, class_weight="balanced", random_state=42, other params sklearn defaults). The Day 3 run recorded in this file measured 80% accuracy on a 5-image test set (1 tampered), where one error moves accuracy by 20 points, so neither Day 3 number is a stable baseline.

**Honesty / leakage controls:**
- Only image-derived features used; filename, folder, label, CSV metadata never enter X
- One training run evaluated; no hyperparameter tuning, no split re-rolling, no sample removal
- Exact-duplicate guard added to `load_dataset()` (fails if identical files exist)
- Rerun with same config reproduces identical split, dataset hash and metrics
- Saved production model is refit on all 75 docs after holdout evaluation (Day 3 convention); the metrics above come from the model trained on 60 docs
- `predict_proba` output is an uncalibrated RF vote fraction, not a real-world fraud probability

**Likely limitation (from measured results):**
- Train-only 5-fold CV (test set never used) gives 58.3% ± 14.9% — so 60% is not a bad split; the features do not separate the classes well
- Tampering is local (small patches/overlays) but all 22 features are whole-image statistics; top importances are `grayscale_std`, `grayscale_mean`, `blockiness`, `dark_pixel_ratio` — these vary with document type and content more than with tampering
- 5 false positives are genuine docs (3 bank statements); 1 miss is an ID proof with `text_overlay`
- 4 features are constant on this dataset (`ela_max`, `copy_move_matches`, `exif_edit_flag`, `image_width`)
- Next step candidates: region-level/localised features (e.g. max/percentile of per-block ELA and noise z-scores), per-doc-type normalisation, larger dataset

**Test changes (pre-existing failures, not caused by Day 5):**
- `tests/test_model.py`: 3 assertions updated from 15 → 22 features (stale since Day 4 schema change); index test uses `as_posix()` so it passes on Windows (was comparing `genuine\x.png` with `genuine/x.png`). No assertions removed.

**Environment issue (resolved):** `python-multipart` added to `requirements.txt`.

Details: `logs/training_log.txt`

### Founder B (API) - Day 5
- [x] Load testing completed
- [x] 5 concurrent users supported
- [x] Performance metrics collected
- [x] Load test report generated
- [x] Optimization recommendations documented

**Performance & Load Test Highlights:**
- **Automated Test Suite**: `tests/test_api_load.py` (covers `/health`, `/verify`, distinct concurrent uploads, stress tiers)
- **Standalone CLI Tool**: `scripts/load_test.py` (supports live server targeting, in-process ASGI fallback, custom payloads, telemetry)
- **Official Performance Report**: `logs/api_load_report.txt`
- **5 Concurrent Users Acceptance Results**:
  * `/health`: 10.23 ms average latency, 476.21 req/s throughput, 100% success rate (50/50)
  * `/verify`: 0.939 s average latency (well below 2.0s requirement), 5.20 req/s throughput, 100% success rate (50/50), 0 unhandled failures
  * Simultaneous uploads: Isolated execution, zero race conditions, zero file collisions
- **Stress Tiers Profiled**:
  * 5 Users: 0.868s avg latency, 5.62 req/s throughput, 100% success rate
  * 10 Users: 1.602s avg latency, 5.46 req/s throughput, 100% success rate
  * 20 Users: 3.079s avg latency, 5.47 req/s throughput, 100% success rate
- **Optimizations Implemented**:
  * Offloaded CPU-heavy model inference to worker thread pool with `asyncio.to_thread` in `backend/main.py`
  * Added thread-safe double-checked locking to `ModelAPI` in `backend/model_utils.py` to eliminate concurrent initialization races
  * Ensured zero blocking on the asyncio event loop for lightning-fast `/health` responses under load


## Day 6 — Comprehensive API Testing

### Founder B (API) - Day 6
- [x] Comprehensive API test suite created
- [x] 15+ tests implemented
- [x] Authentication tests passing
- [x] File validation tests passing
- [x] Error handling tests passing
- [x] Test report generated

**Test Suite:** `tests/test_api_comprehensive.py`: 31 tests (25 test functions, plus parametrized cases), all passing in under 1 s
| Category | Tests | Covers |
|---|---|---|
| Health endpoint | 3 | 200 status, `status` field, `version` field |
| Authentication | 4 | missing key → 401, invalid key → 403, valid key → 200, `/health` stays public |
| File validation | 8 | empty → 400, unsupported types (txt/zip/exe/no extension) → 400, oversized → 413, corrupted image → 400, missing `file` field → 400 |
| Verify endpoint | 10 | real-model prediction, `tamper_risk_score` / `confidence` / `interpretation` types and ranges, exact JSON schema, model receives filename + bytes, all allowed extensions accepted |
| Error handling | 6 | inference exception → 500, model `ValueError` → 400, bad `threshold` → 422, unknown route → 404 (before auth), wrong method → 405 |

**Fixtures (`tests/conftest.py`):** `client`, `test_api_key` / `auth_headers` (patches `config.API_KEY`), `sample_image_bytes`, `sample_image_path`, `temp_file` factory, `mock_prediction`, `mock_model` (patches `backend.main.analyze_document`). None are autouse, so existing tests are unaffected.

**Report:** `logs/api_test_report.txt`, generated by `scripts/generate_api_test_report.py` from the pytest JUnit XML and pytest-cov output. Backend coverage from this module alone: 81% (`main.py` 91%, `error_handlers.py` 84%).

**Bug fixed: `/verify` deadlock on first request.** `ModelAPI.predict()` acquired `self._lock` and then called `load_model()`, which acquires the same non-reentrant `threading.Lock`. Any `/verify` call made before the model was loaded (e.g. without the startup lifespan, as in `TestClient(app)` tests) hung forever. `predict()` now calls `load_model()` directly, since that method already locks and re-checks. The Day 5 load tests did not catch this because the model was already loaded when they ran.

**Commands:**
```bash
pytest tests/test_api_comprehensive.py -v
pytest -v
pytest tests/test_api_comprehensive.py --cov=backend --cov-report=term-missing
python scripts/generate_api_test_report.py
```
`pytest-cov` added to `requirements.txt` for the coverage summary.

## Day 6 (Sept 26) — Final Validation

### Founder A (Model)

- [x] Final validation script created (`scripts/final_validation.py`)
- [x] Frozen 15-image test set evaluated (10 genuine / 5 tampered)
- [x] Per-image predictions generated (all 15, errors listed)
- [x] Confusion matrix generated
- [x] Edge cases tested
- [x] Validation report saved (`logs/final_validation_report.txt`)
- [x] Model readiness documented: **NOT ready** (below target, see below)

**Model validated:** `data/models/fraud_model.pkl` (v1.0.0, schema `day4-v2-22`, RandomForest n_estimators=100, class_weight=balanced, random_state=42) — the model the API loads. No local-forensics model was validated: `fraud_model_local_v1.pkl` was never finalized (Day-5 CV stopped before grid/final, awaiting review).

**Leakage finding:** the deployed `fraud_model.pkl` was refit on all 75 images after the Day-5 holdout, so it has trained on all 15 frozen test images (it scores 15/15 on them — memorization, not accuracy). Valid metrics below come from a held-out replica: same class, params, seed and features, fitted in memory on the 60 frozen TRAIN images only (not saved). It reproduces the Day-5 holdout confusion matrix exactly.

Metrics (held-out replica, n=15, positive class = tampered):
- Accuracy: 60.00% (9/15)
- Precision: 44.44%
- Recall: 80.00%
- F1: 0.5714
- TN=5, FP=5, FN=1, TP=4 — FPR 50.0% (5/10 genuine flagged), FNR 20.0% (1/5 tampered missed)
- Errors: 5 genuine flagged (3 bank statements, 1 payslip, 1 ID proof), 1 missed ID proof with `text_overlay`

Edge cases (via `ModelAPI.predict`, temp images made from a TRAIN image):
- Small images (80x60, 16x16): PASS
- Large images (2550x3150, 4000x5000): PASS — downscaled to 2000px; ~13-16 s, ~160 MB peak
- Grayscale (L, LA): PASS
- Low-quality (blur + JPEG q=5, JPEG q=1): PASS
- No robustness fixes were needed.

Test-set integrity: PASS (15/15 test + 60/60 train SHA-256 match, dataset fingerprint matches, no test image in augmentation v1/v2/v2.1, feature cache or CV results)

Training leakage: PASS for the held-out replica; **FAIL for the deployed artifact** (trained on test images).

READY FOR DAY-6 VALIDATION: **NO** — 60% is below the 78% target, a genuine document has a 50% chance of being flagged, and the deployed artifact has no clean held-out measurement. Not production ready. Next steps (not done on Day 6, to avoid tuning on the test set): decide on the local-forensics representation from CV only, finalize one model trained on TRAIN only, and collect a new held-out set (the current 15 images have now been evaluated twice with the same model config).

Tests: full suite 249 passed, 2 skipped, 1 xfailed, 0 failed (includes 3 new tests in `tests/test_final_validation.py` and Founder B's Day 5/6 API tests; `python-multipart` now installed from `requirements.txt`).

## Day 7 — Final Dataset Index & Statistics

### Founder C (Data)
- [x] Final dataset index confirmed: `data/synthetic/index.csv` — 75 rows, numbered 001–075 with no gaps
- [x] Statistics script created: `scripts/dataset_stats.py` (read-only; computes everything from `index.csv` + images; `--format markdown` for tables)
- [x] Statistics report created: `DATASET.md` (all numbers pasted from the script output)
- [x] `DATA_FORMAT_NOTES.md` rewritten for the current dataset layout; unsafe generator instructions removed (see below)
- [x] Validation: `pytest tests/test_dataset.py -v` — 17 passed, 0 failed
- [x] No dataset images or `index.csv` rows modified (sha256 of all 81 files in `data/synthetic/` unchanged)

**Final dataset (from `scripts/dataset_stats.py`):**

| | Count |
|---|---|
| Indexed documents | 75 |
| Genuine / tampered | 48 (64.0%) / 27 (36.0%) |
| Payslip / bank statement / ID proof | 30 / 23 / 22 |
| Tampered by type (payslip / bank / ID) | 13 / 8 / 6 |
| Tamper methods (blur / color / noise / text) | 4 / 8 / 9 / 6 |
| Day 1 payslips outside the index | 5 |
| Missing / unreadable / mismatched files | 0 / 0 / 0 |

**Generator safety findings (documented, generator code not changed):**
- Running `scripts/generate_tampered_documents.py` with no flag overwrites the 5 Day 1 payslips in `data/synthetic/` without warning (the old `DATA_FORMAT_NOTES.md` told users to run exactly this).
- Re-running `--day4` reproduces documents 026–075 byte-for-byte under new numbers (fixed seed 4), creating duplicate content.
- `--day2` on the current dataset stops with `FileExistsError` before writing.
- `DATA_FORMAT_NOTES.md` now says not to run any generator mode against `data/synthetic/`, and documents a tested workflow that writes to a separate directory with a new seed.

## Day 7 — Demo Dataset Preparation

### Founder A (Model)

- [x] Demo dataset created in `data/demo/` (5 images, byte-identical copies, SHA-256 verified)
- [x] `data/demo/DEMO_IMAGES.md` documents each image; `data/demo/demo_manifest.json` holds the machine-readable record
- [x] Reproducible builder: `scripts/prepare_demo_dataset.py` (selection rules fixed in code before scoring; refuses to re-score)
- [x] Integrity verified; 4 checks in `tests/test_demo_dataset.py`
- [ ] No model change, retraining, threshold change or test-set evaluation (by design)

**Selected images:**

| Demo file | Source | Actual label |
|---|---|---|
| genuine_payslip_01.png | `data/synthetic/payslip_002.png` (Day-1 payslip) | genuine |
| genuine_payslip_02.png | `data/synthetic/payslip_003.png` (Day-1 payslip) | genuine |
| tampered_payslip_01.png | `data/augmented/train_v2_1/tampered/payslip_002_aug1.png` (text replacement) | tampered |
| tampered_payslip_02.png | `data/augmented/train_v2_1/tampered/payslip_037_aug1.png` (text replacement + resampling) | tampered |
| edge_case_01.png | `data/augmented/train_v2_1/genuine/payslip_006_aug2.png` (whole-page JPEG q=78) | genuine |

Labels come from the Day-1 generator (no tampering code, commit bf18a4c) and the `train-aug-v2.1` manifest. Day-1 `payslip_001.png` was excluded: the Day-2 generator reused the Faker seed, which makes it a near-copy of training image `genuine/payslip_001.png`.

**Integrity:** frozen test overlap PASS (hashes, source documents, pixel near-copy check); hash verification PASS; label provenance PASS; frozen test set 15/15 unchanged.

**Scoring (once, deployed `data/models/fraud_model.pkl` v1.0.0 / `day4-v2-22` via `ModelAPI.predict`, tampered if risk >= 0.5):**

| Demo file | Actual | Predicted | Risk | Correct |
|---|---|---|---|---|
| genuine_payslip_01.png | genuine | genuine | 0.36 | YES |
| genuine_payslip_02.png | genuine | tampered | 0.59 | NO |
| tampered_payslip_01.png | tampered | genuine | 0.37 | NO |
| tampered_payslip_02.png | tampered | genuine | 0.14 | NO |
| edge_case_01.png | genuine | genuine | 0.35 | YES |

2/5 correct. This is a demo record, not an accuracy estimate; the only valid measurement is the Day-6 held-out 60% (9/15).

**Limitations:**
- The deployed model was trained on all 75 images, including the frozen test set. No demo file is a training image, but images 3–5 are views of its training documents, and genuine payslip 01 has the same employee record as training image `genuine/payslip_002.png` (different month and figures).
- The two tampered payslips are Day-5 training augmentations. Their edits are visible but land on the wrong fields (an account-style number in an allowance cell, an amount over the "Employee Name:" label), so they are not realistic fraud scenarios. Founder C's training-split tampered payslips were not used because the deployed model trained on those exact files.
- The API returns no flagged regions, so a demo cannot show where an edit is. The risk score is uncalibrated.
- Not production-ready; the 78% target is not met.

Tests: `tests/test_demo_dataset.py` + `tests/test_final_validation.py`: 7 passed.

## Day 8 — Final Integration & QA

### Founder C (Data)

- [x] Final dataset validation (read-only) — 75 indexed documents: 48 genuine / 27 tampered; 30 payslips / 23 bank statements / 22 ID proofs. Every indexed path exists, no duplicate filenames or paths, no byte-identical images, all PNGs readable, labels match folders, doc types match filenames, tamper methods valid. The 5 Day 1 payslips remain present and outside the index; nothing unexpected in `data/synthetic/`
- [x] `pytest tests/test_dataset.py -q` — 17 passed; `scripts/dataset_stats.py` output matches `DATASET.md` (9/9 tables)
- [x] Demo set checked read-only (Founder A's `data/demo/`): 5 readable images; checksums, sizes, dimensions and recorded predictions match `demo_manifest.json`; manifest model hash matches `fraud_model.pkl`. Two demo images are byte copies of Day 1 payslips `payslip_002.png` / `payslip_003.png`
- [x] Documentation verified against the current code; `README.md` created; `ARCHITECTURE.md`, `DEPLOYMENT.md`, `DATASET.md`, `DATA_FORMAT_NOTES.md` updated for the Day 6–7 additions (demo set, validation and statistics scripts) and the demo/dataset relationship
- [x] No dataset, demo, model, code or test files modified

**Full test suite (`python -m pytest -q`, from the repository root at `a555e07`):**

256 collected — **252 passed, 1 failed, 2 skipped, 1 xfailed, 0 errors**.

The single failure is `tests/test_demo_dataset.py::test_hashes_match_record_and_source` (Founder A's Day 7 demo test): `FileNotFoundError` for `data/augmented/train_v2_1/tampered/payslip_002_aug1.png`. The test re-hashes each demo image's source, and three sources are under `data/augmented/train_v2_1/`, which is git-ignored, so they are absent from a fresh clone. This is a known Founder A demo-dataset issue; Founder C did **not** modify or fix it. The other 3 demo tests pass.

Environment note: the bare `pytest` command in the QA environment is an isolated tool install without the project's dependencies (`ModuleNotFoundError: No module named 'fastapi'`, no tests collected). Use `python -m pytest` with the project interpreter.

**Open items for other owners (reported, not changed by Founder C):**
- `tests/test_demo_dataset.py` depends on git-ignored augmented images (Founder A)
- `API_DOCS.md` still describes 15 features, a "calibrated" score and bounding-box `flagged_regions` (Founder B)
- `.env.example` still has `MODEL_PATH=models/model.pt` (Founder B)

The 75-document synthetic dataset remains validated and unchanged. Not production-ready; see `README.md` and `DEPLOYMENT.md` §10.

## Day 7 — Final Code Cleanup & README

### Founder B (API) - Day 7

- [x] Code cleanup completed
- [x] Refactoring completed
- [x] README.md created
- [x] Documentation verified
- [x] Project structure validated
- [x] Release notes generated

**Refactoring report**

| Area | Change |
|---|---|
| Removed test-only shortcut (behaviour fix) | `ModelAPI.predict()` returned a canned "Document verified successfully" result for any upload whose bytes were exactly `fake image bytes for verification`, without running the model. Removed; those bytes are now decoded like any upload and rejected as corrupted. `test_verify_success` now uploads a real fixture image; the old stub test is replaced by `test_model_api_has_no_magic_bytes_shortcut`. |
| Configuration | New in `backend/config.py`: `APP_VERSION`, `ALLOWED_EXTENSIONS`, `TAMPER_THRESHOLD` (env, default 0.5), `CORS_ORIGINS` (env, default `*`). Removed the hardcoded `"0.1.0"` (×4), `0.5` threshold, duplicated extension set and `allow_origins=["*"]`. |
| Duplicate code | The 413 message was built in two places; now `error_handlers.file_size_exceeded_message()`. |
| Dead code | Removed unused `ALLOWED_MIME_TYPES` (MIME types were never checked), unused `get_model_api()`, 43 unused imports / empty f-strings, and 10 unused variables in `scripts/` (generator output verified byte-identical before/after on 12 documents). |
| Type hints & docstrings | Every function in `backend/` has a return type; all `backend/` functions are documented (class docstrings cover `__init__`). Added docstrings to the 7 undocumented `core/` functions. |
| Logging | A feature-count mismatch between the model and extractor now logs a warning instead of truncating silently. |
| Formatting & lint | Added `ruff.toml`; `backend/` formatted with `ruff format`; `ruff check .` passes. `core/` and `scripts/` formatting left as-is to avoid churn in other founders' code. |
| Accuracy of docs | OpenAPI text and `API_DOCS.md` no longer claim 15 features, a calibrated score, bounding-box `flagged_regions` or "Production Ready"; `threshold` documented as ignored. `.env.example` fixed (`MODEL_PATH=models/model.pt` → `data/models/fraud_model.pkl`, new variables added). `DEPLOYMENT.md` updated for the new variables. The load-test report template no longer says "15 forensic features". |
| Tests | `tests/test_demo_dataset.py::test_hashes_match_record_and_source` failed on every fresh clone because 3 sources are git-ignored. Demo-image vs manifest hashes are still always checked; the source comparison runs only when the source file exists locally. |

No model, dataset, feature-extraction or prediction change: the demo payslip `genuine_payslip_01.png` scores 0.36 before and after, matching Founder A's Day 7 record.

**Project structure validation**

| Required | Status |
|---|---|
| `core/`, `backend/`, `scripts/`, `tests/`, `data/`, `logs/` | Present |
| `README.md` | Present — rewritten with overview, features, architecture, install, env setup, run, endpoints, curl/PowerShell/Python examples, real responses, testing, load testing, structure, troubleshooting, contributors |
| `API_DOCS.md`, `DEPLOYMENT.md`, `ARCHITECTURE.md`, `PROGRESS.md` | Present |
| Additional | `DATASET.md`, `DATA_FORMAT_NOTES.md`, `data/demo/DEMO_IMAGES.md`, `.env.example`, `ruff.toml`, `logs/release_notes_v1.txt` |
| Missing | None |

**Final validation checklist**

- [x] API starts with `uvicorn backend.main:app`; model loads at startup
- [x] Real-server curl checks: `/health` 200, `/verify` 200 (demo payslip), no key 401, wrong key 403, `.md` upload 400, `/docs` 200
- [x] All tests pass: `python -m pytest -q` → 253 passed, 2 skipped, 1 xfailed, 0 failed
- [x] `python -m pytest tests/test_api_comprehensive.py -v` → 31 passed
- [x] No import errors: every module in `backend/`, `core/`, `scripts/` imports
- [x] No linting issues: `ruff check .` passes; `ruff format --check backend` passes
- [x] No broken references: every relative link and anchor in the Markdown docs resolves
- [x] Documentation exists and matches the code (feature count, score semantics, `flagged_regions`, `threshold`, config variables)

**Cleanup recommendations (not done — outside Day 7 scope or owned by other founders)**

1. Implement or remove the `threshold` query parameter (API change; needs frontend agreement).
2. Wire `ModelAPI.extract_flagged_regions()` into `/verify`, or remove it, once a localisation approach is chosen (Founder A).
3. Replace `print()` in `core/model_trainer.py` (e.g. the `Model loaded:` startup line) with `logging`.
4. Add docstrings to the 79 undocumented helper functions in `scripts/` and finish type hints there (21 functions lack return types).
5. Pin dependency versions compatible with `fraud_model.pkl`; add CI running `ruff check .` and `python -m pytest`.
6. Replace the single shared API key with hashed per-client keys and constant-time comparison.
7. Starlette warns that `TestClient` over `httpx` is deprecated; revisit when upgrading FastAPI.

**Deliverables:** `README.md`, `logs/release_notes_v1.txt`, `ruff.toml`, updated `API_DOCS.md`, `DEPLOYMENT.md`, `.env.example`.

## Day 8 — End-to-End Testing

### Founder B (API) - Day 8

- [x] E2E testing completed
- [x] Demo images validated
- [x] Environment validation completed
- [x] Stability verified
- [x] Reports generated

**New scripts**
- `scripts/e2e_test.py` starts a real uvicorn server and tests it over HTTP only: authentication, input validation, all 5 demo images and a 20-request reliability run. For each demo image it also recomputes ELA, the 22 features and the model score in-process and requires the API's score to match exactly. Exits 0 only if every check passes.
- `scripts/validate_environment.py` checks Python, every package in `requirements.txt`, required files, the model (loads, 22 features, SHA-256 matches the demo manifest), configuration, and a live `/health` + `/verify`. `--fresh-clone` first clones the repository from GitHub into an empty directory, creates a new virtualenv, installs requirements with `pip --no-cache-dir`, and runs the checks and the full test suite there.

**1. Fresh environment test** (`logs/environment_validation.txt`): **PASS**
- `git clone` → new venv → `pip install --no-cache-dir -r requirements.txt` (25 s) → 37/37 checks → full suite **253 passed, 2 skipped, 1 xfailed, 0 failed** inside the clean clone. No setup issues.
- Server starts and answers `/health` in 1.7–4.1 s (includes model load); `/verify` on a demo image returns 200 with score 0.36.

**2. End-to-end workflow** (`logs/e2e_test_report.txt`): **PASS, 65/65 checks**

| Stage | Result |
|---|---|
| Server start-up | `/health` 200, `/docs` served |
| Authentication | no key 401, wrong key 403, valid key 200, `/health` public |
| Input validation | empty 400, `.txt` 400, no extension 400, corrupted 400, 10 MB + 1 byte 413, `threshold=5` 422, no file 400, unknown route 404, `GET /verify` 405 |
| ELA → features → model | For all 5 demo images: ELA heatmap produced, 22 finite features, in-process score identical to the API's |
| JSON response | exactly `tamper_risk_score`, `confidence`, `flagged_regions`, `interpretation`, values in range |

**3. Demo images** (all 5 through the API):

| File | Actual | Predicted | Risk | Confidence | Latency |
|---|---|---|---|---|---|
| genuine_payslip_01.png | genuine | genuine | 0.36 | 0.64 | 262 ms |
| genuine_payslip_02.png | genuine | tampered | 0.59 | 0.59 | 228 ms |
| tampered_payslip_01.png | tampered | genuine | 0.37 | 0.63 | 399 ms |
| tampered_payslip_02.png | tampered | genuine | 0.14 | 0.86 | 271 ms |
| edge_case_01.png | genuine | genuine | 0.35 | 0.65 | 260 ms |

All 5 scores are identical to Founder A's Day 7 manifest. 2/5 are classified correctly — the same result as Day 7, and a demonstration record, not an accuracy estimate (held-out accuracy remains 60%).

**4. Reliability** (20 consecutive `/verify` requests, the 5 demo images cycled 4 times):
- 20/20 HTTP 200, no crashes (server still healthy afterwards)
- Predictions consistent: every repeat of an image returned exactly the same score
- Memory: server RSS 233.4 MB → 233.5 MB after 5 requests (warm-up) → 234.4 MB after 20 (+0.8 MB); no leak detected. Across three runs the post-warm-up change was +0.2 MB, −19.4 MB and +0.8 MB
- Latency: avg 276 ms, median 274 ms, p95 306 ms, max 317 ms

**5. Cross-machine validation:** step-by-step instructions for macOS/Linux and Windows (setup, the two validation scripts, manual curl check, expected demo scores, machine-specific fixes) are in `DEPLOYMENT.md` §11. The expected demo scores give a quick way to spot a dependency difference between machines.

**Fixed during Day 8**
- `pytest` from the repository root collected `scripts/e2e_test.py` (it matches pytest's `*_test.py` pattern) and reported 4 errors. `pytest.ini` now sets `testpaths = tests`.
- README now states Python 3.11+ (the versions actually verified) instead of 3.10+.

**Scope notes**
- All runs were on one Linux x86_64 cloud container (Python 3.11.15, 4 CPUs). The fresh-clone test rules out working-copy problems on that machine type; Windows and macOS were not run on Day 8 — use DEPLOYMENT.md §11 on those machines.
- Latency here (~0.3 s) is lower than the Day 5 in-process figure (0.94 s at 5 concurrent users) because these requests are sequential and the hardware differs.
- Tested against the Day 7 branch (`claude/peaceful-galileo-g4625a`), which is not yet merged into `main`.

## Day 8 — Model Stability Validation

### Founder A (Model) - Day 8

> Record for model **v1**. Later on Day 8 the model was replaced by **v2** to reach held-out accuracy above 75% (next section). The two scripts below now follow the deployed model's feature schema, and the reports in `logs/` describe v2; the v1 numbers here remain the v1 record.

- [x] Demo images validated
- [x] Stability testing completed
- [x] Accuracy revalidated
- [x] Model consistency verified
- [x] Validation reports generated

Run on 2026-09-27 against `main` at `0d549f7` (after the Day 10 release). Python 3.11.15, scikit-learn 1.9.1, numpy 2.4.6, opencv-python 5.0.0.93, Pillow 12.3.0, Linux x86_64, 4 CPUs. Model under test: `data/models/fraud_model.pkl` (v1.0.0, schema `day4-v2-22`, SHA-256 `bcf4d02f15980f68…`, the same file as Day 7). No model, dataset, demo image, threshold or API change.

**New scripts and reports**
- `scripts/model_stability_test.py` → `logs/model_stability_report.txt` (+ `logs/model_stability_results.json`). Five gating checks (repeatability, execution paths, feature extraction, ELA, drift against the Day-7 record) and two informational analyses (input sensitivity, retraining seed). Exit 0 when the gating checks pass.
- `scripts/model_validation.py` → `logs/model_validation_report.txt` (+ `logs/model_validation_results.json`). Model integrity, demo validation, stability gate, accuracy revalidation on the frozen test set, inference robustness, final checklist. Exits 1 while any checklist item fails — currently the demo item.
- `tests/test_model_stability.py` — 8 tests (helpers, drift detection, Day-6 report parsing, one real-model repeatability check).
- Full suite: `python -m pytest -q` → **290 passed, 2 skipped, 1 xfailed, 0 failed** (282 before + 8 new); `ruff check .` passes.
- Docs: the new scripts and reports are listed in `README.md`, `ARCHITECTURE.md` (§3, §10) and `DEPLOYMENT.md`; the README and DEMO_GUIDE test counts are updated to 290.

**1. Stability — PASS** (10 runs per demo image through `ModelAPI.predict(bytes)`, the call `/verify` makes)

| Image | Label (all 10 runs) | Risk score | Confidence | Max deviation |
|---|---|---|---|---|
| genuine_payslip_01.png | genuine | 0.36 | 0.64 | 0.0000 |
| genuine_payslip_02.png | tampered | 0.59 | 0.59 | 0.0000 |
| tampered_payslip_01.png | genuine | 0.37 | 0.63 | 0.0000 |
| tampered_payslip_02.png | genuine | 0.14 | 0.86 | 0.0000 |
| edge_case_01.png | genuine | 0.35 | 0.65 | 0.0000 |

- All 50 runs returned identical scores and confidences, so they are well within the ±2% tolerance. The label never changed.
- Execution paths: the same numbers came back from a file path, a freshly loaded model, the `/verify` route (in-process TestClient), 4 concurrent threads on one shared `ModelAPI`, and a separate Python process (35 extra calls).
- Feature extraction: all 22 features were bitwise identical over 10 runs and after a lossless PNG re-encode. ELA: heatmap SHA-256 and ELA score identical over 10 runs, and the ELA features in the vector match the heatmap.
- Drift: the model SHA-256 is unchanged since Day 7, and all 5 scores equal the Day-7 manifest (Δ 0.0000).
- Reproducibility: refitting the recorded configuration (seed 42) on the same 75 images reproduces the deployed file's probabilities exactly.

**2. Demo validation — 2/5 correct** (unchanged from Day 7)

| Image Name | Expected Result | Actual Result | Confidence | Risk Score | Pass/Fail |
|---|---|---|---|---|---|
| genuine_payslip_01.png | genuine | genuine | 0.64 | 0.36 | PASS |
| genuine_payslip_02.png | genuine | tampered | 0.59 | 0.59 | FAIL |
| tampered_payslip_01.png | tampered | genuine | 0.63 | 0.37 | FAIL |
| tampered_payslip_02.png | tampered | genuine | 0.86 | 0.14 | FAIL |
| edge_case_01.png | genuine | genuine | 0.65 | 0.35 | PASS |

The two missed edits cover 0.07% and 0.38% of the page, while all 22 features are whole-image statistics.

**3. Accuracy revalidation — unchanged** (frozen 15-image test set, Day-6 held-out method: the deployed configuration fitted in memory on the 60 TRAIN images)
- Accuracy 60.00% (9/15), precision 44.44%, recall 80.00%, F1 0.5714. TN 5, FP 5, FN 1, TP 4; FPR 50%.
- 95% Wilson intervals: accuracy 35.8–80.2%, precision 18.9–73.3%, recall 37.6–96.4%.
- Identical to Day 5 (confusion matrix) and to Day 6 (all 15 per-image scores). Test-set integrity 8/8 checks pass.
- The "~80%" figure is the Day-3 result: 4 of 5 test images, from a 15-feature model trained on 25 documents. It is not a measurement of this model.
- The frozen test set was only re-scored. Nothing was trained, tuned or selected on it.

**4. Unstable outputs (informational; not caused by nondeterminism)**
- Input sensitivity: 12 of 30 content-preserving variants change the label (JPEG re-save at q=95 and q=75, a 2-px crop, a 98% resize, ±3% brightness). A −3% brightness change alone flips 4 of 5 images. In those cases the largest feature shifts are in the page-wide ELA statistics (`ela_high_ratio`, `ela_p95`, `ela_mean`).
- Retraining seed: with seeds 0–9 the demo scores move by up to 0.13. genuine_payslip_02 (0.59 in the deployed model) is labelled tampered by only 4 of 10 other seeds. With 500 trees the seed noise drops from a mean std of 0.029 to 0.017.
- Near the threshold: genuine_payslip_02 at 0.59.

**5. Final model checklist — 5/6**
- [x] Model loads successfully
- [ ] All demo images predict correctly — 2/5
- [x] Scores stable within ±2% — 50/50 runs identical
- [x] No corrupted model files — SHA-256 matches the record, 100/100 trees, refit reproduces the file
- [x] No inference crashes — 0 in 65 `ModelAPI` calls. Corrupt, truncated, empty and non-image uploads are rejected with a `ValueError` (HTTP 400).
- [x] Metrics generated

**Verdict:** the model is deterministic and reproducible, but not accurate enough: 60% held-out against the 78% target, half of the genuine test documents flagged, and 2/5 demo images correct. It is **not production-ready**. The Day-10 release verdict (ready for the live demo, not for fraud decisions) is unchanged.

**Suggested improvements** (full lists in both reports)
1. Accuracy is the blocker, not stability. Move to region-level features: the Day-5 local-forensics representation scored 73.3% in train-only CV. Choose the model on TRAIN only.
2. Collect fresh held-out data (100+ documents). The frozen 15 have now been scored on Day 5, 6 and 8, so they must not be used to choose between new models.
3. Set the threshold on validation data and add a "needs review" band around it. genuine_payslip_02 shows why a hard cut at 0.5 is fragile.
4. Train with re-saved, resized and exposure-varied copies of TRAIN images, and prefer local ELA measures over page-wide ones.
5. Pin scikit-learn, numpy, OpenCV and Pillow. Record the model SHA-256 per release, and re-run both Day 8 scripts after any dependency or model change.

## Day 8 — Model v2: Held-out Accuracy Above 75%

### Founder A (Model) - Day 8 (model v2)

- [x] Held-out accuracy above 75%: **86.67% (13/15)** on the frozen test set and **92.62% (138/149)** on a fresh holdout set
- [x] New local tamper features: `core/tamper_features.py`, schema `local-tamper-v2`, 29 features
- [x] Model v2 trained without any test or holdout document; its training files and SHA-256 hashes are recorded in the model file
- [x] API, scripts, demo record, tests and docs updated for model v2; v1 kept for rollback
- [x] Stability re-verified on model v2

**Why v1 was stuck at 60%.** The dataset's edits are small local rectangles (noise patch, colour shift, blur, white box with text), but all 22 v1 features are whole-page statistics, which barely move when a small rectangle is edited.

**Method (the configuration was fixed before any held-out document was scored)**
1. Studied the generator's four edit types on TRAIN images only. The 15 frozen test images were not opened.
2. Generated a DEV set of 150 new documents (seed 7001, numbered 1001+) with `scripts/generate_extra_documents.py`. It uses Founder C's generator unmodified, via the safe workflow in `DATA_FORMAT_NOTES.md` §6. Before using it, I checked that this machine reproduces `data/synthetic/` documents 026–075 byte for byte (seed 4), so new documents follow the dataset's distribution. There were 0 duplicates or near-copies.
3. Designed the features by repeated 5-fold cross-validation on TRAIN (60) + DEV (150) only. Each feature measures the largest region of the page whose noise, paper colour, sharpness or ink darkness differs from the rest of that same page. Every feature family helps: removing noise costs 13 points, ink 6, colour 6.
4. Found and fixed a robustness problem in cross-validation. A model trained on pristine PNGs flagged **every** genuine page re-saved as JPEG q75. Adding 3 processed copies of every training document (JPEG, resize, brightness, contrast), applied equally to both classes, cut false alarms on processed genuine copies from 69% to 18%, with no loss on pristine files.
5. Froze and trained the model with `scripts/train_tamper_model.py` (RandomForest, 300 trees, class_weight=balanced, seed 42, threshold 0.5 unchanged). Grouped cross-validation gave **91.3% ± 3.9%** on pristine files (2.1% of genuine flagged) and 75.2% on processed copies (`logs/training_log_v2.txt`).
6. Only then generated the HOLDOUT set: 149 documents (seed 7002, numbered 2001+). One ID card that differed from a training image in only 0.38% of pixels was excluded by the near-copy rule, before scoring. Then scored TEST and HOLDOUT once with the frozen file (`scripts/model_validation.py`).

**Held-out results (the deployed file itself, through `ModelAPI`)**

| Set | n | Accuracy (95% CI) | Precision | Recall | F1 | Genuine flagged | Tampered missed |
|---|---|---|---|---|---|---|---|
| Frozen TEST | 15 | **86.67%** (62.1–96.3%) | 71.43% | 100.00% | 0.833 | 2/10 | 0/5 |
| HOLDOUT | 149 | **92.62%** (87.3–95.8%) | 98.25% | 84.85% | 0.911 | 1/83 | 10/66 |
| Combined | 164 | 92.07% (86.9–95.3%) | 95.31% | 85.92% | 0.904 | 3/93 | 10/71 |

Leakage checks pass: no TEST or HOLDOUT image is among the 210 recorded training files, HOLDOUT was created after the model, and the frozen split is intact (8/8). The v1 file had been trained on TEST, so v1 could only be measured through a replica. v2 is measured directly.

On the same 15 test images, accuracy goes from **60.00% (v1) to 86.67% (v2)**, +26.67 points. The two test errors are genuine bank statements scored 0.58 and 0.50; the second sits exactly on the threshold, which I did not change. Of the 10 holdout misses:
- 3 are blurs on blank paper that changed **no pixel**, so they are undetectable;
- 2 are white boxes on white paper, which leave only the inserted words;
- 5 are visible edits the model missed (2 colour shifts, 3 text overlays on bank statements).

**Demo set: 4/5 correct (v1: 2/5).** I re-scored it once with `scripts/prepare_demo_dataset.py --rescore`; the v1 scores are kept in `scoring_history`.

| Image Name | Expected Result | Actual Result | Confidence | Risk Score | Pass/Fail |
|---|---|---|---|---|---|
| genuine_payslip_01.png | genuine | genuine | 0.94 | 0.06 | PASS |
| genuine_payslip_02.png | genuine | genuine | 0.94 | 0.06 | PASS |
| tampered_payslip_01.png | tampered | tampered | 0.79 | 0.79 | PASS |
| tampered_payslip_02.png | tampered | genuine | 0.85 | 0.15 | FAIL |
| edge_case_01.png | genuine | genuine | 0.78 | 0.22 | PASS |

`tampered_payslip_02` uses the Day-5 augmenter's text replacement, an edit type v2 was not trained on.

**Stability of model v2: PASS** (`logs/model_stability_report.txt`)
- All 50 repeat runs returned identical scores. Results are also identical across a file path, a reloaded model, the `/verify` route, threads and a separate process. The 29 features are bitwise stable, and there is no drift against the demo record.
- Rebuilding the training data from scratch and refitting with seed 42 **reproduces the deployed file's probabilities exactly**.
- Input sensitivity: 8 of 30 content-preserving variants flip the label (v1: 12/30). A **2-px crop flips 4 of 5 demo images**, because cropping was not among the training processing operations. JPEG re-save or a 98% resize makes the model miss `tampered_payslip_01`.
- Retraining seed: no demo label depends on the seed (v1: 1/5); mean score std 0.016.

**Final model checklist: 6/7** (`logs/model_validation_report.txt`)
- [x] Model loads successfully
- [ ] All demo images predict correctly — 4/5
- [x] Scores stable within ±2% — 50/50 runs identical
- [x] No corrupted model files — SHA-256 matches the demo record, schema and structure are valid, and all 210 recorded training files are unchanged
- [x] No inference crashes — 0 in 229 `ModelAPI` calls; invalid files are rejected with HTTP 400
- [x] Metrics generated — TEST and HOLDOUT, leakage checks pass
- [x] Held-out accuracy above 75% — TEST 86.67%, HOLDOUT 92.62%

**Integration**
- `core/feature_schemas.py` maps the `feature_schema_version` recorded in a model file to its extractor. `ModelAPI` uses it and refuses an unknown schema or a feature-count mismatch (the same "expects N features" error as before). `DocumentFraudDetector.predict` follows the model's schema.
- Model v1 is kept byte for byte as `data/models/fraud_model_v1.pkl`. To roll back, set `MODEL_PATH` to it. `tests/test_local_forensics.py::test_day5_model_preserved` now checks that file's SHA-256.
- Made schema-aware: `scripts/e2e_test.py`, `scripts/validate_environment.py`, `scripts/performance_review.py`, `scripts/model_stability_test.py`, `scripts/model_validation.py`.
- `scripts/final_validation.py` now validates model v1 only and exits with code 3 for other models; `logs/final_validation_report.txt` remains the v1 record.
- Also updated: `scripts/run_demo.py` text, the OpenAPI description in `backend/main.py`, and `tests/test_health.py` (reads the version from the model file). `requirements.txt` now declares `scipy`.
- Checks on model v2: end-to-end over real HTTP **65/65 PASS** and environment validation **38/38 PASS** (both run to scratch, so Founder B's dated reports are unchanged). New tests in `tests/test_tamper_model.py` (12). Full suite: `python -m pytest -q` → **302 passed, 2 skipped, 1 xfailed, 0 failed**; `ruff check .` passes.
- Docs updated for v2: `README.md`, `ARCHITECTURE.md`, `DEPLOYMENT.md`, `API_DOCS.md`, `DEMO_GUIDE.md`, `DATA_FORMAT_NOTES.md` (new `data/generated/` sets), `data/demo/DEMO_IMAGES.md`.

**Not re-run with v2:** the load tests and performance review (Founder B's figures are for v1). Feature extraction takes about 120 ms per page for v2, against about 200 ms for v1, and end-to-end `/verify` latency was about 0.33 s.

**Limitations**
- Synthetic data from one generator only. Real documents have not been tested.
- Re-saved, resized and cropped files are handled less reliably: about 75% in cross-validation on processed copies, and a 2-px crop flips 4/5 demo images.
- Non-document images (photos, blank pages) still get a score, often above 0.5.
- TEST and HOLDOUT have now each been used once for v2. A future model needs a new holdout set (new seed).
- Scores are uncalibrated, and `flagged_regions` is still empty, although the v2 features already locate the anomalous region.

## Day 9 — Bug Fixes & Production Hardening

### Founder B (API) - Day 9

- [x] Code audit completed
- [x] Security review completed
- [x] Performance review completed
- [x] Bug fixes applied
- [x] Production hardening completed

**Code audit.** Reviewed `backend/`, the image/PDF loading path in `core/`, `tests/` and the scripts that touch the API or models, then sent 19 hostile and edge-case files through `/verify` and repeated the probe after fixing. **13 bugs fixed** (details and before/after table: `logs/bug_fix_report.txt`):

| # | Bug | Before | After |
|---|---|---|---|
| B1 | Tiny / very narrow images (1×1, 20000×10) | 500 (OpenCV crash) | 400 |
| B2 | Decompression bomb (140 KB PNG, 12000×12000) | accepted, 9.4 s, ~943 MB | 400 in 0.01 s |
| B3 | Huge / encrypted PDF pages | rendered without limit / unclear error | 400 before rendering |
| B4 | Browser CORS preflight | 401 — cross-origin frontends could not call `/verify` | 200 |
| B5 | File just under 10 MB | 413 (form framing counted) | accepted |
| B6 | Upload size checked only after the full body was received | — | rejected before parsing (live: 12 MB → 413 in 1 ms) |
| B7 | Server temp paths in 400 messages | leaked | uploaded name only |
| B8 | Exception text in 500 messages | leaked | generic + request ID |
| B9 | Missing / broken model | 500 on `/verify`, `/health` "ok", or crash at start-up | 503 + degraded `/health`, auto-recovers |
| B10 | Model with wrong feature count | silently truncated | refused at load |
| B11 | Default concurrency (4 threads) | 2.30 req/s at 4 clients | 2 slots: 3.47 req/s |
| B12 | Log injection via filename control characters | raw | escaped |
| B13 | `scripts/e2e_test.py --output` outside the repo | crash | works |

**Security review** (`logs/security_review.txt`): 26 items. 14 fixed, 6 checked and not vulnerable (including path traversal: the client filename only supplies the extension, and uploads go to random temp files that are always deleted), 3 accepted for the prototype (pickle model loading, public docs, operator-controlled `MODEL_PATH`), and 4 open before a real deployment (default-key policy, per-client keys, HTTPS and rate limiting, dependency pinning).

**Performance review** (`logs/performance_review.txt`, new `scripts/performance_review.py`, 4-CPU Linux):
- Start-up: uvicorn launch → `/health` 200 in 1.65 s (model load 3 ms); idle RSS 179 MB
- Prediction: 270 ms mean per `/verify` (feature extraction ~200 ms, model ~30 ms); the double image decode in `core/` is only ~8% of that, so it was left alone
- Large inputs: 4000×5000 JPEG 1.9 s, A4 PDF 1.1 s
- CPU: ~275 ms of CPU per request; throughput 3.7/s (1 client), 4.0/s (2), 3.8/s (4) with the new default of 2 slots
- Memory: peak 490 MB under concurrent load. The soak test showed it levels off (~450–500 MB) and stays at ~270–310 MB with `MALLOC_ARENA_MAX=2`: memory kept by the allocator per thread, not a leak. The 20-request E2E run shows +0.0 MB after warm-up.

**Production hardening**
- *Logging:* one format for `backend.*`/`core.*` loggers with a request ID on every line; one access-log line per request; `LOG_LEVEL` setting; `X-Request-ID` accepted from clients (validated) and returned on every response.
- *Startup validation:* logs the effective configuration; warns about the default or a short API key and `CORS_ORIGINS=*`; refuses a model whose feature count does not match the extractor.
- *Graceful error recovery:* a missing or broken model no longer stops the server; `/verify` returns 503 and retries the load on each request, so restoring the file recovers without a restart.
- *Health monitoring:* `/health` reports `model_loaded`, `model_version`, `uptime_seconds` and `detail`, and returns 503 when degraded, so it works as a load-balancer readiness check.
- *Limits:* upload size enforced before parsing; 40 MP pixel limit; 16 px minimum side; at most `MAX_CONCURRENT_ANALYSES` analyses at once.
- New settings: `MAX_IMAGE_PIXELS`, `MAX_CONCURRENT_ANALYSES`, `LOG_LEVEL` (in `config.py`, `.env.example` and the docs).

**Documentation review:** README, API_DOCS, DEPLOYMENT and ARCHITECTURE checked against the new behaviour and updated: new `/health` shape and 503, new 400/500/503 messages, request IDs, new settings, `MALLOC_ARENA_MAX`, middleware order, degraded start-up. Two stale ARCHITECTURE statements from before Day 7 (the MIME allow-list, the test shim) were also corrected. All relative links and anchors resolve.

**Tests:** 29 new (`tests/test_hardening.py` 28, `tests/test_health.py` +1). Full suite **282 passed, 2 skipped, 1 xfailed, 0 failed**; E2E 65/65. Four existing tests were updated for intentional behaviour changes (listed in the bug report); none were weakened. The one that asserted the 500 leak now asserts that the text is withheld.

**Not changed (owned by other founders):** the double decode and the `print()` start-up messages in `core/`.

### Founder C (Data/QA) - Day 9

- [x] Final synthetic dataset revalidated on latest `main` (`c2f3754`): `python -m pytest tests/test_dataset.py -q` — 17 passed
- [x] Dataset unchanged: 75 indexed documents — 48 genuine / 27 tampered; 30 payslips / 23 bank statements / 22 ID proofs; 5 Day 1 payslips outside the index; `data/` byte-identical to Day 8
- [x] `scripts/dataset_stats.py` still matches `DATASET.md` (9/9 tables; 0 missing, 0 unreadable, 0 mismatched files)
- [x] Day 8 demo-test issue confirmed resolved: `test_hashes_match_record_and_source` was changed by Founder B (commit `04f079f`, recorded in Founder B's Day 7 section, merged via PR #11 after Founder C's Day 8 QA) to compare sources only when present. `tests/test_demo_dataset.py`: 4 passed on current `main`. The augmented source images are still git-ignored
- [x] Documentation synchronized with latest `main`; stale Founder C statements corrected:
  - `DEPLOYMENT.md` header no longer claims runtime code and dependencies are unchanged since `8644f37`
  - `DEPLOYMENT.md` §8 generator warning now covers both overwrite and duplicate-append modes
  - `ARCHITECTURE.md` §10 demo-test note changed from open issue to resolved
- [x] Dataset statements in Founder B's updated `README.md`, `ARCHITECTURE.md` and `DEPLOYMENT.md` checked; all correct, no change needed. `DATASET.md` and `DATA_FORMAT_NOTES.md` unchanged
- [x] No code, test, dataset, demo or model files modified

Full test suite not re-run by Founder C on Day 9; the latest full-suite figures are Founder B's (above).

## Day 9 — Model Optimization & Final Review

### Founder A (Model) - Day 9

- [x] Model review completed
- [x] Feature importance analyzed
- [x] Robustness testing completed
- [x] ML reports generated
- [x] Release notes created

Model under review: `data/models/fraud_model.pkl`, v2.0.0, `local-tamper-v2`, SHA-256 `1d374648e967af72…`. The model file and threshold are unchanged. One serving optimization was applied with identical outputs (below).

**New scripts and reports**
- `scripts/feature_importance.py` → `logs/feature_importance_report.txt`. It measures MDI, out-of-fold permutation importance (grouped CV) and partial-dependence direction, and retrains without each feature family. It uses the model's training data only.
- `scripts/model_robustness_test.py` → `logs/model_robustness_report.txt`. It scores the 164 held-out documents under 13 degradations (low resolution, rotation, cropping, blur, JPEG re-save) through `ModelAPI`.
- `scripts/model_review.py` → `logs/model_review_report.txt`. It covers performance against earlier reports, prediction quality, speed, memory and size, forest alternatives (CV only) and the release checklist.
- `logs/model_release_notes.txt`: v2.0.0 release notes covering metrics, robustness, performance, limitations, failure cases, rollback and v3 plan.
- `tests/test_model_review.py`: 7 tests.

**1. Performance review** (the deployed file on held-out documents; identical to the Day 8 v2 record)

| Set | n | Accuracy (95% CI) | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|
| Frozen TEST | 15 | 86.67% (62.1–96.3%) | 71.43% | 100.00% | 0.833 | 20.0% | 0.0% |
| HOLDOUT | 149 | 92.62% (87.3–95.8%) | 98.25% | 84.85% | 0.911 | 1.2% | 15.2% |
| Combined | 164 | 92.07% (86.9–95.3%) | 95.31% | 85.92% | 0.904 | 3.2% | 14.1% |

- Previous reports on the same 15 test images: v1 60.00% (Day 5, 6 and 8) → v2 86.67% (Day 8 and Day 9).
- Weak classes:
  - Tampered is the weaker class: recall 85.9% against 96.8% specificity, so the model errs towards "genuine".
  - By document type, bank statements are weakest: 83.0%, against 96.4% for payslips and ID proofs.
  - By edit method, blur (8/11) and text overlay (14/19) are caught least often.
- False positives: 3 genuine bank statements (0.58, 0.50, 0.59).
- False negatives: 10.
  - 3 are blurs on blank paper that changed no pixel.
  - 2 are white-on-white overlays that left only the inserted words.
  - 5 are visible edits the model missed.

**2. Feature importance** (training data only)
- Retraining without a family costs: colour −7.6 points, noise −7.1, ink −5.2, softness −1.0.
- The top single feature is `colour_rare_blob_0_5pct` (permutation −4.8 points).
- The fingerprint family and 5 single features add nothing measurable. Removing them leaves CV accuracy at 0.924, so they are candidates to drop in v3.
- The ink family helps on pristine files but not on re-saved copies.

**3. Prediction quality**
- Genuine scores have a median of 0.08 (maximum 0.59); tampered scores have a median of 0.90.
- All 60 documents scoring 0.6 or more are tampered. The 0.2–0.6 band is mixed (9 of 30 tampered).
- 7 documents fall within 0.1 of the threshold, and 2 of them are correct.
- Edge cases:
  - Demo set: 4/5 correct.
  - Non-document fixtures score 0.55–0.62, so they are called tampered.
  - Invalid files are rejected with HTTP 400.

**4. Robustness** (164 held-out documents, 92.1% unmodified; 0 crashes in 2,366 predictions)

| Degradation | Accuracy |
|---|---|
| JPEG q85 re-save | 73.2% |
| Downscale 75% / 50% / 50% down-up | 75.0% / 50.0% / 54.9% |
| Rotate 90° / 180° / skew 1° / skew 3° | 74.4% / 64.0% / 80.5% / 70.1% |
| Crop 2 px / 5% / 10% | 79.9% / 84.8% / 86.6% |
| Blur radius 0.5 / 1.0 | 84.1% / 85.4% |

The model is reliable only on clean, upright, full-page renders. Bank statements are the most fragile type.

**5. Optimization**
- **Applied:** `ModelAPI` now runs the forest single-threaded (`n_jobs=1`) with one `predict_proba` pass per document.
  - Typical page: **297 → 136 ms** (2.2× faster).
  - Outputs are identical on 175 files (164 held-out, 5 demo, 6 fixtures).
  - Cause: the model file was saved with `n_jobs=-1`, so every single-document call started a thread pool. Forest time was 184 ms, and is now 21 ms.
- **Measured:**
  - Features take 113 of the 136 ms on a typical page; the colour and ink groups are the most expensive.
  - A 4000×5000 JPEG takes 1,055 ms, of which decoding and downscaling take 464 ms.
  - Memory: model load adds 12 MB (164 MB process after load); peak 289 MB.
  - Size: 5.5 MB, 300 trees.
- **Forest alternatives** (CV on training data only): 200 trees match 300 (0.924) at 3.6 MB; 100 trees give 0.905 at 1.8 MB.
- **Recommendations for v3:**
  - Train with rotation, skew, crop, downscale and blur copies, and normalise pages first.
  - Evaluate on a **new** holdout set, because TEST and HOLDOUT are now used.
  - Calibrate scores and add a "needs review" band.
  - Return `flagged_regions`.
  - Drop the features that add nothing.
  - Use Pillow draft-mode decoding for large JPEGs.
  - Validate on real documents.

**6. Release checklist: 6/6** (`logs/model_review_report.txt`)
- [x] Model file loads successfully (SHA-256 matches the demo record)
- [x] Metrics documented
- [x] Feature importance generated
- [x] Robustness tests completed
- [x] Demo images validated (5/5 scored and equal to the record; 4/5 correct)
- [x] No prediction crashes (0 in 2,545 predictions)

**Verdict:** the ML component is ready for demos and internal evaluation on clean synthetic documents. It is **not production-ready for real uploads**: it has never been evaluated on real documents, and accuracy falls to 50–87% on degraded pages.

Tests: `python -m pytest -q` → **309 passed, 2 skipped, 1 xfailed, 0 failed**; `ruff check .` passes. With the optimized `ModelAPI`, the end-to-end test over real HTTP gives **65/65 PASS** with an average `/verify` latency of 167 ms (about 330 ms on Day 8). Environment validation: **39/39 PASS**. Both were run to scratch, so Founder B's dated reports are unchanged. Docs updated: README (documentation table, limitations), ARCHITECTURE (scripts, single-document inference, §9–§10), DEPLOYMENT (tooling, model quality).

## Day 9 follow-up — `flagged_regions` (localisation)

### Founder A (Model) - localisation

`POST /verify` returned `"flagged_regions": []` for every document. For model v2 it now returns boxes around the anomalous regions.

**Implementation**
- `core/tamper_features.py`: every anomaly mask records its connected regions while the features are computed. Masks cover noise, rare or new paper colour and softness on 8×8 px tiles, plus runs of glyphs with a rare ink darkness.
  - `extract_with_regions(path)` returns the features and the regions from one pass. `locate_regions(path)` returns only the regions.
  - Each region is kept only if it exceeds its mask's minimum size (`LOCATE_MASKS`, chosen on DEV only). Boxes that overlap or lie within 8 px are merged; the result is sorted by area and capped at 5.
  - Coordinates are rescaled from the analysed copy (longest side ≤ 2000 px) to the uploaded image's pixels. PDFs use page 1 at 200 DPI.
  - Region fields are `x`, `y`, `width`, `height`, `area` and `type`; `type` is `noise`, `colour`, `softness` or `ink`, joined by `+` after a merge. These match `FlaggedRegion` in `backend/main.py`.
  - `extract_tamper_features` output is unchanged (identical on 167 files, and asserted in the tests), so the model file and its SHA-256 records stay valid. Cost: about 3 ms per page.
- `core/feature_schemas.py`: a schema may provide `extract_with_regions`. Only `local-tamper-v2` does; for v1 (`day4-v2-22`) it is `None`.
- `backend/model_utils.py`: regions are returned only when `tamper_risk_score >= TAMPER_THRESHOLD`. Otherwise the response would show boxes on a document reported as genuine: 9 of 93 held-out genuine pages have a box before gating, 2 after. The interpretation then adds "N suspicious region(s) located (see flagged_regions)." v1 models still return `[]`.

**Accuracy** (`scripts/localization_eval.py` → `logs/localization_report.txt`). True edit rectangles come from replaying the generator (`edit_footprints`). A hit is a returned box that intersects the edit.

| Set | Visible edits | Hit | Top-1 | Box precision | Genuine with boxes |
|---|---|---|---|---|---|
| DEV (thresholds chosen here) | 59 | 49 (83.1%) | 49 | 54/54 | 0/87 |
| Held-out: HOLDOUT + TEST, measured once | 67 | 55 (82.1%) | 52 | 61/69 | 2/93 |

- Held-out by method: noise patch 28/28, blur 7/7, colour shift 10/13, text overlay 10/19. Text overlays on bank statements are the main miss (7), because the inserted words share the page's ink darkness.
- HOLDOUT and the frozen TEST set were not used to choose anything.
- The demo image `tampered_payslip_01.png` gets an `ink` box at (198, 321), 96 × 9 px, over the replaced account number (recorded edit `[197, 321, 55, 12]`).
- **Limitation:** boxes assume the training resolution, as the score does. On DEV pages resampled to 0.75×, 1.5× or 2.5×, every genuine page gets locator boxes, and the model flags 4, 55 and 86 of 87 genuine pages. Normalising page resolution is v3 work.

**Tests and tools**
- `tests/test_localization.py` (13 tests):
  - generator edits on a TRAIN page are boxed;
  - a genuine page gets `[]`;
  - the features are unchanged;
  - coordinates map to the uploaded image's pixels, and PDF sizes are right;
  - merge, scale and limit rules;
  - the demo ink box;
  - v2 regions only when flagged, v1 never;
  - `/verify` returns valid `FlaggedRegion` items.
- `scripts/e2e_test.py` validates region structure. It fails if regions appear on a document reported as genuine.
- `scripts/model_validation.py` no longer requires `flagged_regions == []`. It checks that boxes lie inside the page and appear only for a tampering verdict.
- `scripts/run_demo.py` prints each located region under its document.
- Docs updated: README (response fields, example, limitations, status), ARCHITECTURE §3, §4, §8, §9, §10, API_DOCS (region fields), DEMO_GUIDE, DEPLOYMENT, and the release notes (`logs/model_release_notes.txt` §10).

Checks:
- `python -m pytest -q`: **322 passed, 2 skipped, 1 xfailed, 0 failed** (309 before, plus 13 new). `ruff check .` passes.
- `python scripts/e2e_test.py` over real HTTP: **65/65 PASS**; `tampered_payslip_01.png` returns `flagged_regions: 1 (ink at 198,321 96x9)` and the other demo images return none. The run wrote to scratch, so Founder B's dated report is unchanged.
- `scripts/model_validation.py` was re-run. All figures are unchanged, and all 5 demo outputs pass the new region check.
- `logs/model_review_report.txt` is kept as the Day 9 record. Its script's localisation recommendation is updated for the next run.

## Day 10 — Final QA & Release

### Founder B (API) - Day 10

- [x] Final QA completed
- [x] All tests passing
- [x] Release validation completed
- [x] Documentation verified
- [x] Deployment ready (for demos and internal pilots, with the operator steps listed below; **not** for real fraud decisions)
- [x] Final report generated

**Release verdict** (`logs/final_release_report.txt`): the backend is **ready for the live demo and investor presentation** and for an internal pilot once the operator steps below are done. It is **not ready for production fraud decisions**: the model's held-out accuracy is 60% against a 78% target, on synthetic data only. The limiting factor is the model, not the API.

**1. Final API validation** (live server, HTTP): E2E **65/65** — `/health`, `/verify`, authentication, 9 validation/error cases, pipeline stages for all 5 demo images, 20-request stability run. Environment validation **38/38** locally and in a fresh clone.

**2. Final test execution:** `pytest -v` in a fresh clone's virtualenv → **282 passed, 2 skipped, 1 xfailed, 0 failed, 0 warnings**. Both remaining warnings were fixed at the source, not filtered: `httpx2` was added for Starlette's TestClient, and Pillow's decompression-bomb warning is now turned into the existing 400. The 2 skips and 1 xfail belong to Founder A's unfinished experimental local-forensics model. On this container the bare `pytest` on PATH is a separate tool install without the project's packages, so use the venv's `pytest` or `python -m pytest`.

**3. Final performance validation** (live uvicorn, over HTTP; `logs/final_load_report.txt`):

| Target | Measured | |
|---|---|---|
| `/verify` avg < 2.0 s at 5 users | 1.19 s (p95 1.62 s, max 1.74 s), 4.0 req/s | met |
| `/health` avg < 50 ms | 7.5 ms | met |
| Success > 95% | 100% | met |
| No 5xx under load | 0 | met |
| 10 / 20 users (no target) | 1.67 s / 3.23 s avg | 20 users over 2 s |
| Model accuracy 78% | 60% | **not met** |

`scripts/load_test.py` fix: its report printed fixed verdicts and numbers regardless of the run (e.g. "YES (0.939s achieved)", "99.99% availability", "memory < 45 MB"), and labelled the tester's own memory as the server's. Verdicts, observations and acceptance criteria are now computed from each run.

**4. Release checklist:** README, API_DOCS, DEPLOYMENT, ARCHITECTURE, DEMO_GUIDE and PROGRESS present and current; all Markdown links and anchors resolve; `ruff check .` passes; no secrets committed. README updated with the Day 10 load results against the targets and the current test counts (the Day 9 README still said 281).

**5. Demo readiness:**
- `scripts/run_demo.py` — one command: health → key refused → bad file refused → all 5 demo documents with score, verdict, correctness and latency. It shows 2/5 correct and the 60% accuracy figure on screen. Verified in the fresh clone.
- `DEMO_GUIDE.md` — demo execution guide (preparation checklist, three ways to run the demo, expected results, talk track, backup plan), API usage guide (curl, PowerShell, Python, JavaScript, response fields, input rules) and troubleshooting guide.

**Operator steps before any shared deployment:** set `API_KEY` and `CORS_ORIGINS`, put the service behind HTTPS, and set `MALLOC_ARENA_MAX=2`. **Open engineering items:** rate limiting, per-client keys, pinned dependencies with CI, a container image, and a multi-worker load test.

### Founder C (Data/QA) - Day 10

- [x] Final dataset validation completed on `main` at `3d9f551`
- [x] `python -m pytest tests/test_dataset.py -q` — 17 passed
- [x] `python scripts/dataset_stats.py` — 75 indexed documents: 48 genuine / 27 tampered; 30 payslips / 23 bank statements / 22 ID proofs; 5 Day 1 payslips outside the index; 0 missing files, 0 unreadable images, 0 metadata mismatches
- [x] All 9 statistics tables in `DATASET.md` match `scripts/dataset_stats.py` output
- [x] Data integrity checks passed: all 81 files in `data/synthetic/` byte-identical to the Day 7 baseline; `data/synthetic/`, `data/demo/` and `data/models/` (89 files) byte-identical to the Day 8 baseline; model SHA-256 matches `data/demo/demo_manifest.json`; no unexpected files under `data/`
- [x] No dataset, model or demo changes made on Day 10; the generator was not run
- [x] Required documentation verified present and checked against the code: `README.md`, `API_DOCS.md`, `ARCHITECTURE.md`, `DEPLOYMENT.md`, `DATASET.md`, `DATA_FORMAT_NOTES.md`, `PROGRESS.md` (endpoints, 22-feature count, model path, configuration settings, dataset counts, demo set, known limitations)
- [x] Stale Founder C statements corrected:
  - `DEPLOYMENT.md`: removed the note that parts of `API_DOCS.md` still describe a 15-feature model (no longer true)
  - `ARCHITECTURE.md` §8: `interpretation` now described as using the configurable `TAMPER_THRESHOLD` (default `0.5`) instead of a fixed `0.5` rule
- [x] Founder C's Day 8 open items are resolved — all by Founder B in commit `04f079f` (Founder B's Day 7 cleanup, merged via PR #11), not by Founder C:
  - `tests/test_demo_dataset.py` no longer fails when git-ignored augmented sources are absent
  - `API_DOCS.md` now describes 22 features, an uncalibrated score and an always-empty `flagged_regions`
  - `.env.example` now uses `MODEL_PATH=data/models/fraud_model.pkl`

Founder C did not re-run the full test suite on Day 10. The full-suite, E2E and load figures in the summary below are Founder B's recorded results.


### Founder A (Model) - Day 10

- [x] Final model validation completed
- [x] Demo images verified
- [x] Accuracy rechecked
- [x] Performance tested
- [x] Release report generated
- [x] Model approved for deployment: for the live demo, the investor presentation and an internal pilot on synthetic or non-decision documents. **Not** approved for automated fraud decisions on real documents (conditions in `logs/final_model_release_report.txt` §5).

Model: `data/models/fraud_model.pkl`, v2.0.0 (`local-tamper-v2`), SHA-256 `1d374648e967af72...`. Unchanged since Day 8.

**Scripts**
- `scripts/final_model_validation.py`: final validation, release review and sign-off. Writes `logs/final_model_release_report.txt`, `logs/model_signoff_checklist.txt` and `logs/final_model_release_results.json`.
- `scripts/model_accuracy_eval.py`: accuracy check. Writes `logs/final_accuracy_report.txt`.
- `scripts/inference_benchmark.py`: performance benchmark. Writes `logs/inference_benchmark_report.txt`.
- `logs/model_limitations.txt`: known limitations.
- `python scripts/final_model_validation.py --rerun` regenerates everything in one run (about 2.5 minutes).

**1. Model validation: 33/33 checks, 0 warnings**
- **Load:** the file loads through both loaders. Its SHA-256 matches the demo manifest and the release notes. The schema, 29 feature names, 300 fitted trees and class order are correct. The recorded training files are unchanged.
- **Dependencies:**
  - The installed scikit-learn, numpy, OpenCV, Pillow and SciPy match the versions recorded in the model file.
  - `pip check` is clean, and PDF, OpenCV, SciPy, FastAPI and uvicorn import.
  - **Found and fixed:** `requirements.txt` did not pin scikit-learn, although the model is a pickle. An environment that already had another version would have kept it and failed or warned at load. It is now pinned to `scikit-learn==1.9.1`.
- **Serialization:** unpickling raises no warnings, and the metadata is complete. A re-pickled copy gives identical probabilities. The rollback model v1 loads, and its SHA-256 matches the release notes.
- **Runtime:** these inputs are processed: PNG, JPEG, grayscale, RGBA, 80×80, 2400×1800 and an A4 PDF. Corrupt, truncated, empty and non-image files are rejected with a controlled error (HTTP 400). 0 crashes.

**2. Demo images** (10 runs each; identical over HTTP)

| File Name | Prediction | Confidence | Risk Score | Pass/Fail |
|---|---|---|---|---|
| genuine_payslip_01.png | genuine | 0.9433 | 0.0567 | PASS |
| genuine_payslip_02.png | genuine | 0.9433 | 0.0567 | PASS |
| tampered_payslip_01.png | tampered | 0.7933 | 0.7933 | PASS (ink region located) |
| tampered_payslip_02.png | genuine | 0.8467 | 0.1533 | FAIL (known: amount replaced + local resample, not in v2's training edits) |
| edge_case_01.png | genuine | 0.7767 | 0.2233 | PASS |

- **Consistent:** every score equals the Day 8 record, and every HTTP response equals the in-process output.
- **Stable:** the largest deviation over 10 runs is 0.0000.
- **Confidence:** always max(score, 1 − score), and none is below 0.6.

**3. Accuracy** (`logs/final_accuracy_report.txt`; nothing tuned)

| Set | n | Accuracy (95% CI) | Precision | Recall | F1 | TN/FP/FN/TP |
|---|---|---|---|---|---|---|
| Frozen TEST | 15 | 86.67% (62.1–96.3%) | 71.43% | 100.00% | 0.833 | 8/2/0/5 |
| HOLDOUT | 149 | 92.62% (87.3–95.8%) | 98.25% | 84.85% | 0.911 | 82/1/10/56 |
| Combined | 164 | 92.07% (86.9–95.3%) | 95.31% | 85.92% | 0.904 | 90/3/10/61 |

- The confusion matrices are identical to the Day 8 validation and the Day 9 review.
- On the same 15 TEST images, v1 scored 60.00% and v2 scores 86.67%.
- Requirements met: above 75% held-out and the 78% sprint target, on both sets.

**4. Performance** (`logs/inference_benchmark_report.txt`; 4-CPU container; targets fixed before measuring, 5/5 met)

| Measurement | Min | Average | Max |
|---|---|---|---|
| Model load (`ModelAPI.load_model`) | 12 ms | 14 ms | 15 ms |
| In-process `ModelAPI.predict`, 169 documents × 3 (507 predictions) | 75 ms | 124 ms | 267 ms |
| HTTP `/verify`, sequential (5 demo images × 10) | 148 ms | 171 ms | 311 ms |
| HTTP `/verify`, 5 concurrent users × 10 requests | 257 ms | 530 ms | 743 ms |
| 4000×5000 JPEG page | – | 1,131 ms | 1,459 ms |

- **Stability:** all 507 repeated predictions are identical, and the median run-to-run variation per document is 3.6%.
- **Load and memory:** 0 crashes; 100% success at 5 users (8.3 requests/s, v1: 1.19 s average); server peak memory 309 MB.
- Today's typical-page time is higher than Day 9's 136 ms. On this machine the Day 9 and current `tamper_features.py` time the same (138.7 vs 137.6 ms) and give identical features, so the difference is the machine, not the code.

**5. Release review: 16/16 checks**
- **Features:** the extractor is chosen from the model's schema. It gives 29 finite features in column order, deterministically, and the served extractor equals the training extractor.
- **ELA:** the deployed v2 does not use ELA; `ela_analysis` is called 0 times per v2 prediction. The ELA pipeline still works for the v1 rollback model, which reproduces its recorded demo scores.
- **Prediction logic:** API score and confidence equal the forest probabilities on independently extracted features. The verdict follows `score >= TAMPER_THRESHOLD`, and regions appear only for a tampering verdict.
- **Thresholds:** config, model metadata and demo manifest all use 0.5. Startup validation warns about an out-of-range threshold. The `threshold` query parameter has no effect, as documented.
- **Non-blocking issues:**
  - Non-document test images are reported as tampered (0.55–0.62).
  - All 3 held-out false positives are bank statements.
  - Founder B's Day 10 reports (`final_release_report.txt`, `final_load_report.txt`) describe model v1. They are left unchanged as dated records.

**6. Sign-off checklist: 6/6** (`logs/model_signoff_checklist.txt`)
- [x] Model file loads correctly
- [x] Demo images validated
- [x] Accuracy report generated
- [x] Performance validated
- [x] No critical bugs found
- [x] Release report completed

**Known limitations** (`logs/model_limitations.txt`):
- Synthetic data from one generator only.
- Accuracy falls to 50–87% on re-saved, downscaled, rotated, skewed or cropped pages.
- Bank statements are the weakest type (83.0%).
- Non-document images still get a score.
- Scores are uncalibrated.
- Replaced-text edits were never trained.
- Only PDF page 1 is analysed.
- At a realistic 1% fraud rate, only about 1 flag in 5 would be a real edit (arithmetic from the measured rates).

Checks:
- `python -m pytest -q`: **338 passed, 2 skipped, 1 xfailed, 0 failed, 0 warnings**. That is 322 before plus 16 in `tests/test_final_release.py`, which include a guard that the committed sign-off matches the model file.
- `python scripts/e2e_test.py` over real HTTP: 65/65 PASS (run to scratch).
- `ruff check .` passes.
- Docs updated: README (status, limitations, documentation table), ARCHITECTURE (scripts, tests, §9 throughput, §10), DEPLOYMENT (dependency pinning), DEMO_GUIDE and `logs/model_release_notes.txt` §8 and §11.

---

## Final Summary (end of sprint)

Compiled by Founder C on Day 10 from results recorded in this file and in `logs/`. Each result is attributed to the founder who produced it.

**Dataset (Founder C; re-verified on Day 10)**
- 75 indexed synthetic documents: 48 genuine / 27 tampered
- 30 payslips / 23 bank statements / 22 ID proofs
- 5 Day 1 payslips kept outside the index (2 of them copied into Founder A's `data/demo/`)
- `tests/test_dataset.py`: 17 passed; statistics in `DATASET.md`, reproducible with `scripts/dataset_stats.py`
- All documents are synthetic and watermarked; no real personal data

**Model (Founder A; `logs/final_validation_report.txt`)**
- Random Forest, 100 trees, 22 features (schema `day4-v2-22`), `data/models/fraud_model.pkl`
- Held-out accuracy **60% (9/15)** on the frozen test set, below the 78% target
- Validated on synthetic data only; `tamper_risk_score` is an uncalibrated vote fraction
- Documented limitation: the deployed model file was refit on all 75 documents, including the 15 test images, so it has no clean held-out measurement of its own; the 60% comes from a replica trained on the 60 training images only

**API / tests / E2E (Founder B; recorded in Founder B's Day 9–10 sections and `logs/`)**
- Full suite, reported by Founder B: 282 passed, 2 skipped, 1 xfailed, 0 failed
- End-to-end: 65/65 checks (`logs/e2e_test_report.txt`)
- Load test: `/verify` averages 1.19 s with 5 concurrent users (`logs/final_load_report.txt`)
- These figures were not independently re-run by Founder C

**Final status (Founder B's release verdict, `logs/final_release_report.txt`)**
- **Not production-ready.** Not suitable for real fraud decisions.
- Recorded as ready for a live demo, and for an internal pilot once the documented operator steps are done (set `API_KEY` and `CORS_ORIGINS`, run behind HTTPS, set `MALLOC_ARENA_MAX=2`)

**Known limitations**
- Model accuracy and data limits as above (60%, synthetic only, uncalibrated scores)
- The tamper decision uses the server-side `TAMPER_THRESHOLD` setting (default `0.5`); the `threshold` query parameter on `/verify` is validated but ignored
- `flagged_regions` is always empty; the API does not show where an edit is
- Not implemented: HTTPS/TLS, rate limiting, per-client API keys, pinned dependencies (listed as open engineering items by Founder B)
- Documentation wording discrepancy left unchanged in this Founder C task: `DEPLOYMENT.md` states "281 passed" after Day 9, while `README.md` and Founder B's Day 9–10 entries state 282

**Addendum (Founder A, Day 10, model v2).** The model figures in this summary describe model v1. The deployed model is now v2.0.0, signed off 6/6 on Day 10:
- **Accuracy:** 86.67% on the frozen TEST set and 92.62% on 149 held-out documents, above the 78% target.
- **Demo and speed:** 4/5 demo images correct, and `/verify` averages 0.5 s at 5 concurrent users.
- **Localisation:** `flagged_regions` returns boxes for flagged documents.
- **Dependencies:** scikit-learn is now pinned.
- **Status:** still not production-ready for real fraud decisions (synthetic data only).

See "Founder A (Model) - Day 10" above and `logs/final_model_release_report.txt`.
