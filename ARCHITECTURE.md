# Architecture — Document Fraud Detection Backend

> Describes the system as of Founder B's Day 9 hardening (request tracing, upload-size
> middleware, image-dimension guards, readiness-aware `/health`, degraded start-up).
> `core/` and the model file are unchanged since this document was first written at
> `8644f37`; offline/demo assets are listed in §10.
> Statements are marked **verified** only where they were checked against the code
> or by running it. This is a prototype: it is **not production-ready** and has not
> been evaluated on real-world documents.

---

## 1. Purpose and scope

The backend accepts an uploaded document image (payslip, bank statement or ID proof),
measures whether any part of the page behaves differently from the rest of the same page
(noise, paper colour, sharpness, ink darkness), and returns a tamper-risk score from a
Random Forest classifier (model v2, since Day 8).

In scope today:

- A FastAPI service with two endpoints: `GET /health` and `POST /verify`.
- An analysis engine that turns one image into the feature vector of the loaded model:
  29 local tamper features for model v2 (`local-tamper-v2`); the Day-4 22-feature
  extractor remains for the v1 rollback model.
- A synthetic-data pipeline that generates, indexes and validates a labelled
  training set, generated development / held-out sets, and scripts that train and
  validate the model.

Out of scope / not implemented: real-document evaluation, calibrated probabilities,
region-level localisation in API responses, multi-page PDF analysis, containerisation,
HTTPS, rate limiting and horizontal scaling.

---

## 2. High-level architecture

```
                         ┌──────────────────────────── backend/ ───────────────────────────┐
 Client ──HTTP──▶ API-key middleware ──▶ /verify handler ──▶ file validation ──▶ model_api │
                 (backend/main.py)       (backend/main.py)   (error_handlers.py)  .predict()│
                         └──────────────────────────────────────────────────────────┬──────┘
                                                                                    │
                ┌───────────────────────────── core/ ─────────────────────────────┐ │
                │ feature_schemas.get_schema(model metadata) → extractor           │ │
                │ tamper_features.extract_with_regions()     ◀── temporary file ◀──┘
                │   load_image → 8 px tile maps → 29 features → vector (+ regions) │
                │ model_trainer.DocumentFraudDetector.load()  (unpickles model)    │
                └─────────────────────────────────┬───────────────────────────────┘
                                                  ▼
                   data/models/fraud_model.pkl  (RandomForestClassifier, model v2)
                                                  ▼
                                         JSON response to client

 Offline (not part of the running service):
   scripts/generate_tampered_documents.py ─▶ data/synthetic/ ─▶ tests/test_dataset.py
   scripts/generate_extra_documents.py ─▶ data/generated/dev_v1/, holdout_v1/
   frozen TRAIN + dev_v1 ─▶ scripts/train_tamper_model.py ─▶ data/models/fraud_model.pkl
   frozen TEST + holdout_v1 ─▶ scripts/model_validation.py (held-out evaluation)
```

---

## 3. Main components

### `backend/` — HTTP service (Founder B)

| File | Role |
|---|---|
| `main.py` | FastAPI app, start-up validation and model loading (lifespan), middleware stack, `/health` and `/verify` routes, response schemas |
| `config.py` | All settings from the environment (and `.env` via python-dotenv): API key, model path, upload and image limits, concurrency, CORS, log level |
| `middleware.py` | `RequestContextMiddleware` (request IDs, access log) and `UploadSizeLimitMiddleware` (rejects oversized bodies before parsing) |
| `logging_config.py` | One log format for `backend.*` / `core.*` loggers, with the request ID on every line |
| `model_utils.py` | `ModelAPI` class and module-level `model_api` instance: model loading and validation, header-only dimension checks, the prediction pipeline |
| `error_handlers.py` | File-validation helpers and centralised JSON error responses |
| `exceptions.py` | Custom exception types mapped to HTTP status codes |

### `core/` — analysis engine (Founder A)

| File | Role | Used by the API? |
|---|---|---|
| `document_analyzer.py` | `load_image` (shared image loader), `ela_analysis`, `extract_features`, `FEATURE_NAMES` (22, model v1) | **Yes** (`load_image`; the 22 features only for a v1 model) |
| `tamper_features.py` | `extract_tamper_features`, `FEATURE_NAMES` (29, schema `local-tamper-v2`): model v2 features; `extract_with_regions` / `locate_regions`: the anomalous regions behind them (`flagged_regions`) | **Yes** |
| `feature_schemas.py` | Registry mapping a model's recorded `feature_schema_version` to its feature names and extractor | **Yes** |
| `model_trainer.py` | `DocumentFraudDetector`: v1 dataset loading, training, evaluation, `save()` / `load()` | **Yes** (`load()` only) |
| `forensic_model.py` | Experimental combined feature set (Day 5 research) | **No** |
| `local_forensics.py` | Experimental region-level forensics features (Day 5 research) | **No** |

`forensic_model.py` and `local_forensics.py` are imported only by
`scripts/ml_recovery.py`, `tests/test_local_forensics.py`, and each other
(`forensic_model.py` imports `local_forensics.py`). No file in `backend/` imports
either module, so they are **not wired into the API** and do not affect `/verify`
results (verified by searching imports).

### `scripts/` — offline tooling

| Script | Purpose |
|---|---|
| `generate_tampered_documents.py` | Synthetic document generator (Founder C). `--day2` built the original 25 documents; `--day4` appended documents 026–075 |
| `train_model.py` | Trains the Random Forest and saves it (defaults to overwriting `data/models/fraud_model.pkl`) |
| `inspect_features.py` | Feature statistics / audit |
| `augment_training_data.py`, `ml_recovery.py` | Day 5 experimental augmentation and cross-validation work |
| `load_test.py` | API load-testing tool |
| `generate_api_test_report.py` | Builds `logs/api_test_report.txt` from a pytest run |
| `dataset_stats.py` | Read-only dataset statistics (Founder C, Day 7); see §10 |
| `final_validation.py` | Frozen-test-set validation of model v1 (Founder A, Day 6); see §10 |
| `generate_extra_documents.py` | Generated development (`dev_v1`) and held-out (`holdout_v1`) sets (Founder A, Day 8); see §10 |
| `train_tamper_model.py` | Trains model v2 on frozen TRAIN + `dev_v1` with processed copies; grouped CV (Founder A, Day 8) |
| `feature_importance.py` | Feature importance of the deployed model on its training data (Founder A, Day 9); see §10 |
| `model_robustness_test.py` | Held-out accuracy under low resolution, rotation, cropping and blur (Founder A, Day 9); see §10 |
| `model_review.py` | Final model review: performance, prediction quality, optimisation, release checklist (Founder A, Day 9); see §10 |
| `localization_eval.py` | Measures `flagged_regions` against the true edit rectangles (Founder A); see §10 |
| `final_model_validation.py` | Final model validation, release review and sign-off (Founder A, Day 10); see §10 |
| `model_accuracy_eval.py` | Final held-out accuracy check with comparison to earlier metrics (Founder A, Day 10); see §10 |
| `inference_benchmark.py` | Inference benchmark: in-process and HTTP latency, stability, 5-user load (Founder A, Day 10); see §10 |
| `prepare_demo_dataset.py` | Builds and scores the demo set in `data/demo/` (Founder A, Day 7); see §10 |
| `model_stability_test.py` | Demo-image repeatability, feature/ELA consistency, drift and sensitivity (Founder A, Day 8); see §10 |
| `model_validation.py` | Demo validation, accuracy revalidation and final model checklist (Founder A, Day 8); see §10 |

### `data/`

| Path | Contents |
|---|---|
| `models/fraud_model.pkl` | The only model file; loaded by the API |
| `synthetic/genuine/`, `synthetic/tampered/` | 75 indexed PNG documents (48 genuine, 27 tampered) |
| `synthetic/index.csv` | One metadata row per indexed document |
| `synthetic/payslip_001.png` … `payslip_005.png` | Original Day 1 payslips, kept outside the index (used by `tests/test_analyzer.py`) |
| `augmented/…/manifest.csv`, `splits/day5_frozen_split.json` | Day 5 experiment metadata (augmented images themselves are git-ignored) |
| `demo/` | 5 demo images + manifest (Founder A, Day 7); not loaded by the API — see §10 |

### `tests/`

pytest suites for the analyzer (`test_analyzer.py`), model and trainer
(`test_model.py`, `test_model_trainer.py`), model utilities (`test_model_utils.py`),
API (`test_health.py`, `test_verify.py`, `test_error_handlers.py`,
`test_api_comprehensive.py`, `test_api_load.py`), local forensics
(`test_local_forensics.py`), dataset integrity (`test_dataset.py`), final model
validation (`test_final_validation.py`), the demo set (`test_demo_dataset.py`) and the Day 8
stability / validation scripts (`test_model_stability.py`), model v2 (`test_tamper_model.py`:
features, schema registry, data isolation and provenance), the Day 9 review tooling
(`test_model_review.py`), localisation (`test_localization.py`: boxes on known edits,
coordinate scaling, v1/v2 and threshold gating, `/verify` schema) and the Day 10 release
tooling (`test_final_release.py`: requirement and target gating, dependency and serialization
checks, ELA and threshold review, and that the committed sign-off matches the model file).
Shared API fixtures live in `tests/conftest.py`.

---

## 4. Request flow — `POST /verify` (verified by code reading and in-process requests)

0. **Middleware stack** (outermost first; `backend/main.py`, `backend/middleware.py`)
   1. `RequestContextMiddleware` — assigns a request ID (the client's `X-Request-ID`
      if short and safe, else a new one), returns it in `X-Request-ID`, logs one
      access line per request.
   2. `CORSMiddleware` — answers browser preflight before authentication.
      Credentials are allowed only for an explicit `CORS_ORIGINS` list, never with `*`.
   3. **API-key check** — `/health`, `/docs`, `/redoc`, `/openapi.json` are public;
      unknown routes → `404` (before authentication); missing `X-API-Key` → `401`;
      wrong key → `403` (constant-time comparison).
   4. `UploadSizeLimitMiddleware` — body above `MAX_FILE_SIZE` + 64 KB framing → `413`
      before the multipart form is parsed (from `Content-Length`, or by counting
      bytes for chunked uploads).
1. **File validation** (`verify_document` + `backend/error_handlers.py`)
   - No `file` field → `400`.
   - Filename missing or extension not in `.jpg`, `.jpeg`, `.png`, `.pdf` → `400`
     (validation is by extension; content is checked when decoded).
   - At most `MAX_FILE_SIZE + 1` bytes are read; empty upload → `400`, oversize → `413`.
2. **Concurrency slot** — at most `MAX_CONCURRENT_ANALYSES` analyses run at once
   (default `min(2, CPUs)`); further requests wait.
3. **Inference** runs in a worker thread (`asyncio.to_thread`) via
   `model_api.predict(content, filename)`; only the base name of the client's
   filename is used, and only for its extension.
4. **Temporary file** — the upload bytes are written to a `NamedTemporaryFile`
   with the original extension; it is deleted after prediction.
   **Dimension check** (header only, nothing decoded): more than `MAX_IMAGE_PIXELS`
   (40 MP; PDF page 1 measured at 200 DPI) → `400`; shortest side under 16 px after
   scaling to 2000 px → `400`; password-protected or empty PDF → `400`.
5. **Feature extraction** — the extractor of the model's recorded feature schema
   (`core.feature_schemas`); for model v2, `core.tamper_features.extract_with_regions(path)`
   (the features of `extract_tamper_features` plus the anomalous regions, from one pass):
   - `load_image`: converts to RGB; PDFs are rendered from **page 1 only** at 200 DPI;
     images whose longest side exceeds 2000 px are downscaled.
   - Builds per-tile (8 x 8 px) maps of noise residual, paper colour, sharpness and ink
     darkness, and measures the largest region that differs from the rest of the page.
   - Returns 29 features in the fixed `FEATURE_NAMES` order, and the regions (section 8).
6. **Classification** — one `model.predict_proba` pass on the 29-value vector; the
   regions are kept only if the score reaches `TAMPER_THRESHOLD`.
7. **Response** — a JSON object (section 8). Unreadable images → `400`; model
   unavailable → `503`; other failures → `500` with a generic message and the
   request ID (details are logged, never returned). Client-facing messages never
   contain server paths. All errors use one shape:
   `{"success": false, "error": "...", "code": <status>, "detail": "..."}`.

### The 29 features of model v2 (`core/tamper_features.py`, schema `local-tamper-v2`)

Regions touching the page edge or wider than 60% of the page are treated as layout
(headers, footers, borders) and ignored.

| Group | Features | What it detects |
|---|---|---|
| Noise (7) | `noise_floor`, `noise_excess_p999`, `noise_blob_k2`, `noise_blob_k2_fill`, `noise_blob_k4`, `noise_blob_k4_fill`, `noise_anomalous_frac` | A region whose pixel noise (tile median of \|gray − median3x3\|) is above the page's own floor |
| Colour (11) | `colour_rare_blob_{0_5,1,2,4}pct` (+ `_fill`), `colour_novel_blob_d10`, `colour_novel_blob_d20`, `colour_novel_p999` | A region of flat tiles whose paper colour is rare on the page or far from every common page colour (colour cast, pasted box on a tinted background) |
| Softness (5) | `soft_blob_s1_0`, `soft_total_s1_0`, `soft_blob_s1_6`, `soft_total_s1_6`, `soft_frac` | Content tiles with weak gradients for their contrast (a blurred region in sharp text) |
| Ink (2) | `ink_rare_line`, `ink_rare_frac` | A run of glyphs whose darkness level is rare on the page (inserted text) |
| Fingerprint (4) | `jpeg_grid`, `flat_frac`, `paper_colours`, `ink_levels` | How the file was saved, so a re-saved genuine page is not read as an edit |

Model v2 is trained on the 60 frozen TRAIN images and 150 generated development
documents, each with 3 processed copies (JPEG, resize, brightness, contrast) for both
classes (`scripts/train_tamper_model.py`).

### Model v1 features (rollback file `fraud_model_v1.pkl`, schema `day4-v2-22`)

| Group | Features |
|---|---|
| ELA statistics | `ela_mean`, `ela_std`, `ela_max`, `ela_p95`, `ela_high_ratio` |
| ELA localisation | `ela_block_max_ratio`, `ela_hotspot_count` |
| Noise / sharpness | `noise_std`, `noise_block_inconsistency`, `laplacian_var` |
| Compression | `blockiness`, `bytes_per_pixel` |
| Other forensic signals | `copy_move_matches`, `exif_edit_flag`, `text_height_std` |
| Geometry | `image_width`, `image_height`, `aspect_ratio` |
| Intensity / texture | `grayscale_mean`, `grayscale_std`, `edge_density`, `dark_pixel_ratio` |

All 22 are whole-image statistics, which is why v1 reached only 60% held-out accuracy.

---

## 5. Model (verified by loading the pickle)

| Property | Model v2 (deployed) | Model v1 (`fraud_model_v1.pkl`) |
|---|---|---|
| File | `data/models/fraud_model.pkl` (≈5.7 MB) | `data/models/fraud_model_v1.pkl` (≈272 KB) |
| Format | pickled dict `{"model": ..., "metadata": ...}` | same |
| Estimator | scikit-learn `RandomForestClassifier` | same |
| Trees / class weights / random state | 300 / `balanced` / 42 | 100 / `balanced` / 42 |
| Feature schema | `local-tamper-v2` (29 features) | `day4-v2-22` (22 features) |
| Labels | `genuine` = 0, `tampered` = 1 | same |
| Training data | 60 frozen TRAIN + 150 generated DEV documents, + 3 processed copies each (840 rows); the file lists every training file with its SHA-256 | all 75 synthetic documents, including the 15 frozen test images |
| Held-out accuracy | **86.67%** (13/15) frozen TEST; **92.62%** (138/149) generated HOLDOUT | 60% (9/15), held-out replica only |

Model v2 is measured directly: the deployed file never saw the frozen TEST set or the
HOLDOUT set, and HOLDOUT was generated after the model was saved
(`scripts/model_validation.py`, `logs/model_validation_report.txt`). Cross-validation on
the training documents: 91.3% ± 3.9% on original files, 75.2% on processed copies
(`logs/training_log_v2.txt`). Retraining with the recorded seed reproduces the file
exactly (`logs/model_stability_report.txt` §7).

Model v1 was refit on all 75 documents, including the 15 frozen test images, so its
60% comes from a replica fitted on the 60 training images only
(`scripts/final_validation.py`, `logs/final_validation_report.txt`).

---

## 6. Dataset and training pipeline

1. **Generate** — `scripts/generate_tampered_documents.py` renders synthetic
   payslips, bank statements and Aadhaar/PAN-style ID proofs with Faker + Pillow.
   Every image carries a "SYNTHETIC / TEST DATA" watermark. Tampered copies use one of
   `blur_region`, `color_shift`, `noise_patch`, `text_overlay`. No real personal data
   is used.
2. **Index** — each indexed document has a row in `data/synthetic/index.csv`
   (`filename, path, doc_type, label, tamper_method, width, height, file_size_bytes`).
3. **Validate** — `tests/test_dataset.py` checks 75 documents (48 genuine /
   27 tampered; 30 payslips / 23 bank statements / 22 ID proofs), CSV↔file
   consistency, duplicates, image readability, labels, tamper methods and
   unexpected files.
4. **Generate more (Day 8)** — `scripts/generate_extra_documents.py` runs the same
   generator with new seeds into `data/generated/dev_v1/` (150 documents, training) and
   `data/generated/holdout_v1/` (149 documents, evaluation only), never into
   `data/synthetic/`. Near-copies of existing images are excluded by rule.
5. **Train** — `scripts/train_tamper_model.py` (model v2) trains on the 60 frozen TRAIN
   images + `dev_v1` with processed copies and saves `data/models/fraud_model.pkl`.
   `scripts/train_model.py` is the v1 trainer (75 documents, 22 features).
6. **Validate** — `scripts/model_validation.py` scores the saved file on the frozen TEST
   set and `holdout_v1`; `scripts/model_stability_test.py` checks determinism.
7. **Serve** — the API loads the saved model at startup.

Dataset details: `DATA_FORMAT_NOTES.md`.

---

## 7. Model loading

- **At startup:** the lifespan hook logs the effective configuration, warns about the
  default API key, a short key or `CORS_ORIGINS=*`, then calls `model_api.load_model()`.
  - The model's recorded `feature_schema_version` must be known to
    `core/feature_schemas.py`, and the model must expect exactly that schema's
    features (29 for v2, 22 for v1) in the same order; otherwise it is refused.
  - Model missing, unreadable or incompatible → the error is logged and the
    **application starts in degraded mode** (it does not crash).
- **Single-document inference (Day 9):** after loading, `ModelAPI` sets the forest to
  `n_jobs=1` and `predict()` makes one `predict_proba` pass, taking the argmax as the class
  (what `RandomForestClassifier.predict` does). Joblib threads inside one prediction only
  added start-up cost and competed with the request slots; a typical page went from ~314 ms
  to ~144 ms in `ModelAPI.predict`, with identical outputs on 175 files
  (`logs/model_review_report.txt` §3).
- **Lazily / recovery:** if `predict()` runs while no model is loaded, it calls
  `load_model()` first, so fixing the model file recovers without a restart.
  `load_model()` is guarded by a `threading.Lock` so concurrent requests do not load
  the model twice.
- **Model still unavailable:** `/verify` returns **HTTP 503** naming the reason.
- `/health` is a readiness check: `200` with `"status": "ok"` when the model is
  loaded, `503` with `"status": "degraded"` and a `detail` otherwise (verified).

---

## 8. `/verify` response fields

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

(`data/demo/tampered_payslip_01.png`; the box covers the replaced account number.)

| Field | Meaning (actual behaviour) |
|---|---|
| `tamper_risk_score` | `predict_proba` value for the `tampered` class, rounded to 4 decimals. An **uncalibrated** Random Forest vote fraction, not a real-world fraud probability |
| `confidence` | `predict_proba` value of the predicted class, rounded to 4 decimals |
| `flagged_regions` | Anomalous regions, largest first, at most 5 (see below). `[]` when the score is below `TAMPER_THRESHOLD`, when nothing stands out, or when the loaded model's schema cannot localise (v1, `day4-v2-22`). The older ELA helper `extract_flagged_regions` in `model_utils.py` is not called |
| `interpretation` | Text chosen by comparing the score with the server-side `TAMPER_THRESHOLD` setting (environment variable, default `0.5`): `tamper_risk_score >= TAMPER_THRESHOLD` → "Evidence of digital tampering detected…", otherwise "Document verified successfully. Low risk…" |

**How regions are found** (`core/tamper_features.py`). While computing the features, every
anomaly mask records its connected regions (8 x 8 px tiles; regions touching the page edge or
wider than 60% of the page are page layout and ignored), and ink analysis records runs of
glyphs sharing a darkness level that is rare on the page. `_select_regions` keeps the regions
of the masks in `LOCATE_MASKS` that exceed that mask's minimum size (set on the DEV set so
genuine DEV pages have almost none), merges boxes that overlap or lie within 8 px, sorts them
by area and rescales them to the uploaded image's pixels (the analysis runs on the copy
downscaled to 2000 px; PDFs: page 1 at 200 DPI). Each region has `x`, `y`, `width`,
`height`, `area` (pixel area of the anomalous tiles; for an ink run, its box) and `type`
(`noise`, `colour`, `softness`, `ink`, or several joined by `+` after a merge). The model's
features are unchanged (`extract_tamper_features` gives identical values), so the trained
model and its SHA-256 records stay valid. Cost: about 3 ms per page.

**Why only when flagged:** regions exist on some genuine pages too (9 of 93 held-out genuine
pages have a locator box); gating on the score removes most of them (2 of 93 remain) and
keeps the response consistent with the interpretation. The interpretation then adds
"N suspicious region(s) located (see flagged_regions)."

**Accuracy** (`scripts/localization_eval.py` → `logs/localization_report.txt`; a hit is a box
intersecting the true edit rectangle recovered by replaying the generator):

| Set | Visible edits | Hit | Top-1 hit | Box precision | Genuine pages with boxes |
|---|---|---|---|---|---|
| DEV (design set, optimistic) | 59 | 49 (83.1%) | 49 (83.1%) | 54/54 | 0/87 |
| Held-out (HOLDOUT + TEST) | 67 | 55 (82.1%) | 52 (77.6%) | 61/69 | 2/93 |

Noise patches and blurred regions are found almost always; text overlays are the weak spot
(10 of 19 held-out), mostly on bank statements where the page's own text has the overlay's
ink darkness. The tile thresholds assume the generator's page size: on resampled pages
(0.75x, 1.5x, 2.5x) every genuine DEV page gets locator boxes, and the score is unreliable
there too.

The optional `threshold` query parameter is validated (`0.0`–`1.0`, otherwise `422`)
but **not used**: it does not change the score, the interpretation or any other
field (verified).

---

## 9. Known limitations

- **Synthetic data only.** Model v2 is trained and evaluated on documents from one
  synthetic generator (86.67% on 15 frozen test documents, 92.62% on 149 held-out
  documents); performance on real documents is unknown.
- **Undetectable labels.** Some generated "tampered" documents change no pixel (a blur on
  blank paper); 3 of the 10 missed held-out edits are of this kind.
- **Processing sensitivity.** Re-saved, resized, rotated or cropped files are handled much
  less reliably than original renders. On the 164 held-out documents (92.1% unmodified):
  JPEG q85 73.2%, 50% downscale 50.0%, 180° rotation 64.0%, 3° skew 70.1%, 2-px crop 79.9%,
  light blur ~85% (`logs/model_robustness_report.txt`). Non-document images still receive a
  score.
- **Uncalibrated scores.** `tamper_risk_score` must not be read as a probability of fraud.
- **Regions are anomalies, not proof.** `flagged_regions` finds 82.1% of held-out edits;
  text overlays whose ink matches the page are missed, and boxes are unreliable on pages
  resampled away from the training resolution (section 8). Model v1 returns no regions.
- **`threshold` is accepted but ignored.**
- **Only the first page of a PDF is analysed.**
- **Extension-based upload validation**; file content is only checked when decoded.
- **One process, threads for concurrency:** analyses contend for the GIL; throughput
  peaks at about 8-9 documents/s on 4 CPUs with model v2 (`logs/inference_benchmark_report.txt`;
  v1: about 4/s, `logs/performance_review.txt`).
- **Experimental modules** (`forensic_model.py`, `local_forensics.py`) are not part of
  the served pipeline.
- **Not production-ready** — see `DEPLOYMENT.md` §10.

---

## 10. Offline and demo assets (not runtime components)

These were added after the runtime architecture above was documented. None of them is
imported by `backend/` or loaded by the API; they are repository tooling and reference
material. Runtime behaviour (§4–§8) is unchanged by them.

| Path | Owner | What it is |
|---|---|---|
| `data/demo/` | Founder A (Day 7) | 5 demo images (2 genuine payslips, 2 tampered payslips, 1 edge case) with `demo_manifest.json` and `DEMO_IMAGES.md`. For API / frontend demonstration only — **not a validation set** |
| `scripts/prepare_demo_dataset.py` | Founder A (Day 7) | Builds `data/demo/` and scores it once per deployed model through `ModelAPI` (`--rescore` for a new model; earlier records kept in `scoring_history`) |
| `tests/test_demo_dataset.py` | Founder A (Day 7) | Checks the demo images, their recorded hashes and sources, frozen-test-set separation and recorded predictions |
| `scripts/final_validation.py` | Founder A (Day 6) | Verifies the frozen 15-image test split and measures model v1's held-out performance (§5); exits with code 3 for other models |
| `tests/test_final_validation.py` | Founder A (Day 6) | Unit tests for the final-validation metrics |
| `scripts/model_stability_test.py` | Founder A (Day 8) | Scores each demo image repeatedly through `ModelAPI` and checks execution paths, features, ELA and drift; writes `logs/model_stability_report.txt` |
| `scripts/model_validation.py` | Founder A (Day 8) | Model integrity, demo validation, held-out accuracy of the deployed file on frozen TEST + `holdout_v1`, error analysis and the final model checklist; writes `logs/model_validation_report.txt` |
| `tests/test_model_stability.py` | Founder A (Day 8) | Unit tests for the Day 8 scripts plus one real-model repeatability check |
| `data/generated/dev_v1/`, `holdout_v1/` | Founder A (Day 8) | Generated documents (seeds 7001 / 7002, numbered 1001+ / 2001+) with `index.csv` and `manifest.json` (SHA-256 per file, excluded near-copies). `dev_v1` is training data; `holdout_v1` is evaluation-only |
| `scripts/generate_extra_documents.py` | Founder A (Day 8) | Creates the two sets with Founder C's generator (unmodified) via the safe workflow in `DATA_FORMAT_NOTES.md` §6; also replays generator runs to measure each edit's changed pixels |
| `scripts/train_tamper_model.py` | Founder A (Day 8) | Trains model v2; writes `logs/training_log_v2.txt` |
| `tests/test_tamper_model.py` | Founder A (Day 8) | Model v2 features, schema registry, data isolation and provenance |
| `data/models/fraud_model_v1.pkl` | Founder A (Day 5) | Model v1, byte-identical to the file deployed until Day 8; rollback only |
| `scripts/feature_importance.py` | Founder A (Day 9) | MDI, out-of-fold permutation importance, partial-dependence direction and family checks on the model's training data; writes `logs/feature_importance_report.txt` |
| `scripts/model_robustness_test.py` | Founder A (Day 9) | Scores the 164 held-out documents under 13 degradations through `ModelAPI`; writes `logs/model_robustness_report.txt` |
| `scripts/model_review.py` | Founder A (Day 9) | Performance vs earlier reports, prediction quality, speed / memory / size, forest alternatives (CV), release checklist; writes `logs/model_review_report.txt` |
| `scripts/localization_eval.py` | Founder A | `flagged_regions` against the true edit rectangles (replayed generator) on DEV, held-out and resampled DEV pages; writes `logs/localization_report.txt` |
| `scripts/model_accuracy_eval.py` | Founder A (Day 10) | Frozen TEST + HOLDOUT through `ModelAPI`: accuracy, precision, recall, F1, confusion matrices, comparison with every earlier record, release requirements; writes `logs/final_accuracy_report.txt` |
| `scripts/inference_benchmark.py` | Founder A (Day 10) | Model load, in-process min/avg/max over 169 documents x 3, large inputs, real-server HTTP (sequential and 5 users), output stability; writes `logs/inference_benchmark_report.txt` |
| `scripts/final_model_validation.py` | Founder A (Day 10) | Load / dependency / serialization / runtime checks, demo validation, release review (features, ELA, prediction logic, thresholds), sign-off; writes `logs/final_model_release_report.txt` and `logs/model_signoff_checklist.txt` |
| `logs/model_limitations.txt` | Founder A (Day 10) | Known limitations: dataset, edge cases, expected failure scenarios, future improvements |
| `tests/test_model_review.py` | Founder A (Day 9) | Inference optimisation, robustness variants and summaries, importance helpers |
| `scripts/dataset_stats.py` | Founder C (Day 7) | Read-only statistics for the 75-document indexed dataset; source of `DATASET.md` |

Relationship to the indexed dataset:

- Two demo images are byte-identical copies of Day 1 payslips kept outside the index
  (`genuine_payslip_01.png` = `data/synthetic/payslip_002.png`,
  `genuine_payslip_02.png` = `data/synthetic/payslip_003.png`).
- The other three are derived from git-ignored Day 5 augmentations of indexed documents
  (`genuine/payslip_002.png`, `genuine/payslip_037.png`, `genuine/payslip_006.png`), as
  recorded in `demo_manifest.json`.
- Demo images are **not** part of the 75-document index and are not counted in
  `DATASET.md`.

Resolved issue: during Founder C's Day 8 QA,
`tests/test_demo_dataset.py::test_hashes_match_record_and_source` failed on a fresh clone
with `FileNotFoundError` (e.g. `data/augmented/train_v2_1/tampered/payslip_002_aug1.png`),
because three demo sources live under the git-ignored `data/augmented/train_v2_1/`.
Founder B then changed the test (commit `04f079f`, merged to `main` in PR #11): each demo
image is still checked against its recorded SHA-256, and the source comparison runs only
when the source file exists locally. The augmented source images remain git-ignored and
are still absent from a fresh clone; the demo tests pass on current `main` (4 passed).
