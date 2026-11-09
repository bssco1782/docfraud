# Day 5 — Augmentation v2 + Corrected Cross-Validation (Founder A)

Date: 2026-09-25 · Status: **CV complete, awaiting review.** No grid search, no final model, no frozen-test evaluation, no commit.

Data labels used throughout: **TRAINING DATA** (60 original training images + augmented views of them), **CV DATA** (held-out folds of the 60 originals), **FROZEN TEST DATA** (15 images — not read by any step in this report except SHA-256 integrity checks).

## 1. Executive summary

- The augmentation label shortcut is fixed. In `train-aug-v2.1` whole-page processing happens at the same rate (50.0 % / 50.0 %) for genuine and tampered images across the whole training pool, op types are balanced within each class, and a probe that tries to predict the label from file size, dimensions, brightness, noise, JPEG grid, format and metadata scores **0.504 balanced accuracy (chance 0.5)**. All 16 automated validation checks pass.
- **Corrected augmentation does not improve the detector.** On the 60 training originals (5-fold CV, identical folds):
  - B (local-forensics-v1, no augmentation): accuracy 0.733 ± 0.082, FP 6 (15.8 %), FN 10 (45.5 %).
  - B + aug-v2.1: 0.633 ± 0.041, FP 13 (34.2 %), FN 9.
  - C + aug-v2.1: 0.700 ± 0.113, FP 9 (23.7 %), FN 9 (40.9 %).
- **No configuration reaches 78 % CV accuracy.** Best observed: 0.733 (B).
- Root cause is the representation, not the data or the model: when trained with label-balanced processing, the local features still cannot tell a **processed genuine page** from a **processed tampered page** (B: 25 of 38 processed genuine views flagged, JPEG re-saved genuine 6 of 7 flagged), and they detect only **7 of 27 purely local edits** with no whole-page processing.

## 2. Previous augmentation problem (train-aug-v1 — INVALID)

v1 gave whole-page processing (JPEG / brightness / noise / resize) only to augmented *genuine* images and only local edits to augmented *tampered* images. The model learned "processed ⇒ genuine, untouched ⇒ tampered": B + aug-v1 pooled confusion `[[15, 23], [0, 22]]` — 23 of 38 genuine originals flagged. v1 is excluded from every experiment here (cache records dropped, version-namespaced keys); its images and CV results are kept only as history (`data/augmented/train_v1/`, `cv_train-aug-v1` in the results JSON).

An intermediate `train-aug-v2` (v2.0) already equalised processing rates but lacked ID / name replacement, region duplication, whole-page contrast and combined whole-page ops. It is superseded by v2.1 and kept only as history (`data/augmented/train_v2/`, `cv_train-aug-v2`). Its slightly better B+aug numbers (0.700, FP 11) are **not** used for any decision — picking between augmentation versions by CV score would be tuning the data to the metric.

## 3. Augmentation v2.1 design (TRAINING DATA only)

Generator: `scripts/augment_training_data.py` · version `train-aug-v2.1` · output `data/augmented/train_v2_1/` · manifest `manifest.csv` · base seed 9000 (per-source seed 9000 + i) · re-running produces a byte-identical manifest.

- **Sources:** the 38 genuine images of the frozen split's TRAIN list only (hash-checked against the frozen split; generator refuses any path or hash that belongs to the frozen test set).
- **Per source:** 2 tampered views + 1 genuine view → 114 augmented views of 38 documents. **These are not 114 independent documents.**
- **Local edits** (tampered views), 17 types spread evenly (4–5 each): number / date / amount / account / ID (PAN- or Aadhaar-style) / name replacement, token copy-move, field-region duplication, local blur, sharpen, brightness, contrast, colour shift, noise, local JPEG, local resampling, and combinations (a text/copy edit followed by JPEG / resample / blur / noise).
  - Edits target real text tokens found by connected components (not blank paper, not a fixed place). Text replacement samples the local background from a ring around the word, matches ink colour from the original word, scales font size to the word height, and uses one of 5 fonts. Pixel edits blend through a Gaussian-feathered mask (no hard rectangle). Size, location, severity and content are randomised; every edit must change pixels (mean |Δ| ≥ 1.5) or it is retried.
- **Whole-page processing** (both classes): mild JPEG (q 70–95), brightness (±10 %), contrast (±10 %), Gaussian noise (σ 0.5–3), resize (0.8–1.2×), or a combination of two. Genuine views are always processed (an unprocessed genuine view would be an exact duplicate of its source, which is already in the pool); the number of processed tampered views is set so the pool-level processed rate is identical per class (38/76 genuine, 49/98 tampered).
- **Provenance per row:** file, label, source path, source SHA-256, source type, local transform + parameters, bounding box, whole-page transform + parameters, seed, output SHA-256, version.

## 4. Dataset counts

| Set | Genuine | Tampered | Total | Notes |
|---|---|---|---|---|
| TRAINING originals | 38 | 22 | 60 | independent documents |
| TRAINING augmented (v2.1) | 38 | 76 | 114 | views of 38 training documents |
| TRAINING pool used when "+aug" | 76 | 98 | 174 | 60 independent documents |
| FROZEN TEST | 10 | 5 | 15 | untouched; hashes verified |

## 7. Feature representation

- A: `day4-v2-22` (22 whole-image features; 4 constant, most document-type dependent — see audit).
- B: `local-forensics-v1` (52 features): 8 per-tile signals (luminance ELA, noise residual, sharpness on ink tiles, gradient, high-frequency energy, paper chroma, intensity, contrast) scored as robust z against the same document within content classes, summarised by max / p99 / p95 / mean / anomalous fraction / neighbour inconsistency, plus 4 region-level (4×4 grid) summaries.
- C: A + B (74).
- Constant local features on the 60 originals: `lf_ela_neighbor_incons`, `lf_noise_p95`, `lf_sharpness_neighbor_incons` (not constant on the full pool). Flagged, not removed.

## 8. CV methodology

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the 60 training originals — the same folds as every previous run (A, B, C reproduce the previous per-fold confusion matrices exactly).
- **Group rule:** the grouping unit is the original source document. In each fold, augmented views are added to training only if their source document is in the training folds; views of held-out sources are never trained on (asserted every fold). Validation always uses originals only, so all six experiments are scored on identical samples.
- RandomForest `n_estimators=100, class_weight="balanced", random_state=42` (Day-5 baseline, untuned). Decision threshold fixed at the RF default; not tuned.
- Test separation asserted in code before every experiment: no frozen-test key or hash in training/CV records, cache, augmentation sources or outputs.

## Trade-offs between technically viable candidates (not a ranking)

| Candidate | Genuine flagged (FPR) | Tampered missed (FNR) | Comment |
|---|---|---|---|
| B | **6 / 38 (15.8 %)** | 10 / 22 (45.5 %) | Lowest review burden and most stable F1 (± 0.052), but misses almost half of tampering; trained only on pristine PNGs, so its FPR on scanned / JPEG uploads is expected to be far higher (held-out processed genuine views: 25/38 flagged even for B trained with processing examples). |
| C + aug-v2.1 | 9 / 38 (23.7 %) | **9 / 22 (40.9 %)** | One fewer miss than B at the cost of 3 more false alarms; fold variance high (accuracy ± 0.113). |
| B + aug-v2.1 | 13 / 38 (34.2 %) | 9 / 22 (40.9 %) | More than a third of genuine documents flagged — too heavy a manual-review load. |

For an NBFC queue, each false positive is a genuine applicant delayed and a reviewer's time spent; each false negative is a forged income/identity document passing. At current signal strength no configuration gives an acceptable pair: lowering misses costs disproportionately many false alarms. The pre-registered F1 rule points to B (B 0.604 vs C+aug 0.600, within 0.02 → lower F1 std); this is recorded as a rule output, not a deployment decision.

## Error analysis (CV errors only — FROZEN TEST not used)

- **Genuine incorrectly flagged — layout / document confusion:** B's 6 FPs are split evenly across document types (2 each); `id_proof_064`, `id_proof_072`, `bank_stmt_012`, `bank_stmt_051` are flagged by all six A/B/C experiments (with and without augmentation) → consistent template elements (borders, coloured bands, watermark) producing anomaly peaks. 52–61 % of B's importance (B+aug 0.525, B 0.613) sits on layout-sensitive signal families (gradient, high-frequency, intensity, contrast); a doc-type probe on B's features is 1.000 accurate.
- **Compression / preprocessing artefact:** processed genuine views are flagged at 66 % (B) — JPEG re-save 6/7, resize 5/6, combined 5/6. Local anomaly signals shift when the whole page is processed.
- **Weak tamper signal / local feature failure:** colour shift (3/7 detected by B), blur (1/4) in original data; date / ID / account replacement (1/5 each), copy-move (1/5), resampling (1/4) in synthetic data. Detected well: noise (7/7 original with aug; 4/4 synthetic), region duplication (4/5), name / number replacement (4/5), local JPEG and blur (3/4).
- **Unrealistic synthetic tampering:** token choice is random, so replacement content sometimes does not match the field's meaning (e.g. a date over a label). Forensically plausible (matched font, ink, background), semantically imperfect.

## 16. Limitations

- 60 independent training documents (22 tampered); one error moves fold accuracy by ~8 points; std 0.04–0.15.
- Three fixed synthetic templates, lossless PNG, one generator; every feature set fully identifies document type.
- Original tampering covers only 4 simple methods; synthetic edits are rendered with host fonts (Windows) that differ from the documents' original font.
- CV on pristine originals overstates real-world performance: real uploads are JPEG / scanned / photographed, where the current features raise false alarms.
- No probability calibration; model output is an uncalibrated RF vote fraction.

## 17. Recommended next engineering step

Make local-forensics invariant to whole-page processing before any tuning: estimate each document's own processing fingerprint (JPEG quality from quantisation / blockiness, page-level noise floor, resampling period) and score tiles **relative to that fingerprint** (e.g. ELA at the estimated quality, noise normalised by the page noise floor, double-compression consistency per 8×8 block) instead of absolute levels. Accept it only if held-out processed-genuine false alarms fall substantially while original-document CV holds or improves. Then add text-line consistency features (stroke width, ink colour, glyph height within a line) for replacement edits.

<!-- DATA SECTIONS BELOW ARE GENERATED by `python scripts/ml_recovery.py report` from logs/day5_ml_recovery_results.json. Narrative sections are written by hand. -->

## Experiment comparison — CV DATA (60 training originals, 5-fold, mean ± std)

| Model | Features | Augmentation | Accuracy | Precision | Recall | F1 | FP | FN | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|
| A | day4-v2-22 (22) | none | 0.583 ± 0.149 | 0.440 ± 0.136 | 0.370 ± 0.201 | 0.387 ± 0.154 | 11 | 14 | 28.9% | 63.6% |
| B | local-forensics-v1 (52) | none | 0.733 ± 0.082 | 0.717 ± 0.163 | 0.540 ± 0.049 | 0.604 ± 0.052 | 6 | 10 | 15.8% | 45.5% |
| C | day4 + local (74) | none | 0.667 ± 0.053 | 0.553 ± 0.126 | 0.450 ± 0.118 | 0.491 ± 0.110 | 8 | 12 | 21.1% | 54.5% |
| A + aug | day4-v2-22 (22) | train-aug-v2.1 | 0.517 ± 0.122 | 0.297 ± 0.190 | 0.370 ± 0.256 | 0.322 ± 0.206 | 15 | 14 | 39.5% | 63.6% |
| B + aug | local-forensics-v1 (52) | train-aug-v2.1 | 0.633 ± 0.041 | 0.513 ± 0.097 | 0.590 ± 0.156 | 0.531 ± 0.075 | 13 | 9 | 34.2% | 40.9% |
| C + aug | day4 + local (74) | train-aug-v2.1 | 0.700 ± 0.113 | 0.657 ± 0.218 | 0.590 ± 0.092 | 0.600 ± 0.103 | 9 | 9 | 23.7% | 40.9% |

Ablations (layout-shortcut Day-4 features removed):

| Experiment | Accuracy | F1 | FP | FN |
|---|---|---|---|---|
| A_minus_shortcuts | 0.600 ± 0.122 | 0.429 ± 0.161 | 11 | 13 |
| A_minus_shortcuts+aug | 0.517 ± 0.097 | 0.378 ± 0.216 | 18 | 11 |
| C_minus_shortcuts | 0.717 ± 0.041 | 0.528 ± 0.138 | 6 | 11 |
| C_minus_shortcuts+aug | 0.650 ± 0.097 | 0.556 ± 0.112 | 12 | 9 |

## Fold-level results — CV DATA

Each cell: accuracy / precision / recall / F1, confusion [[TN, FP], [FN, TP]].

| Fold | A | B | C | A + aug | B + aug | C + aug |
|---|---|---|---|---|---|---|
| 1 | 0.750/0.600/0.750/0.667 [[6, 2], [1, 3]] | 0.750/0.667/0.500/0.571 [[7, 1], [2, 2]] | 0.583/0.333/0.250/0.286 [[6, 2], [3, 1]] | 0.417/0.286/0.500/0.364 [[3, 5], [2, 2]] | 0.583/0.400/0.500/0.444 [[5, 3], [2, 2]] | 0.500/0.333/0.500/0.400 [[4, 4], [2, 2]] |
| 2 | 0.667/0.500/0.250/0.333 [[7, 1], [3, 1]] | 0.833/1.000/0.500/0.667 [[8, 0], [2, 2]] | 0.750/0.667/0.500/0.571 [[7, 1], [2, 2]] | 0.583/0.000/0.000/0.000 [[7, 1], [4, 0]] | 0.667/0.500/0.500/0.500 [[6, 2], [2, 2]] | 0.833/1.000/0.500/0.667 [[8, 0], [2, 2]] |
| 3 | 0.667/0.500/0.250/0.333 [[7, 1], [3, 1]] | 0.750/0.667/0.500/0.571 [[7, 1], [2, 2]] | 0.667/0.500/0.500/0.500 [[6, 2], [2, 2]] | 0.667/0.500/0.750/0.600 [[5, 3], [1, 3]] | 0.583/0.429/0.750/0.545 [[4, 4], [1, 3]] | 0.750/0.600/0.750/0.667 [[6, 2], [1, 3]] |
| 4 | 0.500/0.400/0.400/0.400 [[4, 3], [3, 2]] | 0.750/0.750/0.600/0.667 [[6, 1], [2, 3]] | 0.667/0.667/0.400/0.500 [[6, 1], [3, 2]] | 0.583/0.500/0.400/0.444 [[5, 2], [3, 2]] | 0.667/0.667/0.400/0.500 [[6, 1], [3, 2]] | 0.750/0.750/0.600/0.667 [[6, 1], [2, 3]] |
| 5 | 0.333/0.200/0.200/0.200 [[3, 4], [4, 1]] | 0.583/0.500/0.600/0.545 [[4, 3], [2, 3]] | 0.667/0.600/0.600/0.600 [[5, 2], [2, 3]] | 0.333/0.200/0.200/0.200 [[3, 4], [4, 1]] | 0.667/0.571/0.800/0.667 [[4, 3], [1, 4]] | 0.667/0.600/0.600/0.600 [[5, 2], [2, 3]] |

## Pooled confusion matrices — CV DATA (all 60 originals, each predicted once)

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| A | 27 | 11 | 14 | 8 |
| B | 32 | 6 | 10 | 12 |
| C | 30 | 8 | 12 | 10 |
| A + aug | 23 | 15 | 14 | 8 |
| B + aug | 25 | 13 | 9 | 13 |
| C + aug | 29 | 9 | 9 | 13 |

## Document-type analysis — CV DATA

**A**

| Doc type | Genuine | Tampered | Accuracy | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|---|---|---|
| bank_statement | 12 | 8 | 0.500 | 0.333 | 0.250 | 0.286 | [[8, 4], [6, 2]] |
| id_proof | 12 | 4 | 0.812 | 0.600 | 0.750 | 0.667 | [[10, 2], [1, 3]] |
| payslip | 14 | 10 | 0.500 | 0.375 | 0.300 | 0.333 | [[9, 5], [7, 3]] |

**B**

| Doc type | Genuine | Tampered | Accuracy | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|---|---|---|
| bank_statement | 12 | 8 | 0.700 | 0.667 | 0.500 | 0.571 | [[10, 2], [4, 4]] |
| id_proof | 12 | 4 | 0.812 | 0.600 | 0.750 | 0.667 | [[10, 2], [1, 3]] |
| payslip | 14 | 10 | 0.708 | 0.714 | 0.500 | 0.588 | [[12, 2], [5, 5]] |

**C**

| Doc type | Genuine | Tampered | Accuracy | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|---|---|---|
| bank_statement | 12 | 8 | 0.650 | 0.571 | 0.500 | 0.533 | [[9, 3], [4, 4]] |
| id_proof | 12 | 4 | 0.750 | 0.500 | 0.500 | 0.500 | [[10, 2], [2, 2]] |
| payslip | 14 | 10 | 0.625 | 0.571 | 0.400 | 0.471 | [[11, 3], [6, 4]] |

**A + aug**

| Doc type | Genuine | Tampered | Accuracy | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|---|---|---|
| bank_statement | 12 | 8 | 0.450 | 0.200 | 0.125 | 0.154 | [[8, 4], [7, 1]] |
| id_proof | 12 | 4 | 0.750 | 0.500 | 0.750 | 0.600 | [[9, 3], [1, 3]] |
| payslip | 14 | 10 | 0.417 | 0.333 | 0.400 | 0.364 | [[6, 8], [6, 4]] |

**B + aug**

| Doc type | Genuine | Tampered | Accuracy | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|---|---|---|
| bank_statement | 12 | 8 | 0.650 | 0.556 | 0.625 | 0.588 | [[8, 4], [3, 5]] |
| id_proof | 12 | 4 | 0.750 | 0.500 | 0.500 | 0.500 | [[10, 2], [2, 2]] |
| payslip | 14 | 10 | 0.542 | 0.462 | 0.600 | 0.522 | [[7, 7], [4, 6]] |

**C + aug**

| Doc type | Genuine | Tampered | Accuracy | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|---|---|---|
| bank_statement | 12 | 8 | 0.700 | 0.625 | 0.625 | 0.625 | [[9, 3], [3, 5]] |
| id_proof | 12 | 4 | 0.812 | 0.600 | 0.750 | 0.667 | [[10, 2], [1, 3]] |
| payslip | 14 | 10 | 0.625 | 0.556 | 0.500 | 0.526 | [[10, 4], [5, 5]] |

Document-type probe (predict doc type from features, 5-fold): A_global_day4 = 1.000, B_local_v1 = 1.000, A_minus_shortcuts = 1.000

Feature-importance share (RF fit on all training data of that experiment):

| Model | Layout-sensitive share | Forensic share | Largest groups |
|---|---|---|---|
| B_local_v1 | 0.61 | 0.39 | local_hf 0.19, local_gradient 0.18, local_intensity 0.15, local_chroma 0.11, local_contrast 0.10 |
| B_local_v1+aug | 0.52 | 0.48 | local_sharpness 0.17, local_gradient 0.15, local_hf 0.14, local_intensity 0.13, local_contrast 0.10 |
| C_global_plus_local | 0.55 | 0.45 | day4_other 0.16, local_intensity 0.14, local_gradient 0.13, day4_shortcut 0.11, local_hf 0.10 |
| C_global_plus_local+aug | 0.42 | 0.58 | day4_other 0.22, local_sharpness 0.12, local_hf 0.10, local_gradient 0.10, day4_shortcut 0.09 |

Layout-sensitive groups: day4_shortcut, local_contrast, local_gradient, local_hf, local_intensity.

Constant local features — train originals: ['lf_ela_neighbor_incons', 'lf_noise_p95', 'lf_sharpness_neighbor_incons']; full training pool: none. Flagged, not removed.


## Tampering-method analysis

### ORIGINAL tampering — CV DATA (held-out originals)

| Method | n | A detected / missed / recall | B detected / missed / recall | C detected / missed / recall | A + aug detected / missed / recall | B + aug detected / missed / recall | C + aug detected / missed / recall |
|---|---|---|---|---|---|---|---|
| blur_region | 4 | 1 / 3 / 0.25 | 1 / 3 / 0.25 | 1 / 3 / 0.25 | 0 / 4 / 0.00 | 1 / 3 / 0.25 | 1 / 3 / 0.25 |
| color_shift | 7 | 3 / 4 / 0.43 | 3 / 4 / 0.43 | 2 / 5 / 0.29 | 2 / 5 / 0.29 | 3 / 4 / 0.43 | 2 / 5 / 0.29 |
| noise_patch | 7 | 3 / 4 / 0.43 | 5 / 2 / 0.71 | 5 / 2 / 0.71 | 4 / 3 / 0.57 | 7 / 0 / 1.00 | 7 / 0 / 1.00 |
| text_overlay | 4 | 1 / 3 / 0.25 | 3 / 1 / 0.75 | 2 / 2 / 0.50 | 2 / 2 / 0.50 | 2 / 2 / 0.50 | 3 / 1 / 0.75 |

### SYNTHETIC train-aug-v2.1 tampering — TRAINING DATA, held out by source document

Model trained on the other 4 folds' originals + their augmentations; scored on augmented images whose source document is in the held-out fold. Synthetic edits, NOT independent real fraud samples.

| Label:local op | n | A correct | B correct | C correct |
|---|---|---|---|---|
| genuine:none | 38 | 11 | 13 | 9 |
| tampered:combo | 4 | 0 | 2 | 2 |
| tampered:copy_move | 5 | 2 | 1 | 1 |
| tampered:local_blur | 4 | 0 | 3 | 1 |
| tampered:local_brightness | 4 | 2 | 2 | 1 |
| tampered:local_color | 4 | 4 | 3 | 4 |
| tampered:local_contrast | 4 | 2 | 2 | 4 |
| tampered:local_jpeg | 4 | 1 | 3 | 3 |
| tampered:local_resample | 4 | 2 | 1 | 1 |
| tampered:local_sharpen | 4 | 3 | 2 | 1 |
| tampered:noise_injection | 4 | 3 | 4 | 4 |
| tampered:region_duplicate | 5 | 3 | 4 | 4 |
| tampered:text_replace_account | 5 | 1 | 1 | 2 |
| tampered:text_replace_amount | 5 | 2 | 3 | 2 |
| tampered:text_replace_date | 5 | 0 | 1 | 0 |
| tampered:text_replace_id | 5 | 2 | 1 | 2 |
| tampered:text_replace_name | 5 | 4 | 4 | 4 |
| tampered:text_replace_number | 5 | 2 | 4 | 3 |

By whole-page processing (robustness to benign scanning / re-saving):

| Label:whole-page op | n | A correct | B correct | C correct |
|---|---|---|---|---|
| genuine:ANY_whole_page | 38 | 11 | 13 | 9 |
| genuine:global_brightness | 7 | 5 | 4 | 2 |
| genuine:global_combo | 6 | 2 | 1 | 1 |
| genuine:global_contrast | 6 | 2 | 4 | 2 |
| genuine:global_jpeg | 7 | 0 | 1 | 0 |
| genuine:global_noise | 6 | 1 | 2 | 3 |
| genuine:global_resize | 6 | 1 | 1 | 1 |
| tampered:ANY_whole_page | 49 | 27 | 34 | 34 |
| tampered:global_brightness | 8 | 7 | 6 | 7 |
| tampered:global_combo | 8 | 4 | 5 | 5 |
| tampered:global_contrast | 8 | 4 | 4 | 4 |
| tampered:global_jpeg | 9 | 8 | 9 | 8 |
| tampered:global_noise | 8 | 0 | 4 | 3 |
| tampered:global_resize | 8 | 4 | 6 | 7 |
| tampered:untouched | 27 | 6 | 7 | 5 |

## False positives / false negatives — CV DATA

- **A** — FP (11): genuine/bank_stmt_048.png, genuine/id_proof_072.png, genuine/payslip_027.png, genuine/payslip_007.png, genuine/bank_stmt_016.png, genuine/bank_stmt_051.png, genuine/payslip_002.png, genuine/bank_stmt_012.png, genuine/id_proof_064.png, genuine/payslip_003.png, genuine/payslip_006.png
  - FN (14): tampered/bank_stmt_046.png (noise_patch), tampered/bank_stmt_017.png (blur_region), tampered/id_proof_025.png (blur_region), tampered/payslip_045.png (color_shift), tampered/bank_stmt_014.png (text_overlay), tampered/payslip_026.png (color_shift), tampered/payslip_041.png (noise_patch), tampered/bank_stmt_057.png (text_overlay), tampered/bank_stmt_060.png (text_overlay), tampered/payslip_009.png (blur_region), tampered/bank_stmt_055.png (noise_patch), tampered/payslip_034.png (noise_patch), tampered/payslip_035.png (color_shift), tampered/payslip_044.png (color_shift)
- **B** — FP (6): genuine/id_proof_072.png, genuine/payslip_029.png, genuine/bank_stmt_051.png, genuine/bank_stmt_012.png, genuine/id_proof_064.png, genuine/payslip_008.png
  - FN (10): tampered/bank_stmt_018.png (color_shift), tampered/bank_stmt_046.png (noise_patch), tampered/bank_stmt_017.png (blur_region), tampered/id_proof_025.png (blur_region), tampered/payslip_026.png (color_shift), tampered/payslip_041.png (noise_patch), tampered/bank_stmt_057.png (text_overlay), tampered/payslip_009.png (blur_region), tampered/payslip_035.png (color_shift), tampered/payslip_044.png (color_shift)
- **C** — FP (8): genuine/bank_stmt_048.png, genuine/id_proof_072.png, genuine/payslip_027.png, genuine/payslip_007.png, genuine/payslip_029.png, genuine/bank_stmt_051.png, genuine/bank_stmt_012.png, genuine/id_proof_064.png
  - FN (12): tampered/bank_stmt_018.png (color_shift), tampered/bank_stmt_046.png (noise_patch), tampered/payslip_004.png (color_shift), tampered/bank_stmt_017.png (blur_region), tampered/id_proof_025.png (blur_region), tampered/payslip_026.png (color_shift), tampered/payslip_041.png (noise_patch), tampered/bank_stmt_057.png (text_overlay), tampered/id_proof_062.png (text_overlay), tampered/payslip_009.png (blur_region), tampered/payslip_035.png (color_shift), tampered/payslip_044.png (color_shift)
- **A + aug** — FP (15): genuine/bank_stmt_048.png, genuine/id_proof_069.png, genuine/id_proof_072.png, genuine/payslip_005.png, genuine/payslip_038.png, genuine/payslip_027.png, genuine/bank_stmt_056.png, genuine/payslip_007.png, genuine/payslip_029.png, genuine/bank_stmt_051.png, genuine/payslip_002.png, genuine/bank_stmt_012.png, genuine/id_proof_064.png, genuine/payslip_003.png, genuine/payslip_006.png
  - FN (14): tampered/bank_stmt_018.png (color_shift), tampered/bank_stmt_046.png (noise_patch), tampered/bank_stmt_017.png (blur_region), tampered/bank_stmt_050.png (blur_region), tampered/id_proof_025.png (blur_region), tampered/payslip_045.png (color_shift), tampered/payslip_026.png (color_shift), tampered/bank_stmt_057.png (text_overlay), tampered/bank_stmt_060.png (text_overlay), tampered/payslip_009.png (blur_region), tampered/bank_stmt_055.png (noise_patch), tampered/payslip_034.png (noise_patch), tampered/payslip_035.png (color_shift), tampered/payslip_044.png (color_shift)
- **B + aug** — FP (13): genuine/id_proof_072.png, genuine/payslip_001.png, genuine/payslip_005.png, genuine/bank_stmt_054.png, genuine/payslip_039.png, genuine/payslip_007.png, genuine/payslip_028.png, genuine/payslip_029.png, genuine/payslip_037.png, genuine/bank_stmt_051.png, genuine/bank_stmt_011.png, genuine/bank_stmt_012.png, genuine/id_proof_064.png
  - FN (9): tampered/bank_stmt_018.png (color_shift), tampered/payslip_032.png (color_shift), tampered/bank_stmt_017.png (blur_region), tampered/id_proof_025.png (blur_region), tampered/payslip_026.png (color_shift), tampered/bank_stmt_057.png (text_overlay), tampered/id_proof_062.png (text_overlay), tampered/payslip_009.png (blur_region), tampered/payslip_035.png (color_shift)
- **C + aug** — FP (9): genuine/bank_stmt_048.png, genuine/id_proof_072.png, genuine/payslip_001.png, genuine/payslip_005.png, genuine/payslip_028.png, genuine/payslip_029.png, genuine/bank_stmt_051.png, genuine/bank_stmt_012.png, genuine/id_proof_064.png
  - FN (9): tampered/bank_stmt_018.png (color_shift), tampered/payslip_032.png (color_shift), tampered/bank_stmt_017.png (blur_region), tampered/id_proof_025.png (blur_region), tampered/payslip_026.png (color_shift), tampered/bank_stmt_057.png (text_overlay), tampered/payslip_009.png (blur_region), tampered/payslip_035.png (color_shift), tampered/payslip_044.png (color_shift)

## Augmentation audit — TRAINING DATA

- Rows: 114 ({'tampered': 76, 'genuine': 38}) from 38 training genuine source documents — augmented VIEWS, not independent documents.
- Whole-page processed rate over the full training pool: genuine 0.5, tampered 0.5; untouched: genuine 0.5, tampered 0.5.
- Augmented-only processed rate: {'genuine': 1.0, 'tampered': 0.6447} (genuine variants are always processed because an unprocessed genuine variant would duplicate its source; the pool-level rate is what the model sees).
- Whole-page ops — genuine: {'none': 38, 'global_brightness': 7, 'global_combo': 6, 'global_contrast': 6, 'global_jpeg': 7, 'global_resize': 6, 'global_noise': 6}; tampered: {'none': 49, 'global_brightness': 8, 'global_jpeg': 9, 'global_resize': 8, 'global_noise': 8, 'global_combo': 8, 'global_contrast': 8}.
- Local ops: {'local_blur': 4, 'text_replace_amount': 5, 'local_jpeg': 4, 'region_duplicate': 5, 'text_replace_date': 5, 'local_brightness': 4, 'text_replace_number': 5, 'noise_injection': 4, 'local_contrast': 4, 'local_resample': 4, 'text_replace_name': 5, 'copy_move': 5, 'text_replace_id': 5, 'text_replace_account': 5, 'local_color': 4, 'combo:text_replace_amount+local_resample': 2, 'combo:region_duplicate+local_jpeg': 1, 'local_sharpen': 4, 'combo:text_replace_number+local_jpeg': 1}.
- Fonts used for text edits: {'null': 43, 'arial.ttf': 8, 'tahoma.ttf': 7, 'segoeui.ttf': 7, 'verdana.ttf': 6, 'pillow_default': 5}.
- Single-descriptor label AUC (0.5 = no signal): {'file_bytes': 0.53, 'width': 0.502, 'height': 0.513, 'aspect': 0.509, 'global_brightness': 0.523, 'global_noise': 0.513, 'jpeg_grid': 0.521}.
- Combined shortcut probe (file size, dims, aspect, brightness, noise, JPEG grid, format, mode, metadata; grouped 5-fold): balanced accuracy 0.5036 (chance 0.5).
- Dimensions by label (top): {'genuine': {'850x1050': 25, '850x1100': 22, '850x550': 21, '925x1197': 1, '1001x1295': 1, '915x592': 1}, 'tampered': {'850x1050': 33, '850x1100': 27, '850x550': 26, '830x1074': 1, '772x999': 1, '781x1011': 1}}.
- Distributions by label (p5 / p50 / p95): {'file_bytes': {'genuine': [36885.5, 71291.5, 602403.0], 'tampered': [36763.55, 73674.0, 536872.6]}, 'width': {'genuine': [850.0, 850.0, 874.5], 'tampered': [847.0, 850.0, 932.45]}, 'height': {'genuine': [550.0, 1050.0, 1100.0], 'tampered': [550.0, 1050.0, 1100.0]}, 'aspect': {'genuine': [0.773, 0.81, 1.545], 'tampered': [0.773, 0.81, 1.545]}, 'global_brightness': {'genuine': [217.143, 223.183, 229.803], 'tampered': [216.647, 223.232, 230.031]}, 'global_noise': {'genuine': [0.0, 0.0, 1.0], 'tampered': [0.0, 0.0, 1.0]}, 'jpeg_grid': {'genuine': [0.842, 0.903, 0.957], 'tampered': [0.855, 0.906, 0.999]}}.
- Audit checks: {'processed_rate_gap_le_0.05': True, 'combined_probe_balanced_acc_le_0.60': True, 'every_single_descriptor_auc_le_0.65': True, 'single_format_and_no_metadata': True, 'no_test_source_or_hash': True, 'provenance_complete': True, 'files_match_manifest': True} → PASS.

## Automated validation — all data

- [x] 60 original training images intact
- [x] 15 frozen test images unchanged
- [x] 0 test images in augmentation
- [x] 0 test images in cv
- [x] 0 test records in training cache
- [x] augmentation v1 and v2.0 excluded
- [x] augmentation provenance complete
- [x] whole page processing not label dependent
- [x] no metadata or format leakage
- [x] no dimension leakage
- [x] no compression label leakage
- [x] combined shortcut probe near chance
- [x] sources and augmentations same fold
- [x] no duplicate source leakage
- [x] deterministic seed
- [x] cv folds reproducible

Non-augmented experiments identical to the previous run fold-by-fold: True.

## Comparison with previous runs — CV DATA

| Run | B+aug acc | B+aug F1 | B+aug FP | B+aug FN | C+aug acc | C+aug FP | C+aug FN |
|---|---|---|---|---|---|---|---|
| train-aug-v2.1 | 0.633 | 0.531 | 13 | 9 | 0.700 | 9 | 9 |
| train-aug-v1 | 0.617 | 0.661 | 23 | 0 | 0.583 | 23 | 2 |
| train-aug-v2 | 0.700 | 0.629 | 11 | 7 | 0.717 | 10 | 7 |
