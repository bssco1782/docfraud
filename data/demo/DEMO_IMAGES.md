# DocFraud Demo Dataset

## Purpose

Five images for API / frontend demonstration only. They show what the current pipeline returns for a clean payslip, an edited payslip and a compressed page. **This is not a validation set.** Five images cannot measure accuracy; the valid measurement is the model v2 held-out results: 86.67% (13/15) on the frozen test set and 92.62% (138/149) on the generated holdout set (`logs/model_validation_report.txt`). Predictions below are recorded as returned, including wrong ones.

All documents are synthetic and watermarked "SYNTHETIC / TEST DATA". No real PII.

## Demo Images

### 1. Genuine Payslip 01
- File: `genuine_payslip_01.png`
- Actual label: **genuine** (from Day-1 generator output (scripts/generate_tampered_documents.py @ bf18a4c, generate_payslip_batch: render only, no tampering code); PROGRESS.md Day 1)
- Prediction: **genuine**
- Risk score: 0.0567 (confidence 0.9433)
- Correct: YES
- Dimensions: 850x1050 RGB, 70,484 bytes
- Provenance: data/synthetic/payslip_002.png - Founder C Day-1 synthetic payslip, commit bf18a4c
- SHA-256: `257b7a496877f57c83b7eea739808c7380b7f4b9c5399924b93ac807683da97e`
- Visible change: None - untouched generator output.
- Relation to training data: Image not in any training set. Sibling record: training image genuine/payslip_002.png has the same company, employee, employee ID and bank account (different month and salary figures).

### 2. Genuine Payslip 02
- File: `genuine_payslip_02.png`
- Actual label: **genuine** (from Day-1 generator output (scripts/generate_tampered_documents.py @ bf18a4c, generate_payslip_batch: render only, no tampering code); PROGRESS.md Day 1)
- Prediction: **genuine**
- Risk score: 0.0567 (confidence 0.9433)
- Correct: YES
- Dimensions: 850x1050 RGB, 71,041 bytes
- Provenance: data/synthetic/payslip_003.png - Founder C Day-1 synthetic payslip, commit bf18a4c
- SHA-256: `b6f60b5562aeb19c1e3416be9f2b6305e855021f8b04cd3e58ae5161692ec6a2`
- Visible change: None - untouched generator output.
- Relation to training data: Image not in any training set. No matching record in the 75-image dataset (nearest by pixels: genuine/payslip_006.png, a different company and employee).

### 3. Tampered Payslip 01
- File: `tampered_payslip_01.png`
- Actual label: **tampered** (from data/augmented/train_v2_1/manifest.csv (aug_version train-aug-v2.1))
- Prediction: **tampered**
- Risk score: 0.7933 (confidence 0.7933)
- Correct: YES
- Dimensions: 850x1050 RGB, 70,624 bytes
- Provenance: data/augmented/train_v2_1/tampered/payslip_002_aug1.png - view of training document genuine/payslip_002.png (source sha256 d7dd343ba62e...); local edit: text_replace_account {"bg": [255, 255, 255], "font": "tahoma.ttf", "font_size": 12, "ink": [55, 55, 55], "mean_abs_change": 65.195, "text": "73437992415073"} bbox [197, 321, 55, 12]; whole-page: none {}
- SHA-256: `0c495d11d65f1be576a87c77bf0605b89f946337b5c261db09d51b3032654b23`
- Visible change: Earnings table: the Medical Allowance value '₹ 1,568' is overwritten with the 14-digit string '73437992415073' in a different font. Visible on close inspection; not a plausible salary edit.
- Relation to training data: Image not in any training set of the deployed model, but its source document genuine/payslip_002.png is a training image of it; only the edit is new.

### 4. Tampered Payslip 02
- File: `tampered_payslip_02.png`
- Actual label: **tampered** (from data/augmented/train_v2_1/manifest.csv (aug_version train-aug-v2.1))
- Prediction: **genuine**
- Risk score: 0.1533 (confidence 0.8467)
- Correct: NO
- Dimensions: 850x1050 RGB, 70,828 bytes
- Provenance: data/augmented/train_v2_1/tampered/payslip_037_aug1.png - view of training document genuine/payslip_037.png (source sha256 9a5ffcd59c68...); local edit: combo:text_replace_amount+local_resample {"local_resample": {"factor": 0.3663854649463466, "feather_sigma": 2.3409473717402043, "upsample": "bilinear"}, "mean_abs_change": 21.43, "text_replace_amount": {"bg": [255, 255, 255], "font": "pillow_default", "font_size": 15, "ink": [81, 81, 81], "text": "2,290,395"}} bbox [15, 130, 134, 25]; whole-page: none {}
- SHA-256: `5d8f1df01e4157f09eeaa724ad2d14c71881eaf1eb2324c3420c34e2db68d8d4`
- Visible change: Employee details: the 'Employee Name:' label is overwritten with a blurred '2,290,395' (text replacement + local resampling). Visible; not a plausible salary edit.
- Relation to training data: Image not in any training set of the deployed model, but its source document genuine/payslip_037.png is a training image of it; only the edit is new.

### 5. Edge Case
- File: `edge_case_01.png`
- Actual label: **genuine** (from data/augmented/train_v2_1/manifest.csv (aug_version train-aug-v2.1))
- Prediction: **genuine**
- Risk score: 0.2233 (confidence 0.7767)
- Correct: YES
- Dimensions: 850x1050 RGB, 313,775 bytes
- Provenance: data/augmented/train_v2_1/genuine/payslip_006_aug2.png - view of training document genuine/payslip_006.png (source sha256 1da476683b40...); local edit: none {} bbox null; whole-page: global_jpeg {"quality": 78}
- SHA-256: `00b9613d10a1c6077a91624969e7d6ff1e1afebf1a3376a522b8a722fed6b8e4`
- Visible change: Whole page JPEG-compressed (quality 78); no content change. File is ~4.5x larger than the other PNGs because JPEG noise compresses poorly as PNG.
- Relation to training data: Image not in any training set of the deployed model, but its source document genuine/payslip_006.png is a training image of it; only the compression is new.
- Edge-case type: JPEG-compressed page (whole-page {"quality": 78}), stored as PNG

Demo predictions correct: 4/5 (not an accuracy estimate).

## Selection Rules

Fixed in `scripts/prepare_demo_dataset.py` before scoring; predictions played no part.

- Genuine: first two Day-1 payslips (`data/synthetic/payslip_*.png`, outside `index.csv`) that are not a near-copy of any of the 75 dataset images (< 0.4% of pixels differing). `payslip_001.png` is excluded: the Day-2 generator reused the Faker seed and it is a near-copy of training image `genuine/payslip_001.png`. Selected: `payslip_002.png`, `payslip_003.png`.
- Tampered: first two `train-aug-v2.1` tampered payslip views whose only change is a local text replacement (`text_replace_*`, no whole-page processing). The augmenter writes the new value over a detected text token, which is not always the field the edit type names - see "Visible change".
- Edge case: first `train-aug-v2.1` genuine payslip view whose only change is whole-page JPEG compression (typical of forwarded or re-saved scans).

## Integrity Checks

- Frozen test overlap: PASS (file hashes, source documents, pixel near-copies)
- Hash verification: PASS (demo copies byte-identical to sources; augmented sources match manifest; 15/15 frozen test hashes unchanged)
- Label provenance: PASS
- Demo count: 5
- Genuine payslips: 2
- Tampered payslips: 2
- Edge cases: 1

Full check list: `demo_manifest.json`.

## Model Information

- Artifact: `data/models/fraud_model.pkl` (SHA-256 `1d374648e967af72...`)
- Version: 2.0.0, feature schema local-tamper-v2 (29 features, `core.tamper_features.extract_tamper_features`)
- RandomForest {'n_estimators': 300, 'class_weight': 'balanced', 'random_state': 42}; trained 2026-09-27T08:56:01.350616+00:00 on 210 documents (60 frozen TRAIN + 150 generated DEV, plus processed copies)
- Scoring path: `backend.model_utils.ModelAPI.predict` (same as `/verify`)
- Predicted label: tampered if `tamper_risk_score >= 0.5` (ModelAPI's rule, not tuned)
- Scored once at 2026-09-27T09:05:37+00:00 (UTC)

## Scoring History

Earlier models scored the same five files once each (kept in `demo_manifest.json`):

- Model 1.0.0 (day4-v2-22, SHA-256 `bcf4d02f15980f68...`), scored 2026-09-25T10:14:17+00:00: 2/5 correct - genuine_payslip_01.png 0.36, genuine_payslip_02.png 0.59, tampered_payslip_01.png 0.37, tampered_payslip_02.png 0.14, edge_case_01.png 0.35

## Limitations

- None of the 5 files is a training image of model v2, but images 3-5 are edited / compressed views of its training documents (frozen TRAIN images), and genuine payslip 01 is a sibling of training image `genuine/payslip_002.png`. Only genuine payslip 02 has no related training record.
- Model v2 was trained on the generator's four edit types (blur, colour shift, noise patch, text overlay). The two tampered demo payslips use a different edit (text replacement from the Day-5 augmenter), so they test generalisation to an edit type the model has not seen.
- The two tampered payslips are Founder A's Day-5 training augmentations, not Founder C's dataset tampering methods. Their edits land on the wrong fields (an account-style number in an allowance cell, an amount over a field label), so they are not realistic fraud scenarios.
- Risk score is an uncalibrated RandomForest vote fraction, not a fraud probability.
- `/verify` returns no flagged regions, so the demo cannot show where an edit is.
- Synthetic documents only; real NBFC documents may behave differently.
- Not production-ready.
