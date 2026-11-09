# Dataset — Final Index & Statistics

> Founder C — Day 7. Describes the synthetic dataset in `data/synthetic/` as indexed by
> `data/synthetic/index.csv`. Every number below was produced by
> `scripts/dataset_stats.py` from the files in the repository; none was entered by hand.

**All documents are synthetic.** They were generated with Faker + Pillow, contain no real
personal data, and every image carries a visible "SYNTHETIC / TEST DATA" watermark.
These statistics describe **this dataset only**. They say nothing about real-world
document distributions or real-world tampering.

---

## 1. Overview

| Item | Location |
|---|---|
| Index | `data/synthetic/index.csv` (one row per indexed document) |
| Genuine documents | `data/synthetic/genuine/` |
| Tampered documents | `data/synthetic/tampered/` |
| Day 1 payslips (not indexed) | `data/synthetic/payslip_001.png` … `payslip_005.png` |
| Generator | `scripts/generate_tampered_documents.py` |
| Statistics script | `scripts/dataset_stats.py` (read-only) |
| Validation tests | `tests/test_dataset.py` |

Index columns: `filename, path, doc_type, label, tamper_method, width, height, file_size_bytes`.
`path` is relative to `data/synthetic/` and always equals `<label>/<filename>`.
`tamper_method` is empty for genuine documents.

Documents are numbered `001`–`075` with no gaps: `001`–`025` were created on Day 2 and
`026`–`075` were appended on Day 4.

---

## 2. Final index summary

| Metric | Value |
|---|---|
| Total indexed documents | 75 |
| Day 1 payslips outside the index | 5 |

---

## 3. Genuine / tampered distribution

| Label | Count | Share |
|---|---:|---:|
| genuine | 48 | 64.0% |
| tampered | 27 | 36.0% |

---

## 4. Document-type distribution

| Document type | Count | Share |
|---|---:|---:|
| payslip | 30 | 40.0% |
| bank_statement | 23 | 30.7% |
| id_proof | 22 | 29.3% |

Labels by document type:

| Document type | genuine | tampered | Total | % tampered |
|---|---:|---:|---:|---:|
| payslip | 17 | 13 | 30 | 43.3% |
| bank_statement | 15 | 8 | 23 | 34.8% |
| id_proof | 16 | 6 | 22 | 27.3% |

---

## 5. Tamper-method distribution

| Tamper method | Count | Share of 27 tampered |
|---|---:|---:|
| blur_region | 4 | 14.8% |
| color_shift | 8 | 29.6% |
| noise_patch | 9 | 33.3% |
| text_overlay | 6 | 22.2% |

Tamper methods by document type:

| Document type | blur_region | color_shift | noise_patch | text_overlay |
|---|---:|---:|---:|---:|
| payslip | 1 | 6 | 5 | 1 |
| bank_statement | 2 | 1 | 2 | 3 |
| id_proof | 1 | 1 | 2 | 2 |

Method definitions (`_apply_tampering` in the generator): `blur_region` Gaussian-blurs a
rectangle, `color_shift` shifts the colour of a rectangle, `noise_patch` adds random
noise to a rectangle, `text_overlay` covers a rectangle with "ALTERED VALUE" text.
Each tampered document has exactly one method applied to one region.

---

## 6. Image and file statistics

All indexed images are RGB PNGs.

| Document type | Dimensions (px) | Mean bytes | Min bytes | Max bytes |
|---|---|---:|---:|---:|
| payslip | 850x1050 (30) | 74,639 | 70,187 | 96,618 |
| bank_statement | 850x1100 (23) | 70,106 | 57,247 | 83,190 |
| id_proof | 850x550 (22) | 38,485 | 35,854 | 51,025 |

| File size (all indexed) | Bytes |
|---|---:|
| Total | 4,698,269 |
| Mean | 62,644 |
| Median | 70,777 |
| Min | 35,854 |
| Max | 96,618 |

Observations (factual, from the tables above):

- Image size is fixed by document type: every image is 850 px wide and the height
  depends only on the type. Image dimensions therefore identify the document type,
  not whether a document was tampered.
- The tampered share differs by type (43.3% of payslips vs 27.3% of ID proofs), so
  size- or type-related statistics are partly confounded with the label.

---

## 7. Integrity and validation status

Read-only check performed by `scripts/dataset_stats.py`:

| Integrity check | Count |
|---|---:|
| Missing files | 0 |
| Unreadable images | 0 |
| Size/bytes mismatches vs index.csv | 0 |

`pytest tests/test_dataset.py -v` — **17 passed, 0 failed** (Day 7). The suite checks:
75 indexed rows; 48 genuine / 27 tampered in both the index and the folders; the
30 / 23 / 22 document-type split; every row points to an existing file; no duplicate
filenames or paths; no required field empty; every image opens as a PNG whose size and
byte count match the index; label matches folder; `tamper_method` present exactly on
tampered rows and from the known set; `doc_type` matches the filename prefix; no
unindexed, hidden or non-PNG files; nothing unexpected in `data/synthetic/`; and the
5 Day 1 payslips present and unindexed.

---

## 8. The 5 Day 1 payslips

`data/synthetic/payslip_001.png` … `payslip_005.png` are the original Day 1 test payslips
(850×1050 PNG). They remain **outside** `index.csv` on purpose:

- The indexed dataset was created later (Day 2) with its own `genuine/` / `tampered/`
  layout; the Day 1 files predate it and have no label/tamper metadata.
- `tests/test_analyzer.py` reads images directly from `data/synthetic/`, so these files
  must stay where they are.
- The training code (`core/model_trainer.py`) and `scripts/dataset_stats.py` read only
  `genuine/` and `tampered/`, so the Day 1 files never enter training or statistics.

They share filenames with `genuine/payslip_001.png` etc. but are different files.
**Do not overwrite them** — see the warning in `DATA_FORMAT_NOTES.md`.

### Demo images are not additional dataset documents

- The final indexed synthetic dataset contains **75** documents — exactly the rows of
  `index.csv` in `genuine/` and `tampered/`.
- The 5 Day 1 payslips above are retained **outside** the 75-row index.
- Two of those Day 1 payslips were copied, byte-for-byte, into Founder A's Day 7 demo set:
  `data/demo/genuine_payslip_01.png` = `data/synthetic/payslip_002.png` and
  `data/demo/genuine_payslip_02.png` = `data/synthetic/payslip_003.png`. The remaining demo
  images are derived from augmentations of indexed documents.
- The images in `data/demo/` are therefore **not** counted as additional indexed documents,
  and no statistic in this file includes them. `scripts/dataset_stats.py` reads only
  `data/synthetic/index.csv`, `genuine/` and `tampered/`.

The demo copies are one more reason never to overwrite the Day 1 payslips: the demo
manifest records their SHA-256 and names them as sources.

---

## 9. Reproducing these statistics

From the repository root:

```bash
python scripts/dataset_stats.py                    # plain-text report
python scripts/dataset_stats.py --format markdown  # the tables used in this file
pytest tests/test_dataset.py -v                    # validation suite
```

The script only reads `data/synthetic/index.csv` and the images it references. It prints
to stdout, writes no files, and exits non-zero if any referenced file is missing,
unreadable or inconsistent with the index. Output is deterministic.
