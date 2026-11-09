# Demo Guide — Document Fraud Detection API

Everything needed to run a live demo of the backend: preparation, the demo script,
what to say, API usage for people who want to try it themselves, and fixes for
problems that come up on demo day.

- [Part 1 — Demo execution guide](#part-1--demo-execution-guide)
- [Part 2 — API usage guide](#part-2--api-usage-guide)
- [Part 3 — Troubleshooting guide](#part-3--troubleshooting-guide)

---

## Part 1 — Demo execution guide

### 1.1 The day before

| Check | Command | Expected |
|---|---|---|
| Fresh install works | see [DEPLOYMENT.md §11.2](DEPLOYMENT.md#112-fresh-setup) | no errors |
| Environment is valid | `python scripts/validate_environment.py` | `Overall: PASS` |
| Full workflow works | `python scripts/e2e_test.py` | `Overall result: PASS (65/65 checks passed)` |
| Demo runs end to end | `python scripts/run_demo.py` | `Demo complete.` |
| Demo scores unchanged | compare step 4 with the table in [1.4](#14-what-the-audience-will-see) | identical scores |

Run everything on the laptop you will present from, on the network you will use.
No internet connection is needed during the demo itself.

### 1.2 Thirty minutes before

```bash
cd docfraud-backend
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python scripts/run_demo.py           # dry run; must end with "Demo complete."
```

Close other heavy applications (analysis is CPU-bound), set the terminal font large,
and open <http://127.0.0.1:8000/docs> in a browser tab if you plan to show Swagger.

### 1.3 Running the demo

**Option A — one command (recommended).** Starts the API, walks through four steps and stops the API:

```bash
python scripts/run_demo.py --pause   # press Enter to advance each step
```

**Option B — live server + Swagger UI.** Better for technical audiences:

```bash
export API_KEY="demo-presentation-key"        # PowerShell: $env:API_KEY="demo-presentation-key"
uvicorn backend.main:app --port 8000
```

Then in the browser at <http://127.0.0.1:8000/docs>: click **Authorize**, enter the key,
open **POST /verify** → **Try it out**, choose a file from `data/demo/` and **Execute**.

**Option C — curl.** See [Part 2](#part-2--api-usage-guide).

### 1.4 What the audience will see

| Step | Shows | Result |
|---|---|---|
| 1. Health check | Service up, model loaded, version | `200`, `"status": "ok"`, `"model_loaded": true` |
| 2. Security | Missing key / wrong key | `401` / `403` with a clear message |
| 3. Validation | A `.txt` upload | `400 Invalid file format` |
| 4. Analysis | The 5 demo documents | table below, ~0.3 s each |

| Document | Actual | Risk score | Verdict | Correct |
|---|---|---|---|---|
| genuine_payslip_01.png | genuine | 0.06 | genuine | yes |
| genuine_payslip_02.png | genuine | 0.06 | genuine | yes |
| tampered_payslip_01.png | tampered | 0.79 | tampered | yes |
| tampered_payslip_02.png | tampered | 0.15 | genuine | **no** |
| edge_case_01.png | genuine | 0.22 | genuine | yes |

Under `tampered_payslip_01.png` the demo prints the region the API located:
`-> ink region at x=198, y=321 (96 x 9 px)`, the replaced account number (its ink darkness
is rare on the page). Documents below the threshold get no regions.

These scores are for model v2 (Day 8) and are deterministic: the same files and model always
give the same numbers. (Model v1 scored 0.36 / 0.59 / 0.37 / 0.14 / 0.35, 2 of 5 correct.)

### 1.5 Suggested talk track

1. **Problem.** Edited payslips, bank statements and ID documents are a common fraud
   vector; manual review is slow.
2. **What the backend does.** Upload → API-key check → validation → 29 local tamper
   measurements (does any region's noise, colour, sharpness or ink differ from the rest of
   the page?) → Random Forest → risk score and explanation, in about 0.3 seconds per document.
3. **Engineering quality** (steps 1–3). Authentication, strict validation, uniform
   errors, request IDs in logs, a readiness-aware health check, protection against
   oversized uploads and decompression bombs, 338 automated tests.
4. **Results — say it plainly.** Model v2 reaches 86.7% on the 15 frozen test documents
   and 92.6% on 149 fresh held-out documents (v1: 60%), 4 of 5 on the demo set. All of
   this is synthetic data from one generator; re-saved or cropped files are handled less
   reliably, and real documents have not been tested.
5. **Where.** For a flagged document the API also returns boxes around the suspicious
   parts (`flagged_regions`); on held-out synthetic documents a box lands on 82% of visible
   edits. Text overlays whose ink matches the page are often missed.
6. **Next step.** Real consented documents, robustness to re-saving and cropping, and score
   calibration.

Do **not** present the score as a probability of fraud, or the demo set as an accuracy
measurement. See [README — Limitations](README.md#limitations).

### 1.6 Backup plan

| Problem during the demo | Fallback |
|---|---|
| Server will not start | Show `logs/e2e_test_report.txt` and `logs/final_release_report.txt` |
| Port 8000 busy | `python scripts/run_demo.py --port 8001` |
| Laptop is slow | Close other apps; results are the same, only timings change |
| No projector for the terminal | Swagger UI in the browser (Option B) |

---

## Part 2 — API usage guide

Full reference: [API_DOCS.md](API_DOCS.md). Base URL for a local run: `http://127.0.0.1:8000`.

### 2.1 Authentication

Send the server's `API_KEY` in the `X-API-Key` header on every `/verify` call.
`/health`, `/docs`, `/redoc` and `/openapi.json` need no key.

### 2.2 Check the service

```bash
curl http://127.0.0.1:8000/health
```

`200` with `"status": "ok"` means the model is loaded and `/verify` will work.
`503` with `"status": "degraded"` means it will not; `detail` says why.

### 2.3 Verify a document

**curl (macOS / Linux / Git Bash)**

```bash
curl -X POST http://127.0.0.1:8000/verify \
  -H "X-API-Key: demo-presentation-key" \
  -F "file=@data/demo/genuine_payslip_01.png"
```

**PowerShell**

```powershell
curl.exe -X POST http://127.0.0.1:8000/verify `
  -H "X-API-Key: demo-presentation-key" `
  -F "file=@data/demo/genuine_payslip_01.png"
```

**Python**

```python
import requests

with open("data/demo/genuine_payslip_01.png", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/verify",
        headers={"X-API-Key": "demo-presentation-key"},
        files={"file": ("genuine_payslip_01.png", f, "image/png")},
        timeout=60,
    )
print(r.status_code, r.headers["X-Request-ID"], r.json())
```

**JavaScript (browser or Node 18+)**

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);            // an <input type="file">
const r = await fetch("http://127.0.0.1:8000/verify", {
  method: "POST",
  headers: { "X-API-Key": "demo-presentation-key" },
  body: form,
});
console.log(r.status, await r.json());
```

Browser calls from another origin work once that origin is allowed: start the server
with `CORS_ORIGINS=https://your-frontend.example` (the default `*` allows any origin).

### 2.4 Reading the response

```json
{
  "tamper_risk_score": 0.0567,
  "confidence": 0.9433,
  "flagged_regions": [],
  "interpretation": "Document verified successfully. Low risk of digital tampering detected (risk score: 0.06, confidence: 0.94)."
}
```

| Field | Meaning |
|---|---|
| `tamper_risk_score` | Share of the model's trees voting "tampered" (0–1). At or above 0.5 the interpretation reports tampering. **Not** a calibrated probability. |
| `confidence` | Vote share of the predicted class |
| `flagged_regions` | Boxes (`x`, `y`, `width`, `height` in image pixels, `area`, `type`) around the suspicious parts of the page. Empty for genuine-looking documents; only filled when the score reaches 0.5 |
| `interpretation` | One-sentence summary |

### 2.5 Input rules

| Rule | Limit | Error |
|---|---|---|
| File types | `.jpg`, `.jpeg`, `.png`, `.pdf` (first page only) | `400` |
| File size | 1 byte – 10 MB | `400` empty, `413` too large |
| Image size | at most 40 megapixels; shortest side ≥ 16 px | `400` |
| PDFs | not password-protected, at least one page | `400` |

Every error has the same shape: `{"success": false, "error": "...", "code": 400, "detail": "..."}`.
Every response has an `X-Request-ID` header; quote it when reporting a problem.

---

## Part 3 — Troubleshooting guide

### 3.1 Setup and start-up

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'backend'` | Command run outside the repository root | `cd docfraud-backend` first |
| `ModuleNotFoundError: No module named 'fastapi'` (in `pytest`) | A global `pytest` without the project's packages | Activate the venv; use `python -m pytest` |
| `Address already in use` | Port taken by an earlier server | Use `--port 8001`, or stop the other process |
| Start-up log: `Starting in DEGRADED mode` | Model file missing or unreadable | Restore `data/models/fraud_model.pkl` from git; the next request reloads it |
| Start-up log: `API_KEY is the public default` | `API_KEY` not set | Fine for a local demo; set it before sharing the server |
| `ImportError: libGL.so.1` (Linux) | OpenCV system library missing | `sudo apt-get install -y libgl1 libglib2.0-0` |

### 3.2 Requests

| Response | Meaning | Fix |
|---|---|---|
| `401 Missing X-API-Key header` | No key sent | Add `-H "X-API-Key: <key>"` |
| `403 Invalid API Key` | Key does not match the server's | Use the value the server was started with |
| `400 Invalid file format` | Unsupported extension | Use `.jpg`, `.jpeg`, `.png` or `.pdf` |
| `400 … corrupted file` | File cannot be decoded | Re-export the image; check it opens normally |
| `400 … pixel limit` / `too small` | Image over 40 MP or under 16 px on a side | Resize it |
| `413 Payload Too Large` | File over 10 MB | Compress or resize it |
| `503 … not available` | Model not loaded | See `/health` → `detail`; restore the model file |
| `500 … (request id X)` | Unexpected error | Find `req=X` in the server log for the full traceback |
| Browser: CORS error | Frontend origin not allowed | Restart with `CORS_ORIGINS=<your origin>` |

### 3.3 Results look different

| Symptom | Cause | Fix |
|---|---|---|
| Demo scores differ from the table in 1.4 | Different scikit-learn / numpy / Pillow / OpenCV versions | Compare `pip freeze` with `logs/environment_validation.txt`; reinstall matching versions |
| Slow responses | Other CPU-heavy apps, or many requests at once | Close apps; about 8-9 documents/s is the expected maximum per process (4 CPUs) |
| A real document gets a surprising score | The model has only seen synthetic documents | Expected at this stage; see README Limitations |

More: [DEPLOYMENT.md §9](DEPLOYMENT.md#9-troubleshooting) and
[DEPLOYMENT.md §11.6](DEPLOYMENT.md#116-machine-specific-problems).
