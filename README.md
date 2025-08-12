# PhishLens MVP

Real-time phishing detection for the browser with explainable warnings.
Chrome extension + Flask API + dashboard to review detection events.

## Quick Start (Windows)

- __One-click run__: double‑click `run_phishlens.bat`.
  - Creates `.venv` if missing, installs `requirements.txt`, starts backend at `http://127.0.0.1:8000`, and opens `dashboard/index.html`.
- __Load the extension__: `chrome://extensions` → Enable Developer mode → Load unpacked → select `extension/`.

## Project Structure

- `backend/` — Flask server (`/check`, `/events`, `/health`)
- `extension/` — Chrome extension (Manifest V3)
- `dashboard/` — Minimal HTML dashboard that lists detections

## Architecture

```text
                ┌──────────────────────┐
                │   Chrome Extension   │
                │  (Manifest V3)       │
                │                      │
                │  service_worker.js   │
                │   • on tab load      │
                │   • capture HTML     │
                │   • POST /check      │
                └─────────┬────────────┘
                          │ JSON {url, html}
                          ▼
                 ┌──────────────────────┐
                 │     Backend (Flask)  │
                 │  `backend/app.py`    │
                 │  Endpoints:          │
                 │   • /check           │
                 │   • /events (GET/POST)│
                 │   • /health          │
                 │                      │
                 │ Feature Extraction   │
                 │  `features.py`       │
                 │ Heuristic Scoring    │
                 │  `model.py`          │
                 │ Explainability Map   │
                 │  `explain.py`        │
                 └─────────┬────────────┘
                           │ writes events
                           ▼
                 ┌──────────────────────┐
                 │   SQLite Storage     │
                 │  `storage.py`        │
                 │  table: events       │
                 └─────────┬────────────┘
                           │ reads
                           ▼
                 ┌──────────────────────┐
                 │     Dashboard        │
                 │  `dashboard/*.html`  │
                 │  fetch GET /events   │
                 └──────────────────────┘
```

Flow:
- __Extension__ captures page HTML → calls backend `/check`.
- __Backend__ extracts URL/HTML features → scores risk → returns `risk_score`, `reasons`, `highlights` and logs event.
- __Extension__ shows banner/popup; optional highlights on “Explain”.
- __Dashboard__ lists stored events via `/events`.

## Backend: Setup & Run

Windows PowerShell:

```powershell
# From project root
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r backend\requirements.txt

# Run Flask server
python backend\app.py
```

If activation is blocked, you can run commands with the venv python directly:

```powershell
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r backend\requirements.txt
.\.venv\Scripts\python backend\app.py
```

API:
- `GET /health`
- `POST /check` → `{ url, html }` → `{ risk_score, reasons, highlights, meta }`
- `GET /events` → recent detections

## Dashboard

Open `dashboard/index.html` in a browser. It fetches from `http://127.0.0.1:8000/events`.

## Chrome Extension (Manifest V3)

1. Go to `chrome://extensions` → enable Developer mode.
2. Click "Load unpacked" and select the `extension/` folder.
3. Open any HTTP(S) website. The extension will analyze on page load.
4. Click the extension icon to see score and reasons. Use "Explain" to highlight elements.

Ensure backend is running on `http://127.0.0.1:8000`.

## Troubleshooting

- __Dependency build errors__: use the included `requirements.txt` (no Rust toolchain required).
- __Port already in use__: stop other services using 8000 or change port in `backend/app.py`.
- __No banner/highlights__: open DevTools → Console and check for extension errors; ensure backend `/health` is OK.

## Contributing

- __Prereqs__
  - Python 3.10+ on PATH (`py --version`), Chrome for extension testing.
- __Branching__
  - Create feature branches: `git checkout -b feat/<short-name>`.
- __Run locally__
  - `run_phishlens.bat` or follow Backend + Extension steps above.
- __Coding standards__
  - Python: small, testable functions in `backend/`; keep endpoints thin.
  - JS: avoid inline styles where possible; reuse CSS classes like `.phishlens-highlight`.
- __Add dependencies__
  - Add to `requirements.txt` (root) and `backend/requirements.txt` if backend-only.
- __Testing__
  - Manual: test with safe sites (GitHub, Wikipedia) and sample phish URLs.
  - Check `/health`, `/events` in browser.
- __PR checklist__
  - Description includes purpose, screenshots if UI changed, and test notes.
  - No console errors; `POST /check` works on a couple of sites.

## Notes

- Heuristic model only (fast baseline). Swap to ML later via `backend/model.py` and serialized model.
- Event storage: SQLite at `backend/phishlens.db`.
- CORS is permissive for local development.
