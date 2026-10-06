"""
Tests for features_14.py — 14-Feature PhishNot Extractor
Run with: python backend/test_features_14.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from features_14 import extract_url_features_14, FEATURE_14, N_FEATURES_14

PASS = 0
FAIL = 0

def check(desc: str, got, expected, tol: float = 0):
    global PASS, FAIL
    ok = (abs(got - expected) <= tol) if isinstance(expected, float) else (got == expected)
    status = "PASS" if ok else "FAIL"
    if not ok:
        FAIL += 1
        print(f"  {status}  {desc}")
        print(f"         expected={expected}  got={got}")
    else:
        PASS += 1
        print(f"  {status}  {desc}")


def test_all_keys_present(url: str):
    f = extract_url_features_14(url)
    missing = [k for k in FEATURE_14 if k not in f]
    check(f"all 14 keys present ({url[:40]})", len(missing), 0)
    check(f"exactly 14 keys ({url[:40]})", len(f), N_FEATURES_14)


print("=" * 60)
print("Feature Extractor Tests — features_14.py")
print("=" * 60)

# ── Test 1: All keys always returned ─────────────────────────────
print("\n[1] Key completeness")
for u in [
    "https://example.com",
    "https://example.com/login",
    "http://192.168.1.1/login",
    "malformed-no-scheme",
    "",
]:
    test_all_keys_present(u)

# ── Test 2: Simple hostname ───────────────────────────────────────
print("\n[2] Simple hostname: https://example.com")
f = extract_url_features_14("https://example.com")
check("qty_dot_domain = 1",     f["qty_dot_domain"],     1.0)
check("domain_length = 11",     f["domain_length"],      11.0)   # "example.com"
check("qty_vowels_domain >= 4", f["qty_vowels_domain"],  4.0)    # e,a,e,o
check("directory_length = 0",   f["directory_length"],   0.0)
check("qty_slash_directory = 0",f["qty_slash_directory"],0.0)
check("file_length = 0",        f["file_length"],        0.0)
check("params_length = 0",      f["params_length"],      0.0)

# ── Test 3: URL with path and file ────────────────────────────────
print("\n[3] Path + file: https://example.com/login")
f = extract_url_features_14("https://example.com/login")
check("directory_length = 0",   f["directory_length"],   0.0)   # dir="/"
check("file_length = 5",        f["file_length"],        5.0)   # "login"
check("qty_dot_file = 0",       f["qty_dot_file"],       0.0)

# ── Test 4: Deep path ─────────────────────────────────────────────
print("\n[4] Deep path: https://example.com/a/b/c/file.php?id=123")
f = extract_url_features_14("https://example.com/a/b/c/file.php?id=123")
check("qty_slash_directory = 2", f["qty_slash_directory"], 2.0)  # /a/b/c → 2 internal
check("directory_length >= 4",   f["directory_length"],    5.0)  # "a/b/c"
check("qty_dot_file = 1",        f["qty_dot_file"],        1.0)  # "file.php"
check("file_length = 8",         f["file_length"],         8.0)
check("params_length = 6",       f["params_length"],       6.0)  # "id=123"

# ── Test 5: Subdomain ─────────────────────────────────────────────
print("\n[5] Subdomain: https://sub.example.com/path/file.html?a=1&b=2")
f = extract_url_features_14("https://sub.example.com/path/file.html?a=1&b=2")
check("qty_dot_domain = 2",     f["qty_dot_domain"],     2.0)
check("domain_length = 15",     f["domain_length"],      15.0)  # "sub.example.com"
check("qty_slash_directory = 0",f["qty_slash_directory"],0.0)   # /path → 0 internal
check("file_length = 9",        f["file_length"],        9.0)   # "file.html"
check("qty_dot_file = 1",       f["qty_dot_file"],       1.0)
check("params_length = 7",      f["params_length"],      7.0)   # "a=1&b=2"

# ── Test 6: IP address ────────────────────────────────────────────
print("\n[6] IP: http://192.168.1.1/login")
f = extract_url_features_14("http://192.168.1.1/login")
check("qty_dot_domain = 3",     f["qty_dot_domain"],     3.0)
check("domain_length = 11",     f["domain_length"],      11.0)  # "192.168.1.1"
check("file_length = 5",        f["file_length"],        5.0)   # "login"

# ── Test 7: Trailing slash ────────────────────────────────────────
print("\n[7] Trailing slash: https://example.com/")
f = extract_url_features_14("https://example.com/")
check("file_length = 0",        f["file_length"],        0.0)
check("directory_length = 0",   f["directory_length"],   0.0)

# ── Test 8: Query string only (no path) ──────────────────────────
print("\n[8] Query only: https://example.com?a=123")
f = extract_url_features_14("https://example.com?a=123")
check("params_length = 5",      f["params_length"],      5.0)   # "a=123"

# ── Test 9: Bare domain (no scheme) ──────────────────────────────
print("\n[9] No scheme: paypal-verify.xyz")
f = extract_url_features_14("paypal-verify.xyz")
check("qty_dot_domain = 1",     f["qty_dot_domain"],     1.0)
check("domain_length = 17",     f["domain_length"],      17.0)

# ── Test 10: Malformed URL ────────────────────────────────────────
print("\n[10] Empty URL")
f = extract_url_features_14("")
# Should not raise; all values should be -1
for key in FEATURE_14:
    check(f"{key} = -1 on empty URL", f[key], -1.0)

# ── Summary ───────────────────────────────────────────────────────
print("")
print("=" * 60)
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed  ({FAIL} failed)")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
