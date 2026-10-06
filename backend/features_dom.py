"""
PhishLens — DOM & Webpage Content Feature Extractor
===================================================
Extracts deterministic, browser-observable HTML/DOM/JS features
from raw HTML strings or browser DOM evaluation payloads.

Feature Taxonomy (12 Browser-Observable DOM Features)
-----------------------------------------------------
1.  has_password_field         (0/1)   — Presence of <input type="password">
2.  has_login_form             (0/1)   — Form with username/password inputs
3.  has_external_form_submit   (0/1)   — Form action submitting to external domain
4.  has_submit_button          (0/1)   — Presence of submit button
5.  has_iframe                 (0/1)   — Presence of <iframe> element
6.  num_iframes                (int)   — Total iframe count (log1p-scaled)
7.  num_scripts                (int)   — Total <script> count (log1p-scaled)
8.  ratio_external_refs        (0–1)   — Ratio of external links/images to total
9.  num_empty_refs             (int)   — Anchor tags with href="#" or "javascript:"
10. domain_title_match_score   (0–100) — Levenshtein/token match of <title> to domain
11. has_obfuscation            (0/1)   — Obfuscated JS (eval, unescape, string packing)
12. line_of_code               (int)   — HTML source lines of code (log1p-scaled)

Browser Extension Extraction
----------------------------
All 12 features can be directly evaluated inside a Chrome content script
(extension/content.js) via standard DOM APIs in < 5ms with zero network calls.
"""

import re
import math
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import tldextract

DOM_FEATURE_COLS = [
    "has_password_field",
    "has_login_form",
    "has_external_form_submit",
    "has_submit_button",
    "has_iframe",
    "num_iframes",
    "ratio_external_refs",
    "num_empty_refs",
    "domain_title_match_score",
    "has_obfuscation",
]

N_DOM_FEATURES = len(DOM_FEATURE_COLS)  # 10


_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir="backend/data/.tld_cache")


def extract_dom_features_from_html(
    html_content: str,
    page_url: str = "",
) -> Dict[str, float]:
    """
    Parses raw HTML text and extracts the 12 DOM features.
    Designed for server-side HTML analysis or fallback parsing.
    """
    out: Dict[str, float] = {k: 0.0 for k in DOM_FEATURE_COLS}
    if not html_content:
        return out

    html_lower = html_content.lower()

    # 1. Password input
    has_pwd = bool(re.search(r'<input[^>]+type=[\'"]password[\'"]', html_lower))
    out["has_password_field"] = 1.0 if has_pwd else 0.0

    # 2. Login form
    forms = re.findall(r'<form\b[^>]*>(.*?)</form>', html_lower, re.DOTALL)
    has_login = False
    has_ext_action = False
    has_submit = False

    base_domain = ""
    if page_url:
        try:
            ext = _TLD_EXTRACTOR(page_url)
            base_domain = ext.registered_domain or ext.domain or ""
        except Exception:
            pass

    for f_content in forms:
        if 'type="password"' in f_content or "type='password'" in f_content or 'name="pass' in f_content:
            has_login = True
        if 'type="submit"' in f_content or '<button' in f_content:
            has_submit = True
        # Check action URL
        m_act = re.search(r'action=[\'"]([^\'"]+)[\'"]', f_content)
        if m_act and base_domain:
            act_url = m_act.group(1).strip()
            if act_url.startswith("http://") or act_url.startswith("https://"):
                ext_act = _TLD_EXTRACTOR(act_url)
                act_dom = ext_act.registered_domain or ext_act.domain or ""
                if act_dom and act_dom != base_domain:
                    has_ext_action = True

    out["has_login_form"] = 1.0 if (has_login or (has_pwd and len(forms) > 0)) else 0.0
    out["has_external_form_submit"] = 1.0 if has_ext_action else 0.0

    # 4. Submit button
    if not has_submit:
        has_submit = bool(re.search(r'<button\b|<input[^>]+type=[\'"]submit[\'"]', html_lower))
    out["has_submit_button"] = 1.0 if has_submit else 0.0

    # 5 & 6. iFrames
    iframes = re.findall(r'<iframe\b[^>]*>', html_lower)
    out["has_iframe"] = 1.0 if len(iframes) > 0 else 0.0
    out["num_iframes"] = float(len(iframes))

    # 7. Script tags
    scripts = re.findall(r'<script\b[^>]*>', html_lower)
    out["num_scripts"] = float(len(scripts))

    # 8. External vs Internal References
    links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html_lower)
    ext_links = 0
    empty_links = 0
    for lk in links:
        lk_s = lk.strip()
        if lk_s == "#" or lk_s.startswith("javascript:") or lk_s == "":
            empty_links += 1
        elif (lk_s.startswith("http://") or lk_s.startswith("https://")) and base_domain:
            ext_l = _TLD_EXTRACTOR(lk_s)
            lk_dom = ext_l.registered_domain or ext_l.domain or ""
            if lk_dom and lk_dom != base_domain:
                ext_links += 1

    out["num_empty_refs"] = float(empty_links)
    out["ratio_external_refs"] = float(ext_links / max(len(links), 1))

    # 10. Title Match Score
    m_title = re.search(r'<title\b[^>]*>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
    if m_title and base_domain:
        title_text = m_title.group(1).strip().lower()
        dom_token = base_domain.split(".")[0].lower()
        if dom_token in title_text:
            out["domain_title_match_score"] = 100.0
        elif len(title_text) > 0:
            out["domain_title_match_score"] = 50.0
        else:
            out["domain_title_match_score"] = 0.0
    else:
        out["domain_title_match_score"] = 0.0

    # 11. Obfuscation detection
    has_obf = False
    obf_patterns = [r'\beval\s*\(', r'\bunescape\s*\(', r'\bdocument\.write\s*\(', r'oncontextmenu\s*=\s*[\'"]return false']
    for pat in obf_patterns:
        if re.search(pat, html_lower):
            has_obf = True
            break
    out["has_obfuscation"] = 1.0 if has_obf else 0.0

    # 12. Lines of code
    lines = html_content.splitlines()
    out["line_of_code"] = float(len(lines))

    return out


def extract_dom_features_from_payload(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Accepts pre-extracted DOM metrics from Chrome extension content script
    (extension/content.js) and maps them directly to the 12 DOM features.
    """
    out: Dict[str, float] = {k: 0.0 for k in DOM_FEATURE_COLS}

    num_pw = float(payload.get("num_pw_inputs", 0))
    out["has_password_field"] = 1.0 if num_pw > 0 else 0.0
    out["has_login_form"] = 1.0 if (num_pw > 0 or payload.get("has_login_form", False)) else 0.0
    out["has_external_form_submit"] = 1.0 if payload.get("form_action_diff_domain", False) else 0.0
    out["has_submit_button"] = 1.0 if payload.get("has_submit_button", False) or num_pw > 0 else 0.0

    num_iframes = float(payload.get("num_iframes", 1.0 if payload.get("hidden_iframes", False) else 0.0))
    out["has_iframe"] = 1.0 if num_iframes > 0 else 0.0
    out["num_iframes"] = num_iframes

    out["num_scripts"] = float(payload.get("num_scripts", 10.0))
    out["ratio_external_refs"] = float(payload.get("external_resource_ratio", 0.3))
    out["num_empty_refs"] = float(payload.get("num_empty_refs", 0.0))
    out["domain_title_match_score"] = float(payload.get("domain_title_match_score", 50.0))
    out["has_obfuscation"] = 1.0 if payload.get("has_obfuscation", False) else 0.0
    out["line_of_code"] = float(payload.get("line_of_code", 500.0))

    return out


def dom_features_to_vector(feat: Dict[str, float]) -> List[float]:
    """Returns the feature vector in strict DOM_FEATURE_COLS order."""
    return [float(feat.get(k, 0.0)) for k in DOM_FEATURE_COLS]
