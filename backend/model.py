"""
PhishLens Ensemble Risk Scoring Model
--------------------------------------
Hierarchical 5-category ensemble:
  1. domain_risk   — IP, punycode, suspicious TLD, deep subdomains
  2. url_risk      — obfuscation, long URLs, high path entropy, bad chars
  3. brand_risk    — typosquatting, logo mismatch, brand-in-subdomain
  4. dom_risk      — cross-domain forms, insecure inputs, hidden iframes
  5. ml_risk       — GradientBoostingClassifier probability (sklearn)

Non-linear boosters fire on the highest-confidence phishing combos.
"""

import math
from typing import Dict, Any, Tuple


def calculate_domain_risk(feat: Dict[str, Any]) -> float:
    score = 0.0
    if feat.get("has_ip", 0.0) > 0.5:
        score += 0.40
    if feat.get("has_punycode", 0.0) > 0.5:
        score += 0.35
    if feat.get("is_suspicious_tld", 0.0) > 0.5:
        score += 0.30

    subdomain_count = float(feat.get("subdomain_count", 0.0))
    if subdomain_count >= 3:
        score += min(0.10 * (subdomain_count - 1), 0.30)

    domain_len = float(feat.get("domain_len", 0.0))
    if domain_len > 35:
        score += 0.15

    return min(score, 1.0)


def calculate_url_risk(feat: Dict[str, Any]) -> float:
    score = 0.0
    if feat.get("num_at", 0.0) >= 1.0:
        score += 0.35
    if feat.get("has_port", 0.0) > 0.5:
        score += 0.20

    kw_hits = float(feat.get("kw_hit_count", 0.0))
    if kw_hits > 0:
        score += min(0.20 + 0.10 * (kw_hits - 1), 0.45)

    num_dashes = float(feat.get("num_dashes", 0.0))
    if num_dashes >= 3:
        score += min(0.05 * num_dashes, 0.25)

    entropy = float(feat.get("entropy_path", 0.0))
    if entropy > 4.5:
        score += 0.25
    elif entropy > 3.8:
        score += 0.15

    url_len = float(feat.get("url_len", 0.0))
    if url_len > 120:
        score += 0.15

    if feat.get("has_https", 1.0) < 0.5:
        score += 0.15

    return min(score, 1.0)


def calculate_brand_risk(feat: Dict[str, Any]) -> float:
    score = 0.0
    typosquat = float(feat.get("typosquat_score", 0.0))
    if typosquat >= 0.7:
        score += typosquat * 0.55

    if feat.get("brand_in_subdomain", 0.0) > 0.5:
        score += 0.45

    if feat.get("logo_mismatch", 0.0) > 0.5:
        score += 0.40

    if feat.get("brand_text_hit", 0.0) > 0.5:
        score += 0.25

    return min(score, 1.0)


def calculate_dom_risk(feat: Dict[str, Any]) -> float:
    score = 0.0
    if feat.get("form_action_diff_domain", 0.0) > 0.5:
        score += 0.45
    if feat.get("form_insecure_http", 0.0) > 0.5:
        score += 0.35
    if feat.get("insecure_password_input", 0.0) > 0.5:
        score += 0.40
    if feat.get("empty_action_form", 0.0) > 0.5:
        score += 0.25
    if feat.get("hidden_iframes", 0.0) > 0.5:
        score += 0.20

    num_pw = float(feat.get("num_pw_inputs", 0.0))
    if num_pw > 0 and feat.get("has_https", 1.0) < 0.5:
        score += 0.30

    return min(score, 1.0)


def predict_risk(feat: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
    """
    Hybrid ensemble: 4 heuristic category scores + ML classifier probability.
    Returns (risk_score 0–1, risk_level str, category_scores dict).
    """
    # ── Heuristic category scores ─────────────────────────────────────────────
    domain_score = calculate_domain_risk(feat)
    url_score    = calculate_url_risk(feat)
    brand_score  = calculate_brand_risk(feat)
    dom_score    = calculate_dom_risk(feat)

    # ── ML probability (lazy-loads / trains the pkl on first call) ────────────
    try:
        from ml_model import get_ml_proba
        ml_score = get_ml_proba(feat)
    except Exception:
        ml_score = 0.0

    # ── Weighted linear combination (weights sum to 1.0) ──────────────────────
    raw_score = (
        domain_score * 0.20 +
        url_score    * 0.15 +
        brand_score  * 0.25 +
        dom_score    * 0.20 +
        ml_score     * 0.20
    )

    # ── Non-linear boosters for high-confidence phishing combos ──────────────
    # 1. Brand spoofing + credential input → critical
    if brand_score > 0.4 and (feat.get("num_pw_inputs", 0.0) > 0 or dom_score > 0.3):
        raw_score = max(raw_score, 0.82 + 0.15 * brand_score)

    # 2. Cross-domain form + password → critical phishing
    if feat.get("form_action_diff_domain", 0.0) > 0.5 and feat.get("num_pw_inputs", 0.0) > 0:
        raw_score = max(raw_score, 0.88)

    # 3. IP hostname + suspicious keyword + password → blatant phishing
    if feat.get("has_ip", 0.0) > 0.5 and feat.get("suspicious_kw", 0.0) > 0.5:
        raw_score = max(raw_score, 0.85)

    # 4. ML is highly confident by itself → enforce at least SUSPICIOUS
    if ml_score >= 0.85:
        raw_score = max(raw_score, 0.55)

    final_score = round(max(0.0, min(1.0, raw_score)), 3)

    if final_score >= 0.80:
        risk_level = "DANGEROUS"
    elif final_score >= 0.50:
        risk_level = "SUSPICIOUS"
    else:
        risk_level = "SAFE"

    category_scores = {
        "domain_risk": round(domain_score, 3),
        "url_risk":    round(url_score, 3),
        "brand_risk":  round(brand_score, 3),
        "dom_risk":    round(dom_score, 3),
        "ml_risk":     round(ml_score, 3),
    }

    return final_score, risk_level, category_scores
