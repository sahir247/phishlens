from typing import Dict, List, Tuple
from urllib.parse import urlparse

REASON_TEMPLATES = {
    "form_action_diff_domain": "Form submits to a different domain",
    "form_insecure_http": "Form submits over insecure HTTP",
    "logo_mismatch": "Contains brand logo but domain does not match",
    "brand_text_hit": "Brand name appears in title/meta",
    "suspicious_kw": "Suspicious keywords present in URL",
    "num_pw_inputs": "Page asks for a password",
    "has_ip": "URL uses an IP address instead of domain",
    "num_at": "URL contains '@' which can obfuscate destination",
    "subdomain_count": "Unusually many subdomains",
    "entropy_path": "High URL entropy (random-looking path/query)",
}


def reasons_for(features: Dict[str, float], url: str) -> Tuple[List[str], List[str]]:
    reasons: List[str] = []
    selectors: List[str] = []

    def add(key: str):
        label = REASON_TEMPLATES.get(key)
        if label:
            reasons.append(label)

    # Map based on feature flags/thresholds
    if features.get("form_action_diff_domain", 0) > 0.5:
        add("form_action_diff_domain")
        selectors.append("form")
    if features.get("form_insecure_http", 0) > 0.5:
        add("form_insecure_http")
        selectors.append("form")
    if features.get("logo_mismatch", 0) > 0.5:
        add("logo_mismatch")
        selectors.append("img")
    if features.get("brand_text_hit", 0) > 0.5:
        add("brand_text_hit")
    if features.get("suspicious_kw", 0) > 0.5:
        add("suspicious_kw")
    if features.get("num_pw_inputs", 0) >= 1:
        add("num_pw_inputs")
        selectors.append("input[type='password']")
    if features.get("has_ip", 0) > 0.5:
        add("has_ip")
    if features.get("num_at", 0) >= 1:
        add("num_at")
    if features.get("subdomain_count", 0) >= 3:
        add("subdomain_count")
    if features.get("entropy_path", 0) >= 4.0:
        add("entropy_path")

    # Ensure unique selectors
    selectors = list(dict.fromkeys(selectors))

    return reasons, selectors
