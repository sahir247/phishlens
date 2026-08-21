from typing import Dict, List, Tuple, Any

REASON_DEFINITIONS = {
    "typosquatting": {
        "title": "Suspected Brand Typosquatting",
        "desc": "Domain name closely mimics a trusted brand ({brand})",
        "severity": "CRITICAL",
    },
    "brand_in_subdomain": {
        "title": "Deceptive Brand Subdomain",
        "desc": "Uses brand name '{brand}' in subdomain to deceive visitors",
        "severity": "CRITICAL",
    },
    "form_action_diff_domain": {
        "title": "Cross-Domain Form Submission",
        "desc": "Login or input form submits data to a different, external domain",
        "severity": "CRITICAL",
    },
    "insecure_password_input": {
        "title": "Unencrypted Password Field",
        "desc": "Page collects passwords over unencrypted HTTP protocol",
        "severity": "CRITICAL",
    },
    "has_ip": {
        "title": "Raw IP Hostname",
        "desc": "URL uses a raw IP address instead of a standard domain name",
        "severity": "HIGH",
    },
    "has_punycode": {
        "title": "IDN Homograph / Punycode",
        "desc": "Contains internationalized punycode characters resembling real letters",
        "severity": "HIGH",
    },
    "logo_mismatch": {
        "title": "Spoofed Brand Logo",
        "desc": "Page displays brand logos ({brand}) while hosted on an unrelated domain",
        "severity": "HIGH",
    },
    "is_suspicious_tld": {
        "title": "High-Risk Top-Level Domain",
        "desc": "Uses a TLD ({tld}) frequently associated with phishing and spam",
        "severity": "MEDIUM",
    },
    "form_insecure_http": {
        "title": "Insecure Form Action",
        "desc": "Form action submits credentials over unencrypted HTTP",
        "severity": "HIGH",
    },
    "num_at": {
        "title": "Obfuscated Destination (@ symbol)",
        "desc": "URL contains '@' characters which can disguise the actual destination",
        "severity": "MEDIUM",
    },
    "suspicious_kw": {
        "title": "Phishing Keywords in URL",
        "desc": "Contains security or credential harvesting keywords in the URL",
        "severity": "MEDIUM",
    },
    "subdomain_count": {
        "title": "Excessive Subdomain Levels",
        "desc": "Unusually deep subdomain hierarchy ({count} levels)",
        "severity": "LOW",
    },
    "entropy_path": {
        "title": "High Path Entropy",
        "desc": "URL path contains randomized, high-entropy character sequences",
        "severity": "LOW",
    },
    "hidden_iframes": {
        "title": "Hidden Deceptive iFrames",
        "desc": "Contains zero-dimension or hidden iframe elements",
        "severity": "MEDIUM",
    }
}


def reasons_for(features: Dict[str, Any], url: str) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    """Analyzes features and returns:
    1. human-readable short reasons (list of strings for backward compatibility)
    2. detailed structured reasons (list of dicts with title, desc, severity)
    3. DOM CSS selectors for suspicious elements to highlight
    """
    reasons_text: List[str] = []
    structured_reasons: List[Dict[str, str]] = []
    selectors: List[str] = []

    detected_brand = features.get("detected_brand") or features.get("detected_brand_dom") or "trusted brand"
    tld = features.get("tld", "")

    def add_reason(key: str, **kwargs):
        defn = REASON_DEFINITIONS.get(key)
        if defn:
            desc = defn["desc"].format(**kwargs)
            reasons_text.append(f"{defn['title']}: {desc}")
            structured_reasons.append({
                "key": key,
                "title": defn["title"],
                "description": desc,
                "severity": defn["severity"],
            })

    # Typosquatting
    if float(features.get("typosquat_score", 0.0)) >= 0.7:
        add_reason("typosquatting", brand=detected_brand.capitalize())

    # Brand in Subdomain
    if float(features.get("brand_in_subdomain", 0.0)) > 0.5:
        add_reason("brand_in_subdomain", brand=detected_brand.capitalize())

    # Form Submission across domains
    if float(features.get("form_action_diff_domain", 0.0)) > 0.5:
        add_reason("form_action_diff_domain")
        selectors.append("form")

    # Insecure password input
    if float(features.get("insecure_password_input", 0.0)) > 0.5:
        add_reason("insecure_password_input")
        selectors.append("input[type='password']")

    # Logo brand mismatch
    if float(features.get("logo_mismatch", 0.0)) > 0.5:
        add_reason("logo_mismatch", brand=detected_brand.capitalize())
        selectors.append(f"img[src*='{detected_brand.lower()}']")
        selectors.append("img")

    # Raw IP Hostname
    if float(features.get("has_ip", 0.0)) > 0.5:
        add_reason("has_ip")

    # Punycode / Homograph
    if float(features.get("has_punycode", 0.0)) > 0.5:
        add_reason("has_punycode")

    # Suspicious TLD
    if float(features.get("is_suspicious_tld", 0.0)) > 0.5:
        add_reason("is_suspicious_tld", tld=f".{tld}" if tld else "")

    # Insecure HTTP form
    if float(features.get("form_insecure_http", 0.0)) > 0.5:
        add_reason("form_insecure_http")
        selectors.append("form")

    # At-symbol obfuscation
    if float(features.get("num_at", 0.0)) >= 1.0:
        add_reason("num_at")

    # Suspicious keywords
    if float(features.get("suspicious_kw", 0.0)) > 0.5:
        add_reason("suspicious_kw")

    # Hidden iframes
    if float(features.get("hidden_iframes", 0.0)) > 0.5:
        add_reason("hidden_iframes")
        selectors.append("iframe")

    # Excessive subdomains
    sub_cnt = int(features.get("subdomain_count", 0))
    if sub_cnt >= 3:
        add_reason("subdomain_count", count=sub_cnt)

    # Entropy
    if float(features.get("entropy_path", 0.0)) >= 4.2:
        add_reason("entropy_path")

    # Remove duplicates
    unique_selectors = list(dict.fromkeys(selectors))

    return reasons_text, structured_reasons, unique_selectors
