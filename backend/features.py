import re
import math
import tldextract
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Optional, Any

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "update", "account", "confirm",
    "password", "reset", "bank", "invoice", "pay", "wallet",
    "signin", "authenticate", "billing", "recover", "unlock",
    "security", "support", "kyc", "credential", "auth"
]

BRANDS = [
    "paypal", "microsoft", "apple", "google", "amazon", "facebook",
    "netflix", "bankofamerica", "chase", "wellsfargo", "instagram",
    "binance", "coinbase", "steam", "twitter", "linkedin", "dhl",
    "fedex", "adobe", "dropbox", "ebay", "spotify", "citibank"
]

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "work", "click",
    "country", "stream", "loan", "cfd", "buzz", "icu", "fit",
    "gdn", "cam", "sbs", "rest", "bar", "surf"
}

LEET_REPLACEMENTS = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's',
    '7': 't', '8': 'b', '@': 'a', '$': 's'
}


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    c = Counter(s)
    l = len(s)
    return -sum((cnt / l) * math.log2(cnt / l) for cnt in c.values())


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalize_leet(text: str) -> str:
    res = []
    for ch in text.lower():
        res.append(LEET_REPLACEMENTS.get(ch, ch))
    return "".join(res)


def detect_brand_typosquatting(domain_name: str) -> Tuple[float, Optional[str]]:
    """Detects if a domain name is typosquatting a known brand.
    Returns (typosquat_score, detected_brand).
    """
    cleaned_domain = domain_name.lower().split(".")[0]
    norm_domain = normalize_leet(cleaned_domain)

    # Direct match on the brand is handled separately (legit or subdomain spoofing)
    best_score = 0.0
    detected_brand = None

    for brand in BRANDS:
        if cleaned_domain == brand:
            # Exact domain matches brand name directly
            continue

        # Check if brand is embedded in domain with hyphens/prefixes (e.g. paypal-verify)
        if brand in cleaned_domain or brand in norm_domain:
            best_score = max(best_score, 0.85)
            detected_brand = brand
            continue

        # Levenshtein distance check
        dist = levenshtein_distance(cleaned_domain, brand)
        if dist in (1, 2) and len(cleaned_domain) >= 4 and abs(len(cleaned_domain) - len(brand)) <= 2:
            sim = 1.0 - (dist / max(len(cleaned_domain), len(brand)))
            if sim > best_score:
                best_score = sim
                detected_brand = brand

    return best_score, detected_brand


def extract_url_features(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    registered_domain = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
    subdomain = ext.subdomain or ""
    subdomains_list = [p for p in subdomain.split('.') if p]

    path_q = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")

    # IP address check (standard, hex, or octal patterns)
    hostname = parsed.hostname or ""
    is_ip = 1.0 if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", hostname) else 0.0
    has_hex_ip = 1.0 if re.match(r"^0x[0-9a-fA-F]+", hostname) else 0.0

    # Punycode / IDN Homograph check
    has_punycode = 1.0 if "xn--" in hostname.lower() else 0.0

    # Suspicious TLD check
    tld = (ext.suffix or "").lower().split(".")[-1]
    is_suspicious_tld = 1.0 if tld in SUSPICIOUS_TLDS else 0.0

    # Typosquatting / Brand impersonation in domain
    typosquat_score, spoofed_brand = detect_brand_typosquatting(ext.domain or "")

    # Check if brand is in subdomains while registered domain is different
    brand_in_subdomain = 0.0
    for b in BRANDS:
        if b in subdomain.lower() and b != ext.domain.lower():
            brand_in_subdomain = 1.0
            if not spoofed_brand:
                spoofed_brand = b
            break

    # Keyword search in URL
    kw_hits = [k for k in SUSPICIOUS_KEYWORDS if k in url.lower()]
    suspicious_kw = 1.0 if len(kw_hits) > 0 else 0.0

    # Brand search in whole URL
    url_brand_hits = [b for b in BRANDS if b in url.lower()]
    brand_in_url = 1.0 if len(url_brand_hits) > 0 else 0.0
    if not spoofed_brand and url_brand_hits:
        # If brand appears in path/query but not registered domain
        if ext.domain.lower() not in url_brand_hits:
            spoofed_brand = url_brand_hits[0]

    # Obfuscation indicators
    num_dashes = url.count("-")
    num_at = url.count("@")
    num_underscores = url.count("_")
    num_percent = url.count("%")
    has_port = 1.0 if (parsed.port and parsed.port not in (80, 443)) else 0.0

    features: Dict[str, Any] = {
        # Domain Risk
        "has_ip": is_ip or has_hex_ip,
        "has_punycode": has_punycode,
        "is_suspicious_tld": is_suspicious_tld,
        "subdomain_count": float(len(subdomains_list)),
        "domain_len": len(registered_domain or ""),
        
        # URL Risk
        "url_len": len(url),
        "path_len": len(parsed.path or ""),
        "query_len": len(parsed.query or ""),
        "num_dashes": float(num_dashes),
        "num_at": float(num_at),
        "num_slashes": float(url.count("/")),
        "num_percent": float(num_percent),
        "num_underscores": float(num_underscores),
        "has_port": has_port,
        "entropy_path": shannon_entropy(path_q),
        "has_https": 1.0 if parsed.scheme == "https" else 0.0,
        "suspicious_kw": suspicious_kw,
        "kw_hit_count": float(len(kw_hits)),

        # Brand Risk
        "brand_in_url": brand_in_url,
        "brand_in_subdomain": brand_in_subdomain,
        "typosquat_score": typosquat_score,
        "detected_brand": spoofed_brand or "",

        # Metadata
        "domain": domain,
        "registered_domain": registered_domain,
        "tld": tld,
        "hostname": hostname,
    }
    return features


def extract_html_features(html: str, base_url: str, detected_brand: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    selectors: List[str] = []
    base_ext = tldextract.extract(base_url)
    base_reg_domain = base_ext.registered_domain or ""

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    pw_inputs = soup.find_all("input", {"type": "password"})
    text_inputs = [i for i in inputs if i.get("type") in (None, "text", "email", "tel")]

    # Brand text search in title & meta description
    brand_hit = False
    found_brand = detected_brand
    text_blob = (soup.title.string if soup.title and soup.title.string else "")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        text_blob += " " + str(meta_desc.get("content", ""))
    
    # Check og:title / og:site_name
    og_site = soup.find("meta", property="og:site_name") or soup.find("meta", attrs={"name": "og:site_name"})
    if og_site and og_site.get("content"):
        text_blob += " " + str(og_site.get("content", ""))

    text_blob = text_blob.lower()

    for b in BRANDS:
        if b in text_blob:
            # If brand found in text and base domain is not the brand's domain
            if b != base_ext.domain.lower():
                brand_hit = True
                if not found_brand:
                    found_brand = b
                break

    # Form analysis
    different_action_domain = 0.0
    insecure_form = 0.0
    empty_action_form = 0.0

    for f in forms:
        action = (f.get("action") or "").strip()
        f_id = f.get("id")
        f_name = f.get("name")
        f_selector = f"form#{f_id}" if f_id else (f"form[name='{f_name}']" if f_name else "form")

        if action:
            action_abs = urljoin(base_url, action)
            act_host = tldextract.extract(action_abs)
            act_reg_domain = act_host.registered_domain or ""
            
            if base_reg_domain and act_reg_domain and (base_reg_domain != act_reg_domain):
                different_action_domain = 1.0
                selectors.append(f_selector)
            
            if action.startswith("http://"):
                insecure_form = 1.0
                selectors.append(f_selector)
        else:
            # Empty action on a form containing password field is suspicious
            if f.find("input", {"type": "password"}):
                empty_action_form = 1.0
                selectors.append(f_selector)

    # Logo / Image brand mismatch analysis
    logo_mismatch = 0.0
    imgs = soup.find_all(["img", "svg"])
    for img in imgs:
        src = (img.get("src") or "").lower()
        alt = (img.get("alt") or "").lower()
        title = (img.get("title") or "").lower()
        
        target_brands_to_check = [found_brand] if found_brand else BRANDS
        for b in target_brands_to_check:
            if not b:
                continue
            if b in src or b in alt or b in title:
                # Brand present in image but domain doesn't match the brand
                if b != base_ext.domain.lower():
                    logo_mismatch = 1.0
                    if img.get("id"):
                        selectors.append(f"{img.name}#{img.get('id')}")
                    elif img.get("class"):
                        cls_name = ".".join(img.get("class")[:2])
                        selectors.append(f"{img.name}.{cls_name}")
                    elif img.get("src"):
                        selectors.append(f"img[src*='{b}']")
                    else:
                        selectors.append("img")
                break

    # Additional deceptive elements
    onsubmit_handlers = 1.0 if any(f.get("onsubmit") for f in forms) else 0.0
    
    hidden_iframes = 0.0
    for iframe in soup.find_all("iframe"):
        w = str(iframe.get("width", "")).strip()
        h = str(iframe.get("height", "")).strip()
        style = (iframe.get("style") or "").lower()
        if w in ("0", "1", "0px", "1px") or h in ("0", "1", "0px", "1px") or "display:none" in style or "visibility:hidden" in style:
            hidden_iframes = 1.0
            if iframe.get("id"):
                selectors.append(f"iframe#{iframe.get('id')}")
            else:
                selectors.append("iframe")

    # Password input without HTTPS
    parsed_base = urlparse(base_url)
    insecure_password_input = 1.0 if (pw_inputs and parsed_base.scheme != "https") else 0.0
    if pw_inputs:
        selectors.append("input[type='password']")

    features: Dict[str, Any] = {
        # DOM & Credential Risk
        "num_forms": float(len(forms)),
        "num_inputs": float(len(inputs)),
        "num_pw_inputs": float(len(pw_inputs)),
        "num_text_inputs": float(len(text_inputs)),
        "form_action_diff_domain": different_action_domain,
        "form_insecure_http": insecure_form,
        "empty_action_form": empty_action_form,
        "insecure_password_input": insecure_password_input,
        "brand_text_hit": 1.0 if brand_hit else 0.0,
        "logo_mismatch": logo_mismatch,
        "onsubmit_handlers": onsubmit_handlers,
        "hidden_iframes": hidden_iframes,
        "detected_brand_dom": found_brand or "",
    }

    return features, list(dict.fromkeys(selectors))
