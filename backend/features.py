import re
import math
import tldextract
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "update", "account", "confirm",
    "password", "reset", "bank", "invoice", "pay", "wallet",
]
BRANDS = [
    "paypal", "microsoft", "apple", "google", "amazon", "facebook",
    "netflix", "bankofamerica", "chase", "wellsfargo", "instagram",
]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    c = Counter(s)
    l = len(s)
    return -sum((cnt/l) * math.log2(cnt/l) for cnt in c.values())


def extract_url_features(url: str) -> Dict[str, float]:
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
    subdomain = ext.subdomain or ""

    path_q = (parsed.path or "") + ("?" + parsed.query if parsed.query else "")
    features: Dict[str, float] = {
        "url_len": len(url),
        "path_len": len(parsed.path or ""),
        "query_len": len(parsed.query or ""),
        "num_dashes": url.count("-"),
        "num_at": url.count("@"),
        "num_slashes": url.count("/"),
        "has_ip": 1.0 if re.match(r"^\d+\.\d+\.\d+\.\d+$", parsed.hostname or "") else 0.0,
        "subdomain_count": float(len([p for p in subdomain.split('.') if p])),
        "entropy_path": shannon_entropy(path_q),
        "has_https": 1.0 if parsed.scheme == "https" else 0.0,
        "suspicious_kw": 1.0 if any(k in url.lower() for k in SUSPICIOUS_KEYWORDS) else 0.0,
        "brand_kw": 1.0 if any(b in url.lower() for b in BRANDS) else 0.0,
        "domain": domain,
        "registered_domain": f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain,
    }
    return features


def extract_html_features(html: str, base_url: str) -> Tuple[Dict[str, float], List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    selectors: List[str] = []

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    pw_inputs = soup.find_all("input", {"type": "password"})

    brand_hit = False
    text_blob = (soup.title.string if soup.title and soup.title.string else "")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        text_blob += " " + meta_desc.get("content", "")
    text_blob = (text_blob or "").lower()

    for b in BRANDS:
        if b in text_blob:
            brand_hit = True
            break

    different_action_domain = 0.0
    insecure_form = 0.0
    for f in forms:
        action = f.get("action", "").strip()
        if action:
            action_abs = urljoin(base_url, action)
            base_host = tldextract.extract(base_url)
            act_host = tldextract.extract(action_abs)
            if (base_host.registered_domain or "") != (act_host.registered_domain or ""):
                different_action_domain = 1.0
                selectors.append(f.name + ("#" + f.get("id") if f.get("id") else ""))
        if action.startswith("http://"):
            insecure_form = 1.0

    logo_mismatch = 0.0
    imgs = soup.find_all("img")
    for img in imgs:
        src = (img.get("src") or "").lower()
        alt = (img.get("alt") or "").lower()
        for b in BRANDS:
            if b in src or b in alt:
                # If brand present in image but domain doesn't match the brand keyword in base_url
                if b not in base_url.lower():
                    logo_mismatch = 1.0
                    # try construct a selector-ish string
                    if img.get("id"):
                        selectors.append(f"img#{img.get('id')}")
                    elif img.get("class"):
                        selectors.append("img." + ".".join(img.get("class")))
                    else:
                        selectors.append("img[src*='" + b + "']")
                break

    onsubmit_handlers = 1.0 if any(f.get("onsubmit") for f in forms) else 0.0
    hidden_iframes = 1.0 if any((i.get("width") == "0" or i.get("height") == "0") for i in soup.find_all("iframe")) else 0.0

    features: Dict[str, float] = {
        "num_forms": float(len(forms)),
        "num_inputs": float(len(inputs)),
        "num_pw_inputs": float(len(pw_inputs)),
        "form_action_diff_domain": different_action_domain,
        "form_insecure_http": insecure_form,
        "brand_text_hit": 1.0 if brand_hit else 0.0,
        "logo_mismatch": logo_mismatch,
        "onsubmit_handlers": onsubmit_handlers,
        "hidden_iframes": hidden_iframes,
    }

    return features, selectors
