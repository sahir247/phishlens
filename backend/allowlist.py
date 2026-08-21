"""
PhishLens Domain Allowlist
--------------------------
Maps known brand names to their *official* registered domains.
When a URL's registered domain is in this set for the detected brand,
the page is considered legitimate and risk is suppressed — preventing
false positives on the real PayPal, Apple, Microsoft, etc. sites.
"""

from typing import Dict, Set

# brand_name (lowercase) → set of legitimate registered domains
BRAND_ALLOWLIST: Dict[str, Set[str]] = {
    "paypal": {
        "paypal.com", "paypal.co.uk", "paypal.de", "paypal.fr",
        "paypal.com.au", "paypal.ca", "paypal.es", "paypal.it",
        "paypal.me",
    },
    "microsoft": {
        "microsoft.com", "microsoftonline.com", "live.com",
        "outlook.com", "hotmail.com", "office.com", "azure.com",
        "xbox.com", "bing.com", "msn.com", "windowsazure.com",
        "microsoftstore.com",
    },
    "apple": {
        "apple.com", "icloud.com", "itunes.com", "me.com",
        "apple.co",
    },
    "google": {
        "google.com", "google.co.uk", "google.de", "google.fr",
        "google.ca", "google.com.au", "gmail.com", "youtube.com",
        "googlemail.com", "googleapis.com", "goo.gl", "google.co.in",
        "accounts.google.com",
    },
    "amazon": {
        "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr",
        "amazon.co.jp", "amazon.ca", "amazon.com.au", "amazon.in",
        "amazon.es", "amazon.it", "amzn.to", "aws.amazon.com",
    },
    "facebook": {
        "facebook.com", "fb.com", "fb.me", "instagram.com",
        "whatsapp.com", "messenger.com", "meta.com",
    },
    "instagram": {
        "instagram.com", "facebook.com", "meta.com",
    },
    "twitter": {
        "twitter.com", "t.co", "x.com",
    },
    "netflix": {
        "netflix.com", "netflix.net",
    },
    "bankofamerica": {
        "bankofamerica.com", "bofa.com",
    },
    "chase": {
        "chase.com", "jpmorganchase.com",
    },
    "wellsfargo": {
        "wellsfargo.com",
    },
    "linkedin": {
        "linkedin.com",
    },
    "binance": {
        "binance.com", "binance.us",
    },
    "coinbase": {
        "coinbase.com",
    },
    "steam": {
        "steampowered.com", "steamcommunity.com",
    },
    "adobe": {
        "adobe.com", "adobeid.com",
    },
    "dropbox": {
        "dropbox.com", "dropboxapi.com",
    },
    "ebay": {
        "ebay.com", "ebay.co.uk", "ebay.de", "ebay.fr",
        "ebay.com.au", "ebay.ca",
    },
    "spotify": {
        "spotify.com",
    },
    "citibank": {
        "citi.com", "citibank.com", "citibankonline.com",
    },
    "dhl": {
        "dhl.com", "dhl.de",
    },
    "fedex": {
        "fedex.com",
    },
}

# Build a fast reverse lookup: registered_domain → brand
_DOMAIN_TO_BRAND: Dict[str, str] = {}
for _brand, _domains in BRAND_ALLOWLIST.items():
    for _d in _domains:
        _DOMAIN_TO_BRAND[_d.lower()] = _brand


def is_allowlisted(registered_domain: str, detected_brand: str = "") -> bool:
    """
    Returns True if the registered_domain is an official domain for
    either the detected_brand or *any* brand in the allowlist.
    This avoids flagging paypal.com as suspicious just because 'paypal'
    appears as a keyword in the URL.
    """
    rd = (registered_domain or "").lower()
    if not rd:
        return False

    # Check if domain is officially allowlisted at all
    if rd in _DOMAIN_TO_BRAND:
        return True

    # Also check subdomains of known brands (e.g. accounts.google.com → google.com)
    parts = rd.split(".")
    if len(parts) >= 2:
        apex = ".".join(parts[-2:])
        if apex in _DOMAIN_TO_BRAND:
            return True

    return False


def get_brand_for_domain(registered_domain: str) -> str:
    """Return the brand name for a known domain, or empty string."""
    rd = (registered_domain or "").lower()
    if rd in _DOMAIN_TO_BRAND:
        return _DOMAIN_TO_BRAND[rd]
    parts = rd.split(".")
    if len(parts) >= 2:
        apex = ".".join(parts[-2:])
        if apex in _DOMAIN_TO_BRAND:
            return _DOMAIN_TO_BRAND[apex]
    return ""
