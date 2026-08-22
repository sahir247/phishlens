"""
PhishLens — 14-Feature PhishNot Baseline
=========================================
Deterministic URL feature extractor for the 14 features used
in the PhishNot-style baseline (Vrbančič 2020 feature subset).

Feature taxonomy
----------------
URL-structural (9)   — computed from the URL string alone, zero latency
  1. qty_dot_domain         — dots in hostname
  2. qty_vowels_domain      — vowel count in hostname
  3. domain_length          — hostname length (chars)
  4. qty_dot_directory      — dots in path directory
  5. qty_slash_directory    — slashes in path directory
  6. directory_length       — path directory length (chars)
  7. qty_dot_file           — dots in filename
  8. file_length            — filename length (chars)
  9. params_length          — query-string length (chars)

External / network (5) — require server-side lookup; return -1 when unavailable
  10. time_response         — TCP response time in ms   (backend only)
  11. asn_ip                — ASN number of resolved IP (backend only)
  12. time_domain_activation— WHOIS registration date as Unix timestamp
  13. time_domain_expiration — WHOIS expiry date as Unix timestamp
  14. ttl_hostname           — DNS TTL in seconds        (backend only)

Browser-extension compatibility
---------------------------------
AVAILABLE_DIRECTLY_IN_EXTENSION : features 1-9  (URL parsing, no network)
REQUIRES_BACKEND                : features 10-14 (DNS/WHOIS/TCP lookups)
EXPENSIVE_OR_OPTIONAL           : features 10, 11, 14 (skip in fast path)

Missing-value convention
-------------------------
All numeric features use -1 to signal "not available / lookup failed".
The classifier is trained with -1 present in the data so it learns to
handle missings naturally (HistGradientBoostingClassifier supports NaN;
for other models we impute -1 → median at train time).
"""

from __future__ import annotations

import re
import time
import socket
import struct
import logging
from urllib.parse import urlparse, unquote
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Published 14 feature names (in paper order) ───────────────────────────────
FEATURE_14 = [
    "qty_dot_domain",
    "qty_vowels_domain",
    "domain_length",
    "qty_dot_directory",
    "qty_slash_directory",
    "directory_length",
    "qty_dot_file",
    "file_length",
    "params_length",
    "time_response",
    "asn_ip",
    "time_domain_activation",
    "time_domain_expiration",
    "ttl_hostname",
]

N_FEATURES_14 = len(FEATURE_14)  # 14

# Feature taxonomy for browser-extension consumers
AVAILABLE_IN_EXTENSION = FEATURE_14[:9]   # URL-structural only
REQUIRES_BACKEND       = FEATURE_14[9:]   # network/WHOIS features

_VOWELS = set("aeiouAEIOU")
_MISSING = -1.0   # sentinel for unavailable values


# ═══════════════════════════════════════════════════════════════════════════════
# URL Structural Features  (features 1-9)
# These are computed purely from the URL string — no network calls.
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_url(url: str):
    """
    Robustly parse a URL, adding a scheme if missing.
    Returns a urllib.parse.ParseResult.
    """
    url = url.strip()
    if not url:
        raise ValueError("Empty URL")
    if "://" not in url:
        url = "http://" + url
    return urlparse(url)


def _split_path_parts(path: str):
    """
    Split URL path into (directory, filename).

    Rules:
      - path = /a/b/c/file.php  →  directory="/a/b/c", filename="file.php"
      - path = /a/b/c/          →  directory="/a/b/c", filename=""
      - path = /                →  directory="/",       filename=""
      - path = ""               →  directory="",        filename=""
      - path = /file.php        →  directory="/",       filename="file.php"

    Ignores query strings and fragments (caller should strip them first).
    """
    path = path.split("?")[0].split("#")[0]   # safety strip
    if not path or path == "/":
        return path, ""
    parts = path.rstrip("/").split("/")
    # Last segment is filename if it exists and path doesn't end with /
    if path.endswith("/"):
        directory = path.rstrip("/") or "/"
        filename  = ""
    else:
        filename  = parts[-1]
        directory = "/".join(parts[:-1]) or "/"
    return directory, filename


def extract_url_features_14(url: str) -> Dict[str, float]:
    """
    Extract the 9 URL-structural features from a raw URL string.
    Returns a dict with keys = FEATURE_14[:9], values = float.
    Network features (indices 9-13) are set to _MISSING here;
    use enrich_with_network_features() to fill them in.
    """
    out: Dict[str, float] = {f: _MISSING for f in FEATURE_14}

    try:
        parsed   = _parse_url(url)
        hostname = parsed.hostname or ""
        path     = unquote(parsed.path or "")
        query    = parsed.query or ""

        # Strip port from hostname if present (urlparse already handles this)
        directory, filename = _split_path_parts(path)

        # 1. qty_dot_domain
        out["qty_dot_domain"]     = float(hostname.count("."))

        # 2. qty_vowels_domain
        out["qty_vowels_domain"]  = float(sum(1 for c in hostname if c in _VOWELS))

        # 3. domain_length
        out["domain_length"]      = float(len(hostname))

        # 4. qty_dot_directory
        out["qty_dot_directory"]  = float(directory.count("."))

        # 5. qty_slash_directory
        # Count slashes that represent path depth (not the leading /)
        out["qty_slash_directory"] = float(max(0, directory.count("/") - 1))

        # 6. directory_length  (exclude leading slash for consistency with Vrbancic)
        out["directory_length"]   = float(len(directory.lstrip("/")))

        # 7. qty_dot_file
        out["qty_dot_file"]       = float(filename.count("."))

        # 8. file_length
        out["file_length"]        = float(len(filename))

        # 9. params_length
        out["params_length"]      = float(len(query))

    except Exception as exc:
        logger.debug("Feature extraction failed for %s: %s", url, exc)
        # Leave all as _MISSING

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Network / External Features  (features 10-14)
# These require backend lookups.  Each has a timeout and returns _MISSING
# on any failure so the classifier degrades gracefully.
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_ip(hostname: str, timeout: float = 3.0) -> Optional[str]:
    """Resolve hostname to first IPv4/IPv6 address."""
    old = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout)
        info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC,
                                  socket.SOCK_STREAM)
        if info:
            return info[0][4][0]
    except Exception:
        pass
    finally:
        socket.setdefaulttimeout(old)
    return None


def _get_time_response(hostname: str, port: int = 80, timeout: float = 3.0) -> float:
    """Measure TCP connection latency to hostname:port in milliseconds."""
    try:
        s = socket.create_connection((hostname, port), timeout=timeout)
        t0 = time.monotonic()
        s.close()
        return round((time.monotonic() - t0) * 1000, 1)
    except Exception:
        return _MISSING


def _get_asn_ip(ip: str) -> float:
    """
    Look up the ASN for an IP via the Cymru DNS service (no API key needed).
    Returns ASN as float, or _MISSING on failure.
    """
    if not ip:
        return _MISSING
    try:
        # Reverse the IP octets for IPv4 lookup
        parts = ip.split(".")
        if len(parts) == 4:
            rev = ".".join(reversed(parts))
            result = socket.getaddrinfo(
                f"{rev}.origin.asn.cymru.com", None,
                socket.AF_UNSPEC, socket.SOCK_STREAM,
            )
            # Returns TXT-style; parse ASN from CNAME-like result
            # Fallback: use whois approach
    except Exception:
        pass
    # Simpler fallback: use ipinfo.io (free, no auth for basic)
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://ipinfo.io/{ip}/org",
            headers={"User-Agent": "PhishLens/1.0"},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            data = r.read().decode().strip()
        # Format: "AS12345 ISP Name" — extract the number
        m = re.match(r"AS(\d+)", data)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return _MISSING


def _get_whois_times(hostname: str) -> tuple[float, float]:
    """
    Return (activation_ts, expiration_ts) as Unix timestamps, or (_MISSING, _MISSING).
    Uses rdap.org (no auth, JSON response) — same source as trust.py.
    """
    try:
        import urllib.request, json as _json
        # Strip leading www.
        domain = re.sub(r"^www\.", "", hostname.lower())
        req = urllib.request.Request(
            f"https://rdap.org/domain/{domain}",
            headers={"User-Agent": "PhishLens/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = _json.loads(r.read())

        activation = _MISSING
        expiration = _MISSING

        for ev in data.get("events", []):
            action = ev.get("eventAction", "").lower()
            date_s = ev.get("eventDate", "")
            if not date_s:
                continue
            try:
                from datetime import datetime, timezone
                # ISO 8601 with optional Z/offset
                date_s = date_s.rstrip("Z")
                dt = datetime.fromisoformat(date_s).replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
                if action == "registration":
                    activation = ts
                elif action == "expiration":
                    expiration = ts
            except Exception:
                pass

        return activation, expiration
    except Exception:
        return _MISSING, _MISSING


def _get_ttl(hostname: str) -> float:
    """
    Get DNS TTL for hostname using socket / system resolver.
    Falls back to dnspython if available.
    Returns TTL in seconds as float, or _MISSING.
    """
    try:
        import dns.resolver   # optional; install dnspython if available
        answers = dns.resolver.resolve(hostname, "A", lifetime=3)
        return float(answers.rrset.ttl)
    except ImportError:
        pass
    except Exception:
        pass
    return _MISSING


def enrich_with_network_features(
    features: Dict[str, float],
    url: str,
    timeout: float = 3.0,
    skip_expensive: bool = False,
) -> Dict[str, float]:
    """
    Fill in the 5 external features in-place and return the dict.

    Parameters
    ----------
    features      : dict from extract_url_features_14()
    url           : original URL string
    timeout       : per-lookup timeout in seconds
    skip_expensive: if True, skip time_response and asn_ip (slow lookups)
    """
    out = features.copy()
    try:
        parsed   = _parse_url(url)
        hostname = parsed.hostname or ""
        port     = parsed.port or (443 if parsed.scheme == "https" else 80)

        if not hostname:
            return out

        # 10. time_response
        if not skip_expensive:
            out["time_response"] = _get_time_response(hostname, port, timeout)

        # 11. asn_ip + resolve for later
        ip = _resolve_ip(hostname, timeout)
        if not skip_expensive and ip:
            out["asn_ip"] = _get_asn_ip(ip)

        # 12 + 13. WHOIS times (via rdap.org — same as trust.py)
        activation, expiration = _get_whois_times(hostname)
        out["time_domain_activation"] = activation
        out["time_domain_expiration"] = expiration

        # 14. DNS TTL
        out["ttl_hostname"] = _get_ttl(hostname)

    except Exception as exc:
        logger.debug("Network enrichment failed for %s: %s", url, exc)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: extract all 14 features (URL-only, no network calls)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_14_fast(url: str) -> Dict[str, float]:
    """
    Extract only the 9 URL-structural features.
    Network features remain -1.
    Suitable for browser-extension / low-latency path.
    """
    return extract_url_features_14(url)


def features_to_vector(feat: Dict[str, float]) -> list[float]:
    """Return a list in canonical FEATURE_14 order for model input."""
    return [float(feat.get(f, _MISSING)) for f in FEATURE_14]
