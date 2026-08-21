"""
PhishLens TTL Result Cache
--------------------------
Wraps cachetools.TTLCache to cache /check results by a hash of
(url + first 500 chars of html).  Avoids redundant ML/feature
extraction when the same page is re-analysed within the TTL window.
"""

import hashlib
from typing import Optional, Dict, Any

from cachetools import TTLCache

from config import CACHE_TTL, CACHE_MAX_SIZE

_cache: TTLCache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)


def _make_key(url: str, html: str) -> str:
    raw = url + html[:500]
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()


def get(url: str, html: str) -> Optional[Dict[str, Any]]:
    """Return cached result or None."""
    return _cache.get(_make_key(url, html))


def put(url: str, html: str, result: Dict[str, Any]) -> None:
    """Store a result in the cache."""
    _cache[_make_key(url, html)] = result


def size() -> int:
    return len(_cache)


def clear() -> None:
    _cache.clear()
