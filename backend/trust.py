"""
PhishLens Dynamic Trust Engine
--------------------------------
Computes a continuous trust_score (0–1) for a domain from three runtime signals:

  1. Tranco rank  — top-1M list from tranco-list.eu, loaded once at startup
                    and cached to disk. Rank ≤ 500 → very trusted.
  2. RDAP age     — domain registration date via rdap.org (free, no API key).
                    Domains > 2 years old are established. < 30 days = red flag.
  3. HTTPS        — baseline hygiene (no weight on its own, breaks ties).

The trust_score is used in model.py to dampen the final risk score.
High trust → cannot be DANGEROUS unless ML + heuristics are both very confident.

Architecture
------------
  - _TrancoIndex   : singleton, lazy-loaded, disk-cached JSON (top 10k only)
  - _RDAPCache     : disk-backed per-domain age cache (7-day TTL)
  - compute_trust  : public API used by features.py
"""

import os
import io
import json
import time
import zipfile
import logging
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Tuple

from config import DATA_DIR

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TRANCO_URL         = "https://tranco-list.eu/top-1m.csv.zip"
TRANCO_CACHE_FILE  = os.path.join(DATA_DIR, "tranco_top10k.json")
TRANCO_CACHE_TTL   = 7 * 86400          # refresh weekly
TRANCO_TOP_N       = 10_000             # how many to keep in memory

RDAP_CACHE_FILE    = os.path.join(DATA_DIR, "rdap_age_cache.json")
RDAP_CACHE_TTL     = 7 * 86400          # re-check weekly
RDAP_TIMEOUT       = 3                  # seconds per HTTP call
RDAP_BASE          = "https://rdap.org/domain/"


# ── Tranco Index ───────────────────────────────────────────────────────────────
class _TrancoIndex:
    """Singleton: maps registered_domain → Tranco rank (1 = most popular)."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._data: Dict[str, int] = {}   # domain → rank
                obj._loaded_at: float = 0.0
                obj._loading = False
                cls._instance = obj
        return cls._instance

    # ── public ────────────────────────────────────────────────────────────────
    def rank(self, domain: str) -> Optional[int]:
        """Return rank or None if not in top-10k."""
        self._ensure_loaded()
        return self._data.get(domain.lower().lstrip("www."))

    def loaded(self) -> bool:
        return bool(self._data)

    # ── internal ──────────────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._data and (time.time() - self._loaded_at) < TRANCO_CACHE_TTL:
            return
        if self._loading:
            return          # another thread is loading; caller gets stale data
        self._loading = True
        t = threading.Thread(target=self._load, daemon=True)
        t.start()

    def _load(self):
        try:
            # 1. Try disk cache first
            if os.path.exists(TRANCO_CACHE_FILE):
                stat = os.stat(TRANCO_CACHE_FILE)
                if (time.time() - stat.st_mtime) < TRANCO_CACHE_TTL:
                    with open(TRANCO_CACHE_FILE, "r", encoding="utf-8") as f:
                        self._data = json.load(f)
                    self._loaded_at = time.time()
                    logger.info(f"Tranco: loaded {len(self._data):,} entries from disk cache.")
                    return

            # 2. Download fresh copy
            logger.info("Tranco: downloading top-1M list (background thread)…")
            req = urllib.request.Request(TRANCO_URL, headers={"User-Agent": "PhishLens/0.3"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()

            # 3. Unzip and parse CSV
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as csvf:
                    lines = csvf.read().decode("utf-8", errors="ignore").splitlines()

            data: Dict[str, int] = {}
            for line in lines[:TRANCO_TOP_N]:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        rank = int(parts[0])
                        domain = parts[1].lower().lstrip("www.")
                        data[domain] = rank
                    except ValueError:
                        pass

            self._data = data
            self._loaded_at = time.time()

            # 4. Persist to disk
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(TRANCO_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)

            logger.info(f"Tranco: downloaded and cached {len(data):,} entries.")

        except Exception as e:
            logger.warning(f"Tranco load failed: {e}  (trust engine will still work via RDAP)")
        finally:
            self._loading = False


_tranco = _TrancoIndex()


# ── RDAP Domain Age Cache ─────────────────────────────────────────────────────
class _RDAPCache:
    """Disk-backed cache: registered_domain → age_days (float or None)."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._mem: Dict[str, dict] = {}
                obj._dirty = False
                obj._last_flush = time.time()
                cls._instance = obj
                obj._load_disk()
        return cls._instance

    def get(self, domain: str) -> Optional[float]:
        """Return cached age_days or None if not cached / stale."""
        entry = self._mem.get(domain)
        if not entry:
            return None
        if (time.time() - entry["ts"]) > RDAP_CACHE_TTL:
            return None
        return entry.get("age_days")   # may be None (lookup failed)

    def set(self, domain: str, age_days: Optional[float]):
        self._mem[domain] = {"age_days": age_days, "ts": time.time()}
        self._dirty = True
        # Flush to disk every 60 s max
        if time.time() - self._last_flush > 60:
            self._flush()

    def _load_disk(self):
        try:
            if os.path.exists(RDAP_CACHE_FILE):
                with open(RDAP_CACHE_FILE, "r", encoding="utf-8") as f:
                    self._mem = json.load(f)
        except Exception:
            self._mem = {}

    def _flush(self):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(RDAP_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._mem, f)
            self._dirty = False
            self._last_flush = time.time()
        except Exception as e:
            logger.debug(f"RDAP cache flush error: {e}")


_rdap_cache = _RDAPCache()


# ── RDAP Lookup ───────────────────────────────────────────────────────────────
def _fetch_domain_age_days(domain: str) -> Optional[float]:
    """
    Query rdap.org for the domain's registration date.
    Returns age in days or None on failure.
    Non-blocking: called from a background thread.
    """
    # Strip any leading www.
    domain = domain.lower().lstrip("www.")
    url = RDAP_BASE + domain

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                    "User-Agent": "PhishLens/0.3"})
        with urllib.request.urlopen(req, timeout=RDAP_TIMEOUT) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))

        # Parse registration event date
        events = data.get("events", [])
        for ev in events:
            action = (ev.get("eventAction") or "").lower()
            if action in ("registration", "registered"):
                date_str = ev.get("eventDate", "")
                if date_str:
                    # Normalise timezone: rdap uses ISO 8601
                    date_str = date_str.replace("Z", "+00:00")
                    reg_dt = datetime.fromisoformat(date_str)
                    if reg_dt.tzinfo is None:
                        reg_dt = reg_dt.replace(tzinfo=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - reg_dt).days
                    return float(max(age_days, 0))

    except Exception:
        pass

    return None


def get_domain_age_days(domain: str) -> Optional[float]:
    """
    Returns domain age in days (cached). Launches background lookup on miss.
    Returns None when age is unknown (first call for this domain).
    """
    domain = domain.lower().lstrip("www.")

    # Cache hit
    cached = _rdap_cache.get(domain)
    if cached is not None or domain in _rdap_cache._mem:
        return cached

    # Cache miss — fetch in background so we don't block the request
    def _bg():
        age = _fetch_domain_age_days(domain)
        _rdap_cache.set(domain, age)

    t = threading.Thread(target=_bg, daemon=True)
    t.start()
    return None  # first call returns None; subsequent calls hit cache


# ── Public API ────────────────────────────────────────────────────────────────
def compute_trust(registered_domain: str, has_https: bool) -> Tuple[float, Dict]:
    """
    Returns (trust_score 0–1, signals dict).

    trust_score interpretation:
      0.0  — completely unknown / newly registered
      0.3  — appears in Tranco top-10k OR domain is 1–2 years old
      0.6  — Tranco top-2k AND/OR domain > 2 years old
      0.85 — Tranco top-500 AND domain > 5 years old
      1.0  — maximum (Tranco top-100 + very old domain)

    model.py uses this to dampen the final risk score.
    """
    domain = registered_domain.lower().lstrip("www.")
    signals: Dict = {"domain": domain}

    # ── Tranco rank signal ────────────────────────────────────────────────────
    rank = _tranco.rank(domain)
    signals["tranco_rank"] = rank

    if rank is None:
        tranco_contribution = 0.0
    elif rank <= 100:
        tranco_contribution = 0.65
    elif rank <= 500:
        tranco_contribution = 0.55
    elif rank <= 2_000:
        tranco_contribution = 0.42
    elif rank <= 5_000:
        tranco_contribution = 0.30
    else:                          # 5k – 10k
        tranco_contribution = 0.20

    # ── RDAP domain age signal ────────────────────────────────────────────────
    age_days = get_domain_age_days(domain)
    signals["domain_age_days"] = age_days

    if age_days is None:
        age_contribution = 0.0       # unknown — no benefit
    elif age_days < 30:
        age_contribution = -0.25     # new domain → negative trust
    elif age_days < 180:
        age_contribution = 0.05
    elif age_days < 365:
        age_contribution = 0.12
    elif age_days < 730:
        age_contribution = 0.22
    elif age_days < 1825:           # < 5 years
        age_contribution = 0.32
    else:                           # 5+ years old
        age_contribution = 0.42

    # ── HTTPS baseline ────────────────────────────────────────────────────────
    https_contribution = 0.05 if has_https else 0.0
    signals["has_https"] = has_https

    # ── Combine ───────────────────────────────────────────────────────────────
    raw = tranco_contribution + age_contribution + https_contribution
    trust_score = round(max(0.0, min(1.0, raw)), 3)
    signals["trust_score"] = trust_score

    return trust_score, signals


def warm_up():
    """Call at startup to trigger Tranco download in the background."""
    _tranco._ensure_loaded()
