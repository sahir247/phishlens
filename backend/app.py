"""
PhishLens API Server — v0.3.0
--------------------------------
Production-grade Flask backend featuring:
  - Rate limiting (Flask-Limiter)
  - Structured rotating log file
  - TTL result cache (skip redundant analysis)
  - Domain allowlist (suppress brand false-positives)
  - Batch scan endpoint (POST /check/batch, max 20 URLs)
  - 24-hour trend endpoint (GET /stats/trend)
  - Full analysis pipeline: URL features → HTML features → ML ensemble → explain
"""

import os
# Suppress joblib "could not find physical cores" warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
import json
import logging
import logging.handlers
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import config
from features import extract_url_features, extract_html_features
from explain import reasons_for
from model import predict_risk
from storage import DB, DetectionEvent
from allowlist import is_allowlisted
import cache as url_cache

# ── Logging Setup ─────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    log = logging.getLogger("phishlens")
    log.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    # Rotating file
    try:
        fh = logging.handlers.RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except Exception as e:
        log.warning(f"Could not open log file {config.LOG_FILE}: {e}")
    return log


logger = _setup_logging()

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

DASHBOARD_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dashboard")
)

logger.info("PhishLens v0.3.0 starting — loading ML model…")
try:
    from ml_model import _load_model
    _load_model()
except Exception as e:
    logger.warning(f"ML model warm-up failed (will retry on first request): {e}")

# Start Tranco download in background (non-blocking)
try:
    from trust import warm_up as _trust_warm_up
    _trust_warm_up()
    logger.info("Trust engine: Tranco download started in background.")
except Exception as e:
    logger.warning(f"Trust engine warm-up failed: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _analyse(url_str: str, html_str: str) -> dict:
    """Core analysis pipeline. Returns the full result dict."""
    url_features = extract_url_features(url_str)
    detected_brand = url_features.get("detected_brand")

    html_features, html_selectors = extract_html_features(
        html_str, url_str, detected_brand=detected_brand
    )
    all_features = {**url_features, **html_features}

    risk_score, risk_level, category_scores = predict_risk(all_features)
    reasons_text, structured_reasons, explain_selectors = reasons_for(all_features, url_str)
    highlights = list(dict.fromkeys(html_selectors + explain_selectors))

    brand_target = (
        url_features.get("detected_brand")
        or html_features.get("detected_brand_dom")
        or ""
    )

    meta = {
        "domain":            url_features.get("domain", ""),
        "registered_domain": url_features.get("registered_domain", ""),
        "hostname":          url_features.get("hostname", ""),
        "has_https":         bool(url_features.get("has_https")),
        "brand_target":      brand_target,
        "ts":                datetime.utcnow().timestamp(),
    }

    return {
        "risk_score":         float(risk_score),
        "risk_level":         risk_level,
        "brand_target":       brand_target,
        "category_scores":    category_scores,
        "reasons":            reasons_text,
        "structured_reasons": structured_reasons,
        "highlights":         highlights,
        "trust_signals":      url_features.get("trust_signals", {}),
        "meta":               meta,
    }


def _allowlist_safe_result(url_str: str, registered_domain: str, brand: str) -> dict:
    """Return a SAFE allowlist-override result without running ML."""
    meta = {
        "domain":            registered_domain,
        "registered_domain": registered_domain,
        "hostname":          registered_domain,
        "has_https":         url_str.startswith("https"),
        "brand_target":      "",
        "ts":                datetime.utcnow().timestamp(),
    }
    return {
        "risk_score":         0.02,
        "risk_level":         "SAFE",
        "brand_target":       "",
        "category_scores":    {
            "domain_risk": 0.0, "url_risk": 0.0,
            "brand_risk": 0.0, "dom_risk": 0.0, "ml_risk": 0.0,
        },
        "reasons":            [],
        "structured_reasons": [],
        "highlights":         [],
        "meta":               meta,
        "allowlisted":        True,
        "allowlisted_brand":  brand,
    }


def _persist_event(result: dict, url_str: str):
    try:
        with DB.session() as s:
            evt = DetectionEvent(
                url=url_str,
                risk_score=result["risk_score"],
                risk_level=result["risk_level"],
                brand_target=result["brand_target"],
                reasons_json=json.dumps(result["reasons"]),
                structured_reasons_json=json.dumps(result["structured_reasons"]),
                category_scores_json=json.dumps(result["category_scores"]),
                meta_json=json.dumps(result["meta"]),
                ts=datetime.utcnow().timestamp(),
            )
            s.add(evt)
            s.commit()
    except Exception as e:
        logger.error(f"DB persist error: {e}")


# ── Static Serving ────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.get("/dashboard")
@app.get("/dashboard/")
@app.get("/dashboard/<path:filename>")
def serve_dashboard(filename="index.html"):
    return send_from_directory(DASHBOARD_DIR, filename)


@app.get("/styles.css")
def serve_styles():
    return send_from_directory(DASHBOARD_DIR, "styles.css")


@app.get("/main.js")
def serve_main_js():
    return send_from_directory(DASHBOARD_DIR, "main.js")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({
        "status":  "ok",
        "service": "PhishLens API",
        "version": "0.3.0",
        "cache_size": url_cache.size(),
        "ts":      datetime.utcnow().timestamp(),
    })


# ── Primary Check ─────────────────────────────────────────────────────────────
@app.post("/check")
@limiter.limit(config.RATE_LIMIT_CHECK)
def check():
    data = request.get_json(force=True) or {}
    url = data.get("url", "")
    html = data.get("html", "")
    if not url:
        return jsonify({"error": "Missing 'url'"}), 400

    url_str  = str(url).strip()
    html_str = str(html or "")

    # ── Cache hit ──────────────────────────────────────────────────────────
    cached = url_cache.get(url_str, html_str)
    if cached:
        logger.debug(f"CACHE HIT  {url_str[:80]}")
        return jsonify({**cached, "cached": True})

    # ── Allowlist check ────────────────────────────────────────────────────
    from features import extract_url_features as _eu
    _uf = _eu(url_str)
    reg_domain = _uf.get("registered_domain", "")
    detected_b = _uf.get("detected_brand", "")

    if is_allowlisted(reg_domain, detected_b):
        from allowlist import get_brand_for_domain
        canonical_brand = get_brand_for_domain(reg_domain)
        result = _allowlist_safe_result(url_str, reg_domain, canonical_brand)
        logger.info(f"ALLOWLIST  score=0.02  {url_str[:80]}")
        url_cache.put(url_str, html_str, result)
        return jsonify(result)

    # ── Full analysis ──────────────────────────────────────────────────────
    result = _analyse(url_str, html_str)
    _persist_event(result, url_str)
    url_cache.put(url_str, html_str, result)

    logger.info(
        f"CHECK  score={result['risk_score']:.3f}  "
        f"level={result['risk_level']:<10}  {url_str[:80]}"
    )
    return jsonify(result)


# ── Batch Check ───────────────────────────────────────────────────────────────
@app.post("/check/batch")
@limiter.limit(config.RATE_LIMIT_BATCH)
def check_batch():
    """Analyse up to BATCH_MAX_URLS URLs in one call.

    Request body:
        {"urls": ["https://...", ...], "html_map": {"https://...": "<html>..."}}

    html_map is optional; missing entries default to empty string.
    """
    data = request.get_json(force=True) or {}
    urls = data.get("urls", [])
    html_map: dict = data.get("html_map", {})

    if not isinstance(urls, list) or not urls:
        return jsonify({"error": "Provide a non-empty 'urls' list"}), 400

    if len(urls) > config.BATCH_MAX_URLS:
        return jsonify({
            "error": f"Batch limit is {config.BATCH_MAX_URLS} URLs per call"
        }), 400

    results = []
    for raw_url in urls:
        url_str  = str(raw_url).strip()
        html_str = str(html_map.get(url_str, "") or "")

        # Cache check
        cached = url_cache.get(url_str, html_str)
        if cached:
            results.append({**cached, "url": url_str, "cached": True})
            continue

        # Allowlist check
        from features import extract_url_features as _eu
        _uf = _eu(url_str)
        reg_domain = _uf.get("registered_domain", "")
        if is_allowlisted(reg_domain, _uf.get("detected_brand", "")):
            from allowlist import get_brand_for_domain
            res = _allowlist_safe_result(url_str, reg_domain, get_brand_for_domain(reg_domain))
            url_cache.put(url_str, html_str, res)
            results.append({**res, "url": url_str})
            continue

        res = _analyse(url_str, html_str)
        _persist_event(res, url_str)
        url_cache.put(url_str, html_str, res)
        results.append({**res, "url": url_str})

    logger.info(f"BATCH  count={len(results)}")
    return jsonify({"count": len(results), "results": results})


# ── Simulator ─────────────────────────────────────────────────────────────────
@app.post("/simulate")
@limiter.limit(config.RATE_LIMIT_SIMULATE)
def simulate():
    """Simulate detection with full feature breakdown for the dashboard."""
    data = request.get_json(force=True) or {}
    url     = data.get("url", "https://example.com")
    html    = data.get("html", "")
    persist = data.get("persist", False)

    url_str  = str(url).strip()
    html_str = str(html or "")

    url_features = extract_url_features(url_str)
    detected_brand = url_features.get("detected_brand")
    html_features, html_selectors = extract_html_features(
        html_str, url_str, detected_brand=detected_brand
    )
    all_features = {**url_features, **html_features}

    risk_score, risk_level, category_scores = predict_risk(all_features)
    reasons_text, structured_reasons, explain_selectors = reasons_for(all_features, url_str)
    highlights   = list(dict.fromkeys(html_selectors + explain_selectors))
    brand_target = url_features.get("detected_brand") or html_features.get("detected_brand_dom") or ""

    meta = {
        "domain":            url_features.get("domain", ""),
        "registered_domain": url_features.get("registered_domain", ""),
        "hostname":          url_features.get("hostname", ""),
        "has_https":         bool(url_features.get("has_https")),
        "brand_target":      brand_target,
        "ts":                datetime.utcnow().timestamp(),
    }

    if persist:
        _persist_event({
            "risk_score": float(risk_score), "risk_level": risk_level,
            "brand_target": brand_target, "reasons": reasons_text,
            "structured_reasons": structured_reasons,
            "category_scores": category_scores, "meta": meta,
        }, url_str)

    logger.info(
        f"SIM  score={risk_score:.3f}  level={risk_level:<10}  {url_str[:80]}"
    )

    return jsonify({
        "risk_score":         float(risk_score),
        "risk_level":         risk_level,
        "brand_target":       brand_target,
        "category_scores":    category_scores,
        "reasons":            reasons_text,
        "structured_reasons": structured_reasons,
        "highlights":         highlights,
        "raw_features":       all_features,
        "meta":               meta,
    })


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/stats")
def stats():
    try:
        return jsonify(DB.get_stats())
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.get("/stats/trend")
def stats_trend():
    """Return hourly scan counts for the past 24 h."""
    try:
        hours = min(int(request.args.get("hours", 24)), 168)   # cap at 7 days
        return jsonify({"hours": hours, "trend": DB.get_trend(hours)})
    except Exception as e:
        logger.error(f"Trend error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Events ────────────────────────────────────────────────────────────────────
@app.get("/events")
def list_events():
    limit  = min(int(request.args.get("limit", 100)), 500)
    offset = int(request.args.get("offset", 0))
    level  = request.args.get("level", "").upper()
    search = request.args.get("search", "").strip().lower()
    brand  = request.args.get("brand", "").strip().lower()

    with DB.session() as s:
        query = s.query(DetectionEvent)

        if level and level != "ALL":
            query = query.filter(DetectionEvent.risk_level == level)
        if brand:
            query = query.filter(DetectionEvent.brand_target.ilike(f"%{brand}%"))
        if search:
            query = query.filter(
                (DetectionEvent.url.ilike(f"%{search}%"))
                | (DetectionEvent.reasons_json.ilike(f"%{search}%"))
                | (DetectionEvent.brand_target.ilike(f"%{search}%"))
            )

        total_count = query.count()
        rows = (
            query.order_by(DetectionEvent.ts.desc())
            .offset(offset).limit(limit).all()
        )

        return jsonify({
            "total": total_count, "limit": limit, "offset": offset,
            "events": [r.to_dict() for r in rows],
        })


@app.get("/events/<int:event_id>")
def get_event(event_id: int):
    with DB.session() as s:
        evt = s.query(DetectionEvent).filter(DetectionEvent.id == event_id).first()
        if not evt:
            return jsonify({"error": "Event not found"}), 404
        return jsonify(evt.to_dict())


@app.delete("/events/<int:event_id>")
def delete_event(event_id: int):
    with DB.session() as s:
        evt = s.query(DetectionEvent).filter(DetectionEvent.id == event_id).first()
        if not evt:
            return jsonify({"error": "Event not found"}), 404
        s.delete(evt)
        s.commit()
        logger.info(f"DELETE event id={event_id}")
        return jsonify({"success": True, "deleted_id": event_id})


@app.delete("/events")
def clear_events():
    with DB.session() as s:
        s.query(DetectionEvent).delete()
        s.commit()
    url_cache.clear()
    logger.info("CLEAR all events")
    return jsonify({"success": True, "message": "All detection events cleared"})


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info(f"PhishLens listening on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=False)
