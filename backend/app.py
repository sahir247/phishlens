import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from features import extract_url_features, extract_html_features
from explain import reasons_for
from model import predict_risk
from storage import DB, DetectionEvent

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "ts": datetime.utcnow().timestamp()})


@app.post("/check")
def check():
    data = request.get_json(force=True) or {}
    url = data.get("url", "")
    html = data.get("html", "")
    if not url or not html:
        return jsonify({"error": "Missing 'url' or 'html'"}), 400

    url_features = extract_url_features(str(url))
    html_features, html_selectors = extract_html_features(html, str(url))

    features = {**url_features, **html_features}
    risk_score = predict_risk(features)

    reasons, selectors = reasons_for(features, str(url))
    highlights = list(set(html_selectors + selectors))

    meta = {
        "domain": url_features.get("domain", ""),
        "ts": datetime.utcnow().timestamp(),
    }

    with DB.session() as s:
        evt = DetectionEvent(
            url=str(url),
            risk_score=float(risk_score),
            reasons_json=json.dumps(reasons),
            ts=datetime.utcnow().timestamp(),
        )
        s.add(evt)
        s.commit()

    return jsonify({
        "risk_score": float(risk_score),
        "reasons": reasons,
        "highlights": highlights,
        "meta": meta,
    })


@app.post("/events")
def add_event():
    data = request.get_json(force=True) or {}
    url = data.get("url", "")
    risk_score = float(data.get("risk_score", 0.0))
    reasons = data.get("reasons", [])
    ts = float(data.get("ts") or datetime.utcnow().timestamp())
    with DB.session() as s:
        e = DetectionEvent(
            url=str(url),
            risk_score=risk_score,
            reasons_json=json.dumps(reasons),
            ts=ts,
        )
        s.add(e)
        s.commit()
        s.refresh(e)
        return jsonify({
            "id": e.id,
            "url": e.url,
            "risk_score": e.risk_score,
            "reasons": json.loads(e.reasons_json or "[]"),
            "ts": e.ts,
        })


@app.get("/events")
def list_events():
    try:
        limit = int(request.args.get("limit", 100))
    except Exception:
        limit = 100
    with DB.session() as s:
        rows = s.query(DetectionEvent).order_by(DetectionEvent.ts.desc()).limit(limit).all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "url": r.url,
                "risk_score": r.risk_score,
                "reasons": json.loads(r.reasons_json or "[]"),
                "ts": r.ts,
            })
        return jsonify(out)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
