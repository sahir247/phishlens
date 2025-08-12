from typing import Dict

# Lightweight heuristic-based risk pending ML pipeline
# Returns value in [0,1]

def predict_risk(feat: Dict[str, float]) -> float:
    score = 0.0

    # Weighting of features (quick baseline)
    weights = {
        "suspicious_kw": 0.25,
        "brand_kw": 0.1,
        "brand_text_hit": 0.1,
        "logo_mismatch": 0.2,
        "form_action_diff_domain": 0.2,
        "form_insecure_http": 0.15,
        "num_pw_inputs": 0.1,
        "has_ip": 0.2,
        "num_at": 0.1,
        "subdomain_count": 0.08,
        "entropy_path": 0.05,
        "url_len": 0.02,
    }

    # Normalize some numeric features
    norm = {
        "subdomain_count": min(feat.get("subdomain_count", 0.0) / 5.0, 1.0),
        "entropy_path": min(feat.get("entropy_path", 0.0) / 5.0, 1.0),
        "url_len": min(feat.get("url_len", 0.0) / 200.0, 1.0),
        "num_at": min(feat.get("num_at", 0.0), 1.0),
        "num_pw_inputs": min(feat.get("num_pw_inputs", 0.0) / 2.0, 1.0),
    }

    for k, w in weights.items():
        v = feat.get(k)
        if v is None:
            v = norm.get(k, 0.0)
        else:
            # if key is in norm list, replace with normalized value
            if k in norm:
                v = norm[k]
        score += w * float(v)

    # clamp
    score = max(0.0, min(1.0, score))
    return score
