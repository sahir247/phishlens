"""
PhishLens ML Classifier  —  Inference Module
---------------------------------------------
This module handles model loading and inference only.
Training is done separately via train_model.py which:
  - Downloads real phishing + benign URL datasets
  - Extracts features using features.py
  - Runs a full professional ML pipeline
  - Saves a calibrated CalibratedClassifierCV to MODEL_PATH

This file intentionally contains the synthetic fallback ONLY as a last resort
when no real model exists.  The correct path is to run train_model.py first.

Usage:
    python train_model.py        # train on real data (recommended)
    python ml_model.py           # fallback: train on synthetic data
"""

import os
import json
import logging
import numpy as np

import joblib

from config import MODEL_PATH

logger = logging.getLogger(__name__)

# ── Feature column order (must match build_feature_vector) ────────────────────
FEATURE_COLS = [
    "has_ip",
    "has_punycode",
    "is_suspicious_tld",
    "subdomain_count",
    "domain_len",
    "url_len",
    "num_dashes",
    "num_at",
    "entropy_path",
    "has_https",
    "suspicious_kw",
    "kw_hit_count",
    "brand_in_url",
    "brand_in_subdomain",
    "typosquat_score",
    "num_pw_inputs",
    "form_action_diff_domain",
    "form_insecure_http",
    "logo_mismatch",
    "hidden_iframes",
]

N_FEATURES = len(FEATURE_COLS)


# ── Synthetic Dataset Generator ───────────────────────────────────────────────

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _generate_phishing_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Phishing pages cluster around high-risk feature combinations:
    brand impersonation + credential harvesting patterns.
    """
    samples = np.zeros((n, N_FEATURES))
    idx = {c: i for i, c in enumerate(FEATURE_COLS)}

    # has_ip: 20% of phishing use IP
    samples[:, idx["has_ip"]] = rng.choice([0, 1], size=n, p=[0.80, 0.20])
    # has_punycode: 15% use homograph
    samples[:, idx["has_punycode"]] = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    # suspicious tld: 55%
    samples[:, idx["is_suspicious_tld"]] = rng.choice([0, 1], size=n, p=[0.45, 0.55])
    # subdomain_count: phishing loves deep subdomains
    samples[:, idx["subdomain_count"]] = rng.integers(0, 6, size=n)
    # domain_len: often longer
    samples[:, idx["domain_len"]] = rng.integers(10, 60, size=n)
    # url_len: usually long
    samples[:, idx["url_len"]] = rng.integers(60, 250, size=n)
    # num_dashes: keyword stuffing with hyphens
    samples[:, idx["num_dashes"]] = rng.integers(1, 10, size=n)
    # num_at: 12% use @
    samples[:, idx["num_at"]] = rng.choice([0, 1], size=n, p=[0.88, 0.12])
    # entropy_path: high entropy (obfuscated tokens)
    samples[:, idx["entropy_path"]] = rng.uniform(3.0, 5.5, size=n)
    # has_https: 45% of phishing now uses HTTPS
    samples[:, idx["has_https"]] = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    # suspicious keywords: 85%
    samples[:, idx["suspicious_kw"]] = rng.choice([0, 1], size=n, p=[0.15, 0.85])
    # kw_hit_count: 1-5 keywords
    samples[:, idx["kw_hit_count"]] = rng.integers(1, 6, size=n) * samples[:, idx["suspicious_kw"]]
    # brand_in_url: 80%
    samples[:, idx["brand_in_url"]] = rng.choice([0, 1], size=n, p=[0.20, 0.80])
    # brand_in_subdomain: 50%
    samples[:, idx["brand_in_subdomain"]] = rng.choice([0, 1], size=n, p=[0.50, 0.50])
    # typosquat_score: 0.6-1.0
    samples[:, idx["typosquat_score"]] = rng.uniform(0.0, 1.0, size=n)
    # num_pw_inputs: 0-2
    samples[:, idx["num_pw_inputs"]] = rng.integers(0, 3, size=n)
    # form_action_diff_domain: 60%
    samples[:, idx["form_action_diff_domain"]] = rng.choice([0, 1], size=n, p=[0.40, 0.60])
    # form_insecure_http: 35%
    samples[:, idx["form_insecure_http"]] = rng.choice([0, 1], size=n, p=[0.65, 0.35])
    # logo_mismatch: 55%
    samples[:, idx["logo_mismatch"]] = rng.choice([0, 1], size=n, p=[0.45, 0.55])
    # hidden_iframes: 25%
    samples[:, idx["hidden_iframes"]] = rng.choice([0, 1], size=n, p=[0.75, 0.25])

    return samples


def _generate_safe_samples(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Legitimate pages: mostly HTTPS, short/normal URLs, no cross-domain forms.
    """
    samples = np.zeros((n, N_FEATURES))
    idx = {c: i for i, c in enumerate(FEATURE_COLS)}

    samples[:, idx["has_ip"]] = rng.choice([0, 1], size=n, p=[0.99, 0.01])
    samples[:, idx["has_punycode"]] = rng.choice([0, 1], size=n, p=[0.995, 0.005])
    samples[:, idx["is_suspicious_tld"]] = rng.choice([0, 1], size=n, p=[0.95, 0.05])
    samples[:, idx["subdomain_count"]] = rng.integers(0, 3, size=n)
    samples[:, idx["domain_len"]] = rng.integers(4, 25, size=n)
    samples[:, idx["url_len"]] = rng.integers(15, 100, size=n)
    samples[:, idx["num_dashes"]] = rng.integers(0, 3, size=n)
    samples[:, idx["num_at"]] = rng.choice([0, 1], size=n, p=[0.99, 0.01])
    samples[:, idx["entropy_path"]] = rng.uniform(0.0, 3.5, size=n)
    samples[:, idx["has_https"]] = rng.choice([0, 1], size=n, p=[0.10, 0.90])
    samples[:, idx["suspicious_kw"]] = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    samples[:, idx["kw_hit_count"]] = rng.integers(0, 2, size=n) * samples[:, idx["suspicious_kw"]]
    samples[:, idx["brand_in_url"]] = rng.choice([0, 1], size=n, p=[0.75, 0.25])
    samples[:, idx["brand_in_subdomain"]] = rng.choice([0, 1], size=n, p=[0.96, 0.04])
    samples[:, idx["typosquat_score"]] = rng.uniform(0.0, 0.3, size=n)
    samples[:, idx["num_pw_inputs"]] = rng.integers(0, 2, size=n)
    samples[:, idx["form_action_diff_domain"]] = rng.choice([0, 1], size=n, p=[0.97, 0.03])
    samples[:, idx["form_insecure_http"]] = rng.choice([0, 1], size=n, p=[0.98, 0.02])
    samples[:, idx["logo_mismatch"]] = rng.choice([0, 1], size=n, p=[0.97, 0.03])
    samples[:, idx["hidden_iframes"]] = rng.choice([0, 1], size=n, p=[0.98, 0.02])

    return samples


def _build_training_data(n_total: int = 10_000, seed: int = 42):
    rng = _rng(seed)
    n_phish = n_total // 2
    n_safe = n_total - n_phish

    X_phish = _generate_phishing_samples(n_phish, rng)
    X_safe = _generate_safe_samples(n_safe, rng)

    X = np.vstack([X_phish, X_safe])
    y = np.array([1] * n_phish + [0] * n_safe)

    # Shuffle
    perm = rng.permutation(n_total)
    return X[perm], y[perm]


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save(model_path: str = MODEL_PATH) -> None:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    logger.info("PhishLens ML: generating synthetic training dataset (10 000 samples)…")
    X, y = _build_training_data(10_000)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("gbc", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            min_samples_split=15,
            random_state=42,
        )),
    ])

    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    logger.info(f"PhishLens ML: 5-fold CV AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    clf.fit(X, y)
    joblib.dump(clf, model_path)
    logger.info(f"PhishLens ML: model saved -> {model_path}")


# ── Inference ─────────────────────────────────────────────────────────────────

_clf = None
_clf_meta: dict = {}


def _load_model() -> object:
    global _clf, _clf_meta
    if _clf is not None:
        return _clf

    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "No trained model found at %s.  "
            "Run 'python backend/train_model.py' to train on real data.  "
            "Falling back to synthetic training (accuracy will be lower).",
            MODEL_PATH,
        )
        train_and_save(MODEL_PATH)

    _clf = joblib.load(MODEL_PATH)

    # Load metadata if available (produced by train_model.py)
    meta_path = os.path.join(os.path.dirname(MODEL_PATH), "data", "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                _clf_meta = json.load(f)
            m = _clf_meta.get("metrics", {})
            logger.info(
                "ML model loaded  |  type=%s  trained=%s  AUC=%.4f  F1=%.4f  FPR=%.4f",
                _clf_meta.get("model_type", "unknown"),
                _clf_meta.get("trained_at", "?")[:10],
                m.get("auc_roc", 0),
                m.get("f1_score", 0),
                m.get("false_positive_rate", 0),
            )
        except Exception as e:
            logger.debug("Could not load model metadata: %s", e)
            logger.info("ML model loaded from %s.", MODEL_PATH)
    else:
        logger.info("ML classifier loaded (no metadata file found).")

    return _clf


def build_feature_vector(feat: dict) -> np.ndarray:
    """Convert a features dict to a numpy array in FEATURE_COLS order."""
    return np.array(
        [float(feat.get(c, 0.0)) for c in FEATURE_COLS],
        dtype=np.float64,
    ).reshape(1, -1)


def get_ml_proba(feat: dict) -> float:
    """Return phishing probability (0–1) from the trained classifier."""
    try:
        model = _load_model()
        vec = build_feature_vector(feat)
        proba = model.predict_proba(vec)[0][1]  # class=1 (phishing)
        return float(proba)
    except Exception as e:
        logger.error(f"PhishLens ML inference error: {e}")
        return 0.0


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    train_and_save()
    print(f"Model saved to {MODEL_PATH}")
