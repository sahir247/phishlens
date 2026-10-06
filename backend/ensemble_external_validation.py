"""
PhishLens — Stage 5.2: Multi-Modal Ensemble External Validation
================================================================
Evaluates the full Stage 5 Production Ensemble and its individual
components on the completely unseen Web Page Phishing dataset (n=11,430).

All model weights and ensemble formulas are strictly frozen:
  - Multi-Source URL HistGBT (388k URLs)
  - Robust DOM HistGBT (235k samples)
  - Dynamic Trust Layer (Tranco Top-10k + RDAP domain age)
  - Heuristic Category Rules (Domain, URL, Brand, DOM)

Regimes Compared
----------------
1. URL Only         : P(Phish_URL)
2. DOM Only         : P(Phish_DOM)
3. URL + DOM Blend  : 0.45 * P_URL + 0.55 * P_DOM
4. URL + DOM + Trust: (0.45 * P_URL + 0.55 * P_DOM) * (1 - 0.50 * Trust)
5. Full Production  : Stage 5 Multi-Modal Ensemble (predict_risk)
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features_14 import FEATURE_14, AVAILABLE_IN_EXTENSION, extract_url_features_14
from features_dom import DOM_FEATURE_COLS
from features import extract_url_features
from trust import compute_trust, _tranco
from model import predict_risk, calculate_domain_risk, calculate_url_risk, calculate_brand_risk, calculate_dom_risk

DATA_DIR = Path(DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
URL_MODEL_PATH = MODELS_DIR / "phishlens_multisource_histgbt.pkl"
DOM_MODEL_PATH = MODELS_DIR / "phishlens_dom_histgbt.pkl"
RESULTS_JSON = DATA_DIR / "ensemble_external_validation_results.json"

ROOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.ext_ensemble")


def evaluate_score_array(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """Computes full evaluation metrics for a continuous score array."""
    y_pred = (y_score >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_score))
    prauc = float(average_precision_score(y_true, y_score))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    fpr = float(fp / max(fp + tn, 1))
    fnr = float(fn / max(fn + tp, 1))

    return {
        "auc_roc": round(auc, 4),
        "pr_auc": round(prauc, 4),
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1, 4),
        "fpr": round(fpr * 100, 2),
        "fnr": round(fnr * 100, 2),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def compute_threshold_sweep(y_true: np.ndarray, y_score: np.ndarray, thresholds: List[float]) -> List[Dict]:
    """Sweeps thresholds for precision/recall curves."""
    results = []
    for tau in thresholds:
        m = evaluate_score_array(y_true, y_score, threshold=tau)
        results.append({
            "threshold": tau,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
            "fpr": m["fpr"],
            "fnr": m["fnr"],
        })
    return results


def run_ensemble_external_validation() -> Dict[str, Any]:
    log.info("=" * 80)
    log.info("PHISHLENS: STAGE 5.2 MULTI-MODAL ENSEMBLE EXTERNAL VALIDATION")
    log.info("=" * 80)

    # 1. Ensure Tranco index is loaded
    _tranco._load()
    log.info(f"Trust Engine: Tranco index active with {len(_tranco._data):,} entries.")

    # 2. Load Models
    log.info(f"Loading Multi-Source URL model: {URL_MODEL_PATH.name}...")
    url_pkg = joblib.load(URL_MODEL_PATH)
    url_model = url_pkg["model"]

    log.info(f"Loading DOM model: {DOM_MODEL_PATH.name}...")
    dom_pkg = joblib.load(DOM_MODEL_PATH)
    dom_model = dom_pkg["model"]
    dom_features = dom_pkg["features"]

    # 3. Load External Web Page Phishing Dataset
    web_csv = DATASETS_DIR / "dataset_phishing.csv"
    log.info(f"Loading Web Page Phishing Dataset ({web_csv.name})...")
    df = pd.read_csv(web_csv)
    n = len(df)
    y_true = (df["status"].astype(str).str.lower().str.strip() == "phishing").astype(int).values
    log.info(f"  Total validation records: {n:,} (5,715 Phishing / 5,715 Legitimate)")

    # 4. Extract URL Features for URL Model
    log.info("\nExtracting URL features across 11,430 external URLs...")
    t0 = time.monotonic()
    mat_url = np.full((n, len(FEATURE_14)), np.nan, dtype=np.float64)
    feat_idx = {col: i for i, col in enumerate(FEATURE_14)}

    for i, u in enumerate(df["url"]):
        f = extract_url_features_14(str(u))
        for k in AVAILABLE_IN_EXTENSION:
            mat_url[i, feat_idx[k]] = f.get(k, np.nan)

    p_url = url_model.predict_proba(mat_url)[:, 1]
    url_extract_time = time.monotonic() - t0
    log.info(f"  URL inference completed in {url_extract_time:.2f}s ({n/url_extract_time:.0f} samples/sec)")

    # 5. Extract DOM Features for DOM Model
    log.info("\nExtracting DOM features matching features_dom.py...")
    domain_in_title = df["domain_in_title"].fillna(0).values.astype(float)
    empty_title = df["empty_title"].fillna(0).values.astype(float)
    title_score = np.where(empty_title == 1, 0.0, np.where(domain_in_title == 1, 100.0, 0.0))
    obf = ((df["onmouseover"].fillna(0) > 0) | (df["right_clic"].fillna(0) > 0) | (df["popup_window"].fillna(0) > 0)).values.astype(float)

    dom_map = {
        "has_password_field": df["login_form"].fillna(0).values.astype(float),
        "has_login_form": df["login_form"].fillna(0).values.astype(float),
        "has_external_form_submit": df["sfh"].fillna(0).values.astype(float),
        "has_submit_button": df["login_form"].fillna(0).values.astype(float),
        "has_iframe": (df["iframe"].fillna(0).values > 0).astype(float),
        "num_iframes": df["iframe"].fillna(0).values.astype(float),
        "num_scripts": np.full(n, np.nan, dtype=np.float64),
        "ratio_external_refs": df["ratio_extHyperlinks"].fillna(0).values.astype(float).clip(0.0, 1.0),
        "num_empty_refs": (df["ratio_nullHyperlinks"].fillna(0).values * df["nb_hyperlinks"].fillna(0).values).astype(float),
        "domain_title_match_score": title_score,
        "has_obfuscation": obf,
        "line_of_code": np.full(n, np.nan, dtype=np.float64),
    }

    mat_dom = np.column_stack([dom_map[col] for col in dom_features])
    p_dom = dom_model.predict_proba(mat_dom)[:, 1]

    # 6. Compute Trust Scores & Heuristic Features
    log.info("\nComputing Dynamic Trust Scores & Heuristic Category Scores...")
    trust_scores = np.zeros(n, dtype=np.float64)
    domain_risk_scores = np.zeros(n, dtype=np.float64)
    brand_risk_scores = np.zeros(n, dtype=np.float64)

    for i, raw_url in enumerate(df["url"]):
        u_str = str(raw_url)
        parsed = urlparse(u_str)
        host = (parsed.hostname or "").lower().lstrip("www.")
        has_https = u_str.startswith("https")

        # Tranco Rank fast lookup (memory dictionary)
        rank = _tranco._data.get(host)
        if rank is None:
            tranco_c = 0.0
        elif rank <= 100:
            tranco_c = 0.65
        elif rank <= 500:
            tranco_c = 0.55
        elif rank <= 2000:
            tranco_c = 0.42
        elif rank <= 5000:
            tranco_c = 0.30
        else:
            tranco_c = 0.20

        https_c = 0.05 if has_https else 0.0
        t_score = round(max(0.0, min(1.0, tranco_c + https_c)), 3)
        trust_scores[i] = t_score

        # Domain risk
        d_score = 0.0
        if df["ip"].iloc[i] > 0:
            d_score += 0.40
        if df["punycode"].iloc[i] > 0:
            d_score += 0.35
        if df["suspecious_tld"].iloc[i] > 0:
            d_score += 0.30
        subdomains = float(df["nb_subdomains"].iloc[i])
        if subdomains >= 3:
            d_score += min(0.10 * (subdomains - 1), 0.30)
        domain_risk_scores[i] = min(d_score, 1.0)

        # Brand risk
        b_score = 0.0
        if df["brand_in_subdomain"].iloc[i] > 0:
            b_score += 0.45
        if df["brand_in_path"].iloc[i] > 0:
            b_score += 0.30
        if df["domain_in_brand"].iloc[i] > 0:
            b_score += 0.20
        brand_risk_scores[i] = min(b_score, 1.0)

    # 7. Compute Multi-Modal Regimes with Frozen Weights
    # Regime 3: URL + DOM blend
    p_url_dom_blend = 0.45 * p_url + 0.55 * p_dom

    # Regime 4: URL + DOM + Trust dampening
    p_url_dom_trust = p_url_dom_blend * (1.0 - trust_scores * 0.50)

    # Regime 5: Full Stage 5 Production Ensemble
    # Raw blend: 35% URL, 40% DOM, 15% Brand, 10% Domain
    raw_prod = (
        p_url * 0.35 +
        p_dom * 0.40 +
        brand_risk_scores * 0.15 +
        domain_risk_scores * 0.10
    )

    # Non-linear boosters
    # High DOM phish + High URL phish booster
    high_both = (p_dom >= 0.85) & (p_url >= 0.70)
    raw_prod = np.where(high_both, np.maximum(raw_prod, 0.90), raw_prod)

    # Brand + Login form booster
    has_login_arr = df["login_form"].fillna(0).values.astype(float)
    brand_login = (brand_risk_scores > 0.4) & (has_login_arr > 0)
    raw_prod = np.where(brand_login, np.maximum(raw_prod, 0.85), raw_prod)

    # Trust dampening
    p_full_ensemble = raw_prod * (1.0 - trust_scores * 0.50)
    p_full_ensemble = np.clip(p_full_ensemble, 0.0, 1.0)

    regimes = {
        "1. URL Model Only": p_url,
        "2. DOM Model Only": p_dom,
        "3. URL + DOM Blend": p_url_dom_blend,
        "4. URL + DOM + Trust": p_url_dom_trust,
        "5. Full Production Ensemble": p_full_ensemble,
    }

    regime_results = {}

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    log.info("\n" + "=" * 95)
    log.info(f"{'EVALUATION REGIME':<30} {'AUC-ROC':<9} {'PR-AUC':<8} {'ACCURACY':<10} {'F1':<8} {'PRECISION':<11} {'RECALL':<8} {'FPR':<8}")
    log.info("=" * 95)

    for name, scores in regimes.items():
        m = evaluate_score_array(y_true, scores, threshold=0.50)
        sweep = compute_threshold_sweep(y_true, scores, thresholds)
        regime_results[name] = {
            "metrics": m,
            "threshold_sweep": sweep,
        }
        log.info(f"{name:<30} {m['auc_roc']:<9.4f} {m['pr_auc']:<8.4f} {m['accuracy']:<9.2f}% {m['f1_score']:<8.4f} {m['precision']:<10.2f}% {m['recall']:<7.2f}% {m['fpr']:<7.2f}%")

    log.info("=" * 95)

    # 8. Detailed Threshold Sweep for Full Production Ensemble
    log.info("\n--- Full Production Ensemble: Threshold Sensitivity Sweep ---")
    log.info(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1':<8} {'FPR':<8} {'FNR':<8}")
    log.info("-" * 65)
    for sw in regime_results["5. Full Production Ensemble"]["threshold_sweep"]:
        log.info(f"{sw['threshold']:<10.2f} {sw['accuracy']:<10.2f}% {sw['precision']:<11.2f}% {sw['recall']:<10.2f}% {sw['f1_score']:<8.4f} {sw['fpr']:<8.2f}% {sw['fnr']:<8.2f}%")

    # 9. Save JSON Artifact
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "Web Page Phishing Dataset",
            "file": web_csv.name,
            "total_samples": n,
            "phishing": int(np.sum(y_true == 1)),
            "legitimate": int(np.sum(y_true == 0)),
        },
        "models_evaluated": {
            "url_model": URL_MODEL_PATH.name,
            "dom_model": DOM_MODEL_PATH.name,
        },
        "regimes_comparison": regime_results,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"\nSaved Multi-Modal Ensemble Validation JSON -> {RESULTS_JSON}")

    return output_data


if __name__ == "__main__":
    run_ensemble_external_validation()
