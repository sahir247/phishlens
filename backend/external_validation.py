"""
PhishLens — Multi-Dataset External Validation Framework
=========================================================
Evaluates the trained E5 14-feature HistGradientBoosting model (and the
9-feature URL-only baseline) against unseen, independent external datasets.

Datasets Evaluated
------------------
1. Vrbancic 2020 (Held-out Test Split, n=13,298) — Control Baseline
2. LegitPhish (Mendeley, n=101,218 valid) — Threat intelligence URLs
3. PhiUSIIL (UCI, n=235,795 valid) — Large-scale 2024 benchmark
4. Web Page Phishing (n=11,430 valid) — Balanced URL+DOM benchmark

Label Semantics Normalization
-----------------------------
All evaluations strictly adhere to PhishLens binary ground-truth:
  - Label 1 = PHISHING (Malicious)
  - Label 0 = LEGITIMATE (Benign)

Dataset label mapping:
  - Vrbancic 2020 : phishing == 0 -> 1 (Phish), phishing == 1 -> 0 (Legit)
  - LegitPhish    : ClassLabel == 0.0 -> 1 (Phish), ClassLabel == 1.0 -> 0 (Legit)
  - PhiUSIIL      : label == 0 -> 1 (Phish), label == 1 -> 0 (Legit)
  - Web Page      : status == 'phishing' -> 1, status == 'legitimate' -> 0

Usage
-----
    python backend/external_validation.py
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features_14 import (
    FEATURE_14, N_FEATURES_14, AVAILABLE_IN_EXTENSION,
    extract_url_features_14, enrich_with_network_features,
)
from experiments import load_base_14, make_canonical_splits, make_histgbt

DATA_DIR = Path(DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_E5_PATH = MODELS_DIR / "phishlens_14_histgbt.pkl"
MODEL_9_PATH  = MODELS_DIR / "phishlens_9_url_histgbt.pkl"
RESULTS_JSON  = DATA_DIR / "external_validation_results.json"

ROOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.extval")

RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model Preparation & Artifact Persistence
# ═══════════════════════════════════════════════════════════════════════════════

def get_or_train_e5_models() -> Tuple[CalibratedClassifierCV, CalibratedClassifierCV, np.ndarray, np.ndarray]:
    """
    Trains E5 (14-feature HistGBT) and E_URL9 (9-feature URL-only HistGBT)
    on the canonical Vrbancic training split and calibrates on val split.
    Saves models to disk and returns (model_e5, model_9, X_test_vrb, y_test_vrb).
    """
    df14, y = load_base_14()
    X14 = df14.values
    X9  = df14[AVAILABLE_IN_EXTENSION].values

    idx_train, idx_val, idx_test = make_canonical_splits(len(y), y)

    # ── Train & calibrate E5 (14-feature) ─────────────────────────────────────
    if MODEL_E5_PATH.exists():
        log.info(f"Loading existing E5 model artifact ({MODEL_E5_PATH.name})...")
        pkg_e5 = joblib.load(MODEL_E5_PATH)
        model_e5 = pkg_e5["model"]
    else:
        log.info("Training canonical E5 14-feature HistGBT on Vrbancic...")
        base_e5 = make_histgbt()
        base_e5.fit(X14[idx_train], y[idx_train])
        model_e5 = CalibratedClassifierCV(base_e5, method="sigmoid", cv="prefit")
        model_e5.fit(X14[idx_val], y[idx_val])
        joblib.dump({"model": model_e5, "features": FEATURE_14}, MODEL_E5_PATH)
        log.info(f"Saved E5 model -> {MODEL_E5_PATH} ({MODEL_E5_PATH.stat().st_size//1024} KB)")

    # ── Train & calibrate 9-feature URL-only model ────────────────────────────
    if MODEL_9_PATH.exists():
        log.info(f"Loading existing 9-feature model artifact ({MODEL_9_PATH.name})...")
        pkg_9 = joblib.load(MODEL_9_PATH)
        model_9 = pkg_9["model"]
    else:
        log.info("Training dedicated 9-feature URL-only HistGBT on Vrbancic...")
        base_9 = make_histgbt()
        base_9.fit(X9[idx_train], y[idx_train])
        model_9 = CalibratedClassifierCV(base_9, method="sigmoid", cv="prefit")
        model_9.fit(X9[idx_val], y[idx_val])
        joblib.dump({"model": model_9, "features": AVAILABLE_IN_EXTENSION}, MODEL_9_PATH)
        log.info(f"Saved 9-feature model -> {MODEL_9_PATH} ({MODEL_9_PATH.stat().st_size//1024} KB)")

    X_test_vrb = X14[idx_test]
    y_test_vrb = y[idx_test]
    return model_e5, model_9, X_test_vrb, y_test_vrb


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Fast Deterministic Feature Extraction on Raw URLs
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_from_url_series(
    urls: pd.Series,
    desc: str = "dataset",
) -> Tuple[np.ndarray, float]:
    """
    Extracts the 14 features across a Pandas series of URLs using features_14.py.
    Network features default to np.nan (unobserved / browser-only mode).
    Returns (feature_matrix_shape_(n,14), extraction_time_seconds).
    """
    t0 = time.monotonic()
    n = len(urls)
    feat_matrix = np.full((n, N_FEATURES_14), np.nan, dtype=np.float64)

    # Pre-map feature indices for speed
    feat_idx = {col: i for i, col in enumerate(FEATURE_14)}

    for idx, raw_url in enumerate(urls):
        if not isinstance(raw_url, str) or not raw_url.strip():
            continue
        try:
            feats = extract_url_features_14(raw_url)
            for k, val in feats.items():
                if k in feat_idx:
                    # Map -1.0 sentinel in network features to NaN for HistGBT
                    if val < 0 and k not in AVAILABLE_IN_EXTENSION:
                        feat_matrix[idx, feat_idx[k]] = np.nan
                    else:
                        feat_matrix[idx, feat_idx[k]] = val
        except Exception:
            pass

    elapsed = time.monotonic() - t0
    log.info(f"Extracted {n:,} URLs from {desc} in {elapsed:.2f}s ({(elapsed/n)*1000:.4f} ms/url, {n/elapsed:,.0f} urls/sec)")
    return feat_matrix, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Metrics & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    infer_time_ms: float,
) -> Dict:
    auc   = roc_auc_score(y_true, y_prob)
    prauc = average_precision_score(y_true, y_prob)
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr   = fp / max(fp + tn, 1)
    fnr   = fn / max(fn + tp, 1)

    return {
        "auc_roc":          round(float(auc), 4),
        "pr_auc":           round(float(prauc), 4),
        "accuracy":         round(float(acc), 4),
        "precision":        round(float(prec), 4),
        "recall_tpr":       round(float(rec), 4),
        "f1_score":         round(float(f1), 4),
        "fpr":              round(float(fpr), 4),
        "fnr":              round(float(fnr), 4),
        "tn":               int(tn),
        "fp":               int(fp),
        "fn":               int(fn),
        "tp":               int(tp),
        "infer_ms_per_url": round(float(infer_time_ms), 4),
    }


def evaluate_dataset(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    model_e5: CalibratedClassifierCV,
    model_9: CalibratedClassifierCV,
    provenance: str = "",
) -> Dict:
    n_total = len(y)
    n_phish = int(np.sum(y == 1))
    n_legit = int(np.sum(y == 0))
    pos_ratio = n_phish / max(n_total, 1)

    log.info("=" * 80)
    log.info(f"EVALUATION: {name}")
    log.info(f"  Provenance : {provenance}")
    log.info(f"  Total Rows : {n_total:,}  |  Phishing (1): {n_phish:,} ({pos_ratio*100:.1f}%)  |  Legit (0): {n_legit:,} ({(1-pos_ratio)*100:.1f}%)")
    log.info("-" * 80)

    # 1. E5 (14-feature model) evaluated in URL-only mode (missing network features = NaN)
    t0 = time.monotonic()
    for _ in range(3):
        _ = model_e5.predict_proba(X[:min(500, n_total)])
    infer_ms_e5 = (time.monotonic() - t0) / 3 / min(500, n_total) * 1000

    y_pred_e5 = model_e5.predict(X)
    y_prob_e5 = model_e5.predict_proba(X)[:, 1]
    metrics_e5 = compute_metrics(y, y_pred_e5, y_prob_e5, infer_ms_e5)

    log.info(f"  [E5: 14-Feature HistGBT (URL-Only Mode)]")
    log.info(f"    AUC-ROC: {metrics_e5['auc_roc']:.4f} | PR-AUC: {metrics_e5['pr_auc']:.4f} | Accuracy: {metrics_e5['accuracy']*100:.2f}% | F1: {metrics_e5['f1_score']:.4f}")
    log.info(f"    Precision: {metrics_e5['precision']*100:.2f}% | Recall (TPR): {metrics_e5['recall_tpr']*100:.2f}% | FPR: {metrics_e5['fpr']*100:.2f}% | FNR: {metrics_e5['fnr']*100:.2f}%")
    log.info(f"    Confusion Matrix: TN={metrics_e5['tn']:,} FP={metrics_e5['fp']:,} | FN={metrics_e5['fn']:,} TP={metrics_e5['tp']:,}")

    # 2. Dedicated 9-feature URL-only model
    X9 = X[:, :len(AVAILABLE_IN_EXTENSION)]
    t0 = time.monotonic()
    for _ in range(3):
        _ = model_9.predict_proba(X9[:min(500, n_total)])
    infer_ms_9 = (time.monotonic() - t0) / 3 / min(500, n_total) * 1000

    y_pred_9 = model_9.predict(X9)
    y_prob_9 = model_9.predict_proba(X9)[:, 1]
    metrics_9 = compute_metrics(y, y_pred_9, y_prob_9, infer_ms_9)

    log.info(f"  [Dedicated 9-Feature URL-Only HistGBT]")
    log.info(f"    AUC-ROC: {metrics_9['auc_roc']:.4f} | PR-AUC: {metrics_9['pr_auc']:.4f} | Accuracy: {metrics_9['accuracy']*100:.2f}% | F1: {metrics_9['f1_score']:.4f}")
    log.info(f"    Precision: {metrics_9['precision']*100:.2f}% | Recall (TPR): {metrics_9['recall_tpr']*100:.2f}% | FPR: {metrics_9['fpr']*100:.2f}% | FNR: {metrics_9['fnr']*100:.2f}%")

    return {
        "dataset_name":     name,
        "provenance":       provenance,
        "total_samples":    n_total,
        "phishing_samples": n_phish,
        "legit_samples":    n_legit,
        "phishing_pct":     round(pos_ratio * 100, 2),
        "e5_14feat_metrics": metrics_e5,
        "dedicated_9feat_metrics": metrics_9,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Enriched Sample Validation (Testing 14-feature Network Boost)
# ═══════════════════════════════════════════════════════════════════════════════

from concurrent.futures import ThreadPoolExecutor, as_completed

def _enrich_single_url(u: str, base_f: Dict[str, float], feat_idx: Dict[str, int]) -> Tuple[np.ndarray, bool]:
    row = np.full(N_FEATURES_14, np.nan, dtype=np.float64)
    # Default base features
    for k, v in base_f.items():
        if k in AVAILABLE_IN_EXTENSION:
            row[feat_idx[k]] = v
        else:
            row[feat_idx[k]] = np.nan
    hit = False
    try:
        enr_f = enrich_with_network_features(base_f, u, timeout=1.5, skip_expensive=False)
        for k, v in enr_f.items():
            if v < 0 and k not in AVAILABLE_IN_EXTENSION:
                row[feat_idx[k]] = np.nan
            else:
                row[feat_idx[k]] = v
        if enr_f.get("ttl_hostname", -1) > 0 or enr_f.get("time_response", -1) > 0 or enr_f.get("time_domain_activation", -1) > 0:
            hit = True
    except Exception:
        pass
    return row, hit


def run_enriched_sample_validation(
    urls: pd.Series,
    y: np.ndarray,
    model_e5: CalibratedClassifierCV,
    sample_size: int = 200,
    desc: str = "sample",
) -> Optional[Dict]:
    """
    Takes a balanced sample of URLs, executes concurrent backend network enrichment
    (TCP latency, DNS TTL, WHOIS via RDAP), and measures E5 performance with
    full 14 features vs 9 URL-only features on the SAME sample.
    """
    log.info(f"\nRunning Network-Enriched Validation on {sample_size} balanced URLs from {desc} (concurrent lookups)...")
    
    idx_phish = np.where(y == 1)[0]
    idx_legit = np.where(y == 0)[0]
    
    n_each = min(sample_size // 2, len(idx_phish), len(idx_legit))
    rng = np.random.default_rng(RANDOM_SEED)
    sel_p = rng.choice(idx_phish, size=n_each, replace=False)
    sel_l = rng.choice(idx_legit, size=n_each, replace=False)
    sel_idx = np.concatenate([sel_p, sel_l])
    rng.shuffle(sel_idx)

    sample_urls = urls.iloc[sel_idx].values
    sample_y = y[sel_idx]

    feat_matrix_unenriched = np.full((len(sample_urls), N_FEATURES_14), np.nan, dtype=np.float64)
    feat_matrix_enriched   = np.full((len(sample_urls), N_FEATURES_14), np.nan, dtype=np.float64)
    feat_idx = {col: i for i, col in enumerate(FEATURE_14)}

    # Base unenriched
    for i, u in enumerate(sample_urls):
        base_f = extract_url_features_14(u)
        for k, v in base_f.items():
            if k in AVAILABLE_IN_EXTENSION:
                feat_matrix_unenriched[i, feat_idx[k]] = v
            else:
                feat_matrix_unenriched[i, feat_idx[k]] = np.nan

    t0 = time.monotonic()
    successful_lookups = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(_enrich_single_url, u, extract_url_features_14(u), feat_idx): i
            for i, u in enumerate(sample_urls)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                row, hit = fut.result()
                feat_matrix_enriched[i] = row
                if hit:
                    successful_lookups += 1
            except Exception:
                feat_matrix_enriched[i] = feat_matrix_unenriched[i]

    elapsed = time.monotonic() - t0
    log.info(f"Enriched {len(sample_urls)} URLs in {elapsed:.2f}s ({successful_lookups}/{len(sample_urls)} network hits)")

    # Compare Unenriched vs Enriched on this sample
    prob_unenr = model_e5.predict_proba(feat_matrix_unenriched)[:, 1]
    pred_unenr = model_e5.predict(feat_matrix_unenriched)
    m_unenr = compute_metrics(sample_y, pred_unenr, prob_unenr, 0.05)

    prob_enr = model_e5.predict_proba(feat_matrix_enriched)[:, 1]
    pred_enr = model_e5.predict(feat_matrix_enriched)
    m_enr = compute_metrics(sample_y, pred_enr, prob_enr, 0.05)

    log.info(f"  Enrichment Comparison on {desc} ({len(sample_urls)} URLs):")
    log.info(f"    URL-Only (9 feat)  -> AUC: {m_unenr['auc_roc']:.4f} | Acc: {m_unenr['accuracy']*100:.1f}% | FPR: {m_unenr['fpr']*100:.1f}% | FNR: {m_unenr['fnr']*100:.1f}%")
    log.info(f"    Enriched (14 feat) -> AUC: {m_enr['auc_roc']:.4f} | Acc: {m_enr['accuracy']*100:.1f}% | FPR: {m_enr['fpr']*100:.1f}% | FNR: {m_enr['fnr']*100:.1f}%")
    
    return {
        "sample_size": len(sample_urls),
        "enrichment_duration_s": round(elapsed, 2),
        "url_only_sample_metrics": m_unenr,
        "enriched_sample_metrics": m_enr,
    }



# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main Execution Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("=" * 80)
    log.info("PHISHLENS MULTI-DATASET EXTERNAL VALIDATION FRAMEWORK")
    log.info("=" * 80)

    # 1. Prepare / verify E5 model on Vrbancic
    model_e5, model_9, X_test_vrb, y_test_vrb = get_or_train_e5_models()

    all_results = {}

    # ── Benchmark 0: Vrbancic 2020 (Held-Out Test Control) ───────────────────
    res_vrb = evaluate_dataset(
        name="Vrbancic 2020 (Test Set Control)",
        X=X_test_vrb,
        y=y_test_vrb,
        model_e5=model_e5,
        model_9=model_9,
        provenance="Held-out 15% stratified test split (same data distribution)",
    )
    all_results["vrbancic_control"] = res_vrb

    # ── Benchmark 1: Web Page Phishing (dataset_phishing.csv) ─────────────────
    path_wp = DATASETS_DIR / "dataset_phishing.csv"
    if path_wp.exists():
        df_wp = pd.read_csv(path_wp)
        # Normalize labels: status == 'phishing' -> 1, 'legitimate' -> 0
        y_wp = (df_wp["status"].str.strip().str.lower() == "phishing").astype(int).values
        urls_wp = df_wp["url"].astype(str)
        X_wp, _ = extract_features_from_url_series(urls_wp, desc="Web Page Phishing (dataset_phishing.csv)")
        
        res_wp = evaluate_dataset(
            name="Web Page Phishing Benchmark (11,430 URLs)",
            X=X_wp,
            y=y_wp,
            model_e5=model_e5,
            model_9=model_9,
            provenance="Balanced URL+DOM dataset (5,715 Phish / 5,715 Legit)",
        )
        # Run enriched sample
        enr_wp = run_enriched_sample_validation(urls_wp, y_wp, model_e5, sample_size=100, desc="Web Page Phishing")
        res_wp["enriched_sample_analysis"] = enr_wp
        all_results["web_page_phishing"] = res_wp

    # ── Benchmark 2: LegitPhish / Mendeley (url_features_extracted1.csv) ─────
    path_lp = DATASETS_DIR / "url_features_extracted1.csv"
    if path_lp.exists():
        df_lp = pd.read_csv(path_lp)
        df_lp = df_lp.dropna(subset=["URL", "ClassLabel"])
        # Normalize labels: ClassLabel 0.0 -> 1 (Phishing), ClassLabel 1.0 -> 0 (Legit)
        y_lp = (df_lp["ClassLabel"].astype(float) == 0.0).astype(int).values
        urls_lp = df_lp["URL"].astype(str)
        X_lp, _ = extract_features_from_url_series(urls_lp, desc="LegitPhish (url_features_extracted1.csv)")
        
        res_lp = evaluate_dataset(
            name="LegitPhish / Mendeley Benchmark (101,218 URLs)",
            X=X_lp,
            y=y_lp,
            model_e5=model_e5,
            model_9=model_9,
            provenance="Verified threat-intel URLs (URLHaus, PhishTank, 63% Phish / 37% Legit)",
        )
        enr_lp = run_enriched_sample_validation(urls_lp, y_lp, model_e5, sample_size=100, desc="LegitPhish")
        res_lp["enriched_sample_analysis"] = enr_lp
        all_results["legit_phish"] = res_lp

    # ── Benchmark 3: PhiUSIIL (PhiUSIIL_Phishing_URL_Dataset.csv) ─────────────
    path_phi = DATASETS_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if path_phi.exists():
        df_phi = pd.read_csv(path_phi)
        df_phi = df_phi.dropna(subset=["URL", "label"])
        # Normalize labels: label == 0 -> 1 (Phishing), label == 1 -> 0 (Legit)
        y_phi = (df_phi["label"].astype(int) == 0).astype(int).values
        urls_phi = df_phi["URL"].astype(str)
        X_phi, _ = extract_features_from_url_series(urls_phi, desc="PhiUSIIL (PhiUSIIL_Phishing_URL_Dataset.csv)")
        
        res_phi = evaluate_dataset(
            name="PhiUSIIL 2024 Benchmark (235,795 URLs)",
            X=X_phi,
            y=y_phi,
            model_e5=model_e5,
            model_9=model_9,
            provenance="UCI 2024 dataset with URL + webpage source code (43% Phish / 57% Legit)",
        )
        enr_phi = run_enriched_sample_validation(urls_phi, y_phi, model_e5, sample_size=100, desc="PhiUSIIL")
        res_phi["enriched_sample_analysis"] = enr_phi
        all_results["phiusiil"] = res_phi

    # ── Summary Cross-Dataset Comparison Table ─────────────────────────────────
    log.info("\n" + "=" * 115)
    log.info("CROSS-DATASET GENERALIZATION SUMMARY TABLE (E5 Model Evaluated Across All Datasets)")
    log.info("=" * 115)
    log.info(f"{'Dataset':<35} {'Samples':>9} {'Phish%':>8} {'AUC-ROC':>9} {'PR-AUC':>8} {'Accuracy':>9} {'F1':>7} {'FPR':>7} {'FNR':>7}")
    log.info("-" * 115)

    for k, d in all_results.items():
        m = d["e5_14feat_metrics"]
        log.info(f"{d['dataset_name'][:35]:<35} {d['total_samples']:>9,} {d['phishing_pct']:>7.1f}% {m['auc_roc']:>9.4f} {m['pr_auc']:>8.4f} {m['accuracy']*100:>8.2f}% {m['f1_score']:>7.4f} {m['fpr']*100:>6.2f}% {m['fnr']*100:>6.2f}%")

    log.info("=" * 115)

    # Save to JSON
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nAll detailed metrics saved -> {RESULTS_JSON}")
    log.info(f"Total validation runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
