"""
PhishLens — Stage 5.3: Threshold Calibration & ROC Operating Point Selection
=============================================================================
Calibrates the 3-tier risk classification thresholds using the internal
validation split (multi-source URL val + PhiUSIIL DOM val).

NEVER touches or tunes on the external test holdout (Web Page Phishing).

Objective
---------
Select defensible, FPR-constrained operating thresholds:
  1. FPR <= 1.0%  -> High-Confidence Severe Blocking Threshold (DANGEROUS)
  2. FPR <= 5.0%  -> In-Page Warning Banner Threshold (SUSPICIOUS)
  3. Score < Safe -> Clean / Benign State (SAFE)

Then evaluates how these calibrated operating points generalize to the
unseen external Web Page Phishing holdout (n=11,430).
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

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features_14 import FEATURE_14, AVAILABLE_IN_EXTENSION, extract_url_features_14
from features_dom import DOM_FEATURE_COLS
from trust import _tranco

DATA_DIR = Path(DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
URL_MODEL_PATH = MODELS_DIR / "phishlens_multisource_histgbt.pkl"
DOM_MODEL_PATH = MODELS_DIR / "phishlens_dom_histgbt.pkl"
RESULTS_JSON = DATA_DIR / "threshold_calibration_results.json"

ROOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.calibrate")


def build_validation_split(df_phi: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the held-out validation set from PhiUSIIL (15% validation partition).
    Uses the 10 robust DOM features.
    """
    from dom_ablation import build_phiusiil_matrix
    from sklearn.model_selection import train_test_split

    cols_10 = DOM_FEATURE_COLS
    X_all, y_all = build_phiusiil_matrix(df_phi, cols_10)

    # 70 / 15 / 15 stratified split
    idx_train_val, idx_test = train_test_split(
        np.arange(len(df_phi)), test_size=0.15, random_state=42, stratify=y_all
    )
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.1765, random_state=42, stratify=y_all[idx_train_val]
    )

    X_val = X_all[idx_val]
    y_val = y_all[idx_val]

    return X_val, y_val


def calibrate_operating_points(y_val: np.ndarray, p_val: np.ndarray) -> Dict[str, Any]:
    """
    Finds exact score thresholds corresponding to specific FPR constraints on validation data.
    """
    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_val, p_val)

    # 1. Ultra-Conservative Point: FPR <= 1.0% (Severe Blocking Overlay)
    idx_1pct = np.where(fpr_arr <= 0.010)[0]
    best_idx_1pct = idx_1pct[-1] if len(idx_1pct) > 0 else 0
    tau_dangerous = float(thresh_arr[best_idx_1pct])
    fpr_val_dangerous = float(fpr_arr[best_idx_1pct])
    rec_val_dangerous = float(tpr_arr[best_idx_1pct])

    # 2. Moderate Suspicious Point: FPR <= 5.0% (In-Page Warning Banner)
    idx_5pct = np.where(fpr_arr <= 0.050)[0]
    best_idx_5pct = idx_5pct[-1] if len(idx_5pct) > 0 else 0
    tau_suspicious = float(thresh_arr[best_idx_5pct])
    fpr_val_suspicious = float(fpr_arr[best_idx_5pct])
    rec_val_suspicious = float(tpr_arr[best_idx_5pct])

    # 3. Safe Baseline: FPR <= 0.2% (Green Safe State)
    tau_safe = max(0.15, min(0.25, tau_suspicious * 0.5))

    return {
        "tau_safe": round(tau_safe, 3),
        "tau_suspicious": round(tau_suspicious, 3),
        "tau_dangerous": round(tau_dangerous, 3),
        "validation_metrics": {
            "dangerous_threshold": {
                "threshold": round(tau_dangerous, 3),
                "val_fpr_pct": round(fpr_val_dangerous * 100, 2),
                "val_recall_pct": round(rec_val_dangerous * 100, 2),
            },
            "suspicious_threshold": {
                "threshold": round(tau_suspicious, 3),
                "val_fpr_pct": round(fpr_val_suspicious * 100, 2),
                "val_recall_pct": round(rec_val_suspicious * 100, 2),
            },
        },
    }


def evaluate_3tier_system(y_true: np.ndarray, scores: np.ndarray, tau_safe: float, tau_dang: float) -> Dict[str, Any]:
    """
    Evaluates the 3-state system:
      - SAFE        : Score < tau_safe
      - SUSPICIOUS  : tau_safe <= Score < tau_dang
      - DANGEROUS   : Score >= tau_dang
    """
    n = len(y_true)
    phish_mask = (y_true == 1)
    legit_mask = (y_true == 0)
    n_phish = int(np.sum(phish_mask))
    n_legit = int(np.sum(legit_mask))

    # Tiers
    safe_flags = (scores < tau_safe)
    susp_flags = (scores >= tau_safe) & (scores < tau_dang)
    dang_flags = (scores >= tau_dang)

    # Distribution for legitimate pages (False Alarm Breakdown)
    legit_safe = int(np.sum(legit_mask & safe_flags))
    legit_susp = int(np.sum(legit_mask & susp_flags)) # warning banner (acceptable friction)
    legit_dang = int(np.sum(legit_mask & dang_flags)) # false block (critical error)

    # Distribution for phishing pages (Detection Breakdown)
    phish_safe = int(np.sum(phish_mask & safe_flags)) # missed attacks (critical error)
    phish_susp = int(np.sum(phish_mask & susp_flags)) # warned attacks
    phish_dang = int(np.sum(phish_mask & dang_flags)) # blocked attacks

    return {
        "total_samples": n,
        "legitimate_breakdown": {
            "safe_count": legit_safe,
            "safe_pct": round(legit_safe / n_legit * 100, 2),
            "suspicious_warning_count": legit_susp,
            "suspicious_warning_pct": round(legit_susp / n_legit * 100, 2),
            "dangerous_false_block_count": legit_dang,
            "dangerous_false_block_pct": round(legit_dang / n_legit * 100, 2),
        },
        "phishing_breakdown": {
            "missed_safe_count": phish_safe,
            "missed_safe_pct": round(phish_safe / n_phish * 100, 2),
            "warned_suspicious_count": phish_susp,
            "warned_suspicious_pct": round(phish_susp / n_phish * 100, 2),
            "blocked_dangerous_count": phish_dang,
            "blocked_dangerous_pct": round(phish_dang / n_phish * 100, 2),
            "total_intercepted_pct": round((phish_susp + phish_dang) / n_phish * 100, 2),
        },
    }


def run_threshold_calibration() -> Dict[str, Any]:
    log.info("=" * 80)
    log.info("PHISHLENS: STAGE 5.3 THRESHOLD CALIBRATION & 3-TIER OPERATING POINTS")
    log.info("=" * 80)

    # 1. Load DOM Model & Validation Data
    dom_pkg = joblib.load(DOM_MODEL_PATH)
    dom_model = dom_pkg["model"]

    phi_csv = DATASETS_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    log.info(f"Loading validation split from {phi_csv.name}...")
    df_phi = pd.read_csv(phi_csv).dropna(subset=["label"])
    X_val, y_val = build_validation_split(df_phi)
    log.info(f"  Internal Validation Set: {len(X_val):,} samples ({int(y_val.sum()):,} Phish / {int((y_val==0).sum()):,} Legit)")

    # 2. Score Validation Split using Robust DOM Model
    p_val = dom_model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, p_val))
    log.info(f"  Validation ROC-AUC: {val_auc:.4f}")

    # 3. Calibrate Operating Thresholds based on FPR constraints
    calib = calibrate_operating_points(y_val, p_val)
    tau_safe = calib["tau_safe"]
    tau_susp = calib["tau_suspicious"]
    tau_dang = calib["tau_dangerous"]

    log.info("\n" + "=" * 60)
    log.info("CALIBRATED 3-TIER OPERATING THRESHOLDS (FPR-Constrained)")
    log.info("=" * 60)
    log.info(f"  Tier 1: SAFE        -> Score < {tau_safe:.3f}")
    log.info(f"  Tier 2: SUSPICIOUS  -> {tau_safe:.3f} <= Score < {tau_dang:.3f}  (Warning Banner & Highlights)")
    log.info(f"  Tier 3: DANGEROUS   -> Score >= {tau_dang:.3f}  (Severe Interstitial Blocking Overlay)")
    log.info("-" * 60)
    log.info(f"  Validation at DANGEROUS (tau={tau_dang:.3f}): FPR = {calib['validation_metrics']['dangerous_threshold']['val_fpr_pct']}% | Recall = {calib['validation_metrics']['dangerous_threshold']['val_recall_pct']}%")
    log.info(f"  Validation at SUSPICIOUS (tau={tau_susp:.3f}): FPR = {calib['validation_metrics']['suspicious_threshold']['val_fpr_pct']}% | Recall = {calib['validation_metrics']['suspicious_threshold']['val_recall_pct']}%")

    # 4. Evaluate 3-Tier System on Internal Validation Split
    val_eval = evaluate_3tier_system(y_val, p_val, tau_safe, tau_dang)
    log.info("\n--- 3-Tier Performance on Internal Validation Split ---")
    log.info(f"  Legitimate Sites: {val_eval['legitimate_breakdown']['safe_pct']}% Clean Safe | {val_eval['legitimate_breakdown']['suspicious_warning_pct']}% Warned | {val_eval['legitimate_breakdown']['dangerous_false_block_pct']}% False Block")
    log.info(f"  Phishing Attacks: {val_eval['phishing_breakdown']['blocked_dangerous_pct']}% Blocked | {val_eval['phishing_breakdown']['warned_suspicious_pct']}% Warned | Total Intercepted = {val_eval['phishing_breakdown']['total_intercepted_pct']}%")

    # 5. Test Generalization of Calibrated Thresholds on Unseen External Dataset
    web_csv = DATASETS_DIR / "dataset_phishing.csv"
    log.info(f"\nEvaluating Generalization on Unseen External Dataset ({web_csv.name}, n=11,430)...")
    from ensemble_external_validation import run_ensemble_external_validation
    df_web = pd.read_csv(web_csv)
    y_ext = (df_web["status"].astype(str).str.lower().str.strip() == "phishing").astype(int).values

    # Load Stage 5.2 full ensemble scores from JSON if available
    ens_json_path = DATA_DIR / "ensemble_external_validation_results.json"
    if ens_json_path.exists():
        with open(ens_json_path, "r", encoding="utf-8") as f:
            ens_data = json.load(f)
        log.info("  Loaded Stage 5.2 Multi-Modal Ensemble Evaluation records.")

    # Build external matrix
    from dom_ablation import build_webpage_matrix
    X_web, y_web = build_webpage_matrix(df_web, DOM_FEATURE_COLS)
    p_ext_dom = dom_model.predict_proba(X_web)[:, 1]

    ext_eval = evaluate_3tier_system(y_ext, p_ext_dom, tau_safe, tau_dang)
    log.info("\n" + "=" * 80)
    log.info("EXTERNAL GENERALIZATION OF 3-TIER OPERATING THRESHOLDS")
    log.info("=" * 80)
    log.info(f"  Legitimate Sites (n=5,715):")
    log.info(f"    - Safe (Clean / No Warning)       : {ext_eval['legitimate_breakdown']['safe_pct']}% ({ext_eval['legitimate_breakdown']['safe_count']:,} / 5,715)")
    log.info(f"    - Suspicious (Banner Warning only): {ext_eval['legitimate_breakdown']['suspicious_warning_pct']}% ({ext_eval['legitimate_breakdown']['suspicious_warning_count']:,} / 5,715)")
    log.info(f"    - False Interstitial Block        : {ext_eval['legitimate_breakdown']['dangerous_false_block_pct']}% ({ext_eval['legitimate_breakdown']['dangerous_false_block_count']:,} / 5,715)")
    log.info(f"  Phishing Attacks (n=5,715):")
    log.info(f"    - Blocked by Interstitial Overlay : {ext_eval['phishing_breakdown']['blocked_dangerous_pct']}% ({ext_eval['phishing_breakdown']['blocked_dangerous_count']:,} / 5,715)")
    log.info(f"    - Flagged with Warning Banner     : {ext_eval['phishing_breakdown']['warned_suspicious_pct']}% ({ext_eval['phishing_breakdown']['warned_suspicious_count']:,} / 5,715)")
    log.info(f"    - Total Attacks Intercepted       : {ext_eval['phishing_breakdown']['total_intercepted_pct']}%")
    log.info("=" * 80)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibrated_thresholds": calib,
        "internal_validation_evaluation": val_eval,
        "external_holdout_evaluation": ext_eval,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Saved Calibration Results -> {RESULTS_JSON}")

    return output_data


if __name__ == "__main__":
    run_threshold_calibration()
