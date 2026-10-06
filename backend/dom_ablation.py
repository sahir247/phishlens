"""
PhishLens — DOM Feature Ablation Study
=======================================
Trains and evaluates 4 DOM feature configurations on PhiUSIIL (n=235,795)
and tests each on the independent Web Page Phishing dataset (n=11,430).

Ablation Configurations
-----------------------
- D1: 12-Feature Baseline (Includes LineOfCode & NoOfJS)
- D2: 11-Feature (Excludes LineOfCode)
- D3: 10-Feature (Excludes LineOfCode & NoOfJS)
- D4: 9-Feature Robust DOM Core (Password, Login, ExtSubmit, SubmitBtn, iFrame, NumIframes, ExtRefRatio, TitleMatch, Obfuscation)

Goal: Identify the robust DOM model candidate that minimizes cross-dataset degradation.
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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features_dom import DOM_FEATURE_COLS

DATA_DIR = Path(DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = DATA_DIR / "dom_ablation_results.json"
ROOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.dom_ablation")

RANDOM_SEED = 42

# Define Feature Sets
FEATURE_SETS = {
    "D1_12feat_baseline": DOM_FEATURE_COLS,
    "D2_11feat_no_loc": [c for c in DOM_FEATURE_COLS if c != "line_of_code"],
    "D3_10feat_no_loc_scripts": [c for c in DOM_FEATURE_COLS if c not in ("line_of_code", "num_scripts")],
    "D4_9feat_robust_core": [
        "has_password_field",
        "has_login_form",
        "has_external_form_submit",
        "has_submit_button",
        "has_iframe",
        "num_iframes",
        "ratio_external_refs",
        "domain_title_match_score",
        "has_obfuscation",
    ],
}


def load_raw_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads PhiUSIIL and Web Page Phishing raw DataFrames."""
    phi_path = DATASETS_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    web_path = DATASETS_DIR / "dataset_phishing.csv"

    log.info(f"Loading PhiUSIIL from {phi_path.name}...")
    df_phi = pd.read_csv(phi_path).dropna(subset=["label"])
    
    log.info(f"Loading Web Page Phishing from {web_path.name}...")
    df_web = pd.read_csv(web_path).dropna(subset=["status"])

    return df_phi, df_web


def build_phiusiil_matrix(df_phi: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Builds PhiUSIIL feature matrix for a specific feature subset."""
    n = len(df_phi)
    y = (df_phi["label"].astype(int).values == 0).astype(int)

    # Master mapping dict
    ext = df_phi["NoOfExternalRef"].fillna(0).values.astype(float)
    self_ref = df_phi["NoOfSelfRef"].fillna(0).values.astype(float)
    emp = df_phi["NoOfEmptyRef"].fillna(0).values.astype(float)
    tot = ext + self_ref + emp + 1.0
    ratio_ext = (ext / tot).clip(0.0, 1.0)

    feature_map = {
        "has_password_field": df_phi["HasPasswordField"].fillna(0).values.astype(float),
        "has_login_form": np.maximum(df_phi["HasPasswordField"].fillna(0).values, (df_phi["HasSubmitButton"].fillna(0).values > 0).astype(float)),
        "has_external_form_submit": df_phi["HasExternalFormSubmit"].fillna(0).values.astype(float),
        "has_submit_button": df_phi["HasSubmitButton"].fillna(0).values.astype(float),
        "has_iframe": (df_phi["NoOfiFrame"].fillna(0).values > 0).astype(float),
        "num_iframes": df_phi["NoOfiFrame"].fillna(0).values.astype(float),
        "num_scripts": df_phi["NoOfJS"].fillna(0).values.astype(float),
        "ratio_external_refs": ratio_ext,
        "num_empty_refs": emp,
        "domain_title_match_score": df_phi["DomainTitleMatchScore"].fillna(50).values.astype(float),
        "has_obfuscation": df_phi["HasObfuscation"].fillna(0).values.astype(float),
        "line_of_code": df_phi["LineOfCode"].fillna(500).values.astype(float),
    }

    X = np.column_stack([feature_map[col] for col in feature_cols])
    return X, y


def build_webpage_matrix(df_web: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Builds Web Page Phishing feature matrix for a specific feature subset."""
    n = len(df_web)
    y = (df_web["status"].astype(str).str.lower().str.strip() == "phishing").astype(int).values

    domain_in_title = df_web["domain_in_title"].fillna(0).values.astype(float)
    empty_title = df_web["empty_title"].fillna(0).values.astype(float)
    title_score = np.where(empty_title == 1, 0.0, np.where(domain_in_title == 1, 100.0, 0.0))

    obf = ((df_web["onmouseover"].fillna(0) > 0) | (df_web["right_clic"].fillna(0) > 0) | (df_web["popup_window"].fillna(0) > 0)).values.astype(float)

    feature_map = {
        "has_password_field": df_web["login_form"].fillna(0).values.astype(float),
        "has_login_form": df_web["login_form"].fillna(0).values.astype(float),
        "has_external_form_submit": df_web["sfh"].fillna(0).values.astype(float),
        "has_submit_button": df_web["login_form"].fillna(0).values.astype(float),
        "has_iframe": (df_web["iframe"].fillna(0).values > 0).astype(float),
        "num_iframes": df_web["iframe"].fillna(0).values.astype(float),
        "num_scripts": np.full(n, np.nan, dtype=np.float64),
        "ratio_external_refs": df_web["ratio_extHyperlinks"].fillna(0).values.astype(float).clip(0.0, 1.0),
        "num_empty_refs": (df_web["ratio_nullHyperlinks"].fillna(0).values * df_web["nb_hyperlinks"].fillna(0).values).astype(float),
        "domain_title_match_score": title_score,
        "has_obfuscation": obf,
        "line_of_code": np.full(n, np.nan, dtype=np.float64),
    }

    X = np.column_stack([feature_map[col] for col in feature_cols])
    return X, y


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, y_prob))
    prauc = float(average_precision_score(y_true, y_prob))
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


def run_dom_ablation() -> Dict[str, Any]:
    log.info("=" * 80)
    log.info("STARTING DOM FEATURE ABLATION STUDY (D1 -> D4)")
    log.info("=" * 80)

    df_phi, df_web = load_raw_datasets()
    log.info(f"PhiUSIIL: {len(df_phi):,} samples | Web Page Phishing: {len(df_web):,} samples")

    # Canonical PhiUSIIL Split indices (70/15/15)
    y_phi_all = (df_phi["label"].astype(int).values == 0).astype(int)
    idx_train_val, idx_test = train_test_split(
        np.arange(len(df_phi)), test_size=0.15, random_state=RANDOM_SEED, stratify=y_phi_all
    )
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=0.1765, random_state=RANDOM_SEED, stratify=y_phi_all[idx_train_val]
    )

    log.info(f"PhiUSIIL Split: Train={len(idx_train):,} | Val={len(idx_val):,} | Test={len(idx_test):,}")

    ablation_results = {}
    best_candidate_name = None
    best_cross_dataset_score = -1.0
    best_model_pkg = None

    for name, cols in FEATURE_SETS.items():
        log.info(f"\n" + "-" * 60)
        log.info(f"Evaluating Config [{name}] ({len(cols)} features)...")
        log.info(f"Features: {cols}")

        # Build feature matrices
        X_phi, y_phi = build_phiusiil_matrix(df_phi, cols)
        X_web, y_web = build_webpage_matrix(df_web, cols)

        # Train model
        t0 = time.monotonic()
        base_clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_leaf_nodes=45,
            min_samples_leaf=25,
            l2_regularization=0.05,
            random_state=RANDOM_SEED,
        )
        base_clf.fit(X_phi[idx_train], y_phi[idx_train])

        cal = CalibratedClassifierCV(base_clf, method="sigmoid", cv="prefit")
        cal.fit(X_phi[idx_val], y_phi[idx_val])
        train_time = time.monotonic() - t0

        # Evaluate on PhiUSIIL Held-Out Test Set
        y_prob_phi_test = cal.predict_proba(X_phi[idx_test])[:, 1]
        m_phi = evaluate_predictions(y_phi[idx_test], y_prob_phi_test)

        # Evaluate on External Web Page Phishing Dataset
        y_prob_web = cal.predict_proba(X_web)[:, 1]
        m_web = evaluate_predictions(y_web, y_prob_web)

        # Cross-dataset robust score: harmonic mean or balanced average
        cross_score = 0.5 * (m_phi["auc_roc"] + m_web["auc_roc"])

        log.info(f"  [{name}] Train Time: {train_time:.2f}s")
        log.info(f"  PhiUSIIL Held-Out : AUC={m_phi['auc_roc']:.4f} | F1={m_phi['f1_score']:.4f} | Recall={m_phi['recall']}% | FPR={m_phi['fpr']}%")
        log.info(f"  Web Page External : AUC={m_web['auc_roc']:.4f} | F1={m_web['f1_score']:.4f} | Recall={m_web['recall']}% | FPR={m_web['fpr']}%")
        log.info(f"  Combined Mean AUC : {cross_score:.4f}")

        ablation_results[name] = {
            "num_features": len(cols),
            "feature_names": cols,
            "train_time_seconds": round(train_time, 2),
            "phiusiil_held_out": m_phi,
            "webpage_external": m_web,
            "combined_mean_auc": round(cross_score, 4),
        }

        if cross_score > best_cross_dataset_score:
            best_cross_dataset_score = cross_score
            best_candidate_name = name
            best_model_pkg = {
                "model": cal,
                "features": cols,
                "config_name": name,
                "metrics_phiusiil": m_phi,
                "metrics_webpage": m_web,
            }

    # Summary Table
    log.info("\n" + "=" * 80)
    log.info("DOM ABLATION STUDY COMPARISON TABLE")
    log.info("=" * 80)
    log.info(f"{'Config':<25} {'Feats':<6} {'Phi Heldout AUC':<16} {'Phi FPR':<10} {'Web Ext AUC':<14} {'Web Ext FPR':<12} {'Mean AUC':<10}")
    log.info("-" * 93)
    for k, v in ablation_results.items():
        phi_m = v["phiusiil_held_out"]
        web_m = v["webpage_external"]
        log.info(f"{k:<25} {v['num_features']:<6} {phi_m['auc_roc']:<16.4f} {phi_m['fpr']:<10.2f}% {web_m['auc_roc']:<14.4f} {web_m['fpr']:<12.2f}% {v['combined_mean_auc']:<10.4f}")
    log.info("=" * 80)
    log.info(f"Best Robust Candidate: {best_candidate_name} (Combined AUC: {best_cross_dataset_score:.4f})")

    # Save results
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configurations_evaluated": ablation_results,
        "best_candidate": best_candidate_name,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"Saved DOM Ablation Results -> {RESULTS_JSON}")

    return output_data


if __name__ == "__main__":
    run_dom_ablation()
