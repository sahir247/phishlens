"""
PhishLens — Stage 4: DOM Content Classifier Training Pipeline
==============================================================
Trains and calibrates a dedicated HistGradientBoosting model
on webpage source code, HTML, and DOM structure signals.

Features Used (12 Browser-Observable DOM Signals)
-------------------------------------------------
1.  has_password_field
2.  has_login_form
3.  has_external_form_submit
4.  has_submit_button
5.  has_iframe
6.  num_iframes
7.  num_scripts
8.  ratio_external_refs
9.  num_empty_refs
10. domain_title_match_score
11. has_obfuscation
12. line_of_code

Training Source
---------------
- PhiUSIIL Webpage Dataset (235,795 rows with webpage source code features)
- Web Page Phishing Dataset (11,430 rows with DOM/form/iframe features)

Output Model
------------
backend/data/models/phishlens_dom_histgbt.pkl
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features_dom import DOM_FEATURE_COLS, N_DOM_FEATURES

DATA_DIR = Path(DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DOM_PATH = MODELS_DIR / "phishlens_dom_histgbt.pkl"
RESULTS_DOM_JSON = DATA_DIR / "dom_model_results.json"

ROOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.traindom")

RANDOM_SEED = 42


def load_dom_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the unified DOM feature matrix from PhiUSIIL and Web Page Phishing.
    Returns X (n, 12) and y (n,).
    """
    log.info("Loading DOM/HTML training datasets...")
    rows_X = []
    rows_y = []

    # 1. PhiUSIIL (235,795 rows)
    phi_path = DATASETS_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if phi_path.exists():
        log.info(f"Mapping PhiUSIIL ({phi_path.name})...")
        df_phi = pd.read_csv(phi_path).dropna(subset=["label"])
        y_phi = (df_phi["label"].astype(int).values == 0).astype(int)

        n_phi = len(df_phi)
        mat_phi = np.zeros((n_phi, N_DOM_FEATURES), dtype=np.float64)

        # Mapping to 10 canonical DOM_FEATURE_COLS (D3 Robust Configuration)
        mat_phi[:, 0] = df_phi["HasPasswordField"].fillna(0).values.astype(float)
        mat_phi[:, 1] = np.maximum(df_phi["HasPasswordField"].fillna(0).values,
                                   (df_phi["HasSubmitButton"].fillna(0).values > 0).astype(float))
        mat_phi[:, 2] = df_phi["HasExternalFormSubmit"].fillna(0).values.astype(float)
        mat_phi[:, 3] = df_phi["HasSubmitButton"].fillna(0).values.astype(float)
        mat_phi[:, 4] = (df_phi["NoOfiFrame"].fillna(0).values > 0).astype(float)
        mat_phi[:, 5] = df_phi["NoOfiFrame"].fillna(0).values.astype(float)

        # ratio_external_refs: ext / (self + ext + empty + 1)
        ext = df_phi["NoOfExternalRef"].fillna(0).values.astype(float)
        self_ref = df_phi["NoOfSelfRef"].fillna(0).values.astype(float)
        emp = df_phi["NoOfEmptyRef"].fillna(0).values.astype(float)
        tot = ext + self_ref + emp + 1.0
        mat_phi[:, 6] = (ext / tot).clip(0, 1)

        mat_phi[:, 7] = emp
        mat_phi[:, 8] = df_phi["DomainTitleMatchScore"].fillna(50).values.astype(float)
        mat_phi[:, 9] = df_phi["HasObfuscation"].fillna(0).values.astype(float)

        rows_X.append(mat_phi)
        rows_y.append(y_phi)
        log.info(f"  PhiUSIIL: {n_phi:,} DOM samples mapped (10 robust features).")


    X_all = np.vstack(rows_X)
    y_all = np.concatenate(rows_y)
    log.info(f"Total DOM dataset: {len(X_all):,} samples ({int(y_all.sum()):,} Phish / {int((y_all==0).sum()):,} Legit)")
    return X_all, y_all


def train_dom_classifier() -> Dict:
    X, y = load_dom_training_data()

    # Stratified 70/15/15 split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=RANDOM_SEED, stratify=y_train_val
    )

    log.info(f"Split: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

    t0 = time.monotonic()
    base_clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=45,
        min_samples_leaf=25,
        l2_regularization=0.05,
        random_state=RANDOM_SEED,
    )
    base_clf.fit(X_train, y_train)

    cal = CalibratedClassifierCV(base_clf, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)
    train_duration = time.monotonic() - t0
    log.info(f"DOM model trained and calibrated in {train_duration:.2f}s")

    # Evaluation on held-out test set
    y_pred = cal.predict(X_test)
    y_prob = cal.predict_proba(X_test)[:, 1]

    auc   = roc_auc_score(y_test, y_prob)
    prauc = average_precision_score(y_test, y_prob)
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    cm    = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr   = fp / max(fp + tn, 1)
    fnr   = fn / max(fn + tp, 1)

    log.info("\n" + "=" * 80)
    log.info("STAGE 4: DOM CONTENT CLASSIFIER EVALUATION RESULTS")
    log.info("=" * 80)
    log.info(f"  AUC-ROC:   {auc:.4f}  (PR-AUC: {prauc:.4f})")
    log.info(f"  Accuracy:  {acc*100:.2f}%")
    log.info(f"  F1-Score:  {f1:.4f}")
    log.info(f"  Precision: {prec*100:.2f}%")
    log.info(f"  Recall:    {rec*100:.2f}%")
    log.info(f"  FPR:       {fpr*100:.2f}%  (False alarms on legit pages)")
    log.info(f"  FNR:       {fnr*100:.2f}%  (Missed phishing attacks)")
    log.info(f"  Confusion Matrix: TN={tn:,} FP={fp:,} | FN={fn:,} TP={tp:,}")

    # Feature Importance
    log.info("\nPermutation Feature Importance for DOM Model (AUC drop)...")
    r_imp = permutation_importance(cal, X_test, y_test, n_repeats=3, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1)
    imp_series = pd.Series(r_imp.importances_mean, index=DOM_FEATURE_COLS).sort_values(ascending=False)
    log.info(f"{'Feature':<28} {'Importance':>12}  Bar")
    log.info("-" * 55)
    for feat_name, imp_val in imp_series.items():
        bar = "#" * max(0, int(imp_val * 400))
        log.info(f"{feat_name:<28} {imp_val:>12.4f}  {bar}")

    # Save model
    joblib.dump({"model": cal, "features": DOM_FEATURE_COLS, "train_samples": len(X_train)}, MODEL_DOM_PATH)
    log.info(f"\nSaved DOM Model -> {MODEL_DOM_PATH} ({MODEL_DOM_PATH.stat().st_size//1024} KB)")

    results = {
        "auc_roc": round(float(auc), 4),
        "pr_auc": round(float(prauc), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall_tpr": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "feature_importances": imp_series.to_dict(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_duration_s": round(train_duration, 2),
    }

    with open(RESULTS_DOM_JSON, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    train_dom_classifier()
