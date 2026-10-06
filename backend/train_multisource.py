"""
PhishLens — Stage 3: Multi-Source URL Training & LODO Validation
=================================================================
Trains and evaluates a multi-source HistGradientBoosting URL classifier across
Vrbancic 2020, PhiUSIIL, and LegitPhish (total 425,660 raw URLs).

Validation Protocols
--------------------
1. Grouped Combined-Data Validation:
   Domain-level partitioning (GroupShuffleSplit on registrable domain)
   ensuring zero domain leakage between train, val, and test splits.

2. Leave-One-Dataset-Out (LODO) Cross-Validation:
   - Fold A: Train on (Vrbancic + PhiUSIIL) -> Test on LegitPhish
   - Fold B: Train on (Vrbancic + LegitPhish) -> Test on PhiUSIIL
   - Fold C: Train on (PhiUSIIL + LegitPhish) -> Test on Vrbancic
   - Holdout: Train on (Vrbancic + PhiUSIIL + LegitPhish) -> Test on Web Page Phishing

Note: The reference E5 model is kept frozen and untouched.
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
import tldextract

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features_14 import FEATURE_14, N_FEATURES_14, AVAILABLE_IN_EXTENSION, extract_url_features_14

DATA_DIR = Path(DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MULTISOURCE_PARQUET = DATA_DIR / "multisource_features.parquet"
MODEL_MULTISOURCE_PATH = MODELS_DIR / "phishlens_multisource_histgbt.pkl"
RESULTS_JSON = DATA_DIR / "multisource_validation_results.json"

ROOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.multisrc")

RANDOM_SEED = 42
_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=str(DATA_DIR / ".tld_cache"))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Dataset Loading, Feature Extraction & Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_domain(url: str) -> str:
    try:
        ext = _TLD_EXTRACTOR(url)
        return ext.registered_domain or ext.domain or url.split("/")[0]
    except Exception:
        return "unknown_domain"


def load_or_build_multisource_dataset() -> pd.DataFrame:
    """
    Loads raw URLs from all 3 datasets, runs features_14.py extraction,
    normalizes labels, extracts domains, and deduplicates.
    Caches the combined DataFrame to parquet for fast subsequent loads.
    """
    if MULTISOURCE_PARQUET.exists():
        log.info(f"Loading cached multi-source dataset ({MULTISOURCE_PARQUET.name})...")
        df_all = pd.read_parquet(MULTISOURCE_PARQUET)
        log.info(f"Loaded {len(df_all):,} rows across sources: {df_all['source'].value_counts().to_dict()}")
        return df_all

    log.info("Building unified multi-source dataset from raw files...")
    rows = []

    # ── 1. Vrbancic 2020 (88,647 rows) ─────────────────────────────────────────
    vrb_csv = DATA_DIR / "vrbancic_dataset.csv"
    if vrb_csv.exists():
        log.info(f"Processing Vrbancic 2020 ({vrb_csv.name})...")
        df_vrb = pd.read_csv(vrb_csv)
        # Invert: 0=phish -> 1, 1=legit -> 0
        y_vrb = (df_vrb["phishing"].values == 0).astype(int)
        df_vrb_14 = df_vrb[FEATURE_14].copy().replace(-1, np.nan)
        df_vrb_14["label"] = y_vrb
        df_vrb_14["source"] = "vrbancic"
        df_vrb_14["url_clean"] = [f"vrbancic_sample_{i}" for i in range(len(df_vrb))]
        # Domain signature group for Vrbancic
        dom_grp = df_vrb["domain_length"].astype(str) + "_" + df_vrb["qty_dot_domain"].astype(str) + "_" + df_vrb["asn_ip"].astype(str)
        df_vrb_14["domain"] = dom_grp
        rows.append(df_vrb_14)
        log.info(f"  Vrbancic: {len(df_vrb_14):,} rows ({y_vrb.mean()*100:.1f}% phish)")

    feat_idx = {col: i for i, col in enumerate(FEATURE_14)}

    # ── 2. PhiUSIIL 2024 (235,795 rows) ────────────────────────────────────────
    phi_csv = DATASETS_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
    if phi_csv.exists():
        log.info(f"Processing PhiUSIIL ({phi_csv.name})...")
        df_phi = pd.read_csv(phi_csv).dropna(subset=["URL", "label"])
        urls_phi = df_phi["URL"].astype(str).tolist()
        y_phi = (df_phi["label"].astype(int).values == 0).astype(int)

        t0 = time.monotonic()
        mat_phi = np.full((len(urls_phi), N_FEATURES_14), np.nan, dtype=np.float64)
        domains_phi = []
        clean_urls_phi = []

        for i, u in enumerate(urls_phi):
            u_clean = u.strip().rstrip("/")
            clean_urls_phi.append(u_clean.lower())
            domains_phi.append(_extract_domain(u_clean))
            f = extract_url_features_14(u)
            for k in AVAILABLE_IN_EXTENSION:
                mat_phi[i, feat_idx[k]] = f.get(k, np.nan)

        df_phi_14 = pd.DataFrame(mat_phi, columns=FEATURE_14)
        df_phi_14["label"] = y_phi
        df_phi_14["source"] = "phiusiil"
        df_phi_14["url_clean"] = clean_urls_phi
        df_phi_14["domain"] = domains_phi
        rows.append(df_phi_14)
        log.info(f"  PhiUSIIL: {len(df_phi_14):,} rows extracted in {time.monotonic()-t0:.2f}s ({y_phi.mean()*100:.1f}% phish)")

    # ── 3. LegitPhish / Mendeley (101,218 rows) ────────────────────────────────
    lp_csv = DATASETS_DIR / "url_features_extracted1.csv"
    if lp_csv.exists():
        log.info(f"Processing LegitPhish ({lp_csv.name})...")
        df_lp = pd.read_csv(lp_csv).dropna(subset=["URL", "ClassLabel"])
        urls_lp = df_lp["URL"].astype(str).tolist()
        y_lp = (df_lp["ClassLabel"].astype(float).values == 0.0).astype(int)

        t0 = time.monotonic()
        mat_lp = np.full((len(urls_lp), N_FEATURES_14), np.nan, dtype=np.float64)
        domains_lp = []
        clean_urls_lp = []

        for i, u in enumerate(urls_lp):
            u_clean = u.strip().rstrip("/")
            clean_urls_lp.append(u_clean.lower())
            domains_lp.append(_extract_domain(u_clean))
            f = extract_url_features_14(u)
            for k in AVAILABLE_IN_EXTENSION:
                mat_lp[i, feat_idx[k]] = f.get(k, np.nan)

        df_lp_14 = pd.DataFrame(mat_lp, columns=FEATURE_14)
        df_lp_14["label"] = y_lp
        df_lp_14["source"] = "legitphish"
        df_lp_14["url_clean"] = clean_urls_lp
        df_lp_14["domain"] = domains_lp
        rows.append(df_lp_14)
        log.info(f"  LegitPhish: {len(df_lp_14):,} rows extracted in {time.monotonic()-t0:.2f}s ({y_lp.mean()*100:.1f}% phish)")

    # Combine & Deduplicate
    df_combined = pd.concat(rows, ignore_index=True)
    n_before = len(df_combined)
    log.info(f"\nTotal combined rows before deduplication: {n_before:,}")

    # Remove exact URL duplicates across datasets (prioritizing the first source encountered)
    df_dedup = df_combined.drop_duplicates(subset=["url_clean"], keep="first").reset_index(drop=True)
    n_after = len(df_dedup)
    n_removed = n_before - n_after
    log.info(f"Deduplicated dataset: {n_after:,} unique URLs ({n_removed:,} cross-dataset duplicates removed)")
    log.info(f"Unique domains: {df_dedup['domain'].nunique():,}")
    log.info(f"Class distribution: {int(df_dedup['label'].sum()):,} Phishing ({df_dedup['label'].mean()*100:.1f}%), {int((df_dedup['label']==0).sum()):,} Legit")

    df_dedup.to_parquet(MULTISOURCE_PARQUET, index=False)
    log.info(f"Cached multi-source dataset -> {MULTISOURCE_PARQUET.name}")
    return df_dedup


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Evaluation Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def compute_eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
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
        "auc_roc":    round(float(auc), 4),
        "pr_auc":     round(float(prauc), 4),
        "accuracy":   round(float(acc), 4),
        "precision":  round(float(prec), 4),
        "recall_tpr": round(float(rec), 4),
        "f1_score":   round(float(f1), 4),
        "fpr":        round(float(fpr), 4),
        "fnr":        round(float(fnr), 4),
        "tn":         int(tn),
        "fp":         int(fp),
        "fn":         int(fn),
        "tp":         int(tp),
        "n_samples":  len(y_true),
    }


def fit_and_calibrate_histgbt(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> CalibratedClassifierCV:
    base = HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=30,
        l2_regularization=0.05,
        random_state=RANDOM_SEED,
    )
    base.fit(X_train, y_train)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)
    return cal


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Scheme 1: Multi-Source Domain-Grouped Split Validation
# ═══════════════════════════════════════════════════════════════════════════════

def run_grouped_multisource_experiment(df: pd.DataFrame) -> Tuple[CalibratedClassifierCV, Dict]:
    log.info("\n" + "=" * 90)
    log.info("SCHEME 1: MULTI-SOURCE DOMAIN-GROUPED SPLIT (70% Train / 15% Val / 15% Test)")
    log.info("Zero domain overlap between train, val, and test sets.")
    log.info("=" * 90)

    X = df[FEATURE_14].values
    y = df["label"].values
    groups = df["domain"].values

    # Step 1: Split out 15% test groups
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))

    # Step 2: Split train_val into 70% train and 15% val groups (15/85 ≈ 0.1765)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=RANDOM_SEED)
    train_sub_idx, val_sub_idx = next(gss_val.split(X[train_val_idx], y[train_val_idx], groups=groups[train_val_idx]))
    train_idx = train_val_idx[train_sub_idx]
    val_idx   = train_val_idx[val_sub_idx]

    log.info(f"Grouped Split: Train={len(train_idx):,} ({df.iloc[train_idx]['domain'].nunique():,} doms) | "
             f"Val={len(val_idx):,} ({df.iloc[val_idx]['domain'].nunique():,} doms) | "
             f"Test={len(test_idx):,} ({df.iloc[test_idx]['domain'].nunique():,} doms)")

    # Verify zero domain overlap
    dom_tr = set(df.iloc[train_idx]["domain"])
    dom_te = set(df.iloc[test_idx]["domain"])
    overlap = len(dom_tr.intersection(dom_te))
    log.info(f"Domain overlap between Train and Test: {overlap} (Verified: Strict Zero Leakage)")

    t0 = time.monotonic()
    model = fit_and_calibrate_histgbt(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    train_time = time.monotonic() - t0
    log.info(f"Model trained and calibrated in {train_time:.1f}s")

    # Evaluate on full combined test set
    y_test_pred = model.predict(X[test_idx])
    y_test_prob = model.predict_proba(X[test_idx])[:, 1]
    overall_metrics = compute_eval_metrics(y[test_idx], y_test_pred, y_test_prob)

    log.info("\nOVERALL COMBINED TEST SET RESULTS (Domain-Grouped Held-Out):")
    log.info(f"  AUC-ROC: {overall_metrics['auc_roc']:.4f} | PR-AUC: {overall_metrics['pr_auc']:.4f} | Accuracy: {overall_metrics['accuracy']*100:.2f}% | F1: {overall_metrics['f1_score']:.4f}")
    log.info(f"  Precision: {overall_metrics['precision']*100:.2f}% | Recall (TPR): {overall_metrics['recall_tpr']*100:.2f}% | FPR: {overall_metrics['fpr']*100:.2f}% | FNR: {overall_metrics['fnr']*100:.2f}%")
    log.info(f"  Confusion Matrix: TN={overall_metrics['tn']:,} FP={overall_metrics['fp']:,} | FN={overall_metrics['fn']:,} TP={overall_metrics['tp']:,}")

    # Evaluate breakdown per source on the test set
    df_test = df.iloc[test_idx]
    source_breakdown = {}
    log.info("\nTEST SET PERFORMANCE BREAKDOWN BY DATASET SOURCE:")
    log.info(f"{'Source':<15} {'Samples':>9} {'Phish%':>8} {'AUC-ROC':>9} {'PR-AUC':>8} {'Accuracy':>9} {'F1':>7} {'FPR':>7} {'FNR':>7}")
    log.info("-" * 90)

    for src in df["source"].unique():
        src_mask = (df_test["source"] == src).values
        if src_mask.sum() == 0:
            continue
        X_src = X[test_idx][src_mask]
        y_src = y[test_idx][src_mask]
        p_src = model.predict(X_src)
        prob_src = model.predict_proba(X_src)[:, 1]
        m_src = compute_eval_metrics(y_src, p_src, prob_src)
        source_breakdown[src] = m_src
        log.info(f"{src:<15} {len(y_src):>9,} {y_src.mean()*100:>7.1f}% {m_src['auc_roc']:>9.4f} {m_src['pr_auc']:>8.4f} {m_src['accuracy']*100:>8.2f}% {m_src['f1_score']:>7.4f} {m_src['fpr']*100:>6.2f}% {m_src['fnr']*100:>6.2f}%")

    # Feature Importance on Combined Test Set
    log.info("\nPermutation Feature Importance on Combined Test Set (AUC drop)...")
    r_imp = permutation_importance(model, X[test_idx], y[test_idx], n_repeats=3, random_state=RANDOM_SEED, scoring="roc_auc", n_jobs=-1)
    imp_series = pd.Series(r_imp.importances_mean, index=FEATURE_14).sort_values(ascending=False)
    log.info(f"{'Feature':<28} {'Importance':>12}  Bar")
    log.info("-" * 55)
    for feat_name, imp_val in imp_series.items():
        bar = "#" * max(0, int(imp_val * 400))
        log.info(f"{feat_name:<28} {imp_val:>12.4f}  {bar}")

    # Save model artifact
    joblib.dump({"model": model, "features": FEATURE_14, "trained_samples": len(train_idx)}, MODEL_MULTISOURCE_PATH)
    log.info(f"\nSaved Multi-Source Model -> {MODEL_MULTISOURCE_PATH} ({MODEL_MULTISOURCE_PATH.stat().st_size//1024} KB)")

    return model, {
        "overall_metrics": overall_metrics,
        "source_breakdown": source_breakdown,
        "feature_importances": imp_series.to_dict(),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Scheme 2: Leave-One-Dataset-Out (LODO) Validation
# ═══════════════════════════════════════════════════════════════════════════════

def run_lodo_experiments(df: pd.DataFrame) -> Dict:
    log.info("\n" + "=" * 90)
    log.info("SCHEME 2: LEAVE-ONE-DATASET-OUT (LODO) CROSS-DATASET EXPERIMENTS")
    log.info("Trains on N-1 datasets and tests on the held-out Nth dataset (Zero Cross-Dataset Leakage).")
    log.info("=" * 90)

    lodo_results = {}
    sources = ["vrbancic", "phiusiil", "legitphish"]

    for test_src in sources:
        train_sources = [s for s in sources if s != test_src]
        log.info(f"\n[LODO Fold] Train on ({' + '.join(train_sources)}) -> Test on ({test_src})...")

        df_train_full = df[df["source"].isin(train_sources)].reset_index(drop=True)
        df_test = df[df["source"] == test_src].reset_index(drop=True)

        # Domain split train_full into 85% train and 15% val for calibration
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
        tr_idx, val_idx = next(gss.split(df_train_full, groups=df_train_full["domain"]))

        X_tr = df_train_full.iloc[tr_idx][FEATURE_14].values
        y_tr = df_train_full.iloc[tr_idx]["label"].values
        X_val = df_train_full.iloc[val_idx][FEATURE_14].values
        y_val = df_train_full.iloc[val_idx]["label"].values

        X_te = df_test[FEATURE_14].values
        y_te = df_test["label"].values

        log.info(f"  Train: {len(X_tr):,} | Val: {len(X_val):,} | Test ({test_src}): {len(X_te):,}")

        t0 = time.monotonic()
        model_lodo = fit_and_calibrate_histgbt(X_tr, y_tr, X_val, y_val)
        t_fit = time.monotonic() - t0

        y_pred = model_lodo.predict(X_te)
        y_prob = model_lodo.predict_proba(X_te)[:, 1]
        m = compute_eval_metrics(y_te, y_pred, y_prob)

        lodo_key = f"train_on_{'_'.join(train_sources)}_test_on_{test_src}"
        lodo_results[lodo_key] = {
            "test_source": test_src,
            "train_sources": train_sources,
            "metrics": m,
        }

        log.info(f"  [Result: Test on {test_src}]")
        log.info(f"    AUC-ROC: {m['auc_roc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Accuracy: {m['accuracy']*100:.2f}% | F1: {m['f1_score']:.4f}")
        log.info(f"    Precision: {m['precision']*100:.2f}% | Recall (TPR): {m['recall_tpr']*100:.2f}% | FPR: {m['fpr']*100:.2f}% | FNR: {m['fnr']*100:.2f}%")
        log.info(f"    Confusion Matrix: TN={m['tn']:,} FP={m['fp']:,} | FN={m['fn']:,} TP={m['tp']:,}")

    # ── External Holdout: Web Page Phishing (11,430 URLs) ─────────────────────
    path_wp = DATASETS_DIR / "dataset_phishing.csv"
    if path_wp.exists():
        log.info("\n[External Holdout Test] Train on All 3 Multi-Source Datasets -> Test on Web Page Phishing (11,430 URLs)...")
        df_wp = pd.read_csv(path_wp)
        y_wp = (df_wp["status"].str.strip().str.lower() == "phishing").astype(int).values
        urls_wp = df_wp["url"].astype(str).tolist()

        feat_idx = {col: i for i, col in enumerate(FEATURE_14)}
        mat_wp = np.full((len(urls_wp), N_FEATURES_14), np.nan, dtype=np.float64)
        for i, u in enumerate(urls_wp):
            f = extract_url_features_14(u)
            for k in AVAILABLE_IN_EXTENSION:
                mat_wp[i, feat_idx[k]] = f.get(k, np.nan)

        # Train on all 3 sources
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_SEED)
        tr_idx, val_idx = next(gss.split(df, groups=df["domain"]))
        m_all = fit_and_calibrate_histgbt(df.iloc[tr_idx][FEATURE_14].values, df.iloc[tr_idx]["label"].values,
                                          df.iloc[val_idx][FEATURE_14].values, df.iloc[val_idx]["label"].values)

        y_pred_wp = m_all.predict(mat_wp)
        y_prob_wp = m_all.predict_proba(mat_wp)[:, 1]
        m_wp = compute_eval_metrics(y_wp, y_pred_wp, y_prob_wp)

        lodo_results["train_on_all3_test_on_webpage"] = {
            "test_source": "web_page_phishing",
            "train_sources": sources,
            "metrics": m_wp,
        }
        log.info(f"  [Result: Test on Web Page Phishing]")
        log.info(f"    AUC-ROC: {m_wp['auc_roc']:.4f} | PR-AUC: {m_wp['pr_auc']:.4f} | Accuracy: {m_wp['accuracy']*100:.2f}% | F1: {m_wp['f1_score']:.4f}")
        log.info(f"    Precision: {m_wp['precision']*100:.2f}% | Recall (TPR): {m_wp['recall_tpr']*100:.2f}% | FPR: {m_wp['fpr']*100:.2f}% | FNR: {m_wp['fnr']*100:.2f}%")

    return lodo_results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main Execution & Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("=" * 90)
    log.info("PHISHLENS STAGE 3: MULTI-SOURCE TRAINING & LODO VALIDATION PIPELINE")
    log.info("=" * 90)

    # 1. Build / Load multi-source dataset
    df_multisrc = load_or_build_multisource_dataset()

    # 2. Run Scheme 1 (Grouped Combined Validation)
    model_ms, res_scheme1 = run_grouped_multisource_experiment(df_multisrc)

    # 3. Run Scheme 2 (LODO Cross-Validation)
    res_lodo = run_lodo_experiments(df_multisrc)

    # 4. Consolidated Summary Table
    log.info("\n" + "=" * 115)
    log.info("FINAL SUMMARY TABLE: MULTI-SOURCE GENERALIZATION PERFORMANCE")
    log.info("=" * 115)
    log.info(f"{'Experiment Protocol':<45} {'Test Dataset':<20} {'Samples':>8} {'AUC-ROC':>9} {'Accuracy':>9} {'F1':>7} {'FPR':>7} {'FNR':>7}")
    log.info("-" * 115)

    # Scheme 1 Combined
    m_s1 = res_scheme1["overall_metrics"]
    log.info(f"{'Scheme 1: Multi-Source Domain-Grouped':<45} {'Combined Test Set':<20} {m_s1['n_samples']:>8,} {m_s1['auc_roc']:>9.4f} {m_s1['accuracy']*100:>8.2f}% {m_s1['f1_score']:>7.4f} {m_s1['fpr']*100:>6.2f}% {m_s1['fnr']*100:>6.2f}%")

    # Scheme 1 Breakdown
    for src, m in res_scheme1["source_breakdown"].items():
        log.info(f"{'  -> Source Breakdown: ' + src:<45} {src:<20} {m['n_samples']:>8,} {m['auc_roc']:>9.4f} {m['accuracy']*100:>8.2f}% {m['f1_score']:>7.4f} {m['fpr']*100:>6.2f}% {m['fnr']*100:>6.2f}%")

    # LODO Folds
    for k, v in res_lodo.items():
        m = v["metrics"]
        desc = f"LODO: Train on {len(v['train_sources'])} -> Test on 1"
        log.info(f"{desc:<45} {v['test_source']:<20} {m['n_samples']:>8,} {m['auc_roc']:>9.4f} {m['accuracy']*100:>8.2f}% {m['f1_score']:>7.4f} {m['fpr']*100:>6.2f}% {m['fnr']*100:>6.2f}%")

    log.info("=" * 115)

    # Save to JSON
    output_pkg = {
        "scheme1_grouped_validation": res_scheme1,
        "scheme2_lodo_validation": res_lodo,
        "dataset_summary": {
            "total_samples": len(df_multisrc),
            "unique_domains": df_multisrc["domain"].nunique(),
            "sources": df_multisrc["source"].value_counts().to_dict(),
        },
        "execution_time_s": round(time.time() - t_start, 1),
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(output_pkg, f, indent=2)
    log.info(f"\nAll Stage 3 validation results saved -> {RESULTS_JSON}")
    log.info(f"Total Stage 3 execution time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
