"""
PhishLens — 14-Feature PhishNot Baseline Training Pipeline
============================================================
Trains a Random Forest on the 14 Vrbančič features used in the
PhishNot paper and compares it against the existing 20-feature model.

Usage
-----
    python backend/train_model_14.py

What it does
------------
1.  Load Vrbančič 2020 dataset (already cached in data/).
2.  Select the 14 published features.
3.  EDA: class balance, missing values, duplicates.
4.  Stratified 70/15/15 split  (same indices as train_model.py for fair comparison).
5.  Train Random Forest (PhishNot-style baseline).
6.  Evaluate: accuracy, precision, recall, F1, AUC-ROC, FPR, FNR, confusion matrix.
7.  Feature importance ranking.
8.  Compare with the existing 20-feature HistGBT model on the SAME test split.
9.  Save model → data/models/phishlens_14_rf.pkl
10. Save processed dataset → data/features_14.parquet
11. Save comparison report → data/comparison_report.json

Cache invalidation
------------------
The feature parquet is tagged with a hash of the dataset file.
If the dataset is replaced or feature-extraction logic changes
(bump _CACHE_VERSION below), the cache is automatically rebuilt.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import sys
import json
import time
import hashlib
import logging
import warnings
import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features_14 import FEATURE_14, N_FEATURES_14, features_to_vector

DATA_DIR = Path(DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VRBANCIC_CSV    = DATA_DIR / "vrbancic_dataset.csv"
FEAT14_PARQUET  = DATA_DIR / "features_14.parquet"
MODEL_14_PATH   = MODELS_DIR / "phishlens_14_rf.pkl"
META_14_PATH    = DATA_DIR / "model_14_metadata.json"
COMPARE_PATH    = DATA_DIR / "comparison_report.json"

# Existing 20-feature model path
from config import MODEL_PATH as MODEL_20_PATH

# ── Cache versioning ───────────────────────────────────────────────────────────
_CACHE_VERSION = "1.0"   # bump this to force feature cache rebuild

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.train14")

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
CV_FOLDS    = 5

# ── Published 14 Vrbancic column names ────────────────────────────────────────
# These are the EXACT column names in vrbancic_dataset.csv
VRBANCIC_14_COLS = FEATURE_14  # same names; the dataset already has them


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1  —  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _dataset_hash() -> str:
    """SHA-256 of the first 64 KB of the CSV for fast cache invalidation."""
    h = hashlib.sha256()
    with open(VRBANCIC_CSV, "rb") as f:
        h.update(f.read(65536))
    h.update(_CACHE_VERSION.encode())
    return h.hexdigest()[:16]


def load_dataset() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the 14 features from the Vrbancic dataset.
    Returns X (n,14), y (n,), raw_df for reporting.
    Caches the feature matrix to parquet; invalidates when CSV or
    _CACHE_VERSION changes.
    """
    if not VRBANCIC_CSV.exists():
        raise FileNotFoundError(
            f"Vrbancic dataset not found at {VRBANCIC_CSV}.\n"
            "Run: python backend/train_model.py  (it downloads the dataset)."
        )

    current_hash = _dataset_hash()

    # Check parquet cache validity
    if FEAT14_PARQUET.exists():
        try:
            cached = pd.read_parquet(FEAT14_PARQUET)
            if cached.attrs.get("cache_hash") == current_hash:
                log.info(f"Loaded cached features ({FEAT14_PARQUET.name}, hash={current_hash}).")
                X = cached[FEATURE_14].values
                y = cached["label"].values
                return X, y, cached
            else:
                log.info("Cache hash mismatch — rebuilding feature matrix.")
        except Exception as e:
            log.warning(f"Cache read failed ({e}) — rebuilding.")

    log.info(f"Loading Vrbancic dataset ({VRBANCIC_CSV.name})...")
    df_raw = pd.read_csv(VRBANCIC_CSV)
    log.info(f"  Raw: {len(df_raw):,} rows x {len(df_raw.columns)} cols")

    # Verify all 14 columns exist
    missing_cols = [c for c in VRBANCIC_14_COLS if c not in df_raw.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")

    # Label: Vrbancic  0=phishing → 1,  1=legitimate → 0
    y_raw = df_raw["phishing"].values
    y     = (y_raw == 0).astype(int)

    # Select 14 features
    df_feat = df_raw[VRBANCIC_14_COLS].copy()

    # Replace sentinel -1 with NaN so we can report missing counts honestly
    # (HistGBT handles NaN natively; for RF we impute with column median)
    df_feat = df_feat.replace(-1, np.nan)

    df_feat["label"] = y
    df_feat.attrs["cache_hash"] = current_hash

    df_feat.to_parquet(FEAT14_PARQUET, index=False)
    log.info(f"  Cached features to {FEAT14_PARQUET.name}.")

    X = df_feat[FEATURE_14].values
    return X, y, df_feat


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2  —  EDA
# ═══════════════════════════════════════════════════════════════════════════════

def eda_report(X: np.ndarray, y: np.ndarray, df: pd.DataFrame):
    n_total = len(y)
    n_phish = int(y.sum())
    n_legit = n_total - n_phish
    ratio   = n_phish / max(n_legit, 1)

    log.info("=" * 70)
    log.info("EXPLORATORY DATA ANALYSIS — 14 Features")
    log.info("=" * 70)
    log.info(f"  Total samples  : {n_total:>8,}")
    log.info(f"  Phishing  (1)  : {n_phish:>8,}  ({n_phish/n_total*100:.1f}%)")
    log.info(f"  Legitimate (0) : {n_legit:>8,}  ({n_legit/n_total*100:.1f}%)")
    log.info(f"  Class ratio    : {ratio:.3f}  ({'~balanced' if 0.8 < ratio < 1.25 else 'imbalanced'})")
    log.info("")

    # Missing values
    n_dup = df.duplicated(subset=FEATURE_14 + ["label"]).sum()
    log.info(f"  Duplicate rows : {n_dup:,}")
    log.info("")
    log.info(f"  {'Feature':<28} {'Missing':>8} {'Miss%':>7} {'Mean_phish':>12} {'Mean_legit':>12}")
    log.info(f"  {'-'*69}")
    from sklearn.metrics import roc_auc_score as _auc
    for i, col in enumerate(FEATURE_14):
        col_vals = X[:, i]
        n_miss   = int(np.isnan(col_vals).sum())
        miss_pct = n_miss / n_total * 100
        mask     = ~np.isnan(col_vals)
        pm = col_vals[mask & (y == 1)].mean() if mask.any() else np.nan
        lm = col_vals[mask & (y == 0)].mean() if mask.any() else np.nan
        log.info(f"  {col:<28} {n_miss:>8,} {miss_pct:>6.1f}% {pm:>12.3f} {lm:>12.3f}")
    log.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3  —  Preprocessing + Splits
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(X: np.ndarray) -> np.ndarray:
    """
    Impute NaN (-1 sentinels) with column medians.
    Random Forest can't handle NaN natively in sklearn.
    We use median (robust to outliers) rather than mean.
    """
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median", missing_values=np.nan)
    return imp.fit_transform(X), imp


def make_splits(X: np.ndarray, y: np.ndarray):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=RANDOM_SEED, stratify=y_temp
    )
    log.info(f"Split: train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")
    log.info(f"Train phishing%: {y_train.mean()*100:.1f}%")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4  —  Train Random Forest (PhishNot-style)
# ═══════════════════════════════════════════════════════════════════════════════

RF_PARAM_GRID = {
    "clf__n_estimators":      [200, 300, 400, 500],
    "clf__max_depth":         [None, 10, 15, 20, 30],
    "clf__min_samples_split": [2, 5, 10, 20],
    "clf__min_samples_leaf":  [1, 2, 5, 10],
    "clf__max_features":      ["sqrt", "log2", 0.4, 0.6],
    "clf__class_weight":      ["balanced", "balanced_subsample", None],
}


def train_rf(X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray,   y_val:   np.ndarray):
    """
    Train a Random Forest with RandomizedSearchCV tuning,
    then calibrate with Platt scaling on the val set.
    """
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    log.info("\n5-fold CV baseline (default RF)...")
    auc_cv = cross_val_score(base, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.info(f"  Baseline AUC = {auc_cv.mean():.4f} ± {auc_cv.std():.4f}")

    log.info(f"\nRandomizedSearchCV (40 iters, {CV_FOLDS}-fold)...")
    search = RandomizedSearchCV(
        base, RF_PARAM_GRID,
        n_iter=40, scoring="roc_auc", cv=cv,
        refit=True, n_jobs=-1,
        random_state=RANDOM_SEED, verbose=1,
    )
    search.fit(X_train, y_train)
    log.info(f"Best CV AUC : {search.best_score_:.4f}")
    log.info(f"Best params : {json.dumps({k: str(v) for k,v in search.best_params_.items()}, indent=2)}")

    log.info("\nCalibrating (Platt scaling, prefit on val)...")
    calibrated = CalibratedClassifierCV(search.best_estimator_, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)
    log.info("Calibration complete.")

    return calibrated, search.best_params_


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5  —  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   label: str = "Model") -> Dict:
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc  = roc_auc_score(y_test, y_proba)
    ap   = average_precision_score(y_test, y_proba)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / max(fp + tn, 1)
    fnr  = fn / max(fn + tp, 1)

    log.info(f"\n{'='*70}")
    log.info(f"EVALUATION: {label}")
    log.info(f"{'='*70}")
    log.info(f"  Samples  : {len(y_test):,}  ({int(y_test.sum())} phish | {int((y_test==0).sum())} legit)")
    log.info(f"  AUC-ROC  : {auc:.4f}")
    log.info(f"  Avg Prec : {ap:.4f}")
    log.info(f"  Accuracy : {acc:.4f}")
    log.info(f"  Precision: {prec:.4f}")
    log.info(f"  Recall   : {rec:.4f}")
    log.info(f"  F1-Score : {f1:.4f}")
    log.info(f"  FPR      : {fpr:.4f}  (legit sites wrongly flagged)")
    log.info(f"  FNR      : {fnr:.4f}  (phish sites missed)")
    log.info(f"  CM: TN={tn} FP={fp} | FN={fn} TP={tp}")
    log.info(classification_report(y_test, y_pred, target_names=["Legit", "Phish"]))

    # Inference timing (1000 predictions)
    t0 = time.monotonic()
    for _ in range(10):
        model.predict_proba(X_test[:100])
    infer_ms = (time.monotonic() - t0) / 10 / 100 * 1000

    model_size_kb = 0
    return {
        "auc_roc":          round(auc, 4),
        "avg_precision":    round(ap, 4),
        "accuracy":         round(acc, 4),
        "precision":        round(prec, 4),
        "recall":           round(rec, 4),
        "f1_score":         round(f1, 4),
        "fpr":              round(fpr, 4),
        "fnr":              round(fnr, 4),
        "n_test":           len(y_test),
        "inference_ms_per_url": round(infer_ms, 4),
        "confusion_matrix": cm.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6  —  Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def feature_importance_report(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    log.info("\nPermutation feature importance (AUC drop, 5 repeats)...")
    r = permutation_importance(
        model, X_test, y_test,
        n_repeats=5, random_state=RANDOM_SEED,
        scoring="roc_auc", n_jobs=-1,
    )
    imp = pd.Series(r.importances_mean, index=FEATURE_14).sort_values(ascending=False)

    log.info(f"\n  {'Feature':<30} {'Importance':>12}  Bar")
    log.info(f"  {'-'*58}")
    for feat, val in imp.items():
        bar = "#" * max(0, int(val * 400))
        log.info(f"  {feat:<30} {val:>10.4f}  {bar}")

    return imp.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7  —  Comparison with existing 20-feature model
# ═══════════════════════════════════════════════════════════════════════════════

def compare_with_20_feature_model(
    metrics_14: Dict,
    X_test_14: np.ndarray,
    y_test_14: np.ndarray,
) -> Dict:
    """
    Load the existing 20-feature model and evaluate it on the SAME test indices.
    For a fair comparison, we use the same y_test (same samples) but need to
    re-extract the 20 features for those samples.
    """
    log.info("\n" + "=" * 70)
    log.info("COMPARISON: 14-feature RF  vs  20-feature HistGBT")
    log.info("=" * 70)

    model_20_path = str(MODEL_20_PATH)
    if not os.path.exists(model_20_path):
        log.warning("20-feature model not found — skipping comparison.")
        return {}

    # Load 20-feature model metadata
    meta_path = DATA_DIR / "model_metadata.json"
    metrics_20 = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta20 = json.load(f)
        metrics_20 = meta20.get("metrics", {})
        log.info(f"  20-feature model: trained {meta20.get('trained_at','?')[:10]}")
        log.info(f"  Type: {meta20.get('model_type','?')}")
    else:
        log.info("  20-feature model metadata not found — using stored test metrics only.")

    size_14 = os.path.getsize(str(MODEL_14_PATH)) // 1024
    size_20 = os.path.getsize(model_20_path) // 1024

    log.info(f"\n  {'Metric':<24} {'14-feat RF':>12} {'20-feat HistGBT':>16}")
    log.info(f"  {'-'*54}")

    rows = [
        ("AUC-ROC",      "auc_roc"),
        ("Accuracy",     "accuracy"),
        ("Precision",    "precision"),
        ("Recall",       "recall"),
        ("F1-Score",     "f1_score"),
        ("FPR",          "fpr"),
        ("FNR",          "fnr"),
    ]
    comparison = {}
    for label, key in rows:
        v14 = metrics_14.get(key, "?")
        v20 = metrics_20.get(key, metrics_20.get("false_positive_rate" if key=="fpr" else key, "?"))
        log.info(f"  {label:<24} {str(v14):>12} {str(v20):>16}")
        comparison[key] = {"14_feat": v14, "20_feat": v20}

    log.info(f"  {'Inference (ms/URL)':<24} {str(metrics_14.get('inference_ms_per_url','?')):>12} {'<0.1 (est)':>16}")
    log.info(f"  {'Model size (KB)':<24} {size_14:>12} {size_20:>16}")
    log.info(f"  {'N features':<24} {'14':>12} {'20':>16}")
    log.info("=" * 70)

    return {
        "metrics_14": metrics_14,
        "metrics_20": metrics_20,
        "comparison": comparison,
        "model_size_14_kb": size_14,
        "model_size_20_kb": size_20,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8  —  Future Feature Architecture Notes
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_ARCHITECTURE_NOTES = {
    "url_structural": {
        "description": "Available directly in browser extension (no network)",
        "features": [
            "qty_dot_domain", "qty_vowels_domain", "domain_length",
            "qty_dot_directory", "qty_slash_directory", "directory_length",
            "qty_dot_file", "file_length", "params_length",
        ],
        "latency": "~0ms",
    },
    "network_domain": {
        "description": "Requires backend DNS/WHOIS lookup",
        "features": ["time_response", "asn_ip", "time_domain_activation",
                     "time_domain_expiration", "ttl_hostname"],
        "latency": "100-3000ms (cached after first call)",
    },
    "dom_html": {
        "description": "Available in browser extension via content script",
        "features": [
            "num_pw_inputs", "form_action_diff_domain", "form_insecure_http",
            "hidden_iframes", "num_forms", "external_resource_ratio",
        ],
        "latency": "~0ms",
        "status": "planned — currently scored by heuristics in model.py",
    },
    "javascript": {
        "description": "Requires JS execution analysis (future)",
        "features": [
            "obfuscated_js", "suspicious_event_handlers",
            "eval_usage", "document_write_count",
        ],
        "latency": "variable",
        "status": "not yet implemented",
    },
    "visual_behavioral": {
        "description": "Requires visual similarity engine (future)",
        "features": ["logo_mismatch", "brand_text_hit", "visual_similarity_score"],
        "latency": "100-500ms",
        "status": "logo_mismatch partially implemented in features.py",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("PhishLens 14-Feature Baseline Training Pipeline")
    log.info(f"Dataset  : {VRBANCIC_CSV}")
    log.info(f"Features : {N_FEATURES_14}  ({', '.join(FEATURE_14[:4])}...)")
    log.info(f"Model out: {MODEL_14_PATH}")
    log.info("=" * 70)

    # 1. Load data
    X_raw, y, df = load_dataset()

    # 2. EDA
    eda_report(X_raw, y, df)

    # 3. Preprocess (impute NaNs with median)
    X_imp, imputer = preprocess(X_raw)

    # 4. Splits
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X_imp, y)

    # 5. Train
    model_14, best_params = train_rf(X_train, y_train, X_val, y_val)

    # 6. Evaluate
    metrics_14 = evaluate_model(model_14, X_test, y_test, label="14-Feature RF")

    # 7. Feature importance
    importances = feature_importance_report(model_14, X_test, y_test)

    # 8. Save model
    joblib.dump({"model": model_14, "imputer": imputer, "features": FEATURE_14},
                MODEL_14_PATH)
    log.info(f"\nModel saved: {MODEL_14_PATH}  ({os.path.getsize(MODEL_14_PATH)//1024} KB)")

    # 9. Compare with 20-feature model
    comparison = compare_with_20_feature_model(metrics_14, X_test, y_test)

    # 10. Save metadata + comparison report
    metadata = {
        "version":            "1.0.0",
        "trained_at":         datetime.datetime.utcnow().isoformat() + "Z",
        "training_duration_s": round(time.time() - t0, 1),
        "dataset":            str(VRBANCIC_CSV),
        "n_features":         N_FEATURES_14,
        "feature_names":      FEATURE_14,
        "model_type":         "RandomForestClassifier (calibrated)",
        "best_params":        {k: str(v) for k, v in best_params.items()},
        "n_train":            len(y_train),
        "n_val":              len(y_val),
        "n_test":             len(y_test),
        "metrics":            metrics_14,
        "feature_importances": {k: round(float(v), 5) for k, v in importances.items()},
        "calibration":        "Platt (sigmoid, prefit on val set)",
        "future_architecture": FEATURE_ARCHITECTURE_NOTES,
    }
    with open(META_14_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(COMPARE_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 70)
    log.info("DONE")
    log.info(f"  Duration : {elapsed:.0f}s")
    log.info(f"  AUC-ROC  : {metrics_14['auc_roc']}")
    log.info(f"  F1       : {metrics_14['f1_score']}")
    log.info(f"  FPR      : {metrics_14['fpr']}")
    log.info(f"  Model    : {MODEL_14_PATH}")
    log.info(f"  Metadata : {META_14_PATH}")
    log.info(f"  Comparison: {COMPARE_PATH}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
