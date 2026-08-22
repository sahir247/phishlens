"""
PhishLens ML Training Pipeline  —  Professional Grade  v2.0
=============================================================
Fixes v1.0 data leakage (scheme bias from bare-domain datasets).

Dataset
-------
Primary : Vrbancic et al. (2020)  — 88k real URLs, 111 pre-extracted features.
          DOI: 10.17632/h3cgnj8hft.1  |  No API key required.
          Features extracted from ACTUAL URLs with full paths, subdomains, params.
          label: 0 = phishing, 1 = legitimate  (inverted — we handle this)

Feature Mapping  (Vrbancic col → our FEATURE_COL)
--------------------------------------------------
qty_hyphen_url        → num_dashes
qty_underline_url     → num_underscores
qty_at_url            → num_at
qty_percent_url       → num_percent
qty_slash_url         → num_slashes
length_url            → url_len
domain_in_ip          → has_ip
domain_length         → domain_len
tls_ssl_certificate   → has_https    (1=valid TLS, 0=none)
email_in_url          → num_at (supplement)
url_shortened         → suspicious_kw (shortened URL is suspicious)
time_domain_activation→ domain_age_days (converted: days since activation)
qty_dot_domain        → subdomain_count (dots in domain ≈ subdomain depth)
directory_length      → path_len
params_length         → query_len

HTML-derived features (num_pw_inputs, form_action_diff_domain,
form_insecure_http, logo_mismatch, hidden_iframes) are 0 for all samples
— these are scored by heuristics in model.py, not the ML classifier.

Pipeline
--------
1.  Download & cache Vrbancic CSV (88k rows, ~11 MB)
2.  Map 15 Vrbancic cols → our 20 FEATURE_COLS
3.  EDA: class balance, feature distributions, correlation
4.  Stratified split  70 / 15 / 15
5.  Model zoo 5-fold CV  (LR, RF, GBT, HistGBT)
6.  Hyperparameter tuning on winner  (RandomizedSearchCV)
7.  Platt calibration on val set
8.  Test set evaluation  (AUC, F1, FPR)
9.  Permutation feature importance
10. Save model + metadata JSON
"""

import os
# Suppress joblib "could not find physical cores" warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
import sys
import json
import time
import logging
import warnings
import datetime
import io
import urllib.request
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import MODEL_PATH, DATA_DIR
from ml_model import FEATURE_COLS, N_FEATURES

DATA_DIR = Path(DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)

VRBANCIC_URL       = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_full.csv"
RAW_CSV_CACHE      = DATA_DIR / "vrbancic_dataset.csv"
FEAT_PARQUET_CACHE = DATA_DIR / "features_v2.parquet"
MODEL_META_FILE    = DATA_DIR / "model_metadata.json"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.train")

# ── Hyperparameters ───────────────────────────────────────────────────────────
RANDOM_SEED   = 42
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15
CV_FOLDS      = 5
N_ITER_SEARCH = 50
MAX_SAMPLES   = 80_000   # cap for speed; dataset has 88k


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1  —  Data Acquisition
# ═══════════════════════════════════════════════════════════════════════════════

def download_vrbancic() -> pd.DataFrame:
    """Download and cache the Vrbancic et al. (2020) phishing dataset."""
    if RAW_CSV_CACHE.exists():
        age_h = (time.time() - RAW_CSV_CACHE.stat().st_mtime) / 3600
        if age_h < 168:    # reuse for 1 week
            log.info(f"Loading cached dataset ({RAW_CSV_CACHE.name}, {age_h:.0f}h old).")
            return pd.read_csv(RAW_CSV_CACHE)

    log.info(f"Downloading Vrbancic dataset (~11 MB) from GitHub...")
    req = urllib.request.Request(VRBANCIC_URL, headers={"User-Agent": "PhishLens-Trainer/2.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8", errors="ignore")

    df = pd.read_csv(io.StringIO(raw))
    df.to_csv(RAW_CSV_CACHE, index=False)
    log.info(f"Downloaded {len(df):,} rows × {len(df.columns)} cols. Cached.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2  —  Feature Mapping  (Vrbancic → PhishLens FEATURE_COLS)
# ═══════════════════════════════════════════════════════════════════════════════

def map_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map Vrbancic columns to our FEATURE_COLS vector.
    Returns X (n_samples, N_FEATURES) and y (n_samples,).
    """
    if FEAT_PARQUET_CACHE.exists():
        age_h = (time.time() - FEAT_PARQUET_CACHE.stat().st_mtime) / 3600
        if age_h < 168:
            log.info(f"Loading cached feature matrix ({FEAT_PARQUET_CACHE.name}).")
            feat_df = pd.read_parquet(FEAT_PARQUET_CACHE)
            return feat_df[FEATURE_COLS].values, feat_df["label"].values

    log.info("Mapping Vrbancic features to PhishLens feature space...")

    # Label: Vrbancic uses 1=legit, 0=phishing  →  invert to 1=phishing, 0=benign
    y_raw = df["phishing"].values
    y = (y_raw == 0).astype(int)   # 0(phishing) → 1, 1(legit) → 0

    n = len(df)
    feat_matrix = np.zeros((n, N_FEATURES), dtype=np.float64)
    feat_idx = {col: i for i, col in enumerate(FEATURE_COLS)}

    # ── Direct mappings  (our_feature_col → vrbancic_col) ─────────────────────
    # Only features that actually exist in FEATURE_COLS:
    # has_ip, has_punycode, is_suspicious_tld, subdomain_count, domain_len,
    # url_len, num_dashes, num_at, entropy_path, has_https, suspicious_kw,
    # kw_hit_count, brand_in_url, brand_in_subdomain, typosquat_score,
    # num_pw_inputs, form_action_diff_domain, form_insecure_http,
    # logo_mismatch, hidden_iframes
    direct = {
        "url_len":    "length_url",
        "num_dashes": "qty_hyphen_url",
        "num_at":     "qty_at_url",
        "domain_len": "domain_length",
        "has_ip":     "domain_in_ip",
        "has_https":  "tls_ssl_certificate",
    }
    for our_col, vr_col in direct.items():
        if vr_col in df.columns:
            vals = df[vr_col].fillna(0).values.astype(float)
            vals = np.where(vals < 0, 0.0, vals)   # -1 sentinel → 0
            feat_matrix[:, feat_idx[our_col]] = vals

    # ── Derived: subdomain_count = max(0, dots_in_domain - 1) ─────────────────
    if "qty_dot_domain" in df.columns:
        dots = df["qty_dot_domain"].fillna(1).values.astype(float)
        feat_matrix[:, feat_idx["subdomain_count"]] = np.maximum(0, dots - 1)

    # ── Derived: suspicious_kw = url_shortened OR email_in_url ────────────────
    kw_signal = np.zeros(n)
    for col in ["url_shortened", "email_in_url"]:
        if col in df.columns:
            kw_signal = np.maximum(kw_signal,
                                   df[col].fillna(0).values.astype(float).clip(0))
    feat_matrix[:, feat_idx["suspicious_kw"]] = (kw_signal > 0).astype(float)
    feat_matrix[:, feat_idx["kw_hit_count"]]  = kw_signal

    # ── Derived: entropy_path — from directory_length (path complexity proxy) ──
    # entropy_path IS in FEATURE_COLS (index 8)
    if "directory_length" in df.columns:
        dir_len = df["directory_length"].fillna(0).values.astype(float).clip(0)
        # Approximate entropy: log2(len+1) scaled to [0,6]
        approx_entropy = np.log1p(dir_len) / np.log(256) * 6.0
        feat_matrix[:, feat_idx["entropy_path"]] = approx_entropy

    # ── Derived: punycode — use qty_tilde_url as proxy (obfuscation signal) ───
    if "qty_tilde_url" in df.columns:
        feat_matrix[:, feat_idx["has_punycode"]] = (
            df["qty_tilde_url"].fillna(0).values.astype(float).clip(0) > 0
        ).astype(float)

    # ── Derived: brand_in_url — use server_client_domain as proxy ─────────────
    if "server_client_domain" in df.columns:
        feat_matrix[:, feat_idx["brand_in_url"]] = (
            df["server_client_domain"].fillna(0).values.astype(float).clip(0) > 0
        ).astype(float)

    # ── Derived: typosquat_score — from qty_vowels_domain (unusual domain) ────
    # High vowel ratio + medium length → possibly squatted
    # (best proxy available from Vrbancic features)
    if "qty_vowels_domain" in df.columns and "domain_length" in df.columns:
        vowels  = df["qty_vowels_domain"].fillna(0).values.astype(float).clip(0)
        dom_len = df["domain_length"].fillna(1).values.astype(float).clip(1)
        # Vowel ratio: normal English ≈ 0.38; gibberish domains deviate
        vowel_ratio = vowels / dom_len
        # Squatty score: domains that are mid-length and have odd vowel ratios
        odd_ratio = np.abs(vowel_ratio - 0.38)
        typosquat = (odd_ratio * 2.0).clip(0, 1)
        feat_matrix[:, feat_idx["typosquat_score"]] = typosquat

    # ── HTML features: all 0 (not available in URL-only dataset) ──────────────
    # num_pw_inputs, form_action_diff_domain, form_insecure_http,
    # logo_mismatch, hidden_iframes → remain 0  (handled by heuristic scoring)

    # ── Verify no NaN/Inf ─────────────────────────────────────────────────────
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Cache ─────────────────────────────────────────────────────────────────
    feat_df = pd.DataFrame(feat_matrix, columns=FEATURE_COLS)
    feat_df["label"] = y
    feat_df.to_parquet(FEAT_PARQUET_CACHE, index=False)
    log.info(f"Feature matrix built. Shape: {feat_matrix.shape}")
    return feat_matrix, y


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3  —  EDA
# ═══════════════════════════════════════════════════════════════════════════════

def eda_report(X: np.ndarray, y: np.ndarray):
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["label"] = y
    n_p = int(y.sum())
    n_b = int((y == 0).sum())
    log.info("=" * 68)
    log.info("EXPLORATORY DATA ANALYSIS")
    log.info("=" * 68)
    log.info(f"  Samples total  : {len(y):>8,}")
    log.info(f"  Phishing  (1)  : {n_p:>8,}  ({n_p/len(y)*100:.1f}%)")
    log.info(f"  Benign    (0)  : {n_b:>8,}  ({n_b/len(y)*100:.1f}%)")
    log.info(f"  Imbalance ratio: {n_p/max(n_b,1):.2f}")
    log.info("")
    log.info(f"  {'Feature':<30} {'Phish':>10} {'Benign':>10} {'AUC(1v1)':>10}")
    log.info(f"  {'-'*62}")
    for col in FEATURE_COLS:
        pm  = df[df.label == 1][col].mean()
        bm  = df[df.label == 0][col].mean()
        # Quick univariate AUC as discriminative power indicator
        from sklearn.metrics import roc_auc_score as _auc
        try:
            ua = _auc(y, X[:, FEATURE_COLS.index(col)])
            ua = max(ua, 1 - ua)   # always ≥ 0.5
        except Exception:
            ua = 0.5
        log.info(f"  {col:<30} {pm:>10.3f} {bm:>10.3f} {ua:>10.4f}")
    log.info("=" * 68)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4  —  Model Zoo
# ═══════════════════════════════════════════════════════════════════════════════

def _make_zoo() -> Dict:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=RANDOM_SEED,
                class_weight="balanced", solver="lbfgs",
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=RANDOM_SEED,
                class_weight="balanced", n_jobs=-1, max_depth=12,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.08,
                max_depth=4, subsample=0.85,
                min_samples_split=15, random_state=RANDOM_SEED,
            )),
        ]),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_depth=5,
            random_state=RANDOM_SEED, class_weight="balanced",
        ),
    }


PARAM_GRIDS = {
    "GradientBoosting": {
        "clf__n_estimators":      [200, 300, 400, 500],
        "clf__learning_rate":     [0.03, 0.05, 0.08, 0.10, 0.15],
        "clf__max_depth":         [3, 4, 5, 6],
        "clf__subsample":         [0.7, 0.8, 0.85, 0.9, 1.0],
        "clf__min_samples_split": [5, 10, 15, 20, 30],
        "clf__min_samples_leaf":  [1, 3, 5, 10],
    },
    "HistGradientBoosting": {
        "max_iter":            [200, 300, 400, 500],
        "learning_rate":       [0.03, 0.05, 0.08, 0.10, 0.15],
        "max_depth":           [3, 4, 5, 6, None],
        "min_samples_leaf":    [10, 20, 30, 50],
        "l2_regularization":   [0.0, 0.05, 0.1, 0.5],
        "max_leaf_nodes":      [None, 15, 31, 63],
    },
    "RandomForest": {
        "clf__n_estimators":      [200, 300, 400],
        "clf__max_depth":         [None, 10, 15, 20],
        "clf__min_samples_split": [2, 5, 10],
        "clf__max_features":      ["sqrt", "log2", 0.4, 0.6],
    },
    "LogisticRegression": {
        "clf__C":       [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "clf__penalty": ["l2"],
    },
}


def compare_models(X_train: np.ndarray, y_train: np.ndarray) -> str:
    zoo = _make_zoo()
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    log.info("=" * 68)
    log.info("MODEL COMPARISON  (5-fold stratified CV, AUC-ROC primary metric)")
    log.info("=" * 68)
    log.info(f"  {'Model':<24} {'AUC±std':>14} {'F1':>8} {'Prec':>8} {'Rec':>8}")
    log.info(f"  {'-'*64}")

    best_name, best_auc = None, -1.0
    for name, model in zoo.items():
        t0 = time.time()
        auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc",   n_jobs=-1)
        f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1",        n_jobs=-1)
        pr  = cross_val_score(model, X_train, y_train, cv=cv, scoring="precision", n_jobs=-1)
        rc  = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall",    n_jobs=-1)
        marker = ""
        if auc.mean() > best_auc:
            best_auc, best_name = auc.mean(), name
            marker = "  <-- BEST"
        log.info(
            f"  {name:<24} {auc.mean():.4f}±{auc.std():.4f}"
            f"  {f1.mean():.4f}  {pr.mean():.4f}  {rc.mean():.4f}"
            f"  ({time.time()-t0:.0f}s){marker}"
        )

    log.info("=" * 68)
    log.info(f"Winner: {best_name}  (val AUC = {best_auc:.4f})")
    return best_name


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5  —  Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════════

def tune(name: str, X_train: np.ndarray, y_train: np.ndarray):
    zoo        = _make_zoo()
    base_model = zoo[name]
    grid       = PARAM_GRIDS.get(name, {})

    if not grid:
        log.info(f"No param grid for {name}.")
        base_model.fit(X_train, y_train)
        return base_model

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    log.info(f"\nTuning {name} — RandomizedSearchCV ({N_ITER_SEARCH} iters, 5-fold)...")
    search = RandomizedSearchCV(
        base_model, grid,
        n_iter=N_ITER_SEARCH, scoring="roc_auc",
        cv=cv, refit=True, n_jobs=-1,
        random_state=RANDOM_SEED, verbose=1,
    )
    search.fit(X_train, y_train)
    log.info(f"Best CV AUC : {search.best_score_:.4f}")
    log.info(f"Best params : {json.dumps({k: str(v) for k, v in search.best_params_.items()}, indent=2)}")
    return search.best_estimator_


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6  —  Calibration
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate(model, X_val: np.ndarray, y_val: np.ndarray):
    """Platt scaling on the held-out val set (prefit=True → no data reuse)."""
    log.info("\nCalibrating probabilities (Platt scaling, prefit)...")
    cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)
    log.info("Calibration complete.")
    return cal


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7  —  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
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

    log.info("")
    log.info("=" * 68)
    log.info("FINAL TEST SET EVALUATION  (held-out, never seen during training)")
    log.info("=" * 68)
    log.info(f"  Samples     : {len(y_test):,}  ({int(y_test.sum())} phishing | {int((y_test==0).sum())} benign)")
    log.info(f"  AUC-ROC     : {auc:.4f}   (target: >0.96)")
    log.info(f"  Avg Prec    : {ap:.4f}")
    log.info(f"  Accuracy    : {acc:.4f}")
    log.info(f"  Precision   : {prec:.4f}   (of flagged URLs, how many are truly phishing)")
    log.info(f"  Recall      : {rec:.4f}   (of phishing URLs, how many we catch)")
    log.info(f"  F1-Score    : {f1:.4f}")
    log.info(f"  FP Rate     : {fpr:.4f}   << critical: fraction of legit sites wrongly flagged")
    log.info("")
    log.info(f"  Confusion matrix  (rows=actual, cols=predicted):")
    log.info(f"    TN={tn:>6}  FP={fp:>6}   (benign correctly/wrongly flagged)")
    log.info(f"    FN={fn:>6}  TP={tp:>6}   (phishing missed/caught)")
    log.info("")
    log.info(classification_report(y_test, y_pred, target_names=["Benign", "Phishing"]))
    log.info("=" * 68)

    return {
        "auc_roc":             round(auc, 4),
        "avg_precision":       round(ap, 4),
        "accuracy":            round(acc, 4),
        "precision":           round(prec, 4),
        "recall":              round(rec, 4),
        "f1_score":            round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "confusion_matrix":    cm.tolist(),
        "n_test":              len(y_test),
        "n_tp": int(tp), "n_tn": int(tn), "n_fp": int(fp), "n_fn": int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8  —  Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def feat_importance(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    log.info("\nPermutation feature importance (AUC drop, 5 repeats)...")
    r = permutation_importance(
        model, X_test, y_test,
        n_repeats=5, random_state=RANDOM_SEED,
        scoring="roc_auc", n_jobs=-1,
    )
    imp = pd.Series(r.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)
    log.info(f"  {'Feature':<30} {'Imp (AUC drop)':>16}  Bar")
    log.info(f"  {'-'*58}")
    for feat_name, val in imp.items():
        bar = "#" * max(0, int(val * 300))
        log.info(f"  {feat_name:<30} {val:>14.4f}  {bar}")
    return imp.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_save(model_path: str = MODEL_PATH):
    t0 = time.time()
    log.info("=" * 68)
    log.info("PhishLens ML Training Pipeline  v2.0")
    log.info(f"Dataset  : Vrbancic et al. 2020 (88k real URLs, 111 features)")
    log.info(f"Model out: {model_path}")
    log.info(f"Seed     : {RANDOM_SEED}")
    log.info("=" * 68)

    # 1. Data
    df = download_vrbancic()
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
        log.info(f"Sampled {MAX_SAMPLES:,} rows for training speed.")

    # 2. Feature mapping
    X, y = map_features(df)

    # 3. EDA
    eda_report(X, y)

    # 4. Stratified splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, random_state=RANDOM_SEED, stratify=y_temp
    )
    log.info(f"\nSplit: train={len(y_train):,} | val={len(y_val):,} | test={len(y_test):,}")
    log.info(f"Train phishing%: {y_train.mean()*100:.1f}%")

    # 5. Model zoo CV
    winner = compare_models(X_train, y_train)

    # 6. Hyperparameter tuning
    tuned = tune(winner, X_train, y_train)

    # 7. Calibration on val set
    calibrated = calibrate(tuned, X_val, y_val)

    # 8. Test evaluation
    metrics = evaluate(calibrated, X_test, y_test)

    # 9. Feature importance
    importances = feat_importance(calibrated, X_test, y_test)

    # 10. Save
    joblib.dump(calibrated, model_path)

    metadata = {
        "version":            "2.0.0",
        "trained_at":         datetime.datetime.utcnow().isoformat() + "Z",
        "training_duration_s": round(time.time() - t0, 1),
        "dataset":            "Vrbancic et al. 2020 (88k real URLs)",
        "dataset_url":        VRBANCIC_URL,
        "model_type":         winner,
        "model_path":         str(model_path),
        "feature_cols":       FEATURE_COLS,
        "n_features":         N_FEATURES,
        "n_train":            len(y_train),
        "n_val":              len(y_val),
        "n_test":             len(y_test),
        "calibration":        "Platt (sigmoid, prefit on val set)",
        "hp_tuning":          f"RandomizedSearchCV n_iter={N_ITER_SEARCH}, cv={CV_FOLDS}-fold",
        "metrics":            metrics,
        "feature_importances": {k: round(float(v), 5) for k, v in importances.items()},
    }
    with open(MODEL_META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 68)
    log.info("TRAINING COMPLETE")
    log.info(f"  Duration : {elapsed:.0f}s")
    log.info(f"  AUC-ROC  : {metrics['auc_roc']}")
    log.info(f"  F1       : {metrics['f1_score']}")
    log.info(f"  FPR      : {metrics['false_positive_rate']}  (legit sites wrongly flagged)")
    log.info(f"  Saved    : {model_path}")
    log.info(f"  Meta     : {MODEL_META_FILE}")
    log.info("=" * 68)


if __name__ == "__main__":
    train_and_save()
