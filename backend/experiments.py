"""
PhishLens — Feature Ablation Study
=====================================
Runs 5 experiments to isolate which features drive performance,
using a SINGLE canonical split across all experiments.

Experiments
-----------
  E1  14-feat RF               (baseline, already done — re-run for reference)
  E2  12-feat RF (no WHOIS)    (remove time_domain_activation, time_domain_expiration)
  E3  HistGBT + "existing 20 equivalent" + WHOIS  (best-of-existing + domain age)
  E4  14-feat + entropy_path   (HistGBT — what happens when we add 20-model's top feature?)
  E5  14-feat HistGBT          (architecture swap: same features, smaller model)

Design decisions
----------------
- Same 88,647 rows, RANDOM_SEED=42, same stratified 70/15/15 split
- RF experiments use fixed hyperparams (n_estimators=150, max_depth=20,
  class_weight=balanced) — no re-tuning, isolates feature effect not model tuning
- HistGBT experiments use sklearn defaults — fast, small, NaN-native
- All results calibrated with Platt scaling on the val set
- Single output table for direct comparison

Usage
-----
    python backend/experiments.py
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from config import DATA_DIR
from features_14 import FEATURE_14

DATA_DIR  = Path(DATA_DIR)
FEAT14_PQ = DATA_DIR / "features_14.parquet"
VRB_CSV   = DATA_DIR / "vrbancic_dataset.csv"
OUT_JSON  = DATA_DIR / "ablation_results.json"

RANDOM_SEED = 42
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.ablation")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_base_14() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load the cached 14-feature matrix. Returns df (with NaN) and y."""
    if not FEAT14_PQ.exists():
        raise FileNotFoundError(
            f"{FEAT14_PQ} not found. Run: python backend/train_model_14.py first."
        )
    df = pd.read_parquet(FEAT14_PQ)
    y  = df["label"].values
    return df[FEATURE_14], y


def load_vrbancic_raw() -> pd.DataFrame:
    """Load the raw Vrbancic CSV (needed for Exp 3 extra columns)."""
    log.info(f"Loading raw Vrbancic CSV for Exp 3 features…")
    df = pd.read_csv(VRB_CSV)
    df = df.replace(-1, np.nan)   # sentinel -> NaN
    return df


def build_exp3_features(df_raw: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """
    Exp 3: Reconstruct the 'existing 20-feature model' feature set from
    Vrbancic columns, then augment with WHOIS domain-age features.

    Maps the same columns that train_model.py uses for PhishLens features,
    staying in raw Vrbancic name-space for transparency.

    Features (15 total):
      URL/domain (9)  : length_url, qty_hyphen_url, qty_at_url, domain_length,
                        qty_dot_domain, domain_in_ip, tls_ssl_certificate,
                        url_shortened, directory_length
      Derived (2)     : entropy_path (log1p of directory_length),
                        typosquat_proxy (vowel-ratio deviation)
      WHOIS (2)       : time_domain_activation, time_domain_expiration
      Obfuscation (1) : qty_tilde_url (punycode proxy)
      Brand (1)       : server_client_domain
    """
    n = len(df_raw)
    out = pd.DataFrame(index=df_raw.index)

    def col(name, default=0.0):
        if name in df_raw.columns:
            return df_raw[name].fillna(default).values.astype(float)
        return np.full(n, default, dtype=float)

    # ── Direct ────────────────────────────────────────────────────────────────
    out["length_url"]           = col("length_url")
    out["qty_hyphen_url"]       = col("qty_hyphen_url")
    out["qty_at_url"]           = col("qty_at_url")
    out["domain_length"]        = col("domain_length")
    out["qty_dot_domain"]       = col("qty_dot_domain")
    out["domain_in_ip"]         = col("domain_in_ip")
    out["tls_ssl_certificate"]  = col("tls_ssl_certificate")
    out["directory_length"]     = col("directory_length")
    out["qty_tilde_url"]        = (col("qty_tilde_url") > 0).astype(float)
    out["server_client_domain"] = (col("server_client_domain") > 0).astype(float)

    # ── Derived: entropy_path proxy ───────────────────────────────────────────
    dl = df_raw["directory_length"].fillna(0).values.astype(float).clip(0) \
         if "directory_length" in df_raw.columns else np.zeros(n)
    out["entropy_path"] = np.log1p(dl) / np.log(256) * 6.0

    # ── Derived: typosquat proxy (vowel ratio deviation from 0.38) ────────────
    vowels  = col("qty_vowels_domain", 0.0)
    dom_len = col("domain_length", 1.0).clip(1)
    odd     = np.abs(vowels / dom_len - 0.38)
    out["typosquat_proxy"] = (odd * 2.0).clip(0, 1)

    # ── Obfuscation: url_shortened OR email_in_url -> suspicious_kw ───────────
    kw = np.zeros(n)
    for c in ["url_shortened", "email_in_url"]:
        if c in df_raw.columns:
            kw = np.maximum(kw, df_raw[c].fillna(0).values.astype(float).clip(0))
    out["suspicious_kw"] = (kw > 0).astype(float)

    # ── WHOIS ─────────────────────────────────────────────────────────────────
    out["time_domain_activation"] = col("time_domain_activation", np.nan)
    out["time_domain_expiration"]  = col("time_domain_expiration",  np.nan)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Splits — canonical, shared across all experiments
# ═══════════════════════════════════════════════════════════════════════════════

def make_canonical_splits(n: int, y: np.ndarray):
    """
    Returns train/val/test INDICES (not data), shared across all experiments.
    Same seed and ratios as train_model_14.py.
    """
    idx = np.arange(n)
    idx_temp, idx_test, y_temp, _ = train_test_split(
        idx, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
    idx_train, idx_val, _, _ = train_test_split(
        idx_temp, y_temp, test_size=val_frac, random_state=RANDOM_SEED, stratify=y_temp
    )
    return idx_train, idx_val, idx_test


# ═══════════════════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════════════════

def make_rf() -> Pipeline:
    """
    Fixed RF config (best params from train_model_14.py tuning).
    No re-search — isolates feature effect.
    """
    return Pipeline([
        ("imp",    SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=150, max_depth=20,
            min_samples_split=2, min_samples_leaf=1,
            max_features=0.4, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])


def make_histgbt() -> HistGradientBoostingClassifier:
    """
    HistGBT with defaults + tuned params from existing model.
    Handles NaN natively — no imputation needed.
    """
    return HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.08,
        max_leaf_nodes=63, min_samples_leaf=30,
        l2_regularization=0.05,
        random_state=RANDOM_SEED,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Training + evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_eval(
    X: np.ndarray, y: np.ndarray,
    idx_train, idx_val, idx_test,
    model_factory,
    label: str,
) -> Dict:
    t0 = time.monotonic()

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    model = model_factory()
    model.fit(X_train, y_train)

    # Platt calibration on val set
    cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)

    y_pred  = cal.predict(X_test)
    y_proba = cal.predict_proba(X_test)[:, 1]

    auc  = roc_auc_score(y_test, y_proba)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / max(fp + tn, 1)
    fnr  = fn / max(fn + tp, 1)

    elapsed = time.monotonic() - t0

    # Model size estimate
    tmp = DATA_DIR / f"_tmp_{label.replace(' ','_')}.pkl"
    joblib.dump(cal, tmp)
    size_kb = tmp.stat().st_size // 1024
    tmp.unlink()

    # Inference speed (100 samples, 10 repeats)
    t1 = time.monotonic()
    for _ in range(10):
        cal.predict_proba(X_test[:100])
    infer_ms = (time.monotonic() - t1) / 10 / 100 * 1000

    return {
        "label":          label,
        "auc":            round(auc,  4),
        "accuracy":       round(acc,  4),
        "precision":      round(prec, 4),
        "recall":         round(rec,  4),
        "f1":             round(f1,   4),
        "fpr":            round(fpr,  4),
        "fnr":            round(fnr,  4),
        "size_kb":        size_kb,
        "infer_ms":       round(infer_ms, 4),
        "train_s":        round(elapsed, 1),
        "n_features":     X.shape[1],
        "n_test":         len(y_test),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Print comparison table
# ═══════════════════════════════════════════════════════════════════════════════

def print_table(results: List[Dict]):
    cols = ["label", "n_features", "auc", "accuracy", "f1", "fpr", "fnr", "size_kb", "infer_ms"]
    w = [32, 5, 7, 9, 7, 7, 7, 9, 10]
    header = "  ".join(f"{c:>{w[i]}}" for i, c in enumerate(cols))
    sep    = "  ".join("-" * ww for ww in w)
    log.info("")
    log.info("=" * 105)
    log.info("ABLATION STUDY — ALL EXPERIMENTS")
    log.info("=" * 105)
    log.info(f"  {header}")
    log.info(f"  {sep}")
    for r in results:
        row = "  ".join(
            f"{str(r.get(c,'?')):>{w[i]}}" for i, c in enumerate(cols)
        )
        log.info(f"  {row}")
    log.info("=" * 105)
    log.info("  (fpr/fnr: lower is better | size_kb: lower is better)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("PhishLens Feature Ablation Study — 5 Experiments")
    log.info(f"Seed={RANDOM_SEED}, split=70/15/15, same indices across all")
    log.info("=" * 70)

    # ── Load base 14-feature matrix ───────────────────────────────────────────
    log.info("\nLoading 14-feature base matrix…")
    df14, y = load_base_14()
    X14 = df14.values                        # NaN preserved for HistGBT
    X14_imp = SimpleImputer(strategy="median").fit_transform(X14)  # for RF

    n = len(y)
    idx_train, idx_val, idx_test = make_canonical_splits(n, y)
    log.info(f"Split: train={len(idx_train):,} | val={len(idx_val):,} | test={len(idx_test):,}")

    # ── Exp 3 extra features (raw CSV) ────────────────────────────────────────
    log.info("\nBuilding Exp 3 feature set (existing-equivalent + WHOIS)…")
    df_raw = load_vrbancic_raw()
    # Align rows: features_14.parquet was built from the full CSV in order
    df_exp3 = build_exp3_features(df_raw, y)
    X3 = df_exp3.values   # NaN-aware for HistGBT

    # ── Exp 4: 14 features + entropy_path ─────────────────────────────────────
    dir_col = df_raw["directory_length"].fillna(0).values.astype(float).clip(0) \
              if "directory_length" in df_raw.columns else np.zeros(n)
    entropy = (np.log1p(dir_col) / np.log(256) * 6.0).reshape(-1, 1)
    X4 = np.hstack([X14, entropy])   # NaN-aware (entropy has no NaN)

    results = []

    # ── E1: 14-feat RF (reference) ────────────────────────────────────────────
    log.info("\n[E1] 14-feature RF (reference)…")
    r = train_and_eval(X14_imp, y, idx_train, idx_val, idx_test,
                       make_rf, "E1: 14-feat RF")
    results.append(r)
    log.info(f"     AUC={r['auc']}  F1={r['f1']}  FPR={r['fpr']}  size={r['size_kb']}KB")

    # ── E2: 12-feat RF (no WHOIS) ─────────────────────────────────────────────
    log.info("\n[E2] 12-feature RF (remove time_domain_activation + time_domain_expiration)…")
    no_whois = [c for c in FEATURE_14
                if c not in ("time_domain_activation", "time_domain_expiration")]
    X2_imp = SimpleImputer(strategy="median").fit_transform(
        df14[no_whois].values
    )
    r = train_and_eval(X2_imp, y, idx_train, idx_val, idx_test,
                       make_rf, "E2: 12-feat RF (no WHOIS)")
    results.append(r)
    log.info(f"     AUC={r['auc']}  F1={r['f1']}  FPR={r['fpr']}  size={r['size_kb']}KB")

    # ── E3: existing-equivalent + WHOIS, HistGBT ──────────────────────────────
    log.info("\n[E3] Existing-equivalent features + WHOIS -> HistGBT…")
    r = train_and_eval(X3, y, idx_train, idx_val, idx_test,
                       make_histgbt, "E3: Existing+WHOIS HistGBT")
    results.append(r)
    log.info(f"     AUC={r['auc']}  F1={r['f1']}  FPR={r['fpr']}  size={r['size_kb']}KB")

    # ── E4: 14 features + entropy_path, HistGBT ───────────────────────────────
    log.info("\n[E4] 14-feat + entropy_path -> HistGBT…")
    r = train_and_eval(X4, y, idx_train, idx_val, idx_test,
                       make_histgbt, "E4: 14+entropy HistGBT")
    results.append(r)
    log.info(f"     AUC={r['auc']}  F1={r['f1']}  FPR={r['fpr']}  size={r['size_kb']}KB")

    # ── E5: 14 features, HistGBT ──────────────────────────────────────────────
    log.info("\n[E5] 14-feature HistGBT (deployment candidate)…")
    r = train_and_eval(X14, y, idx_train, idx_val, idx_test,
                       make_histgbt, "E5: 14-feat HistGBT")
    results.append(r)
    log.info(f"     AUC={r['auc']}  F1={r['f1']}  FPR={r['fpr']}  size={r['size_kb']}KB")

    # ── Reference row from metadata ───────────────────────────────────────────
    meta20_path = DATA_DIR / "model_metadata.json"
    if meta20_path.exists():
        with open(meta20_path) as f:
            m20 = json.load(f).get("metrics", {})
        results.append({
            "label":      "BASELINE: 20-feat HistGBT",
            "n_features": 20,
            "auc":        m20.get("auc_roc", "?"),
            "accuracy":   m20.get("accuracy", "?"),
            "f1":         m20.get("f1_score", "?"),
            "fpr":        m20.get("false_positive_rate", m20.get("fpr", "?")),
            "fnr":        "?",
            "size_kb":    1195,
            "infer_ms":   "<0.1",
            "train_s":    228,
        })

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(results)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
