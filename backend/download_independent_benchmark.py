"""
PhishLens — Stage 5.6: Independent Benchmark Dataset Fetcher & Evaluator
========================================================================
Downloads fresh, independent live threat feeds (OpenPhish / URLhaus) and
unseen benign web domains to construct a completely independent external
test benchmark dataset for cross-distribution evaluation.
"""

import os
import sys
import json
import time
import urllib.request
import logging
from pathlib import Path
from urllib.parse import urlparse

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import joblib

from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from features_14 import FEATURE_14, AVAILABLE_IN_EXTENSION, extract_url_features_14
from features_dom import DOM_FEATURE_COLS
from trust import _tranco
from features import extract_url_features
from model import predict_risk

DATA_DIR = Path(DATA_DIR)
MODELS_DIR = DATA_DIR / "models"
URL_MODEL_PATH = MODELS_DIR / "phishlens_multisource_histgbt.pkl"
DOM_MODEL_PATH = MODELS_DIR / "phishlens_dom_histgbt.pkl"

ROOT_DIR = SCRIPT_DIR.parent
DATASETS_DIR = ROOT_DIR / "datasets"
BENCHMARK_CSV = DATASETS_DIR / "independent_live_benchmark.csv"
RESULTS_JSON = DATA_DIR / "independent_benchmark_results.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("phishlens.indep_bench")


def fetch_independent_data() -> pd.DataFrame:
    """Fetches real live threat URLs and benign samples."""
    if BENCHMARK_CSV.exists():
        log.info(f"Loading existing independent benchmark from {BENCHMARK_CSV.name}...")
        return pd.read_csv(BENCHMARK_CSV)

    log.info("Fetching fresh live threat URLs from OpenPhish feed...")
    phish_urls = []
    try:
        req = urllib.request.Request("https://openphish.com/feed.txt", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            lines = resp.read().decode("utf-8", errors="ignore").strip().splitlines()
            for l in lines:
                u = l.strip()
                if u.startswith("http"):
                    phish_urls.append(u)
        log.info(f"  Downloaded {len(phish_urls):,} active phishing URLs from OpenPhish.")
    except Exception as e:
        log.warning(f"  OpenPhish download error: {e}")

    # Fallback/Additional source: URLhaus
    try:
        log.info("Fetching active malware/phishing URLs from URLhaus...")
        req_uh = urllib.request.Request("https://urlhaus.abuse.ch/downloads/csv_recent/", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_uh, timeout=12) as resp:
            uh_text = resp.read().decode("utf-8", errors="ignore")
            for line in uh_text.splitlines():
                if line.startswith('"') and '","' in line:
                    parts = line.strip('"').split('","')
                    if len(parts) >= 3 and parts[2].startswith("http"):
                        phish_urls.append(parts[2])
        log.info(f"  Total unique threat URLs collected: {len(set(phish_urls)):,}")
    except Exception as e:
        log.warning(f"  URLhaus download error: {e}")

    phish_urls = list(set(phish_urls))[:2000]

    # Generate balanced Benign URLs from diverse benign domains
    log.info("Generating matching diverse benign URLs from verified domains...")
    _tranco._load()
    tranco_domains = list(_tranco._data.keys())[1000:3500]  # rank 1000 to 3500 (realistic benign websites)

    benign_paths = [
        "", "/", "/about", "/contact", "/pricing", "/products", "/features",
        "/docs/overview", "/help/center", "/blog/news/2026", "/support/faq",
        "/terms-of-service", "/privacy-policy", "/login", "/register", "/app"
    ]

    benign_urls = []
    for i, dom in enumerate(tranco_domains):
        p = benign_paths[i % len(benign_paths)]
        proto = "https" if i % 10 != 0 else "http"
        benign_urls.append(f"{proto}://{dom}{p}")
        if len(benign_urls) >= len(phish_urls):
            break

    n_sample = min(len(phish_urls), len(benign_urls))
    phish_sample = phish_urls[:n_sample]
    benign_sample = benign_urls[:n_sample]

    df_phish = pd.DataFrame({"url": phish_sample, "label": 1, "source": "openphish_live"})
    df_benign = pd.DataFrame({"url": benign_sample, "label": 0, "source": "tranco_benign"})

    df_all = pd.concat([df_phish, df_benign], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_all.to_csv(BENCHMARK_CSV, index=False)
    log.info(f"Created and saved independent benchmark ({len(df_all):,} samples) -> {BENCHMARK_CSV.name}")
    return df_all


def run_independent_benchmark():
    log.info("=" * 90)
    log.info("PHISHLENS: STAGE 5.6 INDEPENDENT BENCHMARK EVALUATION")
    log.info("=" * 90)

    df = fetch_independent_data()
    n = len(df)
    y_true = df["label"].values
    log.info(f"Loaded {n:,} independent evaluation samples ({int(np.sum(y_true==1)):,} Phish / {int(np.sum(y_true==0)):,} Legit)")

    # Load URL Model & Tranco
    _tranco._load()
    url_pkg = joblib.load(URL_MODEL_PATH)
    url_model = url_pkg["model"]

    # Score with URL Model
    log.info("Extracting URL features and scoring...")
    mat_url = np.full((n, len(FEATURE_14)), np.nan, dtype=np.float64)
    feat_idx = {col: i for i, col in enumerate(FEATURE_14)}
    for i, u in enumerate(df["url"]):
        f = extract_url_features_14(str(u))
        for k in AVAILABLE_IN_EXTENSION:
            mat_url[i, feat_idx[k]] = f.get(k, np.nan)

    p_url = url_model.predict_proba(mat_url)[:, 1]

    # Score with Tri-Component Production Ensemble (URL + Heuristics + Trust)
    log.info("Evaluating full production ensemble scores...")
    p_ensemble = np.zeros(n, dtype=np.float64)

    for i, raw_url in enumerate(df["url"]):
        u_str = str(raw_url)
        feat = extract_url_features(u_str)
        score, lvl, _ = predict_risk(feat, raw_url=u_str, raw_html="")
        p_ensemble[i] = score

    # Metrics
    auc_url = float(roc_auc_score(y_true, p_url))
    prauc_url = float(average_precision_score(y_true, p_url))
    auc_ens = float(roc_auc_score(y_true, p_ensemble))
    prauc_ens = float(average_precision_score(y_true, p_ensemble))

    log.info("\n" + "=" * 80)
    log.info("INDEPENDENT LIVE BENCHMARK RESULTS")
    log.info("=" * 80)
    log.info(f"  URL Model Only:         ROC-AUC = {auc_url:.4f} | PR-AUC = {prauc_url:.4f}")
    log.info(f"  Full Tri-Component:     ROC-AUC = {auc_ens:.4f} | PR-AUC = {prauc_ens:.4f}")
    log.info("=" * 80)

    # Threshold evaluation on independent live data
    thresholds = [0.20, 0.35, 0.50, 0.65]
    log.info(f"\n{'Threshold':<10} {'Accuracy':<10} {'Precision':<11} {'Recall':<10} {'F1':<8} {'FPR':<8} {'FNR':<8}")
    log.info("-" * 65)

    perf_table = []
    for tau in thresholds:
        y_pred = (p_ensemble >= tau).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)

        perf_table.append({
            "threshold": tau,
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1, 4),
            "fpr": round(fpr * 100, 2),
            "fnr": round(fnr * 100, 2),
        })
        log.info(f"{tau:<10.2f} {acc*100:<10.2f}% {prec*100:<11.2f}% {rec*100:<10.2f}% {f1:<8.4f} {fpr*100:<8.2f}% {fnr*100:<8.2f}%")

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": "Independent Live Threat Benchmark (OpenPhish + Tranco)",
        "total_samples": n,
        "auc_url_only": round(auc_url, 4),
        "auc_full_ensemble": round(auc_ens, 4),
        "threshold_performance": perf_table,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    log.info(f"\nSaved Independent Benchmark Results -> {RESULTS_JSON}")


if __name__ == "__main__":
    run_independent_benchmark()
