# PhishLens — Model Comparison Report

## Results Summary

| Metric | 14-feat RF | 20-feat HistGBT | Δ |
|--------|-----------|-----------------|---|
| **AUC-ROC** | **0.9933** | 0.9793 | +1.4% |
| **Accuracy** | **0.9637** | 0.9323 | +3.1% |
| **Precision** | **0.9749** | 0.9451 | +3.0% |
| **Recall** | **0.9694** | 0.9518 | +1.8% |
| **F1-Score** | **0.9722** | 0.9485 | +2.5% |
| **FPR** | **0.0472** | 0.1046 | **−55% false positives** |
| **FNR** | **0.0306** | ~0.050 | −38% missed phish |
| Inference (ms/URL) | 0.375 | <0.1 | 3.75× slower |
| Model size | **96 MB** | 1.2 MB | **82× larger** |
| N features | 14 | 20 | |

> Evaluated on held-out test set. Both models trained on Vrbančič 2020 (88,647 URLs).
> Same RANDOM_SEED=42, same stratified split strategy.

---

## Feature Importance — 14-Feature RF

| Rank | Feature | Importance (AUC drop) | Type |
|------|---------|----------------------|------|
| 1 | `directory_length` | 0.0362 | URL-structural |
| 2 | `file_length` | 0.0294 | URL-structural |
| 3 | `time_domain_activation` | 0.0277 | Network/WHOIS |
| 4 | `qty_slash_directory` | 0.0159 | URL-structural |
| 5 | `asn_ip` | 0.0052 | Network |
| 6 | `qty_dot_domain` | 0.0051 | URL-structural |
| 7 | `ttl_hostname` | 0.0044 | Network/DNS |
| 8 | `time_domain_expiration` | 0.0030 | Network/WHOIS |
| 9 | `domain_length` | 0.0029 | URL-structural |
| 10 | `params_length` | 0.0022 | URL-structural |
| 11 | `time_response` | 0.0022 | Network |
| 12 | `qty_dot_directory` | 0.0021 | URL-structural |
| 13 | `qty_vowels_domain` | 0.0014 | URL-structural |
| 14 | `qty_dot_file` | 0.0014 | URL-structural |

**Key insight**: URL-structural features (path depth, file length) dominate, but
`time_domain_activation` ranks #3 — domain age is a powerful signal that the
20-feature model couldn't access.

---

## Feature Importance — 20-Feature HistGBT (for comparison)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `entropy_path` | 0.2324 |
| 2 | `url_len` | 0.0288 |
| 3 | `subdomain_count` | 0.0202 |
| 4 | `typosquat_score` | 0.0114 |
| 5–20 | (all others) | < 0.008 |

One feature (`entropy_path`) drives ~78% of the 20-feature model's signal —
brittle and lacks domain-age information.

---

## Key Trade-offs

### 14-feature RF wins on:
- All accuracy metrics (AUC, F1, precision, recall)
- **FPR halved**: 4.72% vs 10.46% — far fewer legitimate sites wrongly flagged
- Has access to WHOIS age signals (most powerful known phishing indicator)

### 14-feature RF loses on:
- **Model size**: 96 MB vs 1.2 MB — unacceptable for an extension backend
- **Inference speed**: 0.375ms vs <0.1ms (still fast enough, but 3.75× slower)
- **5 features require backend lookups** — can't run purely in browser
- **53.6% missing** on directory features (URL-only records in dataset)

---

## Missing Value Analysis

| Feature | Missing% | Impact |
|---------|----------|--------|
| `params_length` | **91.6%** | Imputed with median; degrades signal |
| `qty_dot_directory` … `file_length` | **53.6%** | Half of URL-only records have no path |
| `time_domain_activation` | 27.5% | WHOIS often unavailable |
| `time_domain_expiration` | 29.6% | WHOIS often unavailable |

---

## Differences from Published PhishNot Methodology

1. **Dataset**: We use the full Vrbančič 2020 CSV (88,647 rows). The original PhishNot paper may have used a different subset or version.
2. **Missing values**: We impute with median. The paper may have used a different strategy or excluded rows.
3. **Calibration**: We apply Platt scaling on a held-out val set. The paper uses raw RF probabilities.
4. **Tuning**: We run RandomizedSearchCV (40 iters). The paper uses default RF parameters (n_estimators=100).
5. **Class imbalance**: We use `class_weight="balanced"`. The paper treats the dataset as-is.

> We do **not** claim exact reproduction of the published paper results.

---

## Features NOT Available in Browser Extension

| Feature | Reason | Workaround |
|---------|--------|------------|
| `time_response` | Requires TCP connection | Backend `/check` endpoint |
| `asn_ip` | Requires DNS + IP lookup | Backend enrichment |
| `time_domain_activation` | Requires WHOIS/RDAP | Backend (cached via `trust.py`) |
| `time_domain_expiration` | Requires WHOIS/RDAP | Backend (cached via `trust.py`) |
| `ttl_hostname` | Requires DNS query | Backend enrichment |

Features 1–9 (URL-structural) are fully available in-extension with zero latency.

---

## Recommendation

The 14-feature model is strictly more accurate, but at **96 MB it's not deployable** as-is.

**Next steps**:
1. Reduce RF to `n_estimators=50–100` and `max_depth=10–15` to get model size under 5 MB
2. Or switch to `GradientBoosting` / `HistGBT` on the 14 features — same accuracy, <2 MB
3. Add WHOIS age (`time_domain_activation`) as a new feature to the 20-feature pipeline
4. The strongest combined approach: HistGBT on ~8 features (directory_length, file_length,
   time_domain_activation, qty_slash_directory, asn_ip, qty_dot_domain, url_len, entropy_path)

---

## Files Created

| File | Purpose |
|------|---------|
| `backend/features_14.py` | Deterministic URL parser + network enrichment |
| `backend/train_model_14.py` | Full training pipeline (reproduce with `python backend/train_model_14.py`) |
| `backend/test_features_14.py` | 53 unit tests — all passing |
| `backend/data/features_14.parquet` | Processed dataset (cache-versioned) |
| `backend/data/models/phishlens_14_rf.pkl` | Trained + calibrated RF model |
| `backend/data/model_14_metadata.json` | Training metadata + metrics |
| `backend/data/comparison_report.json` | Machine-readable comparison |
