# PhishLens — Ablation Study Results

All 5 experiments share **seed=42, 70/15/15 split, same indices**.  
Apples-to-apples: feature effect is isolated, not model tuning or sampling.

---

## Results Table

| Experiment | Feat | AUC | Accuracy | F1 | **FPR** | FNR | Size | ms/URL |
|-----------|------|-----|---------|---|---------|-----|------|--------|
| E1: 14-feat RF | 14 | 0.9933 | 96.4% | 0.9724 | 4.72% | 3.01% | 48 MB | 0.297 |
| E2: 12-feat RF (no WHOIS) | 12 | 0.9875 | 95.2% | 0.9635 | 7.44% | 3.39% | 66 MB | 0.281 |
| E3: Existing-equiv + WHOIS (HistGBT) | 15 | 0.9897 | 95.4% | 0.9645 | 5.70% | 4.06% | 1.6 MB | 0.109 |
| E4: 14-feat + entropy_path (HistGBT) | 15 | **0.9942** | **96.7%** | **0.9748** | **4.61%** | **2.61%** | 1.9 MB | 0.141 |
| **E5: 14-feat HistGBT** ✅ | **14** | **0.9942** | **96.7%** | **0.9748** | **4.61%** | **2.61%** | **1.9 MB** | **0.063** |
| BASELINE: 20-feat HistGBT | 20 | 0.9793 | 93.2% | 0.9485 | 10.46% | ~5% | 1.2 MB | <0.1 |

---

## Key Findings

### Finding 1 — WHOIS features matter, but URL structure carries most of the weight

**E1 (with WHOIS) vs E2 (without WHOIS):**

| | With WHOIS | Without WHOIS | Delta |
|--|--|--|--|
| AUC | 0.9933 | 0.9875 | −0.006 |
| FPR | 4.72% | 7.44% | **+2.72 pp** |

Removing `time_domain_activation` + `time_domain_expiration` costs 0.6% AUC and **57% more false positives**.
Domain age is a real signal — but the URL path features (`directory_length`, `file_length`, `qty_slash_directory`)
alone reach 98.75% AUC. The user's intuition was correct.

### Finding 2 — `entropy_path` adds ZERO incremental value over the 14-feature set

**E4 vs E5 (identical results):**

- E4: 14 features + `entropy_path` → AUC=**0.9942**, FPR=**4.61%**
- E5: 14 features only       → AUC=**0.9942**, FPR=**4.61%**

`entropy_path` in our system is `log1p(directory_length) / log(256) * 6`, a monotonic transform
of `directory_length` which is already in the 14 features. The model learns the same information.
The existing model's 78% dependency on `entropy_path` was a symptom of having too few real features —
not that `entropy_path` is special.

### Finding 3 — RF is worse AND bigger than HistGBT on the same features

| | E1 (RF) | E5 (HistGBT) |
|--|--|--|
| AUC | 0.9933 | **0.9942** |
| FPR | 4.72% | **4.61%** |
| Size | 48 MB | **1.9 MB** |
| Speed | 0.297ms | **0.063ms** |

HistGBT strictly dominates RF on all four axes simultaneously. The 96 MB RF from the full
tuning run was not worth it even on accuracy grounds.

### Finding 4 — The "existing equivalent + WHOIS" (E3) underperforms 14-feat

E3 reconstructs the existing model's features from Vrbancic columns and adds WHOIS.
AUC=**0.9897**, FPR=**5.70%**. Still better than the original 20-feat baseline,
but 0.45 FPR percentage points worse than E5. The 14 PhishNot features are a better
selection than the mapping we designed.

---

## The Winner: E5

> **14-feat HistGBT: AUC=0.9942, FPR=4.61%, 1.9 MB, 0.063ms/URL**

- **Deployable**: 1.9 MB, fits comfortably as a backend model
- **Fast**: 0.063ms per URL — 5× faster than the RF at inference
- **NaN-native**: HistGBT handles missing WHOIS values without imputation
- **Best overall**: Wins on AUC, F1, FPR, FNR simultaneously vs all other experiments
- **vs existing baseline**: FPR halved (10.46% → 4.61%), AUC +1.5%, F1 +2.6%

---

## What the user predicted correctly

> *"The result doesn't support the idea that WHOIS/domain age is solely responsible for the improvement."*

✅ Confirmed. E2 (no WHOIS) still reaches 0.9875 AUC — URL path structure carries the bulk.

> *"Don't reduce the RF to 50–100 trees and assume it'll solve the 96 MB problem."*

✅ Confirmed. E5 (HistGBT, 14 features) gets **better AUC** than E1 (RF) at **26× smaller** size.
Switching the algorithm is the right move, not shrinking the RF.

> *"HistGBT on the 14 is the one I'd be most interested in."*

✅ E5 = deployment winner, validated by experiment.

---

## Next Step Options

### Option A — Deploy E5 immediately
Retrain E5 with proper `RandomizedSearchCV` tuning (currently used fixed params).
Expected to improve slightly (currently ~0.9940 with `learning_rate=0.08, max_leaf_nodes=63`).
Update `ml_model.py` to load from the 14-feature schema.

### Option B — Test on external dataset first
Both models trained/tested on Vrbancic distribution. An external validation set
(e.g., PhishTank + Tranco live sample) would reveal distribution shift.
This is the honest way to know if 0.9942 holds in the wild.

### Option C — Add remaining features to E5
The 5 network features (`time_response`, `asn_ip`, `ttl_hostname`, `time_domain_activation`,
`time_domain_expiration`) are currently populated from WHOIS/DNS at inference time.
`trust.py` already does RDAP lookups. Integration is straightforward.

**Recommendation: Option A + B in parallel** — deploy E5 to replace the existing model
while collecting live URL samples for external validation.

---

## Files Produced

| File | Contents |
|------|---------|
| [`backend/experiments.py`](file:///c:/Users/parve/OneDrive/Desktop/github/phishlens/backend/experiments.py) | Ablation runner (reproducible) |
| `backend/data/ablation_results.json` | Machine-readable results |
| [`backend/features_14.py`](file:///c:/Users/parve/OneDrive/Desktop/github/phishlens/backend/features_14.py) | 14-feature extractor (URL + network) |
| [`backend/train_model_14.py`](file:///c:/Users/parve/OneDrive/Desktop/github/phishlens/backend/train_model_14.py) | Full 14-feat RF pipeline |
| `backend/data/models/phishlens_14_rf.pkl` | Trained RF (reference, not for deploy) |

