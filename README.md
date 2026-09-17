# Strategic Financial Aid Yield Optimizer

**Domain:** Admissions & Executive Functions
**Type:** Machine Learning pipeline + REST API (Google Colab)

A machine learning system that predicts an admitted student's probability of enrolling (**yield**) given their academic, demographic, and financial-aid profile — and uses that prediction to recommend financial aid packages that balance enrollment targets, net tuition revenue, and socioeconomic diversity goals.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup & Usage](#setup--usage)
  - [1. Dataset Generation](#1-dataset-generation)
  - [2. Model Training](#2-model-training)
  - [3. Testing & Inference](#3-testing--inference)
  - [4. REST API Endpoint](#4-rest-api-endpoint)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Aid-Package Optimizer](#aid-package-optimizer)
- [Testing Strategy](#testing-strategy)
- [Known Limitations](#known-limitations)
- [Future Work](#future-work)

---

## Problem Statement

Institutions often struggle to balance accessibility and fiscal sustainability. Rigid, broad-brush discounting rules either under-fund high-need students or over-subsidize students who would have enrolled regardless — misallocating aid budgets, depressing yield rates, and straining financial sustainability.

This project addresses that by predicting individual price sensitivity and yield probability, then using those predictions to optimize financial aid allocation — maximizing enrollment and institutional revenue while protecting socioeconomic diversity goals.

## Solution Overview

The system has two layers:

1. **A yield-prediction model** — a `scikit-learn` classification pipeline that takes an applicant's academic, demographic, and aid-package features and outputs a probability of enrollment.
2. **An aid-package optimizer** — built on top of the model, it searches a grid of possible aid percentages for a given applicant and recommends the one that maximizes **expected net revenue** (`yield_probability × net_price`), subject to:
   - a minimum yield-probability floor (protects enrollment-count targets), and
   - an equity floor guaranteeing a minimum aid percentage to high-need applicants (protects diversity goals against pure revenue-maximization).

The trained model is served behind a **Flask REST API** that accepts a JSON applicant record and returns a JSON prediction, and persists every prediction to disk for auditability.

## Dataset

No public dataset pairs individual financial-aid offers with enrollment outcomes anywhere — for Tamil Nadu, India-wide, or internationally. That data lives in institutions' private admissions/CRM systems and is not published.

This project therefore uses a **synthetic, Tamil Nadu-calibrated dataset**, generated programmatically (see `generate_tn_dataset()` in Notebook 1):

| Feature | Description |
|---|---|
| `district` | One of 20 Tamil Nadu districts |
| `category` | TN admissions reservation category (OC / BC / MBC / SC / ST), population-weighted |
| `urban` | Urban (1) vs. rural (0) household |
| `family_income` | Annual household income (₹), log-normal, shifted by category and urban/rural |
| `first_gen`, `parent_grad` | First-generation college student flag; parental graduate status |
| `cutoff_12th`, `entrance_score` | Academic strength proxies (12th-std cutoff %, entrance/counselling score) |
| `college_tier` | Autonomous / Affiliated / Self-financing — drives base tuition |
| `tuition`, `distance_km`, `competing_offers` | Institutional and contextual factors |
| `merit_aid_pct`, `need_aid_pct`, `total_aid_pct`, `aid_amount`, `net_price` | The financial aid package |
| `enrolled` | **Target label** — generated from a price-sensitivity + affordability + merit logit, so it has a genuinely learnable relationship to the features |

To use real institutional data instead, load it into a DataFrame with the same column names — no other code needs to change.

## Repository Structure

```
.
├── Strategic_Financial_Aid_Yield_Optimizer_dataset_generator.ipynb   # Notebook 1
├── Strategic_Financial_Aid_Yield_Optimizer_train.ipynb               # Notebook 2
├── Strategic_Financial_Aid_Yield_Optimizer_test.ipynb                # Notebook 3
├── Strategic_Financial_Aid_Yield_Optimizer_endpoint.ipynb            # Notebook 4
└── README.md
```

Each notebook is self-contained and designed to run in **Google Colab (free CPU tier)** — no GPU, no paid APIs, no non-standard installs beyond what Colab ships with by default.

| Artifact | Produced by | Consumed by |
|---|---|---|
| `tn_synthetic_aid_dataset.csv` | Notebook 1 | Notebook 2 |
| `yield_model_pipeline.pkl` | Notebook 2 | Notebook 3, Notebook 4 |
| `yield_test_holdout.csv` | Notebook 2 | Notebook 3 |

## Requirements

- A Google account (for Google Colab)
- No local installation needed — everything runs in the Colab runtime
- Libraries used: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `flask`, `requests` — all preinstalled in Colab's default runtime

## Setup & Usage

Run the notebooks in order the **first** time. After that, only Notebook 4 (or Notebook 3) needs to be reopened for inference — no retraining required unless the data or model changes.

### 1. Dataset Generation
**Notebook:** `Strategic_Financial_Aid_Yield_Optimizer_dataset_generator.ipynb`

- Generates the synthetic Tamil Nadu applicant pool (default: 6,000 records)
- Saves `tn_synthetic_aid_dataset.csv` to `/content/`
- Runs basic sanity checks (missing values, class balance, yield rate by segment)

### 2. Model Training
**Notebook:** `Strategic_Financial_Aid_Yield_Optimizer_train.ipynb`

- Loads the CSV from Notebook 1
- Splits into train/test (80/20, stratified)
- Builds a `ColumnTransformer` (`StandardScaler` for numeric features, `OneHotEncoder` for categorical features) feeding a `RandomForestClassifier`, wrapped in a single `sklearn.pipeline.Pipeline`
- Evaluates on the held-out test set: **Accuracy ≈ 0.65, ROC-AUC ≈ 0.70** on the default synthetic dataset
- Serializes the trained pipeline with the standard library `pickle` module to `yield_model_pipeline.pkl`
- Saves the held-out test split separately as `yield_test_holdout.csv`, so later notebooks can evaluate on genuinely unseen data without retraining

### 3. Testing & Inference
**Notebook:** `Strategic_Financial_Aid_Yield_Optimizer_test.ipynb`

- Loads the pretrained pipeline (`pickle.load`) — no training happens here
- Runs a structured unit-test suite (see [Testing Strategy](#testing-strategy))
- Demonstrates single-applicant scoring, batch scoring, and the aid-package optimizer

### 4. REST API Endpoint
**Notebook:** `Strategic_Financial_Aid_Yield_Optimizer_endpoint.ipynb`

- Loads the pretrained pipeline from `/content/yield_model_pipeline.pkl`
- Starts a Flask server on a background thread inside Colab (`localhost:5000`)
- Exposes `GET /health` and `POST /predict`
- Persists every prediction as a JSON file directly under `/content/`
- Tests itself from within the notebook via the `requests` library — no external tunneling required
- Includes an optional, clearly-marked section for exposing the endpoint publicly via `ngrok`, for cases where an external client (Postman, curl, a frontend app) needs to reach it

## API Reference

### `GET /health`
Liveness check.

**Response — `200 OK`**
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`
Scores one applicant and returns their predicted enrollment probability.

**Request body** — JSON object with all of the following fields:

```json
{
  "district": "Madurai",
  "category": "MBC",
  "urban": 0,
  "family_income": 210000,
  "first_gen": 1,
  "parent_grad": 0,
  "cutoff_12th": 79.0,
  "entrance_score": 128.0,
  "college_tier": "Tier-2 (Affiliated)",
  "tuition": 110000,
  "distance_km": 35.0,
  "competing_offers": 1,
  "merit_aid_pct": 0.20,
  "need_aid_pct": 0.35,
  "total_aid_pct": 0.28,
  "aid_amount": 30800,
  "net_price": 79200
}
```

**Response — `200 OK`**
```json
{
  "yield_probability": 0.3799,
  "predicted_enrolled": false,
  "saved_to": "/content/prediction_20260917T094223353915.json"
}
```

**Response — `400 Bad Request`** (missing fields, wrong field types, or a non-object body)
```json
{ "error": "Missing required fields: ['urban', 'family_income', ...]" }
```

**Response — `500 Internal Server Error`** (unexpected inference failure)
```json
{ "error": "<exception message>" }
```

### Output persistence
Every successful `/predict` call writes its full input/output record to `/content/prediction_<timestamp>.json`, and overwrites `/content/latest_prediction.json` with the same record — giving both a complete audit trail and one fixed, known path to the most recent result.

## Model Details

| | |
|---|---|
| **Algorithm** | `RandomForestClassifier` (300 trees, max depth 10, min samples/leaf 5) |
| **Preprocessing** | `StandardScaler` (numeric features) + `OneHotEncoder` (categorical features), via `ColumnTransformer` |
| **Packaging** | Preprocessing and model bundled into one `sklearn.pipeline.Pipeline`, so the serialized artifact is self-contained |
| **Serialization** | Standard library `pickle` (`.pkl`) |
| **Held-out performance** | Accuracy ≈ 0.65, ROC-AUC ≈ 0.70 (on the default 6,000-row synthetic dataset) |

**Why a Random Forest:** tabular data, a few thousand rows, and a free-tier CPU runtime — a forest trains in seconds, produces usable `predict_proba` probabilities, and its feature importances double as an explainability artifact for an admissions committee. `GradientBoostingClassifier` or `xgboost` are reasonable upgrades once real institutional data is available and squeezing out marginal AUC matters more than training speed.

## Aid-Package Optimizer

Given one applicant, the optimizer (defined in Notebooks 3 and 4) grid-searches aid percentages from 0% to 80% and selects the package that maximizes:

```
expected_revenue = yield_probability(aid_pct) × net_price(aid_pct)
```

subject to:
- **`min_yield_prob`** — a floor on predicted enrollment probability
- **`equity_floor_pct`** — a minimum aid percentage guaranteed to applicants whose income-based need index exceeds a threshold, regardless of what pure revenue-maximization would otherwise select

This equity constraint exists because an unconstrained, revenue-only search systematically under-funds high-need applicants who would likely enroll even with less aid — the opposite of the diversity goal the system is meant to serve. Before trusting this optimizer's output on a real applicant pool, audit its recommended packages by `category` and `district` to confirm it isn't quietly reproducing existing funding gaps.

## Testing Strategy

Notebook 3 runs a structured suite of 8 tests directly against the pretrained pipeline:

| Test | Verifies |
|---|---|
| `test_bundle_integrity` | Saved artifact contains everything inference needs |
| `test_probability_output_validity` | `predict_proba` outputs are valid probabilities in `[0, 1]` |
| `test_batch_inference_shape` | Output length matches input row count |
| `test_holdout_auc_threshold` | Pretrained model clears a minimum AUC bar on unseen data |
| `test_new_unseen_applicant` | Model accepts a hand-written applicant not present in any CSV |
| `test_prediction_determinism` | Identical input produces identical output on repeat calls |
| `test_aid_direction_sanity` | Population-level average yield does not decrease as aid increases |
| `test_robust_to_extreme_inputs` | Plausible-but-extreme values don't crash inference |

Notebook 4's validation-test cell additionally exercises the API's error handling directly: missing fields, wrong field types, and a non-object request body, each confirmed to return `400` with a descriptive message rather than crashing.

## Known Limitations

- **Synthetic data**: all results, metrics, and example predictions in this project are based on generated data calibrated to be *realistic*, not on real institutional records. Model performance on real data will differ.
- **Development server**: the Flask app in Notebook 4 uses Werkzeug's built-in development server (explicitly warned against for production use). It's appropriate for prototyping and demonstration inside Colab, not for production traffic.
- **Session-scoped storage**: files saved to `/content/` do not persist once the Colab runtime is recycled. Download or copy artifacts (model file, prediction logs) elsewhere if they need to survive a session reset.
- **Non-monotonic model behavior**: as a tree ensemble, the model is not mathematically guaranteed to be monotonic in any single feature (e.g., more aid does not strictly guarantee a higher yield probability for every individual applicant) — `test_aid_direction_sanity` checks this at the population level rather than per-applicant for this reason.

## Future Work

- Replace the synthetic generator with real, anonymized admissions/aid/enrollment data
- Add authentication and rate limiting before exposing the endpoint beyond local testing
- Swap the development Flask server for a production WSGI server (e.g., Gunicorn) if deployed outside Colab
- Extend the optimizer to jointly allocate aid across an entire incoming class subject to a fixed total budget, rather than optimizing each applicant independently
- Add model monitoring (prediction drift, fairness metrics by category/district) if deployed against live data
