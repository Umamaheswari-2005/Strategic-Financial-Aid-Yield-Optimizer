# Strategic Financial Aid Yield Optimizer

A machine learning-based financial-aid and enrollment prediction system designed to estimate **student enrollment probability** from academic, financial, demographic, institutional, geographic, and financial-aid related characteristics.

The project supports two dataset environments:

* 🇮🇳 **Tamil Nadu / Indian Dataset**
* 🇺🇸 **US Financial Aid Dataset**

The project includes synthetic dataset generation, machine-learning model training, model testing, enrollment probability prediction, financial-aid scenario analysis, Flask REST API deployment, Redis-based prediction caching, and RabbitMQ-based asynchronous event processing.

> **Note:** The datasets used in this project are synthetic. They are intended for educational, research, prototyping, and system-development purposes and do not represent real student admission or financial-aid records.

---

# 📌 Project Overview

Educational institutions need to understand how different academic, financial, geographic, and aid-related factors may be associated with a student's enrollment decision.

A student's enrollment decision can depend on factors such as:

* Family or household financial situation
* Financial aid and scholarship support
* Academic performance
* Institutional characteristics
* Distance from the institution
* Student engagement
* Competing educational offers
* Residency status
* Demonstrated interest
* Net price after financial aid

The **Strategic Financial Aid Yield Optimizer** uses machine-learning models to estimate the probability that an admitted student will enroll.

The project is implemented as a modular workflow:

```text
Dataset Generation
        ↓
Data Preparation
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Testing
        ↓
Enrollment Probability Prediction
        ↓
Financial-Aid Scenario Analysis
        ↓
Flask REST API
        ↓
Redis Caching + RabbitMQ Messaging
```

---

# 🎯 Objectives

The main objectives of the project are:

* Generate synthetic financial-aid and enrollment datasets.
* Prepare applicant data for machine-learning models.
* Predict whether a student is likely to enroll.
* Estimate enrollment probability using `predict_proba()`.
* Evaluate trained models using classification metrics.
* Analyze important model features.
* Test the effect of different financial-aid levels on predicted enrollment probability.
* Provide applicant-level predictions through a Flask REST API.
* Cache prediction results using Redis.
* Publish prediction events through RabbitMQ.
* Process prediction events asynchronously using a RabbitMQ consumer.
* Maintain separate workflows for Tamil Nadu and US datasets.

---

# 🏗️ Project Architecture

```text
                         ┌───────────────────────────┐
                         │    Synthetic Dataset      │
                         │       Generation          │
                         └─────────────┬─────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  │                                         │
                  ▼                                         ▼
       ┌──────────────────────┐                 ┌──────────────────────┐
       │ Tamil Nadu Dataset   │                 │ US Dataset           │
       │ 6,000 × 18           │                 │ 6,000 × 25           │
       └──────────┬───────────┘                 └──────────┬───────────┘
                  │                                        │
                  ▼                                        ▼
       ┌──────────────────────┐                 ┌──────────────────────┐
       │ Random Forest Model  │                 │ XGBoost Model        │
       │ + preprocessing      │                 │ + model-ready data   │
       └──────────┬───────────┘                 └──────────┬───────────┘
                  │                                        │
                  └────────────────┬───────────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │ Model Testing        │
                         │ & Validation         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Enrollment           │
                         │ Probability          │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
             ┌────────────┐ ┌──────────────┐ ┌──────────────┐
             │ Flask API  │ │    Redis     │ │  RabbitMQ    │
             │ /predict   │ │    Cache     │ │   Queue      │
             └────────────┘ └──────────────┘ └──────┬───────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │   Consumer   │
                                            │ Event Process│
                                            └──────────────┘
```

---

# 📂 Project Structure

```text
Strategic-Financial-Aid-Yield-Optimizer/
│
├── Strategic_Financial_Aid_Yield_Optimizer_dataset_generator.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_train.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_test.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_endpoint.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_end_to_end.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_rabbitmq.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_redis.ipynb
│
├── tn_synthetic_aid_dataset.csv
├── us_financial_aid_yield_dataset.csv
├── us_dataset_model_ready.csv
│
├── yield_model_pipeline.pkl
│
└── README.md
```

The project is divided into separate notebooks so that dataset generation, model training, testing, API deployment, caching, and message processing can be developed and tested independently.

---

# 📓 Project Notebooks

## 1. Dataset Generation

### Notebook

```text
Strategic_Financial_Aid_Yield_Optimizer_dataset_generator.ipynb
```

This notebook contains synthetic dataset generation for both the Tamil Nadu and US environments.

The notebook contains separate dataset-generation sections for:

* Tamil Nadu / Indian dataset
* US financial-aid dataset

---

# 🇮🇳 Tamil Nadu Dataset

The Tamil Nadu synthetic dataset contains:

```text
Records : 6,000
Columns : 18
```

### Dataset File

```text
tn_synthetic_aid_dataset.csv
```

### Dataset Columns

| Column             | Description                        |
| ------------------ | ---------------------------------- |
| `district`         | Applicant's district               |
| `category`         | Student category                   |
| `urban`            | Urban/non-urban indicator          |
| `family_income`    | Family income                      |
| `first_gen`        | First-generation student indicator |
| `parent_grad`      | Parent graduation indicator        |
| `cutoff_12th`      | 12th-standard academic score       |
| `entrance_score`   | Entrance examination score         |
| `college_tier`     | Institutional/college tier         |
| `tuition`          | Tuition amount                     |
| `distance_km`      | Distance from institution          |
| `competing_offers` | Number of competing offers         |
| `merit_aid_pct`    | Merit-based aid percentage         |
| `need_aid_pct`     | Need-based aid percentage          |
| `total_aid_pct`    | Total aid percentage               |
| `aid_amount`       | Financial-aid amount               |
| `net_price`        | Amount remaining after aid         |
| `enrolled`         | Enrollment target variable         |

The target variable is:

```text
enrolled
```

where:

```text
0 = Not Enrolled
1 = Enrolled
```

---

# 🇺🇸 US Financial Aid Dataset

The raw US dataset contains:

```text
Records : 6,000
Columns : 25
```

### Dataset File

```text
us_financial_aid_yield_dataset.csv
```

### Raw Dataset Columns

| Column                        | Description                          |
| ----------------------------- | ------------------------------------ |
| `state_residency`             | Applicant's state of residence       |
| `is_in_state`                 | In-state applicant indicator         |
| `urban_centric_locale`        | Geographic/locale category           |
| `student_aid_index`           | Student financial-aid index          |
| `adjusted_gross_income`       | Adjusted gross income                |
| `first_gen`                   | First-generation indicator           |
| `pell_eligible`               | Pell eligibility indicator           |
| `hs_gpa`                      | High-school GPA                      |
| `sat_act_percentile`          | SAT/ACT percentile                   |
| `institutional_tier`          | Institution classification           |
| `cost_of_attendance`          | Cost of attendance                   |
| `miles_from_campus`           | Distance from campus                 |
| `fafsa_submitted_month`       | FAFSA submission month               |
| `fafsa_month_sin`             | Sine transformation of FAFSA month   |
| `fafsa_month_cos`             | Cosine transformation of FAFSA month |
| `demonstrated_interest`       | Demonstrated student interest        |
| `merit_scholarship_amt`       | Merit scholarship amount             |
| `need_grant_amt`              | Need-based grant amount              |
| `total_aid_package`           | Total financial-aid package          |
| `net_price`                   | Net price after aid                  |
| `enrolled`                    | Enrollment target                    |
| `net_price_to_income_ratio`   | Net price relative to income         |
| `financial_aid_discount_rate` | Financial-aid discount rate          |
| `unmet_financial_need_gap`    | Remaining financial-need gap         |
| `engagement_velocity`         | Student engagement measure           |

---

# 📊 US Model-Ready Dataset

The project also contains a processed US dataset:

```text
us_dataset_model_ready.csv
```

It contains:

```text
Records : 6,000
Columns : 24
```

The model-ready dataset removes raw fields that are not directly used by the trained model and represents categorical information in numerical form.

### Model-Ready Features

```text
is_in_state
student_aid_index
adjusted_gross_income
first_gen
pell_eligible
hs_gpa
sat_act_percentile
institutional_tier
cost_of_attendance
miles_from_campus
fafsa_month_sin
fafsa_month_cos
demonstrated_interest
merit_scholarship_amt
need_grant_amt
net_price
enrolled
net_price_to_income_ratio
financial_aid_discount_rate
unmet_financial_need_gap
engagement_velocity
urban_centric_locale_Rural
urban_centric_locale_Suburb
urban_centric_locale_Town
```

The target variable remains:

```text
enrolled
```

---

# 🤖 2. Model Training

### Notebook

```text
Strategic_Financial_Aid_Yield_Optimizer_train.ipynb
```

The training notebook contains model-training workflows for the project datasets.

---

## 🇮🇳 Tamil Nadu Model

The Tamil Nadu model uses:

```text
Random Forest Classifier
```

### Training Workflow

```text
Load Dataset
     ↓
Select Features
     ↓
Separate Features and Target
     ↓
80/20 Train-Test Split
     ↓
Numerical Preprocessing
     ↓
Categorical Preprocessing
     ↓
Random Forest Training
     ↓
Prediction
     ↓
Model Evaluation
     ↓
Feature Importance
     ↓
Save Model
```

### Features

Numerical features include:

```text
urban
family_income
first_gen
parent_grad
cutoff_12th
entrance_score
tuition
distance_km
competing_offers
merit_aid_pct
need_aid_pct
total_aid_pct
aid_amount
net_price
```

Categorical features include:

```text
district
category
college_tier
```

### Preprocessing

Numerical features use:

```text
StandardScaler
```

Categorical features use:

```text
OneHotEncoder
```

with:

```text
handle_unknown = "ignore"
```

### Random Forest Configuration

```text
n_estimators     = 300
max_depth        = 10
min_samples_leaf = 5
random_state     = 42
n_jobs            = -1
```

The preprocessing and classifier are combined into a single scikit-learn `Pipeline`.

---

# 🇺🇸 US Model

The US model uses:

```text
XGBoost Classifier
```

The US dataset is already converted into a model-ready numerical representation before model training.

### US Model Configuration

```text
n_estimators     = 300
max_depth        = 5
learning_rate    = 0.05
subsample        = 0.8
colsample_bytree = 0.8
eval_metric      = logloss
random_state     = 42
n_jobs            = -1
```

The model is stored inside a scikit-learn `Pipeline` containing:

```text
Preprocessor
     ↓
XGBoost Classifier
```

Since the US model-ready features are numerical/binary, no additional one-hot encoding or feature scaling is required during model training.

---

# 📈 Model Evaluation

The training workflow evaluates model performance using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification report
* ROC curve where applicable
* Feature importance

The model also produces enrollment probabilities using:

```python
predict_proba()
```

The probability for class `1` represents the estimated enrollment probability.

---

# 💾 Trained Model

The trained pipeline is saved as:

```text
yield_model_pipeline.pkl
```

The exact object structure depends on the model-training workflow.

### Tamil Nadu Model

The Tamil Nadu training workflow saves a bundle containing:

```text
pipeline
num_feats
cat_feats
target
trained_on
metrics
notes
```

### US Model

The US training workflow saves the trained pipeline directly:

```text
Pipeline
 ├── preprocessor
 └── classifier
```

This distinction is important when loading the model. The US pipeline should be loaded directly rather than attempting to access:

```python
bundle['pipeline']
```

---

# 🧪 3. Model Testing

### Notebook

```text
Strategic_Financial_Aid_Yield_Optimizer_test.ipynb
```

The testing notebook contains separate testing workflows for the Tamil Nadu and US models.

---

## Tamil Nadu Model Testing

The Tamil Nadu testing workflow:

```text
Load Model
     ↓
Load Holdout Dataset
     ↓
Check Model Bundle
     ↓
Validate Feature Configuration
     ↓
Run Predictions
     ↓
Evaluate Test Results
```

The testing workflow verifies the trained model and its required feature configuration before performing inference.

---

## US Model Testing

The US testing workflow loads the trained XGBoost pipeline directly.

The model's expected feature columns are obtained from:

```python
pipeline.feature_names_in_
```

The testing workflow reproduces the transformations used during model preparation.

### US preprocessing includes:

* Institutional-tier mapping
* Locale one-hot representation
* Net-price-to-income transformation
* Exact feature-column alignment

---

# 💰 Financial-Aid Scenario Analysis

The US model-testing notebook includes an aid-level scenario analysis.

Example aid levels include:

```text
$0
$5,000
$10,000
$15,000
$20,000
$25,000
```

For each aid level, the system calculates:

```text
Cost of Attendance
        ↓
Financial Aid
        ↓
Net Price
        ↓
Enrollment Probability
```

This allows the project to examine how predicted enrollment probability changes as financial aid increases.

The testing workflow evaluates the resulting price-sensitivity curve and checks whether predicted enrollment probability generally increases with additional aid, allowing a small tolerance for model variation.

---

# 🌐 4. Flask REST API

### Notebook

```text
Strategic_Financial_Aid_Yield_Optimizer_endpoint.ipynb
```

The Flask notebook exposes the trained model through HTTP endpoints.

### API Architecture

```text
Client
  │
  │ JSON Applicant Data
  ▼
┌─────────────────────────┐
│      Flask API          │
│                         │
│      /health            │
│      /predict           │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│     ML Pipeline         │
│                         │
│  Preprocessing          │
│       +                 │
│  Classifier             │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Enrollment Prediction   │
│ + Probability           │
└─────────────────────────┘
```

---

# 🔎 API Endpoints

## Health Check

```http
GET /health
```

The endpoint checks whether the Flask API is running and whether the model has been loaded.

Example:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## Prediction

```http
POST /predict
```

The endpoint accepts applicant information in JSON format and returns a predicted enrollment outcome and enrollment probability.

---

# 🛡️ API Validation

The API validates incoming applicant data.

### Missing Fields

If required applicant fields are missing:

```text
HTTP 400 Bad Request
```

is returned.

### Invalid Numeric Fields

If a field expected to be numeric contains invalid data, the API returns:

```text
HTTP 400 Bad Request
```

### Invalid JSON Structure

If the request body is not a JSON object, the request is rejected.

---

# 💾 Prediction Output

The Flask API saves prediction results to JSON files in the Colab filesystem.

The API creates:

```text
prediction_<timestamp>.json
```

and also maintains:

```text
latest_prediction.json
```

Each saved prediction contains:

```text
timestamp
input
output
```

This provides a persistent record of predictions during the active Colab session.

---

# ⚡ 5. Redis Integration

### Notebook

```text
Strategic_Financial_Aid_Yield_Optimizer_redis.ipynb
```

Redis is used as a caching layer for prediction results.

### Redis Workflow

```text
Prediction Request
       ↓
Generate Cache Key
       ↓
Check Redis
       ↓
Cache Hit ──────→ Return Cached Result
       │
       └─ Cache Miss
              ↓
        Run ML Prediction
              ↓
        Store Result in Redis
              ↓
        Return Prediction
```

The Redis implementation demonstrates:

* Connection
* `SET`
* `GET`
* Update
* Delete
* Expiration/TTL

A prediction cache entry can be configured with a time-to-live so that old prediction results are automatically removed.

### Example Cache Key

```text
yield:demo_applicant_001
```

---

# 📨 6. RabbitMQ Integration

### Notebook

```text
Strategic_Financial_Aid_Yield_Optimizer_rabbitmq.ipynb
```

RabbitMQ is used as a message broker for prediction events.

### RabbitMQ Workflow

```text
Flask / Prediction Service
          │
          ▼
    Publish Event
          │
          ▼
┌──────────────────────┐
│ RabbitMQ Queue       │
│                      │
│ yield_prediction_    │
│ queue                │
└──────────┬───────────┘
           │
           ▼
     RabbitMQ Consumer
           │
           ▼
    Process Event
```

The queue used by the project is:

```text
yield_prediction_queue
```

Messages are published as JSON and consumed by a background consumer.

The consumer acknowledges messages after processing them successfully.

---

# 🔄 7. End-to-End Integration

### Notebook

```text
Strategic_Financial_Aid_Yield_Optimizer_end_to_end.ipynb
```

The end-to-end notebook combines the major infrastructure components.

It demonstrates:

* Model loading
* Redis connection
* RabbitMQ connection
* Prediction processing
* Prediction caching
* Event publishing
* RabbitMQ consumption
* Flask API
* Health checking
* Applicant prediction

### End-to-End Flow

```text
                Applicant
                    │
                    ▼
             Flask /predict
                    │
                    ▼
            Validate Request
                    │
                    ▼
              Redis Cache
             ┌──────┴──────┐
             │             │
          Cache Hit     Cache Miss
             │             │
             │             ▼
             │       ML Prediction
             │             │
             │             ▼
             │       Store in Redis
             │             │
             └──────┬──────┘
                    │
                    ▼
             Prediction Result
                    │
                    ▼
             RabbitMQ Event
                    │
                    ▼
               Queue
                    │
                    ▼
                Consumer
                    │
                    ▼
             Event Processing
```

---

# 🧩 Tamil Nadu and US Workflows

The project maintains different model pipelines for the two datasets.

| Component            | Tamil Nadu                          | US                                    |
| -------------------- | ----------------------------------- | ------------------------------------- |
| Dataset              | `tn_synthetic_aid_dataset.csv`      | `us_financial_aid_yield_dataset.csv`  |
| Records              | 6,000                               | 6,000                                 |
| Raw Columns          | 18                                  | 25                                    |
| Model-Ready Dataset  | Not separate                        | `us_dataset_model_ready.csv`          |
| Algorithm            | Random Forest                       | XGBoost                               |
| Numerical Scaling    | StandardScaler                      | Not required                          |
| Categorical Encoding | OneHotEncoder                       | Preprocessed before training          |
| Target               | `enrolled`                          | `enrolled`                            |
| Output               | Enrollment prediction + probability | Enrollment prediction + probability   |
| Testing              | Holdout/model validation            | Scenario + batch + robustness testing |
| API                  | Flask                               | Flask                                 |
| Cache                | Redis                               | Redis                                 |
| Messaging            | RabbitMQ                            | RabbitMQ                              |

---

# 🧪 US Model Testing Scenarios

The US testing notebook contains multiple inference scenarios.

### Test 1 — High-Need Applicant

A high-need applicant with strong financial support and demonstrated interest is evaluated for enrollment probability.

### Test 2 — High-Price Applicant

An applicant with:

* High net price
* Lower engagement
* Greater distance
* Higher financial resources

is evaluated for predicted enrollment probability.

### Test 3 — Aid Sensitivity

Multiple aid levels are tested to generate a price-sensitivity curve.

```text
Aid
 ↓
Net Price
 ↓
Predicted Enrollment Probability
```

### Test 4 — Missing Value

The model is tested with a missing SAT/ACT percentile value.

### Test 5 — Batch Prediction

Multiple applicant records are passed together to verify that the model returns one prediction and probability for each input row.

---

# 🛠️ Technologies Used

| Technology     | Purpose                                                |
| -------------- | ------------------------------------------------------ |
| Python         | Core programming language                              |
| Pandas         | Data loading and manipulation                          |
| NumPy          | Numerical operations and synthetic data generation     |
| Scikit-learn   | Preprocessing, pipelines, Random Forest and evaluation |
| Random Forest  | Tamil Nadu enrollment classification                   |
| XGBoost        | US enrollment classification                           |
| StandardScaler | Numerical feature preprocessing                        |
| OneHotEncoder  | Tamil Nadu categorical feature encoding                |
| Matplotlib     | Model evaluation and visualization                     |
| Flask          | REST API development                                   |
| Requests       | API testing                                            |
| Redis          | Prediction caching                                     |
| RabbitMQ       | Asynchronous prediction-event messaging                |
| Pika           | Python RabbitMQ client                                 |
| Pickle         | Model serialization                                    |
| Google Colab   | Development and execution environment                  |

---

# 📦 Required Python Packages

Install the main Python dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib flask requests redis pika xgboost
```

For Google Colab, XGBoost can also be installed with:

```python
!pip install -q xgboost
```

The infrastructure notebooks install the required Redis/RabbitMQ Python clients as needed.

---

# 🚀 Getting Started

## 1. Open Google Colab

The notebooks are designed to run in a Google Colab environment.

Recommended order:

```text
1. Dataset Generation
        ↓
2. Model Training
        ↓
3. Model Testing
        ↓
4. Redis Setup
        ↓
5. RabbitMQ Setup
        ↓
6. Flask Endpoint
        ↓
7. End-to-End Integration
```

---

# 2. Generate the Dataset

Open:

```text
Strategic_Financial_Aid_Yield_Optimizer_dataset_generator.ipynb
```

Run the appropriate dataset-generation section.

The resulting files are:

```text
tn_synthetic_aid_dataset.csv
us_financial_aid_yield_dataset.csv
```

The US workflow additionally produces the model-ready dataset:

```text
us_dataset_model_ready.csv
```

---

# 3. Train the Model

Open:

```text
Strategic_Financial_Aid_Yield_Optimizer_train.ipynb
```

For the Tamil Nadu dataset, train the Random Forest pipeline.

For the US dataset, train the XGBoost pipeline using the model-ready dataset.

The trained model is saved as:

```text
yield_model_pipeline.pkl
```

---

# 4. Test the Model

Open:

```text
Strategic_Financial_Aid_Yield_Optimizer_test.ipynb
```

Upload the appropriate trained model and required testing dataset.

Run the test cells to verify:

* Model loading
* Feature compatibility
* Predictions
* Enrollment probabilities
* Batch inference
* Scenario analysis
* Model robustness

---

# 5. Start Redis

Open:

```text
Strategic_Financial_Aid_Yield_Optimizer_redis.ipynb
```

The notebook demonstrates starting Redis and performing cache operations.

For a local environment, the default Redis port is:

```text
6379
```

---

# 6. Start RabbitMQ

Open:

```text
Strategic_Financial_Aid_Yield_Optimizer_rabbitmq.ipynb
```

The default AMQP port is:

```text
5672
```

The project uses:

```text
yield_prediction_queue
```

for prediction events.

> For deployment, configure RabbitMQ credentials using secure environment variables or a secret-management system. Do not commit credentials to source control.

---

# 7. Run the Flask API

Open:

```text
Strategic_Financial_Aid_Yield_Optimizer_endpoint.ipynb
```

Load the trained model and start the Flask application.

The API runs on:

```text
http://127.0.0.1:5000
```

Health endpoint:

```text
GET /health
```

Prediction endpoint:

```text
POST /predict
```

---

# 🧪 API Example — Tamil Nadu

Example applicant structure:

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

Send the request to:

```text
POST http://127.0.0.1:5000/predict
```

---

# 🧮 US Model Input

The US model expects the model-ready numerical features.

The model-ready columns include:

```text
is_in_state
student_aid_index
adjusted_gross_income
first_gen
pell_eligible
hs_gpa
sat_act_percentile
institutional_tier
cost_of_attendance
miles_from_campus
fafsa_month_sin
fafsa_month_cos
demonstrated_interest
merit_scholarship_amt
need_grant_amt
net_price
net_price_to_income_ratio
financial_aid_discount_rate
unmet_financial_need_gap
engagement_velocity
urban_centric_locale_Rural
urban_centric_locale_Suburb
urban_centric_locale_Town
```

The US testing notebook contains helper functions that convert raw applicant information into the exact model-ready structure before inference.

---

# 🔐 Configuration and Security

The Redis and RabbitMQ notebooks contain development connection configuration.

For actual deployment:

* Do not commit passwords to GitHub.
* Do not place production credentials directly inside notebooks.
* Use environment variables.
* Use a secrets manager where appropriate.
* Enable TLS for production connections where required.
* Restrict Redis and RabbitMQ network access.
* Rotate credentials if they have been exposed.

Example:

```python
import os

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USERNAME = os.getenv("RABBITMQ_USERNAME")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
```

---

# ⚠️ Model Compatibility

The serialized model should preferably be loaded using versions of Python and machine-learning libraries compatible with the environment in which the model was trained.

In particular, the following should be kept consistent when possible:

```text
Python
scikit-learn
XGBoost
NumPy
```

A version mismatch can produce warnings or, in some cases, inference problems when loading a Pickle model.

---

# 🧩 Project Challenges and Solutions

| Challenge                                                   | Solution                                                            |
| ----------------------------------------------------------- | ------------------------------------------------------------------- |
| Real student financial-aid data was unavailable             | Synthetic datasets were generated                                   |
| Tamil Nadu and US datasets have different schemas           | Separate feature-processing and model workflows were implemented    |
| Numerical and categorical data require different processing | Scikit-learn preprocessing pipelines are used                       |
| US model requires model-ready numerical features            | A separate model-ready US dataset is created                        |
| Different models are suitable for different workflows       | Random Forest is used for Tamil Nadu and XGBoost for the US dataset |
| Model reuse is required after training                      | Trained pipelines are serialized using Pickle                       |
| API requests may contain invalid input                      | Flask validation checks required fields and data types              |
| Repeated predictions can require caching                    | Redis is used to store prediction results                           |
| Prediction processing can be asynchronous                   | RabbitMQ is used for prediction events                              |
| Multiple API requests need to be processed                  | Batch inference is tested in the model-testing workflow             |
| Different aid levels can produce different predictions      | US testing includes aid-level scenario analysis                     |

---

# 📈 Key Features

* ✅ Synthetic financial-aid dataset generation
* ✅ Tamil Nadu applicant dataset
* ✅ US financial-aid dataset
* ✅ US model-ready dataset
* ✅ 6,000-record datasets
* ✅ Enrollment prediction
* ✅ Enrollment probability estimation
* ✅ Tamil Nadu Random Forest model
* ✅ US XGBoost model
* ✅ Numerical feature preprocessing
* ✅ Categorical feature encoding
* ✅ Train/test data separation
* ✅ Model evaluation
* ✅ Accuracy measurement
* ✅ Precision, recall and F1-score
* ✅ ROC-AUC evaluation
* ✅ Feature-importance analysis
* ✅ US financial-aid scenario analysis
* ✅ Batch inference
* ✅ Missing-value inference testing
* ✅ Flask REST API
* ✅ `/health` endpoint
* ✅ `/predict` endpoint
* ✅ JSON request validation
* ✅ Prediction JSON file storage
* ✅ Redis prediction caching
* ✅ RabbitMQ message queue
* ✅ Background RabbitMQ consumer
* ✅ End-to-end prediction workflow
* ✅ Modular Google Colab notebooks

---

# 🔄 Complete End-to-End Workflow

```text
                   DATA GENERATION
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   Tamil Nadu Dataset              US Dataset
    6000 × 18                      6000 × 25
          │                             │
          │                       Model Preparation
          │                             │
          │                             ▼
          │                      US Model-Ready Data
          │                        6000 × 24
          │                             │
          ▼                             ▼
   Random Forest                    XGBoost
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  MODEL TESTING
                         │
                         ▼
             ENROLLMENT PROBABILITY
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
       Financial-Aid            Applicant
        Scenarios               Prediction
             │                       │
             └───────────┬───────────┘
                         ▼
                    FLASK API
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
           Redis                 RabbitMQ
           Cache                  Queue
             │                       │
             │                       ▼
             │                   Consumer
             │                       │
             └───────────┬───────────┘
                         ▼
                  Prediction Result
```

---

# ⚠️ Important Limitations

This project is a prototype based on synthetic data.

The generated datasets do not represent actual:

* Student admissions
* Financial-aid applications
* Institutional enrollment behavior
* Student financial records
* Real-world demographic distributions

Therefore, model predictions should not be interpreted as validated real-world enrollment probabilities.

Before using a system of this type in a production educational environment, it would require:

* Real and appropriately governed institutional data
* Data-quality validation
* Model validation
* Bias and fairness assessment
* Privacy and security controls
* Model monitoring
* Institutional review
* Appropriate human oversight

The current project demonstrates the machine-learning and software architecture required for a financial-aid yield prediction prototype.

---

# 📁 Data Files

## Tamil Nadu

```text
tn_synthetic_aid_dataset.csv
```

Synthetic Tamil Nadu financial-aid and enrollment dataset.

```text
6,000 rows × 18 columns
```

## US Raw Dataset

```text
us_financial_aid_yield_dataset.csv
```

Synthetic US financial-aid and enrollment dataset.

```text
6,000 rows × 25 columns
```

## US Model-Ready Dataset

```text
us_dataset_model_ready.csv
```

Processed US dataset used for model training.

```text
6,000 rows × 24 columns
```

## Trained Model

```text
yield_model_pipeline.pkl
```

Serialized machine-learning pipeline used for inference.

---

# 📚 Notebook Summary

| Notebook                                                          | Purpose                                                                       |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `Strategic_Financial_Aid_Yield_Optimizer_dataset_generator.ipynb` | Generates Tamil Nadu and US synthetic datasets                                |
| `Strategic_Financial_Aid_Yield_Optimizer_train.ipynb`             | Trains the Tamil Nadu Random Forest and US XGBoost workflows                  |
| `Strategic_Financial_Aid_Yield_Optimizer_test.ipynb`              | Tests the trained models and performs inference/scenario tests                |
| `Strategic_Financial_Aid_Yield_Optimizer_endpoint.ipynb`          | Provides Flask REST API endpoints                                             |
| `Strategic_Financial_Aid_Yield_Optimizer_redis.ipynb`             | Demonstrates Redis caching                                                    |
| `Strategic_Financial_Aid_Yield_Optimizer_rabbitmq.ipynb`          | Demonstrates RabbitMQ message publishing and consumption                      |
| `Strategic_Financial_Aid_Yield_Optimizer_end_to_end.ipynb`        | Combines model inference, API, Redis and RabbitMQ into an integrated workflow |

---

# 👩‍💻 Author

**Umamaheswari G**

AI/ML Developer | Machine Learning | Generative AI | Python

---

# 📄 License

This project is intended for:

* Educational purposes
* Research purposes
* Machine-learning experimentation
* Prototype development

The synthetic datasets and prediction system should not be treated as a production financial-aid decision-making system without additional validation, governance, privacy protection, and institutional review.
