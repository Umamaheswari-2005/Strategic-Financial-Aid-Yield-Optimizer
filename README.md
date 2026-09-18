# Strategic Financial Aid Yield Optimizer

A machine learning-based system for predicting **student enrollment probability** and supporting **financial aid optimization** by considering academic, financial, demographic, institutional, and competing-offer factors.

The project uses a **Random Forest Classifier** to estimate the probability that an admitted student will enroll and evaluates different financial-aid scenarios to support decisions that balance **enrollment, expected revenue, and support for students with higher financial need**.

---

## 📌 Project Overview

Educational institutions need to balance two important objectives:

* Increasing student enrollment and yield
* Maintaining financial sustainability while supporting students with higher financial need

A fixed financial-aid percentage may not be suitable for every student because applicants have different academic, financial, demographic, and institutional characteristics.

The **Strategic Financial Aid Yield Optimizer** addresses this challenge by:

1. Generating and preparing a synthetic student dataset.
2. Preprocessing numerical and categorical applicant features.
3. Training a Random Forest classification model.
4. Predicting enrollment probability.
5. Simulating different financial-aid levels.
6. Estimating net price and expected revenue.
7. Applying enrollment, revenue, and equity-related constraints.
8. Providing applicant-level predictions through a Flask REST API.

> **Note:** A synthetic dataset is used because real student admission and financial-aid data is confidential and was not available for this prototype.

---

## 🎯 Objectives

The primary objectives of this project are:

* Predict whether an admitted student is likely to enroll.
* Estimate the probability of enrollment for individual applicants.
* Identify factors that contribute to enrollment decisions.
* Simulate different financial-aid scenarios.
* Estimate the effect of aid on net price and expected revenue.
* Support higher-need students through equity-related constraints.
* Provide a reusable machine-learning pipeline.
* Expose the trained model through a REST API for application integration.

---

## 🏗️ Project Architecture

```text
                    ┌───────────────────────────┐
                    │   Synthetic Dataset       │
                    │   Generation Notebook     │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Dataset                 │
                    │   tn_synthetic_aid_       │
                    │   dataset.csv              │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Model Training Notebook │
                    │                           │
                    │ • Preprocessing           │
                    │ • Train/Test Split        │
                    │ • Random Forest           │
                    │ • Evaluation              │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │   Trained ML Pipeline     │
                    │   yield_model_pipeline.pkl│
                    └─────────────┬─────────────┘
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
              ┌──────────────────┐  ┌──────────────────┐
              │ Model Testing    │  │ Flask REST API   │
              │ Notebook         │  │ Endpoint         │
              └──────────────────┘  └────────┬─────────┘
                                             │
                                             ▼
                                  ┌────────────────────┐
                                  │ Applicant JSON     │
                                  │ Prediction         │
                                  │ + Probability      │
                                  └────────────────────┘
```

---

## 📂 Project Structure

```text
Strategic-Financial-Aid-Yield-Optimizer/
│
├── Strategic_Financial_Aid_Yield_Optimizer_dataset_generator.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_train.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_test.ipynb
├── Strategic_Financial_Aid_Yield_Optimizer_endpoint.ipynb
│
├── tn_synthetic_aid_dataset.csv
├── yield_model_pipeline.pkl
│
├── README.md
└── Documentation/
```

The project is divided into separate notebooks so that **dataset generation, model training, model testing, and API deployment** can be developed and executed independently.

---

# 📓 Google Colab Notebooks

## 1. Synthetic Dataset Generation

**Notebook:** `Strategic_Financial_Aid_Yield_Optimizer_dataset_generator.ipynb`

This notebook is responsible for creating the synthetic student dataset using **Python, NumPy, and Pandas**.

The generated dataset contains:

* **6,000 student records**
* **18 columns**

The dataset represents academic, financial, demographic, institutional, aid-related, and enrollment information.

### Dataset File

```text
tn_synthetic_aid_dataset.csv
```

### Notebook

[Open Dataset Generator in Google Colab](https://colab.research.google.com/drive/1i0nXk5Xb9OvnfmCihBkVGXc1uPlrpPd8?usp=sharing)

---

# 📊 Dataset Description

The dataset contains the following fields:

| Field              | Description                          | Purpose                                                 |
| ------------------ | ------------------------------------ | ------------------------------------------------------- |
| `district`         | Applicant's district                 | Represents geographical/location-related information    |
| `category`         | Student category                     | Represents demographic/category-related characteristics |
| `urban`            | Urban/non-urban indicator            | Represents location and accessibility context           |
| `family_income`    | Family income                        | Helps represent financial affordability and aid need    |
| `first_gen`        | First-generation student indicator   | Represents educational background                       |
| `parent_grad`      | Parent graduation indicator          | Represents parental educational background              |
| `cutoff_12th`      | 12th-standard academic cutoff/score  | Represents academic performance                         |
| `entrance_score`   | Entrance examination score           | Represents academic merit                               |
| `college_tier`     | College institutional tier           | Represents institutional characteristics                |
| `tuition`          | Tuition amount                       | Used to represent the cost before financial aid         |
| `distance_km`      | Distance from student to institution | Represents accessibility and travel considerations      |
| `competing_offers` | Number of competing college offers   | Represents alternative enrollment opportunities         |
| `merit_aid_pct`    | Merit-based aid percentage           | Represents aid awarded based on academic merit          |
| `need_aid_pct`     | Need-based aid percentage            | Represents financial-need-based support                 |
| `total_aid_pct`    | Total aid percentage                 | Represents the overall aid percentage                   |
| `aid_amount`       | Financial aid amount                 | Represents the monetary value of aid                    |
| `net_price`        | Tuition after financial aid          | Represents the effective amount paid after aid          |
| `enrolled`         | Enrollment target variable           | `0 = Not Enrolled`, `1 = Enrolled`                      |

The target variable `enrolled` is generated using a probability-based approach influenced by factors including affordability, academic merit, competing offers, distance, and other student-related factors.

---

# 🤖 2. Model Training

**Notebook:** `Strategic_Financial_Aid_Yield_Optimizer_train.ipynb`

This notebook loads the generated dataset and performs the complete model-training workflow.

### Training Workflow

```text
Load Dataset
      ↓
Data Validation
      ↓
Feature Selection
      ↓
Separate Features & Target
      ↓
Train/Test Split
      ↓
Feature Preprocessing
      ↓
Random Forest Training
      ↓
Model Evaluation
      ↓
Probability Prediction
      ↓
Feature Importance
      ↓
Save Trained Pipeline
```

### Train/Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

Stratified sampling is used to maintain a similar distribution of enrolled and non-enrolled students across the training and testing datasets.

### Preprocessing

The preprocessing pipeline handles:

* **Numerical features** → `StandardScaler`
* **Categorical features** → `OneHotEncoder`

This ensures that the different feature types are converted into a format suitable for machine-learning training.

### Machine Learning Algorithm

The project uses a:

**Random Forest Classifier**

Configuration:

```text
Number of Trees      : 300
Maximum Depth        : 10
Minimum Leaf Size    : 5
```

The Random Forest model is trained using the preprocessed numerical and categorical features.

### Model Evaluation

The model is evaluated using:

* Accuracy
* ROC-AUC
* Classification Report
* ROC Curve

These metrics are used to assess how effectively the model predicts student enrollment.

### Enrollment Probability

The model uses:

```python
predict_proba()
```

to generate the estimated probability that an admitted student will enroll.

Feature-importance analysis is also performed to identify features contributing to the model's predictions.

### Saved Model

The trained pipeline is serialized using Pickle:

```text
yield_model_pipeline.pkl
```

### Notebook

[Open Model Training Notebook in Google Colab](https://colab.research.google.com/drive/14oMuVLanayiex6UrsG5OsTvssNRZyri8?usp=sharing)

---

# 🧪 3. Model Testing

**Notebook:** `Strategic_Financial_Aid_Yield_Optimizer_test.ipynb`

The testing notebook is used to verify the trained machine-learning pipeline independently from the training process.

### Testing Workflow

```text
Load Pretrained Pipeline
          ↓
Prepare Applicant Data
          ↓
Run Prediction
          ↓
Generate Enrollment Probability
          ↓
Validate Output
```

The notebook demonstrates how a new applicant profile can be passed to the trained model to obtain an enrollment prediction and probability.

### Notebook

[Open Model Testing Notebook in Google Colab](https://colab.research.google.com/drive/1D_V0oWqsJlrVuvIgNaMeHVe3gn0a3uw4?usp=sharing)

---

# 🌐 4. Flask REST API Endpoint

**Notebook:** `Strategic_Financial_Aid_Yield_Optimizer_endpoint.ipynb`

A Flask-based REST API is implemented to make the trained ML pipeline accessible through an API endpoint.

### API Workflow

```text
Client Application
       │
       │ JSON Applicant Data
       ▼
┌─────────────────────┐
│ Flask REST API      │
│                     │
│ /predict            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ML Pipeline         │
│                     │
│ Preprocessing       │
│ + Random Forest     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Prediction          │
│ + Probability       │
└─────────────────────┘
```

## API Endpoints

### Health Check

```text
GET /health
```

Used to verify whether the API server is running.

### Prediction

```text
POST /predict
```

The endpoint accepts applicant information as a JSON object and uses the trained pipeline to generate a prediction.

---

## 🔐 API Validation

The endpoint includes validation for:

### Missing Fields

If required applicant fields are missing, the API returns:

```text
400 Bad Request
```

with a message identifying the missing fields.

### Invalid Numeric Data

If a numerical field contains invalid data, the API returns:

```text
400 Bad Request
```

and identifies the fields that must contain numeric values.

### Invalid JSON Structure

If the request body is not a JSON object representing an applicant record, the API returns:

```text
400 Bad Request
```

with an appropriate validation message.

### Notebook

[Open API Endpoint Notebook in Google Colab](https://colab.research.google.com/drive/1Ph0P2JCNwIz5Eq9Btfj28Is7nJFI_WP8?usp=sharing)

---

# 💰 Financial Aid Optimization

The optimization layer evaluates multiple financial-aid scenarios rather than applying one fixed aid percentage to every student.

Aid percentages are simulated from:

```text
0% → 80%
```

in:

```text
5% increments
```

For each scenario, the system estimates:

* Enrollment probability
* Net price
* Expected revenue
* Financial-aid impact

This approach allows different aid levels to be considered for different applicant conditions.

---

## ⚖️ Optimization Constraints

The optimizer considers multiple objectives rather than maximizing enrollment alone.

The major considerations are:

### 1. Enrollment Probability

The selected aid package should support a required minimum enrollment probability.

### 2. Expected Revenue

The system estimates expected net revenue after considering the financial-aid amount.

### 3. Equity Support

An equity-related constraint is included to provide support for students with higher financial need.

The optimizer therefore considers enrollment, expected revenue, and support for higher-need students together.

---

# 🔄 End-to-End Workflow

```text
             ┌──────────────────────┐
             │ Synthetic Data       │
             │ Generation           │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Data Validation &    │
             │ Preprocessing        │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Train/Test Split     │
             │ 80% / 20%            │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Random Forest        │
             │ Classifier           │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Model Evaluation     │
             │ Accuracy / ROC-AUC   │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Enrollment           │
             │ Probability          │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Aid Scenario         │
             │ Simulation           │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Optimization         │
             │ Enrollment + Revenue │
             │ + Equity             │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Applicant Prediction │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │ Flask REST API       │
             └──────────────────────┘
```

---

# 🛠️ Technologies Used

| Technology     | Purpose                                            |
| -------------- | -------------------------------------------------- |
| Python         | Core programming language                          |
| Pandas         | Dataset manipulation and analysis                  |
| NumPy          | Numerical operations and synthetic data generation |
| Scikit-learn   | Machine learning and preprocessing                 |
| Random Forest  | Enrollment classification                          |
| StandardScaler | Numerical feature scaling                          |
| OneHotEncoder  | Categorical feature encoding                       |
| Matplotlib     | Visualization and evaluation plots                 |
| Flask          | REST API development                               |
| Requests       | API testing                                        |
| Pickle         | Model serialization                                |
| Google Colab   | Development and experimentation                    |
| GitHub         | Source-code management and project hosting         |

---

# 📦 Data and Model Files

### Dataset

```text
tn_synthetic_aid_dataset.csv
```

Contains the synthetic student records used for model development.

### Trained Model

```text
yield_model_pipeline.pkl
```

Contains the trained preprocessing and machine-learning pipeline for reuse during testing and API inference.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/Umamaheswari-2005/Strategic-Financial-Aid-Yield-Optimizer.git
```

```bash
cd Strategic-Financial-Aid-Yield-Optimizer
```

## 2. Install Dependencies

Create a Python environment if required and install the necessary packages:

```bash
pip install pandas numpy scikit-learn matplotlib flask requests
```

## 3. Run the Notebooks

The recommended execution order is:

```text
1. Dataset Generator
        ↓
2. Model Training
        ↓
3. Model Testing
        ↓
4. API Endpoint
```

---

# 🧪 API Usage Example

A prediction request can be sent to:

```text
POST http://127.0.0.1:5000/predict
```

with applicant information in JSON format.

Example structure:

```json
{
  "district": "Coimbatore",
  "category": "SC",
  "urban": 1,
  "family_income": 691000,
  "first_gen": 0,
  "parent_grad": 1,
  "cutoff_12th": 92.35,
  "entrance_score": 145.6,
  "college_tier": "Tier-2 (Affiliated)",
  "tuition": 110000,
  "distance_km": 29.8,
  "competing_offers": 3,
  "merit_aid_pct": 0.443,
  "need_aid_pct": 0.43,
  "total_aid_pct": 0.437,
  "aid_amount": 48000,
  "net_price": 62000
}
```

The API processes the applicant record through the saved ML pipeline and returns the prediction response.

---

# 🧩 Project Challenges and Solutions

| Challenge                                                             | Solution                                                                            |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Real student admission and financial-aid data was unavailable         | Created a synthetic dataset for the prototype                                       |
| Different applicants have different financial and academic conditions | Simulated multiple financial-aid scenarios                                          |
| Maximizing enrollment alone can result in excessive aid               | Considered enrollment, expected revenue, and equity constraints together            |
| Training and testing in a single notebook reduced modularity          | Separated dataset generation, training, testing, and API into independent notebooks |
| Reusing the trained model                                             | Saved the complete ML pipeline using Pickle                                         |
| API input errors                                                      | Added missing-field, numeric-type, and JSON-structure validation                    |
| Re-running the Flask notebook can create a duplicate server           | Added a health-check-based server reuse mechanism                                   |

## The original project identified the lack of real student data, the limitations of a fixed aid percentage, and the risk of maximizing enrollment alone as important prototype challenges.

# 📈 Key Features

* ✅ Synthetic student dataset generation
* ✅ 6,000-record dataset
* ✅ 18 applicant-related fields
* ✅ Numerical and categorical preprocessing
* ✅ Stratified 80/20 train-test split
* ✅ Random Forest enrollment classifier
* ✅ Enrollment probability prediction
* ✅ Accuracy and ROC-AUC evaluation
* ✅ Classification report and ROC curve
* ✅ Feature-importance analysis
* ✅ Financial-aid scenario simulation
* ✅ Expected revenue estimation
* ✅ Equity-related optimization constraint
* ✅ Pretrained model serialization
* ✅ Independent model testing
* ✅ Flask REST API
* ✅ API health-check endpoint
* ✅ Request validation
* ✅ GitHub-ready modular project structure

---

# ⚠️ Important Note

This project is a **prototype** based on synthetic data. The generated data does not represent actual student admissions, financial-aid decisions, or enrollment behavior.

For production deployment, the model would require appropriate real-world institutional data, validation, monitoring, privacy controls, and domain-specific evaluation before being used to support actual financial-aid decisions.

---

# 📚 Project Resources

### Google Colab

**Dataset Generation**

[Open Notebook](https://colab.research.google.com/drive/1i0nXk5Xb9OvnfmCihBkVGXc1uPlrpPd8?usp=sharing)

**Model Training**

[Open Notebook](https://colab.research.google.com/drive/14oMuVLanayiex6UrsG5OsTvssNRZyri8?usp=sharing)

**Model Testing**

[Open Notebook](https://colab.research.google.com/drive/1D_V0oWqsJlrVuvIgNaMeHVe3gn0a3uw4?usp=sharing)

**API Endpoint**

[Open Notebook](https://colab.research.google.com/drive/1Ph0P2JCNwIz5Eq9Btfj28Is7nJFI_WP8?usp=sharing)

### GitHub Repository

[Strategic Financial Aid Yield Optimizer](https://github.com/Umamaheswari-2005/Strategic-Financial-Aid-Yield-Optimizer)

### Project Documentation

[Project Documentation](https://docs.google.com/document/d/16AM_bs8FyDamn5Jd_JPJYZQHZZcdLQZ5I7Lmxc6gQg0/edit?usp=sharing)

---

# 👩‍💻 Author

**Umamaheswari G**

AI/ML Developer | Machine Learning | Generative AI | Python

---

## 📄 License

This project is intended for educational, research, and prototype development purposes.
