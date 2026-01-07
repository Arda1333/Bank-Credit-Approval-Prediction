# Credit Approval Prediction with Random Forest

This project implements a **Random Forest–based binary classification pipeline** for the **Credit Approval dataset**.  
The goal is to predict whether a credit application is **approved (+)** or **not approved (−)** using structured tabular data.

The repository contains **two versions of the same implementation**:
- A **Python script** (`random_forest_ml.py`)
- A **Jupyter Notebook** (`random_forest.ipynb`)  

Both files use **identical code**; the notebook additionally includes **executed outputs and visualizations**.

---

## Files in This Repository

### `random_forest_ml.py`
- Pure Python implementation
- Suitable for script-based execution
- Produces metrics and plots when run
- Contains the full end-to-end machine learning pipeline

### `random_forest.ipynb`
- Jupyter Notebook version of the same code
- **Code is identical** to `random_forest_ml.py`
- Includes:
  - Printed outputs
  - Plotted figures
  - Grid search logs
- Intended for **interactive exploration and reporting**

---

## Dataset

- **File name:** `crx.data`
- **Task:** Binary classification (credit approval)
- **Target variable:** `Class`
  - `+` → Approved (mapped to `1`)
  - `-` → Not approved (mapped to `0`)
- Missing values are represented by `?`

### Feature Types
- **Numerical features:**  
  `A2, A3, A8, A11, A14, A15`
- **Categorical features:**  
  All remaining attributes (`A1`, `A4`, `A5`, …)

---

## Machine Learning Pipeline

The model is implemented using **scikit-learn Pipelines** and includes:

### 1. Preprocessing
- **Numerical features**
  - Median imputation
- **Categorical features**
  - Most-frequent imputation
  - One-hot encoding (`handle_unknown="ignore"`)

### 2. Model
- `RandomForestClassifier`
- Key settings:
  - `n_estimators = 200`
  - `class_weight = "balanced"`
  - `random_state = 42`

---

## Evaluation Strategy

### Cross-Validation
- **5-fold Stratified K-Fold**
- Metrics:
  - Accuracy
  - ROC-AUC

### Train / Test Split
- 80% train / 20% test
- Stratified by class label

### Reported Metrics
- Accuracy
- ROC-AUC
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve

---

## Visualizations

The following visual outputs are generated:

- Confusion Matrix (heatmap)
- ROC Curve
- Top-15 Feature Importances
- Decision Tree visualization  
  (one tree from the forest, `max_depth = 3`)

---

## Hyperparameter Tuning

### Grid Search
Performed using `GridSearchCV` with:
- **Scoring:** F1 score
- **Cross-validation:** 5-fold

#### Tuned Parameters
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`

### Tuned Model Evaluation
- Accuracy
- ROC-AUC
- F1 score
- Classification report

---

## Threshold Optimization

After tuning:
- Precision–Recall curve is computed
- F1 score is evaluated across thresholds
- **Optimal decision threshold** is selected to maximize F1
- Final F1 score is reported using this optimized threshold

---

## How to Run

### Python Script
```bash
python random_forest_ml.py
