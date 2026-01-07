"""
Logistic Regression model for the UCI Credit Approval dataset.

This script:
- Loads and preprocesses the data (using data_preprocessing.py)
- Builds a unified preprocessing + logistic regression pipeline
- Evaluates the model with 5-fold stratified cross-validation
- Analyzes coefficients and odds ratios for interpretability
- Plots the most important positive and negative coefficients
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from data_preprocessing import (
    load_raw_data,
    preprocess_data,
    FEATURE_COLS,
    TARGET_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# Use a backend that opens plots in a separate window
matplotlib.use("TkAgg")


# ============================================================================
# 1. Pipeline construction (preprocessing + Logistic Regression)
# ============================================================================

def build_logistic_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline that:
    - imputes and scales numeric features
    - imputes and one-hot encodes categorical features
    - trains a Logistic Regression classifier
    """

    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine both into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Logistic Regression model
    log_reg = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1000,
    )

    # Full pipeline: preprocessing + model
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", log_reg),
    ])

    return clf


# ============================================================================
# 2. Cross-validation evaluation
# ============================================================================

def evaluate_with_cv(clf: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> None:
    """
    Evaluate the logistic regression pipeline using stratified k-fold cross-validation.

    Prints mean and standard deviation for Accuracy and ROC-AUC.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        clf,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    acc_mean = cv_results["test_accuracy"].mean()
    acc_std = cv_results["test_accuracy"].std()

    auc_mean = cv_results["test_roc_auc"].mean()
    auc_std = cv_results["test_roc_auc"].std()

    print("\n=== 5-Fold Cross Validation Results (Logistic Regression) ===")
    print(f"Accuracy (mean ± std): {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"ROC-AUC  (mean ± std): {auc_mean:.4f} ± {auc_std:.4f}")


# ============================================================================
# 3. Coefficient and odds ratio analysis (interpretability)
# ============================================================================

def analyze_coefficients(clf: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Fit the pipeline on the full dataset and compute:

    - Coefficients of the Logistic Regression model
    - Corresponding odds ratios (exp(coef))
    - A table of (feature, coef, odds_ratio), sorted by |coef|

    Returns
    -------
    coef_table : pd.DataFrame
        DataFrame with columns ['feature', 'coef', 'odds_ratio'].
    """
    print("\n=== Coefficient & Odds Ratio Analysis ===")

    # Fit the model on the full data
    clf.fit(X, y)

    # Extract the OneHotEncoder from the pipeline
    ohe = (
        clf.named_steps["preprocess"]
        .named_transformers_["cat"]
        .named_steps["onehot"]
    )

    # Get the expanded categorical feature names after one-hot encoding
    ohe_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)

    # Combine numeric + expanded categorical feature names
    all_features = list(NUMERIC_FEATURES) + list(ohe_feature_names)

    # Extract logistic regression coefficients and compute odds ratios
    coef = clf.named_steps["model"].coef_[0]
    odds_ratio = np.exp(coef)

    coef_table = pd.DataFrame({
        "feature": all_features,
        "coef": coef,
        "odds_ratio": odds_ratio,
    })

    # Sort by absolute magnitude of the coefficient
    coef_table = coef_table.reindex(
        coef_table["coef"].abs().sort_values(ascending=False).index
    )

    print("\nTop 20 features by |coef|:")
    print(coef_table.head(20))

    return coef_table


# ============================================================================
# 4. Coefficient visualization
# ============================================================================

def plot_top_coefficients(coef_table: pd.DataFrame, top_n: int = 10) -> None:
    """
    Plot bar charts for the top positive and top negative coefficients.

    Parameters
    ----------
    coef_table : pd.DataFrame
        Table returned by analyze_coefficients().
    top_n : int
        Number of top positive and negative features to show.
    """
    print("\n=== Plotting Coefficient Importance ===")

    # Top positive and top negative coefficients
    top_positive = coef_table.sort_values(by="coef", ascending=False).head(top_n)
    top_negative = coef_table.sort_values(by="coef", ascending=True).head(top_n)

    # Positive coefficients
    plt.figure(figsize=(10, 6))
    plt.barh(top_positive["feature"], top_positive["coef"], color="green")
    plt.gca().invert_yaxis()
    plt.title("Top Positive Coefficients (Logistic Regression)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.show()

    # Negative coefficients
    plt.figure(figsize=(10, 6))
    plt.barh(top_negative["feature"], top_negative["coef"], color="red")
    plt.gca().invert_yaxis()
    plt.title("Top Negative Coefficients (Logistic Regression)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5. main()
# ============================================================================

def main() -> None:
    print("=== Loading raw data ===")
    df_raw = load_raw_data()
    print("Raw shape:", df_raw.shape)

    print("\n=== Basic preprocessing ===")
    df, X, y = preprocess_data(df_raw)
    print("Processed shape:", df.shape)
    print("Missing values per column:\n", df.isna().sum())
    print("Class distribution:\n", y.value_counts())

    print("\n=== Building Logistic Regression pipeline ===")
    clf = build_logistic_pipeline()
    print("Pipeline constructed.")

    # Cross-validation
    evaluate_with_cv(clf, X, y, n_splits=5)

    # Coefficient analysis
    coef_table = analyze_coefficients(clf, X, y)

    # Optional: save coefficients to CSV for the report
    # coef_table.to_csv(Path(__file__).resolve().parent.parent / "Data" / "logreg_coefficients.csv", index=False)

    # Plot coefficients (comment out if not needed)
    plot_top_coefficients(coef_table, top_n=10)


if __name__ == "__main__":
    main()