"""
Logistic Regression model for the UCI Credit Approval dataset

This script:
- Loads and preprocesses the dataset using the shared preprocessing utilities
- Runs Nested Cross-Validation (outer CV for evaluation, inner CV for hyperparameter tuning)
- Reports CV metrics (AUC, accuracy, precision, recall, F1)
- Fits a final model on the full dataset to save best params + interpretability outputs (coefficients)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from data_preprocessing import load_raw_data, preprocess_data

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def safe_get_feature_names(preprocess: ColumnTransformer):
    """
    Extract feature names from ColumnTransformer after one-hot encoding.
    Works across sklearn versions as best-effort.
    """
    feature_names = []
    for name, trans, cols in preprocess.transformers_:
        if name == "remainder" and trans == "drop":
            continue

        if hasattr(trans, "named_steps"):
            last = list(trans.named_steps.values())[-1]
        else:
            last = trans

        if hasattr(last, "get_feature_names_out"):
            try:
                fn = last.get_feature_names_out(cols)
            except TypeError:
                fn = last.get_feature_names_out()
            feature_names.extend(list(fn))
        else:
            if isinstance(cols, (list, tuple, np.ndarray)):
                feature_names.extend([str(c) for c in cols])
            else:
                feature_names.append(str(cols))
    return feature_names

def build_pipeline_from_df(df: pd.DataFrame):
    """
    Create ColumnTransformer based on df dtypes:
    - numeric: impute median + scale
    - categorical: impute most_frequent + onehot
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in df.columns if c not in numeric_features]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = LogisticRegression(max_iter=2000)

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model),
    ])
    return pipe

def nested_cv_logreg(X: pd.DataFrame, y: np.ndarray, out_dir: Path):
    """
    Nested CV:
      - Outer CV reports unbiased metrics
      - Inner CV selects hyperparameters by ROC-AUC
    """
    outer_cv = StratifiedKFold(5, shuffle=True, random_state=42)

    param_grid = {
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["liblinear"],  # supports l1/l2 for binary
    }

    outer_metrics = []
    fold = 0

    for train_idx, test_idx in outer_cv.split(X, y):
        fold += 1
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = build_pipeline_from_df(X_tr)

        inner_cv = StratifiedKFold(5, shuffle=True, random_state=42)

        grid = GridSearchCV(
            clf,
            param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        grid.fit(X_tr, y_tr)
        best = grid.best_estimator_

        y_prob = best.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "fold": fold,
            "roc_auc": float(roc_auc_score(y_te, y_prob)),
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred, zero_division=0)),
            "recall": float(recall_score(y_te, y_pred, zero_division=0)),
            "f1": float(f1_score(y_te, y_pred, zero_division=0)),
            "best_params": grid.best_params_,
        }
        outer_metrics.append(metrics)
        save_json(out_dir / f"nestedcv_fold_{fold}.json", metrics)

    dfm = pd.DataFrame(outer_metrics)
    summary = {
        "roc_auc_mean": float(dfm["roc_auc"].mean()),
        "roc_auc_std": float(dfm["roc_auc"].std(ddof=1)),
        "accuracy_mean": float(dfm["accuracy"].mean()),
        "accuracy_std": float(dfm["accuracy"].std(ddof=1)),
        "precision_mean": float(dfm["precision"].mean()),
        "precision_std": float(dfm["precision"].std(ddof=1)),
        "recall_mean": float(dfm["recall"].mean()),
        "recall_std": float(dfm["recall"].std(ddof=1)),
        "f1_mean": float(dfm["f1"].mean()),
        "f1_std": float(dfm["f1"].std(ddof=1)),
    }
    save_json(out_dir / "nestedcv_summary.json", summary)
    dfm.to_csv(out_dir / "nestedcv_folds.csv", index=False)

    return summary

def save_coef_analysis(best_model: Pipeline, out_dir: Path):
    preprocess = best_model.named_steps["preprocess"]
    model = best_model.named_steps["model"]

    feat_names = safe_get_feature_names(preprocess)

    coefs = model.coef_.ravel()
    odds = np.exp(coefs)

    df = pd.DataFrame({
        "feature": feat_names,
        "coef": coefs,
        "odds_ratio": odds,
    }).sort_values("coef", ascending=False)

    df.to_csv(out_dir / "logreg_coefficients.csv", index=False)

    df_abs = df.copy()
    df_abs["abs"] = df_abs["coef"].abs()
    df_abs = df_abs.sort_values("abs", ascending=False)

    topk = 15
    top = df_abs.head(topk).sort_values("coef", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["coef"])
    plt.title("Top coefficients (by absolute magnitude)")
    plt.tight_layout()
    plt.savefig(out_dir / "top_coefficients.png")
    plt.close()


def main():
    out_dir = Path("outputs_logreg")
    ensure_dir(out_dir)

    # Load + preprocess using your shared module
    df_raw = load_raw_data()
    _, X, y_series = preprocess_data(df_raw)
    y = y_series.to_numpy(dtype=int)

    # Nested CV evaluation
    summary = nested_cv_logreg(X, y, out_dir)
    print("Nested CV summary:", summary)

    # Fit a single GridSearchCV on the FULL dataset (for best params + interpretability)
    final_clf = build_pipeline_from_df(X)
    final_param_grid = {
        "model__penalty": ["l1", "l2"],
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "model__class_weight": [None, "balanced"],
        "model__solver": ["liblinear"],
    }

    final_cv = StratifiedKFold(5, shuffle=True, random_state=42)
    final_grid = GridSearchCV(
        final_clf,
        final_param_grid,
        scoring="roc_auc",
        cv=final_cv,
        n_jobs=-1,
        refit=True,
    )
    final_grid.fit(X, y)

    best_model = final_grid.best_estimator_
    save_json(out_dir / "best_params.json", final_grid.best_params_)

    save_coef_analysis(best_model, out_dir)

if __name__ == "__main__":
    main()