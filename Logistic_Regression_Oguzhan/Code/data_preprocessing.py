"""
Common data loading and preprocessing utilities for the
UCI Credit Approval dataset (crx.data).

This module is shared by different models (Logistic Regression,
Decision Tree, Random Forest, etc.) so that all of them use
the same preprocessing steps and feature definitions.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# ============================================================================
# Constants and configuration
# ============================================================================

# Path to the dataset, relative to this file:
DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "crx.data"

# Column names according to the UCI Credit Approval dataset
FEATURE_COLS = [f"A{i}" for i in range(1, 16)]  # A1..A15
TARGET_COL = "class"

# Feature type information (from UCI documentation)
NUMERIC_FEATURES = ["A2", "A3", "A8", "A11", "A14", "A15"]
CATEGORICAL_FEATURES = [
    "A1", "A4", "A5", "A6", "A7",
    "A9", "A10", "A12", "A13",
]


# ============================================================================
# Data loading and basic preprocessing
# ============================================================================

def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load the raw dataset from disk and assign column names.

    Parameters
    ----------
    path : Path, optional
        Custom path to the dataset. If None, uses DATA_PATH.

    Returns
    -------
    df_raw : pd.DataFrame
        Raw dataframe with original values (including '?').
    """
    if path is None:
        path = DATA_PATH

    df_raw = pd.read_csv(path, header=None)
    df_raw.columns = FEATURE_COLS + [TARGET_COL]
    return df_raw


def preprocess_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Perform basic preprocessing that is shared across models:
    - Replace '?' with NaN
    - Map class labels '+' and '-' to 1 and 0
    - Split into X (features) and y (target)

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw dataframe as returned by load_raw_data().

    Returns
    -------
    df : pd.DataFrame
        Preprocessed dataframe (NaN instead of '?', numeric target).
    X : pd.DataFrame
        Feature matrix with columns FEATURE_COLS.
    y : pd.Series
        Target vector with values 0/1.
    """
    # Replace '?' with NaN
    df = df_raw.replace("?", np.nan)

   # Map target class: '+' -> 1, '-' -> 0
    df[TARGET_COL] = df[TARGET_COL].map({"+": 1, "-": 0})


    # Split into features and target
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    return df, X, y