"""
Common data loading and preprocessing utilities for the
UCI Credit Approval dataset.

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

FEATURE_COLS = [f"A{i}" for i in range(1, 16)]
TARGET_COL = "class"

NUMERIC_FEATURES = ["A2", "A3", "A8", "A11", "A14", "A15"]
CATEGORICAL_FEATURES = [
    "A1", "A4", "A5", "A6", "A7",
    "A9", "A10", "A12", "A13",
]

DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "crx.data"


def _resolve_default_data_path() -> Path:
    
    if DATA_PATH.exists():
        return DATA_PATH

    local_path = Path(__file__).resolve().parent / "crx.data"
    if local_path.exists():
        return local_path

    cwd_path = Path.cwd() / "crx.data"
    if cwd_path.exists():
        return cwd_path

    raise FileNotFoundError(
        "crx.data not found.\n"
        "Fix options:\n"
        "  (1) Put it under ../Data/crx.data (recommended)\n"
        "  (2) Put it next to data_preprocessing.py\n"
        "  (3) Run from a directory that contains crx.data\n"
        "  (4) Or call load_raw_data(path=Path('.../crx.data'))"
    )


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load the raw dataset and assign column names.
    """
    if path is None:
        path = _resolve_default_data_path()

    df_raw = pd.read_csv(path, header=None)
    df_raw.columns = FEATURE_COLS + [TARGET_COL]
    return df_raw


def preprocess_data(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Shared basic preprocessing:
    - Replace '?' with NaN
    - Map class labels '+' and '-' to 1 and 0
    - Split into X (features) and y (target)

    """
    df = df_raw.replace("?", np.nan)

    df[TARGET_COL] = df[TARGET_COL].map({"+": 1, "-": 0})

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    return df, X, y