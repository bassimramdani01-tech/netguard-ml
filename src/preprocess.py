"""Feature engineering: one-hot encoding + standard scaling.

Categorical columns (protocol_type, service, flag) are one-hot encoded with
``handle_unknown="ignore"`` so that the test set (which contains unseen
values) does not crash the pipeline. Numeric columns are standardised.
Near-constant numeric columns are dropped.
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import data as data_module

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class Preprocessor:
    """Fit-once, transform-many feature pipeline (also used by the demo)."""

    def __init__(self) -> None:
        self.transformer: ColumnTransformer | None = None
        self.feature_names: list[str] = []
        self.dropped_cols: list[str] = []

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        X = X.copy()
        # Drop numeric columns that are constant (e.g. num_outbound_cmds is always 0).
        self.dropped_cols = [c for c in data_module.NUMERIC if X[c].nunique(dropna=True) <= 1]
        keep_num = [c for c in data_module.NUMERIC if c not in self.dropped_cols]

        self.transformer = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 data_module.CATEGORICAL),
                ("num", StandardScaler(), keep_num),
            ],
            remainder="drop",
        )
        self.transformer.fit(X)

        cat_names = self.transformer.named_transformers_["cat"].get_feature_names_out(
            data_module.CATEGORICAL
        )
        self.feature_names = list(cat_names) + list(keep_num)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for c in data_module.NUMERIC:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        return self.transformer.transform(X)

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(MODELS_DIR, "preprocessor.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | None = None) -> "Preprocessor":
        path = path or os.path.join(MODELS_DIR, "preprocessor.joblib")
        return joblib.load(path)
