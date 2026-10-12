"""Model definitions.

We compare four classifiers spanning classic linear, tree-ensemble and
neural approaches. Because intrusion-detection data is heavily imbalanced
(attacks are rare), the linear and forest models use ``class_weight="balanced"``
and XGBoost uses ``scale_pos_weight`` (binary only) so that rare attacks are
not ignored in favour of overall accuracy.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def build_models(task: str = "multiclass", y_train: Optional[np.ndarray] = None,
                 seed: int = 42) -> dict[str, object]:
    """Return a dict of name -> untrained estimator for the given task.

    ``task`` is ``"binary"`` or ``"multiclass"``.
    """
    if task == "binary":
        scale_pos = None
        if y_train is not None:
            neg = int((np.asarray(y_train) == 0).sum())
            pos = int((np.asarray(y_train) == 1).sum())
            scale_pos = neg / max(pos, 1)
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=seed),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
            "XGBoost": XGBClassifier(
                n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, tree_method="hist", scale_pos_weight=scale_pos,
                eval_metric="logloss", random_state=seed, n_jobs=-1),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32), early_stopping=True, n_iter_no_change=10,
                validation_fraction=0.1, batch_size=256, learning_rate_init=1e-3,
                max_iter=200, random_state=seed),
        }

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", eval_metric="mlogloss",
            random_state=seed, n_jobs=-1),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, n_iter_no_change=10,
            validation_fraction=0.1, batch_size=256, learning_rate_init=1e-3,
            max_iter=200, random_state=seed),
    }
