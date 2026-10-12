"""Model interpretability: feature importance + SHAP."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def save_feature_importance(model, feature_names, out_path, top: int = 15) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame()
    imp = np.asarray(model.feature_importances_)
    idx = np.argsort(imp)[::-1][:top]
    names = [feature_names[i] for i in idx]
    vals = imp[idx]
    plt.figure(figsize=(8, 6))
    plt.barh(range(top), vals[::-1], tick_label=names[::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top} feature importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return pd.DataFrame({"feature": names, "importance": vals})


def save_shap_summary(model, X_sample, feature_names, out_path, max_display: int = 15) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # Binary tree explainer returns a 2D array; multiclass returns a list.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    plt.figure()
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
