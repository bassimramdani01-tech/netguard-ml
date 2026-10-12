"""Evaluation metrics and plots."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

sns.set_theme(style="whitegrid")


def binary_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Metrics for the binary task, focusing on the attack (positive) class."""
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_attack": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_attack": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_attack": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_proba is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
    return out


def multi_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def per_class_recall(y_true, y_pred, class_names) -> pd.Series:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return pd.Series(recall, index=class_names)


def save_confusion_matrix(y_true, y_pred, class_names, title, out_path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_roc_curves(y_true, proba_dict: dict[str, np.ndarray], out_path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    for name, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC — binary intrusion detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_per_class_recall_chart(recall_dict: dict[str, pd.Series], out_path) -> None:
    df = pd.DataFrame(recall_dict).T  # rows = models, cols = classes
    df.plot(kind="bar", figsize=(8, 5))
    plt.title("Recall per class (multiclass)")
    plt.ylabel("Recall")
    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    plt.legend(title="Class", loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
