"""Run the full NetGuard pipeline end-to-end.

Usage:
    python scripts/run_all.py

Produces:
    results/  - metrics tables (CSV + Markdown), class distributions, imbalance study
    figures/  - confusion matrices, ROC, per-class recall, feature importance, SHAP
    models/   - trained XGBoost (multiclass) + fitted preprocessor for the demo
"""
from __future__ import annotations

import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import data as data_module
from src.preprocess import Preprocessor
from src import train as train_module
from src import evaluate
from src import interpret

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")
for _d in (RESULTS, FIGURES, MODELS):
    os.makedirs(_d, exist_ok=True)

SEED = 42
GROUP_LABELS = data_module.GROUP_LABELS


def fmt(df: pd.DataFrame) -> str:
    return df.round(4).to_markdown()


def main() -> None:
    t0 = time.time()

    # ---------------------------------------------------------------- data
    print("[1/6] Loading data ...")
    Xtr, Xte, yb_tr, ym_tr, yb_te, ym_te = data_module.load_train_test()

    dist = pd.DataFrame({
        "train": ym_tr.map(data_module.ID_TO_GROUP).value_counts(),
        "test": ym_te.map(data_module.ID_TO_GROUP).value_counts(),
    }).fillna(0).astype(int)
    dist.to_csv(os.path.join(RESULTS, "class_distribution.csv"))
    print(fmt(dist))

    # ----------------------------------------------------------- preprocess
    print("[2/6] Preprocessing (one-hot + scaling) ...")
    prep = Preprocessor().fit(Xtr)
    Xtr_t = prep.transform(Xtr)
    Xte_t = prep.transform(Xte)
    prep.save()
    print(f"  features: {Xtr_t.shape[1]} (dropped constant: {prep.dropped_cols})")

    summary: dict[str, str] = {}

    # ---------------------------------------------------------------- binary
    print("[3/6] Training binary models (normal vs attack) ...")
    bin_models = train_module.build_models("binary", y_train=yb_tr, seed=SEED)
    bin_preds, bin_probas = {}, {}
    for name, m in bin_models.items():
        m.fit(Xtr_t, yb_tr)
        bin_preds[name] = m.predict(Xte_t)
        bin_probas[name] = m.predict_proba(Xte_t)[:, 1]

    bin_rows = []
    for name in bin_models:
        r = evaluate.binary_metrics(yb_te, bin_preds[name], bin_probas[name])
        r["model"] = name
        bin_rows.append(r)
    bin_df = pd.DataFrame(bin_rows).set_index("model")
    bin_df.to_csv(os.path.join(RESULTS, "metrics_binary.csv"))
    summary["binary"] = fmt(bin_df)
    print(fmt(bin_df))

    evaluate.save_confusion_matrix(yb_te, bin_preds["XGBoost"], ["normal", "attack"],
                                   "XGBoost — binary confusion matrix",
                                   os.path.join(FIGURES, "confusion_binary.png"))
    evaluate.save_roc_curves(yb_te, bin_probas, os.path.join(FIGURES, "roc_binary.png"))

    # ------------------------------------------------------- class imbalance
    print("[4/6] Class-imbalance study (Random Forest, binary) ...")
    rf_plain = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf_bal = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1)
    rf_plain.fit(Xtr_t, yb_tr)
    rf_bal.fit(Xtr_t, yb_tr)
    imb = pd.DataFrame({
        "accuracy": [evaluate.binary_metrics(yb_te, rf_plain.predict(Xte_t))["accuracy"],
                     evaluate.binary_metrics(yb_te, rf_bal.predict(Xte_t))["accuracy"]],
        "recall_attack": [evaluate.binary_metrics(yb_te, rf_plain.predict(Xte_t))["recall_attack"],
                          evaluate.binary_metrics(yb_te, rf_bal.predict(Xte_t))["recall_attack"]],
        "f1_attack": [evaluate.binary_metrics(yb_te, rf_plain.predict(Xte_t))["f1_attack"],
                      evaluate.binary_metrics(yb_te, rf_bal.predict(Xte_t))["f1_attack"]],
    }, index=["plain", "class_weight=balanced"])
    imb.to_csv(os.path.join(RESULTS, "imbalance_study.csv"))
    summary["imbalance"] = fmt(imb)
    print(fmt(imb))

    # ------------------------------------------------------------- multiclass
    print("[5/6] Training multiclass models (5 classes) ...")
    multi_models = train_module.build_models("multiclass", seed=SEED)
    multi_preds = {}
    for name, m in multi_models.items():
        m.fit(Xtr_t, ym_tr)
        multi_preds[name] = m.predict(Xte_t)

    multi_rows = []
    for name in multi_models:
        r = evaluate.multi_metrics(ym_te, multi_preds[name])
        r["model"] = name
        multi_rows.append(r)
    multi_df = pd.DataFrame(multi_rows).set_index("model")
    multi_df.to_csv(os.path.join(RESULTS, "metrics_multi.csv"))
    summary["multiclass"] = fmt(multi_df)
    print(fmt(multi_df))

    recall_dict = {name: evaluate.per_class_recall(ym_te, multi_preds[name], GROUP_LABELS)
                   for name in multi_models}
    recall_df = pd.DataFrame(recall_dict).T
    recall_df.to_csv(os.path.join(RESULTS, "per_class_recall.csv"))
    evaluate.save_confusion_matrix(ym_te, multi_preds["XGBoost"], GROUP_LABELS,
                                   "XGBoost — multiclass confusion matrix",
                                   os.path.join(FIGURES, "confusion_multi.png"))
    evaluate.save_per_class_recall_chart(recall_dict, os.path.join(FIGURES, "per_class_recall.png"))

    # --------------------------------------------------------- interpretability
    print("[6/6] Interpretability (feature importance + SHAP) ...")
    xgb_bin = bin_models["XGBoost"]
    imp_df = interpret.save_feature_importance(xgb_bin, prep.feature_names,
                                               os.path.join(FIGURES, "feature_importance.png"))
    imp_df.to_csv(os.path.join(RESULTS, "feature_importance.csv"), index=False)

    # SHAP on a subsample to stay CPU-friendly
    rng = np.random.RandomState(SEED)
    n_samp = min(2000, Xte_t.shape[0])
    idx = rng.choice(Xte_t.shape[0], size=n_samp, replace=False)
    interpret.save_shap_summary(xgb_bin, Xte_t[idx], prep.feature_names,
                                os.path.join(FIGURES, "shap_summary.png"))

    # ------------------------------------------------- save model for demo
    joblib.dump(multi_models["XGBoost"], os.path.join(MODELS, "xgb_multi.joblib"))
    print(f"\nSaved demo model + preprocessor to {MODELS}")

    with open(os.path.join(RESULTS, "summary.md"), "w", encoding="utf-8") as fh:
        for key in ["binary", "imbalance", "multiclass"]:
            fh.write(f"## {key}\n\n{summary[key]}\n\n")

    print(f"\nDone in {time.time() - t0:.1f}s. Artifacts in results/, figures/, models/.")


if __name__ == "__main__":
    main()
