"""NetGuard Gradio demo.

Run after ``python scripts/run_all.py`` has produced ``models/xgb_multi.joblib``
and ``models/preprocessor.joblib``:

    python src/app.py

Then open the printed local URL and paste 41 comma-separated NSL-KDD feature
values (or click a pre-loaded example) to classify a connection.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import joblib

from src import data as data_module
from src.preprocess import Preprocessor

MODEL_PATH = os.path.join(ROOT, "models", "xgb_multi.joblib")
PREPROC_PATH = os.path.join(ROOT, "models", "preprocessor.joblib")
GROUP_LABELS = data_module.GROUP_LABELS


def _load_assets():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(PREPROC_PATH)):
        raise RuntimeError(
            "Model not found. Run `python scripts/run_all.py` first to train it."
        )
    model = joblib.load(MODEL_PATH)
    prep = Preprocessor.load(PREPROC_PATH)
    return model, prep


def _example_rows() -> dict[str, str]:
    """Return a few real rows from the test set for the demo dropdown."""
    te = data_module.load_raw(os.path.join(data_module.DATA_DIR, "KDDTest+.txt"))
    te["group"] = te["label"].map(data_module.ATTACK_TO_GROUP)
    picks = {}
    for group in ["normal", "dos", "probe", "r2l", "u2r"]:
        sub = te[te["group"] == group]
        if len(sub):
            row = sub[data_module.COLUMNS].iloc[0].astype(str)
            picks[group] = ",".join(row.values)
    return picks


def classify(raw: str) -> str:
    model, prep = _load_assets()
    try:
        vals = [v.strip() for v in raw.split(",")]
        if len(vals) != len(data_module.COLUMNS):
            return f"Expected {len(data_module.COLUMNS)} comma-separated values, got {len(vals)}."
        row = pd.DataFrame([vals], columns=data_module.COLUMNS)
        X = prep.transform(row)
        probs = model.predict_proba(X)[0]
        pred = int(np.argmax(probs))
        label = GROUP_LABELS[pred]
        verdict = "ATTACK (malicious)" if pred != 0 else "NORMAL (benign)"
        lines = [f"Prediction: {label}", f"Verdict: {verdict}", ""]
        for i, name in enumerate(GROUP_LABELS):
            lines.append(f"  {name:<8} {probs[i]:.1%}")
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def build_demo():
    import gradio as gr

    examples = _example_rows()
    with gr.Blocks(title="NetGuard — ML Network Intrusion Detection") as demo:
        gr.Markdown(
            "# NetGuard — ML Network Intrusion Detection\n"
            "Classify a network connection as **normal** or an attack family "
            "(DoS / Probe / R2L / U2R) using an XGBoost model trained on NSL-KDD."
        )
        inp = gr.Textbox(
            label="41 NSL-KDD features (comma-separated)",
            lines=2,
            placeholder="0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00",
        )
        with gr.Row():
            for name, sample in examples.items():
                gr.Button(name.capitalize()).click(
                    fn=lambda s=sample: s, outputs=inp)
        classify_btn = gr.Button("Classify", variant="primary")
        out = gr.Textbox(label="Result", lines=6)
        classify_btn.click(fn=classify, inputs=inp, outputs=out)
    return demo


if __name__ == "__main__":
    build_demo().launch()
