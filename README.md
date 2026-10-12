# NetGuard — ML-based Network Intrusion Detection

A reproducible benchmark of four machine-learning classifiers for **network
intrusion detection** on the [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
dataset. The project answers a simple question that most tutorials skip:

> *"Accuracy says 99% — but can the model actually find the attacks that matter?"*

NetGuard shows that the answer is **no**, and demonstrates *why* per-class recall
and macro-F1 — not accuracy — are the right way to evaluate an intrusion-detection
system. It ships with a full training pipeline, generated figures, and an
interactive [Gradio](https://gradio.app) demo.

---

## Key findings

1. **Tree ensembles win on ROC-AUC and binary F1**, but **class-balanced logistic
   regression is the most robust on rare attack families** (R2L / U2R).
2. **R2L attacks are effectively invisible to tree models** — Random Forest and
   XGBoost recall only **2% and 6%** of them, respectively, despite ~97% recall on
   "normal" traffic and high overall accuracy.
3. The NSL-KDD test set contains ~21 attack types **never seen in training**, so the
   honest accuracy (~78–80%) is far below the ~99% reported by tutorials that use a
   random split of the same data.

### Binary detection (normal vs. attack)

| Model | Accuracy | Precision (attack) | Recall (attack) | F1 (attack) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.755 | 0.917 | 0.626 | 0.744 | 0.791 |
| Random Forest | 0.773 | 0.968 | 0.621 | 0.757 | 0.962 |
| **XGBoost** | 0.795 | 0.969 | 0.661 | 0.786 | **0.967** |
| **MLP** | **0.797** | 0.925 | **0.701** | **0.797** | 0.899 |

### Multiclass classification (5 classes)

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.791** | **0.606** | 0.777 | **0.673** |
| Random Forest | 0.742 | 0.510 | 0.697 | 0.496 |
| XGBoost | 0.778 | 0.557 | 0.738 | 0.535 |
| MLP | 0.745 | 0.578 | 0.714 | 0.541 |

### Per-class recall — where the models actually fail

| Model | Normal | DoS | Probe | R2L | U2R |
|---|---|---|---|---|---|
| Logistic Regression | 0.914 | 0.831 | **0.848** | **0.233** | **0.537** |
| Random Forest | 0.972 | 0.769 | 0.613 | 0.021 | 0.104 |
| XGBoost | **0.972** | 0.844 | 0.663 | 0.063 | 0.134 |
| MLP | 0.931 | 0.799 | 0.616 | 0.103 | 0.254 |

---

## Quickstart

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data, trains 8 models, saves figures)
python scripts/run_all.py

# 3. Launch the interactive demo
python src/app.py
```

The pipeline writes:
- `results/` — metrics tables (CSV + Markdown), class distribution, imbalance study
- `figures/` — confusion matrices, ROC curves, per-class recall, feature importance, SHAP
- `models/` — trained XGBoost + preprocessor used by the demo

## Project structure

```
netguard-ml/
├── src/
│   ├── data.py         # dataset download + loading (NSL-KDD)
│   ├── preprocess.py   # one-hot encoding + standard scaling
│   ├── train.py        # Logistic Regression, Random Forest, XGBoost, MLP
│   ├── evaluate.py     # metrics + plots
│   ├── interpret.py    # feature importance + SHAP
│   └── app.py          # Gradio demo
├── scripts/
│   └── run_all.py      # end-to-end pipeline
├── report/
│   └── REPORT.md       # mini research paper
├── results/            # generated metrics
├── figures/            # generated plots
└── models/             # trained artifacts (gitignored)
```

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) is the de-facto standard
benchmark for network intrusion detection, derived from the 1999 KDD Cup data.
Each record is a network connection described by 41 features (e.g. `src_bytes`,
`service`, `flag`, `count`, error rates) labelled `normal` or one of ~20 attack
types grouped into four families: **DoS**, **Probe**, **R2L**, **U2R**. The
training set has 125,973 rows and the test set 22,544 rows; the two are drawn
from different distributions, which makes this a realistic generalisation test.

## Why class imbalance dominates this problem

The training set is severely imbalanced at the attack-family level:

| Class | Train | Test |
|---|---:|---:|
| normal | 67,343 | 9,711 |
| DoS | 45,927 | 7,458 |
| Probe | 11,656 | 2,421 |
| R2L | 995 | 2,887 |
| U2R | 52 | 67 |

With only 52 U2R and 995 R2L training examples, every model under-fits the rare
attacks — which is exactly why "accuracy" is a misleading metric here.

## License

[MIT](LICENSE) © 2026 Bassim Ramdani
