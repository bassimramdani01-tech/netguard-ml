# NetGuard: Benchmarking Machine Learning for Network Intrusion Detection

*Bassim Ramdani — portfolio research report*

## Abstract

Network intrusion detection systems (NIDS) are typically evaluated with overall
accuracy, a metric that hides poor performance on the rare, high-impact attacks
that matter most. This work benchmarks four classifiers — logistic regression,
random forest, XGBoost and a multilayer perceptron — on the NSL-KDD dataset for
both binary (normal vs. attack) and multiclass (five-class) detection. Tree
ensembles achieve the best ROC-AUC (0.967) and binary F1, but class-balanced
logistic regression is the most robust on rare attack families, recalling 23% of
R2L and 54% of U2R attacks where random forest recalls only 2% and 10%. All
models fail to reliably detect R2L and U2R, demonstrating that accuracy is an
inadequate metric for NIDS and that per-class recall and macro-F1 should be
reported instead.

## 1. Introduction

Intrusion detection is the problem of distinguishing benign network traffic from
malicious activity. Because attacks are far rarer than normal traffic, a model
that predicts "normal" for every connection achieves high accuracy while being
completely useless. The standard NSL-KDD benchmark makes this concrete: its test
set contains roughly 21 attack types that never appear in the training set, so a
model must genuinely generalise rather than memorise.

Two questions drive this study:

1. How do classical and tree/neural classifiers compare on **binary** and
   **multiclass** intrusion detection?
2. How does **class imbalance** distort accuracy, and which models remain
   effective on the rare attack families (R2L and U2R)?

## 2. Data

NSL-KDD ([Tavallaee et al., 2009](#references)) is a cleaned version of the 1999
KDD Cup intrusion-detection dataset. Each record describes a network connection
with 41 features — including `src_bytes`, `dst_bytes`, `service`, `flag`,
`protocol_type`, connection counts and error rates — and a label that is either
`normal` or one of ~20 attack types grouped into four families: DoS, Probe, R2L
(remote-to-local) and U2R (user-to-root).

The training set (125,973 rows) and test set (22,544 rows) are drawn from
**different distributions**; the test set contains novel attacks absent from
training, making it a realistic generalisation benchmark rather than a random
holdout.

| Class | Train | Test |
|---|---:|---:|
| normal | 67,343 | 9,711 |
| DoS | 45,927 | 7,458 |
| Probe | 11,656 | 2,421 |
| R2L | 995 | 2,887 |
| U2R | 52 | 67 |

The imbalance is stark at the family level: U2R appears in only 52 training
examples (0.04%) and R2L in 995 (0.8%).

## 3. Method

**Preprocessing.** Categorical features (`protocol_type`, `service`, `flag`) are
one-hot encoded with `handle_unknown="ignore"` so unseen test-set values do not
break the pipeline; numeric features are standardised; constant columns (e.g.
`num_outbound_cmds`, always zero) are dropped. This yields 121 features.

**Models.** Four classifiers spanning linear, ensemble and neural approaches:
logistic regression (`class_weight="balanced"`), random forest (300 trees,
balanced), XGBoost (400 trees, `scale_pos_weight` for binary) and a multilayer
perceptron (64→32 hidden units, early stopping).

**Evaluation.** Accuracy, precision/recall/F1 (macro and weighted), ROC-AUC, and
per-class recall. Interpretability is provided by feature importance and SHAP
([Lundberg & Lee, 2017](#references)).

## 4. Results

### 4.1 Binary detection

| Model | Accuracy | Precision (attack) | Recall (attack) | F1 (attack) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.755 | 0.917 | 0.626 | 0.744 | 0.791 |
| Random Forest | 0.773 | 0.968 | 0.621 | 0.757 | 0.962 |
| XGBoost | 0.795 | 0.969 | 0.661 | 0.786 | **0.967** |
| MLP | **0.797** | 0.925 | **0.701** | **0.797** | 0.899 |

Tree ensembles dominate ROC-AUC (0.962–0.967), but recall on attacks is only
0.62–0.70: roughly a third of attacks are missed even by the best model. This is
the honest cost of the novel-attack test set.

### 4.2 Multiclass classification

| Model | Accuracy | Macro-F1 | Weighted-F1 | Macro-Recall |
|---|---|---|---|---|
| **Logistic Regression** | **0.791** | **0.606** | 0.777 | **0.673** |
| Random Forest | 0.742 | 0.510 | 0.697 | 0.496 |
| XGBoost | 0.778 | 0.557 | 0.738 | 0.535 |
| MLP | 0.745 | 0.578 | 0.714 | 0.541 |

The ranking **reverses** on macro metrics: logistic regression — the weakest
model by ROC-AUC — becomes the best by macro-F1 (0.606) and macro-recall (0.673).
Tree models win accuracy but lose the rare classes.

### 4.3 Per-class recall: where models fail

| Model | Normal | DoS | Probe | R2L | U2R |
|---|---|---|---|---|---|
| Logistic Regression | 0.914 | 0.831 | 0.848 | **0.233** | **0.537** |
| Random Forest | 0.972 | 0.769 | 0.613 | 0.021 | 0.104 |
| XGBoost | 0.972 | 0.844 | 0.663 | 0.063 | 0.134 |
| MLP | 0.931 | 0.799 | 0.616 | 0.103 | 0.254 |

The headline finding: **random forest and XGBoost detect only 2% and 6% of R2L
attacks respectively**, while reaching 97% recall on normal traffic. Class
imbalance, not model quality, is the binding constraint — and it is invisible in
accuracy.

### 4.4 Effect of class weighting (binary)

| Setting | Accuracy | Recall (attack) | F1 (attack) |
|---|---|---|---|
| plain | 0.768 | 0.613 | 0.751 |
| `class_weight="balanced"` | 0.773 | 0.621 | 0.757 |

The effect is small in the *binary* task because the two classes are roughly
balanced (58,630 attack vs. 67,343 normal). The severe imbalance is at the
*multiclass* family level, which is why weighting matters most for R2L/U2R.

### 4.5 Feature importance

XGBoost's top features (binary task) are interpretable for a network operator:
`service_ecr_i` (echo service, strongly tied to the Smurf DoS attack), `flag_SF`
(normal connection flag), `service_http`, `src_bytes`, `wrong_fragment`,
`protocol_type_icmp` and `dst_bytes`. The SHAP summary plot (see `figures/`)
confirms these directions.

## 5. Discussion

Three conclusions follow from the results:

1. **Accuracy is a misleading NIDS metric.** Models that reach ~97% recall on
   normal traffic still miss 94–98% of R2L attacks. Any NIDS evaluation must
   report per-class recall and macro-F1.
2. **Tree ensembles and linear models have complementary strengths.** Trees
   maximise ROC-AUC and binary F1; class-balanced logistic regression is the most
   robust on rare classes. A deployed system should combine or select models by
   the cost of false negatives.
3. **Generalisation is the real challenge.** The ~78–80% accuracy figures are
   realistic — far below the ~99% reported by tutorials that split a single file
   randomly and thus leak test-set attack types into training.

## 6. Limitations and future work

- **Rare-class performance is poor** and is the primary open problem; future work
  should evaluate oversampling (SMOTE/ADASYN), cost-sensitive learning, anomaly
  detection (one-class) and, for U2R/R2L, feature engineering targeted at those
  families.
- **NSL-KDD is a dated benchmark**; results should be extended to modern traffic
  (CICIDS2017, UNSW-NB15) before drawing deployment conclusions.
- A deeper learning comparison (CNNs on flow statistics, or graph models over
  connection topologies) would strengthen the benchmark.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python scripts/run_all.py   # downloads data, trains models, writes results/figures
python src/app.py           # interactive Gradio demo
```

All artifacts (`results/`, `figures/`) are regenerated from the committed code;
the random seed is fixed (42).

## References

- Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). *A detailed
  analysis of the KDD CUP 99 data set*. IEEE CISDA.
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model
  Predictions*. NeurIPS.
