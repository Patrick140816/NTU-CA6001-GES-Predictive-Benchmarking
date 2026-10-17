# NTU CCDS CA6001 — AI Algorithms Fundamentals & Application
## Option 5: Singapore Graduate Employment Survey (GES) Predictive Benchmarking (2013–2024)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626.svg)](https://jupyter.org/)

---

### Project Overview | 项目概述
This repository contains the complete research, algorithmic implementations, and empirical benchmark for **NTU CCDS CA6001 Course Assignment (Option 5)**. 

Using longitudinal official survey records across Singapore's six autonomous universities (NTU, NUS, SIT, SMU, SUSS, SUTD) from **2013 through 2024**, this project investigates the inductive biases, variance-bias trade-offs, and predictive boundaries across four foundational machine learning paradigms:
1. **Generalized Linear Models (M1)**: Ridge & Balanced Logistic Regression with $L_1/L_2$ Regularization.
2. **Bagging Tree Ensembles (M2)**: Decision Trees vs. Multi-Tree Random Forests (Leo Breiman Variance Reduction).
3. **Gradient Boosting Decision Trees (M3)**: LightGBM with Sequential Error Correction & Game-Theoretic SHAP Interpretability.
4. **Deep Tabular Learning (M4)**: PyTorch Multi-Layer Perceptron with Entity Embeddings for high-cardinality institutional/domain categories.
5. **Heterogeneous Blended Ensemble**: Blending gradient boosted trees with deep entity embeddings ($0.6 \text{ M3} + 0.4 \text{ M4}$).

---

### Master Quantitative Benchmark | 核心基准成绩榜单

#### 1. Binary Classification: High Starting Salary Tier Prediction ($\ge 70\text{th}$ percentile within cohort & $\ge 85\%$ FT employment)
> **Temporal Blind Test Set (2023–2024 Cohorts, $N=275$, 53 Positive Classes / 19.27%)**

| Model Paradigm | Algorithm Backbone | 5-Fold CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Test PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1: Generalized Linear** | Logistic Regression ($L_2$) | $0.9177 \pm 0.025$ | 0.6218 | 0.3355 | **0.9811** | 0.5000 | 0.9063 | 0.7593 |
| **M2: Bagging Ensemble** | Random Forest (150 trees) | $0.9292 \pm 0.030$ | 0.6545 | 0.3521 | 0.9434 | 0.5128 | 0.9201 | 0.7771 |
| **M3: Gradient Boosting** | LightGBM GBDT | $0.9310 \pm 0.025$ | 0.6073 | 0.3270 | **0.9811** | 0.4906 | **0.9258** | **0.8125** |
| **M4: Deep Tabular** | PyTorch Entity-Embedding MLP | **$0.9314 \pm 0.021$** | **0.7236** | **0.4080** | 0.9623 | **0.5730** | 0.9177 | 0.7657 |
| **Heterogeneous Blend** | **$0.6 \text{ M3} + 0.4 \text{ M4}$** | — | **0.6873** | **0.3796** | **0.9811** | **0.5474** | **0.9267** | **0.8069** |
| *(Tuned Weight Blend)* | *$0.7 \text{ M3} + 0.3 \text{ M4}$* | — | 0.6655 | 0.3636 | **0.9811** | 0.5306 | **0.9279** | 0.8105 |

#### 2. Continuous Salary Regression (Gross Monthly Salary in SGD)
| Model Paradigm | Algorithm | $R^2$ (Goodness of Fit) | RMSE (SGD) | MAE (SGD) | Inductive Bias Trait |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **M4: Deep Tabular** | PyTorch Entity-Embedding MLP | **0.7487** 🏆 | **S$ 365.88** 🏆 | **S$ 272.73** 🏆 | Smooth continuous manifold interpolation |
| **M2: Bagging Ensemble** | Random Forest Regressor | 0.6941 | S$ 403.71 | S$ 288.51 | Robust to outliers; step-wise plateau limit |
| **M3: Gradient Boosting** | LightGBM Regressor | 0.6495 | S$ 432.13 | S$ 323.82 | Fast convergence; sensitive to high-variance tails |
| **M1: Generalized Linear** | Ridge Regression ($L_2$) | 0.6167 | S$ 451.86 | S$ 376.71 | Flat hyperplane lacks non-linear compensation |

---

### Key Empirical Findings & Theoretical Insights | 核心学术结论

1. **LightGBM Dominates Ranking & Discriminative Ordering**:
   - LightGBM achieves the top single-model test **ROC-AUC (0.9258)** and **PR-AUC (0.8125)**. Sequential negative gradient descent focuses heavily on boundary hard samples.
2. **Deep Tabular Embeddings Excel at Continuous Interpolation**:
   - Neural entity embeddings map universities and academic clusters into continuous geometric proximity ($\cos(\text{NUS}, \text{NTU}) > 0.85$), driving continuous salary regression $R^2$ to **0.7487** (a $21\%$ error reduction over linear baselines).
3. **Macro Wage Inflation & Threshold Drift (Critical Failure Mode Analysis)**:
   - Between 2013 and 2024, Singapore's top-30% salary bar rose from S$3,200 to S$4,800. Models trained on nominal SGD inputs ($x_{t-1}$) over-predict positives at static $p \ge 0.5$ cutoffs (Recall $98\%$, Precision $33\%$). Re-calibrating the decision threshold to $p \ge 0.85$ restores precision to $50.5\%$ and F1 to $0.65$, demonstrating high ranking stability despite label threshold drift.
4. **Heterogeneous Synergy**:
   - Blending tree-based orthogonal axis splitting (LightGBM) with continuous neural manifolds (PyTorch) achieves the coursework peak ROC-AUC of **0.9279**.

---

### Repository Structure | 项目目录结构

```text
ca6001-ges-project/
├── data/
│   ├── raw/
│   │   └── GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv       # Raw survey from Data.gov.sg (1,550 rows)
│   └── processed/
│       ├── ges_cleaned_features.csv                               # Sanitized & lag-engineered data (1,445 rows)
│       ├── train_temporal_2013_2022.csv                           # 5-Fold CV training split (1,170 rows)
│       └── test_temporal_2023_2024.csv                            # Out-of-time blind test split (275 rows)
├── notebooks/                                                     # 12 Executable Notebooks (Dual EN & EN/ZH)
│   ├── 01_data_preprocessing_and_eda.ipynb                        # Step 1: Preprocessing & EDA (English)
│   ├── 01_data_preprocessing_and_eda_bilingual.ipynb              # Step 1: Preprocessing & EDA (Bilingual)
│   ├── 02_m1_logistic_baseline.ipynb                              # Step 2: Linear Baseline (English)
│   ├── 02_m1_logistic_baseline_bilingual.ipynb                    # Step 2: Linear Baseline (Bilingual)
│   ├── 02_m2_bagging_random_forest.ipynb                          # Step 3: Bagging Trees (English)
│   ├── 02_m2_bagging_random_forest_bilingual.ipynb                # Step 3: Bagging Trees (Bilingual)
│   ├── 02_m3_boosting_gbdt_shap.ipynb                             # Step 4: Boosting & SHAP (English)
│   ├── 02_m3_boosting_gbdt_shap_bilingual.ipynb                   # Step 4: Boosting & SHAP (Bilingual)
│   ├── 02_m4_deep_tabular_mlp.ipynb                               # Step 5: Deep Tabular MLP (English)
│   ├── 02_m4_deep_tabular_mlp_bilingual.ipynb                     # Step 5: Deep Tabular MLP (Bilingual)
│   ├── 03_model_evaluation_and_comparison.ipynb                   # Step 6: Master Evaluation (English)
│   └── 03_model_evaluation_and_comparison_bilingual.ipynb         # Step 6: Master Evaluation (Bilingual)
├── outputs/
│   ├── figures/                                                   # 300-DPI Publication Charts (Pure English)
│   │   ├── eda_salary_trends.png
│   │   ├── eda_university_domain_boxplot.png
│   │   ├── eda_correlation_heatmap.png
│   │   ├── m1_logistic_coefficients.png
│   │   ├── m2_depth_overfitting_curve.png
│   │   ├── m3_shap_beeswarm.png
│   │   ├── m4_training_convergence_curves.png
│   │   ├── eval_overlaid_roc_pr_curves.png
│   │   └── eval_confusion_matrices.png
│   └── metrics/                                                   # Standardized Model Metrics JSON & CSV
│       ├── m1_metrics.json
│       ├── m2_metrics.json
│       ├── m3_metrics.json
│       ├── m4_metrics.json
│       ├── master_benchmark_table.csv
│       └── master_metrics.json
├── requirements.txt                                               # Python dependencies
└── README.md                                                      # Project documentation
```

---

### Reproducibility & Execution Sequence | 复现与执行指引

All notebooks are self-contained and pre-executed with cached inline outputs and figures. To replicate the full pipeline from scratch:

```bash
# 1. Clone repository
git clone <your-github-repo-url>
cd <repo-folder>

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebooks in sequence:
# Step 1: Preprocessing & Anti-Leakage Feature Engineering
jupyter nbconvert --to notebook --execute notebooks/01_data_preprocessing_and_eda.ipynb

# Steps 2-5: Train & Benchmark Individual Models
jupyter nbconvert --to notebook --execute notebooks/02_m1_logistic_baseline.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_m2_bagging_random_forest.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_m3_boosting_gbdt_shap.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_m4_deep_tabular_mlp.ipynb

# Step 6: Master Cross-Model Synthesis & Publication Visualizations
jupyter nbconvert --to notebook --execute notebooks/03_model_evaluation_and_comparison.ipynb
```

---

### Team Allocation & ILO Mapping | 团队分工与课程大纲对应

| Member | Module Lead | Primary Algorithmic Scope | Intended Learning Outcomes (ILO) |
| :--- | :--- | :--- | :--- |
| **Member 1** | Module 02-M1 | Generalized Linear Models & $L_1/L_2$ Penalties | **ILO1, ILO2, ILO4** (Linear Decision Boundaries & Sparsity) |
| **Member 2** | Module 02-M2 | Decision Trees & Random Forest Bagging | **ILO1, ILO2, ILO5** (Leo Breiman Variance Reduction Theorem) |
| **Member 3** | Module 02-M3 | Gradient Boosting (LightGBM) & SHAP Game Theory | **ILO1, ILO3, ILO6** (Forward Stagewise Bias Reduction & Attribution) |
| **Member 4** | Module 02-M4 | PyTorch Tabular MLP & Entity Embeddings | **ILO1, ILO3, ILO6** (Manifold Learning & Small-Sample Overfitting) |
| **All Members** | Module 01 & 03 | Preprocessing, Anti-Leakage Pipeline & Master Evaluation | **ILO1–ILO6** (Full Pipeline Synthesis & Critical Discussion) |

---
*Created for NTU CCDS CA6001 Course Assignment (October 2026).*
