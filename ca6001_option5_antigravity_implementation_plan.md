# NTU CCDS CA6001 Group Project Implementation Plan (Option 5 — Jupyter Notebook Spec)
# 南洋理工大学 CCDS 学院 CA6001 小组作业代码与项目实施计划（Jupyter Notebook 重构版）

## 1. Project Charter & Dataset Specification | 项目章程与数据集规范

- **Course | 课程名称**: CA6001 – AI Algorithms Fundamentals & Application (50% Course Grade | 占课程总成绩 50%)
- **Project Topic | 课题名称**:
  - **EN**: Predicting Graduate Employment Outcomes and Starting Salary Tiers Across Singapore's Six Autonomous Universities: A Comparative Study of Linear, Tree-Based Ensemble, and Deep Tabular Learning Models
  - **ZH**: 新加坡六所公立大学毕业生就业率与起薪梯队预测：基于线性回归、树模型集成与深度表格学习的对比研究
- **Official Dataset | 官方数据源**:
  - **Primary Source | 主数据源**: [Graduate Employment Survey (GES) - NTU, NUS, SIT, SMU, SUSS & SUTD (Data.gov.sg)](https://data.gov.sg/dataset/graduate-employment-survey-ntu-nus-sit-smu-suss-sutd) (`d_3c55210de27fcccda2ed0c63fdd2b352`, 2013–2024, 1,550 raw rows / 1,445 clean rows × 12 columns)
  - **Alternative Mirror | 开源镜像**: [Singapore Graduate Employment Survey (Kaggle)](https://www.kaggle.com/datasets/shahnazrasheed/singapore-graduate-employment-survey)
  - **Macroeconomic Enrichment | 宏观特征补充**: [Singapore Statistics (SingStat)](https://www.singstat.gov.sg/)（内置静态 Lookup 字典，确保离线 100% 可复现）
- **Submission Deadline | 最终提交截止时间**: October 28, 2026, 11:59 PM via NTULearn | 2026年10月28日 23:59（通过 NTULearn 提交）

| Deliverable | Hard Limit | Code & Artifact Output in Repository |
| :--- | :--- | :--- |
| **1. Final Report (最终报告)** | Max 10 pages (excl. title page) | `outputs/figures/*.png` + `outputs/metrics/master_comparison.csv` |
| **2. Presentation Slides (演示文稿)** | Max 12 slides (excl. title slide) | High-DPI visualizations (ROC overlay, SHAP summary, Boxplots) |
| **3. Audio / Video (汇报音视频)** | Max 15 minutes (~3.5 min/member) | Executable `.ipynb` cell outputs & model comparison charts |
| **4. Source Code (项目源码)** | Reproducible Notebook repository | Clean, fully run `.ipynb` files in `notebooks/` with `README.md` |

---

## 2. Jupyter Notebook Repository Architecture | Notebook 团队协同目录结构

为了最大化提升 4 人团队的协同效率与同步便利性，避免传统 `.py` 脚本在多环境调试与图表查看时的摩擦，本项目全面采用 **模块化 Jupyter Notebook (`.ipynb`)** 架构。

每份 Notebook 均采用“**独立可运行、内嵌交互可视化、自动持久化高保真图表与指标**”的设计标准：

```text
ca6001-ges-project/
├── data/
│   ├── raw/
│   │   └── GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv   # Raw CSV from Data.gov.sg
│   └── processed/
│       ├── ges_cleaned_features.csv                           # Cleaned & lag-engineered dataset (1,445 rows)
│       ├── train_temporal_2013_2022.csv                       # Temporal train split (2013–2022, 1,170 rows)
│       └── test_temporal_2023_2024.csv                        # Temporal test split (2023–2024, 275 rows)
├── notebooks/
│   ├── 01_data_preprocessing_and_eda.ipynb                    # Step 1: Preprocessing & EDA (Pure English)
│   ├── 01_data_preprocessing_and_eda_bilingual.ipynb          # Step 1: Preprocessing & EDA (Bilingual EN/ZH)
│   ├── 02_m1_logistic_baseline.ipynb                          # Step 2: Logistic Baseline & Ridge (Pure English)
│   ├── 02_m1_logistic_baseline_bilingual.ipynb                # Step 2: Logistic Baseline & Ridge (Bilingual EN/ZH)
│   ├── 02_m2_bagging_random_forest.ipynb                      # Step 3: Decision Tree vs. Random Forest (Pure English)
│   ├── 02_m2_bagging_random_forest_bilingual.ipynb            # Step 3: Decision Tree vs. Random Forest (Bilingual EN/ZH)
│   ├── 02_m3_boosting_gbdt_shap.ipynb                         # Step 4: LightGBM & SHAP (Pure English)
│   ├── 02_m3_boosting_gbdt_shap_bilingual.ipynb               # Step 4: LightGBM & SHAP (Bilingual EN/ZH)
│   ├── 02_m4_deep_tabular_mlp.ipynb                           # Step 5: PyTorch Entity-Embedding MLP (Pure English)
│   ├── 02_m4_deep_tabular_mlp_bilingual.ipynb                 # Step 5: PyTorch Entity-Embedding MLP (Bilingual EN/ZH)
│   ├── 03_model_evaluation_and_comparison.ipynb               # Step 6: Master evaluation, overlaid ROC/PR, Ensemble (Pure English)
│   └── 03_model_evaluation_and_comparison_bilingual.ipynb     # Step 6: Master evaluation, overlaid ROC/PR, Ensemble (Bilingual EN/ZH)
├── outputs/
│   ├── figures/                                               # All 300-DPI PNG charts for Report & Slides
│   └── metrics/                                               # Standardized JSON & CSV metric outputs per model
├── requirements.txt                                           # Python dependencies (including Jupyter stack)
└── README.md                                                  # Execution sequence, reproducibility guide & team allocation
```

### 依赖环境清单 (`requirements.txt`)

```text
jupyter>=1.0.0
ipykernel>=6.25.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
shap>=0.42.0
torch>=2.0.0
```

---

## 3. Data Schema & Anti-Leakage Feature Engineering Spec | 数据字典与防泄露特征规范

### 3.1 Raw Column Dictionary & Sanitization Rules (`notebooks/01_data_preprocessing_and_eda.ipynb`)

原始 MOE GES 数据集共包含 12 列。由于小样本专业（应答人数 `< 30`）在原始 CSV 中填入了字符串 `na` 或 `**`，所有数值列均被存储为文本，必须在 Notebook 01 中执行以下清洗规范：

| Raw Column Name | Role | Null / Dirty Rate | Preprocessing & Sanitization Logic |
| :--- | :--- | :--- | :--- |
| `year` | Input Feature | 0.0% | 转换为整型（`2013`–`2024`）。 |
| `university` | Input Feature | 0.0% | 去除脚注符号（`*`, `#`, `^`）；包含 6 所公立大学（`NTU`, `NUS`, `SIT`, `SMU`, `SUSS`, `SUTD`）。 |
| `school` | Input Feature | 2.1% | 填充缺失值（主要为 SMU 直属课程），统一为 `"SMU Direct Programme"`，剥离脚注符号。 |
| `degree` | Input Feature | 0.0% | 提取布尔特征（Honours、Double Degree、Follow-up 等）后，剥离括号注释与特殊符号生成 `degree_clean`。 |
| `employment_rate_overall` | Survey Outcome | 5.9% (`na`/`**`) | 通过 `pd.to_numeric(..., errors="coerce")` 转为数值。**严禁作为当期 $X_t$ 输入**。 |
| `employment_rate_ft_perm` | Survey Outcome | 5.9% (`na`/`**`) | 转为浮点数。仅用于定义当期分类标签 `target_binary` 以及跨年滞后特征 `lag1_ft_emp_rate`。 |
| `basic_monthly_mean` | Survey Outcome | 5.9% (`na`/`**`) | 转为浮点数。**严格剔除，禁止进入特征矩阵 $X_t$（同源目标泄露）**。 |
| `basic_monthly_median` | Survey Outcome | 5.9% (`na`/`**`) | 转为浮点数。**严格剔除，禁止进入特征矩阵 $X_t$（同源目标泄露）**。 |
| `gross_monthly_mean` | Survey Outcome | 5.9% (`na`/`**`) | 转为浮点数。**严格剔除，禁止进入特征矩阵 $X_t$（同源目标泄露）**。 |
| `gross_monthly_median` | Primary Target | 5.9% (`na`/`**`) | 转为浮点数。剔除缺失目标值的脏数据行（最终保留 1,445 条高质量有效样本）。 |
| `gross_monthly_25_percentile` | Survey Outcome | 5.9% (`na`/`**`) | 转为浮点数。仅用于衍生 $t-1$ 年薪酬离散度指标 `lag1_salary_iqr`。 |
| `gross_monthly_75_percentile` | Survey Outcome | 5.9% (`na`/`**`) | 转为浮点数。仅用于衍生 $t-1$ 年薪酬离散度指标 `lag1_salary_iqr`。 |

### 3.2 Strict Anti-Leakage Feature Engineering Pipeline (防泄露特征工程三大模块)

> **CRITICAL ANTI-LEAKAGE CONTRACT (防同源特征泄露铁律)**:
> 预测第 $t$ 年毕业生结果的特征矩阵 $X_t$，**严禁包含当年任何薪酬与就业调研指标**。所有调研历史表现必须严格向后平移至少 1 年（即仅使用 $t-1$ 及以前的已知历史）。

1. **Module A — 领域与正则结构特征 (Domain & Regex Structural Features)**:
   - `is_double_degree`: 正则匹配 `r"(&| and |Double|Dual)"`，双学位/双主修标记为 `1`，其余为 `0`。
   - `is_honours`: 正则匹配 `r"(Honours|Cum Laude|Hons)"` 标记为 `1`。
   - `is_follow_up`: 针对医科、法学、建筑等需要后续实习考证的跟踪专业标记为 `1`。
   - `domain_cluster`: 关键词映射至 7 大宏观学科集群（`Computing_AI`, `Engineering`, `Business_Finance`, `Medicine_Health`, `Law_PublicPolicy`, `Science_Math`, `Humanities_Design`）。
   - `tfidf_deg_01` ... `tfidf_deg_15`: 对 `degree_clean` 进行 TF-IDF 特征提取（前 15 维高频专业词汇，仅在训练集拟合）。

2. **Module B — 跨年滞后特征 ($t-1$) (Historical Lag Features)**:
   - 按 `["university", "degree_clean", "year"]` 排序并分组进行 `shift(1)`：
     - `lag1_gross_median`: 上一届该专业全职起薪中位数；
     - `lag1_ft_emp_rate`: 上一届该专业全职固定就业率；
     - `lag1_salary_iqr`: 上一届薪酬四分位差（75分位 - 25分位，衡量薪资离散度）。
   - **分层冷启动回填（Hierarchical Cold-Start Imputation）**:
     - 新设专业无历史记录时，依次按 `(year - 1, university, domain_cluster)` 中位数 -> `(year - 1, domain_cluster)` 中位数 -> 历史同期学科中位数分层填补；
     - 添加指示变量 `has_lag_history`（真实历史=1，插补填充=0）。

3. **Module C — 时间趋势与宏观经济背景 (Temporal & Macroeconomic Context)**:
   - `years_since_2013`: `year - 2013`（捕获长期通胀与薪酬上涨趋势）；
   - `macro_regime`: 宏观周期划分（`"Pre_COVID_2013_2019"`, `"COVID_Shock_2020_2021"`, `"Post_COVID_Expansion_2022_2025"`，涵盖至当前最新 2024 调研年）；
   - `sg_cpi_yoy` & `sg_gdp_growth`: 新加坡年均 CPI 同比与 GDP 增速静态 Lookup 字典映射。

### 3.3 预测任务与标签定义 (Target Definitions)

- **主任务 (Task A — High-Competitiveness Binary Classification，全组 4 个模型对比主线)**:
  - 计算各日历年 $t$ 内部的起薪 70 分位点（`q70_year`）；
  - `target_binary = 1` 当且仅当 `(gross_monthly_median >= q70_year) and (employment_rate_ft_perm >= 85.0)`，否则为 `0`；
  - **正负比例**: 正样本占比约 24%–27%（约 1:3 不平衡），能真实检验模型在少数类上的 **Recall**、**Precision**、**F1** 与 **PR-AUC**。
- **扩展任务 (Task B & Task C，供 Notebook 探究与附录拓展)**:
  - `target_tier`: 3 类薪资梯队（`< 3600`, `3600–4500`, `> 4500`）；
  - `target_reg`: 连续起薪数值 `gross_monthly_median`。

---

## 4. Work Breakdown Structure (WBS) & 4-Member Notebook Allocation | 四人分工与 Notebook 规范

每位组员负责一个独立的 Jupyter Notebook，所有 Notebook 共享相同的数据输入接口（`data/processed/ges_cleaned_features.csv`）与标准化评估输出接口：

| Member | Notebook File | Core Models & Algorithms | Key Hyperparameters to Tune | Key Visual Artifacts (`outputs/figures/`) & Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **Member 1**<br>(Data Lead & Linear Baseline) | `notebooks/01_data_preprocessing_and_eda.ipynb`<br>`notebooks/02_m1_logistic_baseline.ipynb` | 数据清洗工程 + `LogisticRegression` (L1/L2 正则) + `Ridge` | 正则强度 $C \in [0.01, 0.1, 1.0, 10.0]$<br>惩罚项 `penalty=['l1', 'l2']`<br>`class_weight=[None, 'balanced']` | `eda_salary_trends.png`<br>`eda_university_domain_boxplot.png`<br>`m1_logistic_coefficients.png`<br>`outputs/metrics/m1_metrics.json` |
| **Member 2**<br>(Bagging Tree Lead) | `notebooks/02_m2_bagging_random_forest.ipynb` | `DecisionTreeClassifier` (无剪枝 vs 剪枝) vs. `RandomForestClassifier` (Bagging 集成) | 树深度 `max_depth=[2, 4, 6, 8, 12, None]`<br>树数量 `n_estimators=[50, 100, 200, 300]`<br>最大特征数 `max_features=['sqrt', 'log2', 0.5]` | `m2_depth_overfitting_curve.png`<br>（展现深浅树训练/验证 F1 变化，论证方差缩减 Variance Reduction）<br>`outputs/metrics/m2_metrics.json` |
| **Member 3**<br>(Boosting Tree Lead) | `notebooks/02_m3_boosting_gbdt_shap.ipynb` | `CatBoostClassifier` / `XGBClassifier` / `LGBMClassifier` + SHAP 特征归因 | 学习率 `learning_rate=[0.03, 0.05, 0.1]`<br>`max_depth=[3, 5, 7]`<br>不平衡权重 `scale_pos_weight=[1.0, 2.5, 3.0]` | `m3_shap_summary.png`<br>（SHAP Beeswarm 图揭示非线性特征重要性与交互归因）<br>`outputs/metrics/m3_metrics.json` |
| **Member 4**<br>(Deep Learning Lead) | `notebooks/02_m4_deep_tabular_mlp.ipynb` | PyTorch `TabularEmbeddingMLP`（`nn.Embedding` 嵌入离散变量 + BatchNorm + Dropout + Linear） | 隐藏层结构 `hidden_dims=[[64, 32], [128, 64, 32]]`<br>`dropout=[0.0, 0.3, 0.5]`<br>`weight_decay=[1e-4, 1e-3]` | `m4_train_val_loss_auc.png`<br>（训练/验证损失与 AUC 曲线对比，讨论小样本表格深度学习正则化）<br>`outputs/metrics/m4_metrics.json` |

---

## 5. Standardized Notebook Evaluation Protocol | 统一 Notebook 评估协议

为了确保各 Notebook 指标能够无缝聚合比对，各模型 Notebook 均内嵌以下标准评估函数逻辑：

```python
# Standardized Evaluation Protocol (Shared Across Notebooks M1 to M4)
# 1. 5-Fold Stratified Cross-Validation (评估模型内生稳定性)
#    StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# 2. Temporal Out-of-Time Split (评估未来泛化性)
#    Train Set: year <= 2022 (~70%)
#    Test Set:  year >= 2023 (~30%)
# 3. 输出指标字典并持久化至 outputs/metrics/{model_name}_metrics.json:
#    {
#      "model_name": "M1_Logistic_L2",
#      "cv_5fold": {
#        "accuracy_mean": ..., "accuracy_std": ...,
#        "precision_mean": ..., "precision_std": ...,
#        "recall_mean": ..., "recall_std": ...,
#        "f1_mean": ..., "f1_std": ...,
#        "roc_auc_mean": ..., "roc_auc_std": ...
#      },
#      "temporal_test": {
#        "accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "roc_auc": ...,
#        "y_true": [...], "y_pred": [...], "y_prob": [...]
#      }
#    }
```

聚合分析 Notebook（`notebooks/03_model_evaluation_and_comparison.ipynb`）读取 `outputs/metrics/` 下的 4 份 JSON 文件，自动生成：
1. `outputs/metrics/master_comparison_table.csv`（全指标主对比表）；
2. `outputs/figures/all_models_roc_pr_curves.png`（4 模型同图叠加的 ROC 曲线与 PR 曲线）；
3. `outputs/figures/all_models_confusion_matrices.png`（2x2 混淆矩阵图）。

---

## 6. Step-by-Step Prompts for Antigravity IDE (Notebook 版) | 逐步执行指令模板

在 Antigravity IDE 中，你可以依次运行以下 **6 个步骤 Prompt**，让 Agent 直接生成并运行完整的 Jupyter Notebooks：

### Step 1 Prompt: 创建预处理与 EDA Notebook (`notebooks/01_data_preprocessing_and_eda.ipynb`)

```markdown
Please create and execute `notebooks/01_data_preprocessing_and_eda.ipynb` for our NTU CA6001 GES project:
1. Initialize directories `data/raw`, `data/processed`, `outputs/figures`, `outputs/metrics`.
2. Load `data/raw/GraduateEmploymentSurveyNTUNUSSITSMUSUSSSUTD.csv`. If not present, download it from Data.gov.sg or load from workspace.
3. Apply all sanitization rules: strip footnote characters (`*`, `#`, `^`, `##`), coerce `na`/`**` to `NaN`, drop records with missing targets (~1,500 rows remain).
4. Build the 3 strict anti-leakage feature modules:
   - Module A: Domain regex flags (`is_double_degree`, `is_honours`, `is_follow_up`), 7 domain clusters, and TF-IDF degree features.
   - Module B: 1-year historical lag features (`lag1_gross_median`, `lag1_ft_emp_rate`, `lag1_salary_iqr`) with hierarchical cold-start imputation and `has_lag_history`.
   - Module C: Macroeconomic context with embedded SingStat CPI and GDP dictionaries.
5. Create targets: `target_binary` (annual top 30% gross median & FT employment >= 85%), `target_tier`, and `target_reg`.
6. Save `data/processed/ges_cleaned_features.csv`, `train_temporal_2013_2022.csv`, and `test_temporal_2023_2024.csv`.
7. Generate and save 3 publication-grade 300-DPI charts to `outputs/figures/`:
   - `eda_salary_trends.png`: Longitudinal median salary trend (2013-2024) across universities.
   - `eda_university_domain_boxplot.png`: Salary distribution across the 6 universities & 7 domain clusters.
   - `eda_correlation_heatmap.png`: Spearman correlation matrix of processed features.
```

### Step 2 Prompt: 创建回归与线性基线模型 Notebook (`notebooks/02_m1_logistic_baseline.ipynb`)

```markdown
Please create and execute `notebooks/02_m1_logistic_baseline.ipynb` for Member 1 (Linear & Logistic Regression Baseline Lead):
1. Load `data/processed/ges_cleaned_features.csv`, `train_temporal_2013_2022.csv`, and `test_temporal_2023_2024.csv`.
2. Build a scikit-learn Pipeline with `ColumnTransformer`:
   - One-Hot Encoding for categorical features (`uni_code`, `domain_cluster`, `macro_regime`).
   - StandardScaler for continuous numerical and lag features.
   - Pass-through for pre-engineered TF-IDF features and binary indicators.
3. Implement and tune models on `target_binary`:
   - Compare `LogisticRegression` with L1 (Lasso) vs. L2 (Ridge) penalty across C = [0.01, 0.1, 1.0, 10.0] with `class_weight='balanced'`.
   - Also implement Ridge Regression on continuous `target_reg` as an auxiliary baseline.
4. Evaluate using the Standardized Protocol:
   - Stratified 5-Fold Cross-Validation on the training split (mean & std for Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
   - Temporal Out-of-Time Test Set evaluation on cohorts 2023–2024.
5. Export artifacts:
   - Pure English 300-DPI feature coefficients bar plot to `outputs/figures/m1_logistic_coefficients.png` (render inline in high-res).
   - Save standardized metric dictionary to `outputs/metrics/m1_metrics.json`.
```

### Step 3 Prompt: 创建 Bagging 树模型（决策树 vs 随机森林）Notebook (`notebooks/02_m2_bagging_random_forest.ipynb`)

```markdown
Please create and execute `notebooks/02_m2_bagging_random_forest.ipynb` for Member 2 (Bagging Tree Lead):
1. Load `data/processed/ges_cleaned_features.csv`, `train_temporal_2013_2022.csv`, and `test_temporal_2023_2024.csv`.
2. Preprocess features using `ColumnTransformer` (Ordinal/One-Hot encoding for trees).
3. Algorithmic Comparison (demonstrating Variance Reduction):
   - Single `DecisionTreeClassifier`: Evaluate across tree depths `max_depth = [2, 4, 6, 8, 10, 15, None]` to demonstrate unpruned overfitting.
   - Bagging Ensemble `RandomForestClassifier`: Tune `n_estimators = [100, 200, 300]`, `max_depth = [4, 6, 8, 12, None]`, and `max_features = ['sqrt', 'log2', 0.5]`.
4. Evaluate using the Standardized Protocol:
   - Stratified 5-Fold CV on training split + Temporal Out-of-Time Test Set on cohorts 2023–2024.
   - Record Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC.
5. Export artifacts:
   - Plot Train vs. Validation F1 Score curves across tree depths to illustrate Variance Reduction and save to `outputs/figures/m2_depth_overfitting_curve.png` (pure English labels, 300 DPI, render inline).
   - Save standardized metric dictionary to `outputs/metrics/m2_metrics.json`.
```

### Step 4 Prompt: 创建 Boosting 提升树模型与 SHAP 解释 Notebook (`notebooks/02_m3_boosting_gbdt_shap.ipynb`)

```markdown
Please create and execute `notebooks/02_m3_boosting_gbdt_shap.ipynb` for Member 3 (Boosting Tree Lead):
1. Load `data/processed/ges_cleaned_features.csv`, `train_temporal_2013_2022.csv`, and `test_temporal_2023_2024.csv`.
2. Implement modern gradient boosting frameworks (`CatBoostClassifier`, `XGBClassifier`, and `LGBMClassifier`):
   - Tune hyperparameters: `learning_rate = [0.03, 0.05, 0.1]`, `max_depth = [3, 5, 7]`, and `scale_pos_weight = [1.0, 2.5, 3.0]` for class imbalance.
   - Demonstrate Bias Reduction through sequential error correction.
3. Evaluate using the Standardized Protocol:
   - Stratified 5-Fold CV on training split + Temporal Out-of-Time Test Set on cohorts 2023–2024.
   - Save standardized metric dictionary to `outputs/metrics/m3_metrics.json`.
4. Model Explainability & Feature Attribution:
   - Calculate TreeExplainer SHAP values on the temporal test set.
   - Export pure English 300-DPI SHAP Beeswarm summary plot showing top 15 features to `outputs/figures/m3_shap_summary.png` (render inline).
```

### Step 5 Prompt: 创建深度表格学习 PyTorch MLP Notebook (`notebooks/02_m4_deep_tabular_mlp.ipynb`)

```markdown
Please create and execute `notebooks/02_m4_deep_tabular_mlp.ipynb` for Member 4 (Deep Learning Lead):
1. Load `data/processed/ges_cleaned_features.csv`, `train_temporal_2013_2022.csv`, and `test_temporal_2023_2024.csv`.
2. Construct PyTorch Tabular Architecture (`TabularEmbeddingMLP`):
   - `nn.Embedding` for categorical inputs (`uni_code`, `domain_cluster`, `macro_regime`).
   - Concatenate embeddings with standardized continuous numerical inputs.
   - Network Backbone: Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear -> BatchNorm1d -> ReLU -> Linear.
   - Compare unregularized (`dropout=0.0`) vs. regularized (`dropout=0.3`, `weight_decay=1e-3`) to investigate tabular deep learning overfitting on small datasets (~1.5k rows).
3. Evaluate using the Standardized Protocol:
   - Stratified 5-Fold CV on training split + Temporal Out-of-Time Test Set on cohorts 2023–2024.
   - Save standardized metric dictionary to `outputs/metrics/m4_metrics.json`.
4. Export artifacts:
   - Plot Training vs. Validation Loss and ROC-AUC curves across 50 epochs to `outputs/figures/m4_train_val_loss_auc.png` (pure English labels, 300 DPI, render inline).
```

### Step 6 Prompt: 创建汇总对比与汇报可视化 Notebook (`notebooks/03_model_evaluation_and_comparison.ipynb`)

```markdown
Please create and execute `notebooks/03_model_evaluation_and_comparison.ipynb` (Master Evaluation & Comparative Synthesis):
1. Ingest metric JSON files from `outputs/metrics/` (`m1_metrics.json` through `m4_metrics.json`).
2. Generate Master Comparison Table:
   - Export `outputs/metrics/master_comparison_table.csv` and display styled table comparing 5-Fold CV (Mean ± Std) and Temporal Test metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC) across M1, M2, M3, and M4.
3. Publication Comparative Visualizations:
   - Plot overlaid ROC Curves and Precision-Recall (PR) Curves across all 4 models on identical axes (`outputs/figures/all_models_roc_pr_curves.png`, pure English, 300 DPI, render inline).
   - Plot 2x2 Annotated Confusion Matrices grid comparing all 4 models on the temporal test set (`outputs/figures/all_models_confusion_matrices.png`, pure English, 300 DPI, render inline).
4. Academic Synthesis Cell:
   - Comprehensive comparative discussion mapping to CA6001 ILO1–ILO6 (Linear bias vs. Tree ensemble flexibility vs. Deep tabular challenges on small N=1.5k).
```

---

## 7. 12-Day Execution Checklist & Report Blueprint | 12 天推进清单与报告结构映射

### 7.1 12-Day Sprint Checklist (Oct 16 – Oct 28, 2026)

- [ ] **Days 1–3 (Oct 16 – Oct 18) — Notebook 01 数据清洗与特征冻结 (Gate 1)**: 完成 Step 1 (`01_data_preprocessing_and_eda.ipynb`)，确认 `ges_cleaned_features.csv` 零当期泄露，导出 3 张 EDA 基础图，锁定训练/测试集。
- [ ] **Days 4–7 (Oct 19 – Oct 22) — 并行 4 大模型独立实验与调优 (Gate 2)**: 4 位组员分别按 Step 2–5 独立开发与运行各自的 Notebook（`02_m1` 至 `02_m4`），产出各模型的独立图表与 JSON 指标文件；随后执行 Step 6 运行 `03_model_evaluation_and_comparison.ipynb` 导出主对比表与综合对比图。
- [ ] **Days 8–10 (Oct 23 – Oct 25) — 10-Page Final Report 报告起草**: 基于 Notebooks 内嵌的结论与 `outputs/figures/` 高清图撰写 10 页报告（不含标题页），逐节对应 **ILO1–ILO6**。
- [ ] **Days 9–11 (Oct 24 – Oct 26) — 12-Slide 汇报幻灯片制作**: 提取 Notebook 核心图表（ROC/PR 对比、SHAP 图、过拟合曲线）制作 12 页精美 PPT（不含标题页）。
- [ ] **Days 11–12 (Oct 27 – Oct 28) — 音视频录制与 NTULearn 提交**: 4 位组员录制 15 分钟汇报音视频（每人约 3.5 分钟），打包 Notebook 仓库，于 **2026年10月28日 23:59 前** 完成 NTULearn 提交。

### 7.2 10-Page Report & 12-Slide Mapping to CA6001 Rubric

| Report Pages | Slide Numbers | Section Title (Bilingual) | Required Code/Figure Artifacts from `outputs/` | Assessed ILOs & Weight |
| :--- | :--- | :--- | :--- | :--- |
| **Page 1** | **Slides 1–2** | **1. Introduction & Problem Definition**<br>问题背景与任务定义 | Dataset summary table from Notebook 01 | Problem Analysis (35%) |
| **Pages 2–3** | **Slides 3–4** | **2. Dataset Analysis & Anti-Leakage Engineering**<br>数据探索与防泄露特征工程 | `eda_salary_trends.png`, `eda_university_domain_boxplot.png` | **ILO4** (35% Design) |
| **Pages 4–5** | **Slides 5–7** | **3. Algorithmic Principles & Comparative Justification**<br>四类核心算法原理与适用性论证 | Algorithm Architecture table + `m1_logistic_coefficients.png` | **ILO1 & ILO2** (35% Design) |
| **Pages 6–8** | **Slides 8–10** | **4. Experimental Setup, Tuning & Performance Evaluation**<br>实验设计、调优与多维性能评估 | `master_comparison_table.csv`, `all_models_roc_pr_curves.png`, `all_models_confusion_matrices.png`, `m3_shap_summary.png` | **ILO3, ILO5 & ILO6** (35% Implementation) |
| **Pages 9–10** | **Slides 11–12** | **5. Critical Discussion, Limitations & Conclusion**<br>深度讨论、算法局限性与总结展望 | `m2_depth_overfitting_curve.png`, `m4_train_val_loss_auc.png` | **ILO2 & ILO6** (20% Report & Slides) |
