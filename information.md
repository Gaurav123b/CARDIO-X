# CARDIO-X — System Architecture, Methodological Specifications & Technical Reference

> **Project Title:** An Intelligent ML-Driven System for Cholesterol Risk Detection, Prevention & Personalized Lifestyle Advisory (CARDIO-X)  
> **Target Problem:** Primary Prevention 10-Year Cardiovascular Disease (CVD) Risk Prediction  
> **Dataset Source:** Longitudinal Framingham Heart Study (FHS)  
> **Implementation Language:** Python 3.13 (Compatible with Python 3.10+)  
> **Repository:** `https://github.com/Gaurav123b/CARDIO-X`

---

## Table of Contents

1. [Executive Summary & Clinical Rationale](#1-executive-summary--clinical-rationale)
2. [Technology Stack & Architectural Principles](#2-technology-stack--architectural-principles)
3. [End-to-End Execution & Output Guide](#3-end-to-end-execution--output-guide)
4. [Dataset Architecture & Dictionary of Clinical Features](#4-dataset-architecture--dictionary-of-clinical-features)
5. [Data Quality Assurance System (`src/data_quality.py`)](#5-data-quality-assurance-system-srcdata_qualitypy)
6. [Exploratory Data Analysis System (`src/eda.py`)](#6-exploratory-data-analysis-system-srcedapy)
7. [Mathematical Formulations & Clinical Feature Engineering](#7-mathematical-formulations--clinical-feature-engineering)
8. [Target Variable Formulation & Primary Prevention Cohort Selection](#8-target-variable-formulation--primary-prevention-cohort-selection)
9. [Preprocessing & Pipeline Isolation Architecture (`src/preprocessing.py`)](#9-preprocessing--pipeline-isolation-architecture-srcpreprocessingpy)
10. [Comprehensive File-by-File Technical Directory](#10-comprehensive-file-by-file-technical-directory)
11. [Verification & Automated Test Suite (`tests/`)](#11-verification--automated-test-suite-tests)
12. [Frequently Asked Questions & Prompt-Learning Guide](#12-frequently-asked-questions--prompt-learning-guide)

---

## 1. Executive Summary & Clinical Rationale

Cardiovascular diseases (CVD) remain the leading cause of morbidity and premature mortality worldwide. A fundamental principle of modern preventive cardiology is **primary prevention**: identifying individuals at high risk of developing cardiovascular events *before* clinical onset occurs, thereby enabling targeted lifestyle modifications and pharmacological interventions (e.g., lipid-lowering statin therapy, antihypertensive management).

The **CARDIO-X** system implements an end-to-end, reproducible, leakage-free machine learning architecture designed to predict the **10-year risk of first incident cardiovascular disease** using baseline clinical measurements from the renowned **Framingham Heart Study (FHS)**.

### Clinical Mission & Objectives
- **Cohort Specialization:** Target strictly asymptomatic primary-prevention participants with no baseline history of Coronary Heart Disease (CHD), Angina Pectoris, Myocardial Infarction (MI), or Stroke.
- **Biomarker Modeling:** Analyze cholesterol pathways (`TOTCHOL`, HDL/LDL fractions), blood pressure dynamics (`SYSBP`, `DIABP`), metabolic indicators (`GLUCOSE`, `BMI`, `DIABETES`), and behavioral factors (`CURSMOKE`, `CIGPDAY`).
- **Mathematical Integrity:** Eliminate multicollinearity between raw blood pressures and derived hemodynamic parameters (Pulse Pressure, Mean Arterial Pressure).
- **Zero-Leakage Assurance:** Fit all imputers, scalers, and transformation parameters exclusively within training splits, locking the held-out test set until final validation.

---

## 2. Technology Stack & Architectural Principles

### Core Technology Stack

| Layer | Technology | Version | Purpose in CARDIO-X |
| :--- | :--- | :--- | :--- |
| **Runtime Language** | Python | `3.13.9` (>=3.10) | Base execution environment; leverages strict type annotations (`typing`). |
| **Tabular Data Processing** | `pandas` | `3.0.6` | Data loading, filtering, longitudinal grouping, attrition tracking, CSV reports. |
| **Vectorized Computation** | `numpy` | `2.5.4` | High-performance numerical computations, quantile derivations, boolean masking. |
| **Machine Learning & Pipelines** | `scikit-learn` | `1.9.1` | `Pipeline`, `ColumnTransformer`, `RobustScaler`, `SimpleImputer`, `StratifiedShuffleSplit`. |
| **Scientific Visualization** | `matplotlib` | `3.11.2` | Core plotting engine for distributions, trajectory plots, and multi-panel figures. |
| **Statistical Data Graphics** | `seaborn` | `0.13.2` | Heatmaps, violin plots, regression lines, and correlation matrices. |
| **Testing & Quality Assurance** | `pytest` | `9.1.1` | Automated regression testing, boundary condition assertions, and leakage checks. |
| **Interactive Research** | `jupyter` / `IPython` | `1.1.1` | Exploratory data analysis notebooks for visual and interactive validation. |

### Architectural Guardrails

```
CARDIO-X PIPELINE ARCHITECTURE

[Raw Data: FraminghamDataset.csv] (Immutable, SHA-256 Verified)
              │
              ▼
   ┌──────────────────────┐
   │ src/data_quality.py  │ ──► Generates reports/data_quality/ (CSVs & Markdown)
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │     src/eda.py       │ ──► Generates reports/figures/ (7 PNGs) & reports/eda_summary.md
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ src/preprocessing.py │
   └──────────┬───────────┘
              │
    ┌─────────┴─────────┐
    │  Cohort Filtering │ ──► Baseline (4,434) ➔ -219 Prior Disease ➔ -232 Indeterminate
    │ & Target Labeling │     ➔ Evaluable Cohort: 3,983 (369 Positive, 3,614 Negative)
    └─────────┬─────────┘
              │
    ┌─────────┴────────────────────────┐
    │ Stratified 80/20 Train/Test Split │
    └─────────┬────────────────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
┌───────────┐   ┌───────────┐
│ Train Set │   │ Test Set  │
│  (3,186)  │   │   (797)   │
└─────┬─────┘   └─────┬─────┘
      │               │
      ▼               ▼
┌───────────┐   ┌───────────┐
│ Fit &     │   │ Transform │ (No parameter estimation on test set;
│ Transform │   │   ONLY    │  Zero Data Leakage guaranteed)
└───────────┘   └───────────┘
```

1. **Source Immutability:** The raw dataset is strictly read-only. Before and after every execution, the file's SHA-256 hash (`e073ce4c60d351f2015c500b5bdf89c3a99ab3dfd0ba525e31a743742d649131`) and byte size (`1,407,548 bytes`) are verified.
2. **Clinical Data Protection:** Raw patient records are strictly excluded via `.gitignore` (`data/*`, with exceptions only for `README.md` and `.gitkeep`).
3. **No Transformation Bleed:** Preprocessing transformations (median imputation, robust scaling) are fit **only** on the training set and applied downstream to the test set.

---

## 3. End-to-End Execution & Output Guide

### How to Run Each Component

All commands are executed from the repository root directory using the virtual environment:

```bash
# Activate Virtual Environment (macOS/Linux)
source .venv/bin/activate

# 1. Verify Environment & Directory Structure
python verify_env.py

# 2. Run Data Quality Analysis
python src/data_quality.py

# 3. Run Exploratory Data Analysis & Generate Figures
python src/eda.py

# 4. Run Preprocessing Pipeline & Stratified Train/Test Partitioning
python src/preprocessing.py

# 5. Run the Complete Automated Test Suite (25 Tests)
pytest tests/
```

### Where to Find the Output Results and What Form They Take

| Step | Script to Execute | Outputs Produced | Output Format & Location |
| :--- | :--- | :--- | :--- |
| **1. Environment** | `python verify_env.py` | Terminal verification log confirming Python 3.13, virtualenv activation, directory integrity, and dependency import status. | CLI stdout output |
| **2. Data Quality** | `python src/data_quality.py` | Complete tabular audits of missingness, descriptive stats, and longitudinal completeness. | 1. `reports/data_quality/data_quality_summary.md`<br>2. `reports/data_quality/missing_values.csv`<br>3. `reports/data_quality/descriptive_statistics.csv`<br>4. `reports/data_quality/dataset_overview.csv` |
| **3. EDA** | `python src/eda.py` | 7 publication-quality visual charts + detailed statistical summary markdown report. | 1. `reports/figures/fig01_distributions_key_biomarkers.png`<br>2. `reports/figures/fig02_missingness_by_period.png`<br>3. `reports/figures/fig03_correlation_matrix_baseline.png`<br>4. `reports/figures/fig04_correlation_lipid_panel_period3.png`<br>5. `reports/figures/fig05_longitudinal_trajectories.png`<br>6. `reports/figures/fig06_engineered_clinical_features.png`<br>7. `reports/figures/fig07_outcome_prevalence_comparison.png`<br>8. `reports/eda_summary.md` |
| **4. Preprocessing** | `python src/preprocessing.py` | Cohort flow attrition tables, target distribution validation, collinearity check, and stratified splits. | 1. Terminal stdout logging flow counts<br>2. `reports/preprocessing_design.md`<br>3. In-memory train/test arrays ready for modeling |
| **5. Testing** | `pytest tests/` | Automated assertion results for 25 tests verifying all mathematical rules, shapes, and isolation. | CLI test report (Green / Passed) |

---

## 4. Dataset Architecture & Dictionary of Clinical Features

### Longitudinal Structure
- **Total Records:** 11,627 clinical examination rows across 39 columns.
- **Unique Participants (`RANDID`):** 4,434 distinct individuals.
- **Examination Periods (`PERIOD`):**
  - `PERIOD == 1` (Exam 1, Baseline): 4,434 participants (100%).
  - `PERIOD == 2` (Exam 2, Follow-up ~6 years): 3,930 participants (88.6%).
  - `PERIOD == 3` (Exam 3, Follow-up ~12 years): 3,263 participants (73.6%).
  - Participants with complete data across all 3 exams: **3,206 individuals**.

### Data Dictionary & Variable Definitions

| Variable Name | Type | Unit / Coding | Clinical Interpretation & Modeling Role |
| :--- | :--- | :--- | :--- |
| `RANDID` | Integer | Identifier | Unique randomized participant identifier (**Excluded from predictors**). |
| `PERIOD` | Integer | 1, 2, 3 | Longitudinal examination cycle. (**Baseline models strictly use `PERIOD == 1`**). |
| `SEX` | Binary | 1 = Male; 2 = Female | Biological sex. Baseline distribution: 56.4% Female, 43.6% Male. |
| `AGE` | Continuous | Years | Age at examination (Baseline range: 32–70 years; Mean: 49.9 years). |
| `EDUCATION` | Categorical | 1, 2, 3, 4 | Highest educational attainment (1: <HS, 2: HS, 3: College, 4: Post-grad). |
| `CURSMOKE` | Binary | 0 = No; 1 = Yes | Current cigarette smoking status at exam (49.4% active smokers at baseline). |
| `CIGPDAY` | Continuous | Cigarettes/Day | Number of cigarettes smoked per day (0–90 cigs/day). |
| `TOTCHOL` | Continuous | mg/dL | Serum Total Cholesterol (Baseline range: 107–696 mg/dL; Median: 234 mg/dL). |
| `HDLC` | Continuous | mg/dL | High-Density Lipoprotein ("good") Cholesterol. **Note: 100% missing in Period 1 & 2; introduced only at Period 3.** |
| `LDLC` | Continuous | mg/dL | Low-Density Lipoprotein ("bad") Cholesterol. **Note: 100% missing in Period 1 & 2; introduced only at Period 3.** |
| `BMI` | Continuous | $\text{kg/m}^2$ | Body Mass Index (Baseline median: $25.4\text{ kg/m}^2$). |
| `SYSBP` | Continuous | mmHg | Systolic Blood Pressure (Baseline range: 83.5–295.0 mmHg; Median: 132.0 mmHg). |
| `DIABP` | Continuous | mmHg | Diastolic Blood Pressure (Baseline range: 30.0–142.5 mmHg; Median: 82.0 mmHg). |
| `BPMEDS` | Binary | 0 = No; 1 = Yes | Use of Anti-Hypertensive Medication at time of exam. |
| `HEARTRTE` | Continuous | Beats/Min | Resting heart rate in ventricular beats per minute. |
| `GLUCOSE` | Continuous | mg/dL | Casual serum glucose level (Baseline median: 80.0 mg/dL). |
| `DIABETES` | Binary | 0 = No; 1 = Yes | Clinically diagnosed diabetes mellitus. |
| `PREVCHD` | Binary | 0 = No; 1 = Yes | Pre-existing Coronary Heart Disease at exam (**Used for cohort exclusion**). |
| `PREVAP` | Binary | 0 = No; 1 = Yes | Pre-existing Angina Pectoris at exam (**Used for cohort exclusion**). |
| `PREVMI` | Binary | 0 = No; 1 = Yes | Pre-existing Myocardial Infarction at exam (**Used for cohort exclusion**). |
| `PREVSTRK` | Binary | 0 = No; 1 = Yes | Pre-existing Stroke at exam (**Used for cohort exclusion**). |
| `PREVHYP` | Binary | 0 = No; 1 = Yes | Pre-existing Hypertension at exam. |
| `CVD` | Binary | 0 = No; 1 = Yes | Cardiovascular disease event occurrence during entire surveillance (up to 24y). |
| `TIMECVD` | Continuous | Days | Time from baseline examination to first CVD event or censoring (0–8,766 days). |
| `DEATH` | Binary | 0 = No; 1 = Yes | All-cause death occurrence during surveillance. |
| `TIMEDTH` | Continuous | Days | Time from baseline examination to death or censoring (0–8,766 days). |

> **Crucial Discovery on Triglycerides:** Triglyceride levels were **not measured or recorded** in this dataset. Models cannot use triglyceride-derived metrics.
> **Crucial Discovery on HDL/LDL:** Period 1 (`PERIOD == 1`) contains 0 measurements of `HDLC` and `LDLC`. Therefore, 10-year baseline prediction models rely on `TOTCHOL` as the primary lipid biomarker.

---

## 5. Data Quality Assurance System (`src/data_quality.py`)

The data quality module executes non-destructive diagnostic evaluations across every variable in the dataset.

### Core Functions & Capabilities
1. `compute_dataset_overview(df)`:
   - Validates dimensions: 11,627 rows $\times$ 39 columns.
   - Calculates memory usage (approx. 3.5 MB) and identifies distinct data types (float64, int64).
2. `compute_missing_values(df)`:
   - Calculates exact null counts and percentages for each variable across the full dataset and per examination period.
   - At baseline (`PERIOD == 1`):
     - `GLUCOSE`: 388 missing ($8.75\%$)
     - `BPMEDS`: 593 missing ($13.37\%$)
     - `TOTCHOL`: 52 missing ($1.17\%$)
     - `CIGPDAY`: 32 missing ($0.72\%$)
     - `BMI`: 19 missing ($0.43\%$)
     - `HEARTRTE`: 1 missing ($0.02\%$)
3. `compute_descriptive_statistics(df)`:
   - Calculates parametric (mean, standard deviation) and non-parametric (median, IQR, min, max, skewness) metrics.
4. `analyze_cholesterol_completeness(df)`:
   - Documents the protocol shift where `HDLC` and `LDLC` were introduced at Exam 3, confirming that Period 1 has 0% completeness for lipoprotein fractions.
5. `check_longitudinal_integrity(df)`:
   - Evaluates participant retention and loss-to-follow-up across cycles.

---

## 6. Exploratory Data Analysis System (`src/eda.py`)

The exploratory data analysis system produces statistical summaries and 7 publication-ready figures (`reports/figures/`):

1. **`fig01_distributions_key_biomarkers.png`**:
   - Multi-panel distribution plots (KDE + Histograms) of `TOTCHOL`, `SYSBP`, `DIABP`, `BMI`, `GLUCOSE`, and `AGE`.
   - Illustrates severe right-skewness in `GLUCOSE` (skewness = 6.21) and `TOTCHOL` (skewness = 1.05), providing the rationale for non-parametric imputers (`SimpleImputer(strategy='median')`) and outlier-resistant scalers (`RobustScaler`).
2. **`fig02_missingness_by_period.png`**:
   - Heatmap depicting missingness rates across all 3 examination periods.
3. **`fig03_correlation_matrix_baseline.png`**:
   - Pearson correlation heatmap for all baseline clinical variables.
   - Reveals collinearity between `SYSBP` and `DIABP` ($r = 0.70$).
4. **`fig04_correlation_lipid_panel_period3.png`**:
   - Analyzes correlations among the full lipid panel (`TOTCHOL`, `HDLC`, `LDLC`) at Exam 3 ($N=3,263$), demonstrating that `TOTCHOL` is strongly correlated with atherogenic `LDLC` ($r = 0.91$).
5. **`fig05_longitudinal_trajectories.png`**:
   - Follows the $N=3,206$ participant cohort who attended all 3 cycles, demonstrating longitudinal age-related increases in systolic blood pressure and total cholesterol.
6. **`fig06_engineered_clinical_features.png`**:
   - Evaluates distributions and outcome relationships of derived physiological measures (`PULSE_PRESSURE` and `MAP`).
7. **`fig07_outcome_prevalence_comparison.png`**:
   - Compares cumulative lifetime events (24-year surveillance) against the 10-year landmark boundary.

---

## 7. Mathematical Formulations & Clinical Feature Engineering

### 1. The 10-Year Temporal Boundary Threshold
The dataset records time-to-event variables in units of **days**. A 10-year follow-up horizon is precisely:
$$\text{Ten-Year Threshold} = 10 \times 365.25 = 3,652.5 \text{ days}$$

### 2. Hemodynamic Formulas
Blood pressure dynamics involve both steady (resistive) and pulsatile (arterial stiffness) components:

- **Pulse Pressure ($\text{PP}$):** Reflects large-artery stiffness and compliance:
  $$\text{PP} = \text{SYSBP} - \text{DIABP}$$
  *Baseline Stats:* Mean: $49.6\text{ mmHg}$; Median: $48.0\text{ mmHg}$; Range: $15.0 - 171.0\text{ mmHg}$.

- **Mean Arterial Pressure ($\text{MAP}$):** Reflects steady-state organ perfusion pressure:
  $$\text{MAP} = \text{DIABP} + \frac{1}{3}(\text{SYSBP} - \text{DIABP}) = \frac{1}{3}\text{SYSBP} + \frac{2}{3}\text{DIABP}$$
  *Baseline Stats:* Mean: $98.8\text{ mmHg}$; Median: $97.7\text{ mmHg}$; Range: $54.8 - 188.7\text{ mmHg}$.

### 3. Exact Mathematical Collinearity Proof
Notice that $\text{PP}$ and $\text{MAP}$ are linear combinations of $\text{SYSBP}$ and $\text{DIABP}$:
$$\begin{bmatrix} \text{PP} \\ \text{MAP} \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ \frac{1}{3} & \frac{2}{3} \end{bmatrix} \begin{bmatrix} \text{SYSBP} \\ \text{DIABP} \end{bmatrix}$$
Because the transformation matrix has rank 2, including all four variables ($\text{SYSBP}, \text{DIABP}, \text{PP}, \text{MAP}$) simultaneously results in an **exact rank-deficient design matrix**:
$$\text{Rank}(\mathbf{X}_{\text{all 4}}) = 2 < 4$$
In linear models (Logistic Regression, ElasticNet), this produces severe variance inflation ($VIF \to \infty$) and unstable regression coefficients. CARDIO-X resolves this by partitioning into disjoint feature sets:
1. `standard_bp`: [`SYSBP`, `DIABP`] (Standard clinical baseline).
2. `hemodynamic`: [`PULSE_PRESSURE`, `MAP`] (Arterial stiffness & perfusion representation).
3. `combined_tree`: [`SYSBP`, `DIABP`, `PULSE_PRESSURE`, `MAP`] (For decision trees/random forests only, which are immune to linear dependence).

### 4. Robust Scaling Formulation
Because clinical biomarkers (`GLUCOSE`, `TOTCHOL`, `SYSBP`) display significant skewness and extreme outliers, standard z-score scaling ($\frac{x - \mu}{\sigma}$) compresses healthy variance. CARDIO-X utilizes `RobustScaler`, centering by the median and scaling by the Interquartile Range (IQR):
$$x_{\text{scaled}} = \frac{x - \text{Median}(x)}{Q_3(x) - Q_1(x)}$$

---

## 8. Target Variable Formulation & Primary Prevention Cohort Selection

### Cohort Construction & Attrition Waterfall

```
┌────────────────────────────────────────────────────────┐
│ Total Baseline Participants at PERIOD == 1             │  N = 4,434
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ Exclude Prior Cardiovascular Disease (-219)
┌────────────────────────────────────────────────────────┐
│ Primary-Prevention Cohort (Disease-Free at Baseline)   │  N = 4,215
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼ Exclude Indeterminate Outcomes (-232)
┌────────────────────────────────────────────────────────┐
│ Final Evaluable Primary-Prevention Cohort              │  N = 3,983
└────────────────────────────────────────────────────────┘
```

#### Step 1: Baseline Population
- All records where `PERIOD == 1`: **$N = 4,434$**.

#### Step 2: Prior Disease Exclusion ($N = 219$)
In primary prevention, patients with pre-existing cardiovascular conditions at baseline cannot experience a *first incident* event. We take the union of:
$$\text{Prior CVD} = \text{PREVCHD} \cup \text{PREVAP} \cup \text{PREVMI} \cup \text{PREVSTRK}$$
Breakdown of the 219 excluded participants:
- Angina Pectoris + CHD (`PREVAP == 1` & `PREVCHD == 1`): **101**
- Angina + MI + CHD (`PREVAP == 1` & `PREVMI == 1` & `PREVCHD == 1`): **43**
- Myocardial Infarction + CHD (`PREVMI == 1` & `PREVCHD == 1`): **39**
- Stroke alone (`PREVSTRK == 1`): **25**
- MI + Stroke + CHD: **4**
- CHD alone (`PREVCHD == 1`): **4**
- Angina + Stroke + CHD: **3**
- **Total Excluded:** $101 + 43 + 39 + 25 + 4 + 4 + 3 = \mathbf{219}$.
- **Primary-Prevention Cohort Remaining:** $4,434 - 219 = \mathbf{4,215}$.

#### Step 3: Indeterminate Outcome Exclusions ($N = 232$)
In a 10-year binary classification task, a participant must have confirmed 10-year status:
1. **Confirmed Positive ($Y = 1$):** Experienced a CVD event within 10 years ($\text{CVD} = 1$ and $\text{TIMECVD} \le 3,652.5\text{ days}$).
2. **Confirmed Negative ($Y = 0$):** Remained alive and disease-free for at least 10 years ($\text{CVD} = 0$ and $\text{TIMECVD} \ge 3,652.5\text{ days}$), **OR** experienced their first CVD event *after* 10 years ($\text{CVD} = 1$ and $\text{TIMECVD} > 3,652.5\text{ days}$).
3. **Indeterminate (Excluded, $N = 232$):** Participants who had $\text{CVD} = 0$, but whose observation ended before 10 years ($\text{TIMECVD} < 3,652.5\text{ days}$):
   - **Non-CVD Deaths ($N = 216$):** Died from non-cardiovascular causes (cancer, trauma) prior to 10 years.
   - **Early Censoring ($N = 16$):** Lost to follow-up or administrative drop-out prior to 10 years.

#### Final Evaluable Cohort Distribution
- **Total Cohort Size:** **3,983 participants**.
- **Positive Cases ($Y = 1$):** **369 ($9.26\%$)**
- **Negative Cases ($Y = 0$):** **3,614 ($90.74\%$)**
- **Class Imbalance:** Approximately **1 : 9.8**.

---

## 9. Preprocessing & Pipeline Isolation Architecture (`src/preprocessing.py`)

### 1. Leakage-Free Stratified Splitting
To evaluate models without data snooping or optimistic bias, the 3,983 evaluable cohort is split into:
- **Training Set ($80\%$):** **3,186 participants** ($295$ positive, $2,891$ negative).
- **Held-Out Test Set ($20\%$):** **797 participants** ($74$ positive, $723$ negative).
- **Stratification:** Stratified by target $Y$ (`random_state=42`) ensuring both sets have an identical event rate of $9.26\%$.

### 2. Feature Set Definitions
Baseline predictors strictly exclude participant IDs (`RANDID`), follow-up durations (`TIMECVD`, `TIMEDTH`), future event flags (`DEATH`, `ANYCHD`), and post-baseline observations:

```python
# 1. Standard Blood Pressure Feature Set (Default for Linear Models)
FEATURE_SET_STANDARD_BP = [
    "AGE", "SEX", "EDUCATION", "CURSMOKE", "CIGPDAY", "BMI",
    "TOTCHOL", "GLUCOSE", "DIABETES", "BPMEDS", "HEARTRTE", "PREVHYP",
    "SYSBP", "DIABP"
]

# 2. Hemodynamic Feature Set (Alternative for Linear Models)
FEATURE_SET_HEMODYNAMIC = [
    "AGE", "SEX", "EDUCATION", "CURSMOKE", "CIGPDAY", "BMI",
    "TOTCHOL", "GLUCOSE", "DIABETES", "BPMEDS", "HEARTRTE", "PREVHYP",
    "PULSE_PRESSURE", "MAP"
]

# 3. Combined Tree Feature Set (Exclusively for Tree Ensembles)
FEATURE_SET_COMBINED_TREE = [
    "AGE", "SEX", "EDUCATION", "CURSMOKE", "CIGPDAY", "BMI",
    "TOTCHOL", "GLUCOSE", "DIABETES", "BPMEDS", "HEARTRTE", "PREVHYP",
    "SYSBP", "DIABP", "PULSE_PRESSURE", "MAP"
]
```

### 3. Pipeline Implementation
The transformation pipeline is built using Scikit-Learn's `ColumnTransformer`:
- **Continuous Features:** Imputed using `SimpleImputer(strategy='median')` followed by `RobustScaler()`.
- **Categorical/Binary Features:** Imputed using `SimpleImputer(strategy='most_frequent')` with passthrough scaling.
- **Strict Execution Rule:** The pipeline is fitted **only** on $X_{\text{train}}$ via `.fit_transform()`. The test partition $X_{\text{test}}$ is transformed strictly via `.transform()`.

---

## 10. Comprehensive File-by-File Technical Directory

```
CARDIO-X/
├── .gitignore                                 # Git configuration safeguarding raw datasets and environments
├── README.md                                  # High-level project repository documentation
├── information.md                             # Comprehensive technical manual & architecture specification
├── requirements.txt                           # Frozen pinned project dependencies
├── verify_env.py                              # Environment and structure verification script
├── data/
│   ├── README.md                              # Dataset documentation and provenance details
│   ├── .gitkeep                               # Directory anchor
│   └── FraminghamDataset.csv                  # Raw clinical dataset (Local only, SHA-256 protected)
├── models/
│   └── .gitkeep                               # Destination for trained model artifacts (*.joblib)
├── notebooks/
│   ├── .gitkeep                               # Directory anchor
│   └── 01_exploratory_data_analysis.ipynb     # Interactive EDA notebook
├── reports/
│   ├── .gitkeep                               # Directory anchor
│   ├── eda_summary.md                         # Detailed markdown report of exploratory data analysis
│   ├── preprocessing_design.md                # Specifications of cohort attrition and pipeline rules
│   ├── data_quality/
│   │   ├── data_quality_summary.md            # Markdown summary of dataset data quality audit
│   │   ├── dataset_overview.csv               # Dimensions, column types, and memory metrics
│   │   ├── descriptive_statistics.csv         # Parametric and non-parametric stats for all columns
│   │   └── missing_values.csv                 # Null counts and rates across all columns and periods
│   └── figures/
│       ├── fig01_distributions_key_biomarkers.png
│       ├── fig02_missingness_by_period.png
│       ├── fig03_correlation_matrix_baseline.png
│       ├── fig04_correlation_lipid_panel_period3.png
│       ├── fig05_longitudinal_trajectories.png
│       ├── fig06_engineered_clinical_features.png
│       └── fig07_outcome_prevalence_comparison.png
├── src/
│   ├── __init__.py                            # Package initialization exposing __version__ = "0.1.0"
│   ├── data_quality.py                        # Data quality audit and diagnostic reporting script
│   ├── eda.py                                 # Statistical analysis and figure generation script
│   └── preprocessing.py                       # Cohort construction, target definition, and pipeline script
└── tests/
    ├── __init__.py                            # Test package marker
    ├── test_environment.py                    # 4 tests: Python version, directories, files, package imports
    ├── test_data_quality.py                   # 6 tests: Dataset integrity, missingness, longitudinal checks
    ├── test_eda.py                            # 6 tests: Skewness, paired cohort, figure generation, immutability
    └── test_preprocessing.py                  # 9 tests: Cohort counts, target boundary, zero-leakage, immutability
```

---

## 11. Verification & Automated Test Suite (`tests/`)

The repository includes a comprehensive 25-test automated testing suite executing via `pytest`:

```bash
$ pytest tests/
============================= test session starts ==============================
collected 25 items

tests/test_data_quality.py ......                                        [ 24%]
tests/test_eda.py ......                                                 [ 48%]
tests/test_environment.py ....                                           [ 64%]
tests/test_preprocessing.py .........                                    [100%]

============================== 25 passed in 5.46s ===============================
```

### Breakdown of Test Suites
1. **`test_environment.py` (4 tests):**
   - Asserts Python >= 3.10.
   - Verifies all 6 root directories exist (`data`, `notebooks`, `src`, `models`, `reports`, `tests`).
   - Verifies essential config files exist (`requirements.txt`, `.gitignore`, `README.md`, `data/README.md`, `src/__init__.py`).
   - Verifies `src` package imports cleanly with version metadata.
2. **`test_data_quality.py` (6 tests):**
   - Asserts raw dataset exists and matches byte size and SHA-256 hash.
   - Validates dimension overview (11,627 rows, 39 columns, 4,434 participants).
   - Validates missingness rates and cholesterol availability patterns.
   - Validates source data immutability.
3. **`test_eda.py` (6 tests):**
   - Validates distribution metrics (IQR, Skewness).
   - Validates mathematical calculations for `CHOL_HDL_RATIO`, `PULSE_PRESSURE`, and `MAP`.
   - Validates paired longitudinal cohort size ($N=3,206$).
   - Validates all 7 figure files are generated and non-empty.
   - Validates source data immutability before and after EDA.
4. **`test_preprocessing.py` (9 tests):**
   - Asserts exactly 4,434 baseline participants (`PERIOD == 1`).
   - Asserts exactly 219 prior-disease exclusions (leaving 4,215).
   - Asserts exactly 232 indeterminate exclusions (leaving 3,983).
   - Asserts target distribution: 369 positive ($9.26\%$) and 3,614 negative ($90.74\%$).
   - Asserts events $>3,652.5\text{ days}$ are labeled $Y = 0$.
   - Asserts zero feature leakage in predictors.
   - Asserts rank-deficiency separation (`standard_bp` vs `hemodynamic`).
   - Asserts 80/20 train/test split isolation (transformers fit only on train).
   - Asserts source dataset immutability.

---

## 12. Frequently Asked Questions & Prompt-Learning Guide

*This section provides quick reference answers designed for study, oral defense, or prompting in Large Language Models (LLMs).*

### Q1: Why did we exclude participants with prior cardiovascular disease at baseline?
**Answer:** The clinical objective is **primary prevention** (identifying risk in asymptomatic individuals before initial disease manifestation). Including individuals who already had a myocardial infarction or stroke would conflate secondary prevention (recurrent event risk management) with primary etiology, distorting feature weights.

### Q2: Why are events occurring after Day 3,652.5 classified as negative ($Y = 0$)?
**Answer:** In a 10-year landmark prediction task, the question is: *"Will this patient experience an event within the next 10 years?"* If an individual remains event-free for 10 full years and only experiences an event at Year 14 (e.g., Day 5,110), they successfully survived the 10-year window event-free. Marking them positive would leak future survival information and violate the 10-year horizon.

### Q3: Why can't we use HDL, LDL, or Triglycerides in the baseline models?
**Answer:** 
1. **HDL/LDL:** High-Density and Low-Density Lipoproteins were not measured at Exams 1 and 2 in the Framingham Study protocol; they were introduced only at Exam 3. Consequently, they are 100% missing at baseline. Imputing 100% missing values would hallucinate data.
2. **Triglycerides:** The variable does not exist anywhere in the Framingham dataset.
Baseline models must therefore use Total Cholesterol (`TOTCHOL`, $98.8\%$ complete) as the primary baseline lipid biomarker.

### Q4: Why can't we include SYSBP, DIABP, Pulse Pressure, and MAP together in Logistic Regression?
**Answer:** Pulse Pressure ($\text{PP} = \text{SYSBP} - \text{DIABP}$) and Mean Arterial Pressure ($\text{MAP} = \frac{1}{3}\text{SYSBP} + \frac{2}{3}\text{DIABP}$) are exact linear combinations of systolic and diastolic blood pressure. In linear models, this causes exact multicollinearity (matrix rank = 2 rather than 4), blowing up parameter variance ($VIF \to \infty$) and causing numerical instability. They must be evaluated in separate feature sets.

### Q5: What is the risk of fitting an imputer or scaler on the entire dataset before splitting?
**Answer:** Fitting transformations on the entire dataset constitutes **data leakage** (specifically distribution snooping). The imputer's median and the scaler's interquartile range would incorporate information from the test set, creating overly optimistic performance estimates that fail to generalize in actual clinical practice.
