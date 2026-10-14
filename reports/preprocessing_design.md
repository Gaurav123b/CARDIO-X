# Cardio-X: Preprocessing Pipeline & Target Definition Design Report

## 1. Executive Summary & Verified Cohort Counts
- **Dataset Path:** `data/FraminghamDataset.csv` (SHA-256: `e073ce4c60d351f2015c500b5bdf89c3a99ab3dfd0ba525e31a743742d649131`).
- **Cohort Strategy:** **Option A1 (Primary Prevention)**: Predict first incident cardiovascular event within 10 years among participants disease-free at baseline.
- **Target Definition:** $\text{TARGET\_10Y\_CVD} = 1$ if $(\text{CVD} == 1 \ \& \ \text{TIMECVD} \le 3652.5\text{ days})$, else $0$.

| Stage | Count | Percentage |
| --- | --- | --- |
| 1. Baseline Examination Records (PERIOD == 1) | 4434 | 100.0% |
| 2. Pre-existing CVD Exclusions (PREVCHD/AP/MI/STRK == 1) | -219 | 4.94% |
| 3. Primary-Prevention Cohort | 4215 | 95.06% |
| 4. Indeterminate Exclusions (Follow-up < 10 Yrs without CVD) | -232 | 5.50% |
|    - Non-CVD Deaths before 10 years (DEATH == 1) | 216 | 5.12% |
|    - Lost to follow-up / Censored < 10 years (DEATH == 0) | 16 | 0.38% |
| 5. Evaluable Primary-Prevention Cohort | 3983 | 100.0% |
|    - Confirmed 10-Year Incident CVD Cases (Y = 1) | 369 | 9.26% |
|    - Confirmed 10-Year Negative Controls (Y = 0) | 3614 | 90.74% |
|      * Followed >= 10 yrs without CVD (CVD == 0, TIMECVD >= 3652.5) | 2987 | 75.00% |
|      * Incident CVD event occurred AFTER 10 yrs (CVD == 1, TIMECVD > 3652.5) | 627 | 15.74% |

---

## 2. Participant-Level Train/Test Split (Stratified 80/20)
The evaluable cohort of 3,983 unique participants is partitioned using `StratifiedShuffleSplit` with fixed `random_state=42`:

| Partition | Total | Positives (Y=1) | Negatives (Y=0) | Event Rate (%) |
| --- | --- | --- | --- | --- |
| Training Set (80%) | 3186 | 295 | 2891 | 9.26% |
| Held-Out Test Set (20%) | 797 | 74 | 723 | 9.28% |
| Total Evaluable Cohort | 3983 | 369 | 3614 | 9.26% |

> [!IMPORTANT]
> **Methodological Rule:** The held-out test set ($N=797$) is strictly preserved for final model benchmarking in Phase 5. It must never be used for feature selection, scaling parameter fitting, or hyperparameter tuning.

---

## 3. Eligible Baseline Predictor Features (Zero Leakage)
All features are measured strictly at baseline examination wave 1 before prospective follow-up began:

| Feature | Type | Range/Levels | Missing | Role |
| --- | --- | --- | --- | --- |
| AGE | Continuous | 32 - 70 | 0 (0.0%) | Vascular aging risk factor |
| SEX | Binary | 0=Female, 1=Male | 0 (0.0%) | Biological sex stratifier |
| CURSMOKE | Binary | 0=No, 1=Yes | 0 (0.0%) | Active tobacco use |
| CIGPDAY | Continuous | 0 - 90 | 28 (0.70%) | Smoking dose intensity |
| educ | Ordinal | 1 < 2 < 3 < 4 | 101 (2.54%) | Educational attainment level |
| SYSBP | Continuous | 83.5 - 295.0 | 0 (0.0%) | Systolic arterial pressure (mmHg) |
| DIABP | Continuous | 30.0 - 150.0 | 0 (0.0%) | Diastolic vascular resistance (mmHg) |
| BPMEDS | Binary | 0=No, 1=Yes | 53 (1.33%) | Antihypertensive medication use |
| PREVHYP | Binary | 0=No, 1=Yes | 0 (0.0%) | Prior hypertension history |
| HEARTRTE | Continuous | 37 - 220 | 1 (0.03%) | Resting heart rate (bpm) |
| TOTCHOL | Continuous | 112 - 696 | 47 (1.18%) | Serum total cholesterol (mg/dL) |
| BMI | Continuous | 14.43 - 56.80 | 17 (0.43%) | Body mass index (kg/m²) |
| DIABETES | Binary | 0=No, 1=Yes | 0 (0.0%) | Diagnosed diabetes mellitus |
| GLUCOSE | Continuous | 39 - 478 | 345 (8.66%) | Fasting blood glucose (mg/dL) |

### Collinear Blood Pressure Feature Sets:
- **Feature Set A (`standard_bp`):** Uses raw `SYSBP` and `DIABP` (standard clinical baseline).
- **Feature Set B (`hemodynamic`):** Uses derived `PULSE_PRESSURE` ($\text{SYSBP} - \text{DIABP}$) and `MAP` ($\text{DIABP} + \frac{1}{3}\text{PP}$), omitting raw BP to prevent linear dependence.
- **Feature Set C (`combined_tree`):** Includes all 4 features, designated strictly for non-linear tree-based ensembles.

---

## 4. Competing-Risk Disclosure & Modeling Simplifications
- **Exclusion of Non-CVD Deaths ($N=216$):** Participants who died of non-cardiovascular causes before 10 years without experiencing a CVD event are excluded from this binary classification cohort.
- **Academic Limitation:** This models the landmark conditional probability $P(\text{CVD within 10y} \mid \text{Followed } \ge 10\text{y or CVD observed } \le 10\text{y})$. It is a recognized simplification of binary machine learning and does not constitute a full competing-risk subdistribution hazards model (e.g. Fine-Gray).

---

## 5. Model-Tailored Preprocessing Architecture
1. **Tree Ensembles (Random Forest, XGBoost):** Imputation only (Median for continuous, Mode for categorical). **No scaling applied** to preserve natural clinical units.
2. **Linear / Regularized Models (Logistic Regression, ElasticNet):** Median/Mode imputation followed by `RobustScaler` on continuous features to prevent extreme physiological spikes (e.g., SYSBP = 295) from dominating gradient descent.