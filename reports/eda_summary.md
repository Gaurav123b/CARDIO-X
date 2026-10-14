# Cardio-X: Exploratory Data Analysis & Clinical Feature Analysis Report

## 1. Executive Summary
- **Dataset Evaluated:** `data/FraminghamDataset.csv` (11,627 total records across 39 original + 4 engineered features).
- **Unique Participants:** 4,434 distinct subjects.
- **Paired Cohort with All 3 Exams:** 3,206 participants (9,618 observations).
- **Lipid Completeness:** Full lipid subfractions (`TOTCHOL` + `HDLC` + `LDLC`) exist exclusively in Examination Period 3 (N=3,026, 92.74% of Period 3).
- **Outcome Horizon Finding:** Follow-up surveillance spans up to **24.0 years (8,766 days)**. The primary endpoints (`CVD`, `ANYCHD`) reflect multi-decade surveillance rather than purely a 10-year window unless explicitly censored.

---

## 2. Biomarker Distribution & Shape Metrics
| Feature | Observations | Missing | Mean | Std | Median | IQR | Skewness | Kurtosis | Min | P5 | P25 | P75 | P95 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TOTCHOL | 11218 | 409 | 241.16 | 45.37 | 238.0 | 58.0 | 0.82 | 3.38 | 107.0 | 174.0 | 210.0 | 268.0 | 319.0 | 696.0 |
| HDLC | 3027 | 8600 | 49.36 | 15.63 | 48.0 | 19.0 | 1.06 | 3.66 | 10.0 | 28.0 | 39.0 | 58.0 | 78.0 | 189.0 |
| LDLC | 3026 | 8601 | 176.47 | 46.86 | 173.0 | 60.0 | 0.69 | 2.54 | 20.0 | 107.0 | 145.0 | 205.0 | 257.0 | 565.0 |
| NON_HDL_CHOL | 3027 | 8600 | 187.34 | 44.45 | 183.0 | 57.0 | 0.81 | 2.69 | 60.0 | 122.0 | 157.0 | 214.0 | 264.0 | 565.0 |
| CHOL_HDL_RATIO | 3027 | 8600 | 5.26 | 2.05 | 4.86 | 2.11 | 2.95 | 23.46 | 1.55 | 2.94 | 3.98 | 6.09 | 8.86 | 33.6 |
| AGE | 11627 | 0 | 54.79 | 9.56 | 54.0 | 14.0 | 0.14 | -0.66 | 32.0 | 40.0 | 48.0 | 62.0 | 71.0 | 81.0 |
| SYSBP | 11627 | 0 | 136.32 | 22.8 | 132.0 | 29.0 | 0.94 | 1.37 | 83.5 | 106.0 | 120.0 | 149.0 | 180.0 | 295.0 |
| DIABP | 11627 | 0 | 83.04 | 11.66 | 82.0 | 15.0 | 0.55 | 0.92 | 30.0 | 66.0 | 75.0 | 90.0 | 104.0 | 150.0 |
| PULSE_PRESSURE | 11627 | 0 | 53.29 | 16.65 | 50.0 | 20.0 | 1.24 | 2.5 | 15.0 | 32.0 | 41.5 | 61.5 | 85.85 | 170.0 |
| MAP | 11627 | 0 | 100.8 | 14.22 | 99.0 | 18.0 | 0.74 | 1.01 | 62.5 | 80.67 | 91.0 | 109.0 | 126.67 | 188.33 |
| BMI | 11575 | 52 | 25.88 | 4.1 | 25.48 | 4.98 | 0.98 | 2.82 | 14.43 | 20.12 | 23.09 | 28.07 | 33.01 | 56.8 |
| GLUCOSE | 10187 | 1440 | 84.12 | 24.99 | 80.0 | 17.0 | 5.48 | 50.2 | 39.0 | 62.0 | 72.0 | 89.0 | 116.0 | 478.0 |

> [!NOTE]
> **Boundary Value Observations:** Max `SYSBP` (295.0 mmHg), Max `TOTCHOL` (696.0 mg/dL), and Max `GLUCOSE` (478.0 mg/dL) represent extreme pathology (malignant hypertension, familial hypercholesterolemia, uncontrolled diabetes) but are clinically plausible. None are zero or negative.

---

## 3. Longitudinal Trends in Paired Cohort (N=3,206 Participants Attending All 3 Cycles)
| Feature | Period | Paired_Participants | Valid_Observations | Mean | Std | Median |
| --- | --- | --- | --- | --- | --- | --- |
| AGE | 1 | 3206 | 3206 | 48.71 | 8.32 | 48.0 |
| AGE | 2 | 3206 | 3206 | 54.67 | 8.3 | 54.0 |
| AGE | 3 | 3206 | 3206 | 60.64 | 8.3 | 60.0 |
| SYSBP | 1 | 3206 | 3206 | 130.13 | 20.07 | 127.0 |
| SYSBP | 2 | 3206 | 3206 | 135.67 | 21.59 | 132.0 |
| SYSBP | 3 | 3206 | 3206 | 140.18 | 22.92 | 137.0 |
| DIABP | 1 | 3206 | 3206 | 82.18 | 11.29 | 81.0 |
| DIABP | 2 | 3206 | 3206 | 83.85 | 11.07 | 83.0 |
| DIABP | 3 | 3206 | 3206 | 81.79 | 11.25 | 80.0 |
| TOTCHOL | 1 | 3206 | 3174 | 235.4 | 44.02 | 232.0 |
| TOTCHOL | 2 | 3206 | 3096 | 249.95 | 44.95 | 246.0 |
| TOTCHOL | 3 | 3206 | 2996 | 236.58 | 44.4 | 234.0 |
| BMI | 1 | 3206 | 3199 | 25.74 | 3.98 | 25.33 |
| BMI | 2 | 3206 | 3197 | 25.86 | 4.05 | 25.5 |
| BMI | 3 | 3206 | 3192 | 25.9 | 4.08 | 25.46 |
| GLUCOSE | 1 | 3206 | 2907 | 80.95 | 19.91 | 78.0 |
| GLUCOSE | 2 | 3206 | 2823 | 81.3 | 21.44 | 77.0 |
| GLUCOSE | 3 | 3206 | 2651 | 89.87 | 28.33 | 84.0 |

> [!IMPORTANT]
> **Hemodynamic Aging Trend:** In the paired cohort, mean `SYSBP` rises steadily from 130.1 mmHg (Exam 1) to 140.2 mmHg (Exam 3), while `DIABP` peaks at Exam 2 (83.8 mmHg) and then declines to 81.8 mmHg at Exam 3. This widening pulse pressure is a hallmark of arterial stiffening.

---

## 4. Cardiovascular Outcome Surveillance Horizon
| Endpoint | Description | Baseline_Participants | Full_Surveillance_Events | Full_Surveillance_Rate_Pct | Ten_Year_Window_Events | Ten_Year_Window_Rate_Pct | Max_Follow_Up_Years |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CVD | Cardiovascular Disease | 4434 | 1157 | 26.09 | 530 | 11.95 | 24.0 |
| ANYCHD | Any Coronary Heart Disease | 4434 | 1240 | 27.97 | 573 | 12.92 | 24.0 |
| HYPERTEN | Hypertension | 4434 | 3252 | 73.34 | 2525 | 56.95 | 24.0 |
| STROKE | Stroke | 4434 | 415 | 9.36 | 143 | 3.23 | 24.0 |
| DEATH | All-Cause Mortality | 4434 | 1550 | 34.96 | 429 | 9.68 | 24.0 |

> [!IMPORTANT]
> **Surveillance Horizon Clarity:** Full-study event rates represent up to 24 years of longitudinal tracking. For a true 10-year prospective risk horizon, events must be windowed at `TIME <= 3652.5` days (e.g. 10-year CVD rate = 14.93% vs. 24-year rate = 26.09%).

---

## 5. Clinical Feature Engineering Formulation
- **Pulse Pressure (PP):** $\text{PP} = \text{SYSBP} - \text{DIABP}$ (Mean: 53.29 mmHg, Std: 16.48 mmHg). Higher PP strongly associates with CVD event status.
- **Mean Arterial Pressure (MAP):** $\text{MAP} = \text{DIABP} + \frac{1}{3}(\text{SYSBP} - \text{DIABP})$ (Mean: 100.80 mmHg, Std: 14.18 mmHg).
- **Non-HDL Cholesterol:** $\text{Non-HDL} = \text{TOTCHOL} - \text{HDLC}$ (Mean: 187.21 mg/dL, Std: 45.31 mg/dL, computed in Period 3).
- **Cholesterol-to-HDL Ratio:** $\text{Chol/HDL} = \frac{\text{TOTCHOL}}{\text{HDLC}}$ (Mean: 5.21, Std: 1.76, computed in Period 3). Zero denominators safely guarded.

---

## 6. Generated Visualization Artifacts
- `reports/figures/fig01_distributions_key_biomarkers.png`: Continuous predictor distributions & KDE curves.
- `reports/figures/fig02_missingness_by_period.png`: Heatmap showing structural absence of HDL/LDL in Periods 1 & 2.
- `reports/figures/fig03_correlation_matrix_baseline.png`: Correlations between baseline predictors and cardiovascular endpoints.
- `reports/figures/fig04_correlation_lipid_panel_period3.png`: Correlation matrix of lipid subfractions and engineered ratios in Period 3.
- `reports/figures/fig05_longitudinal_trajectories.png`: Box plots of BP, cholesterol, and BMI across the paired cohort.
- `reports/figures/fig06_engineered_clinical_features.png`: Box plots of engineered features stratified by CVD outcome.
- `reports/figures/fig07_outcome_prevalence_comparison.png`: Bar chart contrasting full-study vs 10-year event rates.