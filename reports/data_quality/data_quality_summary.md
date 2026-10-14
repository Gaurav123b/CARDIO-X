# Cardio-X: Data Quality and Dataset Validation Report

## 1. Executive Summary
- **Total Rows:** 11,627
- **Total Columns:** 39
- **Distinct Participants:** 4,434
- **Exact Duplicate Rows:** 0
- **Data Integrity:** No row duplicates found; repeated IDs correspond to prospective longitudinal follow-up examination cycles.

---

## 2. Dataset Overview
| Metric | Value |
| --- | --- |
| Total Rows | 11627.0 |
| Total Columns | 39.0 |
| Unique Participants (RANDID) | 4434.0 |
| Exact Duplicate Rows | 0.0 |
| Duplicate (RANDID, PERIOD) Pairs | 0.0 |
| Total Cells | 453453.0 |
| Total Missing Cells | 20075.0 |
| Overall Missingness (%) | 4.43 |
| Memory Footprint (MB) | 3.46 |

---

## 3. Longitudinal Cohort Structure Analysis
The dataset consists of multi-wave prospective examinations per participant (`PERIOD` 1, 2, and 3):

- **Exam Cycle 1 (PERIOD=1):** 4,434 observations (Median follow-up: 0.0 days / ~0.0 years)
- **Exam Cycle 2 (PERIOD=2):** 3,930 observations (Median follow-up: 2174.0 days / ~6.0 years)
- **Exam Cycle 3 (PERIOD=3):** 3,263 observations (Median follow-up: 4361.0 days / ~11.9 years)

### Examination Frequency per Participant:
- Participants with 1 recorded examination(s): **447**
- Participants with 2 recorded examination(s): **781**
- Participants with 3 recorded examination(s): **3,206**

> [!IMPORTANT]
> **Epidemiological Interpretation:** Repeated `RANDID` entries represent longitudinal follow-up observations of the same subjects, NOT duplicate data. Models must account for this structure (e.g., using baseline Exam 1 observations or explicit panel/time-series modeling) to avoid data leakage.

---

## 4. Missing Data Profile
Ranked summary of columns exhibiting missingness:

| Column | Data_Type | Non_Missing_Count | Missing_Count | Missing_Percentage | Missingness_Rank |
| --- | --- | --- | --- | --- | --- |
| LDLC | float64 | 3026 | 8601 | 73.97 | 1 |
| HDLC | float64 | 3027 | 8600 | 73.97 | 2 |
| GLUCOSE | float64 | 10187 | 1440 | 12.38 | 3 |
| BPMEDS | float64 | 11034 | 593 | 5.1 | 4 |
| TOTCHOL | float64 | 11218 | 409 | 3.52 | 5 |
| educ | float64 | 11332 | 295 | 2.54 | 6 |
| CIGPDAY | float64 | 11548 | 79 | 0.68 | 7 |
| BMI | float64 | 11575 | 52 | 0.45 | 8 |
| HEARTRTE | float64 | 11621 | 6 | 0.05 | 9 |

---

## 5. Cholesterol & Lipid Profile Evaluation
- **Total Cholesterol (`TOTCHOL`):** Available in 11,218 rows (96.48%).
- **High-Density Lipoprotein (`HDLC`):** Available in 3,027 rows (26.03%).
- **Low-Density Lipoprotein (`LDLC`):** Available in 3,026 rows (26.03%).
- **Triglycerides:** **Not present** in the dataset.
- **Complete Lipid Profiles (`TOTCHOL` + `HDLC` + `LDLC`):** **3,026 rows (26.03%)**.

### Lipid Completeness by Examination Period:
- **Period 1:** 0 complete lipid profiles out of 4,434 rows (0.0%) [TOTCHOL=4,382, HDLC=0, LDLC=0]
- **Period 2:** 0 complete lipid profiles out of 3,930 rows (0.0%) [TOTCHOL=3,787, HDLC=0, LDLC=0]
- **Period 3:** 3,026 complete lipid profiles out of 3,263 rows (92.74%) [TOTCHOL=3,049, HDLC=3,027, LDLC=3,026]

> [!NOTE]
> **Protocol Disclosure:** `HDLC` and `LDLC` were not included in the Framingham study testing protocol during Examination Periods 1 and 2. Their missingness in early cycles is structural by study design rather than random.

---

## 6. Descriptive Distribution Statistics
Key numeric physiological parameters (quantiles, range, dispersion):

| Column | Count | Missing | Zeros | Mean | Std | Min | 25% | Median (50%) | 75% | Max | IQR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RANDID | 11627 | 0 | 0 | 5004740.92 | 2900877.44 | 2448.0 | 2474378.0 | 5006008.0 | 7472730.0 | 9999312.0 | 4998352.0 |
| SEX | 11627 | 0 | 0 | 1.57 | 0.5 | 1.0 | 1.0 | 2.0 | 2.0 | 2.0 | 1.0 |
| TOTCHOL | 11218 | 409 | 0 | 241.16 | 45.37 | 107.0 | 210.0 | 238.0 | 268.0 | 696.0 | 58.0 |
| AGE | 11627 | 0 | 0 | 54.79 | 9.56 | 32.0 | 48.0 | 54.0 | 62.0 | 81.0 | 14.0 |
| SYSBP | 11627 | 0 | 0 | 136.32 | 22.8 | 83.5 | 120.0 | 132.0 | 149.0 | 295.0 | 29.0 |
| DIABP | 11627 | 0 | 0 | 83.04 | 11.66 | 30.0 | 75.0 | 82.0 | 90.0 | 150.0 | 15.0 |
| CURSMOKE | 11627 | 0 | 6598 | 0.43 | 0.5 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| CIGPDAY | 11548 | 79 | 6598 | 8.25 | 12.19 | 0.0 | 0.0 | 0.0 | 20.0 | 90.0 | 20.0 |
| BMI | 11575 | 52 | 0 | 25.88 | 4.1 | 14.43 | 23.09 | 25.48 | 28.07 | 56.8 | 4.98 |
| DIABETES | 11627 | 0 | 11097 | 0.05 | 0.21 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| BPMEDS | 11034 | 593 | 10090 | 0.09 | 0.28 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| HEARTRTE | 11621 | 6 | 0 | 76.78 | 12.46 | 37.0 | 69.0 | 75.0 | 85.0 | 220.0 | 16.0 |
| GLUCOSE | 10187 | 1440 | 0 | 84.12 | 24.99 | 39.0 | 72.0 | 80.0 | 89.0 | 478.0 | 17.0 |
| educ | 11332 | 295 | 0 | 1.99 | 1.03 | 1.0 | 1.0 | 2.0 | 3.0 | 4.0 | 2.0 |
| PREVCHD | 11627 | 0 | 10785 | 0.07 | 0.26 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| PREVAP | 11627 | 0 | 11000 | 0.05 | 0.23 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| PREVMI | 11627 | 0 | 11253 | 0.03 | 0.18 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| PREVSTRK | 11627 | 0 | 11475 | 0.01 | 0.11 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| PREVHYP | 11627 | 0 | 6283 | 0.46 | 0.5 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| TIME | 11627 | 0 | 4434 | 1957.02 | 1758.78 | 0.0 | 0.0 | 2156.0 | 4252.5 | 4854.0 | 4252.5 |
| PERIOD | 11627 | 0 | 0 | 1.9 | 0.81 | 1.0 | 1.0 | 2.0 | 3.0 | 3.0 | 2.0 |
| HDLC | 3027 | 8600 | 0 | 49.36 | 15.63 | 10.0 | 39.0 | 48.0 | 58.0 | 189.0 | 19.0 |
| LDLC | 3026 | 8601 | 0 | 176.47 | 46.86 | 20.0 | 145.0 | 173.0 | 205.0 | 565.0 | 60.0 |
| DEATH | 11627 | 0 | 8100 | 0.3 | 0.46 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| ANGINA | 11627 | 0 | 9725 | 0.16 | 0.37 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| HOSPMI | 11627 | 0 | 10473 | 0.1 | 0.3 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| MI_FCHD | 11627 | 0 | 9839 | 0.15 | 0.36 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| ANYCHD | 11627 | 0 | 8469 | 0.27 | 0.44 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| STROKE | 11627 | 0 | 10566 | 0.09 | 0.29 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| CVD | 11627 | 0 | 8728 | 0.25 | 0.43 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| HYPERTEN | 11627 | 0 | 2985 | 0.74 | 0.44 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| TIMEAP | 11627 | 0 | 307 | 7241.56 | 2477.78 | 0.0 | 6224.0 | 8766.0 | 8766.0 | 8766.0 | 2542.0 |
| TIMEMI | 11627 | 0 | 141 | 7593.85 | 2136.73 | 0.0 | 7212.0 | 8766.0 | 8766.0 | 8766.0 | 1554.0 |
| TIMEMIFC | 11627 | 0 | 161 | 7543.04 | 2192.12 | 0.0 | 7049.5 | 8766.0 | 8766.0 | 8766.0 | 1716.5 |
| TIMECHD | 11627 | 0 | 407 | 7008.15 | 2641.34 | 0.0 | 5598.5 | 8766.0 | 8766.0 | 8766.0 | 3167.5 |
| TIMESTRK | 11627 | 0 | 60 | 7660.88 | 2011.08 | 0.0 | 7295.0 | 8766.0 | 8766.0 | 8766.0 | 1471.0 |
| TIMECVD | 11627 | 0 | 331 | 7166.08 | 2541.67 | 0.0 | 6004.0 | 8766.0 | 8766.0 | 8766.0 | 2762.0 |
| TIMEDTH | 11627 | 0 | 0 | 7854.1 | 1788.37 | 26.0 | 7797.5 | 8766.0 | 8766.0 | 8766.0 | 968.5 |
| TIMEHYP | 11627 | 0 | 3518 | 3598.96 | 3464.16 | 0.0 | 0.0 | 2429.0 | 7329.0 | 8766.0 | 7329.0 |

---

## 7. Prospective Outcome & Candidate Target Discovery
Audit of potential binary outcome endpoints and associated time-to-event indicators:

| Candidate_Outcome | Observed_Values | Missing_Count | Missing_Pct | Positive_Cases | Positive_Rate_Pct | Associated_Time_Column | Preliminary_Description |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CVD | [0, 1] | 0 | 0.0 | 2899 | 24.93 | TIMECVD | Cardiovascular Disease event during follow-up |
| ANYCHD | [0, 1] | 0 | 0.0 | 3158 | 27.16 | TIMECHD | Any Coronary Heart Disease event during follow-up |
| HYPERTEN | [0, 1] | 0 | 0.0 | 8642 | 74.33 | TIMEHYP | Hypertension status during follow-up |
| STROKE | [0, 1] | 0 | 0.0 | 1061 | 9.13 | TIMESTRK | Stroke event during follow-up |
| DEATH | [0, 1] | 0 | 0.0 | 3527 | 30.33 | TIMEDTH | All-cause mortality during follow-up |
| ANGINA | [0, 1] | 0 | 0.0 | 1902 | 16.36 | TIMEAP | Angina Pectoris event during follow-up |
| HOSPMI | [0, 1] | 0 | 0.0 | 1154 | 9.93 | TIMEMI | Hospitalized Myocardial Infarction |
| MI_FCHD | [0, 1] | 0 | 0.0 | 1788 | 15.38 | TIMEMIFC | Myocardial Infarction or Fatal Coronary Heart Disease |

> [!TIP]
> **Target Selection Notice:** Selection of the primary prediction target (e.g. `CVD`, `ANYCHD`, or hypercholesterolemia risk stratification) will be finalized in subsequent design phases with guide alignment.