# CARDIO-X Phase 2: Machine Learning Target & Feature Audit Report

> **Project:** CARDIO-X — Intelligent ML-Driven System for Cholesterol Risk Detection & 10-Year Cardiovascular Event Prediction  
> **Phase:** Phase 2 — Target & Feature Verification, Leakage Audit, and Evaluation Strategy Formulation  
> **Deliverable Created:** [`data_comparison/ml_target_and_feature_audit.md`](file:///Users/gauravkumar/Desktop/CARDIO-X/data_comparison/ml_target_and_feature_audit.md)  
> **Associated Feature Registry:** [`data_comparison/ml_feature_dictionary.csv`](file:///Users/gauravkumar/Desktop/CARDIO-X/data_comparison/ml_feature_dictionary.csv) (129 audited variables)  
> **Microdata Sources Inspected:**  
> - NHANES 2021–2023 Survey Microdata ($N = 6,001$, 88 variables): [`data_comparison/nhanes_diabetes_2021_2023.csv`](file:///Users/gauravkumar/Desktop/CARDIO-X/data_comparison/nhanes_diabetes_2021_2023.csv)  
> - Framingham Heart Study Microdata ($N = 11,627$ longitudinal records, 39 variables): [`data/FraminghamDataset.csv`](file:///Users/gauravkumar/Desktop/CARDIO-X/data/FraminghamDataset.csv)  
> - Preprocessing & Cohort Pipeline: [`src/preprocessing.py`](file:///Users/gauravkumar/Desktop/CARDIO-X/src/preprocessing.py)  
> - Regression Test Suite: 25 passing automated tests in `tests/`  

---

## 1. Executive Summary & Epistemological Boundaries

To ensure scientific rigor prior to model construction, this audit evaluates the two distinct machine learning tasks comprising the CARDIO-X system:
- **TIER 1 (Remote Screening Engine):** Predict phenotypic hypercholesterolemia (`TARGET_HYPERCHOL`) strictly from non-invasive lifestyle, demographic, and physical examination parameters in contemporary NHANES microdata.
- **TIER 2 (Clinical Risk Engine):** Predict 10-year first incident hard cardiovascular disease events (`TARGET_10Y_CVD`) from verified baseline clinical and laboratory biomarkers in the longitudinal Framingham Heart Study cohort.

### Epistemological Distinction: Verified Facts vs. Assumptions

| Domain | Verified Empirical Facts (Proven from Microdata) | Working Assumptions (To be Empirically Tested) | Unresolved Questions (Requiring User / Clinical Alignment) |
| :--- | :--- | :--- | :--- |
| **Tier 1 Target (`TARGET_HYPERCHOL`)** | Neither target column exists precomputed in the raw CSV files. Lab Total Cholesterol $\ge 240\text{ mg/dL}$ alone labels $1,571$ statin-treated patients ($94.6\%$ of all statin users) as "negative controls". | Composite Option B (`TotChol >= 240` OR `Statin == 1`, prevalence $37.88\%$) accurately represents treated + untreated phenotypic hypercholesterolemia without arbitrary scaling. | Should Tier 1 target Option B (prevalence $37.9\%$), ATP III Composite Option C ($48.3\%$), or a statin-naive subgroup model ($12.4\%$) be the primary benchmark? |
| **Tier 1 Features** | 58 variables are non-invasive; 10 are invasive laboratory panels; 6 directly leak target information (lipid panel, lab-based undiagnosed diabetes). | Lifestyle factors (diet, MET minutes, sleep, smoking) provide sufficient signal to triage patients for laboratory lipid testing. | Should self-reported doctor-diagnosed conditions (`diabetes_diagnosed`, `hypertension_diagnosed`) be included in remote consumer-facing intake forms? |
| **Tier 2 Target (`TARGET_10Y_CVD`)** | Derived via `build_primary_prevention_cohort`: $N = 3,983$ evaluable participants, $369$ events ($9.26\%$), $0$ missing labels. $219$ prevalent disease cases and $232$ indeterminate cases excluded. | Dichotomizing continuous follow-up at 10 years ($3,652.5\text{ days}$) preserves clinical decision alignment with ACC/AHA guidelines. | How should competing risks of non-CVD death ($N = 216$) be reported when communicating 10-year absolute risk to patients? |
| **Tier 2 Features** | All 14 baseline features were measured at Period 1 ($t=0$) strictly before follow-up ($0 - 8,766\text{ days}$). `HDLC` and `LDLC` are $100\%$ missing at baseline. | Median imputation on training folds prevents data leakage while preserving sample size ($N = 3,983$). | Will linear models evaluate `standard_bp` and `hemodynamic` feature sets in isolated pipelines to avoid multicollinearity? |

---

## 2. TASK 1: Verification of Target Variables

### A. TIER 1 Target: NHANES Hypercholesterolemia

#### Target Column Existence
- **Finding:** The column `TARGET_HYPERCHOL` **DOES NOT EXIST** in [`data_comparison/nhanes_diabetes_2021_2023.csv`](file:///Users/gauravkumar/Desktop/CARDIO-X/data_comparison/nhanes_diabetes_2021_2023.csv).
- **Source Columns Available:** `total_cholesterol_mg_dl`, `high_cholesterol_diagnosed`, `cholesterol_medication`, `non_hdl_cholesterol_mg_dl`, `ldl_cholesterol_mg_dl`, `hdl_cholesterol_mg_dl`.

#### Quantitative Evaluation of Candidate Definitions

```
Total NHANES Sample: N = 6,001
├── Missing Total Cholesterol Lab: N = 288 (4.80%)
└── Evaluated on Observed Records: N = 5,713
```

We evaluated five candidate formulations against clinical guidelines (AHA/ACC, NCEP ATP III):

| Formulation Option | Mathematical Definition | Positive Count ($N$) | Negative Count ($N$) | Missing Count ($N$) | Effective Prevalence | Clinical Validity & Critical Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Option A: Strict Lab Cutoff** | `total_cholesterol_mg_dl >= 240` | 625 | 5,088 | 288 | **10.94%** | **Severely Flawed (Confounding by Indication):** Ignores lipid medication. $1,571$ participants taking statins have medically lowered cholesterol ($< 240\text{ mg/dL}$) and are falsely categorized as "healthy controls". |
| **Option B: Clinical Composite (Recommended)** | `(total_cholesterol_mg_dl >= 240) \| (cholesterol_medication == 1)` | 2,196 | 3,578 | 204 | **37.88%** | **High Clinical Validity:** Captures active laboratory dyslipidemia ($N = 535$ untreated/uncontrolled) plus diagnosed hypercholesterolemia managed pharmacologically ($N = 1,660$). Requires no arbitrary numerical scaling. |
| **Option C: ATP III Dyslipidemia Composite** | `(TotChol >= 240) \| (Non-HDL >= 160) \| (Statin == 1)` | 2,788 | 2,987 | 226 | **48.28%** | **Comprehensive Lipid Risk:** Incorporates atherogenic non-HDL cholesterol. Effectively balanced binary classification task. |
| **Option D: Self-Reported Diagnosis** | `high_cholesterol_diagnosed == 1` | 2,413 | 3,554 | 34 | **40.44%** | **Subjective Recall Bias:** Misses 535 unmanaged/undiagnosed participants who have lab TotChol $\ge 240$ but reported never being diagnosed. |
| **Option E: Borderline/High Composite** | `(total_cholesterol_mg_dl >= 200) \| (cholesterol_medication == 1)` | 3,365 | 2,410 | 226 | **58.27%** | **Broad Screening:** Identifies anyone with borderline elevation ($200 - 239\text{ mg/dL}$) or on therapy. |
| **Subgroup Option: Statin-Naive Only** | Option A strictly within `cholesterol_medication == 0` ($N = 4,315$) | 535 | 3,780 | 204 | **12.40%** | **Etiological Purity:** Eliminates pharmacotherapy distortion entirely by filtering out treated individuals. Limited to statin-naive triage. |

#### Empirical Proof of Statin Confounding in Option A
Cross-tabulating measured Total Cholesterol ($\ge 240\text{ mg/dL}$) against Statin Use (`cholesterol_medication`):
- Among statin users ($N = 1,660$):
  - **1,571 ($94.64\%$) have Total Cholesterol $< 240\text{ mg/dL}$** (successful pharmacotherapy).
  - Only $89$ ($5.36\%$) remain $\ge 240\text{ mg/dL}$.
- Among unmedicated participants ($N = 4,315$):
  - **535 ($12.40\%$) have Total Cholesterol $\ge 240\text{ mg/dL}$** (untreated/undiagnosed hypercholesterolemia).
  - $3,780$ ($87.60\%$) have Total Cholesterol $< 240\text{ mg/dL}$.

**Conclusion for Tier 1:** Option A causes severe target distortion. **Option B is the scientifically valid target formulation** for a non-invasive screening tool, capturing true underlying phenotypic hypercholesterolemia while strictly adhering to the user's rule prohibiting arbitrary statin adjustments.

---

### B. TIER 2 Target: Framingham 10-Year Incident CVD

#### Target Column Existence
- **Finding:** The column `TARGET_10Y_CVD` **DOES NOT EXIST** precomputed in [`data/FraminghamDataset.csv`](file:///Users/gauravkumar/Desktop/CARDIO-X/data/FraminghamDataset.csv).
- **Source Columns Available:** `PERIOD`, `PREVCHD`, `PREVAP`, `PREVMI`, `PREVSTRK`, `CVD`, `TIMECVD`, `DEATH`.
- **Derivation Pipeline:** Codified in [`src/preprocessing.py`](file:///Users/gauravkumar/Desktop/CARDIO-X/src/preprocessing.py) via `build_primary_prevention_cohort()`.

#### Step-by-Step Mathematical Derivation Audit

```
Total Framingham Dataset: N = 11,627 records across 3 examination waves
├── Step 1: Baseline Wave Isolation (PERIOD == 1)
│   └── Baseline Participants: N = 4,434 unique individuals (RANDID unique)
│
├── Step 2: Prior Disease Exclusions (Secondary Prevention Exclusion)
│   ├── Prevalent Coronary Heart Disease (PREVCHD == 1): N = 114
│   ├── Prevalent Angina Pectoris (PREVAP == 1): N = 100
│   ├── Prevalent Myocardial Infarction (PREVMI == 1): N = 47
│   ├── Prevalent Stroke (PREVSTRK == 1): N = 21
│   └── Total Prior Disease Exclusions (Union): N = 219 participants
│   └── Primary Prevention Cohort: N = 4,215 disease-free at baseline
│
└── Step 3: 10-Year Outcome Construction & Indeterminate Censoring Exclusions
    ├── Positive 10-Year Events (CVD == 1 & TIMECVD <= 3652.5 days): N = 369
    ├── Confirmed Negative Controls: N = 3,614
    │   ├── Never Developed CVD during >= 10-Year Follow-up (CVD == 0 & TIMECVD >= 3652.5): N = 2,987
    │   └── Developed CVD after 10-Year Horizon (CVD == 1 & TIMECVD > 3652.5): N = 627
    │
    └── Indeterminate Cases (Follow-up Ended < 10 Years without CVD): N = 232 (EXCLUDED)
        ├── Non-CVD Deaths before 10 Years (DEATH == 1 & CVD == 0 & TIMECVD < 3652.5): N = 216
        └── Lost to Follow-up / Early Censored (DEATH == 0 & CVD == 0 & TIMECVD < 3652.5): N = 16
```

#### Final Evaluable Cohort Characteristics

| Cohort Metric | Audited Value | Percentage / Clinical Interpretation |
| :--- | :--- | :--- |
| **Total Evaluable Cohort ($N$)** | **3,983** | Exactly matches the `src/preprocessing.py` primary prevention cohort. |
| **Positive Cases (`TARGET_10Y_CVD == 1`)** | **369** | **9.26%** (First incident hard CVD event within 10 years). |
| **Negative Controls (`TARGET_10Y_CVD == 0`)** | **3,614** | **90.74%** (Confirmed CVD-free survival for $\ge 10$ years). |
| **Missing Target Labels in Evaluable Cohort** | **0** | **0.00%** complete label ascertainment. |
| **Target Reproducibility** | **100% Deterministic** | Perfectly reproducible via automated logic; zero heuristic ambiguity. |

---

## 3. TASK 2: Tier 1 Feature Leakage & Exclusion Audit (NHANES)

The goal of Tier 1 is a **non-invasive screening and lifestyle advisory engine** (deployable via telehealth, digital health apps, or clinic intake forms prior to laboratory blood draws). All 88 variables in NHANES were audited across four distinct leakage risk tiers:

```
                            NHANES VARIABLE AUDIT TAXONOMY (88 Variables)
                                                  │
         ┌────────────────────────────────────────┼────────────────────────────────────────┐
         ▼                                        ▼                                        ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐        ┌──────────────────────────────┐
│ PERMISSIBLE PREDICTORS (58)  │        │ CRITICAL LEAKAGE (12)        │        │ INVASIVE LABS (10)           │
├──────────────────────────────┤        ├──────────────────────────────┤        ├──────────────────────────────┤
│ Demographics: 11 vars        │        │ Target Definitor: 4 vars     │        │ Hepatic Panel: 3 vars        │
│ Anthropometrics: 10 vars     │        │ Lab Lipid Proxies: 4 vars    │        │ Renal Panel: 4 vars          │
│ Hemodynamics / BP: 7 vars    │        │ Lab Glucose/A1c: 2 vars      │        │ Hematology / CBC: 3 vars     │
│ Lifestyle / Habits: 11 vars  │        │ Survey Identifiers: 2 vars   │        │                              │
│ Medical History: 19 vars     │        │                              │        │                              │
└──────────────────────────────┘        └──────────────────────────────┘        └──────────────────────────────┘
```

### A. Critical Target Leakage Variables (STRICT EXCLUSION)

These variables directly define, strongly proxy, or mathematically encode the target label. Including any of them produces artificial 100% accuracy and renders the screening tool useless:

1. **Direct Target Definers (100% Leakage):**
   - `total_cholesterol_mg_dl`: The continuous serum measurement defining hypercholesterolemia.
   - `cholesterol_medication`: Part of the composite target definition (Option B & C). Missing in 26 ($0.43\%$).
   - `high_cholesterol_diagnosed`: Self-reported diagnosis of high cholesterol. Strongly collinear proxy. Missing in 34 ($0.57\%$).
   - `non_hdl_cholesterol_mg_dl`: Continuous total cholesterol minus HDL. Direct mathematical component.
2. **Direct Laboratory Lipid Proxies (Severe Leakage):**
   - `ldl_cholesterol_mg_dl`: Directly correlates with total cholesterol ($r = +0.897$). Missing in 2,850 ($47.49\%$, morning fasting subsample only).
   - `triglycerides_mg_dl`: Atherogenic lipid biomarker ($r = +0.412$). Missing in 2,813 ($46.88\%$).
   - `triglyceride_hdl_ratio`: Derived laboratory ratio.
   - `hdl_cholesterol_mg_dl`: High-density lipoprotein cholesterol. Missing in 288 ($4.80\%$).
3. **Hidden Laboratory Contamination:**
   - `diabetes_undiagnosed`: Identified exclusively via laboratory fasting plasma glucose ($\ge 126\text{ mg/dL}$) or glycohemoglobin (HbA1c $\ge 6.5\%$). Captures 233 undiagnosed cases using blood tests.
   - `glycemic_status`: Categorical score ($0=\text{Normal}, 1=\text{Prediabetes}, 2=\text{Diabetes}$) derived from laboratory HbA1c and glucose.

### B. Invasive Laboratory Variables (EXCLUDED to Maintain Non-Invasive Clinical Role)

These tests require venipuncture and laboratory processing. They violate the premise of a non-invasive intake tool:
- **Hepatic Panel:** `alt_u_l` (5.87% missing), `ast_u_l` (6.07% missing), `ggt_u_l` (5.77% missing).
- **Renal & Uric Acid Panel:** `uric_acid_mg_dl` (5.73% missing), `creatinine_mg_dl` (5.75% missing), `albumin_g_dl` (5.17% missing), `egfr_ml_min_1_73m2` (5.75% missing).
- **Complete Blood Count (CBC):** `white_blood_cell_count_1000_ul` (0.38% missing), `hemoglobin_g_dl` (0.38% missing), `platelet_count_1000_ul` (0.38% missing).

### C. Permissible Non-Invasive Candidate Predictors (58 Variables)

The remaining 58 variables represent verified non-invasive clinical and lifestyle features. A curated subset of 24 top-ranking features is recommended for Tier 1 model development:

| Domain | Recommended Features | Data Type | Missing Count ($N$, %) | Clinical Rationale & Association with Target |
| :--- | :--- | :--- | :--- | :--- |
| **Demographics** | `age_years` | continuous | 0 (0.00%) | Strongest demographic predictor of dyslipidemia ($r = +0.4619$ with Option B). |
| | `sex` | binary | 0 (0.00%) | Sexual dimorphism in post-menopausal lipid metabolism. |
| | `race_ethnicity` | categorical | 0 (0.00%) | Epidemiological disparities in lipid distribution. |
| | `education_level` | ordinal | 236 (3.93%) | Socioeconomic determinant of nutritional access. |
| | `family_income_poverty_ratio` | continuous | 769 (12.81%) | Economic barrier to healthy dietary options. Median imputed. |
| **Anthropometrics** | `bmi_kg_m2` | continuous | 87 (1.45%) | General adiposity metric. |
| | `waist_circumference_cm` | continuous | 276 (4.60%) | Central visceral adiposity ($r = +0.1835$); drives hepatic free fatty acid flux. |
| | `waist_to_height_ratio` | continuous | 284 (4.73%) | Height-normalized visceral adiposity index ($r = +0.2016$). |
| | `weight_change_1yr_kg` | continuous | 184 (3.07%) | Recent weight flux indicating metabolic instability. |
| **Vitals / Hemodynamics** | `avg_systolic_bp_mmhg` | continuous | 190 (3.17%) | Arterial stiffness correlate ($r = +0.2045$). |
| | `avg_diastolic_bp_mmhg` | continuous | 190 (3.17%) | Peripheral vascular resistance. |
| | `pulse_pressure_mmhg` | continuous | 190 (3.17%) | Hemodynamic marker of vascular compliance ($r = +0.2387$). |
| | `avg_pulse_bpm` | continuous | 190 (3.17%) | Resting autonomic tone. |
| | `bp_medication` | binary | 7 (0.12%) | Self-reported hypertension management (cardiometabolic risk marker). |
| **Lifestyle & Habits** | `smoking_status` | categorical | 9 (0.15%) | Tobacco exposure impairs reverse cholesterol transport. |
| | `total_ltp_met_minutes_week`| continuous | 71 (1.18%) | Total physical activity volume (MET-min/week). |
| | `sedentary_minutes_day` | continuous | 44 (0.73%) | Daily sedentary sitting time. |
| | `alcohol_drinks_per_week` | continuous | 36 (0.60%) | Alcohol consumption stimulates hepatic VLDL synthesis. |
| | `sleep_hours_weekday` | continuous | 63 (1.05%) | Sleep deprivation alters ghrelin/leptin and lipid clearance. |
| **Medical History** | `diabetes_diagnosed` | binary | 0 (0.00%) | Patient-reported doctor diagnosis of diabetes (non-invasive). |
| | `general_health` | ordinal | 2 (0.03%) | Self-rated overall health perception (1=Excellent to 5=Poor). |
| | `phq9_score` | continuous | 810 (13.50%) | Depressive symptom burden; linked to lifestyle disengagement. |
| | `coronary_heart_disease` | binary | 255 (4.25%) | Established macrovascular disease history. |
| | `stroke` | binary | 244 (4.07%) | Cerebrovascular disease history. |

---

## 4. TASK 3: Tier 2 Feature Audit & Temporal Validity (Framingham)

### A. Temporal Ordering & Pre-Outcome Measurement Verification

In prospective machine learning, **all predictors $\mathbf{X}$ must be measured at $t = 0$ (baseline) strictly prior to the observation window for the outcome $y$**.
- **Temporal Verification:** All baseline predictors in [`src/preprocessing.py`](file:///Users/gauravkumar/Desktop/CARDIO-X/src/preprocessing.py) were recorded at Examination Period 1 ($1948 - 1952$).
- **Follow-up Timing:** The 10-year follow-up window for incident CVD begins on the day of the baseline examination (`TIME = 0`) and extends to $3,652.5\text{ days}$ ($10\text{ years}$).
- **Conclusion:** There is **zero reverse-causality or post-baseline predictor contamination** in the baseline feature sets.

### B. Baseline Predictors Audit & Missingness in Evaluable Cohort ($N = 3,983$)

| Variable Name | Clinical Role | Data Type | Missing Count ($N$) | Missing Percentage | Imputation Strategy in Pipeline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `AGE` | Chronological age (years) | continuous | 0 | 0.00% | None required (complete). |
| `SEX` | Biological sex | binary | 0 | 0.00% | Recoded to 0/1 indicator. |
| `CURSMOKE` | Current smoking status | binary | 0 | 0.00% | None required (complete). |
| `CIGPDAY` | Cigarettes per day | continuous | 29 | 0.73% | `SimpleImputer(strategy='median')` |
| `educ` | Educational attainment | ordinal | 100 | 2.51% | `SimpleImputer(strategy='most_frequent')` |
| `BPMEDS` | Blood pressure medication | binary | 47 | 1.18% | `SimpleImputer(strategy='most_frequent')` |
| `PREVHYP` | Prevalent hypertension | binary | 0 | 0.00% | None required (complete). |
| `HEARTRTE` | Heart rate (bpm) | continuous | 1 | 0.03% | `SimpleImputer(strategy='median')` |
| `TOTCHOL` | Total serum cholesterol | continuous | 46 | 1.15% | `SimpleImputer(strategy='median')` |
| `BMI` | Body mass index ($\text{kg/m}^2$)| continuous | 10 | 0.25% | `SimpleImputer(strategy='median')` |
| `DIABETES` | Prevalent diabetes | binary | 0 | 0.00% | None required (complete). |
| `GLUCOSE` | Casual blood glucose | continuous | 364 | 9.14% | `SimpleImputer(strategy='median')` |
| `SYSBP` | Systolic blood pressure | continuous | 0 | 0.00% | None required (complete). |
| `DIABP` | Diastolic blood pressure | continuous | 0 | 0.00% | None required (complete). |
| `PULSE_PRESSURE` | Derived: $\text{SYSBP} - \text{DIABP}$ | continuous | 0 | 0.00% | None required (derived). |
| `MAP` | Derived: $\text{DIABP} + \frac{\text{PP}}{3}$ | continuous | 0 | 0.00% | None required (derived). |

#### Critical Finding on Lipid Subfractions (`HDLC`, `LDLC`)
- `HDLC` and `LDLC` have **$3,983$ missing values ($100.0\%$)** at Examination Period 1.
- In the original Framingham Study design, lipoprotein subfractions were not assayed until subsequent examination waves in the late 1960s.
- **Decision:** Models for 10-year baseline primary prevention **must rely exclusively on `TOTCHOL`** and cannot include baseline HDL or LDL.

### C. Critical Target Leakage Exclusions (Post-Baseline Follow-Up Variables)

The Framingham dataset contains 15 variables tracking longitudinal events and follow-up days. **None of these may enter any predictor feature set**:
- **Follow-up Events (Critical Leakage):** `CVD` (target component), `ANYCHD`, `STROKE`, `HOSPMI`, `MI_FCHD`, `ANGINA`, `HYPERTEN`, `DEATH`.
- **Follow-up Timers (Critical Leakage):** `TIMECVD` (target component), `TIMECHD`, `TIMESTRK`, `TIMEMI`, `TIMEMIFC`, `TIMEAP`, `TIMEHYP`, `TIMEDTH`.

### D. Label Uncertainty & Competing Risks Analysis
1. **Competing Risks of Non-CVD Death ($N = 216$):**
   - In `src/preprocessing.py`, participants who died of non-cardiovascular causes (e.g. malignancy, trauma) before 10 years without experiencing a CVD event are excluded as indeterminate.
   - *Justification:* Prevents falsely labeling a participant who died at Year 2 as a "10-year CVD-free survivor".
   - *Limitation:* May induce mild healthy-survivor attrition; in full clinical deployment, survival models with Fine-Gray subdistribution hazards should be explored as an advanced benchmark.
2. **Discrete 10-Year Horizon Thresholding:**
   - 627 participants developed CVD *after* 10 years (e.g. at Year 12 or 15). They are correctly classified as negative controls for the 10-year primary prevention window (`TARGET_10Y_CVD == 0`).
   - Models predict the discrete probability $P(\text{Event} \le 10\text{ years} \mid \mathbf{X}_{\text{baseline}})$, exactly matching the clinical utility window of the ACC/AHA ASCVD Risk Estimator.

---

## 5. TASK 4: Machine Learning Evaluation Strategies

### A. TIER 1: NHANES Hypercholesterolemia Screening Engine

```
                             TIER 1 VALIDATION ARCHITECTURE
                                            │
     ┌──────────────────────────────────────┴──────────────────────────────────────┐
     ▼                                                                             ▼
┌─────────────────────────────────────────┐               ┌─────────────────────────────────────────┐
│ MODEL OPTIMIZATION PIPELINE             │               │ DUAL EVALUATION BENCHMARK               │
├─────────────────────────────────────────┤               ├─────────────────────────────────────────┤
│ • 80/20 Stratified Split                │               │ • Unweighted Metrics (Sample Risk)      │
│   (Train: N ≈ 4,800; Test: N ≈ 1,200)   │               │ • Survey-Weighted Metrics (MEC Weight)  │
│ • SimpleImputer(median / most_frequent) │               │ • Cost-Utility Analysis:                │
│ • Scaler: RobustScaler (linear models)  │               │   Operating at Sensitivity ≥ 80% to     │
│ • Classifier: Logistic, ElasticNet,     │               │   minimize missed dyslipidemia cases    │
│   RandomForest, XGBoost                 │               │ • Calibrated via Sigmoid / Isotonic     │
└─────────────────────────────────────────┘               └─────────────────────────────────────────┘
```

#### 1. Splitting Strategy & Survey Design Accounting
- **Complex Survey Multi-Stage Clustering:** NHANES participants are sampled within 15 strata across 30 Primary Sampling Units (`survey_strata`, `survey_psu`).
- **Splitting Approach:**
  - Stratified 80/20 Train/Test Split, stratifying on `TARGET_HYPERCHOL` and demographic age/sex bands.
  - To prevent optimistic bias from spatial autocorrelation, train and test sets should be audited across PSUs.
- **Survey Weight Application (`survey_weight_mec_2yr`):**
  - *Training:* Pass `sample_weight = survey_weight_mec_2yr` during model fitting to minimize population-level generalization error rather than sample-skewed error.
  - *Reporting:* Report dual metrics: **Sample Metrics** (raw unweighted test fold performance) and **National Population Estimates** (survey-weighted test fold performance representing the non-institutionalized US civilian population).

#### 2. Class Imbalance Handling
- Target Option B exhibits an effective prevalence of **$37.88\%$** ($2,196$ positive / $3,578$ negative).
- Imbalance is mild (ratio $\approx 1 : 1.63$); extensive synthetic oversampling (SMOTE) is unnecessary and can distort probability calibration.
- Standard class weighting (`class_weight='balanced'`) in tree ensembles and logistic regression provides sufficient compensation.

#### 3. Primary Metrics & Probability Calibration
- **Primary Discriminative Metrics:**
  - **PR-AUC (Average Precision):** Baseline random rate = $0.3788$.
  - **ROC-AUC:** Assesses ranking discriminative power across all cutoffs.
- **Calibration & Decision Threshold:**
  - Fit `CalibratedClassifierCV(method='sigmoid', cv=5)` on the training fold.
  - In remote triage, **False Negatives are far more clinically costly than False Positives** (missing an individual with severe dyslipidemia who needs lipid therapy vs. recommending a non-invasive routine blood panel to an individual whose lipids are normal).
  - Select the classification decision threshold corresponding to **Sensitivity $\ge 80\% - 85\%$** rather than the arbitrary $0.50$ default.

---

### B. TIER 2: Framingham 10-Year Clinical CVD Risk Engine

```
                             TIER 2 VALIDATION ARCHITECTURE
                                            │
     ┌──────────────────────────────────────┴──────────────────────────────────────┐
     ▼                                                                             ▼
┌─────────────────────────────────────────┐               ┌─────────────────────────────────────────┐
│ STRICT TRAINING SPLIT (N = 3,186)       │               │ INDEPENDENT TEST BENCHMARK (N = 797)    │
├─────────────────────────────────────────┤               ├─────────────────────────────────────────┤
│ • Positive Events: N = 295 (9.26%)      │               │ • Positive Events: N = 74 (9.28%)       │
│ • Negative Controls: N = 2,891 (90.74%) │               │ • Negative Controls: N = 723 (90.72%)   │
│ • Imputation & Scaling fit on Train fold│               │ • Frozen Preprocessor & Scaler applied  │
│ • Class Weighting: scale_pos_weight=9.79│               │ • Clinical Risk Strata Benchmarking:    │
│ • Cross-Validation: 5-Fold Stratified CV│               │   <5%, 5-7.4%, 7.5-19.9%, ≥20% Risk     │
└─────────────────────────────────────────┘               └─────────────────────────────────────────┘
```

#### 1. Splitting Strategy & Data Partitioning
- **Strict 80/20 Stratified Partition:** Codified with `random_state=42` and `stratify=y` in `src/preprocessing.py`:
  - **Training Set (`X_train`, `y_train`):** $N = 3,186$ ($295$ positive events, $2,891$ negative controls; event prevalence $9.26\%$).
  - **Held-Out Test Set (`X_test`, `y_test`):** $N = 797$ ($74$ positive events, $723$ negative controls; event prevalence $9.28\%$).
- The test set remains completely untouched throughout feature selection and hyperparameter tuning.

#### 2. Rigorous Handling of Low Event Prevalence ($9.26\%$)
- Because only 1 in 11 participants experiences a 10-year event, naive accuracy is a pathological metric (a dummy classifier predicting 0 achieves $90.74\%$ accuracy).
- **Required Metrics Suite:**
  1. **PR-AUC (Average Precision):** Primary metric. Baseline chance = $0.0926$. Any model exceeding $0.35$ demonstrates strong predictive utility.
  2. **ROC-AUC:** Measures concordance ($C$-index). Historical Framingham risk scores achieve $\text{ROC-AUC} \approx 0.74 - 0.78$.
  3. **Brier Score:** Mean squared probability error. Quantifies absolute risk calibration.
- **Cost-Sensitive Learning:**
  - Logistic Regression / Random Forest: `class_weight='balanced'`.
  - XGBoost / LightGBM: `scale_pos_weight = \frac{2891}{295} \approx 9.80$.
  - Avoid SMOTE on clinical tabular data as it generates uncalibrated synthetic points in sparse covariate spaces.

#### 3. Probability Calibration & Clinical Decision Thresholds
- Cardiovascular risk engines inform pharmacological statin initiation based on absolute risk percentages, not discrete binary decisions:
  - **Low 10-Year Risk:** $< 5.0\%$ (Lifestyle advice only).
  - **Borderline Risk:** $5.0\% - 7.4\%$ (Selective risk enhancers considered).
  - **Intermediate Risk:** $7.5\% - 19.9\%$ (Moderate-intensity statin recommended).
  - **High Risk:** $\ge 20.0\%$ (High-intensity statin strongly recommended).
- **Calibration Requirement:** Raw ensemble outputs must be calibrated via `CalibratedClassifierCV(method='sigmoid', cv=5)` fit strictly on `X_train`.
- Calibration curves (observed proportion vs. predicted probability deciles) and Brier scores must be reported.

---

## 6. Summary of Feature Dictionary & Next Implementation Step

### A. Feature Dictionary Deliverable Summary
The companion CSV file [`data_comparison/ml_feature_dictionary.csv`](file:///Users/gauravkumar/Desktop/CARDIO-X/data_comparison/ml_feature_dictionary.csv) documents all 129 variables across both studies:

| Dataset | Tier Assignment Category | Variable Count | Leakage Risk Assessment | Primary Action in Phase 3 Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **Framingham** | `Tier 2 Predictor` | 16 | None | Core clinical predictor matrices (`standard_bp`, `hemodynamic`, `combined_tree`). |
| | `Tier 2 Target Component` | 8 | Critical / High | Target definition (`TARGET_10Y_CVD`) and baseline cohort exclusions. |
| | `Excluded - Post-Baseline Event` | 6 | Critical | Post-baseline follow-up events; excluded to prevent leakage. |
| | `Excluded - Follow-up Timing` | 8 | Critical | Follow-up day counters; excluded to prevent leakage. |
| | `Excluded - Missing at Baseline`| 2 | None (`HDLC`, `LDLC`) | $100\%$ missing at baseline Period 1; excluded. |
| | `Excluded - Identifier` | 1 (`RANDID`) | None | Drop before model fitting. |
| **NHANES** | `Tier 1 Predictor` | 58 | None / Low | Candidate non-invasive lifestyle, demographic, and vital sign features. |
| | `Tier 1 Target Component` | 4 | Critical | Target definition (`TARGET_HYPERCHOL`) components; excluded from predictors. |
| | `Excluded - Target Leakage` | 6 | Critical | Invasive lipid subfractions (`LDL`, `Trig`, `HDL`) and lab diabetes; excluded. |
| | `Excluded - Invasive Laboratory`| 10 | High | Hepatic, renal, and CBC blood panels; excluded to preserve non-invasive nature. |
| | `Excluded - Survey Design` | 7 | None | Survey weights, strata, PSUs; reserved for survey-weighted evaluation. |
| | `Excluded - Redundant Collinear`| 3 | Low | Categorically binned redundancies (`age_group`, `bmi_category`); dropped. |
| **TOTAL** | | **129** | | Complete audit registry established. |

---

### B. Recommended Next Implementation Step (Phase 3 Execution)

With target definitions verified, feature leakage boundaries codified, and evaluation strategies defined, the immediate next action is:

1. **Implement Model Architecture Pipeline in `src/models.py`:**
   - Create a new modular script [`src/models.py`](file:///Users/gauravkumar/Desktop/CARDIO-X/src/models.py) without altering existing preprocessing or EDA modules.
   - Implement baseline model trainers for both tiers:
     - **Tier 1:** Logistic Regression, ElasticNet, Random Forest, XGBoost trained on NHANES non-invasive features predicting `TARGET_HYPERCHOL_OPT_B`.
     - **Tier 2:** Logistic Regression, ElasticNet, Random Forest, XGBoost trained on Framingham baseline features predicting `TARGET_10Y_CVD`.
   - Incorporate `ColumnTransformer` pipelines with training-fold-only imputation and scaling.
   - Implement probability calibration (`CalibratedClassifierCV`) and compute PR-AUC, ROC-AUC, Brier score, and threshold operating points.
2. **Add Automated Unit Tests in `tests/test_models.py`:**
   - Validate that zero target leakage occurs across feature matrices.
   - Verify pipeline immutability and test that model outputs conform to expected probability ranges $[0, 1]$.
