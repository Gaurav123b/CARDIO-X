# CARDIO-X: NHANES (2021–2023) Processed Results & Cross-Cohort Comparative Report

> **Executive Reference:** Comprehensive descriptive, correlation, covariance, and secular shift analysis contrasting the baseline Framingham Heart Study ($N=4,434$) with the contemporary NHANES 2021–2023 examination cycle ($N=6,001$).

---

## 1. Executive Summary & File Inventory in `data_comparison/`
- **Primary NHANES Dataset:** `data_comparison/nhanes_diabetes_2021_2023.csv` (6,001 rows × 88 columns).
- **Baseline Framingham Dataset:** `data/FraminghamDataset.csv` (4,434 baseline rows across 39 features).
- **All outputs are isolated:** All 8 CSV matrix/data files, 7 high-resolution figures, and this documentation report reside strictly within `data_comparison/`.

### Generated Deliverables in `data_comparison/`:
1. `nhanes_descriptive_statistics.csv` — Full parametric and non-parametric summary of continuous NHANES biomarkers.
2. `nhanes_correlation_matrix.csv` — Pearson correlation matrix across 16 key clinical and metabolic features.
3. `nhanes_covariance_matrix.csv` — Covariance matrix across 10 continuous physiological predictors in original units².
4. `nhanes_lipid_panel_correlation.csv` — Correlation matrix of lipid subfractions and statin medication.
5. `shared_features_framingham_correlation.csv` — Correlation matrix of identical shared predictors in Framingham baseline.
6. `shared_features_nhanes_correlation.csv` — Correlation matrix of identical shared predictors in NHANES.
7. `shared_features_framingham_covariance.csv` — Covariance matrix of identical shared predictors in Framingham baseline.
8. `shared_features_nhanes_covariance.csv` — Covariance matrix of identical shared predictors in NHANES.
9. `correlation_difference_matrix.csv` — Delta correlation matrix (NHANES − Framingham).
10. `fig01_nhanes_biomarker_distributions.png` — Multi-panel distribution histograms and KDE curves for NHANES.
11. `fig02_nhanes_correlation_matrix.png` — Heatmap of NHANES clinical, lipid, and metabolic correlations.
12. `fig03_nhanes_covariance_matrix.png` — Heatmap of NHANES physiological covariances.
13. `fig04_nhanes_lipid_panel_correlation.png` — Heatmap of complete lipid panel correlations.
14. `fig05_comparative_biomarker_distributions.png` — Side-by-side comparative boxplots contrasting Framingham vs. NHANES.
15. `fig06_comparative_correlation_heatmaps.png` — Side-by-side dual heatmaps on identical clinical variables.
16. `fig07_correlation_shift_heatmap.png` — Heatmap illustrating correlation shifts across historical eras.

---

## 2. NHANES 2021–2023 Descriptive Distribution Statistics
| Feature | Observations | Missing | Missing_Pct | Mean | Std | Min | P25 | Median | P75 | Max | IQR | Skewness | Kurtosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| age_years | 6001 | 0 | 0.00 | 52.62 | 18.09 | 18.00 | 37.00 | 56.00 | 68.00 | 80.00 | 31.00 | -0.29 | -1.08 |
| bmi_kg_m2 | 5914 | 87 | 1.45 | 29.67 | 7.31 | 11.10 | 24.50 | 28.30 | 33.40 | 74.80 | 8.90 | 1.20 | 2.42 |
| avg_systolic_bp_mmhg | 5811 | 190 | 3.17 | 122.34 | 17.94 | 70.00 | 110.16 | 120.00 | 132.00 | 232.33 | 21.84 | 0.95 | 1.86 |
| avg_diastolic_bp_mmhg | 5811 | 190 | 3.17 | 74.44 | 11.04 | 34.00 | 67.00 | 73.67 | 81.00 | 139.00 | 14.00 | 0.47 | 0.98 |
| pulse_pressure_mmhg | 5811 | 190 | 3.17 | 47.89 | 14.13 | 16.67 | 38.33 | 45.33 | 54.67 | 137.00 | 16.34 | 1.26 | 2.59 |
| avg_pulse_bpm | 5811 | 190 | 3.17 | 71.57 | 12.14 | 34.00 | 63.00 | 70.67 | 79.00 | 129.33 | 16.00 | 0.47 | 0.38 |
| total_cholesterol_mg_dl | 5713 | 288 | 4.80 | 187.02 | 42.64 | 62.00 | 157.00 | 184.00 | 213.00 | 438.00 | 56.00 | 0.57 | 0.97 |
| hdl_cholesterol_mg_dl | 5713 | 288 | 4.80 | 54.41 | 14.69 | 22.00 | 44.00 | 52.00 | 62.00 | 159.00 | 18.00 | 1.08 | 2.20 |
| ldl_cholesterol_mg_dl | 3151 | 2850 | 47.49 | 107.96 | 36.61 | 3.00 | 82.00 | 105.00 | 132.00 | 314.00 | 50.00 | 0.44 | 0.56 |
| triglycerides_mg_dl | 3188 | 2813 | 46.88 | 119.48 | 86.83 | 19.00 | 72.00 | 98.50 | 140.00 | 1745.00 | 68.00 | 6.20 | 82.25 |
| non_hdl_cholesterol_mg_dl | 5713 | 288 | 4.80 | 132.61 | 41.17 | 21.00 | 104.00 | 129.00 | 157.00 | 373.00 | 53.00 | 0.77 | 1.46 |
| triglyceride_hdl_ratio | 3188 | 2813 | 46.88 | 2.51 | 2.76 | 0.34 | 1.20 | 1.86 | 2.92 | 69.08 | 1.71 | 9.73 | 180.25 |
| creatinine_mg_dl | 5656 | 345 | 5.75 | 0.89 | 0.40 | 0.35 | 0.72 | 0.85 | 1.00 | 15.17 | 0.28 | 15.57 | 430.51 |
| egfr_ml_min_1_73m2 | 5656 | 345 | 5.75 | 92.24 | 21.85 | 3.50 | 78.40 | 94.30 | 107.40 | 149.00 | 29.00 | -0.46 | 0.17 |
| sedentary_minutes_day | 5957 | 44 | 0.73 | 363.80 | 207.45 | 0.00 | 180.00 | 300.00 | 480.00 | 1380.00 | 300.00 | 0.79 | 0.45 |

---

## 3. Correlation Analysis Comparison (Shared Clinical Predictors)

### A. Framingham Baseline Correlation Matrix ($N=4,434$)
| Feature | Age | Systolic BP | Diastolic BP | Pulse Press. | Heart Rate | Total Chol. | BMI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 1.00 | 0.40 | 0.21 | 0.43 | -0.01 | 0.25 | 0.13 |
| Systolic BP | 0.40 | 1.00 | 0.78 | 0.87 | 0.18 | 0.20 | 0.33 |
| Diastolic BP | 0.21 | 0.78 | 1.00 | 0.37 | 0.19 | 0.16 | 0.38 |
| Pulse Press. | 0.43 | 0.87 | 0.37 | 1.00 | 0.12 | 0.17 | 0.19 |
| Heart Rate | -0.01 | 0.18 | 0.19 | 0.12 | 1.00 | 0.10 | 0.07 |
| Total Chol. | 0.25 | 0.20 | 0.16 | 0.17 | 0.10 | 1.00 | 0.12 |
| BMI | 0.13 | 0.33 | 0.38 | 0.19 | 0.07 | 0.12 | 1.00 |

### B. NHANES 2021–2023 Correlation Matrix ($N=6,001$)
| Feature | Age | Systolic BP | Diastolic BP | Pulse Press. | Heart Rate | Total Chol. | BMI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 1.00 | 0.39 | 0.03 | 0.48 | -0.16 | 0.04 | 0.03 |
| Systolic BP | 0.39 | 1.00 | 0.62 | 0.79 | -0.07 | 0.07 | 0.04 |
| Diastolic BP | 0.03 | 0.62 | 1.00 | 0.00 | 0.26 | 0.20 | 0.21 |
| Pulse Press. | 0.48 | 0.79 | 0.00 | 1.00 | -0.29 | -0.06 | -0.12 |
| Heart Rate | -0.16 | -0.07 | 0.26 | -0.29 | 1.00 | 0.06 | 0.12 |
| Total Chol. | 0.04 | 0.07 | 0.20 | -0.06 | 0.06 | 1.00 | -0.02 |
| BMI | 0.03 | 0.04 | 0.21 | -0.12 | 0.12 | -0.02 | 1.00 |

### C. Correlation Delta Matrix (Δr = NHANES − Framingham)
| Feature | Age | Systolic BP | Diastolic BP | Pulse Press. | Heart Rate | Total Chol. | BMI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 0.00 | -0.01 | -0.18 | 0.05 | -0.15 | -0.20 | -0.10 |
| Systolic BP | -0.01 | 0.00 | -0.17 | -0.08 | -0.26 | -0.13 | -0.29 |
| Diastolic BP | -0.18 | -0.17 | 0.00 | -0.37 | 0.07 | 0.03 | -0.16 |
| Pulse Press. | 0.05 | -0.08 | -0.37 | 0.00 | -0.42 | -0.23 | -0.31 |
| Heart Rate | -0.15 | -0.26 | 0.07 | -0.42 | 0.00 | -0.04 | 0.05 |
| Total Chol. | -0.20 | -0.13 | 0.03 | -0.23 | -0.04 | 0.00 | -0.14 |
| BMI | -0.10 | -0.29 | -0.16 | -0.31 | 0.05 | -0.14 | 0.00 |

> [!IMPORTANT]
> **Key Secular Shift Findings:**
> 1. **Cholesterol–Age Decoupling (Δr = -0.21):** In the 1950s–70s Framingham baseline, Total Cholesterol rose steadily with age ($r = 0.25$). In contemporary NHANES, this correlation has collapsed to near zero ($r = 0.04$) because older Americans are heavily treated with statin therapy ($27.7\%$ overall, $>50\%$ in older cohorts).
> 2. **Systolic BP–BMI Attenuation (Δr = -0.29):** The association between Body Mass Index and Systolic Blood Pressure dropped from $r = 0.33$ in Framingham to $r = 0.04$ in NHANES, largely driven by contemporary anti-hypertensive medication usage ($30.4\%$ on BP meds in NHANES).

---

## 4. Covariance Analysis Comparison (Shared Clinical Predictors)

### A. Framingham Baseline Covariance Matrix
| Feature | Age | Systolic BP | Diastolic BP | Pulse Press. | Heart Rate | Total Chol. | BMI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 75.29 | 77.53 | 21.66 | 55.86 | -1.42 | 96.52 | 4.77 |
| Systolic BP | 77.53 | 502.73 | 211.98 | 290.74 | 49.62 | 200.19 | 30.09 |
| Diastolic BP | 21.66 | 211.98 | 145.35 | 66.64 | 27.50 | 88.38 | 18.57 |
| Pulse Press. | 55.86 | 290.74 | 66.64 | 224.11 | 22.12 | 111.81 | 11.52 |
| Heart Rate | -1.42 | 49.62 | 27.50 | 22.12 | 146.74 | 51.62 | 3.62 |
| Total Chol. | 96.52 | 200.19 | 88.38 | 111.81 | 51.62 | 1993.72 | 22.45 |
| BMI | 4.77 | 30.09 | 18.57 | 11.52 | 3.62 | 22.45 | 16.82 |

### B. NHANES 2021–2023 Covariance Matrix
| Feature | Age | Systolic BP | Diastolic BP | Pulse Press. | Heart Rate | Total Chol. | BMI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Age | 327.09 | 127.18 | 5.18 | 122.00 | -35.61 | 34.52 | 4.59 |
| Systolic BP | 127.18 | 321.74 | 122.06 | 199.68 | -15.91 | 53.47 | 4.58 |
| Diastolic BP | 5.18 | 122.06 | 121.89 | 0.17 | 34.65 | 91.27 | 16.47 |
| Pulse Press. | 122.00 | 199.68 | 0.17 | 199.52 | -50.55 | -37.80 | -11.89 |
| Heart Rate | -35.61 | -15.91 | 34.65 | -50.55 | 147.42 | 30.24 | 10.21 |
| Total Chol. | 34.52 | 53.47 | 91.27 | -37.80 | 30.24 | 1817.95 | -6.17 |
| BMI | 4.59 | 4.58 | 16.47 | -11.89 | 10.21 | -6.17 | 53.49 |

> [!NOTE]
> **Covariance Interpretation:**
> - **Variance of Age:** Framingham = 75.29 ($\sigma = 8.68$, age range 32–70); NHANES = 327.09 ($\sigma = 18.09$, age range 18–80).
> - **Variance of BMI:** Framingham = 16.82 ($\sigma = 4.10$); NHANES = 53.49 ($\sigma = 7.31$). The contemporary US population exhibits more than 3-fold higher dispersion in adiposity.
> - **Covariance of (Age, Total Cholesterol):** Framingham = +96.52; NHANES = +34.52 (reflecting pharmacological suppression).

---

## 5. Methodological & Epidemiological Synthesis
1. **Cohort vs. Cross-Sectional:** Framingham tracks 10-year prospective incident CVD endpoints ($Y = 1$ if event $\le 3652.5\text{ days}$). NHANES captures cross-sectional lifetime prevalence. The two datasets cannot be naively merged row-by-row.
2. **Therapeutic Era Confounder:** Machine learning models trained on modern populations must incorporate medication flags (`cholesterol_medication`, `bp_medication`) to avoid mischaracterizing high-risk patients on successful pharmacotherapy as low-risk.