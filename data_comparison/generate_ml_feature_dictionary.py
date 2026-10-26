"""
Script to generate data_comparison/ml_feature_dictionary.csv for CARDIO-X Phase 2.
Audits all 88 NHANES variables and 39 (+2 derived) Framingham variables.
Distinguishes valid predictors, target definers, leakage sources, invasive labs, and survey design variables.
"""

import pandas as pd
import numpy as np
from src.preprocessing import build_primary_prevention_cohort

def build_dictionary():
    # 1. Load microdata
    f_raw = pd.read_csv("data/FraminghamDataset.csv")
    f_eval, f_audit = build_primary_prevention_cohort(f_raw)
    n_df = pd.read_csv("data_comparison/nhanes_diabetes_2021_2023.csv")

    rows = []

    # ==========================================
    # FRAMINGHAM VARIABLES (39 raw + 2 derived)
    # ==========================================
    
    framingham_meta = {
        "RANDID": {
            "tier": "Excluded - Identifier",
            "domain": "Survey Design",
            "type": "identifier",
            "leakage": "None",
            "notes": "Unique participant identifier. Must be dropped prior to modeling."
        },
        "SEX": {
            "tier": "Tier 2 Predictor",
            "domain": "Demographics",
            "type": "binary",
            "leakage": "None",
            "notes": "Biological sex (1=Male, 2=Female). Coded to 0/1 indicator in pipeline."
        },
        "TOTCHOL": {
            "tier": "Tier 2 Predictor",
            "domain": "Lipid Panel",
            "type": "continuous",
            "leakage": "None",
            "notes": "Serum total cholesterol (mg/dL) measured at baseline. Key independent biomarker for 10y CVD."
        },
        "AGE": {
            "tier": "Tier 2 Predictor",
            "domain": "Demographics",
            "type": "continuous",
            "leakage": "None",
            "notes": "Age in years at baseline examination (range 32-70)."
        },
        "SYSBP": {
            "tier": "Tier 2 Predictor",
            "domain": "Hemodynamics",
            "type": "continuous",
            "leakage": "None",
            "notes": "Systolic blood pressure (mmHg). Evaluated in standard_bp and combined_tree sets."
        },
        "DIABP": {
            "tier": "Tier 2 Predictor",
            "domain": "Hemodynamics",
            "type": "continuous",
            "leakage": "None",
            "notes": "Diastolic blood pressure (mmHg). Evaluated in standard_bp and combined_tree sets."
        },
        "CURSMOKE": {
            "tier": "Tier 2 Predictor",
            "domain": "Lifestyle / Behavioral",
            "type": "binary",
            "leakage": "None",
            "notes": "Current cigarette smoker at baseline (0=No, 1=Yes)."
        },
        "CIGPDAY": {
            "tier": "Tier 2 Predictor",
            "domain": "Lifestyle / Behavioral",
            "type": "continuous",
            "leakage": "None",
            "notes": "Number of cigarettes smoked per day at baseline (0 if non-smoker)."
        },
        "BMI": {
            "tier": "Tier 2 Predictor",
            "domain": "Anthropometrics",
            "type": "continuous",
            "leakage": "None",
            "notes": "Body mass index (kg/m^2) measured at baseline."
        },
        "DIABETES": {
            "tier": "Tier 2 Predictor",
            "domain": "Medical History",
            "type": "binary",
            "leakage": "None",
            "notes": "Prevalent diabetes mellitus at baseline (0=No, 1=Yes)."
        },
        "BPMEDS": {
            "tier": "Tier 2 Predictor",
            "domain": "Medical History",
            "type": "binary",
            "leakage": "None",
            "notes": "Use of anti-hypertensive medication at baseline (0=No, 1=Yes)."
        },
        "HEARTRTE": {
            "tier": "Tier 2 Predictor",
            "domain": "Hemodynamics",
            "type": "continuous",
            "leakage": "None",
            "notes": "Resting heart rate / ventricular rate (beats/min) at baseline."
        },
        "GLUCOSE": {
            "tier": "Tier 2 Predictor",
            "domain": "Non-Lipid Laboratory",
            "type": "continuous",
            "leakage": "None",
            "notes": "Casual blood glucose (mg/dL) at baseline. Has 9.14% missingness in evaluable cohort; median imputed."
        },
        "educ": {
            "tier": "Tier 2 Predictor",
            "domain": "Demographics",
            "type": "ordinal",
            "leakage": "None",
            "notes": "Education level (1=0-11 yrs, 2=high school, 3=some college, 4=college degree)."
        },
        "PREVCHD": {
            "tier": "Tier 2 Target Component",
            "domain": "Medical History",
            "type": "binary",
            "leakage": "High",
            "notes": "Prevalent coronary disease at baseline. Used exclusively to filter secondary prevention cases."
        },
        "PREVAP": {
            "tier": "Tier 2 Target Component",
            "domain": "Medical History",
            "type": "binary",
            "leakage": "High",
            "notes": "Prevalent angina pectoris at baseline. Excluded in primary prevention definition."
        },
        "PREVMI": {
            "tier": "Tier 2 Target Component",
            "domain": "Medical History",
            "type": "binary",
            "leakage": "High",
            "notes": "Prevalent myocardial infarction at baseline. Excluded in primary prevention definition."
        },
        "PREVSTRK": {
            "tier": "Tier 2 Target Component",
            "domain": "Medical History",
            "type": "binary",
            "leakage": "High",
            "notes": "Prevalent stroke at baseline. Excluded in primary prevention definition."
        },
        "PREVHYP": {
            "tier": "Tier 2 Predictor",
            "domain": "Medical History",
            "type": "binary",
            "leakage": "None",
            "notes": "Prevalent hypertension at baseline (SYSBP>=140, DIABP>=90, or on BPMEDS)."
        },
        "TIME": {
            "tier": "Excluded - Follow-up Timing",
            "domain": "Survey Design",
            "type": "continuous",
            "leakage": "Moderate",
            "notes": "Days from baseline exam. Constant 0 at period 1."
        },
        "PERIOD": {
            "tier": "Tier 2 Target Component",
            "domain": "Survey Design",
            "type": "ordinal",
            "leakage": "High",
            "notes": "Examination wave (1, 2, 3). Used to isolate baseline observation (PERIOD==1)."
        },
        "HDLC": {
            "tier": "Excluded - Missing at Baseline",
            "domain": "Lipid Panel",
            "type": "continuous",
            "leakage": "None",
            "notes": "High-density lipoprotein cholesterol (mg/dL). 100% missing in Period 1 baseline cohort; cannot be used."
        },
        "LDLC": {
            "tier": "Excluded - Missing at Baseline",
            "domain": "Lipid Panel",
            "type": "continuous",
            "leakage": "None",
            "notes": "Low-density lipoprotein cholesterol (mg/dL). 100% missing in Period 1 baseline cohort; cannot be used."
        },
        "DEATH": {
            "tier": "Tier 2 Target Component",
            "domain": "Outcome / Time-to-Event",
            "type": "binary",
            "leakage": "Critical",
            "notes": "All-cause mortality over follow-up. Used to identify competing risks during 10-year horizon."
        },
        "ANGINA": {
            "tier": "Excluded - Post-Baseline Event",
            "domain": "Outcome / Time-to-Event",
            "type": "binary",
            "leakage": "Critical",
            "notes": "Incident angina over follow-up. Target leakage if used as predictor."
        },
        "HOSPMI": {
            "tier": "Excluded - Post-Baseline Event",
            "domain": "Outcome / Time-to-Event",
            "type": "binary",
            "leakage": "Critical",
            "notes": "Incident hospitalized MI over follow-up. Target leakage if used as predictor."
        },
        "MI_FCHD": {
            "tier": "Excluded - Post-Baseline Event",
            "domain": "Outcome / Time-to-Event",
            "type": "binary",
            "leakage": "Critical",
            "notes": "Incident MI or fatal CHD over follow-up. Target leakage if used as predictor."
        },
        "ANYCHD": {
            "tier": "Excluded - Post-Baseline Event",
            "domain": "Outcome / Time-to-Event",
            "type": "binary",
            "leakage": "Critical",
            "notes": "Incident coronary heart disease over follow-up. Target leakage if used as predictor."
        },
        "STROKE": {
            "tier": "Excluded - Post-Baseline Event",
            "domain": "Outcome / Time-to-Event",
            "type": "binary",
            "leakage": "Critical",
            "notes": "Incident stroke over follow-up. Target leakage if used as predictor."
        },
        "CVD": {
            "tier": "Tier 2 Target Component",
            "domain": "Outcome / Time-to-Event",
            "type": "binary",
            "leakage": "Critical",
            "notes": "Overall incident CVD event indicator. Core source column for TARGET_10Y_CVD."
        },
        "HYPERTEN": {
            "tier": "Excluded - Post-Baseline Event",
            "domain": "Outcome / Time-to-Event",
            "type": "binary",
            "leakage": "Critical",
            "notes": "Incident hypertension over follow-up. Target leakage if used as predictor."
        },
        "TIMEAP": {
            "tier": "Excluded - Follow-up Timing",
            "domain": "Outcome / Time-to-Event",
            "type": "continuous",
            "leakage": "Critical",
            "notes": "Follow-up days to incident angina. Target leakage."
        },
        "TIMEMI": {
            "tier": "Excluded - Follow-up Timing",
            "domain": "Outcome / Time-to-Event",
            "type": "continuous",
            "leakage": "Critical",
            "notes": "Follow-up days to incident MI. Target leakage."
        },
        "TIMEMIFC": {
            "tier": "Excluded - Follow-up Timing",
            "domain": "Outcome / Time-to-Event",
            "type": "continuous",
            "leakage": "Critical",
            "notes": "Follow-up days to recognized MI/fatal CHD. Target leakage."
        },
        "TIMECHD": {
            "tier": "Excluded - Follow-up Timing",
            "domain": "Outcome / Time-to-Event",
            "type": "continuous",
            "leakage": "Critical",
            "notes": "Follow-up days to incident CHD. Target leakage."
        },
        "TIMESTRK": {
            "tier": "Excluded - Follow-up Timing",
            "domain": "Outcome / Time-to-Event",
            "type": "continuous",
            "leakage": "Critical",
            "notes": "Follow-up days to incident stroke. Target leakage."
        },
        "TIMECVD": {
            "tier": "Tier 2 Target Component",
            "domain": "Outcome / Time-to-Event",
            "type": "continuous",
            "leakage": "Critical",
            "notes": "Follow-up days to first incident CVD event or censoring. Core source column for TARGET_10Y_CVD."
        },
        "TIMEDTH": {
            "tier": "Excluded - Follow-up Timing",
            "domain": "Outcome / Time-to-Event",
            "type": "continuous",
            "leakage": "Critical",
            "notes": "Follow-up days to all-cause death. Target leakage."
        },
        "TIMEHYP": {
            "tier": "Excluded - Follow-up Timing",
            "domain": "Outcome / Time-to-Event",
            "type": "continuous",
            "leakage": "Critical",
            "notes": "Follow-up days to incident hypertension. Target leakage."
        },
        "PULSE_PRESSURE": {
            "tier": "Tier 2 Predictor",
            "domain": "Hemodynamics",
            "type": "continuous",
            "leakage": "None",
            "notes": "Derived baseline feature (SYSBP - DIABP). Evaluated in hemodynamic and combined_tree sets."
        },
        "MAP": {
            "tier": "Tier 2 Predictor",
            "domain": "Hemodynamics",
            "type": "continuous",
            "leakage": "None",
            "notes": "Derived baseline feature (DIABP + PP/3). Evaluated in hemodynamic and combined_tree sets."
        }
    }

    for col, meta in framingham_meta.items():
        if col in ["PULSE_PRESSURE", "MAP"]:
            raw_miss = 0
            raw_pct = 0.0
            eval_miss = 0
            eval_pct = 0.0
        else:
            raw_miss = int(f_raw[col].isna().sum())
            raw_pct = round(raw_miss / len(f_raw) * 100, 2)
            eval_miss = int(f_eval[col].isna().sum()) if col in f_eval.columns else int(f_raw[f_raw["PERIOD"]==1][col].isna().sum())
            eval_pct = round(eval_miss / len(f_eval) * 100, 2)

        rows.append({
            "dataset": "Framingham",
            "variable_name": col,
            "tier_assignment": meta["tier"],
            "clinical_domain": meta["domain"],
            "data_type": meta["type"],
            "raw_missing_count": raw_miss,
            "raw_missing_pct": raw_pct,
            "evaluable_missing_count": eval_miss,
            "evaluable_missing_pct": eval_pct,
            "leakage_risk_level": meta["leakage"],
            "notes_and_recommendation": meta["notes"]
        })

    # ==========================================
    # NHANES VARIABLES (88 variables)
    # ==========================================

    # Categorization map for NHANES
    def classify_nhanes(col):
        # Target Definitor / Component
        if col in ["total_cholesterol_mg_dl", "high_cholesterol_diagnosed", "cholesterol_medication", "non_hdl_cholesterol_mg_dl"]:
            return ("Tier 1 Target Component", "Lipid Panel", "Critical", 
                    "Direct component of proposed TARGET_HYPERCHOL definitions. Excluded from predictors.")
        # Direct Lab Proxies / Lipid Panel Leakage
        if col in ["ldl_cholesterol_mg_dl", "triglycerides_mg_dl", "triglyceride_hdl_ratio", "hdl_cholesterol_mg_dl"]:
            return ("Excluded - Target Leakage", "Lipid Panel", "Critical",
                    "Laboratory lipid subfraction directly correlated with target. Severe leakage for non-invasive model.")
        # Invasive Laboratory Chemistry & Hematology
        if col in ["alt_u_l", "ast_u_l", "ggt_u_l"]:
            return ("Excluded - Invasive Laboratory", "Non-Lipid Laboratory", "High",
                    "Hepatic transaminase enzyme panel. Inaccessible in non-invasive telemedicine triage.")
        if col in ["uric_acid_mg_dl", "creatinine_mg_dl", "albumin_g_dl", "egfr_ml_min_1_73m2"]:
            return ("Excluded - Invasive Laboratory", "Non-Lipid Laboratory", "High",
                    "Renal and metabolic serum panel. Inaccessible in non-invasive triage.")
        if col in ["white_blood_cell_count_1000_ul", "hemoglobin_g_dl", "platelet_count_1000_ul"]:
            return ("Excluded - Invasive Laboratory", "Non-Lipid Laboratory", "High",
                    "Complete blood count (CBC). Inaccessible in non-invasive screening.")
        if col in ["diabetes_undiagnosed", "glycemic_status"]:
            return ("Excluded - Target Leakage", "Non-Lipid Laboratory", "Critical",
                    "Constructed using laboratory fasting plasma glucose and HbA1c. Excluded to maintain purely non-invasive inputs.")
        # Survey Design Variables
        if col in ["participant_id", "survey_cycle", "survey_weight_interview_2yr", "survey_weight_mec_2yr", "survey_weight_fasting_2yr", "survey_strata", "survey_psu"]:
            return ("Excluded - Survey Design", "Survey Design", "None",
                    "Survey design and weighting variables. survey_weight_mec_2yr used in evaluation; not feature predictors.")
        
        # Valid Demographics
        if col in ["age_years", "sex", "race_ethnicity", "born_in_us"]:
            return ("Tier 1 Predictor", "Demographics", "None", "Core non-invasive demographic predictor.")
        if col in ["age_group", "education_code"]:
            return ("Excluded - Redundant Collinear", "Demographics", "Low", "Redundant binned coding of age_years or education_level.")
        if col in ["education_level", "marital_status", "work_status", "has_health_insurance", "routine_healthcare_place"]:
            return ("Tier 1 Predictor", "Demographics", "None", "Socio-demographic lifestyle and healthcare access predictor.")
        if col in ["family_income_poverty_ratio", "income_poverty_category"]:
            return ("Tier 1 Predictor", "Demographics", "None", "Economic hardship indicators. continuous ratio has 12.8% missingness; category is binned.")
        if col == "pregnant":
            return ("Tier 1 Predictor", "Demographics", "Low", "Pregnancy status. Alters lipid physiology; examine exclusions or indicator.")
        
        # Valid Anthropometrics
        if col in ["bmi_kg_m2", "weight_kg", "height_cm", "waist_circumference_cm", "hip_circumference_cm", 
                   "arm_circumference_cm", "waist_to_height_ratio", "waist_to_hip_ratio", "weight_change_1yr_kg", "tried_to_lose_weight_past_year"]:
            return ("Tier 1 Predictor", "Anthropometrics", "None", "Non-invasive body composition measurement. Central adiposity correlates with dyslipidemia.")
        if col == "bmi_category":
            return ("Excluded - Redundant Collinear", "Anthropometrics", "Low", "Categorical binning of continuous bmi_kg_m2.")
        
        # Valid Hemodynamics
        if col in ["avg_systolic_bp_mmhg", "avg_diastolic_bp_mmhg", "avg_pulse_bpm", "pulse_pressure_mmhg", "hypertension_diagnosed", "bp_medication"]:
            return ("Tier 1 Predictor", "Hemodynamics", "None", "Non-invasive vital signs and blood pressure management history.")
        if col == "hypertension_measured":
            return ("Tier 1 Predictor", "Hemodynamics", "None", "Clinical stage hypertension indicator based on measured BP cutoffs (>=130/80).")
        
        # Valid Lifestyle & Behavioral
        if col in ["moderate_ltp_minutes_week", "vigorous_ltp_minutes_week", "total_ltp_met_minutes_week", "meets_activity_guidelines", 
                   "sedentary_minutes_day", "smoked_100_cigarettes", "smoking_status", "alcohol_drinks_per_week", "heavy_alcohol_use", 
                   "sleep_hours_weekday", "sleep_hours_weekend"]:
            return ("Tier 1 Predictor", "Lifestyle / Behavioral", "None", "Self-reported physical activity, sedentary behavior, tobacco, alcohol, and sleep habits.")
        
        # Patient-Reported Health Outcomes
        if col in ["phq9_score", "depression_screen_positive", "general_health", "general_health_label", "mental_health_care_past_year"]:
            return ("Tier 1 Predictor", "Medical History", "None", "Self-reported mental and general health perception. phq9 has 13.5% missingness.")
        if col in ["asthma", "congestive_heart_failure", "coronary_heart_disease", "angina", "heart_attack", "stroke", 
                   "liver_condition", "thyroid_problem", "cancer_any", "weak_or_failing_kidneys", "heart_disease_or_attack"]:
            return ("Tier 1 Predictor", "Medical History", "Low", "Self-reported chronic disease history. Permissible non-invasive medical history features.")
        if col in ["diabetes", "diabetes_diagnosed"]:
            return ("Tier 1 Predictor", "Medical History", "Low", "Self-reported doctor-diagnosed diabetes history. Distinct from undiagnosed lab-based diabetes.")

        return ("Tier 1 Candidate", "Other", "Low", "Inspect individually.")

    for col in n_df.columns:
        tier, domain, leakage, notes = classify_nhanes(col)
        raw_miss = int(n_df[col].isna().sum())
        raw_pct = round(raw_miss / len(n_df) * 100, 2)
        dtype = str(n_df[col].dtype)
        if dtype in ["int64", "float64"]:
            if n_df[col].nunique() == 2 and set(n_df[col].dropna().unique()).issubset({0, 1}):
                vtype = "binary"
            elif n_df[col].nunique() <= 5:
                vtype = "ordinal"
            else:
                vtype = "continuous"
        elif dtype == "object" or dtype == "str":
            vtype = "categorical"
        else:
            vtype = dtype

        if col == "participant_id":
            vtype = "identifier"

        rows.append({
            "dataset": "NHANES",
            "variable_name": col,
            "tier_assignment": tier,
            "clinical_domain": domain,
            "data_type": vtype,
            "raw_missing_count": raw_miss,
            "raw_missing_pct": raw_pct,
            "evaluable_missing_count": raw_miss,
            "evaluable_missing_pct": raw_pct,
            "leakage_risk_level": leakage,
            "notes_and_recommendation": notes
        })

    dict_df = pd.DataFrame(rows)
    out_path = "data_comparison/ml_feature_dictionary.csv"
    dict_df.to_csv(out_path, index=False)
    print(f"Successfully created {out_path} with {len(dict_df)} feature definitions.")

    # Summary table
    print("\nSummary by Dataset and Tier Assignment:")
    print(dict_df.groupby(["dataset", "tier_assignment"]).size())

if __name__ == "__main__":
    build_dictionary()
