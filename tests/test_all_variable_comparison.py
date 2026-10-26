"""
Unit test suite for CARDIO-X all-variable comparative analysis module.
Validates inventory generation, mapping classifications, correlation logic,
Fisher z-transformations, multiple testing adjustments, and deliverable integrity.
Uses synthetic fixtures for mathematical verification.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from data_comparison.run_all_variable_comparison import (
    build_variable_inventory,
    build_variable_mapping,
    compute_correlation_differences_and_tests,
    compute_pairwise_correlation_matrix,
    harmonize_comparable_data,
    load_datasets,
)


# ---------------------------------------------------------------------------
# Synthetic Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_cohorts():
    """Provides synthetic DataFrames mimicking Framingham baseline and NHANES."""
    np.random.seed(42)
    n = 200

    f_synth = pd.DataFrame({
        "AGE": np.random.randint(35, 70, size=n),
        "TOTCHOL": np.random.normal(230, 40, size=n),
        "SYSBP": np.random.normal(130, 20, size=n),
        "DIABP": np.random.normal(82, 10, size=n),
        "HEARTRTE": np.random.normal(75, 11, size=n),
        "BMI": np.random.normal(25.5, 3.8, size=n),
        "SEX": np.random.choice([1, 2], size=n),
        "CURSMOKE": np.random.choice([0, 1], size=n),
        "DIABETES": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "BPMEDS": np.random.choice([0, 1], size=n, p=[0.92, 0.08]),
        "educ": np.random.choice([1.0, 2.0, 3.0, 4.0], size=n),
        "PREVCHD": np.random.choice([0, 1], size=n, p=[0.97, 0.03]),
        "PREVAP": np.random.choice([0, 1], size=n, p=[0.98, 0.02]),
        "PREVMI": np.random.choice([0, 1], size=n, p=[0.98, 0.02]),
        "PREVSTRK": np.random.choice([0, 1], size=n, p=[0.99, 0.01]),
        "PERIOD": [1] * n
    })
    f_synth["PULSE_PRESSURE"] = f_synth["SYSBP"] - f_synth["DIABP"]
    f_synth["MAP"] = f_synth["DIABP"] + f_synth["PULSE_PRESSURE"] / 3.0

    # Inject minor missingness
    f_synth.loc[0:4, "TOTCHOL"] = np.nan
    f_synth.loc[5:7, "BMI"] = np.nan

    n_synth = pd.DataFrame({
        "age_years": np.random.randint(18, 80, size=n),
        "total_cholesterol_mg_dl": np.random.normal(190, 42, size=n),
        "avg_systolic_bp_mmhg": np.random.normal(124, 17, size=n),
        "avg_diastolic_bp_mmhg": np.random.normal(76, 11, size=n),
        "pulse_pressure_mmhg": np.random.normal(48, 14, size=n),
        "avg_pulse_bpm": np.random.normal(73, 12, size=n),
        "bmi_kg_m2": np.random.normal(29.8, 7.1, size=n),
        "sex": np.random.choice(["male", "female"], size=n),
        "smoking_status": np.random.choice(["never", "former", "current"], size=n, p=[0.55, 0.30, 0.15]),
        "diabetes_diagnosed": np.random.choice([0, 1], size=n, p=[0.88, 0.12]),
        "bp_medication": np.random.choice([0, 1], size=n, p=[0.70, 0.30]),
        "education_code": np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n),
        "coronary_heart_disease": np.random.choice([0.0, 1.0], size=n, p=[0.95, 0.05]),
        "angina": np.random.choice([0.0, 1.0], size=n, p=[0.97, 0.03]),
        "heart_attack": np.random.choice([0.0, 1.0], size=n, p=[0.96, 0.04]),
        "stroke": np.random.choice([0.0, 1.0], size=n, p=[0.96, 0.04]),
    })
    n_synth["MAP"] = n_synth["avg_diastolic_bp_mmhg"] + n_synth["pulse_pressure_mmhg"] / 3.0

    return f_synth, n_synth


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestAllVariableComparison:

    def test_load_datasets_and_hemodynamics(self):
        """Verifies dataset loading and mathematical correctness of derived hemodynamics."""
        f_raw, f_base, n_df = load_datasets()
        assert len(f_raw) == 11627
        assert len(f_base) == 4434
        assert len(n_df) == 6001

        # Check derived pulse pressure identity: PP == SBP - DBP
        f_valid_bp = f_base[["SYSBP", "DIABP", "PULSE_PRESSURE"]].dropna()
        np.testing.assert_allclose(
            f_valid_bp["PULSE_PRESSURE"],
            f_valid_bp["SYSBP"] - f_valid_bp["DIABP"],
            rtol=1e-5
        )

        # Check derived MAP identity: MAP == DBP + (SYSBP - DBP)/3
        f_valid_map = f_base[["SYSBP", "DIABP", "MAP"]].dropna()
        np.testing.assert_allclose(
            f_valid_map["MAP"],
            f_valid_map["DIABP"] + (f_valid_map["SYSBP"] - f_valid_map["DIABP"]) / 3.0,
            rtol=1e-5
        )

    def test_variable_inventory_generation(self, synthetic_cohorts):
        """Tests that variable inventory handles missingness, dtypes, and classifications properly."""
        f_synth, _ = synthetic_cohorts
        inv_df = build_variable_inventory(f_synth, "Framingham", baseline_df=f_synth)

        assert len(inv_df) == len(f_synth.columns)
        assert set(inv_df["dataset"].unique()) == {"Framingham"}
        assert "variable_name" in inv_df.columns
        assert "missing_count" in inv_df.columns
        assert "classification" in inv_df.columns

        # Verify missing counts match injected synthetic missingness
        totchol_row = inv_df[inv_df["variable_name"] == "TOTCHOL"].iloc[0]
        assert totchol_row["missing_count"] == 5
        assert totchol_row["missing_pct"] == round(5 / len(f_synth) * 100.0, 2)

    def test_variable_mapping_structure(self, synthetic_cohorts):
        """Tests that variable mapping contains valid classifications (A, B, C, D)."""
        f_synth, n_synth = synthetic_cohorts
        map_df = build_variable_mapping(f_synth, n_synth)

        valid_statuses = {
            "A: Directly comparable",
            "B: Comparable after documented transformation",
            "C: Conceptually related but not directly comparable",
            "D: Not comparable"
        }
        assert set(map_df["comparability_status"].unique()).issubset(valid_statuses)
        assert "framingham_variable" in map_df.columns
        assert "nhanes_variable" in map_df.columns
        assert "exclusion_reason" in map_df.columns

        # Verify key Tier A pairs
        tier_a = map_df[map_df["comparability_status"] == "A: Directly comparable"]
        tier_a_fvars = set(tier_a["framingham_variable"])
        assert {"AGE", "TOTCHOL", "SYSBP", "DIABP", "HEARTRTE", "BMI", "BPMEDS"}.issubset(tier_a_fvars)

    def test_harmonize_comparable_data(self, synthetic_cohorts):
        """Tests that harmonized dataframes contain standardized clinical columns with identical names."""
        f_synth, n_synth = synthetic_cohorts
        f_clean, n_clean = harmonize_comparable_data(f_synth, n_synth)

        expected_continuous = [
            "Age", "Total_Cholesterol", "Systolic_BP", "Diastolic_BP",
            "Pulse_Pressure", "Mean_Arterial_Pressure", "Heart_Rate", "BMI"
        ]
        for col in expected_continuous:
            assert col in f_clean.columns
            assert col in n_clean.columns

        # Verify Sex recoding: Male=1, Female=0
        assert set(f_clean["Sex_Male"].dropna().unique()).issubset({0, 1})
        assert set(n_clean["Sex_Male"].dropna().unique()).issubset({0, 1})

        # Verify Smoking recoding: Current=1, Others=0
        assert set(f_clean["Current_Smoker"].dropna().unique()).issubset({0.0, 1.0})
        assert set(n_clean["Current_Smoker"].dropna().unique()).issubset({0.0, 1.0})

    def test_pairwise_correlation_matrix_math(self):
        """Tests that pairwise correlation respects NaNs and calculates exact sample sizes."""
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "z": [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        })
        corr_mat, n_mat = compute_pairwise_correlation_matrix(df, ["x", "y", "z"], method="pearson")

        # x vs y has 5 complete observations and r = 1.0
        assert n_mat.loc["x", "y"] == 5
        assert pytest.approx(corr_mat.loc["x", "y"], abs=1e-5) == 1.0

        # y vs z has 6 complete observations and r = -1.0
        assert n_mat.loc["y", "z"] == 6
        assert pytest.approx(corr_mat.loc["y", "z"], abs=1e-5) == -1.0

    def test_fisher_z_and_multiple_testing_adjustments(self):
        """Tests Fisher r-to-z test, Bonferroni, and Benjamini-Hochberg FDR adjustments."""
        vars_list = ["v1", "v2", "v3"]
        f_corr = pd.DataFrame(
            [[1.0, 0.40, 0.20], [0.40, 1.0, 0.30], [0.20, 0.30, 1.0]],
            index=vars_list, columns=vars_list
        )
        n_corr = pd.DataFrame(
            [[1.0, 0.10, 0.20], [0.10, 1.0, -0.10], [0.20, -0.10, 1.0]],
            index=vars_list, columns=vars_list
        )
        f_n = pd.DataFrame(1000, index=vars_list, columns=vars_list)
        n_n = pd.DataFrame(1000, index=vars_list, columns=vars_list)

        diff_df = compute_correlation_differences_and_tests(f_corr, n_corr, f_n, n_n, vars_list)
        assert len(diff_df) == 3
        assert "Z_Statistic" in diff_df.columns
        assert "P_Value_Bonferroni" in diff_df.columns
        assert "Q_Value_FDR" in diff_df.columns

        # Verify Bonferroni is always >= raw p-value
        for _, row in diff_df.iterrows():
            assert row["P_Value_Bonferroni"] >= row["P_Value_Raw"]
            assert row["Q_Value_FDR"] >= row["P_Value_Raw"]

    def test_deliverable_files_existence_and_integrity(self):
        """Verifies that all 7 deliverables and 12 figures exist on disk and have non-zero size."""
        base_dir = Path("data_comparison/all_variable_analysis")
        fig_dir = base_dir / "figures"

        required_csvs = [
            "variable_inventory_framingham.csv",
            "variable_inventory_nhanes.csv",
            "variable_mapping.csv",
            "correlation_sample_sizes.csv",
            "correlation_difference_ranking.csv",
            "analysis_summary.json",
            "all_variable_comparison_report.md"
        ]
        for f in required_csvs:
            file_path = base_dir / f
            assert file_path.exists(), f"Missing deliverable: {file_path}"
            assert file_path.stat().st_size > 50, f"File unexpectedly empty: {file_path}"

        # Check JSON integrity
        with open(base_dir / "analysis_summary.json") as jf:
            summary = json.load(jf)
            assert "dataset_dimensions" in summary
            assert summary["dataset_dimensions"]["framingham_baseline"]["rows"] == 4434
            assert summary["dataset_dimensions"]["nhanes_2021_2023"]["rows"] == 6001

        # Check all 12 figures exist and have valid PNG headers
        for i in range(1, 13):
            fig_files = list(fig_dir.glob(f"{i:02d}_*.png"))
            assert len(fig_files) == 1, f"Missing figure with index {i:02d}"
            fig_path = fig_files[0]
            assert fig_path.stat().st_size > 10000, f"Figure unexpectedly small: {fig_path}"
            # Check PNG magic bytes
            with open(fig_path, "rb") as pf:
                header = pf.read(8)
                assert header == b"\x89PNG\r\n\x1a\n", f"Invalid PNG magic bytes: {fig_path}"
