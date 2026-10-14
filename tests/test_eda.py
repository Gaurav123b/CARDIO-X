"""
Unit tests for Cardio-X Exploratory Data Analysis & Clinical Feature Engineering.
Verifies calculation accuracy, zero-division safety, and raw data immutability using mock data.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.eda import (
    calculate_engineered_features,
    compute_distribution_metrics,
    compute_paired_longitudinal_trends,
    analyze_outcome_surveillance_windows,
    run_exploratory_analysis,
)


class TestEDAModule(unittest.TestCase):
    """Test suite for clinical feature engineering and EDA logic."""

    def setUp(self):
        """Build deterministic mock dataframe with clinical & longitudinal properties."""
        self.mock_df = pd.DataFrame({
            "RANDID": [1, 1, 1, 2, 2, 3],
            "PERIOD": [1, 2, 3, 1, 2, 1],
            "TIME": [0, 2190, 4380, 0, 2190, 0],
            "AGE": [40, 46, 52, 50, 56, 60],
            "SYSBP": [120.0, 130.0, 140.0, 150.0, 160.0, 110.0],
            "DIABP": [80.0, 85.0, 80.0, 90.0, 95.0, 70.0],
            "TOTCHOL": [200.0, 220.0, 240.0, 260.0, 250.0, 180.0],
            "HDLC": [np.nan, np.nan, 50.0, np.nan, np.nan, 0.0],  # Includes zero to test safety
            "LDLC": [np.nan, np.nan, 140.0, np.nan, np.nan, 110.0],
            "BMI": [24.0, 25.0, 26.0, 28.0, 29.0, 22.0],
            "GLUCOSE": [80.0, 85.0, 90.0, 95.0, 100.0, 75.0],
            "CVD": [1, 1, 1, 0, 0, 0],
            "TIMECVD": [1825.0, 1825.0, 1825.0, 8766.0, 8766.0, 8766.0],  # 1825 days = 5 yrs (<= 10 yrs)
        })

    def test_engineered_features_calculation(self):
        """Verify accurate calculation of hemodynamic and lipid features."""
        feat_df = calculate_engineered_features(self.mock_df)

        # Pulse Pressure = SYSBP - DIABP
        self.assertAlmostEqual(feat_df.loc[0, "PULSE_PRESSURE"], 40.0)  # 120 - 80
        self.assertAlmostEqual(feat_df.loc[2, "PULSE_PRESSURE"], 60.0)  # 140 - 80

        # Mean Arterial Pressure = DIABP + (SYSBP - DIABP) / 3
        # Row 0: 80 + 40/3 = 93.333...
        self.assertAlmostEqual(feat_df.loc[0, "MAP"], 80.0 + 40.0 / 3.0)

        # Non-HDL Cholesterol = TOTCHOL - HDLC (Row 2: 240 - 50 = 190)
        self.assertAlmostEqual(feat_df.loc[2, "NON_HDL_CHOL"], 190.0)
        # Row 0 has NaN HDLC -> Non-HDL must be NaN
        self.assertTrue(np.isnan(feat_df.loc[0, "NON_HDL_CHOL"]))

        # Cholesterol to HDL Ratio = TOTCHOL / HDLC (Row 2: 240 / 50 = 4.8)
        self.assertAlmostEqual(feat_df.loc[2, "CHOL_HDL_RATIO"], 4.8)

    def test_zero_denominator_safety(self):
        """Verify that HDLC == 0 safely produces NaN without raising ZeroDivisionError."""
        feat_df = calculate_engineered_features(self.mock_df)
        # Row 5 has HDLC == 0.0 -> Ratio must be NaN, not error or inf
        self.assertTrue(np.isnan(feat_df.loc[5, "CHOL_HDL_RATIO"]))

    def test_input_dataframe_immutability_in_feature_calc(self):
        """Verify that calculate_engineered_features does not mutate input dataframe in-place."""
        cols_before = list(self.mock_df.columns)
        _ = calculate_engineered_features(self.mock_df)
        cols_after = list(self.mock_df.columns)
        self.assertEqual(cols_before, cols_after)

    def test_paired_longitudinal_cohort_counting(self):
        """Verify accurate extraction of subjects attending all 3 examination periods."""
        n_paired, long_df = compute_paired_longitudinal_trends(self.mock_df)
        # In mock data, only RANDID=1 has 3 periods (1, 2, 3)
        self.assertEqual(n_paired, 1)
        self.assertFalse(long_df.empty)

    def test_surveillance_window_logic(self):
        """Verify differentiation between full surveillance and 10-year window."""
        surv_df = analyze_outcome_surveillance_windows(self.mock_df)
        cvd_row = surv_df[surv_df["Endpoint"] == "CVD"].iloc[0]

        # In baseline cohort (PERIOD == 1): Total 3 participants (1, 2, 3)
        self.assertEqual(cvd_row["Baseline_Participants"], 3)
        # Participant 1 had event at 1825 days (5 years, <= 10 years) -> 1 event
        self.assertEqual(cvd_row["Full_Surveillance_Events"], 1)
        self.assertEqual(cvd_row["Ten_Year_Window_Events"], 1)

    def test_eda_pipeline_immutability(self):
        """Verify that executing run_exploratory_analysis produces reports and preserves source CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_framingham.csv"
            fig_dir = Path(tmpdir) / "figures"
            rep_dir = Path(tmpdir) / "reports"

            self.mock_df.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as f:
                initial_hash = hashlib.sha256(f.read()).hexdigest()

            result = run_exploratory_analysis(
                input_path=str(csv_path),
                output_reports_dir=str(rep_dir),
                output_figures_dir=str(fig_dir),
            )

            self.assertEqual(result["status"], "success")
            self.assertTrue(result["input_file_unchanged"])

            with open(csv_path, "rb") as f:
                final_hash = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(initial_hash, final_hash)
            self.assertTrue(Path(result["summary_report"]).is_file())
            for fig_file in result["figures"]:
                self.assertTrue(Path(fig_file).is_file())


if __name__ == "__main__":
    unittest.main()
