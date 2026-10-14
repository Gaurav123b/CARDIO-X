"""
Unit tests for Cardio-X Target Definition & Preprocessing Module.
Verifies cohort construction counts, target labeling logic, leakage exclusion,
train/test partition integrity, and raw dataset immutability.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import (
    TEN_YEAR_DAYS,
    build_primary_prevention_cohort,
    extract_feature_sets,
    create_preprocessor,
    split_dataset,
    run_preprocessing_pipeline,
)


class TestPreprocessingModule(unittest.TestCase):
    """Test suite verifying cohort construction, target definitions, and pipelines."""

    @classmethod
    def setUpClass(cls):
        """Load real Framingham dataset once for cohort verification tests."""
        cls.data_path = Path("data/FraminghamDataset.csv")
        cls.has_real_data = cls.data_path.is_file()
        if cls.has_real_data:
            cls.df_raw = pd.read_csv(cls.data_path)

    def setUp(self):
        """Build deterministic synthetic test dataframe for localized unit tests."""
        self.mock_df = pd.DataFrame({
            "RANDID": [101, 102, 103, 104, 105, 106, 107, 108],
            "PERIOD": [1, 1, 1, 1, 1, 1, 1, 1],
            "TIME": [0, 0, 0, 0, 0, 0, 0, 0],
            "AGE": [45, 50, 55, 60, 65, 48, 52, 58],
            "SEX": [1, 2, 1, 2, 1, 2, 1, 2],
            "CURSMOKE": [0, 1, 0, 1, 0, 1, 0, 1],
            "CIGPDAY": [0, 20, 0, 10, 0, 15, 0, 25],
            "educ": [2, 3, 1, 4, 2, 1, 3, 2],
            "SYSBP": [120.0, 140.0, 130.0, 160.0, 135.0, 125.0, 145.0, 150.0],
            "DIABP": [80.0, 90.0, 85.0, 95.0, 85.0, 80.0, 90.0, 95.0],
            "BPMEDS": [0, 1, 0, 1, 0, 0, 1, 0],
            "PREVHYP": [0, 1, 0, 1, 0, 0, 1, 1],
            "HEARTRTE": [70, 75, 80, 85, 72, 68, 76, 82],
            "TOTCHOL": [200.0, 240.0, 220.0, 260.0, 210.0, 190.0, 250.0, 230.0],
            "BMI": [25.0, 28.0, 26.0, 31.0, 27.0, 24.0, 29.0, 30.0],
            "DIABETES": [0, 0, 0, 1, 0, 0, 0, 1],
            "GLUCOSE": [80.0, 90.0, 85.0, 140.0, 82.0, 78.0, 95.0, 130.0],
            # Prior history columns
            "PREVCHD": [0, 1, 0, 0, 0, 0, 0, 0],   # Row 1 has prior CHD -> excluded
            "PREVAP":  [0, 0, 1, 0, 0, 0, 0, 0],   # Row 2 has prior AP  -> excluded
            "PREVMI":  [0, 0, 0, 0, 0, 0, 0, 0],
            "PREVSTRK": [0, 0, 0, 0, 0, 0, 0, 0],
            # Outcome columns
            "CVD":     [1, 1, 0, 1, 0, 0, 1, 0],
            "TIMECVD": [1500.0, 500.0, 8766.0, 4000.0, 8766.0, 2000.0, 3000.0, 8766.0],
            "DEATH":   [0, 1, 0, 0, 0, 1, 0, 0],
            "TIMEDTH": [8766.0, 500.0, 8766.0, 8766.0, 8766.0, 2000.0, 8766.0, 8766.0],
        })

    def test_cohort_assertions_on_actual_dataset(self):
        """Verify exact mathematical cohort counts against the real Framingham dataset."""
        if not self.has_real_data:
            self.skipTest("Raw FraminghamDataset.csv not found.")

        eval_df, audit = build_primary_prevention_cohort(self.df_raw, return_audit=True)

        self.assertEqual(audit["total_baseline_records"], 4434)
        self.assertEqual(audit["prior_disease_exclusions"], 219)
        self.assertEqual(audit["primary_prevention_cohort"], 4215)
        self.assertEqual(audit["indeterminate_exclusions"], 232)
        self.assertEqual(audit["indeterminate_non_cvd_deaths"], 216)
        self.assertEqual(audit["indeterminate_early_censored"], 16)
        self.assertEqual(audit["evaluable_participants"], 3983)
        self.assertEqual(audit["positive_10y_events"], 369)
        self.assertEqual(audit["negative_10y_controls"], 3614)
        self.assertEqual(audit["negative_never_cvd_ge_10y"], 2987)
        self.assertEqual(audit["negative_cvd_after_10y"], 627)
        self.assertEqual(audit["positive_rate_pct"], 9.26)
        self.assertEqual(len(eval_df), 3983)

    def test_synthetic_prior_disease_exclusion(self):
        """Verify that any pre-existing CHD/AP/MI/Stroke case is accurately filtered."""
        eval_df, audit = build_primary_prevention_cohort(self.mock_df, return_audit=True)

        # Rows 102 and 103 had prior disease -> excluded
        self.assertEqual(audit["prior_disease_exclusions"], 2)
        self.assertNotIn(102, eval_df["RANDID"].values)
        self.assertNotIn(103, eval_df["RANDID"].values)

    def test_post_10year_cvd_events_labeled_as_negative(self):
        """Verify that participant with CVD after 10 years is labeled as Y=0."""
        eval_df, _ = build_primary_prevention_cohort(self.mock_df, return_audit=True)

        # Row 104 has CVD == 1 but TIMECVD == 4000 (> 3652.5 days)
        row_104 = eval_df[eval_df["RANDID"] == 104].iloc[0]
        self.assertEqual(row_104["TARGET_10Y_CVD"], 0)

    def test_indeterminate_non_cvd_deaths_excluded(self):
        """Verify that participant who died without CVD before 10 years is excluded."""
        eval_df, audit = build_primary_prevention_cohort(self.mock_df, return_audit=True)

        # Row 106 has CVD == 0, TIMECVD == 2000 (< 3652.5) and DEATH == 1 -> indeterminate
        self.assertNotIn(106, eval_df["RANDID"].values)
        self.assertEqual(audit["indeterminate_exclusions"], 1)

    def test_feature_sets_zero_leakage(self):
        """Verify that feature matrices contain zero time, wave, or prospective outcome columns."""
        eval_df, _ = build_primary_prevention_cohort(self.mock_df, return_audit=True)
        feature_sets, y = extract_feature_sets(eval_df)

        forbidden_patterns = ["TIME", "PERIOD", "DEATH", "STROKE", "CVD", "CHD", "ANGINA", "HOSPMI", "MI_", "RANDID"]

        for name, X in feature_sets.items():
            for col in X.columns:
                for forbidden in forbidden_patterns:
                    self.assertNotIn(forbidden, col, f"Leakage detected: '{col}' in feature set '{name}'")

        # Verify y contains only binary 0/1
        self.assertTrue(set(y.unique()).issubset({0, 1}))

    def test_collinear_feature_separation(self):
        """Verify that standard_bp has only SYSBP/DIABP and hemodynamic has only PP/MAP."""
        eval_df, _ = build_primary_prevention_cohort(self.mock_df, return_audit=True)
        feature_sets, _ = extract_feature_sets(eval_df)

        X_std = feature_sets["standard_bp"]
        self.assertIn("SYSBP", X_std.columns)
        self.assertIn("DIABP", X_std.columns)
        self.assertNotIn("PULSE_PRESSURE", X_std.columns)
        self.assertNotIn("MAP", X_std.columns)

        X_hemo = feature_sets["hemodynamic"]
        self.assertIn("PULSE_PRESSURE", X_hemo.columns)
        self.assertIn("MAP", X_hemo.columns)
        self.assertNotIn("SYSBP", X_hemo.columns)
        self.assertNotIn("DIABP", X_hemo.columns)

    def test_stratified_train_test_split_isolation(self):
        """Verify stratified split proportions and zero participant overlap."""
        if not self.has_real_data:
            self.skipTest("Raw FraminghamDataset.csv not found.")

        eval_df, _ = build_primary_prevention_cohort(self.df_raw, return_audit=True)
        feature_sets, y = extract_feature_sets(eval_df)

        X = feature_sets["standard_bp"]
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=42)

        # Exact split counts on N=3983
        self.assertEqual(len(X_train), 3186)
        self.assertEqual(len(X_test), 797)
        self.assertEqual(len(X_train) + len(X_test), 3983)

        # Exact stratification counts
        self.assertEqual(int(y_train.sum()), 295)
        self.assertEqual(int(y_test.sum()), 74)
        self.assertAlmostEqual(y_train.mean(), 0.0926, places=3)
        self.assertAlmostEqual(y_test.mean(), 0.0928, places=3)

        # Zero index overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        self.assertEqual(len(train_indices.intersection(test_indices)), 0)

    def test_pipeline_preprocessor_fitting(self):
        """Verify preprocessors transform continuous and categorical features cleanly."""
        eval_df, _ = build_primary_prevention_cohort(self.mock_df, return_audit=True)
        feature_sets, y = extract_feature_sets(eval_df)
        X = feature_sets["standard_bp"]

        # Tree preprocessor
        pipe_tree = create_preprocessor(feature_set="standard_bp", model_family="tree")
        X_tree = pipe_tree.fit_transform(X)
        self.assertEqual(X_tree.shape, X.shape)
        # Check no missing values after imputation
        self.assertFalse(np.isnan(X_tree).any())

        # Linear preprocessor
        pipe_linear = create_preprocessor(feature_set="standard_bp", model_family="linear")
        X_linear = pipe_linear.fit_transform(X)
        self.assertEqual(X_linear.shape, X.shape)
        self.assertFalse(np.isnan(X_linear).any())

    def test_pipeline_immutability(self):
        """Verify that running the preprocessing pipeline preserves raw CSV hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            rep_dir = Path(tmpdir) / "reports"

            # Replicate mock data so StratifiedShuffleSplit has sufficient class instances
            test_mock = pd.concat([self.mock_df] * 5, ignore_index=True)
            test_mock["RANDID"] = range(1, len(test_mock) + 1)
            test_mock.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as f:
                initial_hash = hashlib.sha256(f.read()).hexdigest()

            result = run_preprocessing_pipeline(
                input_path=str(csv_path),
                output_reports_dir=str(rep_dir),
                assert_full_cohort=False
            )


            self.assertEqual(result["status"], "success")
            self.assertTrue(result["input_file_unchanged"])


            with open(csv_path, "rb") as f:
                final_hash = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(initial_hash, final_hash)
            self.assertTrue(Path(result["report_file"]).is_file())


if __name__ == "__main__":
    unittest.main()
