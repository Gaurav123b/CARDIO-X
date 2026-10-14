"""
Unit tests for Cardio-X data quality analysis module.
Uses small in-memory mock data to verify computation correctness without mutating files.
"""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_quality import (
    compute_dataset_overview,
    compute_missing_values,
    compute_descriptive_statistics,
    check_longitudinal_integrity,
    analyze_cholesterol_completeness,
    discover_candidate_outcomes,
    run_data_quality_analysis,
)


class TestDataQualityModule(unittest.TestCase):
    """Test suite verifying data quality calculation accuracy and immutability."""

    def setUp(self):
        """Create a tiny, deterministic test dataframe."""
        self.mock_df = pd.DataFrame({
            "RANDID": [101, 101, 102, 103],
            "PERIOD": [1, 2, 1, 1],
            "TIME": [0, 2190, 0, 0],
            "AGE": [45, 51, 60, 39],
            "TOTCHOL": [200.0, 220.0, np.nan, 180.0],
            "HDLC": [np.nan, 50.0, np.nan, 40.0],
            "LDLC": [np.nan, 130.0, np.nan, 110.0],
            "CVD": [0, 1, 0, 0],
        })

    def test_missing_value_calculation(self):
        """Verify accurate missing-value counts and percentage computation."""
        missing_df = compute_missing_values(self.mock_df)
        
        # Check LDLC has 2 missing out of 4 (50.0%)
        ldlc_row = missing_df[missing_df["Column"] == "LDLC"].iloc[0]
        self.assertEqual(ldlc_row["Missing_Count"], 2)
        self.assertEqual(ldlc_row["Missing_Percentage"], 50.0)
        self.assertEqual(ldlc_row["Non_Missing_Count"], 2)

        # Check AGE has 0 missing (0.0%)
        age_row = missing_df[missing_df["Column"] == "AGE"].iloc[0]
        self.assertEqual(age_row["Missing_Count"], 0)
        self.assertEqual(age_row["Missing_Percentage"], 0.0)

    def test_duplicate_row_detection(self):
        """Verify that exact duplicate rows are identified."""
        df_with_dups = pd.concat([self.mock_df, self.mock_df.iloc[[0]]], ignore_index=True)
        overview = compute_dataset_overview(df_with_dups)
        dup_val = overview.loc[overview["Metric"] == "Exact Duplicate Rows", "Value"].values[0]
        self.assertEqual(dup_val, 1)

    def test_cholesterol_completeness_logic(self):
        """Verify complete-case detection across TOTCHOL, HDLC, and LDLC."""
        chol_info = analyze_cholesterol_completeness(self.mock_df)
        
        # Only row 1 (index 1: 220, 50, 130) and row 3 (index 3: 180, 40, 110) are complete
        self.assertEqual(chol_info["complete_chol_cases"], 2)
        self.assertEqual(chol_info["complete_chol_percentage"], 50.0)
        self.assertFalse(chol_info["has_triglycerides"])

    def test_longitudinal_integrity_detection(self):
        """Verify recognition of longitudinal participant exams."""
        long_info = check_longitudinal_integrity(self.mock_df)
        self.assertTrue(long_info["has_randid"])
        self.assertEqual(long_info["unique_participants"], 3)  # 101, 102, 103
        self.assertEqual(long_info["exam_distribution"][1], 2)  # 2 participants with 1 exam (102, 103)
        self.assertEqual(long_info["exam_distribution"][2], 1)  # 1 participant with 2 exams (101)

    def test_missing_input_path_handling(self):
        """Verify that FileNotFoundError is raised when input file does not exist."""
        with self.assertRaises(FileNotFoundError):
            run_data_quality_analysis(input_path="data/non_existent_file.csv")

    def test_input_file_immutability(self):
        """Verify that running data quality analysis leaves the source CSV file unmodified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            out_dir = Path(tmpdir) / "reports"
            
            # Write initial test CSV
            self.mock_df.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as f:
                initial_hash = hashlib.sha256(f.read()).hexdigest()

            # Execute analysis
            result = run_data_quality_analysis(input_path=str(csv_path), output_dir=str(out_dir))
            
            # Verify result flags
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["input_file_unchanged"])

            # Verify file hash directly
            with open(csv_path, "rb") as f:
                final_hash = hashlib.sha256(f.read()).hexdigest()

            self.assertEqual(initial_hash, final_hash)
            
            # Verify generated reports exist
            for r_path in result["reports"].values():
                self.assertTrue(Path(r_path).is_file())


if __name__ == "__main__":
    unittest.main()
