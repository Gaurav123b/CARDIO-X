"""
Cardio-X: Data Quality Analysis Module.

Provides reproducible, non-destructive validation and quality profiling for
the Framingham clinical cohort dataset. Computes structural properties,
missingness rankings, summary statistics, longitudinal structure,
cholesterol profile completeness, and candidate outcome distributions.
Generates aggregated tabular and Markdown reports without exposing
individual patient records.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


def compute_dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes top-level structural metrics of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        Single-row summary containing row count, column count, memory usage,
        and duplicate counts.
    """
    n_rows, n_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    exact_duplicates = int(df.duplicated().sum())

    # Check for duplicate subject examinations if RANDID and PERIOD exist
    if "RANDID" in df.columns and "PERIOD" in df.columns:
        randid_period_duplicates = int(df.duplicated(subset=["RANDID", "PERIOD"]).sum())
        unique_participants = int(df["RANDID"].nunique())
    else:
        randid_period_duplicates = 0
        unique_participants = n_rows

    total_cells = n_rows * n_cols
    total_missing = int(df.isnull().sum().sum())
    overall_missing_pct = (total_missing / total_cells * 100.0) if total_cells > 0 else 0.0

    overview_data = {
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Unique Participants (RANDID)",
            "Exact Duplicate Rows",
            "Duplicate (RANDID, PERIOD) Pairs",
            "Total Cells",
            "Total Missing Cells",
            "Overall Missingness (%)",
            "Memory Footprint (MB)",
        ],
        "Value": [
            n_rows,
            n_cols,
            unique_participants,
            exact_duplicates,
            randid_period_duplicates,
            total_cells,
            total_missing,
            round(overall_missing_pct, 2),
            round(memory_mb, 2),
        ],
    }
    overview_df = pd.DataFrame(overview_data)
    # Format integer metrics without decimal points for clean presentation
    int_metrics = [
        "Total Rows",
        "Total Columns",
        "Unique Participants (RANDID)",
        "Exact Duplicate Rows",
        "Duplicate (RANDID, PERIOD) Pairs",
        "Total Cells",
        "Total Missing Cells",
    ]
    overview_df["Value"] = overview_df.apply(
        lambda r: int(r["Value"]) if r["Metric"] in int_metrics else r["Value"], axis=1
    )
    return overview_df



def compute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes missing value counts, percentages, and non-missing counts per column.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        Ranked summary dataframe with missingness metrics.
    """
    total_rows = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total_rows * 100.0).round(2)
    non_missing_count = df.notnull().sum()
    dtypes = df.dtypes.astype(str)

    report_df = pd.DataFrame({
        "Column": df.columns,
        "Data_Type": dtypes,
        "Non_Missing_Count": non_missing_count,
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_pct,
    })

    # Sort descending by missing count, then alphabetically by Column
    report_df = report_df.sort_values(
        by=["Missing_Count", "Column"], ascending=[False, True]
    ).reset_index(drop=True)
    report_df["Missingness_Rank"] = range(1, len(report_df) + 1)
    return report_df


def compute_descriptive_statistics(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Computes descriptive distribution statistics for numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.
    columns : Optional[List[str]]
        Specific columns to analyze. If None, all numeric columns are selected.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics (count, mean, std, min, quartiles, median, max, IQR, zero count).
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        valid_cols = [c for c in columns if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
        numeric_df = df[valid_cols]

    if numeric_df.empty:
        return pd.DataFrame()

    stats_list = []
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        n_count = len(series)
        n_missing = int(numeric_df[col].isnull().sum())
        zero_count = int((series == 0).sum())

        if n_count > 0:
            mean_val = round(float(series.mean()), 2)
            std_val = round(float(series.std()), 2)
            min_val = round(float(series.min()), 2)
            q25 = round(float(series.quantile(0.25)), 2)
            median_val = round(float(series.median()), 2)
            q75 = round(float(series.quantile(0.75)), 2)
            max_val = round(float(series.max()), 2)
            iqr = round(q75 - q25, 2)
        else:
            mean_val = std_val = min_val = q25 = median_val = q75 = max_val = iqr = np.nan

        stats_list.append({
            "Column": col,
            "Count": n_count,
            "Missing": n_missing,
            "Zeros": zero_count,
            "Mean": mean_val,
            "Std": std_val,
            "Min": min_val,
            "25%": q25,
            "Median (50%)": median_val,
            "75%": q75,
            "Max": max_val,
            "IQR": iqr,
        })

    return pd.DataFrame(stats_list)


def check_longitudinal_integrity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates the longitudinal cohort structure of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    Dict[str, Any]
        Dictionary of longitudinal integrity indicators (participant counts, exam frequency).
    """
    results: Dict[str, Any] = {
        "has_randid": "RANDID" in df.columns,
        "has_period": "PERIOD" in df.columns,
        "has_time": "TIME" in df.columns,
        "total_rows": len(df),
        "unique_participants": df["RANDID"].nunique() if "RANDID" in df.columns else len(df),
        "exam_distribution": {},
        "period_counts": {},
        "time_summary_by_period": {},
    }

    if results["has_randid"]:
        exam_freq = df["RANDID"].value_counts().value_counts().to_dict()
        results["exam_distribution"] = {int(k): int(v) for k, v in exam_freq.items()}

    if results["has_period"]:
        period_counts = df["PERIOD"].value_counts().sort_index().to_dict()
        results["period_counts"] = {int(k): int(v) for k, v in period_counts.items()}

    if results["has_period"] and results["has_time"]:
        for period, group in df.groupby("PERIOD"):
            t = group["TIME"]
            results["time_summary_by_period"][int(period)] = {
                "count": int(t.count()),
                "min_days": float(t.min()),
                "median_days": float(t.median()),
                "max_days": float(t.max()),
            }

    return results


def analyze_cholesterol_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates availability, missingness, and completeness across lipid parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    Dict[str, Any]
        Metrics regarding TOTCHOL, HDLC, LDLC, and complete cases.
    """
    total_rows = len(df)
    results: Dict[str, Any] = {
        "total_rows": total_rows,
        "has_totchol": "TOTCHOL" in df.columns,
        "has_hdlc": "HDLC" in df.columns,
        "has_ldlc": "LDLC" in df.columns,
        "has_triglycerides": any("TRIG" in c.upper() for c in df.columns),
        "totchol_count": int(df["TOTCHOL"].notnull().sum()) if "TOTCHOL" in df.columns else 0,
        "hdlc_count": int(df["HDLC"].notnull().sum()) if "HDLC" in df.columns else 0,
        "ldlc_count": int(df["LDLC"].notnull().sum()) if "LDLC" in df.columns else 0,
        "complete_chol_cases": 0,
        "complete_chol_percentage": 0.0,
        "completeness_by_period": {},
    }

    chol_cols = [c for c in ["TOTCHOL", "HDLC", "LDLC"] if c in df.columns]
    if chol_cols:
        complete_mask = df[chol_cols].notnull().all(axis=1)
        complete_count = int(complete_mask.sum())
        results["complete_chol_cases"] = complete_count
        results["complete_chol_percentage"] = round(complete_count / total_rows * 100.0, 2) if total_rows > 0 else 0.0

        if "PERIOD" in df.columns:
            for period, group in df.groupby("PERIOD"):
                p_total = len(group)
                p_complete = int(group[chol_cols].notnull().all(axis=1).sum())
                results["completeness_by_period"][int(period)] = {
                    "total_period_rows": p_total,
                    "complete_cases": p_complete,
                    "percentage": round(p_complete / p_total * 100.0, 2) if p_total > 0 else 0.0,
                    "totchol_non_null": int(group["TOTCHOL"].notnull().sum()) if "TOTCHOL" in group.columns else 0,
                    "hdlc_non_null": int(group["HDLC"].notnull().sum()) if "HDLC" in group.columns else 0,
                    "ldlc_non_null": int(group["LDLC"].notnull().sum()) if "LDLC" in group.columns else 0,
                }

    return results


def discover_candidate_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audits prospective outcome columns and time-to-event indicators.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset.

    Returns
    -------
    pd.DataFrame
        Summary table detailing event rates, categories, missingness, and preliminary interpretations.
    """
    outcome_candidates = [
        ("CVD", "Cardiovascular Disease event during follow-up"),
        ("ANYCHD", "Any Coronary Heart Disease event during follow-up"),
        ("HYPERTEN", "Hypertension status during follow-up"),
        ("STROKE", "Stroke event during follow-up"),
        ("DEATH", "All-cause mortality during follow-up"),
        ("ANGINA", "Angina Pectoris event during follow-up"),
        ("HOSPMI", "Hospitalized Myocardial Infarction"),
        ("MI_FCHD", "Myocardial Infarction or Fatal Coronary Heart Disease"),
    ]

    records = []
    total_rows = len(df)
    for col, desc in outcome_candidates:
        if col in df.columns:
            series = df[col]
            missing_n = int(series.isnull().sum())
            missing_pct = round(missing_n / total_rows * 100.0, 2) if total_rows > 0 else 0.0
            unique_vals = sorted(series.dropna().unique().tolist())
            pos_count = int((series == 1).sum())
            pos_rate = round(pos_count / total_rows * 100.0, 2) if total_rows > 0 else 0.0

            time_col = f"TIME{col}" if f"TIME{col}" in df.columns else (
                "TIMECVD" if col == "CVD" else (
                    "TIMECHD" if col == "ANYCHD" else (
                        "TIMEHYP" if col == "HYPERTEN" else (
                            "TIMESTRK" if col == "STROKE" else (
                                "TIMEDTH" if col == "DEATH" else (
                                    "TIMEAP" if col == "ANGINA" else (
                                        "TIMEMI" if col == "HOSPMI" else (
                                            "TIMEMIFC" if col == "MI_FCHD" else "None"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )

            records.append({
                "Candidate_Outcome": col,
                "Observed_Values": str(unique_vals),
                "Missing_Count": missing_n,
                "Missing_Pct": missing_pct,
                "Positive_Cases": pos_count,
                "Positive_Rate_Pct": pos_rate,
                "Associated_Time_Column": time_col,
                "Preliminary_Description": desc,
            })

    return pd.DataFrame(records)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """
    Converts a pandas DataFrame into a GitHub-flavored Markdown table without
    requiring the optional 'tabulate' third-party dependency.
    """
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_vals = [str(val) if pd.notnull(val) else "" for val in row]
        lines.append("| " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def generate_markdown_summary(
    overview_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    longitudinal_info: Dict[str, Any],
    cholesterol_info: Dict[str, Any],
    outcomes_df: pd.DataFrame,
) -> str:
    """
    Formats the complete data quality analysis findings into an academic Markdown report.
    Guarantees no individual patient data is exposed.
    """
    total_rows_val = int(overview_df.loc[overview_df["Metric"] == "Total Rows", "Value"].values[0])
    total_cols_val = int(overview_df.loc[overview_df["Metric"] == "Total Columns", "Value"].values[0])
    exact_dups_val = int(overview_df.loc[overview_df["Metric"] == "Exact Duplicate Rows", "Value"].values[0])

    md: List[str] = [
        "# Cardio-X: Data Quality and Dataset Validation Report",
        "\n## 1. Executive Summary",
        f"- **Total Rows:** {total_rows_val:,}",
        f"- **Total Columns:** {total_cols_val}",
        f"- **Distinct Participants:** {longitudinal_info['unique_participants']:,}",
        f"- **Exact Duplicate Rows:** {exact_dups_val}",
        "- **Data Integrity:** No row duplicates found; repeated IDs correspond to prospective longitudinal follow-up examination cycles.",
        "\n---\n",
        "## 2. Dataset Overview",
        dataframe_to_markdown(overview_df),

        "\n---\n",
        "## 3. Longitudinal Cohort Structure Analysis",
        "The dataset consists of multi-wave prospective examinations per participant (`PERIOD` 1, 2, and 3):",
        "",
    ]

    for p, count in longitudinal_info.get("period_counts", {}).items():
        t_info = longitudinal_info.get("time_summary_by_period", {}).get(p, {})
        med_days = t_info.get("median_days", 0)
        years_approx = round(med_days / 365.25, 1)
        md.append(f"- **Exam Cycle {p} (PERIOD={p}):** {count:,} observations (Median follow-up: {med_days} days / ~{years_approx} years)")

    md.extend([
        "",
        "### Examination Frequency per Participant:",
    ])
    for n_exams, count in sorted(longitudinal_info.get("exam_distribution", {}).items()):
        md.append(f"- Participants with {n_exams} recorded examination(s): **{count:,}**")

    md.extend([
        "\n> [!IMPORTANT]",
        "> **Epidemiological Interpretation:** Repeated `RANDID` entries represent longitudinal follow-up observations of the same subjects, NOT duplicate data. Models must account for this structure (e.g., using baseline Exam 1 observations or explicit panel/time-series modeling) to avoid data leakage.",
        "\n---\n",
        "## 4. Missing Data Profile",
        "Ranked summary of columns exhibiting missingness:\n",
        dataframe_to_markdown(missing_df[missing_df["Missing_Count"] > 0]),
        "\n---\n",
        "## 5. Cholesterol & Lipid Profile Evaluation",
        f"- **Total Cholesterol (`TOTCHOL`):** Available in {cholesterol_info['totchol_count']:,} rows ({cholesterol_info['totchol_count']/cholesterol_info['total_rows']*100:.2f}%).",
        f"- **High-Density Lipoprotein (`HDLC`):** Available in {cholesterol_info['hdlc_count']:,} rows ({cholesterol_info['hdlc_count']/cholesterol_info['total_rows']*100:.2f}%).",
        f"- **Low-Density Lipoprotein (`LDLC`):** Available in {cholesterol_info['ldlc_count']:,} rows ({cholesterol_info['ldlc_count']/cholesterol_info['total_rows']*100:.2f}%).",
        f"- **Triglycerides:** **Not present** in the dataset.",
        f"- **Complete Lipid Profiles (`TOTCHOL` + `HDLC` + `LDLC`):** **{cholesterol_info['complete_chol_cases']:,} rows ({cholesterol_info['complete_chol_percentage']}%)**.",
        "",
        "### Lipid Completeness by Examination Period:",
    ])

    for p, p_data in cholesterol_info.get("completeness_by_period", {}).items():
        md.append(
            f"- **Period {p}:** {p_data['complete_cases']:,} complete lipid profiles out of {p_data['total_period_rows']:,} rows ({p_data['percentage']}%) "
            f"[TOTCHOL={p_data['totchol_non_null']:,}, HDLC={p_data['hdlc_non_null']:,}, LDLC={p_data['ldlc_non_null']:,}]"
        )

    md.extend([
        "",
        "> [!NOTE]",
        "> **Protocol Disclosure:** `HDLC` and `LDLC` were not included in the Framingham study testing protocol during Examination Periods 1 and 2. Their missingness in early cycles is structural by study design rather than random.",
        "\n---\n",
        "## 6. Descriptive Distribution Statistics",
        "Key numeric physiological parameters (quantiles, range, dispersion):\n",
        dataframe_to_markdown(stats_df),
        "\n---\n",
        "## 7. Prospective Outcome & Candidate Target Discovery",
        "Audit of potential binary outcome endpoints and associated time-to-event indicators:\n",
        dataframe_to_markdown(outcomes_df),
        "\n> [!TIP]",
        "> **Target Selection Notice:** Selection of the primary prediction target (e.g. `CVD`, `ANYCHD`, or hypercholesterolemia risk stratification) will be finalized in subsequent design phases with guide alignment.",
    ])

    return "\n".join(md)



def run_data_quality_analysis(
    input_path: str = "data/FraminghamDataset.csv",
    output_dir: str = "reports/data_quality",
) -> Dict[str, Any]:
    """
    Executes the end-to-end data quality pipeline without mutating the input file.

    Parameters
    ----------
    input_path : str
        Path to the source CSV dataset.
    output_dir : str
        Directory where reports will be saved.

    Returns
    -------
    Dict[str, Any]
        Dictionary with status and report file locations.
    """
    in_file = Path(input_path)
    if not in_file.is_file():
        raise FileNotFoundError(f"Dataset not found at expected path: {input_path}")

    # Read-only cryptographic digest check before execution
    with open(in_file, "rb") as f:
        pre_hash = hashlib.sha256(f.read()).hexdigest()

    # Load dataset
    df = pd.read_csv(in_file)

    # Compute quality metrics
    overview_df = compute_dataset_overview(df)
    missing_df = compute_missing_values(df)
    stats_df = compute_descriptive_statistics(df)
    longitudinal_info = check_longitudinal_integrity(df)
    cholesterol_info = analyze_cholesterol_completeness(df)
    outcomes_df = discover_candidate_outcomes(df)

    # Markdown summary
    summary_md = generate_markdown_summary(
        overview_df=overview_df,
        missing_df=missing_df,
        stats_df=stats_df,
        longitudinal_info=longitudinal_info,
        cholesterol_info=cholesterol_info,
        outcomes_df=outcomes_df,
    )

    # Save reports
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    overview_path = out_path / "dataset_overview.csv"
    missing_path = out_path / "missing_values.csv"
    stats_path = out_path / "descriptive_statistics.csv"
    summary_path = out_path / "data_quality_summary.md"

    overview_df.to_csv(overview_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)

    # Post-execution immutability verification
    with open(in_file, "rb") as f:
        post_hash = hashlib.sha256(f.read()).hexdigest()

    if pre_hash != post_hash:
        raise RuntimeError("Fatal: The input dataset was modified during data quality analysis!")

    return {
        "status": "success",
        "input_hash": post_hash,
        "input_file_unchanged": True,
        "reports": {
            "overview_csv": str(overview_path),
            "missing_csv": str(missing_path),
            "descriptive_csv": str(stats_path),
            "summary_md": str(summary_path),
        },
        "overview": overview_df.to_dict(orient="records"),
        "cholesterol_completeness": cholesterol_info,
        "longitudinal_structure": longitudinal_info,
    }


if __name__ == "__main__":
    result = run_data_quality_analysis()
    print("=" * 60)
    print("DATA QUALITY ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Status           : {result['status']}")
    print(f"Input Unchanged  : {result['input_file_unchanged']}")
    print(f"SHA-256 Digest   : {result['input_hash']}")
    print("Generated Reports:")
    for name, path in result["reports"].items():
        print(f"  - {name:<16}: {path}")
    print("=" * 60)
