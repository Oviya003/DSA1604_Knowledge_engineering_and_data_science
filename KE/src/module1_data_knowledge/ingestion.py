"""
ExplainAI - Module 1: Data Ingestion and Integrity Validation
Provides reusable dataset loading, schema verification, data type enforcement,
domain integrity checking, duplicate detection, and structured validation reporting.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.config import (
    MATH_DATASET_PATH,
    POR_DATASET_PATH,
    TARGET_COL,
    BINARY_TARGET_COL,
    PASSING_THRESHOLD,
)

# Expected schema definitions for UCI Student Performance datasets
EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2", "G3"
]

EXPECTED_ROW_COUNTS = {
    "Mathematics": 395,
    "Portuguese": 649,
}

VALID_DOMAINS = {
    "school": {"GP", "MS"},
    "sex": {"F", "M"},
    "address": {"U", "R"},
    "famsize": {"LE3", "GT3"},
    "Pstatus": {"T", "A"},
    "Mjob": {"teacher", "health", "services", "at_home", "other"},
    "Fjob": {"teacher", "health", "services", "at_home", "other"},
    "reason": {"home", "reputation", "course", "other"},
    "guardian": {"mother", "father", "other"},
    "schoolsup": {"yes", "no"},
    "famsup": {"yes", "no"},
    "paid": {"yes", "no"},
    "activities": {"yes", "no"},
    "nursery": {"yes", "no"},
    "higher": {"yes", "no"},
    "internet": {"yes", "no"},
    "romantic": {"yes", "no"},
}

NUMERICAL_RANGES = {
    "age": (15, 22),
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 100),
    "G1": (0, 20),
    "G2": (0, 20),
    "G3": (0, 20),
}


class DatasetIngestion:
    """Handles dataset loading, structural inspection, and integrity validation."""

    def __init__(
        self,
        math_path: Optional[Path] = None,
        por_path: Optional[Path] = None,
    ):
        self.math_path = math_path or MATH_DATASET_PATH
        self.por_path = por_path or POR_DATASET_PATH

    def load_raw_dataset(self, subject: str = "Mathematics") -> pd.DataFrame:
        """
        Loads a raw semicolon-delimited dataset without in-place modification.
        Validates file existence before reading.
        """
        target_path = self.math_path if subject == "Mathematics" else self.por_path
        if not target_path.exists():
            raise FileNotFoundError(f"Raw dataset file not found at {target_path}")

        # Read semicolon-delimited CSV into DataFrame
        df = pd.read_csv(target_path, sep=";")
        return df

    def validate_dataset(self, df: pd.DataFrame, subject: str = "Mathematics") -> Dict[str, Any]:
        """
        Executes a comprehensive validation suite against a loaded dataset:
        1. Expected column presence and total column count.
        2. Expected row count verification.
        3. Missing value analysis (per-column and total).
        4. Exact duplicate row detection across all attributes.
        5. Categorical domain consistency checks.
        6. Numerical range and bound checks.
        """
        issues: List[str] = []
        warnings: List[str] = []

        # 1. Column presence check
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
        if missing_cols:
            issues.append(f"Missing expected columns: {missing_cols}")
        if extra_cols:
            warnings.append(f"Unexpected additional columns: {extra_cols}")

        # 2. Row count verification
        expected_rows = EXPECTED_ROW_COUNTS.get(subject)
        actual_rows = len(df)
        if expected_rows is not None and actual_rows != expected_rows:
            warnings.append(f"Row count mismatch for {subject}: expected {expected_rows}, got {actual_rows}")

        # 3. Missing values analysis
        null_counts = df.isnull().sum()
        total_nulls = int(null_counts.sum())
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if total_nulls > 0:
            issues.append(f"Found {total_nulls} missing values in columns: {cols_with_nulls}")

        # 4. Exact duplicate rows
        exact_dups = int(df.duplicated().sum())
        if exact_dups > 0:
            warnings.append(f"Found {exact_dups} exact duplicate rows across all 33 attributes.")

        # 5. Categorical domain validation
        for col, allowed in VALID_DOMAINS.items():
            if col in df.columns:
                invalid_vals = set(df[col].unique()) - allowed
                if invalid_vals:
                    issues.append(f"Column '{col}' has invalid categories: {invalid_vals} (allowed: {allowed})")

        # 6. Numerical range validation
        for col, (low, high) in NUMERICAL_RANGES.items():
            if col in df.columns:
                out_of_bounds = df[(df[col] < low) | (df[col] > high)]
                if len(out_of_bounds) > 0:
                    issues.append(
                        f"Column '{col}' has {len(out_of_bounds)} values outside valid range [{low}, {high}]."
                    )

        status = "PASSED" if len(issues) == 0 else "FAILED"

        return {
            "subject": subject,
            "status": status,
            "total_rows": actual_rows,
            "total_columns": len(df.columns),
            "total_missing_values": total_nulls,
            "columns_with_missing": cols_with_nulls,
            "exact_duplicate_rows": exact_dups,
            "issues": issues,
            "warnings": warnings,
        }

    def load_and_validate_subject(self, subject: str = "Mathematics") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads and validates a single subject dataset, returning the DataFrame and report."""
        df = self.load_raw_dataset(subject)
        report = self.validate_dataset(df, subject)
        return df, report

    def load_all_subjects(self) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
        """Loads and validates both Mathematics and Portuguese datasets."""
        results = {}
        for subject in ["Mathematics", "Portuguese"]:
            df, report = self.load_and_validate_subject(subject)
            results[subject] = (df, report)
        return results

