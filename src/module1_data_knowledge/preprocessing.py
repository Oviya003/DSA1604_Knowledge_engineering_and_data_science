"""
ExplainAI - Module 1: Data Preprocessing and Feature Preparation
Provides feature-target separation, anti-leakage enforcement, descriptive attribute
isolation, and modular, unfitted scikit-learn ColumnTransformer pipelines.

TEMPORAL CONTEXT & ASSUMPTION NOTICE:
-------------------------------------
The UCI Student Performance dataset is retrospective and does not provide
granular feature-collection timestamps. Model A is conceptualized as an
early-stage intake screening scenario under the operational assumption that
demographics, historical failures, and intake survey habits are acquired during
course enrollment. This assumption is documented as a methodological modeling
framework for early risk identification, rather than an empirical property
of the raw UCI survey archive.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import (
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    DEMOGRAPHIC_CATEGORICAL,
    DEMOGRAPHIC_NUMERICAL,
    EARLY_ASSESSMENT_FEATURES,
    DESCRIPTIVE_FEATURES,
    STRICTLY_EXCLUDED_FROM_FEATURES,
    TARGET_COL,
    BINARY_TARGET_COL,
    PASSING_THRESHOLD,
)


class DataPreprocessor:
    """
    Coordinates feature selection, target derivation, and preprocessing pipeline
    assembly without premature data fitting.
    """

    def __init__(self):
        self.categorical_features = list(DEMOGRAPHIC_CATEGORICAL)
        self.numerical_features_a = list(DEMOGRAPHIC_NUMERICAL)
        self.numerical_features_b = list(DEMOGRAPHIC_NUMERICAL) + list(EARLY_ASSESSMENT_FEATURES)

    def extract_features_and_target(
        self,
        df: pd.DataFrame,
        model_experiment: str = "A",
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Extracts feature matrix X, binary ground-truth target y, and descriptive metadata.

        Strict Anti-Leakage Enforcements:
        1. Model A extracts exactly 29 features.
        2. Model B extracts exactly 30 features (Model A + G1).
        3. 'absences' is strictly excluded from X (retained only in descriptive metadata).
        4. 'G2', 'G3', and 'is_at_risk' are strictly excluded from X.
        5. Target y is derived from G3 < 10 (1 = At-Risk, 0 = Passing).
        """
        experiment = model_experiment.upper()
        if experiment == "A":
            feature_cols = list(FEATURES_MODEL_A)
            expected_count = 29
        elif experiment == "B":
            feature_cols = list(FEATURES_MODEL_B)
            expected_count = 30
        else:
            raise ValueError(f"Unknown experiment '{model_experiment}'. Must be 'A' or 'B'.")

        # Validate that all requested feature columns exist in the DataFrame
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Input DataFrame is missing required features: {missing_features}")

        if TARGET_COL not in df.columns:
            raise ValueError(f"Input DataFrame is missing target column '{TARGET_COL}'")

        # Extract feature matrix X (creates an independent copy)
        X = df[feature_cols].copy()

        # Derive binary target y: 1 if G3 < 10 else 0
        y = (df[TARGET_COL] < PASSING_THRESHOLD).astype(int).rename(BINARY_TARGET_COL)

        # Extract descriptive metadata (for profiling, EDA, and audit, without entering models)
        meta_cols = [c for c in (DESCRIPTIVE_FEATURES + ["G1", "G2", "G3"]) if c in df.columns]
        meta_df = df[meta_cols].copy()
        meta_df[BINARY_TARGET_COL] = y.values

        # Strict assertion checks against data leakage
        assert len(X.columns) == expected_count, (
            f"Feature count mismatch for Model {experiment}: expected {expected_count}, got {len(X.columns)}"
        )
        for excluded in STRICTLY_EXCLUDED_FROM_FEATURES:
            assert excluded not in X.columns, f"CRITICAL LEAKAGE: '{excluded}' detected in feature matrix X!"

        return X, y, meta_df

    def build_column_transformer(self, model_experiment: str = "A") -> ColumnTransformer:
        """
        Constructs an UNIFITTED scikit-learn ColumnTransformer.

        Methodological Rule:
        The pipeline is returned un-fitted. Fitting is strictly deferred to the
        training split in Module 2 to prevent any cross-fold information leakage.

        Pipelines:
        - Numerical features: Median Imputation -> StandardScaler
        - Categorical features: Most-Frequent Imputation -> OneHotEncoder(handle_unknown='ignore')
        """
        experiment = model_experiment.upper()
        if experiment == "A":
            cat_cols = self.categorical_features
            num_cols = self.numerical_features_a
        elif experiment == "B":
            cat_cols = self.categorical_features
            num_cols = self.numerical_features_b
        else:
            raise ValueError(f"Unknown experiment '{model_experiment}'. Must be 'A' or 'B'.")

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, num_cols),
                ("cat", cat_pipeline, cat_cols),
            ],
            remainder="drop",
        )

        return preprocessor

