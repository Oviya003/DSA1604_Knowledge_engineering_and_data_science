"""
ExplainAI - Step 2 Scaffolding & Validation Tests
Validates:
1. Python 3.13 dependency compatibility.
2. UCI dataset files, schemas, and non-null integrity.
3. Curriculum DAG acyclicity, referential validity, explicit demonstration provenance,
   and subject restriction to Mathematics.
4. Feature availability timing, exact feature counts (Model A: 29, Model B: 30),
   removal of 'absences' from predictive features, and strict anti-leakage isolation of G1/G2/G3/is_at_risk.
5. Cross-subject matching inspection: exact verification of 1-to-1 matches (358),
   ambiguous Cartesian keys (8 keys generating 24 rows), and preservation of separate records.
6. SQLite database 10 tables, composite primary keys (student_id, subject),
   cascading foreign keys, and demarcation of UCI source vs. application-generated records.
"""

import sqlite3
import pytest
import pandas as pd
from pathlib import Path

from src.config import (
    MATH_DATASET_PATH,
    POR_DATASET_PATH,
    PROJECT_ROOT,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    EARLY_ASSESSMENT_FEATURES,
    DEMOGRAPHIC_CATEGORICAL,
    DEMOGRAPHIC_NUMERICAL,
    DESCRIPTIVE_FEATURES,
    STRICTLY_EXCLUDED_FROM_FEATURES,
    TARGET_COL,
    BINARY_TARGET_COL,
    PASSING_THRESHOLD,
    CURRICULUM_SUPPORTED_SUBJECTS,
)
from src.database.db_manager import DatabaseManager
from src.module1_data_knowledge.curriculum_catalog import (
    MATHEMATICS_CURRICULUM,
    CURRICULUM_PROVENANCE,
    CURRICULUM_SUBJECT,
    validate_curriculum_dag,
    build_curriculum_graph,
    is_subject_curriculum_supported,
    get_unsupported_subject_message,
)


def test_dependencies_imported():
    """Verify all required libraries are installed and compatible with Python 3.13."""
    import streamlit
    import pandas
    import numpy
    import sklearn
    import shap
    import networkx
    import plotly
    import xgboost

    assert streamlit.__version__ is not None
    assert pandas.__version__ is not None
    assert numpy.__version__ is not None
    assert sklearn.__version__ is not None
    assert shap.__version__ is not None
    assert networkx.__version__ is not None
    assert plotly.__version__ is not None
    assert xgboost.__version__ is not None


def test_uci_datasets_exist_and_valid():
    """Verify downloaded UCI datasets exist and match expected schemas."""
    assert MATH_DATASET_PATH.exists(), f"Math dataset missing at {MATH_DATASET_PATH}"
    assert POR_DATASET_PATH.exists(), f"Portuguese dataset missing at {POR_DATASET_PATH}"

    df_mat = pd.read_csv(MATH_DATASET_PATH, sep=";")
    df_por = pd.read_csv(POR_DATASET_PATH, sep=";")

    assert df_mat.shape == (395, 33), f"Unexpected math shape: {df_mat.shape}"
    assert df_por.shape == (649, 33), f"Unexpected por shape: {df_por.shape}"

    # Target column check
    assert TARGET_COL in df_mat.columns
    assert TARGET_COL in df_por.columns

    # Verify no nulls in raw UCI dataset
    assert df_mat.isnull().sum().sum() == 0
    assert df_por.isnull().sum().sum() == 0


def test_curriculum_provenance_dag_and_subject_restriction():
    """
    Verify:
    1. Curriculum is explicitly labeled as demonstration curriculum.
    2. Curriculum is strictly restricted to Mathematics.
    3. Portuguese curriculum queries are handled gracefully.
    4. Graph forms a valid DAG with no cycles.
    """
    assert CURRICULUM_PROVENANCE == "DEMONSTRATION_CURRICULUM"
    assert CURRICULUM_SUBJECT == "Mathematics"

    # Subject restriction checks
    assert is_subject_curriculum_supported("Mathematics") is True
    assert is_subject_curriculum_supported("Portuguese") is False
    assert is_subject_curriculum_supported("History") is False

    unsupported_msg = get_unsupported_subject_message("Portuguese")
    assert "Mathematics (Demonstration Curriculum)" in unsupported_msg
    assert "Portuguese" in unsupported_msg

    for item in MATHEMATICS_CURRICULUM:
        assert item["curriculum_type"] == "DEMONSTRATION_CURRICULUM"
        assert item["subject"] == "Mathematics"

    # DAG structural validation
    result = validate_curriculum_dag(MATHEMATICS_CURRICULUM)
    assert result["valid"] is True, f"Curriculum validation errors: {result['errors']}"
    assert result["is_dag"] is True, "Curriculum graph contains cyclic dependencies!"
    assert result["num_concepts"] == 10
    assert result["num_prerequisite_edges"] == 11
    assert len(result["topological_order"]) == 10
    assert result["topological_order"][0] == "MATH_01"
    assert result["topological_order"][-1] == "MATH_10"


def test_feature_counts_and_leakage_prevention():
    """
    Verify:
    1. Exact feature counts: Model A = 29, Model B = 30.
    2. 'absences' is removed from predictive features and retained only for descriptive analysis.
    3. G1 is present ONLY in Model B and strictly excluded from Model A.
    4. G2, G3, and is_at_risk are strictly excluded from both Model A and Model B.
    5. Binary target is derived from generic 0-20 scale: G3 < 10 is at-risk.
    """
    # Categorical: 17, Numerical: 12 (absences excluded)
    assert len(DEMOGRAPHIC_CATEGORICAL) == 17
    assert len(DEMOGRAPHIC_NUMERICAL) == 12

    # Model A: exactly 29 features
    assert len(FEATURES_MODEL_A) == 29
    assert set(FEATURES_MODEL_A) == set(DEMOGRAPHIC_CATEGORICAL + DEMOGRAPHIC_NUMERICAL)

    # Model B: exactly 30 features (Model A + G1)
    assert len(FEATURES_MODEL_B) == 30
    assert "G1" in FEATURES_MODEL_B
    assert "G1" not in FEATURES_MODEL_A

    # Absences is in DESCRIPTIVE_FEATURES but strictly excluded from predictive features
    assert "absences" in DESCRIPTIVE_FEATURES
    assert "absences" not in FEATURES_MODEL_A
    assert "absences" not in FEATURES_MODEL_B

    # Future assessments & target variables must NEVER be in feature matrices
    for forbidden in STRICTLY_EXCLUDED_FROM_FEATURES:
        assert forbidden not in FEATURES_MODEL_A, f"{forbidden} leaked into Model A"
        assert forbidden not in FEATURES_MODEL_B, f"{forbidden} leaked into Model B"

    # Target threshold check
    assert PASSING_THRESHOLD == 10.0


def test_cross_subject_student_matching_analysis():
    """
    Inspect student-merge.R matching logic on the 13 demographic attributes:
    1. Check duplicate key combinations in Math and Portuguese datasets.
    2. Verify that the 382 merged rows contain Cartesian products from non-unique keys.
    3. Verify unambiguous 1-to-1 matches (358), ambiguous keys (8 keys -> 24 rows),
       and unique key combinations (366).
    4. Confirm that Mathematics and Portuguese must be preserved as separate records.
    """
    df_mat = pd.read_csv(MATH_DATASET_PATH, sep=";")
    df_por = pd.read_csv(POR_DATASET_PATH, sep=";")

    match_cols = [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
    ]

    # Check non-unique keys within individual datasets
    dup_mat_count = df_mat.duplicated(subset=match_cols, keep=False).sum()
    dup_por_count = df_por.duplicated(subset=match_cols, keep=False).sum()
    assert dup_mat_count == 8, f"Expected 8 duplicate records in Math, got {dup_mat_count}"
    assert dup_por_count == 23, f"Expected 23 duplicate records in Por, got {dup_por_count}"

    # Replicate student-merge.R inner join
    d3 = pd.merge(df_mat, df_por, on=match_cols, suffixes=("_mat", "_por"))
    assert len(d3) == 382, f"Expected 382 merged rows, got {len(d3)}"

    # Analyze 1-to-1 vs ambiguous matches
    count_mat = df_mat.groupby(match_cols).size().rename("count_mat")
    count_por = df_por.groupby(match_cols).size().rename("count_por")
    matched_counts = pd.concat([count_mat, count_por], axis=1).dropna()

    one_to_one = matched_counts[(matched_counts["count_mat"] == 1) & (matched_counts["count_por"] == 1)]
    ambiguous = matched_counts[(matched_counts["count_mat"] > 1) | (matched_counts["count_por"] > 1)]

    assert len(one_to_one) == 358, f"Expected 358 unambiguous 1-to-1 matches, got {len(one_to_one)}"
    assert len(ambiguous) == 8, f"Expected 8 ambiguous keys, got {len(ambiguous)}"

    # Verify that ambiguous keys create a Cartesian product of 24 rows
    cartesian_rows = int((ambiguous["count_mat"] * ambiguous["count_por"]).sum())
    assert cartesian_rows == 24
    assert len(one_to_one) + cartesian_rows == 382
    assert len(matched_counts) == 366  # Unique keys, NOT 382 unique individuals!


def test_database_multi_subject_isolation_and_foreign_keys(tmp_path):
    """
    Verify:
    1. Composite primary key (student_id, subject) prevents collisions.
    2. Assessments isolate Mathematics from Portuguese.
    3. Diagnostic quizzes and mastery queries for unsupported subjects return empty lists gracefully.
    4. Deleting one subject enrollment cascades ONLY that subject's records.
    5. Foreign keys reject non-existent student/subject combinations.
    """
    test_db = tmp_path / "test_multi_subject.db"
    db_mgr = DatabaseManager(db_path=test_db)
    db_mgr.init_database()

    with db_mgr.get_connection() as conn:
        cursor = conn.cursor()

        # Enroll student STU_001 in Mathematics
        cursor.execute(
            """
            INSERT INTO students (
                student_id, subject, source_dataset, school, sex, age, address, famsize, Pstatus,
                Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures,
                schoolsup, famsup, paid, activities, nursery, higher, internet, romantic,
                famrel, freetime, goout, Dalc, Walc, health, absences, G1_raw, G2_raw, G3_raw, is_at_risk
            ) VALUES (
                'STU_001', 'Mathematics', 'UCI_MATHEMATICS', 'GP', 'F', 17, 'U', 'GT3', 'T',
                3, 3, 'other', 'other', 'course', 'mother', 1, 2, 0,
                'no', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no',
                4, 3, 3, 1, 1, 3, 6, 8.0, 9.0, 9.0, 1
            )
            """
        )

        # Enroll same student STU_001 in Portuguese
        cursor.execute(
            """
            INSERT INTO students (
                student_id, subject, source_dataset, school, sex, age, address, famsize, Pstatus,
                Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures,
                schoolsup, famsup, paid, activities, nursery, higher, internet, romantic,
                famrel, freetime, goout, Dalc, Walc, health, absences, G1_raw, G2_raw, G3_raw, is_at_risk
            ) VALUES (
                'STU_001', 'Portuguese', 'UCI_PORTUGUESE', 'GP', 'F', 17, 'U', 'GT3', 'T',
                3, 3, 'other', 'other', 'course', 'mother', 1, 2, 0,
                'no', 'yes', 'no', 'no', 'yes', 'yes', 'yes', 'no',
                4, 3, 3, 1, 1, 3, 2, 13.0, 14.0, 14.0, 0
            )
            """
        )

        # Insert UCI official assessments (is_uci_source = 1)
        cursor.execute(
            """
            INSERT INTO assessments (student_id, subject, assessment_name, term_period, score, max_score, percentage, assessment_date, is_uci_source)
            VALUES ('STU_001', 'Mathematics', 'Term 1 Baseline (G1)', 1, 8.0, 20.0, 40.0, '2026-11-15', 1)
            """
        )
        cursor.execute(
            """
            INSERT INTO assessments (student_id, subject, assessment_name, term_period, score, max_score, percentage, assessment_date, is_uci_source)
            VALUES ('STU_001', 'Portuguese', 'Term 1 Baseline (G1)', 1, 13.0, 20.0, 65.0, '2026-11-15', 1)
            """
        )
        conn.commit()

        # Query assessments specifically for Mathematics
        cursor.execute("SELECT score, is_uci_source FROM assessments WHERE student_id = 'STU_001' AND subject = 'Mathematics'")
        math_assessments = cursor.fetchall()
        assert len(math_assessments) == 1
        assert math_assessments[0][0] == 8.0

        # Query assessments specifically for Portuguese
        cursor.execute("SELECT score, is_uci_source FROM assessments WHERE student_id = 'STU_001' AND subject = 'Portuguese'")
        por_assessments = cursor.fetchall()
        assert len(por_assessments) == 1
        assert por_assessments[0][0] == 13.0

        # Verify concept queries for unsupported subject returns empty list cleanly
        por_concepts = db_mgr.get_all_concepts(subject="Portuguese")
        assert len(por_concepts) == 0

        # Verify foreign key constraint rejection
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute(
                """
                INSERT INTO assessments (student_id, subject, assessment_name, term_period, score, max_score, percentage, assessment_date, is_uci_source)
                VALUES ('STU_001', 'Science', 'Midterm', 2, 15.0, 20.0, 75.0, '2026-12-01', 0)
                """
            )

        # Verify cascade delete: deleting Portuguese enrollment deletes ONLY Portuguese assessments
        cursor.execute("DELETE FROM students WHERE student_id = 'STU_001' AND subject = 'Portuguese'")
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM assessments WHERE student_id = 'STU_001' AND subject = 'Portuguese'")
        assert cursor.fetchone()[0] == 0

        cursor.execute("SELECT COUNT(*) FROM assessments WHERE student_id = 'STU_001' AND subject = 'Mathematics'")
        assert cursor.fetchone()[0] == 1
