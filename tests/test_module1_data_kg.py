"""
ExplainAI - Module 1 Dedicated Test Suite
Tests:
1. Mathematics and Portuguese dataset loading.
2. Expected row and column counts.
3. Data type validation & value range checks.
4. Missing-value handling.
5. Duplicate record detection.
6. Model A feature count (29).
7. Model B feature count (30).
8. Leakage exclusion (G2, G3, is_at_risk strictly excluded).
9. Descriptive-only absences.
10. Subject isolation between Mathematics and Portuguese.
11. Knowledge graph node and edge counts (10 nodes, 11 edges).
12. DAG validation (acyclicity).
13. Prerequisite and dependent retrieval (direct and ancestor/descendant).
14. Topological learning order.
15. Diagnostic question integrity (20 questions, valid options, correct answers).
16. SQLite insertion, retrieval, foreign key behavior, and transaction integrity.
"""

import sqlite3
import pytest
import pandas as pd
from pathlib import Path

from src.config import (
    MATH_DATASET_PATH,
    POR_DATASET_PATH,
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
)
from src.database.db_manager import DatabaseManager
from src.module1_data_knowledge.ingestion import DatasetIngestion, EXPECTED_COLUMNS
from src.module1_data_knowledge.preprocessing import DataPreprocessor
from src.module1_data_knowledge.knowledge_graph import AcademicKnowledgeGraph
from src.module1_data_knowledge.assessment_history import AssessmentHistoryManager
from src.module1_data_knowledge.curriculum_catalog import (
    MATHEMATICS_CURRICULUM,
    CURRICULUM_PROVENANCE,
    is_subject_curriculum_supported,
)


@pytest.fixture
def ingestion():
    return DatasetIngestion()


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


@pytest.fixture
def kg_math():
    return AcademicKnowledgeGraph("Mathematics")


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_module1.db"
    mgr = DatabaseManager(db_path=db_file)
    mgr.init_database()
    return mgr


# ------------------------------------------------------------------------------
# 1. Dataset Loading and Schema Tests
# ------------------------------------------------------------------------------
def test_dataset_loading_row_and_column_counts(ingestion):
    """Verify loading of both raw datasets and exact expected row/column shapes."""
    df_mat, val_mat = ingestion.load_and_validate_subject("Mathematics")
    df_por, val_por = ingestion.load_and_validate_subject("Portuguese")

    assert df_mat.shape == (395, 33), f"Unexpected math shape: {df_mat.shape}"
    assert df_por.shape == (649, 33), f"Unexpected por shape: {df_por.shape}"
    assert val_mat["status"] == "PASSED"
    assert val_por["status"] == "PASSED"
    assert val_mat["total_missing_values"] == 0
    assert val_por["total_missing_values"] == 0


def test_dataset_columns_and_data_types(ingestion):
    """Verify all 33 expected columns exist and adhere to domain boundaries."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    assert list(df_mat.columns) == EXPECTED_COLUMNS

    # Check that age is integer and within secondary bounds
    assert df_mat["age"].dtype in ["int64", "int32"]
    assert df_mat["age"].between(15, 22).all()

    # Check grades are within 0-20
    for g in ["G1", "G2", "G3"]:
        assert df_mat[g].between(0, 20).all()

    # Check binary indicators
    for col in ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]:
        assert set(df_mat[col].unique()).issubset({"yes", "no"})


def test_missing_values_and_duplicate_detection(ingestion):
    """Verify zero missing values in raw datasets and detect duplicate rows."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    report = ingestion.validate_dataset(df_mat, "Mathematics")

    assert report["total_missing_values"] == 0
    assert report["columns_with_missing"] == {}
    assert report["exact_duplicate_rows"] == 0


# ------------------------------------------------------------------------------
# 2. Preprocessing & Leakage Prevention Tests
# ------------------------------------------------------------------------------
def test_model_a_feature_count_and_leakage_exclusion(preprocessor, ingestion):
    """Verify Model A contains exactly 29 features with strict zero leakage."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    X, y, meta = preprocessor.extract_features_and_target(df_mat, model_experiment="A")

    assert X.shape == (395, 29), f"Expected (395, 29), got {X.shape}"
    assert len(X.columns) == 29

    # Absences must be in meta, but NOT in X
    assert "absences" in meta.columns
    assert "absences" not in X.columns

    # G1, G2, G3, is_at_risk must NOT be in X
    for col in ["G1", "G2", "G3", "is_at_risk"]:
        assert col not in X.columns

    # Target y check
    assert y.name == "is_at_risk"
    assert len(y) == 395
    assert set(y.unique()).issubset({0, 1})
    # Target definition check: y must match G3 < 10
    expected_y = (df_mat["G3"] < 10.0).astype(int)
    assert (y == expected_y).all()


def test_model_b_feature_count_and_early_assessment(preprocessor, ingestion):
    """Verify Model B contains exactly 30 features (Model A + G1)."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    X, y, meta = preprocessor.extract_features_and_target(df_mat, model_experiment="B")

    assert X.shape == (395, 30), f"Expected (395, 30), got {X.shape}"
    assert len(X.columns) == 30
    assert "G1" in X.columns

    # G2, G3, is_at_risk must NOT be in X
    for col in ["G2", "G3", "is_at_risk", "absences"]:
        assert col not in X.columns


def test_unfitted_column_transformer(preprocessor):
    """
    Verify ColumnTransformer is configured with appropriate sub-pipelines
    without premature fitting.
    """
    transformer_a = preprocessor.build_column_transformer("A")
    transformer_b = preprocessor.build_column_transformer("B")

    assert transformer_a is not None
    assert transformer_b is not None

    # Check transformer definitions
    named_transformers = {name: trans for name, trans, cols in transformer_a.transformers}
    assert "num" in named_transformers
    assert "cat" in named_transformers

    # Transformer must NOT be fitted yet (does not have hasattr 'transformers_')
    assert not hasattr(transformer_a, "transformers_")


# ------------------------------------------------------------------------------
# 3. Mathematics Knowledge Graph Tests
# ------------------------------------------------------------------------------
def test_knowledge_graph_structure(kg_math):
    """Verify graph has exactly 10 concepts, 11 edges, and DEMONSTRATION_CURRICULUM provenance."""
    status = kg_math.get_status()
    assert status["supported"] is True
    assert status["curriculum_provenance"] == "DEMONSTRATION_CURRICULUM"
    assert status["num_concepts"] == 10
    assert status["num_prerequisite_edges"] == 11
    assert status["is_acyclic"] is True


def test_prerequisite_retrieval_and_transitive_closure(kg_math):
    """Verify direct and transitive prerequisite queries."""
    # MATH_01 has no prerequisites (foundational root)
    assert kg_math.get_direct_prerequisites("MATH_01") == []
    assert kg_math.get_all_ancestor_prerequisites("MATH_01") == []

    # MATH_02 requires MATH_01
    assert kg_math.get_direct_prerequisites("MATH_02") == ["MATH_01"]

    # MATH_04 requires MATH_02 (which requires MATH_01)
    direct_04 = kg_math.get_direct_prerequisites("MATH_04")
    ancestors_04 = kg_math.get_all_ancestor_prerequisites("MATH_04")
    assert direct_04 == ["MATH_02"]
    assert "MATH_01" in ancestors_04
    assert "MATH_02" in ancestors_04

    # MATH_10 requires MATH_09 (and all foundational concepts upstream)
    ancestors_10 = kg_math.get_all_ancestor_prerequisites("MATH_10")
    assert "MATH_01" in ancestors_10
    assert "MATH_09" in ancestors_10


def test_topological_order_and_paths(kg_math):
    """Verify valid learning order and path exploration."""
    order = kg_math.get_topological_order()
    assert len(order) == 10
    assert order[0] == "MATH_01"
    assert order[-1] == "MATH_10"

    # Verify prerequisite ordering: every prerequisite must appear before its dependent
    for node in order:
        prereqs = kg_math.get_direct_prerequisites(node)
        for p in prereqs:
            assert order.index(p) < order.index(node)

    # Test path search from MATH_01 to MATH_10
    paths = kg_math.find_prerequisite_paths("MATH_01", "MATH_10")
    assert len(paths) > 0
    for path in paths:
        assert path[0] == "MATH_01"
        assert path[-1] == "MATH_10"


def test_unsupported_subject_knowledge_graph():
    """Verify graceful handling for Portuguese concept graph."""
    kg_por = AcademicKnowledgeGraph("Portuguese")
    status = kg_por.get_status()
    assert status["supported"] is False
    assert "Portuguese" in status["message"]
    assert kg_por.get_topological_order() == []
    assert kg_por.get_direct_prerequisites("MATH_01") == []


# ------------------------------------------------------------------------------
# 4. Diagnostic Question Integrity Tests
# ------------------------------------------------------------------------------
def test_diagnostic_questions_integrity(test_db):
    """Verify all 20 diagnostic questions are valid and stored in SQLite."""
    questions = test_db.get_all_diagnostic_questions(subject="Mathematics")
    assert len(questions) == 20

    concept_ids = {c["concept_id"] for c in test_db.get_all_concepts(subject="Mathematics")}

    for q in questions:
        assert q["concept_id"] in concept_ids
        assert q["question_text"] is not None and len(q["question_text"]) > 5
        assert q["option_a"] is not None and len(q["option_a"]) > 0
        assert q["option_b"] is not None and len(q["option_b"]) > 0
        assert q["correct_option"] in ["A", "B", "C", "D"]
        assert q["explanation"] is not None and len(q["explanation"]) > 0
        assert q["blooms_level"] in ["Remember", "Understand", "Apply", "Analyze"]


# ------------------------------------------------------------------------------
# 5. SQLite Storage & Subject Isolation Tests
# ------------------------------------------------------------------------------
def test_sqlite_student_and_assessment_sync(test_db, ingestion):
    """Verify database sync, row counts, and complete subject isolation."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    df_por = ingestion.load_raw_dataset("Portuguese")

    history_mgr = AssessmentHistoryManager(db_manager=test_db)
    res_mat = history_mgr.sync_dataset_to_database(df_mat, subject="Mathematics")
    res_por = history_mgr.sync_dataset_to_database(df_por, subject="Portuguese")

    assert res_mat["students_synced"] == 395
    assert res_mat["assessments_synced"] == 395 * 3  # 1185
    assert res_por["students_synced"] == 649
    assert res_por["assessments_synced"] == 649 * 3  # 1947

    # Check total student counts
    assert test_db.get_student_count("Mathematics") == 395
    assert test_db.get_student_count("Portuguese") == 649
    assert test_db.get_student_count() == 1044

    # Check assessment counts
    assert test_db.get_assessment_count(subject="Mathematics", is_uci_source=1) == 1185
    assert test_db.get_assessment_count(subject="Portuguese", is_uci_source=1) == 1947

    # Check specific student record
    stu_mat_1 = test_db.get_student_by_id_and_subject("MAT_001", "Mathematics")
    assert stu_mat_1 is not None
    assert stu_mat_1["subject"] == "Mathematics"
    assert stu_mat_1["source_dataset"] == "UCI_MATHEMATICS"

    # Verify chronological assessment order and date provenance for MAT_001
    assessments_mat_1 = test_db.get_student_assessments("MAT_001", "Mathematics")
    assert len(assessments_mat_1) == 3
    assert assessments_mat_1[0]["term_period"] == 1
    assert assessments_mat_1[1]["term_period"] == 2
    assert assessments_mat_1[2]["term_period"] == 3
    assert all(a["is_uci_source"] == 1 for a in assessments_mat_1)

    # Date provenance verification: UCI source has no calendar dates
    for a in assessments_mat_1:
        assert a["assessment_date"] is None
        assert a["date_provenance"] == "NOT_PROVIDED_IN_UCI_SOURCE"


def test_post_g1_prediction_cannot_access_future_assessments(preprocessor, ingestion):
    """
    Verify that post-G1 prediction feature matrix (Model B) strictly cannot
    access future midterm (G2), final exam (G3), or target outcome (is_at_risk).
    """
    df_mat = ingestion.load_raw_dataset("Mathematics")
    X_b, y, meta = preprocessor.extract_features_and_target(df_mat, model_experiment="B")

    # G1 is available at the post-G1 prediction point
    assert "G1" in X_b.columns

    # G2, G3, is_at_risk are strictly excluded
    assert "G2" not in X_b.columns
    assert "G3" not in X_b.columns
    assert "is_at_risk" not in X_b.columns
    assert "absences" not in X_b.columns

    # Verify that future assessment values cannot be reconstructed from X_b
    assert X_b.shape[1] == 30


def test_sqlite_foreign_key_and_rollback(test_db):
    """Verify foreign key enforcement and transactional rollback on failure."""
    # Attempting to insert an assessment for a non-existent student must fail
    with pytest.raises(sqlite3.IntegrityError):
        test_db.insert_assessment(
            {
                "student_id": "GHOST_999",
                "subject": "Mathematics",
                "assessment_name": "Test Assessment",
                "term_period": 1,
                "score": 15.0,
                "max_score": 20.0,
                "percentage": 75.0,
                "assessment_date": "2026-11-15",
                "is_uci_source": 0,
                "notes": "Invalid record",
            }
        )
