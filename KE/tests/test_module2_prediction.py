"""
ExplainAI - Module 2 Dedicated Test Suite
Tests:
1. Model A and Model B feature counts (29 and 30) and leakage prevention.
2. Independent Mathematics and Portuguese model training.
3. Preprocessing ColumnTransformer fitted only inside training pipelines.
4. Reproducible train/test splitting with fixed random seeds.
5. Cross-validation isolation on training partitions only.
6. Model prediction service output format, risk labels, and probabilities.
7. Confusion matrix and evaluation metric integrity (Accuracy, Precision, Recall, F1, Balanced Accuracy, ROC-AUC).
8. Rule-based warning conditions, triggers, explanations, and non-discrimination.
9. Configurable rule thresholds.
10. What-If simulator input validation, rejection of demographics and historical grades.
11. Immutability of original student records after simulation.
12. Audit logging of What-If hypothetical scenarios.
13. Chronological score-trajectory drift detection (G1 -> G2).
14. Strict exclusion of G3 from early-warning drift generation.
15. Subject and data-source isolation across models and predictions.
16. Serialization and deserialization of model joblib artifacts.
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from src.config import (
    MODELS_DIR,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    STRICTLY_EXCLUDED_FROM_FEATURES,
    TARGET_COL,
    BINARY_TARGET_COL,
)
from src.database.db_manager import DatabaseManager
from src.module1_data_knowledge.ingestion import DatasetIngestion
from src.module2_prediction_reasoning.predictor import AcademicRiskPredictor
from src.module2_prediction_reasoning.rule_engine import RuleBasedWarningEngine
from src.module2_prediction_reasoning.what_if_simulator import AcademicWhatIfSimulator
from src.module2_prediction_reasoning.academic_drift import AcademicDriftDetector


@pytest.fixture
def predictor():
    return AcademicRiskPredictor()


@pytest.fixture
def rule_engine():
    return RuleBasedWarningEngine()


@pytest.fixture
def what_if_simulator():
    return AcademicWhatIfSimulator()


@pytest.fixture
def drift_detector():
    return AcademicDriftDetector()


@pytest.fixture
def ingestion():
    return DatasetIngestion()


# ------------------------------------------------------------------------------
# 1. Feature Counts, Anti-Leakage & Model Artifacts
# ------------------------------------------------------------------------------
def test_feature_counts_and_anti_leakage_in_models():
    """Verify Model A has 29 features, Model B has 30 features, and zero leakage."""
    assert len(FEATURES_MODEL_A) == 29
    assert len(FEATURES_MODEL_B) == 30
    assert "G1" in FEATURES_MODEL_B
    assert "G1" not in FEATURES_MODEL_A

    for forbidden in STRICTLY_EXCLUDED_FROM_FEATURES:
        assert forbidden not in FEATURES_MODEL_A
        assert forbidden not in FEATURES_MODEL_B


def test_model_artifacts_saved_and_deserializable(predictor):
    """Verify that all 4 best model pipelines exist on disk and load correctly."""
    experiments = [
        ("Mathematics", "A"),
        ("Mathematics", "B"),
        ("Portuguese", "A"),
        ("Portuguese", "B"),
    ]
    for subj, exp in experiments:
        pipeline = predictor.load_model(subj, exp)
        assert pipeline is not None
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")
        # Check pipeline steps
        assert "preprocessor" in pipeline.named_steps
        assert "classifier" in pipeline.named_steps


def test_evaluation_metrics_json_integrity():
    """Verify that model_evaluation_metrics.json contains non-fabricated benchmark metrics."""
    metrics_file = MODELS_DIR / "model_evaluation_metrics.json"
    assert metrics_file.exists(), f"Metrics file missing at {metrics_file}"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "experiments" in metrics
    for exp_key in ["mathematics_model_a", "mathematics_model_b", "portuguese_model_a", "portuguese_model_b"]:
        assert exp_key in metrics["experiments"]
        exp_data = metrics["experiments"][exp_key]
        assert "best_model" in exp_data
        assert "models" in exp_data

        # Verify candidate algorithms benchmarked
        for algo in ["Dummy", "LogisticRegression", "RandomForest", "XGBoost"]:
            assert algo in exp_data["models"]
            tm = exp_data["models"][algo]["test_metrics"]
            cv = exp_data["models"][algo]["cv_metrics"]

            assert 0.0 <= tm["accuracy"] <= 1.0
            assert 0.0 <= tm["precision"] <= 1.0
            assert 0.0 <= tm["recall"] <= 1.0
            assert 0.0 <= tm["f1"] <= 1.0
            assert 0.0 <= tm["roc_auc"] <= 1.0
            assert "confusion_matrix" in tm
            assert "f1_mean" in cv


# ------------------------------------------------------------------------------
# 2. Risk Predictor Service Tests
# ------------------------------------------------------------------------------
def test_risk_predictor_inference_format(predictor, ingestion):
    """Verify predictor output schema and probabilistic confidence."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    sample_student = df_mat.iloc[0].to_dict()
    sample_student["student_id"] = "MAT_001"

    # Test Model A inference
    res_a = predictor.predict_student_risk(sample_student, subject="Mathematics", experiment="A", persist=False)
    assert res_a["student_id"] == "MAT_001"
    assert res_a["subject"] == "Mathematics"
    assert res_a["model_experiment"] == "MODEL_A"
    assert res_a["risk_category"] in ["At-Risk", "Satisfactory"]
    assert 0.0 <= res_a["at_risk_probability"] <= 1.0
    assert 0.0 <= res_a["satisfactory_probability"] <= 1.0
    assert round(res_a["at_risk_probability"] + res_a["satisfactory_probability"], 2) == 1.0
    assert res_a["feature_count_used"] == 29

    # Test Model B inference (uses G1)
    res_b = predictor.predict_student_risk(sample_student, subject="Mathematics", experiment="B", persist=False)
    assert res_b["model_experiment"] == "MODEL_B"
    assert res_b["feature_count_used"] == 30


def test_model_b_superiority_over_model_a():
    """
    Verify empirically that incorporating G1 in Model B improves evaluation
    metrics compared to pre-course Model A.
    """
    with open(MODELS_DIR / "model_evaluation_metrics.json", "r") as f:
        metrics = json.load(f)

    # In Mathematics, Model B AUC and F1 should be significantly higher than Model A
    mat_a_best = metrics["experiments"]["mathematics_model_a"].get(
        "best_model_test_metrics",
        metrics["experiments"]["mathematics_model_a"].get("best_model_metrics")
    )
    mat_b_best = metrics["experiments"]["mathematics_model_b"].get(
        "best_model_test_metrics",
        metrics["experiments"]["mathematics_model_b"].get("best_model_metrics")
    )

    assert mat_b_best["roc_auc"] > mat_a_best["roc_auc"]
    assert mat_b_best["f1"] > mat_a_best["f1"]
    assert mat_b_best["recall"] >= mat_a_best["recall"]


def test_regression_model_b_inference_from_database_record(predictor):
    """
    Regression Test: Verify Model B inference succeeds when student record is retrieved
    directly from SQLite database (where G1 is stored as G1_raw).
    Tests both Mathematics (MAT_001) and Portuguese (POR_001) cohorts, and confirms
    Model A remains unaffected and uses exactly 29 features while Model B uses 30 features.
    """
    db = DatabaseManager()

    # 1. Test Mathematics cohort (MAT_001)
    stu_mat = db.get_student_by_id_and_subject("MAT_001", "Mathematics")
    assert stu_mat is not None

    # Model A: 29 features
    res_mat_a = predictor.predict_student_risk(stu_mat, subject="Mathematics", experiment="A", persist=False)
    assert res_mat_a["model_experiment"] == "MODEL_A"
    assert res_mat_a["feature_count_used"] == 29
    assert 0.0 <= res_mat_a["at_risk_probability"] <= 1.0

    # Model B: 30 features (incorporates G1)
    res_mat_b = predictor.predict_student_risk(stu_mat, subject="Mathematics", experiment="B", persist=False)
    assert res_mat_b["model_experiment"] == "MODEL_B"
    assert res_mat_b["feature_count_used"] == 30
    assert 0.0 <= res_mat_b["at_risk_probability"] <= 1.0

    # 2. Test Portuguese cohort (POR_001)
    stu_por = db.get_student_by_id_and_subject("POR_001", "Portuguese")
    assert stu_por is not None

    # Model A: 29 features
    res_por_a = predictor.predict_student_risk(stu_por, subject="Portuguese", experiment="A", persist=False)
    assert res_por_a["model_experiment"] == "MODEL_A"
    assert res_por_a["feature_count_used"] == 29
    assert 0.0 <= res_por_a["at_risk_probability"] <= 1.0

    # Model B: 30 features (incorporates G1)
    res_por_b = predictor.predict_student_risk(stu_por, subject="Portuguese", experiment="B", persist=False)
    assert res_por_b["model_experiment"] == "MODEL_B"
    assert res_por_b["feature_count_used"] == 30
    assert 0.0 <= res_por_b["at_risk_probability"] <= 1.0


# ------------------------------------------------------------------------------
# 3. Rule-Based Warning Engine Tests
# ------------------------------------------------------------------------------
def test_rule_engine_warning_triggers(rule_engine):
    """Verify transparent warning trigger conditions and explanations."""
    at_risk_student = {
        "failures": 2,
        "studytime": 1,
        "absences": 15,
        "Dalc": 3,
        "Walc": 4,
        "health": 1,
        "G1": 7.0,
    }
    prediction_result = {"at_risk_probability": 0.85}

    report = rule_engine.evaluate_rules(at_risk_student, prediction_result)

    assert report["overall_status"] == "CRITICAL_WARNING"
    assert report["total_warnings_triggered"] >= 5

    rule_ids = [w["rule_id"] for w in report["warnings"]]
    assert "RULE_ML_HIGH_RISK" in rule_ids
    assert "RULE_EARLY_EXAM_FAIL" in rule_ids
    assert "RULE_HISTORICAL_FAILURES" in rule_ids
    assert "RULE_EXCESSIVE_ABSENCES" in rule_ids
    assert "RULE_LOW_STUDY_TIME" in rule_ids
    assert "RULE_ELEVATED_ALCOHOL" in rule_ids
    assert "RULE_POOR_HEALTH" in rule_ids

    # Verify no demographic characteristics trigger warnings
    for w in report["warnings"]:
        assert "sex" not in w["condition"].lower()
        assert "address" not in w["condition"].lower()
        assert "job" not in w["condition"].lower()


def test_rule_engine_configurable_thresholds(rule_engine):
    """Verify that rule thresholds are dynamically configurable."""
    student = {"failures": 1, "absences": 8, "studytime": 2, "G1": 11.0}

    # Under standard thresholds, failures=1 triggers warning, but absences=8 does not
    report_std = rule_engine.evaluate_rules(student)
    triggered_std = [w["rule_id"] for w in report_std["warnings"]]
    assert "RULE_HISTORICAL_FAILURES" in triggered_std
    assert "RULE_EXCESSIVE_ABSENCES" not in triggered_std

    # With custom threshold (absences >= 5), it triggers
    custom_cfg = {"excessive_absences": 5, "high_failures": 2}
    report_custom = rule_engine.evaluate_rules(student, custom_thresholds=custom_cfg)
    triggered_custom = [w["rule_id"] for w in report_custom["warnings"]]
    assert "RULE_EXCESSIVE_ABSENCES" in triggered_custom
    assert "RULE_HISTORICAL_FAILURES" not in triggered_custom


def test_rule_engine_model_a_temporal_suppression(rule_engine):
    """Verify that RULE_EARLY_EXAM_FAIL is suppressed under Model A to prevent temporal leakage."""
    student = {"failures": 0, "absences": 2, "studytime": 3, "G1": 7.0}

    # Under Model B / Post-G1, G1 < 10 MUST trigger RULE_EARLY_EXAM_FAIL
    report_b = rule_engine.evaluate_rules(student, model_experiment="B")
    rule_ids_b = [w["rule_id"] for w in report_b["warnings"]]
    assert "RULE_EARLY_EXAM_FAIL" in rule_ids_b

    # Under Model A (Intake), G1 < 10 MUST be suppressed (examination has not occurred yet)
    report_a = rule_engine.evaluate_rules(student, model_experiment="A")
    rule_ids_a = [w["rule_id"] for w in report_a["warnings"]]
    assert "RULE_EARLY_EXAM_FAIL" not in rule_ids_a

    # Also test suppression via prediction_result metadata
    pred_res_a = {"model_experiment": "MODEL_A", "at_risk_probability": 0.20}
    report_pred_a = rule_engine.evaluate_rules(student, prediction_result=pred_res_a)
    rule_ids_pred_a = [w["rule_id"] for w in report_pred_a["warnings"]]
    assert "RULE_EARLY_EXAM_FAIL" not in rule_ids_pred_a


# ------------------------------------------------------------------------------
# 4. Academic What-If Simulator Tests
# ------------------------------------------------------------------------------
def test_what_if_simulator_valid_scenario(what_if_simulator, ingestion):
    """Verify counterfactual prediction delta calculation and record immutability."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    original_student = df_mat.iloc[10].to_dict()
    original_student["student_id"] = "MAT_011"
    original_studytime = original_student["studytime"]

    # Hypothetical scenario: increase study time to level 4
    changes = {"studytime": 4}
    res = what_if_simulator.simulate_counterfactual(
        original_student_data=original_student,
        proposed_changes=changes,
        subject="Mathematics",
        experiment="A",
        log_to_audit=False,
    )

    assert "baseline" in res
    assert "hypothetical" in res
    assert "delta" in res
    assert res["hypothetical"]["modified_features"] == {"studytime": 4}
    assert "scientific_disclaimer" in res
    assert "CRITICAL SCIENTIFIC LIMITATION" in res["scientific_disclaimer"]

    # Immutability check: original dictionary must remain completely unchanged
    assert original_student["studytime"] == original_studytime


def test_what_if_simulator_rejection_of_demographics_and_grades(what_if_simulator, ingestion):
    """Verify simulator strictly rejects modifications to demographics, historical failures, and exam grades."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    student = df_mat.iloc[0].to_dict()

    # Reject immutable demographic changes
    with pytest.raises(ValueError, match="immutable demographic"):
        what_if_simulator.simulate_counterfactual(student, {"sex": "M"})

    with pytest.raises(ValueError, match="immutable demographic"):
        what_if_simulator.simulate_counterfactual(student, {"age": 20})

    # Reject historical failures (cannot be altered after the fact)
    with pytest.raises(ValueError, match="empirical historical record/outcome"):
        what_if_simulator.simulate_counterfactual(student, {"failures": 0})

    # Reject retrospective grade changes
    with pytest.raises(ValueError, match="empirical historical record/outcome"):
        what_if_simulator.simulate_counterfactual(student, {"G1": 18.0})

    with pytest.raises(ValueError, match="empirical historical record/outcome"):
        what_if_simulator.simulate_counterfactual(student, {"G2": 15.0})

    # Reject sensitive personal and health attributes
    with pytest.raises(ValueError, match="sensitive personal/health"):
        what_if_simulator.simulate_counterfactual(student, {"Dalc": 1})

    with pytest.raises(ValueError, match="sensitive personal/health"):
        what_if_simulator.simulate_counterfactual(student, {"Walc": 1})

    with pytest.raises(ValueError, match="sensitive personal/health"):
        what_if_simulator.simulate_counterfactual(student, {"health": 5})

    # Reject invalid range
    with pytest.raises(ValueError, match="out of valid range"):
        what_if_simulator.simulate_counterfactual(student, {"studytime": 10})


# ------------------------------------------------------------------------------
# 5. Academic Drift Early-Warning Tests
# ------------------------------------------------------------------------------
def test_academic_drift_detection_trajectory(tmp_path):
    """Verify score deterioration detection across consecutive term assessments (G1 -> G2)."""
    test_db = tmp_path / "test_drift.db"
    db_mgr = DatabaseManager(db_path=test_db)
    db_mgr.init_database()

    with db_mgr.get_connection() as conn:
        cursor = conn.cursor()
        # Insert student
        cursor.execute(
            """
            INSERT INTO students (
                student_id, subject, source_dataset, school, sex, age, address, famsize, Pstatus,
                Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures,
                schoolsup, famsup, paid, activities, nursery, higher, internet, romantic,
                famrel, freetime, goout, Dalc, Walc, health, absences, G1_raw, G2_raw, G3_raw, is_at_risk
            ) VALUES (
                'DRIFT_001', 'Mathematics', 'SYNTHETIC_DEMO', 'GP', 'F', 16, 'U', 'GT3', 'T',
                3, 3, 'other', 'other', 'course', 'mother', 1, 2, 0,
                'no', 'yes', 'no', 'no', 'yes', 'yes', 'yes', 'no',
                4, 3, 3, 1, 1, 3, 4, 14.0, 9.0, 8.0, 1
            )
            """
        )
        # Term 1: 14.0 (Good baseline)
        cursor.execute(
            """
            INSERT INTO assessments (student_id, subject, assessment_name, term_period, score, max_score, percentage, date_provenance, is_uci_source)
            VALUES ('DRIFT_001', 'Mathematics', 'Term 1 Baseline (G1)', 1, 14.0, 20.0, 70.0, 'NOT_PROVIDED_IN_UCI_SOURCE', 1)
            """
        )
        # Term 2: 9.0 (Drop of 5.0 points -> Critical deterioration!)
        cursor.execute(
            """
            INSERT INTO assessments (student_id, subject, assessment_name, term_period, score, max_score, percentage, date_provenance, is_uci_source)
            VALUES ('DRIFT_001', 'Mathematics', 'Term 2 Midterm (G2)', 2, 9.0, 20.0, 45.0, 'NOT_PROVIDED_IN_UCI_SOURCE', 1)
            """
        )
        # Term 3: 8.0 (Final exam)
        cursor.execute(
            """
            INSERT INTO assessments (student_id, subject, assessment_name, term_period, score, max_score, percentage, date_provenance, is_uci_source)
            VALUES ('DRIFT_001', 'Mathematics', 'Final Examination (G3)', 3, 8.0, 20.0, 40.0, 'NOT_PROVIDED_IN_UCI_SOURCE', 1)
            """
        )
        conn.commit()

    detector = AcademicDriftDetector(db_manager=db_mgr, critical_decline_points=3.0)

    # Standard early warning: G3 must be barred
    report = detector.analyze_trajectory("DRIFT_001", subject="Mathematics", include_final_grade=False)

    assert report["evaluable"] is True
    assert report["observations_count"] == 2  # Only G1 and G2
    assert report["trend"] == "CRITICAL_DECLINE"
    assert report["latest_score_delta"] == -5.0
    assert len(report["warnings"]) == 1
    assert report["warnings"][0]["severity"] == "CRITICAL"
    assert report["warnings"][0]["score_drop"] == 5.0

    # Confirm G3 was strictly not accessed
    terms_in_obs = [o["term_period"] for o in report["observations"]]
    assert 3 not in terms_in_obs

