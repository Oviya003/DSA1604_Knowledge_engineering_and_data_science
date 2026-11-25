"""
ExplainAI - Module 3 Dedicated Test Suite
Tests:
1. SHAP Explainer model loading across all 4 experiments (Math A/B, Por A/B).
2. Binary positive-class (At-Risk) SHAP value extraction and shape handling.
3. Transformed feature mapping to original human-readable feature names.
4. Global feature importance calculation and ranking.
5. Strict anti-leakage in SHAP explanation inputs (no G2, G3, is_at_risk, or absences).
6. Knowledge Gap Detective diagnostic question retrieval from SQLite.
7. Diagnostic quiz evaluation, answer validation, and score calculation.
8. Mastery status classification (MASTERED >=0.80, IN_PROGRESS 0.50-0.79, CRITICAL_GAP <0.50).
9. Provenance preservation: is_uci_source = 0 for quiz attempts.
10. Prerequisite DAG traversal in topological order.
11. Accurate distinction between unassessed concepts and knowledge gaps.
12. Unsupported Portuguese subject handling in Knowledge Gap Detective.
13. Personalized Learning Recommender multi-signal synthesis and prioritization.
14. Ethical non-discrimination guard (demographics cannot trigger recommendations).
15. Closed-Loop Improvement Tracker intervention lifecycle (PRESCRIBED -> IN_PROGRESS -> COMPLETED).
16. Before-and-after score delta calculation and mastery progression.
17. Historical intervention and attempt persistence in SQLite.
18. Scientific and causal disclaimers on SHAP, recommender, and tracker outputs.
"""

import pytest
from pathlib import Path
from typing import Dict, Any

from src.database.db_manager import DatabaseManager
from src.module1_data_knowledge.ingestion import DatasetIngestion
from src.module2_prediction_reasoning.predictor import AcademicRiskPredictor
from src.module2_prediction_reasoning.rule_engine import RuleBasedWarningEngine
from src.module3_explainability_support.shap_explainer import (
    AcademicShapExplainer,
    SHAP_SCIENTIFIC_DISCLAIMER,
)
from src.module3_explainability_support.knowledge_gap_detective import (
    KnowledgeGapDetective,
    COARSE_ASSESSMENT_LIMITATION,
    MASTERY_THRESHOLD_MASTERED,
    MASTERY_THRESHOLD_IN_PROGRESS,
)
from src.module3_explainability_support.recommender import (
    PersonalizedLearningRecommender,
    RECOMMENDER_DISCLAIMER,
)
from src.module3_explainability_support.improvement_tracker import (
    ClosedLoopImprovementTracker,
    TRACKER_CAUSAL_DISCLAIMER,
)


@pytest.fixture
def test_db(tmp_path):
    """Provides a fresh, isolated SQLite test database with seeded curriculum."""
    db_file = tmp_path / "test_explainai_m3.db"
    mgr = DatabaseManager(db_path=db_file)
    mgr.init_database()
    return mgr


@pytest.fixture
def shap_service():
    return AcademicShapExplainer()


@pytest.fixture
def gap_detective(test_db):
    return KnowledgeGapDetective(db_manager=test_db)


@pytest.fixture
def recommender(test_db):
    return PersonalizedLearningRecommender(db_manager=test_db)


@pytest.fixture
def tracker(test_db, gap_detective):
    return ClosedLoopImprovementTracker(db_manager=test_db, gap_detective=gap_detective)


@pytest.fixture
def ingestion():
    return DatasetIngestion()


# ------------------------------------------------------------------------------
# 1. SHAP Explainability Service Tests
# ------------------------------------------------------------------------------
def test_shap_explainer_loading_and_instance_explanation(shap_service, ingestion):
    """Verify local SHAP prediction explanation across all 4 experiments."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    df_por = ingestion.load_raw_dataset("Portuguese")

    sample_mat = df_mat.iloc[0].to_dict()
    sample_mat["student_id"] = "MAT_TEST"

    sample_por = df_por.iloc[0].to_dict()
    sample_por["student_id"] = "POR_TEST"

    # Test Mathematics Model A (Logistic Regression)
    res_mat_a = shap_service.explain_instance(sample_mat, subject="Mathematics", experiment="A")
    assert res_mat_a["student_id"] == "MAT_TEST"
    assert res_mat_a["subject"] == "Mathematics"
    assert res_mat_a["experiment"] == "MODEL_A"
    assert "top_risk_increasing_factors" in res_mat_a
    assert "top_risk_decreasing_factors" in res_mat_a
    assert "all_feature_contributions" in res_mat_a
    assert len(res_mat_a["all_feature_contributions"]) == 55
    assert SHAP_SCIENTIFIC_DISCLAIMER == res_mat_a["scientific_disclaimer"]

    # Test Mathematics Model B (Logistic Regression with G1)
    res_mat_b = shap_service.explain_instance(sample_mat, subject="Mathematics", experiment="B")
    assert res_mat_b["experiment"] == "MODEL_B"
    assert len(res_mat_b["all_feature_contributions"]) == 56

    # Test Portuguese Model A (Random Forest Classifier)
    res_por_a = shap_service.explain_instance(sample_por, subject="Portuguese", experiment="A")
    assert res_por_a["classifier_type"] == "RandomForestClassifier"
    assert len(res_por_a["all_feature_contributions"]) == 55

    # Test Portuguese Model B (XGBoost Classifier)
    res_por_b = shap_service.explain_instance(sample_por, subject="Portuguese", experiment="B")
    assert res_por_b["classifier_type"] == "XGBClassifier"
    assert len(res_por_b["all_feature_contributions"]) == 56

    # Verify mathematical reconstruction across all model types
    assert res_mat_a["reconstruction_check"]["is_reconstructed"] is True
    assert res_mat_b["reconstruction_check"]["is_reconstructed"] is True
    assert res_por_a["reconstruction_check"]["is_reconstructed"] is True
    assert res_por_b["reconstruction_check"]["is_reconstructed"] is True


def test_shap_feature_mapping_and_anti_leakage(shap_service, ingestion):
    """Verify transformed features map back to human-readable names with no leaked columns."""
    df_mat = ingestion.load_raw_dataset("Mathematics")
    sample_mat = df_mat.iloc[1].to_dict()

    res = shap_service.explain_instance(sample_mat, subject="Mathematics", experiment="A")
    orig_feats = [c["original_feature"] for c in res["all_feature_contributions"]]

    # Verify no leaked features appear in contributions
    assert "G2" not in orig_feats
    assert "G3" not in orig_feats
    assert "is_at_risk" not in orig_feats
    assert "absences" not in orig_feats

    # Check that aggregated original features contain expected variables
    agg_names = [f["feature"] for f in res["aggregated_original_features"]]
    assert "studytime" in agg_names
    assert "failures" in agg_names
    assert "school" in agg_names


def test_shap_global_feature_importance(shap_service):
    """Verify calculation of global mean absolute SHAP values across training cohort."""
    glob_res = shap_service.compute_global_feature_importance("Mathematics", "A", max_samples=30)
    assert glob_res["subject"] == "Mathematics"
    assert glob_res["experiment"] == "MODEL_A"
    assert len(glob_res["ranked_original_features"]) == 29

    top_feat = glob_res["ranked_original_features"][0]
    assert "original_feature" in top_feat
    assert top_feat["sum_abs_shap"] >= 0.0


# ------------------------------------------------------------------------------
# 2. Knowledge Gap Detective Tests (Signature Feature 2)
# ------------------------------------------------------------------------------
def test_diagnostic_question_retrieval(gap_detective):
    """Verify retrieval of diagnostic questions for curriculum concepts."""
    q_data = gap_detective.get_concept_questions("MATH_01", subject="Mathematics", include_answers=False)
    assert q_data["supported"] is True
    assert q_data["concept_id"] == "MATH_01"
    assert q_data["question_count"] == 2
    for q in q_data["questions"]:
        assert "A" in q["options"]
        assert "B" in q["options"]
        assert "C" in q["options"]
        assert "D" in q["options"]
        assert "correct_option" not in q  # Hidden when include_answers=False


def test_diagnostic_quiz_evaluation_and_mastery_classification(gap_detective, test_db):
    """Verify answer validation, scoring, mastery categorization, and SQLite persistence."""
    # Seed student record
    test_db.insert_student({
        "student_id": "STU_001",
        "subject": "Mathematics",
        "source_dataset": "SYNTHETIC_DEMO",
        "school": "GP",
        "sex": "F",
        "age": 16,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 4,
        "absences": 2,
        "G1_raw": 12.0,
        "G2_raw": 13.0,
        "G3_raw": 14.0,
        "is_at_risk": 0,
    })

    q_data = gap_detective.get_concept_questions("MATH_01", subject="Mathematics", include_answers=True)
    questions = q_data["questions"]

    # Test 100% correct answers -> MASTERED (>= 0.80)
    all_correct = {q["question_id"]: q["correct_option"] for q in questions}
    res_mastered = gap_detective.evaluate_quiz_submission(
        student_id="STU_001",
        concept_id="MATH_01",
        submitted_answers=all_correct,
        quiz_type="BASELINE_DIAGNOSTIC",
    )
    assert res_mastered["score_fraction"] == 1.0
    assert res_mastered["percentage"] == 100.0
    assert res_mastered["mastery_status"] == "MASTERED"
    assert res_mastered["mastery_achieved"] is True

    # Verify attempt logged in SQLite with is_uci_source = 0
    attempts = test_db.get_student_quiz_attempts("STU_001", "Mathematics", "MATH_01")
    assert len(attempts) == 1
    assert attempts[0]["is_uci_source"] == 0
    assert attempts[0]["score"] == 1.0

    # Verify concept_mastery table updated
    mastery_rec = test_db.get_concept_mastery_record("STU_001", "Mathematics", "MATH_01")
    assert mastery_rec is not None
    assert mastery_rec["mastery_status"] == "MASTERED"

    # Test 0% correct answers -> CRITICAL_GAP (< 0.50)
    all_wrong = {q["question_id"]: "Z" for q in questions}
    res_gap = gap_detective.evaluate_quiz_submission(
        student_id="STU_001",
        concept_id="MATH_02",
        submitted_answers=all_wrong,
        quiz_type="BASELINE_DIAGNOSTIC",
    )
    assert res_gap["score_fraction"] == 0.0
    assert res_gap["mastery_status"] == "CRITICAL_GAP"
    assert res_gap["mastery_achieved"] is False


def test_prerequisite_gap_tracing_and_unassessed_distinction(gap_detective, test_db):
    """Verify prerequisite DAG traversal and distinction of unassessed concepts from gaps."""
    # Analyze prerequisites for MATH_05 (Linear Equations) for unassessed student
    analysis = gap_detective.analyze_prerequisite_gaps("STU_NEW", "MATH_05", subject="Mathematics")
    assert analysis["supported"] is True
    assert analysis["total_ancestor_prerequisites"] >= 2
    assert analysis["critical_gap_count"] == 0  # CRITICAL: Unassessed must NOT be counted as critical gaps!
    assert analysis["unassessed_count"] == analysis["total_ancestor_prerequisites"]

    # Verify all unassessed prerequisites are labeled 'NOT_YET_ASSESSED'
    for p in analysis["prerequisite_learning_path"]:
        assert p["mastery_status"] == "NOT_YET_ASSESSED"
        assert p["mastery_score"] is None


def test_unsupported_portuguese_subject_in_gap_detective(gap_detective):
    """Verify that Portuguese returns a graceful unsupported notice without inventing syllabus."""
    res_q = gap_detective.get_concept_questions("MATH_01", subject="Portuguese")
    assert res_q["supported"] is False
    assert "lacks a concept-level curriculum" in res_q["message"].lower() or "mathematics" in res_q["message"].lower()

    res_eval = gap_detective.evaluate_quiz_submission("STU_001", "MATH_01", {}, subject="Portuguese")
    assert res_eval["supported"] is False

    res_prereq = gap_detective.analyze_prerequisite_gaps("STU_001", "MATH_01", subject="Portuguese")
    assert res_prereq["supported"] is False


# ------------------------------------------------------------------------------
# 3. Personalized Learning Recommender Tests
# ------------------------------------------------------------------------------
def test_recommender_generation_and_evidence(recommender, gap_detective):
    """Verify multi-signal synthesis, prioritization, and rationale traceability."""
    student = {
        "student_id": "MAT_001",
        "studytime": 1,
        "failures": 2,
        "absences": 14,
        "G1": 7.0,
    }
    pred_res = {
        "risk_category": "At-Risk",
        "at_risk_probability": 0.82,
        "model_algorithm": "LogisticRegression",
    }
    rule_engine = RuleBasedWarningEngine()
    warn_res = rule_engine.evaluate_rules(student, pred_res, model_experiment="B")

    recs_res = recommender.generate_recommendations(
        student_data=student,
        prediction_result=pred_res,
        warning_report=warn_res,
        gap_analysis=None,
        subject="Mathematics",
        persist=False,
    )

    assert recs_res["total_recommendations"] >= 3
    recs = recs_res["recommendations"]

    # Check priority and category breakdown
    categories = [r["category"] for r in recs]
    assert "ACADEMIC_INTERVENTION" in categories
    assert "STUDY_HABITS" in categories
    assert "ATTENDANCE_SUPPORT" in categories

    for r in recs:
        assert r["priority"] in ["HIGH", "MEDIUM", "LOW"]
        assert len(r["recommendation_text"]) > 10
        assert len(r["rationale"]) > 10
        assert len(r["evidence_source"]) > 5
        # Verify no demographic attributes in recommendations
        assert "sex" not in r["evidence_source"].lower()
        assert "school" not in r["evidence_source"].lower()


# ------------------------------------------------------------------------------
# 4. Closed-Loop Improvement Tracker Tests (Signature Feature 4)
# ------------------------------------------------------------------------------
def test_closed_loop_intervention_lifecycle(tracker, test_db, gap_detective):
    """Verify prescription, status progression, reassessment quiz, and score delta calculation."""
    # Seed student record
    test_db.insert_student({
        "student_id": "MAT_LOOP",
        "subject": "Mathematics",
        "source_dataset": "SYNTHETIC_DEMO",
        "school": "GP",
        "sex": "M",
        "age": 16,
        "address": "U",
        "famsize": "LE3",
        "Pstatus": "T",
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "services",
        "Fjob": "services",
        "reason": "reputation",
        "guardian": "father",
        "traveltime": 1,
        "studytime": 1,
        "failures": 1,
        "schoolsup": "no",
        "famsup": "no",
        "paid": "no",
        "activities": "no",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 3,
        "freetime": 2,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 3,
        "absences": 4,
        "G1_raw": 8.0,
        "G2_raw": 9.0,
        "G3_raw": 9.0,
        "is_at_risk": 1,
    })

    # 1. Prescribe Intervention
    presc = tracker.prescribe_intervention(
        student_id="MAT_LOOP",
        subject="Mathematics",
        target_concept_id="MATH_01",
        intervention_type="REMEDIAL_EXERCISES",
        notes="Review integer addition and multiplication practice sheets.",
    )
    int_id = presc["intervention_id"]
    assert presc["status"] == "PRESCRIBED"

    # 2. Start Intervention
    start_res = tracker.start_intervention(int_id)
    assert start_res["current_status"] == "IN_PROGRESS"

    # 3. Complete Intervention with Post-Quiz Reassessment (100% correct answers)
    q_data = gap_detective.get_concept_questions("MATH_01", subject="Mathematics", include_answers=True)
    answers = {q["question_id"]: q["correct_option"] for q in q_data["questions"]}

    comp_res = tracker.complete_intervention_with_reassessment(int_id, answers)
    assert comp_res["status"] == "COMPLETED"
    assert comp_res["post_quiz_score"] == 1.0
    assert comp_res["new_mastery_status"] == "MASTERED"
    assert comp_res["score_delta"] > 0
    assert TRACKER_CAUSAL_DISCLAIMER == comp_res["scientific_disclaimer"]

    # 4. Verify History Summary
    hist = tracker.get_student_intervention_history("MAT_LOOP", "Mathematics")
    assert hist["total_interventions"] == 1
    assert hist["status_breakdown"]["COMPLETED"] == 1
    assert hist["status_breakdown"]["IN_PROGRESS"] == 0
