"""
ExplainAI - Module 3: Personalized Learning Recommender Service
Synthesizes predictive risk, rule-based alerts, SHAP attributions, and prerequisite
knowledge gaps into transparent, actionable educational recommendations.

ETHICAL & NON-DISCRIMINATION POLICY:
------------------------------------
Recommendations are generated strictly from actionable academic and behavioral signals
(e.g., prerequisite gaps, study routine, early exam scores, recorded absences).
Demographic and protected characteristics (e.g., sex, school, address, parental jobs)
are STRICTLY PROHIBITED from generating punitive or biased recommendations.
"""

from typing import Dict, List, Any, Optional
import datetime

from src.database.db_manager import DatabaseManager

RECOMMENDER_DISCLAIMER = (
    "PEDAGOGICAL ADVISORY NOTICE: Recommendations provide personalized academic suggestions "
    "derived from multivariate predictive signals, rule-based triggers, and diagnostic gap tracing. "
    "They represent advisory learning strategies and DO NOT guarantee improved grades or exam outcomes."
)


class PersonalizedLearningRecommender:
    """Generates traceable, evidence-backed pedagogical recommendations."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def generate_recommendations(
        self,
        student_data: Dict[str, Any],
        prediction_result: Optional[Dict[str, Any]] = None,
        warning_report: Optional[Dict[str, Any]] = None,
        shap_result: Optional[Dict[str, Any]] = None,
        gap_analysis: Optional[Dict[str, Any]] = None,
        subject: str = "Mathematics",
        persist: bool = True,
    ) -> Dict[str, Any]:
        """
        Synthesizes multiple diagnostic signals into prioritized, evidence-backed recommendations.
        """
        student_id = student_data.get("student_id", "ANONYMOUS")
        recommendations: List[Dict[str, Any]] = []

        # ----------------------------------------------------------------------
        # 1. Prerequisite Knowledge Gap Signals (Highest Academic Priority)
        # ----------------------------------------------------------------------
        if gap_analysis and gap_analysis.get("supported"):
            crit_gaps = gap_analysis.get("critical_gaps", [])
            in_prog = gap_analysis.get("in_progress_prerequisites", [])
            target_info = gap_analysis.get("target_concept", {})

            if crit_gaps:
                earliest_gap = crit_gaps[0]
                recommendations.append({
                    "priority": "HIGH",
                    "category": "KNOWLEDGE_GAP",
                    "target_concept_id": earliest_gap["concept_id"],
                    "suggested_action": "REMEDIAL_EXERCISES",
                    "recommendation_text": (
                        f"Complete foundational review and remedial practice for '{earliest_gap['name']}' "
                        f"before advancing to '{target_info.get('name', 'advanced concepts')}'."
                    ),
                    "rationale": (
                        f"Diagnostic testing revealed a critical knowledge gap (score: {earliest_gap.get('mastery_score', 0):.0%}) "
                        f"in '{earliest_gap['name']}', which is a required foundational prerequisite."
                    ),
                    "evidence_source": f"Diagnostic Quiz Gap: {earliest_gap['concept_id']} ({earliest_gap['name']})",
                })

            elif in_prog:
                earliest_prog = in_prog[0]
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "KNOWLEDGE_GAP",
                    "target_concept_id": earliest_prog["concept_id"],
                    "suggested_action": "CONCEPT_REVISION",
                    "recommendation_text": (
                        f"Reinforce practice problems on '{earliest_prog['name']}' to achieve full mastery "
                        f"prior to final term evaluations."
                    ),
                    "rationale": (
                        f"Current mastery level is in progress ({earliest_prog.get('mastery_score', 0):.0%}). "
                        f"Consolidating this concept will strengthen preparation for downstream coursework."
                    ),
                    "evidence_source": f"Diagnostic Quiz Progress: {earliest_prog['concept_id']} ({earliest_prog['name']})",
                })

        # ----------------------------------------------------------------------
        # 2. Rule-Based Academic Warnings Signals
        # ----------------------------------------------------------------------
        if warning_report and "warnings" in warning_report:
            for w in warning_report["warnings"]:
                rule_id = w.get("rule_id")

                if rule_id == "RULE_EARLY_EXAM_FAIL":
                    recommendations.append({
                        "priority": "HIGH",
                        "category": "ACADEMIC_INTERVENTION",
                        "target_concept_id": None,
                        "suggested_action": "TEACHER_1ON1",
                        "recommendation_text": (
                            "Schedule an individual 1-on-1 teacher academic conference to analyze first-period "
                            "exam errors and establish a targeted recovery roadmap."
                        ),
                        "rationale": (
                            f"Student scored below passing benchmark on First Period Exam ({w.get('observed_value')}). "
                            f"Early course intervention is statistically critical to prevent cumulative failure."
                        ),
                        "evidence_source": f"Rule Alert: {w.get('rule_name')} ({w.get('observed_value')})",
                    })

                elif rule_id == "RULE_LOW_STUDY_TIME":
                    recommendations.append({
                        "priority": "HIGH",
                        "category": "STUDY_HABITS",
                        "target_concept_id": None,
                        "suggested_action": "STUDY_SCHEDULE",
                        "recommendation_text": (
                            "Establish a structured weekly study timetable increasing self-study allocation "
                            "from <2 hours to 4-5 hours per week with designated homework review slots."
                        ),
                        "rationale": (
                            f"Current self-reported independent study allocation is Level {student_data.get('studytime', 1)} "
                            f"(<2 hrs/week). Regular study routines strongly correlate with passing performance."
                        ),
                        "evidence_source": f"Rule Alert: {w.get('rule_name')} (Level {student_data.get('studytime', 1)})",
                    })

                elif rule_id == "RULE_EXCESSIVE_ABSENCES":
                    abs_count = student_data.get("absences", 0)
                    recommendations.append({
                        "priority": "HIGH",
                        "category": "ATTENDANCE_SUPPORT",
                        "target_concept_id": None,
                        "suggested_action": "PASTORAL_CHECK",
                        "recommendation_text": (
                            "Conduct pastoral attendance review and arrange peer note-sharing for missed lectures."
                        ),
                        "rationale": (
                            f"Student has accumulated {abs_count} recorded absences, risking significant instructional gaps."
                        ),
                        "evidence_source": f"Descriptive Attendance Record: {abs_count} total absences",
                    })

                elif rule_id == "RULE_HISTORICAL_FAILURES":
                    fails = student_data.get("failures", 0)
                    recommendations.append({
                        "priority": "HIGH",
                        "category": "ACADEMIC_INTERVENTION",
                        "target_concept_id": None,
                        "suggested_action": "PEER_TUTORING",
                        "recommendation_text": (
                            "Enroll in structured peer tutoring sessions focusing on foundational problem-solving strategies."
                        ),
                        "rationale": (
                            f"Record indicates {fails} prior course failure(s). Dedicated peer tutoring mitigates recurring failure patterns."
                        ),
                        "evidence_source": f"Historical Failure Record: {fails} past course failure(s)",
                    })

        # ----------------------------------------------------------------------
        # 3. Model Predicted Risk & SHAP Attributions Signals
        # ----------------------------------------------------------------------
        if prediction_result and prediction_result.get("risk_category") == "At-Risk":
            # If high statistical risk but no specific rule triggered
            if not any(r["category"] == "ACADEMIC_INTERVENTION" for r in recommendations):
                prob = prediction_result.get("at_risk_probability", 0.5)
                recommendations.append({
                    "priority": "HIGH",
                    "category": "ACADEMIC_INTERVENTION",
                    "target_concept_id": None,
                    "suggested_action": "ACADEMIC_ADVISING",
                    "recommendation_text": (
                        "Refer student to academic counseling service for comprehensive learning plan evaluation."
                    ),
                    "rationale": (
                        f"Machine learning screening indicates elevated risk probability ({prob:.1%}). "
                        f"Early institutional engagement prevents progressive disengagement."
                    ),
                    "evidence_source": f"Predictive Model: {prediction_result.get('model_algorithm')} ({prob:.1%} At-Risk Probability)",
                })

        # ----------------------------------------------------------------------
        # 4. Default / Maintenance Recommendation if Profile is Satisfactory
        # ----------------------------------------------------------------------
        if not recommendations:
            recommendations.append({
                "priority": "LOW",
                "category": "MAINTENANCE",
                "target_concept_id": None,
                "suggested_action": "PROGRESS_MAINTENANCE",
                "recommendation_text": (
                    "Maintain current study routine and continue with standard course curriculum pace."
                ),
                "rationale": (
                    "Student profile demonstrates satisfactory academic indicators with no urgent knowledge gaps detected."
                ),
                "evidence_source": "Satisfactory Academic Profile Evaluation",
            })

        # Persist recommendations in SQLite if requested and student_id is known
        if persist and student_id != "ANONYMOUS":
            for rec in recommendations:
                try:
                    self.db.insert_learning_recommendation({
                        "student_id": student_id,
                        "subject": subject,
                        "priority": rec["priority"],
                        "recommendation_text": rec["recommendation_text"],
                        "rationale": rec["rationale"],
                        "status": "ACTIVE",
                    })
                except Exception:
                    pass

        return {
            "student_id": student_id,
            "subject": subject,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
            "generated_timestamp": datetime.datetime.now().isoformat(),
            "scientific_disclaimer": RECOMMENDER_DISCLAIMER,
        }

