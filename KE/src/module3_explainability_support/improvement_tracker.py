"""
ExplainAI - Module 3: Closed-Loop Learning Improvement Tracker (Signature Feature 4)
Orchestrates the end-to-end intervention lifecycle:
Risk Diagnosis -> Recommendation -> Prescription -> Activity Completion -> Reassessment -> Mastery Verification.

SCIENTIFIC LIMITATION NOTICE:
-----------------------------
Before-and-after diagnostic score increases measure observed empirical progress on
reassessment evaluations. They DO NOT constitute controlled clinical or causal proof
that the intervention alone caused the observed gain.
"""

from typing import Dict, List, Any, Optional
import datetime

from src.database.db_manager import DatabaseManager
from src.module3_explainability_support.knowledge_gap_detective import KnowledgeGapDetective

TRACKER_CAUSAL_DISCLAIMER = (
    "CAUSAL LIMITATION: A positive score change between baseline and post-intervention "
    "assessments indicates observed diagnostic progression. It does not establish causal proof "
    "that the intervention alone produced the change, as uncontrolled factors (e.g., extraneous "
    "study, classroom instruction, test familiarity) may contribute."
)


class ClosedLoopImprovementTracker:
    """Manages the full lifecycle of educational interventions and longitudinal outcome tracking."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        gap_detective: Optional[KnowledgeGapDetective] = None,
    ):
        self.db = db_manager or DatabaseManager()
        self.gap_detective = gap_detective or KnowledgeGapDetective(db_manager=self.db)

    def prescribe_intervention(
        self,
        student_id: str,
        subject: str,
        target_concept_id: str,
        intervention_type: str,
        notes: str = "",
        prescribed_by: str = "ExplainAI Support Engine",
    ) -> Dict[str, Any]:
        """
        Creates and stores a new educational intervention prescription.
        Lifecycle state: PRESCRIBED.
        """
        valid_types = ["REMEDIAL_EXERCISES", "PEER_TUTORING", "TEACHER_1ON1", "CONCEPT_REVISION"]
        if intervention_type not in valid_types:
            raise ValueError(f"Intervention type '{intervention_type}' invalid. Must be one of {valid_types}.")

        now_timestamp = datetime.datetime.now().isoformat()
        intervention_id = self.db.insert_intervention({
            "student_id": student_id,
            "subject": subject,
            "target_concept_id": target_concept_id,
            "intervention_type": intervention_type,
            "prescribed_date": now_timestamp,
            "status": "PRESCRIBED",
            "prescribed_by": prescribed_by,
            "notes": notes,
        })

        return {
            "intervention_id": intervention_id,
            "student_id": student_id,
            "subject": subject,
            "target_concept_id": target_concept_id,
            "intervention_type": intervention_type,
            "status": "PRESCRIBED",
            "prescribed_date": now_timestamp,
            "notes": notes,
            "message": f"Intervention #{intervention_id} successfully prescribed for {student_id}.",
        }

    def start_intervention(self, intervention_id: int) -> Dict[str, Any]:
        """Transitions an intervention from PRESCRIBED to IN_PROGRESS."""
        intervention = self.db.get_intervention_by_id(intervention_id)
        if not intervention:
            raise ValueError(f"Intervention #{intervention_id} not found.")

        self.db.update_intervention_status(intervention_id, new_status="IN_PROGRESS")
        return {
            "intervention_id": intervention_id,
            "student_id": intervention["student_id"],
            "target_concept_id": intervention["target_concept_id"],
            "previous_status": intervention["status"],
            "current_status": "IN_PROGRESS",
            "message": f"Intervention #{intervention_id} marked as in progress.",
        }

    def complete_intervention_with_reassessment(
        self,
        intervention_id: int,
        post_quiz_answers: Dict[str, str],
        quiz_type: str = "POST_INTERVENTION_CHECK",
    ) -> Dict[str, Any]:
        """
        Completes the intervention loop:
        1. Retrieves prior baseline score for the target concept.
        2. Evaluates the post-intervention diagnostic quiz attempt.
        3. Updates concept mastery in SQLite.
        4. Calculates empirical progress delta (post vs pre).
        5. Marks the intervention as COMPLETED.
        """
        intervention = self.db.get_intervention_by_id(intervention_id)
        if not intervention:
            raise ValueError(f"Intervention #{intervention_id} not found.")

        student_id = intervention["student_id"]
        subject = intervention["subject"]
        concept_id = intervention["target_concept_id"]

        # Fetch baseline diagnostic performance prior to this completion
        prior_mastery = self.db.get_concept_mastery_record(student_id, subject, concept_id)
        baseline_score = float(prior_mastery["mastery_score"]) if prior_mastery else 0.0
        baseline_status = prior_mastery["mastery_status"] if prior_mastery else "NOT_YET_ASSESSED"

        # Execute post-intervention diagnostic quiz evaluation
        quiz_res = self.gap_detective.evaluate_quiz_submission(
            student_id=student_id,
            concept_id=concept_id,
            submitted_answers=post_quiz_answers,
            quiz_type=quiz_type,
            subject=subject,
        )

        post_score = float(quiz_res["score_fraction"])
        new_mastery_status = quiz_res["mastery_status"]
        score_delta = round(post_score - baseline_score, 4)

        now_timestamp = datetime.datetime.now().isoformat()

        # Update intervention record to COMPLETED with post-quiz score
        self.db.update_intervention_status(
            intervention_id=intervention_id,
            new_status="COMPLETED",
            post_quiz_score=post_score,
            completed_date=now_timestamp,
        )

        # Determine pedagogical next steps
        if new_mastery_status == "MASTERED":
            next_action = "Advance to subsequent curriculum concepts in DAG order."
            outcome_assessment = "SUCCESSFUL_MASTERY_ACHIEVED"
        elif score_delta > 0:
            next_action = "Continue targeted revision; student demonstrated positive progress but has not yet reached mastery."
            outcome_assessment = "POSITIVE_PROGRESS_PARTIAL"
        else:
            next_action = "Schedule 1-on-1 teacher conference for deeper qualitative gap diagnosis."
            outcome_assessment = "PERSISTENT_GAP"

        return {
            "intervention_id": intervention_id,
            "student_id": student_id,
            "subject": subject,
            "target_concept_id": concept_id,
            "status": "COMPLETED",
            "completed_date": now_timestamp,
            "baseline_score": baseline_score,
            "baseline_mastery_status": baseline_status,
            "post_quiz_score": post_score,
            "new_mastery_status": new_mastery_status,
            "score_delta": score_delta,
            "improvement_percentage": round(score_delta * 100.0, 1),
            "outcome_assessment": outcome_assessment,
            "next_action": next_action,
            "quiz_evaluation": quiz_res,
            "scientific_disclaimer": TRACKER_CAUSAL_DISCLAIMER,
        }

    def get_student_intervention_history(
        self,
        student_id: str,
        subject: str = "Mathematics",
    ) -> Dict[str, Any]:
        """
        Retrieves complete intervention history, completion status breakdown,
        and longitudinal progress tracking.
        """
        interventions = self.db.get_student_interventions(student_id, subject)
        quiz_attempts = self.db.get_student_quiz_attempts(student_id, subject)
        mastery_records = self.db.get_student_mastery(student_id, subject)

        status_counts = {"PRESCRIBED": 0, "IN_PROGRESS": 0, "COMPLETED": 0, "CANCELLED": 0}
        for item in interventions:
            st = item.get("status", "PRESCRIBED")
            status_counts[st] = status_counts.get(st, 0) + 1

        return {
            "student_id": student_id,
            "subject": subject,
            "total_interventions": len(interventions),
            "status_breakdown": status_counts,
            "interventions": interventions,
            "quiz_attempts_count": len(quiz_attempts),
            "quiz_attempts": quiz_attempts,
            "concept_mastery_summary": mastery_records,
            "scientific_disclaimer": TRACKER_CAUSAL_DISCLAIMER,
        }

