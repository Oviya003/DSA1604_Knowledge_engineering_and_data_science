"""
ExplainAI - Module 3: Prerequisite-Aware Knowledge Gap Detective (Signature Feature 2)
Provides curriculum-guided diagnostic testing, concept mastery estimation,
and DAG prerequisite gap tracing using the Mathematics demonstration curriculum.

CURRICULUM PROVENANCE & LIMITATION:
-----------------------------------
1. Operates on the 10-concept Mathematics DEMONSTRATION_CURRICULUM.
2. Portuguese language curriculum is unsupported and returns a graceful status message.
3. Each concept currently has two demonstration questions; mastery scores serve
   as coarse indicators rather than comprehensive psychometric evaluations.
4. Unassessed concepts are classified as 'NOT_YET_ASSESSED' and are NEVER
   automatically labeled as knowledge gaps.
"""

from typing import Dict, List, Any, Optional
import datetime

from src.database.db_manager import DatabaseManager
from src.module1_data_knowledge.knowledge_graph import AcademicKnowledgeGraph
from src.module1_data_knowledge.curriculum_catalog import (
    CURRICULUM_PROVENANCE,
    is_subject_curriculum_supported,
    get_unsupported_subject_message,
)

# Configurable Concept Mastery Thresholds (0.0 to 1.0)
MASTERY_THRESHOLD_MASTERED = 0.80     # >= 80% correct
MASTERY_THRESHOLD_IN_PROGRESS = 0.50  # 50% to 79% correct
# < 50% is CRITICAL_GAP

COARSE_ASSESSMENT_LIMITATION = (
    "COARSE ASSESSMENT NOTICE: The demonstration curriculum provides two diagnostic questions "
    "per concept node. Mastery scores provide preliminary pedagogical indications and should "
    "be supplemented with comprehensive teacher assessments."
)


class KnowledgeGapDetective:
    """Manages diagnostic quizzes, concept mastery tracking, and prerequisite gap tracing."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        knowledge_graph: Optional[AcademicKnowledgeGraph] = None,
    ):
        self.db = db_manager or DatabaseManager()
        self.kg = knowledge_graph or AcademicKnowledgeGraph(subject="Mathematics")

    def is_subject_supported(self, subject: str) -> bool:
        """Checks if concept-level curriculum analysis is supported for the subject."""
        return is_subject_curriculum_supported(subject)

    def get_concept_questions(
        self,
        concept_id: str,
        subject: str = "Mathematics",
        include_answers: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieves diagnostic questions for a concept node.
        Hides correct options and explanations by default when serving quizzes.
        """
        if not self.is_subject_supported(subject):
            return {
                "supported": False,
                "subject": subject,
                "message": get_unsupported_subject_message(subject),
                "questions": [],
            }

        concept_info = self.kg.get_concept_details(concept_id)
        if not concept_info:
            raise ValueError(f"Concept '{concept_id}' not found in curriculum.")

        raw_questions = self.db.get_diagnostic_questions_by_concept(concept_id)
        sanitized_questions = []

        for q in raw_questions:
            q_dict = {
                "question_id": q["question_id"],
                "concept_id": q["concept_id"],
                "question_text": q["question_text"],
                "options": {
                    "A": q["option_a"],
                    "B": q["option_b"],
                    "C": q["option_c"],
                    "D": q["option_d"],
                },
                "blooms_level": q["blooms_level"],
            }
            if include_answers:
                q_dict["correct_option"] = q["correct_option"]
                q_dict["explanation"] = q["explanation"]
            sanitized_questions.append(q_dict)

        return {
            "supported": True,
            "subject": subject,
            "concept_id": concept_id,
            "concept_name": concept_info["name"],
            "domain": concept_info["domain"],
            "difficulty": concept_info["difficulty"],
            "question_count": len(sanitized_questions),
            "questions": sanitized_questions,
            "curriculum_provenance": CURRICULUM_PROVENANCE,
            "assessment_limitation": COARSE_ASSESSMENT_LIMITATION,
        }

    def evaluate_quiz_submission(
        self,
        student_id: str,
        concept_id: str,
        submitted_answers: Dict[str, str],
        quiz_type: str = "BASELINE_DIAGNOSTIC",
        subject: str = "Mathematics",
    ) -> Dict[str, Any]:
        """
        Validates answers, computes score, determines mastery state, and persists
        the genuine application-generated attempt (is_uci_source=0).
        """
        if not self.is_subject_supported(subject):
            return {
                "supported": False,
                "subject": subject,
                "message": get_unsupported_subject_message(subject),
            }

        raw_questions = self.db.get_diagnostic_questions_by_concept(concept_id)
        if not raw_questions:
            raise ValueError(f"No diagnostic questions found for concept '{concept_id}'.")

        total_questions = len(raw_questions)
        correct_count = 0
        detailed_evaluations = []

        for q in raw_questions:
            qid = q["question_id"]
            user_ans = str(submitted_answers.get(qid, "")).strip().upper()
            correct_ans = str(q["correct_option"]).strip().upper()
            is_correct = (user_ans == correct_ans)
            if is_correct:
                correct_count += 1

            detailed_evaluations.append({
                "question_id": qid,
                "question_text": q["question_text"],
                "student_answer": user_ans,
                "correct_answer": correct_ans,
                "is_correct": is_correct,
                "explanation": q["explanation"],
            })

        score_fraction = correct_count / total_questions if total_questions > 0 else 0.0
        percentage = round(score_fraction * 100.0, 1)

        # Classify mastery according to defined thresholds
        if score_fraction >= MASTERY_THRESHOLD_MASTERED:
            mastery_status = "MASTERED"
            mastery_achieved = 1
        elif score_fraction >= MASTERY_THRESHOLD_IN_PROGRESS:
            mastery_status = "IN_PROGRESS"
            mastery_achieved = 0
        else:
            mastery_status = "CRITICAL_GAP"
            mastery_achieved = 0

        now_timestamp = datetime.datetime.now().isoformat()

        # Persist attempt in SQLite (is_uci_source = 0)
        attempt_id = self.db.record_quiz_attempt({
            "student_id": student_id,
            "subject": subject,
            "concept_id": concept_id,
            "quiz_type": quiz_type,
            "score": round(score_fraction, 4),
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "mastery_achieved": mastery_achieved,
            "attempt_date": now_timestamp,
            "is_uci_source": 0,
        })

        # Update concept_mastery table
        self.db.upsert_concept_mastery({
            "student_id": student_id,
            "subject": subject,
            "concept_id": concept_id,
            "mastery_score": round(score_fraction, 4),
            "mastery_status": mastery_status,
            "last_assessed": now_timestamp,
        })

        return {
            "attempt_id": attempt_id,
            "student_id": student_id,
            "subject": subject,
            "concept_id": concept_id,
            "quiz_type": quiz_type,
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "score_fraction": round(score_fraction, 4),
            "percentage": percentage,
            "mastery_status": mastery_status,
            "mastery_achieved": bool(mastery_achieved),
            "attempt_date": now_timestamp,
            "question_results": detailed_evaluations,
            "coarse_indicator_notice": COARSE_ASSESSMENT_LIMITATION,
        }

    def analyze_prerequisite_gaps(
        self,
        student_id: str,
        target_concept_id: str,
        subject: str = "Mathematics",
    ) -> Dict[str, Any]:
        """
        Traverses DAG prerequisites for target_concept_id, evaluates student mastery,
        identifies knowledge gaps, and constructs a prioritized prerequisite learning sequence.
        """
        if not self.is_subject_supported(subject):
            return {
                "supported": False,
                "subject": subject,
                "message": get_unsupported_subject_message(subject),
                "target_concept": None,
                "prerequisite_path": [],
            }

        concept_info = self.kg.get_concept_details(target_concept_id)
        if not concept_info:
            raise ValueError(f"Target concept '{target_concept_id}' not found in curriculum.")

        # Get all ancestor prerequisites in topological order (foundational first)
        ancestor_ids = self.kg.get_all_ancestor_prerequisites(target_concept_id)

        # Retrieve student's current mastery records
        all_mastery = {
            m["concept_id"]: m
            for m in self.db.get_student_mastery(student_id, subject)
        }

        prerequisite_evaluations: List[Dict[str, Any]] = []
        critical_gaps: List[Dict[str, Any]] = []
        in_progress_prereqs: List[Dict[str, Any]] = []
        unassessed_prereqs: List[Dict[str, Any]] = []
        mastered_prereqs: List[Dict[str, Any]] = []

        for cid in ancestor_ids:
            c_meta = self.kg.get_concept_details(cid)
            c_name = c_meta["name"] if c_meta else cid
            c_diff = c_meta["difficulty"] if c_meta else "Intermediate"
            c_domain = c_meta["domain"] if c_meta else "Mathematics"

            if cid in all_mastery:
                m_rec = all_mastery[cid]
                m_score = float(m_rec["mastery_score"])
                m_status = m_rec["mastery_status"]
                last_dt = m_rec["last_assessed"]
            else:
                m_score = None
                # CRITICAL: Never label an unassessed concept as a gap!
                m_status = "NOT_YET_ASSESSED"
                last_dt = None

            eval_entry = {
                "concept_id": cid,
                "name": c_name,
                "difficulty": c_diff,
                "domain": c_domain,
                "mastery_score": m_score,
                "mastery_status": m_status,
                "last_assessed": last_dt,
                "direct_prerequisites": c_meta["direct_prerequisites"] if c_meta else [],
            }

            prerequisite_evaluations.append(eval_entry)

            if m_status == "CRITICAL_GAP":
                critical_gaps.append(eval_entry)
            elif m_status == "IN_PROGRESS":
                in_progress_prereqs.append(eval_entry)
            elif m_status == "NOT_YET_ASSESSED":
                unassessed_prereqs.append(eval_entry)
            elif m_status == "MASTERED":
                mastered_prereqs.append(eval_entry)

        # Assess readiness for the target concept
        # Student is considered ready only if all direct prerequisites are Mastered
        direct_prereqs = concept_info["direct_prerequisites"]
        direct_unmastered = [
            cid for cid in direct_prereqs
            if cid not in all_mastery or all_mastery[cid]["mastery_status"] != "MASTERED"
        ]

        if not direct_prereqs:
            readiness_verdict = "READY_TO_STUDY (Foundational Concept - No Prerequisites)"
        elif not direct_unmastered:
            readiness_verdict = "READY_TO_STUDY (All Direct Prerequisites Mastered)"
        elif critical_gaps:
            readiness_verdict = "NOT_READY (Foundational Knowledge Gaps Detected)"
        else:
            readiness_verdict = "PARTIALLY_READY (Prerequisites In Progress or Unassessed)"

        # Target concept's own status
        target_mastery = all_mastery.get(target_concept_id)
        target_status = target_mastery["mastery_status"] if target_mastery else "NOT_YET_ASSESSED"
        target_score = float(target_mastery["mastery_score"]) if target_mastery else None

        return {
            "supported": True,
            "student_id": student_id,
            "subject": subject,
            "target_concept": {
                "concept_id": target_concept_id,
                "name": concept_info["name"],
                "domain": concept_info["domain"],
                "difficulty": concept_info["difficulty"],
                "direct_prerequisites": direct_prereqs,
                "mastery_status": target_status,
                "mastery_score": target_score,
            },
            "readiness_verdict": readiness_verdict,
            "total_ancestor_prerequisites": len(ancestor_ids),
            "critical_gap_count": len(critical_gaps),
            "in_progress_count": len(in_progress_prereqs),
            "unassessed_count": len(unassessed_prereqs),
            "mastered_count": len(mastered_prereqs),
            "prerequisite_learning_path": prerequisite_evaluations,
            "critical_gaps": critical_gaps,
            "in_progress_prerequisites": in_progress_prereqs,
            "unassessed_prerequisites": unassessed_prereqs,
            "recommended_focus_concept": critical_gaps[0] if critical_gaps else (in_progress_prereqs[0] if in_progress_prereqs else (unassessed_prereqs[0] if unassessed_prereqs else None)),
            "curriculum_provenance": CURRICULUM_PROVENANCE,
            "coarse_assessment_notice": COARSE_ASSESSMENT_LIMITATION,
        }
