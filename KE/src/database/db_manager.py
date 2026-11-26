"""
ExplainAI - Database Operations Manager
Handles SQLite database lifecycle, schema provisioning, CRUD operations,
multi-subject isolation, transactional integrity, and rollback handling.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config import DATABASE_PATH, CURRICULUM_SUPPORTED_SUBJECTS
from src.database.schema import SCHEMA_SQL
from src.module1_data_knowledge.curriculum_catalog import (
    MATHEMATICS_CURRICULUM,
    CURRICULUM_PROVENANCE,
    is_subject_curriculum_supported,
    get_unsupported_subject_message,
)


class DatabaseManager:
    """Manages SQLite database lifecycle and operations for ExplainAI."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """Returns a database connection with foreign key support enabled and row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_database(self) -> None:
        """Initializes database schema and populates default curriculum metadata."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            self._seed_curriculum(conn)
            conn.commit()

    def _seed_curriculum(self, conn: sqlite3.Connection) -> None:
        """Seeds the standard mathematics curriculum, prerequisites, and questions if empty."""
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM concepts")
        count = cursor.fetchone()[0]
        if count > 0:
            return  # Already seeded

        for concept in MATHEMATICS_CURRICULUM:
            cursor.execute(
                """
                INSERT OR IGNORE INTO concepts
                (concept_id, subject, name, domain, difficulty, description, curriculum_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    concept["concept_id"],
                    concept.get("subject", "Mathematics"),
                    concept["name"],
                    concept["domain"],
                    concept["difficulty"],
                    concept["description"],
                    concept.get("curriculum_type", CURRICULUM_PROVENANCE),
                ),
            )

            # Insert prerequisites
            for prereq_id in concept["prerequisites"]:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO concept_prerequisites (concept_id, prerequisite_id)
                    VALUES (?, ?)
                    """,
                    (concept["concept_id"], prereq_id),
                )

            # Insert diagnostic questions
            for q in concept.get("diagnostic_questions", []):
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO diagnostic_questions
                    (question_id, concept_id, question_text, option_a, option_b, option_c, option_d, correct_option, explanation, blooms_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        q["question_id"],
                        concept["concept_id"],
                        q["question_text"],
                        q["options"].get("A", ""),
                        q["options"].get("B", ""),
                        q["options"].get("C", ""),
                        q["options"].get("D", ""),
                        q["correct_option"],
                        q["explanation"],
                        q["blooms_level"],
                    ),
                )

    def insert_student(self, student_data: Dict[str, Any]) -> None:
        """Inserts a single student record with transactional safety."""
        sql = """
        INSERT OR REPLACE INTO students (
            student_id, subject, source_dataset, school, sex, age, address, famsize, Pstatus,
            Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures,
            schoolsup, famsup, paid, activities, nursery, higher, internet, romantic,
            famrel, freetime, goout, Dalc, Walc, health, absences, G1_raw, G2_raw, G3_raw, is_at_risk
        ) VALUES (
            :student_id, :subject, :source_dataset, :school, :sex, :age, :address, :famsize, :Pstatus,
            :Medu, :Fedu, :Mjob, :Fjob, :reason, :guardian, :traveltime, :studytime, :failures,
            :schoolsup, :famsup, :paid, :activities, :nursery, :higher, :internet, :romantic,
            :famrel, :freetime, :goout, :Dalc, :Walc, :health, :absences, :G1_raw, :G2_raw, :G3_raw, :is_at_risk
        )
        """
        with self.get_connection() as conn:
            conn.execute(sql, student_data)
            conn.commit()

    def insert_students_batch(self, students: List[Dict[str, Any]]) -> int:
        """Inserts multiple student records in a single transaction with automatic rollback."""
        sql = """
        INSERT OR REPLACE INTO students (
            student_id, subject, source_dataset, school, sex, age, address, famsize, Pstatus,
            Medu, Fedu, Mjob, Fjob, reason, guardian, traveltime, studytime, failures,
            schoolsup, famsup, paid, activities, nursery, higher, internet, romantic,
            famrel, freetime, goout, Dalc, Walc, health, absences, G1_raw, G2_raw, G3_raw, is_at_risk
        ) VALUES (
            :student_id, :subject, :source_dataset, :school, :sex, :age, :address, :famsize, :Pstatus,
            :Medu, :Fedu, :Mjob, :Fjob, :reason, :guardian, :traveltime, :studytime, :failures,
            :schoolsup, :famsup, :paid, :activities, :nursery, :higher, :internet, :romantic,
            :famrel, :freetime, :goout, :Dalc, :Walc, :health, :absences, :G1_raw, :G2_raw, :G3_raw, :is_at_risk
        )
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, students)
            conn.commit()
            return cursor.rowcount

    def insert_assessment(self, assessment_data: Dict[str, Any]) -> int:
        """Inserts a single assessment record."""
        sql = """
        INSERT INTO assessments (
            student_id, subject, assessment_name, term_period, score, max_score, percentage, assessment_date, date_provenance, is_uci_source, notes
        ) VALUES (
            :student_id, :subject, :assessment_name, :term_period, :score, :max_score, :percentage, :assessment_date, :date_provenance, :is_uci_source, :notes
        )
        """
        data = dict(assessment_data)
        if "date_provenance" not in data:
            data["date_provenance"] = "NOT_PROVIDED_IN_UCI_SOURCE" if data.get("is_uci_source") == 1 else "ESTIMATED_DEMONSTRATION"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
            return cursor.lastrowid

    def insert_assessments_batch(self, assessments: List[Dict[str, Any]]) -> int:
        """Inserts multiple assessment records in a single transaction."""
        sql = """
        INSERT INTO assessments (
            student_id, subject, assessment_name, term_period, score, max_score, percentage, assessment_date, date_provenance, is_uci_source, notes
        ) VALUES (
            :student_id, :subject, :assessment_name, :term_period, :score, :max_score, :percentage, :assessment_date, :date_provenance, :is_uci_source, :notes
        )
        """
        prepared_list = []
        for a in assessments:
            rec = dict(a)
            if "date_provenance" not in rec:
                rec["date_provenance"] = "NOT_PROVIDED_IN_UCI_SOURCE" if rec.get("is_uci_source") == 1 else "ESTIMATED_DEMONSTRATION"
            prepared_list.append(rec)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, prepared_list)
            conn.commit()
            return cursor.rowcount

    def get_student_count(self, subject: Optional[str] = None) -> int:
        """Returns the number of student records, optionally filtered by subject."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if subject:
                cursor.execute("SELECT COUNT(*) FROM students WHERE subject = ?", (subject,))
            else:
                cursor.execute("SELECT COUNT(*) FROM students")
            return cursor.fetchone()[0]

    def get_all_students(self, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all students, optionally filtered by subject."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if subject:
                cursor.execute("SELECT * FROM students WHERE subject = ? ORDER BY student_id ASC", (subject,))
            else:
                cursor.execute("SELECT * FROM students ORDER BY subject ASC, student_id ASC")
            students = []
            for row in cursor.fetchall():
                s_dict = dict(row)
                if "G1" not in s_dict and "G1_raw" in s_dict:
                    s_dict["G1"] = s_dict["G1_raw"]
                students.append(s_dict)
            return students

    def get_assessment_count(self, subject: Optional[str] = None, is_uci_source: Optional[int] = None) -> int:
        """Returns the count of assessments according to subject and source criteria."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM assessments WHERE 1=1"
            params: List[Any] = []
            if subject:
                query += " AND subject = ?"
                params.append(subject)
            if is_uci_source is not None:
                query += " AND is_uci_source = ?"
                params.append(is_uci_source)
            cursor.execute(query, params)
            return cursor.fetchone()[0]

    def get_student_by_id_and_subject(self, student_id: str, subject: str) -> Optional[Dict[str, Any]]:
        """Fetches a single student enrollment record by composite key (student_id, subject)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM students WHERE student_id = ? AND subject = ?",
                (student_id, subject),
            )
            row = cursor.fetchone()
            if not row:
                return None
            student_dict = dict(row)
            if "G1" not in student_dict and "G1_raw" in student_dict:
                student_dict["G1"] = student_dict["G1_raw"]
            return student_dict

    def get_all_enrollments_for_student(self, student_id: str) -> List[Dict[str, Any]]:
        """Retrieves all course enrollments (e.g. Mathematics, Portuguese) for a given student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM students WHERE student_id = ? ORDER BY subject ASC",
                (student_id,),
            )
            enrollments = []
            for row in cursor.fetchall():
                e_dict = dict(row)
                if "G1" not in e_dict and "G1_raw" in e_dict:
                    e_dict["G1"] = e_dict["G1_raw"]
                enrollments.append(e_dict)
            return enrollments

    def get_student_assessments(self, student_id: str, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves assessments for a student, optionally filtered by subject, ordered chronologically."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if subject:
                cursor.execute(
                    """
                    SELECT * FROM assessments
                    WHERE student_id = ? AND subject = ?
                    ORDER BY term_period ASC, assessment_date ASC
                    """,
                    (student_id, subject),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM assessments
                    WHERE student_id = ?
                    ORDER BY term_period ASC, assessment_date ASC
                    """,
                    (student_id,),
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_concepts(self, subject: str = "Mathematics") -> List[Dict[str, Any]]:
        """Retrieves all curriculum concepts for a given subject."""
        if not is_subject_curriculum_supported(subject):
            return []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM concepts WHERE subject = ? ORDER BY concept_id ASC",
                (subject,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_concept_prerequisites(self) -> List[Dict[str, str]]:
        """Retrieves all directed prerequisite edges (concept_id, prerequisite_id)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT concept_id, prerequisite_id FROM concept_prerequisites")
            return [dict(row) for row in cursor.fetchall()]

    def get_diagnostic_questions_by_concept(self, concept_id: str) -> List[Dict[str, Any]]:
        """Retrieves diagnostic questions associated with a given concept."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM diagnostic_questions WHERE concept_id = ?", (concept_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_all_diagnostic_questions(self, subject: str = "Mathematics") -> List[Dict[str, Any]]:
        """Retrieves all diagnostic questions for a subject."""
        if not is_subject_curriculum_supported(subject):
            return []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT dq.*, c.name as concept_name, c.domain, c.difficulty
                FROM diagnostic_questions dq
                JOIN concepts c ON dq.concept_id = c.concept_id
                WHERE c.subject = ?
                ORDER BY dq.concept_id ASC, dq.question_id ASC
                """,
                (subject,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_student_mastery(self, student_id: str, subject: str = "Mathematics") -> List[Dict[str, Any]]:
        """Retrieves all concept mastery scores for a student in a specific subject."""
        if not is_subject_curriculum_supported(subject):
            return []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT cm.*, c.name as concept_name, c.domain, c.difficulty
                FROM concept_mastery cm
                JOIN concepts c ON cm.concept_id = c.concept_id
                WHERE cm.student_id = ? AND cm.subject = ?
                ORDER BY cm.concept_id ASC
                """,
                (student_id, subject),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_student_interventions(self, student_id: str, subject: str = "Mathematics") -> List[Dict[str, Any]]:
        """Retrieves all interventions for a student in a specific subject."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT i.*, c.name as concept_name
                FROM interventions i
                JOIN concepts c ON i.target_concept_id = c.concept_id
                WHERE i.student_id = ? AND i.subject = ?
                ORDER BY i.prescribed_date DESC
                """,
                (student_id, subject),
            )
            return [dict(row) for row in cursor.fetchall()]

    def insert_prediction_record(self, data: Dict[str, Any]) -> int:
        """Stores a model inference risk prediction record."""
        sql = """
        INSERT INTO prediction_records (
            student_id, subject, model_experiment, model_algorithm, risk_category, at_risk_probability
        ) VALUES (
            :student_id, :subject, :model_experiment, :model_algorithm, :risk_category, :at_risk_probability
        )
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
            return cursor.lastrowid

    def get_student_predictions(self, student_id: str, subject: str) -> List[Dict[str, Any]]:
        """Retrieves past prediction records for a specific student and subject."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM prediction_records
                WHERE student_id = ? AND subject = ?
                ORDER BY prediction_timestamp DESC
                """,
                (student_id, subject),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_audit_logs(
        self,
        student_id: Optional[str] = None,
        action_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves audit log entries filtered by student_id and/or action_type."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params: List[Any] = []
            if student_id:
                query += " AND student_id = ?"
                params.append(student_id)
            if action_type:
                query += " AND action_type = ?"
                params.append(action_type)
            query += " ORDER BY timestamp DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def log_audit_action(
        self,
        action_type: str,
        details: Dict[str, Any],
        student_id: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> int:
        """Records an action in the audit log for explainability and accountability."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_logs (student_id, subject, action_type, details)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, subject, action_type, json.dumps(details)),
            )
            conn.commit()
            return cursor.lastrowid

    # ==============================================================================
    # Module 3 Operations: Quizzes, Mastery, Recommendations, Interventions
    # ==============================================================================
    def record_quiz_attempt(self, attempt_data: Dict[str, Any]) -> int:
        """
        Stores a genuine application-generated diagnostic quiz attempt (is_uci_source=0).
        """
        sql = """
        INSERT INTO diagnostic_quiz_attempts (
            student_id, subject, concept_id, quiz_type, score,
            total_questions, correct_answers, mastery_achieved, attempt_date, is_uci_source
        ) VALUES (
            :student_id, :subject, :concept_id, :quiz_type, :score,
            :total_questions, :correct_answers, :mastery_achieved, :attempt_date, 0
        )
        """
        data = dict(attempt_data)
        if "subject" not in data:
            data["subject"] = "Mathematics"
        if "is_uci_source" not in data:
            data["is_uci_source"] = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
            return cursor.lastrowid

    def get_student_quiz_attempts(
        self,
        student_id: str,
        subject: str = "Mathematics",
        concept_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves diagnostic quiz attempts for a student, optionally filtered by concept."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if concept_id:
                cursor.execute(
                    """
                    SELECT dqa.*, c.name as concept_name
                    FROM diagnostic_quiz_attempts dqa
                    JOIN concepts c ON dqa.concept_id = c.concept_id
                    WHERE dqa.student_id = ? AND dqa.subject = ? AND dqa.concept_id = ?
                    ORDER BY dqa.attempt_date DESC, dqa.attempt_id DESC
                    """,
                    (student_id, subject, concept_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT dqa.*, c.name as concept_name
                    FROM diagnostic_quiz_attempts dqa
                    JOIN concepts c ON dqa.concept_id = c.concept_id
                    WHERE dqa.student_id = ? AND dqa.subject = ?
                    ORDER BY dqa.attempt_date DESC, dqa.attempt_id DESC
                    """,
                    (student_id, subject),
                )
            return [dict(row) for row in cursor.fetchall()]

    def upsert_concept_mastery(self, mastery_data: Dict[str, Any]) -> int:
        """
        Inserts or updates the student's mastery status for a concept.
        Enforces UNIQUE(student_id, subject, concept_id) constraint.
        """
        sql = """
        INSERT INTO concept_mastery (
            student_id, subject, concept_id, mastery_score, mastery_status, last_assessed
        ) VALUES (
            :student_id, :subject, :concept_id, :mastery_score, :mastery_status, :last_assessed
        )
        ON CONFLICT(student_id, subject, concept_id) DO UPDATE SET
            mastery_score = excluded.mastery_score,
            mastery_status = excluded.mastery_status,
            last_assessed = excluded.last_assessed
        """
        data = dict(mastery_data)
        if "subject" not in data:
            data["subject"] = "Mathematics"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
            return cursor.lastrowid

    def get_concept_mastery_record(
        self,
        student_id: str,
        subject: str,
        concept_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single concept mastery record for a student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM concept_mastery
                WHERE student_id = ? AND subject = ? AND concept_id = ?
                """,
                (student_id, subject, concept_id),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def insert_learning_recommendation(self, rec_data: Dict[str, Any]) -> int:
        """Stores a personalized learning recommendation."""
        sql = """
        INSERT INTO learning_recommendations (
            student_id, subject, priority, recommendation_text, rationale, status
        ) VALUES (
            :student_id, :subject, :priority, :recommendation_text, :rationale, :status
        )
        """
        data = dict(rec_data)
        if "status" not in data:
            data["status"] = "ACTIVE"
        if "subject" not in data:
            data["subject"] = "Mathematics"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
            return cursor.lastrowid

    def get_student_recommendations(
        self,
        student_id: str,
        subject: str = "Mathematics",
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves learning recommendations for a student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    """
                    SELECT * FROM learning_recommendations
                    WHERE student_id = ? AND subject = ? AND status = ?
                    ORDER BY created_at DESC
                    """,
                    (student_id, subject, status),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM learning_recommendations
                    WHERE student_id = ? AND subject = ?
                    ORDER BY created_at DESC
                    """,
                    (student_id, subject),
                )
            return [dict(row) for row in cursor.fetchall()]

    def update_recommendation_status(self, recommendation_id: int, new_status: str) -> bool:
        """Updates the status of a recommendation (e.g. 'ACKNOWLEDGED', 'COMPLETED')."""
        sql = "UPDATE learning_recommendations SET status = ? WHERE recommendation_id = ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, (new_status, recommendation_id))
            conn.commit()
            return cursor.rowcount > 0

    def insert_intervention(self, intervention_data: Dict[str, Any]) -> int:
        """Stores an educational intervention prescription."""
        sql = """
        INSERT INTO interventions (
            student_id, subject, target_concept_id, intervention_type, prescribed_date,
            completed_date, status, prescribed_by, notes, post_quiz_score
        ) VALUES (
            :student_id, :subject, :target_concept_id, :intervention_type, :prescribed_date,
            :completed_date, :status, :prescribed_by, :notes, :post_quiz_score
        )
        """
        data = dict(intervention_data)
        if "status" not in data:
            data["status"] = "PRESCRIBED"
        if "prescribed_by" not in data:
            data["prescribed_by"] = "ExplainAI Support Engine"
        if "completed_date" not in data:
            data["completed_date"] = None
        if "post_quiz_score" not in data:
            data["post_quiz_score"] = None
        if "notes" not in data:
            data["notes"] = ""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, data)
            conn.commit()
            return cursor.lastrowid

    def update_intervention_status(
        self,
        intervention_id: int,
        new_status: str,
        post_quiz_score: Optional[float] = None,
        completed_date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Updates an intervention's lifecycle state, completion timestamp, and post-quiz outcome."""
        updates = ["status = ?"]
        params: List[Any] = [new_status]
        if completed_date is not None:
            updates.append("completed_date = ?")
            params.append(completed_date)
        if post_quiz_score is not None:
            updates.append("post_quiz_score = ?")
            params.append(post_quiz_score)
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        params.append(intervention_id)

        sql = f"UPDATE interventions SET {', '.join(updates)} WHERE intervention_id = ?"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor.rowcount > 0

    def get_intervention_by_id(self, intervention_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single intervention by its primary key ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT i.*, c.name as concept_name
                FROM interventions i
                JOIN concepts c ON i.target_concept_id = c.concept_id
                WHERE i.intervention_id = ?
                """,
                (intervention_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
