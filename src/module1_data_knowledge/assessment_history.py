"""
ExplainAI - Module 1: Assessment History and Student Record Synchronization
Transforms raw UCI student records into persistent database entities and
populates chronological term trajectories (G1, G2, G3).

DATE PROVENANCE & AUTHENTICITY NOTICE:
--------------------------------------
The authentic UCI Student Performance dataset records academic term order
via period evaluations (G1 = Term 1, G2 = Term 2, G3 = Final Grade), but does
NOT include calendar assessment dates. To maintain scientific integrity:
1. Genuine UCI assessment records have assessment_date = None.
2. date_provenance is explicitly set to 'NOT_PROVIDED_IN_UCI_SOURCE'.
3. Chronological sequencing is strictly governed by term_period (1, 2, 3).
4. Calendar dates are never fabricated or represented as authentic historical facts.
5. All genuine UCI records maintain is_uci_source = 1.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd

from src.database.db_manager import DatabaseManager
from src.config import PASSING_THRESHOLD, BINARY_TARGET_COL


class AssessmentHistoryManager:
    """Manages student record ingestion, assessment record creation, and database persistence."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    @staticmethod
    def generate_student_records(df: pd.DataFrame, subject: str = "Mathematics") -> List[Dict[str, Any]]:
        """
        Converts a validated raw DataFrame into strongly typed student enrollment dictionaries
        with unambiguous prefixed identifiers (e.g. MAT_001, POR_001).
        """
        prefix = "MAT" if subject == "Mathematics" else "POR"
        source_dataset = f"UCI_{subject.upper()}"
        records: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            student_id = f"{prefix}_{idx + 1:03d}"
            g1 = float(row["G1"])
            g2 = float(row["G2"])
            g3 = float(row["G3"])
            is_at_risk = 1 if g3 < PASSING_THRESHOLD else 0

            rec = {
                "student_id": student_id,
                "subject": subject,
                "source_dataset": source_dataset,
                "school": str(row["school"]),
                "sex": str(row["sex"]),
                "age": int(row["age"]),
                "address": str(row["address"]),
                "famsize": str(row["famsize"]),
                "Pstatus": str(row["Pstatus"]),
                "Medu": int(row["Medu"]),
                "Fedu": int(row["Fedu"]),
                "Mjob": str(row["Mjob"]),
                "Fjob": str(row["Fjob"]),
                "reason": str(row["reason"]),
                "guardian": str(row["guardian"]),
                "traveltime": int(row["traveltime"]),
                "studytime": int(row["studytime"]),
                "failures": int(row["failures"]),
                "schoolsup": str(row["schoolsup"]),
                "famsup": str(row["famsup"]),
                "paid": str(row["paid"]),
                "activities": str(row["activities"]),
                "nursery": str(row["nursery"]),
                "higher": str(row["higher"]),
                "internet": str(row["internet"]),
                "romantic": str(row["romantic"]),
                "famrel": int(row["famrel"]),
                "freetime": int(row["freetime"]),
                "goout": int(row["goout"]),
                "Dalc": int(row["Dalc"]),
                "Walc": int(row["Walc"]),
                "health": int(row["health"]),
                "absences": int(row["absences"]),
                "G1_raw": g1,
                "G2_raw": g2,
                "G3_raw": g3,
                "is_at_risk": is_at_risk,
            }
            records.append(rec)
        return records

    @staticmethod
    def generate_uci_assessment_records(df: pd.DataFrame, subject: str = "Mathematics") -> List[Dict[str, Any]]:
        """
        Builds chronological assessment trajectory records for Term 1 (G1), Term 2 (G2),
        and Final Examination (G3), tagged with is_uci_source=1 and
        date_provenance='NOT_PROVIDED_IN_UCI_SOURCE'.

        Calendar dates are left as None because the source dataset does not contain dates.
        """
        prefix = "MAT" if subject == "Mathematics" else "POR"
        assessments: List[Dict[str, Any]] = []

        term_configs = [
            {"term": 1, "name": "Term 1 Baseline (G1)", "col": "G1"},
            {"term": 2, "name": "Term 2 Midterm (G2)", "col": "G2"},
            {"term": 3, "name": "Final Examination (G3)", "col": "G3"},
        ]

        for idx, row in df.iterrows():
            student_id = f"{prefix}_{idx + 1:03d}"
            for tc in term_configs:
                score = float(row[tc["col"]])
                pct = round((score / 20.0) * 100.0, 2)
                assessments.append(
                    {
                        "student_id": student_id,
                        "subject": subject,
                        "assessment_name": tc["name"],
                        "term_period": tc["term"],
                        "score": score,
                        "max_score": 20.0,
                        "percentage": pct,
                        "assessment_date": None,  # Explicitly None: no calendar date in UCI source
                        "date_provenance": "NOT_PROVIDED_IN_UCI_SOURCE",
                        "is_uci_source": 1,
                        "notes": f"Authentic {subject} {tc['name']} evaluation from UCI dataset.",
                    }
                )
        return assessments

    def sync_dataset_to_database(self, df: pd.DataFrame, subject: str = "Mathematics") -> Dict[str, int]:
        """
        Ingests students and assessment history into SQLite in atomic batches.
        Returns the counts of inserted students and assessments.
        """
        student_records = self.generate_student_records(df, subject)
        assessment_records = self.generate_uci_assessment_records(df, subject)

        num_students = self.db.insert_students_batch(student_records)
        num_assessments = self.db.insert_assessments_batch(assessment_records)

        return {
            "subject": subject,
            "students_synced": num_students,
            "assessments_synced": num_assessments,
        }
