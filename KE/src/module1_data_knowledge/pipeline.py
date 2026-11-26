"""
ExplainAI - Module 1: Functional Workflow Pipeline Orchestrator
Coordinates the end-to-end execution of Module 1:
Raw CSV Ingestion -> Validation -> Feature Preparation -> Knowledge Graph -> Diagnostic Questions -> SQLite Sync -> Validation Report.
"""

from typing import Dict, Any, Optional
import pandas as pd

from src.database.db_manager import DatabaseManager
from src.module1_data_knowledge.ingestion import DatasetIngestion
from src.module1_data_knowledge.preprocessing import DataPreprocessor
from src.module1_data_knowledge.knowledge_graph import AcademicKnowledgeGraph
from src.module1_data_knowledge.assessment_history import AssessmentHistoryManager


def run_module1_pipeline(db_manager: Optional[DatabaseManager] = None) -> Dict[str, Any]:
    """
    Executes the complete Module 1 functional workflow:
    1. Dataset Ingestion: loads and inspects student-mat.csv and student-por.csv.
    2. Data Validation: checks schema, types, missing values, duplicates, and ranges.
    3. Feature Preprocessing: verifies Model A (29 features) and Model B (30 features) anti-leakage matrices.
    4. Knowledge Graph: constructs Mathematics DAG (10 concepts, 11 edges) and validates topological sort.
    5. Diagnostic Questions: verifies 20 demonstration questions.
    6. SQLite Storage: persists student profiles and chronological assessments with complete subject isolation.
    7. Returns comprehensive workflow execution report.
    """
    db = db_manager or DatabaseManager()
    db.init_database()

    ingestion = DatasetIngestion()
    preprocessor = DataPreprocessor()
    kg_math = AcademicKnowledgeGraph("Mathematics")
    history_mgr = AssessmentHistoryManager(db_manager=db)

    pipeline_report: Dict[str, Any] = {
        "status": "IN_PROGRESS",
        "ingestion": {},
        "preprocessing": {},
        "knowledge_graph": {},
        "diagnostic_questions": {},
        "database_sync": {},
    }

    # Step 1 & 2: Ingestion & Validation
    math_df, math_val = ingestion.load_and_validate_subject("Mathematics")
    por_df, por_val = ingestion.load_and_validate_subject("Portuguese")

    pipeline_report["ingestion"]["Mathematics"] = math_val
    pipeline_report["ingestion"]["Portuguese"] = por_val

    # Step 3: Feature Preprocessing & Anti-Leakage Verification
    X_mat_a, y_mat, meta_mat = preprocessor.extract_features_and_target(math_df, model_experiment="A")
    X_mat_b, _, _ = preprocessor.extract_features_and_target(math_df, model_experiment="B")

    X_por_a, y_por, meta_por = preprocessor.extract_features_and_target(por_df, model_experiment="A")
    X_por_b, _, _ = preprocessor.extract_features_and_target(por_df, model_experiment="B")

    pipeline_report["preprocessing"]["Mathematics"] = {
        "model_a_feature_count": len(X_mat_a.columns),
        "model_b_feature_count": len(X_mat_b.columns),
        "target_distribution": {
            "at_risk_count": int(y_mat.sum()),
            "passing_count": int((1 - y_mat).sum()),
            "at_risk_rate": round(float(y_mat.mean()), 4),
        },
        "absences_in_features": "absences" in X_mat_a.columns or "absences" in X_mat_b.columns,
        "g2_g3_in_features": any(c in X_mat_a.columns or c in X_mat_b.columns for c in ["G2", "G3", "is_at_risk"]),
    }

    pipeline_report["preprocessing"]["Portuguese"] = {
        "model_a_feature_count": len(X_por_a.columns),
        "model_b_feature_count": len(X_por_b.columns),
        "target_distribution": {
            "at_risk_count": int(y_por.sum()),
            "passing_count": int((1 - y_por).sum()),
            "at_risk_rate": round(float(y_por.mean()), 4),
        },
        "absences_in_features": "absences" in X_por_a.columns or "absences" in X_por_b.columns,
        "g2_g3_in_features": any(c in X_por_a.columns or c in X_por_b.columns for c in ["G2", "G3", "is_at_risk"]),
    }

    # Step 4: Knowledge Graph Validation
    kg_status = kg_math.get_status()
    pipeline_report["knowledge_graph"] = {
        "subject": "Mathematics",
        "supported": kg_status["supported"],
        "curriculum_provenance": kg_status.get("curriculum_provenance"),
        "num_concepts": kg_status.get("num_concepts"),
        "num_prerequisite_edges": kg_status.get("num_prerequisite_edges"),
        "is_acyclic": kg_status.get("is_acyclic"),
        "topological_order": kg_math.get_topological_order(),
    }

    # Step 5: Diagnostic Questions
    diag_questions = db.get_all_diagnostic_questions(subject="Mathematics")
    pipeline_report["diagnostic_questions"] = {
        "subject": "Mathematics",
        "total_questions": len(diag_questions),
        "all_linked_to_concepts": all(q["concept_name"] is not None for q in diag_questions),
    }

    # Step 6: SQLite Database Sync
    sync_mat = history_mgr.sync_dataset_to_database(math_df, subject="Mathematics")
    sync_por = history_mgr.sync_dataset_to_database(por_df, subject="Portuguese")

    pipeline_report["database_sync"]["Mathematics"] = sync_mat
    pipeline_report["database_sync"]["Portuguese"] = sync_por
    pipeline_report["database_sync"]["total_students_in_db"] = db.get_student_count()
    pipeline_report["database_sync"]["total_assessments_in_db"] = db.get_assessment_count()

    # Final overall status
    pipeline_report["status"] = "COMPLETED_SUCCESSFULLY"
    return pipeline_report

