"""
ExplainAI - Module 2: Academic Risk Predictor Service
Loads trained model pipelines and provides real-time academic risk inference,
at-risk probability estimation, and persistence in SQLite.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import datetime
import joblib
import pandas as pd
import numpy as np

from src.config import (
    MODELS_DIR,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    SUPPORTED_SUBJECTS,
)
from src.database.db_manager import DatabaseManager


class AcademicRiskPredictor:
    """Provides inference services for academic risk prediction using trained models."""

    def __init__(self, models_dir: Optional[Path] = None, db_manager: Optional[DatabaseManager] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.db = db_manager or DatabaseManager()
        self._loaded_models: Dict[str, Any] = {}

    def _get_model_key(self, subject: str, experiment: str) -> str:
        return f"{subject.lower()}_{experiment.lower()}_best"

    def load_model(self, subject: str, experiment: str) -> Any:
        """Loads a persisted model pipeline from disk with in-memory caching."""
        key = self._get_model_key(subject, experiment)
        if key in self._loaded_models:
            return self._loaded_models[key]

        model_path = self.models_dir / f"{key}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at {model_path}. "
                f"Please run train_models.py to train and serialize models."
            )

        pipeline = joblib.load(model_path)
        self._loaded_models[key] = pipeline
        return pipeline

    def predict_student_risk(
        self,
        student_data: Dict[str, Any],
        subject: str = "Mathematics",
        experiment: str = "A",
        persist: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes model inference for a student record.

        Input Validation & Anti-Leakage:
        - If experiment == 'A': uses exactly the 29 Model A features.
        - If experiment == 'B': uses exactly the 30 Model B features (including G1).
        - G2 and G3 are strictly ignored even if present in student_data.

        Returns:
            Dictionary containing predicted risk category, at-risk probability,
            model algorithm, version, and metadata.
        """
        subj = subject.capitalize()
        exp = experiment.upper()
        if subj not in SUPPORTED_SUBJECTS:
            raise ValueError(f"Subject '{subject}' is not supported. Must be one of {SUPPORTED_SUBJECTS}.")
        if exp not in ["A", "B"]:
            raise ValueError(f"Experiment '{experiment}' is invalid. Must be 'A' or 'B'.")

        pipeline = self.load_model(subj, exp)
        required_features = FEATURES_MODEL_A if exp == "A" else FEATURES_MODEL_B

        # Defensive feature resolution: map G1 from G1_raw or SQLite assessments if not explicitly under 'G1'
        input_data = dict(student_data)
        if exp == "B" and "G1" not in input_data:
            if "G1_raw" in input_data:
                input_data["G1"] = input_data["G1_raw"]
            elif "student_id" in input_data and input_data["student_id"] != "ANONYMOUS":
                try:
                    assessments = self.db.get_student_assessments(input_data["student_id"], subj)
                    t1 = [a for a in assessments if a.get("term_period") == 1]
                    if t1:
                        input_data["G1"] = float(t1[0]["score"])
                except Exception:
                    pass

        # Extract only the required feature subset into a single-row DataFrame
        missing_feats = [f for f in required_features if f not in input_data]
        if missing_feats:
            raise ValueError(f"Student data is missing required features for Model {exp}: {missing_feats}")

        row_dict = {f: [input_data[f]] for f in required_features}
        X_input = pd.DataFrame(row_dict)

        # Execute prediction and probability estimation
        prediction = int(pipeline.predict(X_input)[0])
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(X_input)[0]
            # Probability of positive class (At-Risk = 1)
            at_risk_prob = round(float(probabilities[1]), 4)
        else:
            at_risk_prob = 1.0 if prediction == 1 else 0.0

        risk_category = "At-Risk" if prediction == 1 else "Satisfactory"

        # Determine algorithm name from pipeline
        algo_name = pipeline.named_steps["classifier"].__class__.__name__

        timestamp = datetime.datetime.now().isoformat()
        student_id = student_data.get("student_id", "ANONYMOUS")

        result = {
            "student_id": student_id,
            "subject": subj,
            "model_experiment": f"MODEL_{exp}",
            "model_algorithm": algo_name,
            "risk_label": prediction,
            "risk_category": risk_category,
            "at_risk_probability": at_risk_prob,
            "satisfactory_probability": round(1.0 - at_risk_prob, 4),
            "feature_count_used": len(required_features),
            "model_version": "1.0.0",
            "prediction_timestamp": timestamp,
        }

        # Persist prediction record in SQLite if requested and student_id is known
        if persist and student_id != "ANONYMOUS":
            try:
                self.db.insert_prediction_record(
                    {
                        "student_id": student_id,
                        "subject": subj,
                        "model_experiment": f"MODEL_{exp}",
                        "model_algorithm": algo_name,
                        "risk_category": risk_category,
                        "at_risk_probability": at_risk_prob,
                    }
                )
            except Exception:
                pass  # Gracefully proceed if persistence fails (e.g. unregistered temporary ID)

        return result

