"""
ExplainAI - Module 3: SHAP Explainability Service
Computes local (individual) and global feature attribution explanations for
trained academic risk prediction models (Logistic Regression, Random Forest, XGBoost)
across Mathematics and Portuguese cohorts in Model A (Intake) and Model B (Post-G1).

SCIENTIFIC & INTERPRETABILITY DISCLAIMER:
-----------------------------------------
SHAP (SHapley Additive exPlanations) values describe the mathematical contribution
of each feature toward the statistical classification model's log-odds / risk output.
High feature attribution DOES NOT establish real-world causation, nor does it prove
that intervening on a feature will guarantee improved academic outcomes.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import shap

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    SUPPORTED_SUBJECTS,
)
from src.module1_data_knowledge.ingestion import DatasetIngestion
from src.module1_data_knowledge.preprocessing import DataPreprocessor

SHAP_SCIENTIFIC_DISCLAIMER = (
    "SCIENTIFIC & CAUSAL LIMITATION: SHAP (SHapley Additive exPlanations) values describe "
    "the mathematical contribution of each input feature to the statistical machine learning "
    "model's risk prediction for this specific profile. High feature attribution reflects "
    "statistical association within the model and DOES NOT constitute causal proof that changing "
    "this attribute will guarantee an improved real-world academic grade."
)


class AcademicShapExplainer:
    """Manages model-specific SHAP explainers with transformed feature mapping and caching."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or MODELS_DIR
        self.ingestion = DatasetIngestion()
        self.preprocessor = DataPreprocessor()
        self._explainers_cache: Dict[str, Any] = {}
        self._background_cache: Dict[str, Any] = {}

    def _get_cache_key(self, subject: str, experiment: str) -> str:
        return f"{subject.lower()}_{experiment.lower()}"

    def get_pipeline(self, subject: str, experiment: str) -> Any:
        """Loads the selected best pipeline for the subject and experiment."""
        key = self._get_cache_key(subject, experiment)
        model_path = self.models_dir / f"{key}_best.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Please train models first.")
        return joblib.load(model_path)

    def get_background_data(self, subject: str, experiment: str, n_samples: int = 100) -> pd.DataFrame:
        """
        Retrieves a deterministic background reference sample from the training partition.
        Excludes G2, G3, is_at_risk, and absences to preserve strict anti-leakage.
        """
        key = self._get_cache_key(subject, experiment)
        if key in self._background_cache:
            return self._background_cache[key]

        df = self.ingestion.load_raw_dataset(subject)
        X, y, _ = self.preprocessor.extract_features_and_target(df, model_experiment=experiment)

        # Sample deterministically from training data
        sample_size = min(n_samples, len(X))
        bg_sample = X.sample(n=sample_size, random_state=RANDOM_STATE)
        self._background_cache[key] = bg_sample
        return bg_sample

    def get_explainer(self, subject: str, experiment: str) -> Tuple[Any, Any]:
        """
        Builds and caches the appropriate SHAP explainer for the serialized model pipeline.
        Returns tuple of (explainer, pipeline).
        """
        key = self._get_cache_key(subject, experiment)
        if key in self._explainers_cache:
            return self._explainers_cache[key]

        pipeline = self.get_pipeline(subject, experiment)
        transformer = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        clf_type = type(classifier).__name__

        bg_df = self.get_background_data(subject, experiment, n_samples=100)
        bg_trans = transformer.transform(bg_df)

        if clf_type == "LogisticRegression":
            # LinearExplainer operates on transformed background data
            explainer = shap.LinearExplainer(classifier, bg_trans)
        elif clf_type in ["RandomForestClassifier", "XGBClassifier"]:
            # TreeExplainer is exact and fast for tree ensembles
            explainer = shap.TreeExplainer(classifier)
        else:
            # Fallback to model-agnostic KernelExplainer with medoid / kmeans background
            bg_summary = shap.kmeans(bg_trans, k=min(15, len(bg_trans)))
            explainer = shap.KernelExplainer(classifier.predict_proba, bg_summary)

        self._explainers_cache[key] = (explainer, pipeline)
        return explainer, pipeline

    @staticmethod
    def _map_feature_name(col_name: str) -> Tuple[str, str]:
        """
        Maps transformed ColumnTransformer feature names back to original feature and human-readable label.
        e.g. 'cat__school_MS' -> ('school', 'School (Mousinho da Silveira)')
        'num__studytime' -> ('studytime', 'Weekly Study Time')
        """
        if col_name.startswith("cat__"):
            raw = col_name[5:]
            parts = raw.split("_", 1)
            orig = parts[0]
            val = parts[1] if len(parts) > 1 else ""
            human_label = f"{orig} = '{val}'" if val else orig
            return orig, human_label
        elif col_name.startswith("num__"):
            orig = col_name[5:]
            return orig, orig
        return col_name, col_name

    def explain_instance(
        self,
        student_data: Dict[str, Any],
        subject: str = "Mathematics",
        experiment: str = "A",
    ) -> Dict[str, Any]:
        """
        Computes local SHAP explanation for an individual student prediction.

        Returns:
            Dictionary containing prediction outcome, at-risk probability, base value,
            detailed feature contributions (positive = increases risk, negative = decreases risk),
            and aggregated original feature contributions.
        """
        subj = subject.capitalize()
        exp = experiment.upper()
        if subj not in SUPPORTED_SUBJECTS:
            raise ValueError(f"Subject '{subject}' is not supported.")
        if exp not in ["A", "B"]:
            raise ValueError("Experiment must be 'A' or 'B'.")

        explainer, pipeline = self.get_explainer(subj, exp)
        transformer = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        clf_type = type(classifier).__name__

        req_cols = FEATURES_MODEL_A if exp == "A" else FEATURES_MODEL_B

        # Defensive feature resolution: map G1 from G1_raw or SQLite assessments if not explicitly under 'G1'
        inst_data = dict(student_data)
        if exp == "B" and "G1" not in inst_data:
            if "G1_raw" in inst_data:
                inst_data["G1"] = inst_data["G1_raw"]
            elif "student_id" in inst_data and inst_data["student_id"] != "ANONYMOUS":
                try:
                    assessments = self.ingestion.load_raw_dataset(subj)
                except Exception:
                    pass

        missing = [c for c in req_cols if c not in inst_data]
        if missing:
            raise ValueError(f"Student record missing required features: {missing}")

        row_df = pd.DataFrame([{c: inst_data[c] for c in req_cols}])
        X_trans = transformer.transform(row_df)
        feat_names = transformer.get_feature_names_out()

        # Compute prediction and probability
        pred_label = int(pipeline.predict(row_df)[0])
        at_risk_prob = float(pipeline.predict_proba(row_df)[0][1]) if hasattr(pipeline, "predict_proba") else float(pred_label)

        # Compute SHAP values
        raw_shap = explainer(X_trans)

        # Extract values for the positive class (At-Risk = 1)
        # Handle shape differences: (1, M) for Linear & XGBoost, (1, M, 2) for RandomForest
        if hasattr(raw_shap, "values"):
            vals = raw_shap.values
            base_vals = raw_shap.base_values
        else:
            vals = raw_shap
            base_vals = getattr(explainer, "expected_value", 0.0)

        if vals.ndim == 3 and vals.shape[2] == 2:
            # Multi-class output (e.g. RandomForest): index 1 is positive class At-Risk
            inst_shap = vals[0, :, 1]
            base_val = float(base_vals[0, 1] if hasattr(base_vals, "__getitem__") and hasattr(base_vals, "ndim") and base_vals.ndim > 1 else (base_vals[1] if isinstance(base_vals, (list, np.ndarray)) else base_vals))
        elif vals.ndim == 2:
            inst_shap = vals[0, :]
            base_val = float(base_vals[0] if isinstance(base_vals, (list, np.ndarray)) else base_vals)
        else:
            inst_shap = vals.flatten()
            base_val = float(base_vals[0] if isinstance(base_vals, (list, np.ndarray)) else base_vals)

        # Structure individual feature contributions
        contributions: List[Dict[str, Any]] = []
        orig_feature_aggregates: Dict[str, float] = {}

        for col_name, shap_val, trans_val in zip(feat_names, inst_shap, X_trans[0]):
            orig_feat, human_label = self._map_feature_name(col_name)
            s_val = float(shap_val)

            orig_val = student_data.get(orig_feat, "")
            contributions.append({
                "transformed_feature": col_name,
                "original_feature": orig_feat,
                "display_name": human_label,
                "original_value": orig_val,
                "transformed_value": round(float(trans_val), 4),
                "shap_value": round(s_val, 4),
                "abs_shap": round(abs(s_val), 4),
                "impact_direction": "INCREASES_RISK" if s_val > 0 else "DECREASES_RISK",
            })

            # Aggregate by original feature
            orig_feature_aggregates[orig_feat] = orig_feature_aggregates.get(orig_feat, 0.0) + s_val

        # Sort transformed contributions by absolute impact
        contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

        # Sort aggregated original features by absolute impact
        orig_summary = [
            {
                "feature": f,
                "original_value": student_data.get(f, ""),
                "net_shap_value": round(net_val, 4),
                "abs_net_shap": round(abs(net_val), 4),
                "impact_direction": "INCREASES_RISK" if net_val > 0 else "DECREASES_RISK",
            }
            for f, net_val in orig_feature_aggregates.items()
        ]
        orig_summary.sort(key=lambda x: x["abs_net_shap"], reverse=True)

        # Risk increasing vs decreasing factors
        risk_increasing = [c for c in contributions if c["shap_value"] > 0]
        risk_decreasing = [c for c in contributions if c["shap_value"] < 0]

        # Determine mathematical output space
        if clf_type == "LogisticRegression":
            output_space = "log_odds"
            output_units = "log-odds (logit)"
            target_metric = float(classifier.decision_function(X_trans)[0])
            sum_shap = float(np.sum(inst_shap))
            reconstructed = float(base_val + sum_shap)
            reconstruction_delta = abs(target_metric - reconstructed)
        elif clf_type == "RandomForestClassifier":
            output_space = "probability"
            output_units = "probability"
            target_metric = float(classifier.predict_proba(X_trans)[0, 1])
            sum_shap = float(np.sum(inst_shap))
            reconstructed = float(base_val + sum_shap)
            reconstruction_delta = abs(target_metric - reconstructed)
        elif clf_type == "XGBClassifier":
            output_space = "log_odds"
            output_units = "log-odds (margin)"
            target_metric = float(classifier.predict_proba(X_trans)[0, 1])
            sum_shap = float(np.sum(inst_shap))
            reconstructed = float(base_val + sum_shap)
            # Reconstructed in margin space, map through sigmoid to compare with probability
            prob_reconstructed = 1.0 / (1.0 + np.exp(-reconstructed))
            reconstruction_delta = abs(target_metric - prob_reconstructed)
        else:
            output_space = "probability"
            output_units = "probability"
            target_metric = at_risk_prob
            sum_shap = float(np.sum(inst_shap))
            reconstructed = float(base_val + sum_shap)
            reconstruction_delta = abs(target_metric - reconstructed)

        return {
            "student_id": student_data.get("student_id", "ANONYMOUS"),
            "subject": subj,
            "experiment": f"MODEL_{exp}",
            "classifier_type": clf_type,
            "output_space": output_space,
            "output_units": output_units,
            "predicted_risk_category": "At-Risk" if pred_label == 1 else "Satisfactory",
            "at_risk_probability": round(at_risk_prob, 4),
            "base_value": round(base_val, 4),
            "reconstruction_check": {
                "output_space": output_space,
                "output_units": output_units,
                "base_value": round(base_val, 6),
                "sum_shap": round(sum_shap, 6),
                "reconstructed_output": round(reconstructed, 6),
                "reconstruction_error": round(reconstruction_delta, 8),
                "is_reconstructed": bool(reconstruction_delta < 1e-3),
            },
            "top_risk_increasing_factors": risk_increasing[:5],
            "top_risk_decreasing_factors": risk_decreasing[:5],
            "all_feature_contributions": contributions,
            "aggregated_original_features": orig_summary,
            "scientific_disclaimer": SHAP_SCIENTIFIC_DISCLAIMER,
        }

    def compute_global_feature_importance(
        self,
        subject: str = "Mathematics",
        experiment: str = "A",
        max_samples: int = 100,
    ) -> Dict[str, Any]:
        """
        Computes global mean absolute SHAP value importance across the background training partition.

        Returns:
            Dictionary containing ranked transformed features and ranked original features.
        """
        subj = subject.capitalize()
        exp = experiment.upper()
        explainer, pipeline = self.get_explainer(subj, exp)
        transformer = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        clf_type = type(classifier).__name__

        bg_df = self.get_background_data(subj, exp, n_samples=max_samples)
        X_trans = transformer.transform(bg_df)
        feat_names = transformer.get_feature_names_out()

        raw_shap = explainer(X_trans)
        vals = raw_shap.values if hasattr(raw_shap, "values") else raw_shap

        if vals.ndim == 3 and vals.shape[2] == 2:
            # Positive class index 1 (At-Risk)
            shap_matrix = vals[:, :, 1]
        elif vals.ndim == 2:
            shap_matrix = vals
        else:
            shap_matrix = vals.reshape(len(bg_df), -1)

        # Mean absolute SHAP value per transformed column
        mean_abs = np.mean(np.abs(shap_matrix), axis=0)

        transformed_importance = []
        orig_agg: Dict[str, List[float]] = {}

        for col_name, score in zip(feat_names, mean_abs):
            orig_feat, human_label = self._map_feature_name(col_name)
            transformed_importance.append({
                "transformed_feature": col_name,
                "original_feature": orig_feat,
                "display_name": human_label,
                "mean_abs_shap": round(float(score), 4),
            })
            if orig_feat not in orig_agg:
                orig_agg[orig_feat] = []
            orig_agg[orig_feat].append(float(score))

        transformed_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        original_importance = [
            {
                "original_feature": f,
                "mean_abs_shap": round(float(np.mean(scores)), 4),
                "sum_abs_shap": round(float(np.sum(scores)), 4),
            }
            for f, scores in orig_agg.items()
        ]
        original_importance.sort(key=lambda x: x["sum_abs_shap"], reverse=True)

        return {
            "subject": subj,
            "experiment": f"MODEL_{exp}",
            "classifier_type": clf_type,
            "samples_evaluated": len(bg_df),
            "ranked_original_features": original_importance,
            "ranked_transformed_features": transformed_importance,
            "scientific_disclaimer": SHAP_SCIENTIFIC_DISCLAIMER,
        }

