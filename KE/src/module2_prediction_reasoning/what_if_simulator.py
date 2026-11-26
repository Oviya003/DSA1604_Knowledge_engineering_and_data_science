"""
ExplainAI - Module 2: Academic What-If Simulator (Signature Feature 1)
Enables interactive counterfactual sensitivity analysis by allowing users to explore
how hypothetical changes to actionable profile variables alter model risk probabilities.

CRITICAL SCIENTIFIC & METHODOLOGICAL DISCLAIMER:
------------------------------------------------
The statistical machine learning models estimate associative risk probabilities
learned from observational historical data. A decrease or increase in predicted
risk probability reflects counterfactual model behavior under hypothetical inputs.
It DOES NOT provide causal proof that changing a student's real-world behavior
will guarantee an identical change in their actual academic grade.
"""

from typing import Dict, List, Any, Optional
import copy

from src.module2_prediction_reasoning.predictor import AcademicRiskPredictor
from src.database.db_manager import DatabaseManager


# Explicit allowlist of actionable, academic-support and study-related scenario variables
ACTIONABLE_ALLOWLIST = {
    "studytime": {"type": "int", "min": 1, "max": 4, "description": "Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)"},
    "freetime": {"type": "int", "min": 1, "max": 5, "description": "Free time after school (1: very low to 5: very high)"},
    "goout": {"type": "int", "min": 1, "max": 5, "description": "Going out with friends (1: very low to 5: very high)"},
    "schoolsup": {"type": "categorical", "allowed": ["yes", "no"], "description": "Extra school academic support"},
    "famsup": {"type": "categorical", "allowed": ["yes", "no"], "description": "Family educational support"},
    "paid": {"type": "categorical", "allowed": ["yes", "no"], "description": "Extra paid subject classes"},
    "activities": {"type": "categorical", "allowed": ["yes", "no"], "description": "Extracurricular activities"},
}

# Immutable demographic features that can NEVER be changed in what-if scenarios
IMMUTABLE_DEMOGRAPHICS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "nursery", "internet",
]

# Sensitive personal and health attributes excluded from academic simulation
RESTRICTED_PERSONAL_ATTRIBUTES = ["Dalc", "Walc", "health"]

# Immutable historical assessment, behavioral records, and outcomes that can NEVER be changed retrospectively
IMMUTABLE_HISTORICAL_ATTRIBUTES = ["failures", "G1", "G2", "G3", "absences", "is_at_risk"]
IMMUTABLE_HISTORICAL_GRADES = ["G1", "G2", "G3"]

SCIENTIFIC_DISCLAIMER_TEXT = (
    "CRITICAL SCIENTIFIC LIMITATION: This simulator displays statistical associations "
    "learned by the classification model from observational survey data. A reduction in "
    "predicted risk probability indicates model sensitivity to the adjusted feature values, "
    "but DOES NOT constitute causal proof that intervening on these variables will guarantee "
    "an improved academic outcome in real life."
)


class AcademicWhatIfSimulator:
    """Manages counterfactual what-if simulations against trained predictive models."""

    def __init__(
        self,
        predictor: Optional[AcademicRiskPredictor] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.predictor = predictor or AcademicRiskPredictor()
        self.db = db_manager or DatabaseManager()

    def validate_counterfactual_modifications(self, proposed_changes: Dict[str, Any]) -> List[str]:
        """
        Validates that proposed changes adhere strictly to the actionable allowlist,
        valid domain bounds, and ethical non-modification of demographic attributes.
        """
        errors = []
        for feature, val in proposed_changes.items():
            if feature in IMMUTABLE_DEMOGRAPHICS:
                errors.append(f"Modification rejected: '{feature}' is an immutable demographic attribute.")
            elif feature in IMMUTABLE_HISTORICAL_ATTRIBUTES or feature in IMMUTABLE_HISTORICAL_GRADES:
                errors.append(f"Modification rejected: '{feature}' is an empirical historical record/outcome and cannot be modified retrospectively.")
            elif feature in RESTRICTED_PERSONAL_ATTRIBUTES:
                errors.append(f"Modification rejected: '{feature}' is a sensitive personal/health attribute not allowed as an academic intervention scenario.")
            elif feature not in ACTIONABLE_ALLOWLIST:
                errors.append(f"Modification rejected: '{feature}' is not on the actionable What-If allowlist.")
            else:
                spec = ACTIONABLE_ALLOWLIST[feature]
                if spec["type"] == "int":
                    try:
                        int_val = int(val)
                        if int_val < spec["min"] or int_val > spec["max"]:
                            errors.append(f"Value {val} for '{feature}' out of valid range [{spec['min']}, {spec['max']}].")
                    except (ValueError, TypeError):
                        errors.append(f"Value '{val}' for '{feature}' must be an integer.")
                elif spec["type"] == "categorical":
                    if str(val).lower() not in spec["allowed"]:
                        errors.append(f"Value '{val}' for '{feature}' must be one of {spec['allowed']}.")
        return errors

    def simulate_counterfactual(
        self,
        original_student_data: Dict[str, Any],
        proposed_changes: Dict[str, Any],
        subject: str = "Mathematics",
        experiment: str = "A",
        log_to_audit: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes a counterfactual prediction scenario:
        1. Validates proposed changes against allowlist and value ranges.
        2. Calculates original baseline model prediction without altering original records.
        3. Creates hypothetical in-memory student clone with applied modifications.
        4. Calculates hypothetical model prediction.
        5. Computes probability delta and category transition.
        6. Logs simulation in audit_logs for traceability and accountability.
        """
        validation_errors = self.validate_counterfactual_modifications(proposed_changes)
        if validation_errors:
            raise ValueError(f"Invalid What-If modification request: {validation_errors}")

        # Compute original baseline prediction
        baseline_pred = self.predictor.predict_student_risk(
            student_data=original_student_data,
            subject=subject,
            experiment=experiment,
            persist=False,
        )

        # Create cloned hypothetical record
        hypothetical_data = copy.deepcopy(original_student_data)
        for feat, val in proposed_changes.items():
            spec = ACTIONABLE_ALLOWLIST[feat]
            hypothetical_data[feat] = int(val) if spec["type"] == "int" else str(val).lower()

        # Compute hypothetical counterfactual prediction
        hypothetical_pred = self.predictor.predict_student_risk(
            student_data=hypothetical_data,
            subject=subject,
            experiment=experiment,
            persist=False,
        )

        # Delta metrics
        base_prob = baseline_pred["at_risk_probability"]
        hypo_prob = hypothetical_pred["at_risk_probability"]
        prob_delta = round(hypo_prob - base_prob, 4)

        if prob_delta < -0.005:
            effect_direction = "RISK_REDUCED"
        elif prob_delta > 0.005:
            effect_direction = "RISK_INCREASED"
        else:
            effect_direction = "NO_MEANINGFUL_CHANGE"

        category_changed = baseline_pred["risk_category"] != hypothetical_pred["risk_category"]

        simulation_result = {
            "student_id": original_student_data.get("student_id", "ANONYMOUS"),
            "subject": subject,
            "experiment": f"MODEL_{experiment.upper()}",
            "model_algorithm": baseline_pred["model_algorithm"],
            "baseline": {
                "risk_category": baseline_pred["risk_category"],
                "at_risk_probability": base_prob,
            },
            "hypothetical": {
                "risk_category": hypothetical_pred["risk_category"],
                "at_risk_probability": hypo_prob,
                "modified_features": proposed_changes,
            },
            "delta": {
                "probability_difference": prob_delta,
                "percentage_point_difference": round(prob_delta * 100.0, 2),
                "effect_direction": effect_direction,
                "category_transition": f"{baseline_pred['risk_category']} -> {hypothetical_pred['risk_category']}",
                "category_changed": category_changed,
            },
            "scientific_disclaimer": SCIENTIFIC_DISCLAIMER_TEXT,
        }

        # Log scenario in audit_logs without altering genuine student or assessment records
        if log_to_audit and "student_id" in original_student_data:
            try:
                self.db.log_audit_action(
                    action_type="WHAT_IF_SIMULATION",
                    details=simulation_result,
                    student_id=original_student_data.get("student_id"),
                    subject=subject,
                )
            except Exception:
                pass

        return simulation_result

