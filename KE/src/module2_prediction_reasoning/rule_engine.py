"""
ExplainAI - Module 2: Rule-Based Reasoning Engine
Generates transparent, deterministic academic warnings from student attributes,
early assessment marks, and predicted risk probabilities.

ETHICAL & NON-DISCRIMINATION POLICY:
------------------------------------
Rule-based warnings evaluate strictly actionable, behavioral, and academic
factors (e.g. study time, past failures, early exam performance, attendance).
Demographic and protected characteristics (e.g. sex, address type, parental jobs,
family size, school attended) are STRICTLY FORBIDDEN from triggering academic warnings
or punitive interventions.
"""

from typing import Dict, List, Any, Optional

from src.config import WARNING_THRESHOLDS


class RuleBasedWarningEngine:
    """Evaluates rule-based academic warnings with transparent, configurable thresholds."""

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = dict(WARNING_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)

    def evaluate_rules(
        self,
        student_data: Dict[str, Any],
        prediction_result: Optional[Dict[str, Any]] = None,
        custom_thresholds: Optional[Dict[str, Any]] = None,
        model_experiment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates student attributes against transparent academic warning rules.

        Rules Evaluated:
        1. Model-Predicted Risk (Probability >= threshold, default 0.50)
        2. Early Assessment Struggle (G1 < 10.0; evaluated ONLY for Model B / post-G1;
           suppressed for Model A intake predictions to prevent temporal leakage)
        3. Prior Academic Failures (failures >= threshold, default 1)
        4. Attendance Tally Concern (absences >= threshold, default 10)
        5. Low Self-Reported Study Effort (studytime <= 1, <2 hours/week)
        6. Alcohol Consumption Concern (Dalc >= 3 or Walc >= 4)
        7. Self-Reported Health Vulnerability (health <= 2)

        Returns:
            Dictionary containing list of triggered warnings, summary counts,
            severity breakdown, and evaluation metadata.
        """
        active_thresholds = dict(self.thresholds)
        if custom_thresholds:
            active_thresholds.update(custom_thresholds)

        triggered_warnings: List[Dict[str, Any]] = []

        # Determine if evaluating under Model A (Intake) context
        is_model_a = False
        if model_experiment and str(model_experiment).upper() in ["A", "MODEL_A"]:
            is_model_a = True
        elif prediction_result and prediction_result.get("model_experiment") in ["A", "MODEL_A"]:
            is_model_a = True

        # Rule 1: High Predicted Risk Probability (from ML service)
        risk_prob_threshold = active_thresholds.get("predicted_risk_prob", 0.50)
        if prediction_result and "at_risk_probability" in prediction_result:
            prob = float(prediction_result["at_risk_probability"])
            if prob >= risk_prob_threshold:
                triggered_warnings.append(
                    {
                        "rule_id": "RULE_ML_HIGH_RISK",
                        "rule_name": "High Statistical Risk Score",
                        "severity": "HIGH",
                        "condition": f"Predicted At-Risk Probability >= {risk_prob_threshold:.2f}",
                        "observed_value": f"{prob:.2%}",
                        "explanation": (
                            f"The statistical classification model predicts a {prob:.1%} probability of final "
                            f"academic failure based on intake/early indicators. Immediate proactive academic support is advised."
                        ),
                    }
                )

        # Rule 2: Early Assessment Deficit (G1 < 10)
        # TEMPORAL LEAKAGE RULE: Evaluated ONLY post-G1 (Model B). Suppressed at intake (Model A)
        # because G1 evaluation has not occurred yet.
        if not is_model_a and "G1" in student_data and student_data["G1"] is not None:
            g1_val = float(student_data["G1"])
            g1_threshold = active_thresholds.get("g1_failure_threshold", 10.0)
            if g1_val < g1_threshold:
                triggered_warnings.append(
                    {
                        "rule_id": "RULE_EARLY_EXAM_FAIL",
                        "rule_name": "First Period Examination Deficit",
                        "severity": "HIGH",
                        "condition": f"Term 1 Grade (G1) < {g1_threshold:.1f}",
                        "observed_value": f"{g1_val:.1f} / 20",
                        "explanation": (
                            f"Student scored {g1_val:.1f}/20 on the first period evaluation, falling below the "
                            f"passing benchmark ({g1_threshold:.1f}/20). Immediate foundational review is required."
                        ),
                    }
                )

        # Rule 3: Prior Academic Failures
        fail_threshold = active_thresholds.get("high_failures", 1)
        if "failures" in student_data:
            failures = int(student_data["failures"])
            if failures >= fail_threshold:
                severity = "HIGH" if failures >= 2 else "MEDIUM"
                triggered_warnings.append(
                    {
                        "rule_id": "RULE_HISTORICAL_FAILURES",
                        "rule_name": "Historical Academic Failure Record",
                        "severity": severity,
                        "condition": f"Past Course Failures >= {fail_threshold}",
                        "observed_value": f"{failures} failure(s)",
                        "explanation": (
                            f"Student has a documented record of {failures} prior course failure(s). Prior academic "
                            f"struggles present an ongoing risk for subsequent conceptual mastery."
                        ),
                    }
                )

        # Rule 4: Attendance Concern (from descriptive absences attribute)
        absence_threshold = active_thresholds.get("excessive_absences", 10)
        if "absences" in student_data:
            absences = int(student_data["absences"])
            if absences >= absence_threshold:
                severity = "HIGH" if absences >= 18 else "MEDIUM"
                triggered_warnings.append(
                    {
                        "rule_id": "RULE_EXCESSIVE_ABSENCES",
                        "rule_name": "Chronic Course Absence Tally",
                        "severity": severity,
                        "condition": f"School Absences >= {absence_threshold}",
                        "observed_value": f"{absences} absences",
                        "explanation": (
                            f"Student has accumulated {absences} recorded absences. Frequent absenteeism correlates "
                            f"strongly with learning gaps and missed classroom instruction."
                        ),
                    }
                )

        # Rule 5: Low Weekly Study Time
        study_threshold = active_thresholds.get("low_study_time", 1)
        if "studytime" in student_data:
            studytime = int(student_data["studytime"])
            if studytime <= study_threshold:
                triggered_warnings.append(
                    {
                        "rule_id": "RULE_LOW_STUDY_TIME",
                        "rule_name": "Low Independent Study Allocation",
                        "severity": "MEDIUM",
                        "condition": f"Weekly Study Time Level <= {study_threshold} (< 2 hours/week)",
                        "observed_value": f"Level {studytime} (< 2 hrs/wk)",
                        "explanation": (
                            f"Student reports dedicating less than 2 hours per week to independent course study. "
                            f"Structured study routines and targeted exercise practice are recommended."
                        ),
                    }
                )

        # Rule 6: Elevated Substance / Alcohol Consumption
        high_alc_threshold = active_thresholds.get("high_alcohol", 4)
        dalc = int(student_data.get("Dalc", 1))
        walc = int(student_data.get("Walc", 1))
        if dalc >= 3 or walc >= high_alc_threshold:
            triggered_warnings.append(
                {
                    "rule_id": "RULE_ELEVATED_ALCOHOL",
                    "rule_name": "Elevated Self-Reported Alcohol Consumption",
                    "severity": "MEDIUM",
                    "condition": f"Workday Alcohol >= 3 or Weekend Alcohol >= {high_alc_threshold}",
                    "observed_value": f"Workday: {dalc}/5, Weekend: {walc}/5",
                    "explanation": (
                        "Survey records indicate elevated alcohol consumption habits, which may impair student "
                        "attention, cognitive stamina, and homework completion."
                    ),
                }
            )

        # Rule 7: Health Vulnerability
        poor_health_threshold = active_thresholds.get("poor_health", 2)
        if "health" in student_data:
            health = int(student_data["health"])
            if health <= poor_health_threshold:
                triggered_warnings.append(
                    {
                        "rule_id": "RULE_POOR_HEALTH",
                        "rule_name": "Self-Reported Health Vulnerability",
                        "severity": "LOW",
                        "condition": f"Current Health Status <= {poor_health_threshold} (Poor)",
                        "observed_value": f"Level {health}/5",
                        "explanation": (
                            f"Student reports vulnerable general health (rating {health}/5). Physical wellbeing "
                            f"concerns may impact sustained academic engagement."
                        ),
                    }
                )

        # Compute summary counts
        severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for w in triggered_warnings:
            severity_counts[w["severity"]] += 1

        overall_status = "SAFE"
        if severity_counts["HIGH"] > 0:
            overall_status = "CRITICAL_WARNING"
        elif severity_counts["MEDIUM"] > 0:
            overall_status = "MODERATE_WARNING"
        elif severity_counts["LOW"] > 0:
            overall_status = "MONITORING"

        return {
            "overall_status": overall_status,
            "total_warnings_triggered": len(triggered_warnings),
            "severity_counts": severity_counts,
            "warnings": triggered_warnings,
            "active_thresholds": active_thresholds,
        }

