"""
ExplainAI - Module 2: Academic Drift Early-Warning System (Signature Feature 3)
Implements chronological score-trajectory deterioration monitoring across consecutive
term assessments (e.g. G1 -> G2) to identify early academic declines before final examinations.

TERMINOLOGY & SCOPE NOTICE:
---------------------------
This component performs 'Academic Performance Deterioration Monitoring'. It evaluates
chronological decreases across consecutive evaluation marks. It is NOT a measure of
statistical covariate or concept drift in machine learning data distributions.

ANTI-LEAKAGE SEPARATION:
------------------------
1. The post-G1 ML experiment does NOT access G2 or G3.
2. The drift monitoring component evaluates G1 and G2 observations (as G2 becomes
   available at midterm) to identify deterioration prior to the final exam (G3).
3. Final outcome G3 is NEVER used to fabricate an 'early' warning.
"""

from typing import Dict, List, Any, Optional
from src.config import WARNING_THRESHOLDS
from src.database.db_manager import DatabaseManager


DEFAULT_DECLINE_THRESHOLD = 3.0  # Drop of 3.0 or more points on 0-20 scale is critical
MODERATE_DECLINE_THRESHOLD = 1.5  # Drop of 1.5 to 2.9 points is moderate


class AcademicDriftDetector:
    """Monitors student chronological grade trajectories to detect performance deterioration."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        critical_decline_points: float = DEFAULT_DECLINE_THRESHOLD,
        moderate_decline_points: float = MODERATE_DECLINE_THRESHOLD,
    ):
        self.db = db_manager or DatabaseManager()
        self.critical_threshold = critical_decline_points
        self.moderate_threshold = moderate_decline_points

    def analyze_trajectory(
        self,
        student_id: str,
        subject: str = "Mathematics",
        include_final_grade: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyzes consecutive available term assessments for academic deterioration.

        Ordering & Leakage Enforcement:
        - Relies on term_period ordering (Term 1 -> Term 2 -> Term 3).
        - If include_final_grade is False (standard early-warning setting), only Term 1 (G1)
          and Term 2 (G2) are analyzed. Final grade G3 is excluded.
        - Returns trajectory trend, score deltas, and generated warnings.
        """
        all_assessments = self.db.get_student_assessments(student_id, subject)

        # Filter assessments according to early-warning scope
        evaluated_assessments = []
        for a in all_assessments:
            term = int(a["term_period"])
            if not include_final_grade and term >= 3:
                continue  # Bar final exam G3 from early-warning generation
            evaluated_assessments.append(a)

        # Sort strictly by term_period ascending
        evaluated_assessments.sort(key=lambda x: (x["term_period"], x["assessment_date"] or ""))

        observations = []
        for a in evaluated_assessments:
            observations.append(
                {
                    "assessment_name": a["assessment_name"],
                    "term_period": a["term_period"],
                    "score": float(a["score"]),
                    "date": a["assessment_date"],
                    "date_provenance": a.get("date_provenance", "NOT_PROVIDED_IN_UCI_SOURCE"),
                    "is_uci_source": a["is_uci_source"],
                }
            )

        if len(observations) < 2:
            return {
                "student_id": student_id,
                "subject": subject,
                "evaluable": False,
                "observations_count": len(observations),
                "trend": "INSUFFICIENT_OBSERVATIONS",
                "message": (
                    f"Only {len(observations)} evaluation observation(s) available for {subject}. "
                    f"At least two consecutive evaluations (e.g. G1 and G2) are required to monitor trajectory."
                ),
                "observations": observations,
                "consecutive_deltas": [],
                "warnings": [],
            }

        consecutive_deltas = []
        warnings = []
        max_decline = 0.0

        for i in range(1, len(observations)):
            prev = observations[i - 1]
            curr = observations[i]
            delta = round(curr["score"] - prev["score"], 2)
            consecutive_deltas.append(
                {
                    "from_term": prev["term_period"],
                    "to_term": curr["term_period"],
                    "from_assessment": prev["assessment_name"],
                    "to_assessment": curr["assessment_name"],
                    "previous_score": prev["score"],
                    "current_score": curr["score"],
                    "score_delta": delta,
                }
            )

            # Check for negative drop
            if delta < 0:
                drop_magnitude = abs(delta)
                max_decline = max(max_decline, drop_magnitude)

                if drop_magnitude >= self.critical_threshold:
                    warnings.append(
                        {
                            "warning_id": f"DRIFT_CRITICAL_T{prev['term_period']}_T{curr['term_period']}",
                            "severity": "CRITICAL",
                            "score_drop": drop_magnitude,
                            "from_score": prev["score"],
                            "to_score": curr["score"],
                            "condition": f"Score Drop >= {self.critical_threshold:.1f} points",
                            "explanation": (
                                f"Academic deterioration alert: Grade dropped by {drop_magnitude:.1f} points "
                                f"(from {prev['score']:.1f} to {curr['score']:.1f}) between {prev['assessment_name']} "
                                f"and {curr['assessment_name']}. Immediate educational intervention is warranted."
                            ),
                        }
                    )
                elif drop_magnitude >= self.moderate_threshold:
                    warnings.append(
                        {
                            "warning_id": f"DRIFT_MODERATE_T{prev['term_period']}_T{curr['term_period']}",
                            "severity": "MODERATE",
                            "score_drop": drop_magnitude,
                            "from_score": prev["score"],
                            "to_score": curr["score"],
                            "condition": f"Score Drop >= {self.moderate_threshold:.1f} points",
                            "explanation": (
                                f"Moderate downward trajectory detected: Grade declined by {drop_magnitude:.1f} points "
                                f"(from {prev['score']:.1f} to {curr['score']:.1f}). Remedial reinforcement recommended."
                            ),
                        }
                    )

        # Categorize overall trajectory trend
        latest_delta = consecutive_deltas[-1]["score_delta"]
        if latest_delta <= -self.critical_threshold:
            overall_trend = "CRITICAL_DECLINE"
        elif latest_delta <= -self.moderate_threshold:
            overall_trend = "MODERATE_DECLINE"
        elif latest_delta >= self.moderate_threshold:
            overall_trend = "IMPROVING"
        else:
            overall_trend = "STABLE"

        return {
            "student_id": student_id,
            "subject": subject,
            "evaluable": True,
            "observations_count": len(observations),
            "trend": overall_trend,
            "latest_score_delta": latest_delta,
            "max_decline_magnitude": round(max_decline, 2),
            "observations": observations,
            "consecutive_deltas": consecutive_deltas,
            "warnings": warnings,
            "drift_type": "ACADEMIC_PERFORMANCE_DETERIORATION",
        }

