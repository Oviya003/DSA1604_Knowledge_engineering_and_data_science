"""
ExplainAI - UI View: Personalized Learning Recommendations
Displays actionable pedagogical recommendations synthesized across multiple diagnostic signals.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.database.db_manager import DatabaseManager
from src.module3_explainability_support.recommender import (
    PersonalizedLearningRecommender,
    RECOMMENDER_DISCLAIMER,
)
from src.module2_prediction_reasoning.predictor import AcademicRiskPredictor
from src.module2_prediction_reasoning.rule_engine import RuleBasedWarningEngine
from src.module3_explainability_support.shap_explainer import AcademicShapExplainer
from src.module3_explainability_support.knowledge_gap_detective import KnowledgeGapDetective


def render_recommendations_view(
    db: DatabaseManager,
    recommender: PersonalizedLearningRecommender,
    predictor: AcademicRiskPredictor,
    rule_engine: RuleBasedWarningEngine,
    shap_service: AcademicShapExplainer,
    gap_detective: KnowledgeGapDetective,
    student: Dict[str, Any],
    subject: str,
):
    student_id = student["student_id"]

    st.markdown(
        f"""
        <div class="disclaimer-card">
            <strong>Advisory Pedagogical Guidance Disclaimer:</strong><br>
            {RECOMMENDER_DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute live signals
    with st.spinner("Synthesizing multi-modal academic signals..."):
        try:
            pred_res = predictor.predict_student_risk(student, subject=subject, experiment="B", persist=False)
        except Exception:
            pred_res = None

        warn_res = rule_engine.evaluate_rules(student, pred_res, model_experiment="B")

        try:
            shap_res = shap_service.explain_instance(student, subject=subject, experiment="B")
        except Exception:
            shap_res = None

        gap_analysis = None
        if gap_detective.is_subject_supported(subject):
            try:
                gap_analysis = gap_detective.analyze_prerequisite_gaps(student_id, "MATH_05", subject=subject)
            except Exception:
                gap_analysis = None

        recs_data = recommender.generate_recommendations(
            student_data=student,
            prediction_result=pred_res,
            warning_report=warn_res,
            shap_result=shap_res,
            gap_analysis=gap_analysis,
            subject=subject,
            persist=True,
        )

    recommendations = recs_data.get("recommendations", [])

    # Priority counts summary strip
    high_count = sum(1 for r in recommendations if r.get("priority") == "HIGH")
    med_count = sum(1 for r in recommendations if r.get("priority") == "MEDIUM")
    low_count = sum(1 for r in recommendations if r.get("priority") == "LOW")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Interventions</div>
                <div class="metric-value">{len(recommendations)}</div>
                <div class="metric-subtitle">Active action candidates</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">High Priority</div>
                <div class="metric-value" style="color:#DC2626;">{high_count}</div>
                <div class="metric-subtitle">Immediate remediation required</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Medium Priority</div>
                <div class="metric-value" style="color:#D97706;">{med_count}</div>
                <div class="metric-subtitle">Advisory curricular reinforcement</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Low Priority</div>
                <div class="metric-value" style="color:#059669;">{low_count}</div>
                <div class="metric-subtitle">Supportive enrichment activities</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="saas-card-header">
            <div>
                <h3 class="saas-card-title">Synthesized Pedagogical Action Plan</h3>
                <div class="saas-card-subtitle">Triaged recommendations synthesized across ML risk drivers, SHAP values, and curriculum gaps.</div>
            </div>
            <span class="badge-info">Evidence-Backed</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for r in recommendations:
        prio = r["priority"]
        cat = r["category"]
        prio_badge = "badge-at-risk" if prio == "HIGH" else ("badge-warning" if prio == "MEDIUM" else "badge-safe")
        border_cls = "rec-high" if prio == "HIGH" else ("rec-med" if prio == "MEDIUM" else "rec-low")

        st.markdown(
            f"""
            <div class="rec-card {border_cls}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="{prio_badge}">[{prio} PRIORITY]</span>
                        <strong style="color:#14213D; font-size:14px;">{cat.replace('_', ' ').title()}</strong>
                    </div>
                    <span class="badge-info" style="font-size:11px;">{r.get('suggested_action', 'RECOMMENDATION')}</span>
                </div>
                <div style="font-size:15px; font-weight:700; margin-top:10px; color:#14213D;">
                    {r['recommendation_text']}
                </div>
                <div style="font-size:13px; color:#475569; margin-top:6px; line-height:1.5;">
                    <strong>Pedagogical Rationale:</strong> {r['rationale']}
                </div>
                <div style="margin-top:10px; display:flex; align-items:center; gap:6px; font-size:12px; color:#64748B;">
                    <span style="font-weight:600; color:#1E293B;">Evidence Base:</span>
                    <span style="background:#F1F5F9; border:1px solid #E2E8F0; padding:2px 8px; border-radius:4px;">{r['evidence_source']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Historical Log Section
    # --------------------------------------------------------------------------
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="saas-card-header">
            <div>
                <h3 class="saas-card-title">Historical Recommendations Audit Log</h3>
                <div class="saas-card-subtitle">Stored recommendations in SQLite database.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    persisted_recs = db.get_student_recommendations(student_id, subject=subject)
    if persisted_recs:
        p_df = pd.DataFrame(persisted_recs)[["recommendation_id", "priority", "recommendation_text", "status", "created_at"]]
        p_df = p_df.rename(columns={
            "recommendation_id": "ID",
            "priority": "Priority",
            "recommendation_text": "Recommendation",
            "status": "Status",
            "created_at": "Timestamp",
        })
        st.dataframe(p_df, use_container_width=True)
    else:
        st.info("No prior recommendation records found in database.")

    st.markdown('</div>', unsafe_allow_html=True)
