"""
ExplainAI - UI View: Academic Prediction
Displays Model A and Model B inferences, prediction timing, probability metrics,
and detailed deterministic rule reasoning.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from src.module2_prediction_reasoning.predictor import AcademicRiskPredictor
from src.module2_prediction_reasoning.rule_engine import RuleBasedWarningEngine


def render_prediction_view(
    predictor: AcademicRiskPredictor,
    rule_engine: RuleBasedWarningEngine,
    student: Dict[str, Any],
    subject: str,
):
    student_id = student["student_id"]

    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>Dual-Timeline Forecasting Framework:</strong><br>
            • <strong>Model A (Intake Baseline)</strong>: Evaluates 29 demographic and pre-course indicators at enrollment prior to examinations.<br>
            • <strong>Model B (Post-G1 Interim)</strong>: Incorporates the First Period Examination (G1) for 30 total features, substantially boosting discriminative power.<br>
            • <strong>Probability Calibration Notice</strong>: Probability estimates from fitted classifiers; outputs have not been independently calibrated, and no confidence intervals are provided.<br>
            <em>Strict Anti-Leakage: Midterm mark (G2), final mark (G3), target label (is_at_risk), and cumulative absences are strictly barred from both models.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 1. Dual Model Inference Cards
    # --------------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="saas-card-header" style="margin-bottom:8px;">
                <div>
                    <h3 class="saas-card-title">Model A: Intake Baseline</h3>
                    <div class="saas-card-subtitle">Course Intake • Pre-Examination Timeline</div>
                </div>
                <span class="badge-info">29 Features</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            pred_a = predictor.predict_student_risk(student, subject=subject, experiment="A", persist=True)
            prob_a = pred_a["at_risk_probability"]
            cat_a = pred_a["risk_category"]
            color_a = "#DC2626" if cat_a == "At-Risk" else "#059669"
            badge_a = "badge-at-risk" if cat_a == "At-Risk" else "badge-safe"

            fig_a = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_a * 100,
                number={"suffix": "%", "font": {"size": 28, "color": "#14213D", "family": "Plus Jakarta Sans"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": color_a, "thickness": 0.25},
                    "bgcolor": "#F1F5F9",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "#ECFDF5"},
                        {"range": [50, 100], "color": "#FEF2F2"},
                    ],
                    "threshold": {
                        "line": {"color": "#14213D", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_a.update_layout(
                height=180,
                margin=dict(l=15, r=15, t=10, b=10),
                paper_bgcolor="#FFFFFF",
            )
            st.plotly_chart(fig_a, use_container_width=True)

            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; margin-top:10px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-size:12px; color:#64748B;">Predicted Risk Tier</span>
                        <span class="{badge_a}">{cat_a.upper()}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-size:12px; color:#64748B;">Algorithm Selected</span>
                        <strong style="font-size:12px; color:#14213D;">{pred_a['model_algorithm']}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-size:12px; color:#64748B;">Satisfactory Probability</span>
                        <strong style="font-size:12px; color:#059669;">{pred_a['satisfactory_probability']:.1%}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-size:12px; color:#64748B;">Model Selection Base</span>
                        <span style="font-size:11.5px; color:#64748B;">5-Fold Stratified Training CV</span>
                    </div>
                    <div style="font-size:11px; color:#94A3B8; margin-top:8px; border-top:1px dashed #E2E8F0; padding-top:6px;">
                        * Probability estimates from fitted classifiers; outputs have not been independently calibrated.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error executing Model A: {e}")

    with col2:
        st.markdown(
            """
            <div class="saas-card-header" style="margin-bottom:8px;">
                <div>
                    <h3 class="saas-card-title">Model B: Post-G1 Interim</h3>
                    <div class="saas-card-subtitle">Mid-Term Update • Incorporates First Period Grade</div>
                </div>
                <span class="badge-info">30 Features</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pred_b = None
        try:
            pred_b = predictor.predict_student_risk(student, subject=subject, experiment="B", persist=True)
            prob_b = pred_b["at_risk_probability"]
            cat_b = pred_b["risk_category"]
            color_b = "#DC2626" if cat_b == "At-Risk" else "#059669"
            badge_b = "badge-at-risk" if cat_b == "At-Risk" else "badge-safe"
            algo_b = pred_b["model_algorithm"]
            sat_prob_b = pred_b["satisfactory_probability"]

            fig_b = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_b * 100,
                number={"suffix": "%", "font": {"size": 28, "color": "#14213D", "family": "Plus Jakarta Sans"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": color_b, "thickness": 0.25},
                    "bgcolor": "#F1F5F9",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50], "color": "#ECFDF5"},
                        {"range": [50, 100], "color": "#FEF2F2"},
                    ],
                    "threshold": {
                        "line": {"color": "#14213D", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_b.update_layout(
                height=180,
                margin=dict(l=15, r=15, t=10, b=10),
                paper_bgcolor="#FFFFFF",
            )
            st.plotly_chart(fig_b, use_container_width=True)

            g1_val = student.get('G1', student.get('G1_raw', 'N/A'))
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:12px 14px; margin-top:10px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-size:12px; color:#64748B;">Predicted Risk Tier</span>
                        <span class="{badge_b}">{cat_b.upper()}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-size:12px; color:#64748B;">Algorithm Selected</span>
                        <strong style="font-size:12px; color:#14213D;">{algo_b}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                        <span style="font-size:12px; color:#64748B;">First Period Mark (G1)</span>
                        <strong style="font-size:12px; color:#14213D;">{g1_val} / 20 pts</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="font-size:12px; color:#64748B;">Satisfactory Probability</span>
                        <strong style="font-size:12px; color:#059669;">{sat_prob_b:.1%}</strong>
                    </div>
                    <div style="font-size:11px; color:#94A3B8; margin-top:8px; border-top:1px dashed #E2E8F0; padding-top:6px;">
                        * Probability estimates from fitted classifiers; outputs have not been independently calibrated.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Error executing Model B: {e}")

    # --------------------------------------------------------------------------
    # 2. Deterministic Rule-Based Warning Engine
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="saas-card-header" style="margin-top:20px; margin-bottom:12px;">
            <div>
                <h3 class="saas-card-title">Deterministic Rule-Based Warning Reasoning</h3>
                <div class="saas-card-subtitle">Transparent academic heuristics evaluating actionable behavioral indicators.</div>
            </div>
            <span class="badge-info">Deterministic Logic</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Configure Rule Engine Thresholds", expanded=False):
        c_th1, c_th2, c_th3 = st.columns(3)
        with c_th1:
            th_prob = st.slider("Model Risk Probability Threshold", 0.30, 0.80, 0.50, 0.05)
            th_g1 = st.slider("Term 1 Grade Deficit Threshold (G1)", 8.0, 12.0, 10.0, 0.5)
        with c_th2:
            th_fail = st.number_input("Past Failures Threshold", min_value=1, max_value=3, value=1)
            th_abs = st.number_input("Recorded Absences Concern Threshold", min_value=5, max_value=25, value=10)
        with c_th3:
            th_study = st.selectbox("Low Study Time Level", [1, 2], index=0, format_func=lambda x: f"Level {x} (<2 hrs/wk)" if x==1 else f"Level {x} (2-5 hrs/wk)")
            th_alc = st.slider("Weekend Alcohol Level Threshold", 2, 5, 4, 1)

    custom_cfg = {
        "predicted_risk_prob": th_prob,
        "g1_failure_threshold": th_g1,
        "high_failures": th_fail,
        "excessive_absences": th_abs,
        "low_study_time": th_study,
        "high_alcohol": th_alc,
    }

    warn_report = rule_engine.evaluate_rules(
        student,
        pred_b if 'pred_b' in locals() else None,
        custom_thresholds=custom_cfg,
        model_experiment="B",
    )

    warnings = warn_report.get("warnings", [])
    overall_status = warn_report['overall_status']
    status_badge = "badge-safe" if overall_status == "SATISFACTORY" else ("badge-at-risk" if overall_status == "CRITICAL_RISK" else "badge-warning")

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 14px; border-radius:8px;">
            <span style="font-size:13px; color:#64748B;">Overall Evaluated Status: <strong style="color:#14213D;">{overall_status}</strong></span>
            <span class="{status_badge}">{len(warnings)} Warning(s) Triggered</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if warnings:
        warn_data = []
        for w in warnings:
            warn_data.append({
                "Severity": w["severity"],
                "Rule Name": w["rule_name"],
                "Condition": w["condition"],
                "Observed Value": w["observed_value"],
                "Pedagogical Explanation": w["explanation"],
            })
        st.dataframe(warn_data, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="info-box">
                <strong>SAFE PROFILE:</strong> No active deterministic warnings triggered under current thresholds.
            </div>
            """,
            unsafe_allow_html=True,
        )
