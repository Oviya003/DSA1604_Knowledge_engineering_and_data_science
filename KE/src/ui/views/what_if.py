"""
ExplainAI - UI View: Academic What-If Simulator (Signature Feature 1)
Provides counterfactual sensitivity simulation across actionable variables with causal disclaimers.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from src.module2_prediction_reasoning.what_if_simulator import (
    AcademicWhatIfSimulator,
    ACTIONABLE_ALLOWLIST,
    SCIENTIFIC_DISCLAIMER_TEXT,
)


def render_what_if_view(
    simulator: AcademicWhatIfSimulator,
    student: Dict[str, Any],
    subject: str,
):
    student_id = student["student_id"]

    st.markdown(
        f"""
        <div class="disclaimer-card" style="border-left:4px solid #F59E0B;">
            <strong>Hypothetical Simulation Methodology & Causal Disclaimer:</strong><br>
            {SCIENTIFIC_DISCLAIMER_TEXT}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 1. Actionable Levers vs Locked Indicators (Two-Panel Card)
    # --------------------------------------------------------------------------
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="saas-card-header">
            <div>
                <h3 class="saas-card-title">Scenario Formulation: Actionable Academic Variables</h3>
                <div class="saas-card-subtitle">Adjust modifiable study and institutional support variables to observe associative model shifts.</div>
            </div>
            <span class="badge-warning">Hypothetical Mode</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Study Habits & Dedication**")
        cur_study = student.get("studytime", 1)
        new_study = st.slider(
            "Weekly Study Time Level",
            min_value=1,
            max_value=4,
            value=int(cur_study),
            help="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h",
        )

        cur_free = student.get("freetime", 3)
        new_free = st.slider("Free Time After School (1-5)", 1, 5, int(cur_free))

    with col2:
        st.markdown("**2. Social & Extracurricular**")
        cur_goout = student.get("goout", 3)
        new_goout = st.slider("Going Out with Friends (1-5)", 1, 5, int(cur_goout))

        cur_fail = student.get("failures", 0)
        st.text_input(
            "Historical Class Failures (Locked)",
            value=f"{cur_fail} past failure(s)",
            disabled=True,
            help="Historical past failures cannot be altered retrospectively.",
        )

    with col3:
        st.markdown("**3. Institutional & Family Support**")
        cur_schoolsup = student.get("schoolsup", "no")
        new_schoolsup = st.selectbox("Extra School Support", ["no", "yes"], index=0 if cur_schoolsup=="no" else 1)

        cur_famsup = student.get("famsup", "no")
        new_famsup = st.selectbox("Family Educational Support", ["no", "yes"], index=0 if cur_famsup=="no" else 1)

    # Collect proposed changes
    proposed = {}
    if new_study != cur_study:
        proposed["studytime"] = new_study
    if new_free != cur_free:
        proposed["freetime"] = new_free
    if new_goout != cur_goout:
        proposed["goout"] = new_goout
    if new_schoolsup != cur_schoolsup:
        proposed["schoolsup"] = new_schoolsup
    if new_famsup != cur_famsup:
        proposed["famsup"] = new_famsup

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        exp_choice = st.radio("Simulation Model Baseline:", ["Model A (Intake Baseline)", "Model B (Post-G1)"], horizontal=True)
        exp_letter = "A" if "Model A" in exp_choice else "B"
    with col_ctrl2:
        if proposed:
            st.markdown(f"<div style='margin-top:20px; font-size:12px; color:#2563EB;'><strong>{len(proposed)}</strong> variable(s) modified</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-top:20px; font-size:12px; color:#64748B;'>Adjust sliders above to test scenarios</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. Simulation Outcome Comparison
    # --------------------------------------------------------------------------
    try:
        sim_res = simulator.simulate_counterfactual(
            original_student_data=student,
            proposed_changes=proposed,
            subject=subject,
            experiment=exp_letter,
            log_to_audit=True,
        )

        base = sim_res["baseline"]
        hypo = sim_res["hypothetical"]
        delta = sim_res["delta"]

        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="saas-card-header">
                <div>
                    <h3 class="saas-card-title">Counterfactual Risk Shift Analysis</h3>
                    <div class="saas-card-subtitle">Evaluating simulated probability differential against baseline profile.</div>
                </div>
                <span class="badge-info">Sensitivity Model</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            base_badge = "badge-at-risk" if base["risk_category"] == "At-Risk" else "badge-safe"
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase;">Baseline Profile</span>
                        <span class="{base_badge}">{base['risk_category'].upper()}</span>
                    </div>
                    <div style="font-size:26px; font-weight:800; color:#14213D; margin-top:6px;">{base['at_risk_probability']:.1%}</div>
                    <div style="font-size:12px; color:#64748B; margin-top:4px;">Actual recorded attributes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col2:
            hypo_badge = "badge-at-risk" if hypo["risk_category"] == "At-Risk" else "badge-safe"
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase;">Hypothetical Scenario</span>
                        <span class="{hypo_badge}">{hypo['risk_category'].upper()}</span>
                    </div>
                    <div style="font-size:26px; font-weight:800; color:#3B82F6; margin-top:6px;">{hypo['at_risk_probability']:.1%}</div>
                    <div style="font-size:12px; color:#64748B; margin-top:4px;">Simulated counterfactual profile</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_col3:
            prob_delta = delta.get("probability_difference", delta.get("at_risk_probability_delta", 0.0))
            delta_color = "#059669" if prob_delta < 0 else ("#DC2626" if prob_delta > 0 else "#64748B")
            delta_symbol = "▼" if prob_delta < 0 else ("▲" if prob_delta > 0 else "■")
            effect_text = delta.get("effect_direction", "NO_MEANINGFUL_CHANGE").replace("_", " ").title()
            delta_badge = "badge-safe" if prob_delta < 0 else ("badge-at-risk" if prob_delta > 0 else "badge-info")

            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:11.5px; font-weight:700; color:#64748B; text-transform:uppercase;">Sensitivity Shift (Δ p)</span>
                        <span class="{delta_badge}">{effect_text}</span>
                    </div>
                    <div style="font-size:26px; font-weight:800; color:{delta_color}; margin-top:6px;">{delta_symbol} {abs(prob_delta):.1%}</div>
                    <div style="font-size:12px; color:#64748B; margin-top:4px;">{'Risk reduction observed' if prob_delta < 0 else ('Risk escalation observed' if prob_delta > 0 else 'Zero net shift')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Plotly comparison bar chart
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Baseline Profile",
            x=["At-Risk Probability"],
            y=[base["at_risk_probability"] * 100],
            marker=dict(color="#94A3B8", line=dict(color="#FFFFFF", width=1.5)),
            text=[f"{base['at_risk_probability']:.1%}"],
            textposition="auto",
            textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans"),
        ))
        fig_comp.add_trace(go.Bar(
            name="Hypothetical Scenario",
            x=["At-Risk Probability"],
            y=[hypo["at_risk_probability"] * 100],
            marker=dict(color="#3B82F6", line=dict(color="#FFFFFF", width=1.5)),
            text=[f"{hypo['at_risk_probability']:.1%}"],
            textposition="auto",
            textfont=dict(size=12, color="#FFFFFF", family="Plus Jakarta Sans"),
        ))
        fig_comp.update_layout(
            barmode="group",
            yaxis=dict(
                title=dict(text="Predicted At-Risk Probability (%)", font=dict(size=11, color="#64748B")),
                range=[0, 100],
                gridcolor="#F1F5F9",
            ),
            xaxis=dict(
                gridcolor="#F1F5F9",
            ),
            height=260,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color="#14213D"),
            ),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown(
            '<div style="font-size:11px; color:#64748B; margin-top:4px; text-align:center;">'
            '<em>* Probability estimates from fitted classifiers; outputs have not been independently calibrated, and no confidence intervals are provided.</em>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"What-If Simulation error: {e}")
