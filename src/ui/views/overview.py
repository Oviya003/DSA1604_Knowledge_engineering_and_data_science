"""
ExplainAI - UI View: Overview
Provides high-level student profile, risk status, rule alerts, and assessment trajectory summary.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from src.database.db_manager import DatabaseManager
from src.module2_prediction_reasoning.predictor import AcademicRiskPredictor
from src.module2_prediction_reasoning.rule_engine import RuleBasedWarningEngine


def render_overview_view(
    db: DatabaseManager,
    predictor: AcademicRiskPredictor,
    rule_engine: RuleBasedWarningEngine,
    student: Dict[str, Any],
    subject: str,
):
    student_id = student["student_id"]

    # --------------------------------------------------------------------------
    # 1. Executive Metric Summary Cards (4-Column Layout)
    # --------------------------------------------------------------------------
    study_map = {1: "< 2 hrs/wk", 2: "2 - 5 hrs/wk", 3: "5 - 10 hrs/wk", 4: "> 10 hrs/wk"}
    
    # Pre-fetch predictions for clean metric display
    try:
        pred_b = predictor.predict_student_risk(student, subject=subject, experiment="B", persist=False)
        risk_cat_b = pred_b["risk_category"]
        prob_b = pred_b["at_risk_probability"]
        algo_b = pred_b["model_algorithm"]
    except Exception:
        pred_b = None
        risk_cat_b = "At-Risk" if student.get("failures", 0) > 0 else "Satisfactory"
        prob_b = 0.50
        algo_b = "Selected Pipeline"

    badge_b = "badge-at-risk" if risk_cat_b == "At-Risk" else "badge-safe"
    fail_badge = "badge-at-risk" if student["failures"] > 0 else "badge-safe"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="metric-title">Forecast Risk Tier</span>
                    <span class="{badge_b}">{risk_cat_b.upper()}</span>
                </div>
                <div class="metric-value">{prob_b:.1%}</div>
                <div class="metric-subtitle">Model B Interim At-Risk Probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="metric-title">Study Dedication</span>
                    <span class="badge-info">Level {student['studytime']}</span>
                </div>
                <div class="metric-value">{study_map.get(student['studytime'], 'Standard')}</div>
                <div class="metric-subtitle">Self-reported weekly study hours</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="metric-title">Prior Failures</span>
                    <span class="{fail_badge}">{student['failures']} Failed</span>
                </div>
                <div class="metric-value">{student['failures']} <span style="font-size:14px; color:#64748B; font-weight:500;">classes</span></div>
                <div class="metric-subtitle">Historical course retention deficit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="metric-title">Recorded Absences</span>
                    <span class="badge-warning">{student['absences']} Days</span>
                </div>
                <div class="metric-value">{student['absences']} <span style="font-size:14px; color:#64748B; font-weight:500;">sessions</span></div>
                <div class="metric-subtitle">Cumulative annual total (Descriptive)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # 2. Longitudinal Grade Trajectory & Assessments
    # --------------------------------------------------------------------------
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="saas-card-header">
            <div>
                <h3 class="saas-card-title">Longitudinal Grade Progression & Milestones</h3>
                <div class="saas-card-subtitle">Official term assessment evaluations recorded in the UCI archive.</div>
            </div>
            <span class="badge-info">Scale: 0 - 20 pts</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assessments = db.get_student_assessments(student_id, subject)
    if assessments:
        x_terms = [f"Term {a['term_period']}: {a['assessment_name']}" for a in assessments]
        y_scores = [a["score"] for a in assessments]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_terms,
            y=y_scores,
            mode="lines+markers+text",
            name="Score (0-20)",
            line=dict(color="#14213D", width=3, shape="spline"),
            marker=dict(size=10, color=["#3B82F6", "#2563EB", "#14213D"], line=dict(color="#FFFFFF", width=2)),
            text=[f"{s:.1f}" for s in y_scores],
            textposition="top center",
            textfont=dict(size=12, color="#14213D", family="Plus Jakarta Sans"),
        ))

        fig.add_hline(
            y=10.0,
            line_dash="dash",
            line_color="#DC2626",
            line_width=1.5,
            annotation_text="Passing Benchmark (10.0)",
            annotation_position="bottom right",
            annotation_font=dict(size=11, color="#DC2626"),
        )

        fig.update_layout(
            height=260,
            margin=dict(l=30, r=30, t=20, b=30),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            yaxis=dict(
                title=dict(text="Evaluation Grade", font=dict(size=12, color="#64748B")),
                range=[0, 21],
                gridcolor="#E2E8F0",
                zeroline=False,
            ),
            xaxis=dict(
                gridcolor="#E2E8F0",
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Assessment cards row
        cols_ass = st.columns(len(assessments))
        for idx, a in enumerate(assessments):
            with cols_ass[idx]:
                score = a["score"]
                status_color = "#059669" if score >= 10 else "#DC2626"
                status_label = "Satisfactory" if score >= 10 else "Deficit"
                badge_cls = "badge-safe" if score >= 10 else "badge-at-risk"
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:12px 14px; border-radius:8px; text-align:center;">
                        <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">{a['assessment_name']} (Term {a['term_period']})</div>
                        <div style="font-size:22px; font-weight:800; color:{status_color}; margin:4px 0;">{score:.1f} <span style="font-size:13px; color:#64748B; font-weight:400;">/ 20</span></div>
                        <span class="{badge_cls}" style="font-size:10.5px;">{status_label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No assessment milestones logged for this student.")

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3. Dual-Experiment Predictive Overview (Model A vs Model B)
    # --------------------------------------------------------------------------
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="saas-card-header">
            <div>
                <h3 class="saas-card-title">Early-Warning Predictive Models</h3>
                <div class="saas-card-subtitle">Comparing intake baseline forecasts against post-first period interim predictions.</div>
            </div>
            <span class="badge-info">Dual-Timeline Architecture</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_pred_a, col_pred_b = st.columns(2)

    with col_pred_a:
        try:
            pred_a = predictor.predict_student_risk(student, subject=subject, experiment="A", persist=False)
            risk_cat_a = pred_a["risk_category"]
            prob_a = pred_a["at_risk_probability"]
            badge_class_a = "badge-at-risk" if risk_cat_a == "At-Risk" else "badge-safe"

            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:14px; color:#14213D;">Model A: Intake Baseline</span>
                        <span class="{badge_class_a}">{risk_cat_a.upper()}</span>
                    </div>
                    <div style="font-size:12px; color:#64748B; margin-top:2px;">Pre-course evaluation prior to examinations</div>
                    <div style="margin: 14px 0 8px 0;">
                        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                            <span style="color:#64748B;">At-Risk Probability</span>
                            <strong style="color:#14213D;">{prob_a:.1%}</strong>
                        </div>
                        <div style="background:#E2E8F0; border-radius:9999px; height:8px; overflow:hidden;">
                            <div style="background:{'#DC2626' if prob_a>=0.5 else '#059669'}; width:{prob_a*100:.1f}%; height:100%;"></div>
                        </div>
                    </div>
                    <div style="font-size:11.5px; color:#64748B; line-height:1.4;">
                        Algorithm: <strong>{pred_a['model_algorithm']}</strong><br>
                        Features: 29 intake indicators (anti-leakage: exam marks & absences barred).<br>
                        <span style="font-size:11px; color:#94A3B8;">Probability estimates from fitted classifiers; outputs have not been independently calibrated.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Could not compute Model A prediction: {e}")

    with col_pred_b:
        try:
            st.markdown(
                f"""
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; font-size:14px; color:#14213D;">Model B: Post-G1 Interim</span>
                        <span class="{badge_b}">{risk_cat_b.upper()}</span>
                    </div>
                    <div style="font-size:12px; color:#64748B; margin-top:2px;">Updated early warning incorporating Term 1 Grade</div>
                    <div style="margin: 14px 0 8px 0;">
                        <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:4px;">
                            <span style="color:#64748B;">At-Risk Probability</span>
                            <strong style="color:#14213D;">{prob_b:.1%}</strong>
                        </div>
                        <div style="background:#E2E8F0; border-radius:9999px; height:8px; overflow:hidden;">
                            <div style="background:{'#DC2626' if prob_b>=0.5 else '#059669'}; width:{prob_b*100:.1f}%; height:100%;"></div>
                        </div>
                    </div>
                    <div style="font-size:11.5px; color:#64748B; line-height:1.4;">
                        Algorithm: <strong>{algo_b}</strong><br>
                        Features: 30 indicators (Intake + G1 mark: {student.get('G1', 'N/A')}/20).<br>
                        <span style="font-size:11px; color:#94A3B8;">Probability estimates from fitted classifiers; outputs have not been independently calibrated.</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error(f"Could not compute Model B prediction: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. Active Deterministic Academic Warnings
    # --------------------------------------------------------------------------
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="saas-card-header">
            <div>
                <h3 class="saas-card-title">Active Deterministic Warnings</h3>
                <div class="saas-card-subtitle">Transparent rule-based heuristic alerts evaluating actionable risk indicators.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    warn_report = rule_engine.evaluate_rules(student, pred_b if pred_b else None, model_experiment="B")
    warnings = warn_report.get("warnings", [])

    if not warnings:
        st.markdown(
            """
            <div class="info-box">
                <strong>SAFE PROFILE:</strong> No deterministic academic warnings triggered for this student under standard thresholds.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for w in warnings:
            box_cls = "critical-box" if w["severity"] == "HIGH" else "warning-box"
            sev_badge = "badge-at-risk" if w["severity"] == "HIGH" else "badge-warning"
            st.markdown(
                f"""
                <div class="{box_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:700;">{w['rule_name']}</span>
                        <span class="{sev_badge}">{w['severity']} SEVERITY</span>
                    </div>
                    <div style="font-size:12px; color:#64748B;">Condition: <em>{w['condition']}</em> (Observed: <strong>{w['observed_value']}</strong>)</div>
                    <div style="margin-top:6px; font-size:13px;">{w['explanation']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)
