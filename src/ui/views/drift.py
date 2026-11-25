"""
ExplainAI - UI View: Academic Drift Monitoring (Signature Feature 3)
Monitors chronological score trajectories across consecutive evaluations (G1 -> G2)
to detect rapid academic performance deterioration prior to final examinations.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any

from src.module2_prediction_reasoning.academic_drift import AcademicDriftDetector


def render_drift_view(
    drift_detector: AcademicDriftDetector,
    student: Dict[str, Any],
    subject: str,
):
    student_id = student["student_id"]

    st.markdown(
        """
        <div class="disclaimer-card">
            <strong>Chronological Early-Warning Monitoring:</strong><br>
            Tracking academic momentum between First Period (G1) and Midterm (G2) detects rapid performance deterioration before final exams (G3):<br>
            • <strong>Critical Decline (Δ ≤ -3.0 pts)</strong>: Rapid downward velocity requiring urgent pedagogical intervention.<br>
            • <strong>Moderate Decline (-3.0 &lt; Δ ≤ -1.5 pts)</strong>: Advisory warning signal for academic reinforcement.<br>
            • <strong>Stable / Improving (Δ &gt; -1.5 pts)</strong>: Academic trajectory maintained or advancing positively.<br>
            <em>Anti-Leakage Scope: G3 is strictly excluded from early-warning drift detection. Records are genuine UCI archives with date_provenance='NOT_PROVIDED_IN_UCI_SOURCE'.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    trajectory = drift_detector.analyze_trajectory(student_id, subject=subject, include_final_grade=False)

    if not trajectory.get("evaluable"):
        st.warning(trajectory.get("message", "Insufficient observations to evaluate drift."))
        return

    obs = trajectory["observations"]
    deltas = trajectory["consecutive_deltas"]
    trend = trajectory["trend"]
    warnings = trajectory["warnings"]

    # --------------------------------------------------------------------------
    # 1. Trajectory KPI Cards Strip (4 Columns)
    # --------------------------------------------------------------------------
    d_val = deltas[0]["score_delta"] if deltas else 0.0
    d_color = "#DC2626" if d_val <= -3.0 else ("#D97706" if d_val < 0 else "#059669")
    trend_badge = "badge-at-risk" if "DECLINE" in trend else "badge-safe"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Term 1 Assessment (G1)</div>
                <div class="metric-value">{obs[0]['score']:.1f} <span style="font-size:13px; color:#64748B; font-weight:400;">/ 20</span></div>
                <div class="metric-subtitle">First Period Evaluation (UCI)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Term 2 Assessment (G2)</div>
                <div class="metric-value">{obs[1]['score']:.1f} <span style="font-size:13px; color:#64748B; font-weight:400;">/ 20</span></div>
                <div class="metric-subtitle">Midterm Evaluation (UCI)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Trajectory Shift (Δ)</div>
                <div class="metric-value" style="color:{d_color};">{d_val:+.1f} <span style="font-size:13px; font-weight:400;">pts</span></div>
                <div class="metric-subtitle">Consecutive score velocity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Deterioration Status</div>
                <div style="margin-top:4px;"><span class="{trend_badge}">{trend}</span></div>
                <div class="metric-subtitle">Evaluated trajectory state</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # 2. Interactive Progression Trajectory Chart
    # --------------------------------------------------------------------------
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="saas-card-header">
            <div>
                <h3 class="saas-card-title">Chronological Term-Over-Term Progression</h3>
                <div class="saas-card-subtitle">Visualizing grade trajectory shifts across consecutive evaluations.</div>
            </div>
            <span class="badge-info">Trajectory Analysis</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    x_terms = [f"Term {o['term_period']} ({o['assessment_name']})" for o in obs]
    y_scores = [o["score"] for o in obs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_terms,
        y=y_scores,
        mode="lines+markers+text",
        name="Student Score",
        line=dict(color="#14213D", width=3, shape="spline"),
        marker=dict(size=12, color=["#3B82F6", d_color], line=dict(color="#FFFFFF", width=2)),
        text=[f"{s:.1f} / 20" for s in y_scores],
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
        height=300,
        margin=dict(l=30, r=30, t=20, b=30),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        yaxis=dict(
            title=dict(text="Evaluation Grade (0 - 20)", font=dict(size=12, color="#64748B")),
            range=[0, 21],
            gridcolor="#E2E8F0",
        ),
        xaxis=dict(
            gridcolor="#E2E8F0",
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # 3. Warning Alert Feed
    # --------------------------------------------------------------------------
    if warnings:
        for w in warnings:
            box_cls = "critical-box" if w["severity"] == "CRITICAL" else "warning-box"
            sev_badge = "badge-at-risk" if w["severity"] == "CRITICAL" else "badge-warning"
            st.markdown(
                f"""
                <div class="{box_cls}">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-weight:700;">[{w['alert_type']}] Trajectory Deterioration Warning</span>
                        <span class="{sev_badge}">{w['severity']} SEVERITY</span>
                    </div>
                    <div>Score dropped by <strong>{abs(w['score_drop']):.1f} points</strong> ({w['from_score']:.1f} → {w['to_score']:.1f}).</div>
                    <div style="font-size:12.5px; margin-top:4px;">{w['message']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="info-box">
                <strong>STABLE TRAJECTORY:</strong> No performance deterioration detected between Term 1 and Term 2 evaluations.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)
