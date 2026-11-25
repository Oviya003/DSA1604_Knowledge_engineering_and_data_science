"""
ExplainAI - UI View: SHAP Explainability
Visualizes individual feature contributions and global feature importances
using the actual serialized models and training reference background.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

from src.module3_explainability_support.shap_explainer import (
    AcademicShapExplainer,
    SHAP_SCIENTIFIC_DISCLAIMER,
)


def render_explainability_view(
    shap_service: AcademicShapExplainer,
    student: Dict[str, Any],
    subject: str,
):
    student_id = student["student_id"]

    st.markdown(
        f"""
        <div class="disclaimer-card">
            <strong>Interpreting SHAP Values (SHapley Additive exPlanations):</strong><br>
            SHAP quantifies the contribution of each student attribute toward shifting the predicted risk from the base rate.
            <br><em>{SHAP_SCIENTIFIC_DISCLAIMER}</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_local, tab_global = st.tabs(["Local Student Attribution", "Global Cohort Feature Importance"])

    with tab_local:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="saas-card-header">
                <div>
                    <h3 class="saas-card-title">Instance-Level Risk Factor Decomposition</h3>
                    <div class="saas-card-subtitle">Local feature attributions for this specific student profile.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        exp_choice = st.radio("Select Prediction Experiment to Explain:", ["Model A (Intake)", "Model B (Post-G1)"], horizontal=True)
        exp_letter = "A" if "Model A" in exp_choice else "B"

        with st.spinner("Computing SHAP values for student instance..."):
            try:
                shap_result = shap_service.explain_instance(student, subject=subject, experiment=exp_letter)
                cat = shap_result["predicted_risk_category"]
                prob = shap_result["at_risk_probability"]
                badge_cls = "badge-at-risk" if cat == "At-Risk" else "badge-safe"

                # Metadata chips row
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.markdown(
                        f"""
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 14px; border-radius:8px;">
                            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Predicted Risk</div>
                            <div style="margin-top:4px;"><span class="{badge_cls}">{cat.upper()}</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_m2:
                    st.markdown(
                        f"""
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 14px; border-radius:8px;">
                            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Risk Probability</div>
                            <div style="font-size:18px; font-weight:800; color:#14213D; margin-top:2px;">{prob:.1%}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_m3:
                    st.markdown(
                        f"""
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 14px; border-radius:8px;">
                            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Model Family</div>
                            <div style="font-size:14px; font-weight:700; color:#14213D; margin-top:4px;">{shap_result['classifier_type']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_m4:
                    st.markdown(
                        f"""
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 14px; border-radius:8px;">
                            <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Output Space</div>
                            <div style="font-size:14px; font-weight:700; color:#2563EB; margin-top:4px;">{shap_result.get('output_units', 'margin')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"""
                    <div style="font-size:12px; color:#64748B; background:#F1F5F9; padding:8px 12px; border-radius:6px; margin:12px 0;">
                        <strong>Mathematical Space Interpretation:</strong> SHAP values are evaluated on the <strong>{shap_result.get('output_units', 'margin')}</strong> scale.
                        """
                        + (" For Logistic Regression and XGBoost, attributions represent log-odds shifts before sigmoid conversion; they are not additive percentage points."
                           if "log-odds" in shap_result.get("output_units", "") else
                           " For Random Forest, attributions sum additively to class probabilities.")
                        + """
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Feature attribution horizontal bar chart
                top_inc = shap_result["top_risk_increasing_factors"]
                top_dec = shap_result["top_risk_decreasing_factors"]
                plot_items = list(reversed(top_inc + top_dec))

                if plot_items:
                    y_labels = [f"{item['display_name']} ({item['original_value']})" for item in plot_items]
                    x_vals = [item["shap_value"] for item in plot_items]
                    bar_colors = ["#DC2626" if v > 0 else "#059669" for v in x_vals]

                    fig = go.Figure(go.Bar(
                        x=x_vals,
                        y=y_labels,
                        orientation="h",
                        marker=dict(color=bar_colors, line=dict(color="#FFFFFF", width=1)),
                        text=[f"{v:+.3f}" for v in x_vals],
                        textposition="auto",
                        textfont=dict(size=11, family="Plus Jakarta Sans"),
                    ))
                    fig.update_layout(
                        title=dict(
                            text=f"Key Feature Attributions ({shap_result['experiment']}) — Red Increases Risk, Green Reduces Risk",
                            font=dict(size=13, color="#14213D", family="Plus Jakarta Sans"),
                        ),
                        xaxis=dict(
                            title=dict(text=f"SHAP Attribution ({shap_result.get('output_units', 'log-odds')})", font=dict(size=11, color="#64748B")),
                            gridcolor="#F1F5F9",
                            zerolinecolor="#94A3B8",
                            zerolinewidth=1.5,
                        ),
                        yaxis=dict(
                            gridcolor="#F1F5F9",
                            tickfont=dict(size=11, color="#334155"),
                        ),
                        height=380,
                        margin=dict(l=10, r=20, t=40, b=30),
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#F8FAFC",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Aggregated Original Profile Features
                st.markdown("#### Primary Demographic & Profile Variables")
                st.markdown("<span style='font-size:12px; color:#64748B;'>Aggregates dummy-encoded categorical indicators back to single profile attributes.</span>", unsafe_allow_html=True)
                agg_df = pd.DataFrame(shap_result["aggregated_original_features"])
                agg_df = agg_df.rename(columns={
                    "feature": "Feature",
                    "original_value": "Student Value",
                    "net_shap_value": "Net SHAP Value",
                    "abs_net_shap": "Absolute Impact",
                    "impact_direction": "Direction",
                })
                st.dataframe(agg_df.head(15), use_container_width=True)

            except Exception as e:
                st.error(f"Error computing local SHAP explanation: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab_global:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="saas-card-header">
                <div>
                    <h3 class="saas-card-title">Cohort-Level Feature Importance ({subject})</h3>
                    <div class="saas-card-subtitle">Global mean absolute SHAP values across the entire training population.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        exp_global = st.radio("Global Experiment:", ["Model A (Intake)", "Model B (Post-G1)"], horizontal=True, key="global_exp")
        exp_glob_letter = "A" if "Model A" in exp_global else "B"

        with st.spinner("Calculating global feature importance across training cohort..."):
            try:
                glob_res = shap_service.compute_global_feature_importance(subject, exp_glob_letter, max_samples=100)
                ranked_orig = glob_res["ranked_original_features"]
                top_12 = ranked_orig[:12]
                y_feat = [f["original_feature"] for f in reversed(top_12)]
                x_imp = [f["sum_abs_shap"] for f in reversed(top_12)]

                fig_glob = go.Figure(go.Bar(
                    x=x_imp,
                    y=y_feat,
                    orientation="h",
                    marker=dict(color="#3B82F6", line=dict(color="#FFFFFF", width=1)),
                    text=[f"{v:.3f}" for v in x_imp],
                    textposition="auto",
                    textfont=dict(size=11, family="Plus Jakarta Sans"),
                ))
                fig_glob.update_layout(
                    title=dict(
                        text=f"Global Feature Importance Ranking ({glob_res['experiment']})",
                        font=dict(size=13, color="#14213D", family="Plus Jakarta Sans"),
                    ),
                    xaxis=dict(
                        title=dict(text="Mean Absolute SHAP Value (Impact Magnitude)", font=dict(size=11, color="#64748B")),
                        gridcolor="#F1F5F9",
                    ),
                    yaxis=dict(
                        gridcolor="#F1F5F9",
                        tickfont=dict(size=11, color="#334155"),
                    ),
                    height=420,
                    margin=dict(l=10, r=20, t=40, b=30),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#F8FAFC",
                )
                st.plotly_chart(fig_glob, use_container_width=True)

                st.markdown("#### Complete Cohort Feature Hierarchy")
                st.dataframe(pd.DataFrame(ranked_orig), use_container_width=True)

            except Exception as e:
                st.error(f"Error computing global SHAP importance: {e}")

        st.markdown('</div>', unsafe_allow_html=True)
