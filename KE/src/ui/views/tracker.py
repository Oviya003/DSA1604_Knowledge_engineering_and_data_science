"""
ExplainAI - UI View: Closed-Loop Learning Improvement Tracker (Signature Feature 4)
Manages the end-to-end educational intervention lifecycle with longitudinal outcome verification.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

from src.database.db_manager import DatabaseManager
from src.module3_explainability_support.improvement_tracker import (
    ClosedLoopImprovementTracker,
    TRACKER_CAUSAL_DISCLAIMER,
)
from src.module3_explainability_support.knowledge_gap_detective import (
    KnowledgeGapDetective,
    MASTERY_THRESHOLD_MASTERED,
)


def render_tracker_view(
    db: DatabaseManager,
    tracker: ClosedLoopImprovementTracker,
    gap_detective: KnowledgeGapDetective,
    student: Dict[str, Any],
    subject: str,
):
    student_id = student["student_id"]

    # Header section
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1.25rem;">
            <div>
                <h2 style="margin:0; font-size:1.5rem; font-weight:800; color:#14213D; letter-spacing:-0.02em;">
                    Closed-Loop Learning Improvement Tracker
                </h2>
                <div style="color:#64748B; font-size:0.875rem; margin-top:0.25rem;">
                    Signature Feature 4 &bull; Longitudinal tracking of diagnostic reassessment and observed concept progress
                </div>
            </div>
            <div style="display:flex; gap:0.5rem; align-items:center;">
                <span class="badge badge-info" style="font-size:0.8rem; padding:0.35rem 0.75rem;">
                    {student_id} &bull; {subject}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Required Causal Disclaimer Card
    st.markdown(
        f"""
        <div class="disclaimer-card" style="margin-bottom:1.5rem;">
            <div style="display:flex; align-items:flex-start; gap:0.6rem;">
                <span style="font-size:1.1rem; line-height:1;">⚖️</span>
                <div>
                    <strong style="color:#92400E; font-size:0.825rem; text-transform:uppercase; letter-spacing:0.04em;">
                        Intervention Efficacy Notice
                    </strong>
                    <div style="color:#78350F; font-size:0.825rem; margin-top:0.25rem; line-height:1.45;">
                        {TRACKER_CAUSAL_DISCLAIMER}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs for Workflow
    tab_overview, tab_prescribe, tab_reassess = st.tabs([
        "📋 Intervention History & Progress",
        "✍️ Prescribe New Intervention",
        "🎯 Reassessment & Outcome Loop",
    ])

    with tab_overview:
        history = tracker.get_student_intervention_history(student_id, subject=subject)
        counts = history["status_breakdown"]

        # KPI Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Interventions</div>
                    <div class="metric-value">{history['total_interventions']}</div>
                    <div class="metric-subtitle">Lifetime logged actions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top:3px solid #D97706;">
                    <div class="metric-title">Prescribed / Pending</div>
                    <div class="metric-value" style="color:#D97706;">{counts.get('PRESCRIBED', 0)}</div>
                    <div class="metric-subtitle">Awaiting student start</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top:3px solid #3B82F6;">
                    <div class="metric-title">In Progress</div>
                    <div class="metric-value" style="color:#3B82F6;">{counts.get('IN_PROGRESS', 0)}</div>
                    <div class="metric-subtitle">Active remediation active</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="metric-card" style="border-top:3px solid #059669;">
                    <div class="metric-title">Completed & Reassessed</div>
                    <div class="metric-value" style="color:#059669;">{counts.get('COMPLETED', 0)}</div>
                    <div class="metric-subtitle">Evaluated via diagnostic quiz</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### 📜 Intervention Records Log")
        interventions = history.get("interventions", [])
        if not interventions:
            st.info("No educational interventions currently recorded for this student.")
        else:
            table_data = []
            for item in interventions:
                post_sc = f"{item['post_quiz_score']:.0%}" if item.get('post_quiz_score') is not None else "Pending"
                table_data.append({
                    "ID": f"#{item['intervention_id']}",
                    "Target Concept": f"{item['target_concept_id']} ({item.get('concept_name', '')})",
                    "Modality": item["intervention_type"].replace("_", " "),
                    "Status": item["status"],
                    "Prescribed Date": item["prescribed_date"][:10] if item.get("prescribed_date") else "",
                    "Post-Quiz Outcome": post_sc,
                    "Notes": item.get("notes", ""),
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            # Before-and-After Progress Chart for Completed Interventions
            completed_items = [i for i in interventions if i["status"] == "COMPLETED" and i.get("post_quiz_score") is not None]
            if completed_items:
                st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
                st.markdown("#### 📈 Observed Outcome Progress (Post-Intervention)")
                concept_names = [f"{i['target_concept_id']}" for i in completed_items]
                post_scores = [i["post_quiz_score"] * 100 for i in completed_items]

                fig = go.Figure(go.Bar(
                    x=concept_names,
                    y=post_scores,
                    marker=dict(
                        color="#059669",
                        line=dict(color="#047857", width=1.5),
                    ),
                    text=[f"{s:.1f}%" for s in post_scores],
                    textposition="auto",
                    textfont=dict(size=12, color="#FFFFFF", family="Inter, sans-serif"),
                ))
                fig.add_hline(
                    y=MASTERY_THRESHOLD_MASTERED * 100,
                    line_dash="dash",
                    line_color="#3B82F6",
                    line_width=2,
                    annotation_text=f"Mastery Threshold ({MASTERY_THRESHOLD_MASTERED:.0%})",
                    annotation_position="top right",
                    annotation_font=dict(size=11, color="#1D4ED8", family="Inter, sans-serif"),
                )
                fig.update_layout(
                    title=dict(
                        text="Post-Intervention Diagnostic Quiz Performance",
                        font=dict(size=13, color="#14213D", family="Inter, sans-serif"),
                    ),
                    xaxis=dict(
                        title=dict(text="Concept Node", font=dict(size=12, color="#64748B")),
                        tickfont=dict(size=11, color="#1E293B"),
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title=dict(text="Score (%)", font=dict(size=12, color="#64748B")),
                        range=[0, 105],
                        gridcolor="#F1F5F9",
                        tickfont=dict(size=11, color="#1E293B"),
                    ),
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    height=320,
                    margin=dict(l=20, r=20, t=45, b=25),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"Configured threshold: Post-quiz score ≥ {MASTERY_THRESHOLD_MASTERED:.0%} classifies concept node as MASTERED in KnowledgeGapDetective. "
                    "Demonstrated score changes evaluate diagnostic item performance and do not establish causal improvement in overall course grades."
                )

    with tab_prescribe:
        st.markdown(
            """
            <div style="margin-bottom:1rem;">
                <h4 style="margin:0 0 0.25rem 0; font-size:1.15rem; font-weight:700; color:#14213D;">
                    Prescribe Targeted Educational Intervention
                </h4>
                <div style="color:#64748B; font-size:0.875rem;">
                    Select curriculum concept nodes flagged for remediation and assign tailored pedagogical action plans.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not gap_detective.is_subject_supported(subject):
            st.warning("Intervention concept mapping is currently restricted to Mathematics curriculum concepts.")
        else:
            concepts = db.get_all_concepts(subject="Mathematics")
            c_options = {c["concept_id"]: f"{c['concept_id']}: {c['name']} ({c['difficulty']})" for c in concepts}

            with st.form("prescribe_form"):
                sel_concept = st.selectbox(
                    "Target Concept Node:",
                    options=list(c_options.keys()),
                    format_func=lambda x: c_options[x],
                    help="Select the specific curriculum concept node requiring educational support.",
                )
                sel_type = st.selectbox(
                    "Intervention Modality:",
                    [
                        "REMEDIAL_EXERCISES",
                        "PEER_TUTORING",
                        "TEACHER_1ON1",
                        "CONCEPT_REVISION",
                    ],
                    format_func=lambda x: x.replace("_", " ").title(),
                    help="Educational support delivery channel.",
                )
                prescribed_by = st.text_input(
                    "Prescribing Educator / Team:",
                    value="Academic Support Center",
                    help="Name or department assigning this intervention.",
                )
                notes = st.text_area(
                    "Support Action Plan & Student Goals:",
                    value="Complete targeted foundational problem sets and attend peer review sessions.",
                    rows=3,
                    help="Explicit actionable instructions for the student and mentor.",
                )
                submit_presc = st.form_submit_button("Prescribe Intervention", type="primary", use_container_width=True)

            if submit_presc:
                res = tracker.prescribe_intervention(
                    student_id=student_id,
                    subject=subject,
                    target_concept_id=sel_concept,
                    intervention_type=sel_type,
                    notes=notes,
                    prescribed_by=prescribed_by,
                )
                st.success(res["message"])
                st.rerun()

    with tab_reassess:
        st.markdown(
            """
            <div style="margin-bottom:1rem;">
                <h4 style="margin:0 0 0.25rem 0; font-size:1.15rem; font-weight:700; color:#14213D;">
                    Reassessment & Outcome Verification Loop
                </h4>
                <div style="color:#64748B; font-size:0.875rem;">
                    Administer post-intervention diagnostic quizzes to verify concept recovery and update student mastery status.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_user_int = db.get_student_interventions(student_id, subject=subject)
        active_ints = [i for i in all_user_int if i["status"] in ["PRESCRIBED", "IN_PROGRESS"]]

        if not active_ints:
            st.info("No active pending interventions available for reassessment.")
        else:
            int_map = {
                i["intervention_id"]: f"#{i['intervention_id']} - {i['target_concept_id']} ({i.get('concept_name', '')}) [{i['status']}]"
                for i in active_ints
            }
            chosen_id = st.selectbox(
                "Select Active Intervention:",
                options=list(int_map.keys()),
                format_func=lambda x: int_map[x],
            )
            chosen_rec = next(i for i in active_ints if i["intervention_id"] == chosen_id)

            if chosen_rec["status"] == "PRESCRIBED":
                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:1.25rem; margin:1rem 0;">
                        <div style="font-weight:700; color:#14213D; margin-bottom:0.25rem;">
                            Intervention #{chosen_id} is currently Prescribed
                        </div>
                        <div style="color:#64748B; font-size:0.875rem; margin-bottom:1rem;">
                            Before running post-intervention reassessment quizzes, mark the intervention as in progress once the student begins remediation.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Mark Intervention as In Progress", type="primary", use_container_width=True):
                    tracker.start_intervention(chosen_id)
                    st.success(f"Intervention #{chosen_id} transitioned to IN_PROGRESS.")
                    st.rerun()

            elif chosen_rec["status"] == "IN_PROGRESS":
                target_cid = chosen_rec["target_concept_id"]
                st.markdown(
                    f"""
                    <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:1rem 1.25rem; margin:1rem 0;">
                        <span style="font-weight:700; color:#1E40AF;">Diagnostic Reassessment:</span>
                        <span style="color:#1E3A8A; font-size:0.9rem;">
                            Administer post-intervention check for <code>{target_cid}</code> to verify empirical learning gains.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                q_data = gap_detective.get_concept_questions(target_cid, subject=subject)
                questions = q_data.get("questions", [])

                if not questions:
                    st.info("No questions found for this concept node.")
                else:
                    with st.form(f"reassess_form_{chosen_id}"):
                        reassess_answers = {}
                        for idx, q in enumerate(questions, start=1):
                            st.markdown(
                                f"""
                                <div style="font-weight:600; color:#14213D; font-size:0.95rem; margin-bottom:0.25rem;">
                                    Question {idx}: {q['question_text']}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            ans = st.radio(
                                f"Select answer for Question {idx}:",
                                options=["A", "B", "C", "D"],
                                format_func=lambda x: f"{x}) {q['options'][x]}",
                                key=f"re_{q['question_id']}",
                                label_visibility="collapsed",
                            )
                            reassess_answers[q["question_id"]] = ans
                            st.markdown("<div style='height:0.75rem; border-bottom:1px dashed #E2E8F0; margin-bottom:1rem;'></div>", unsafe_allow_html=True)

                        complete_btn = st.form_submit_button("Submit Reassessment & Evaluate Outcome", type="primary", use_container_width=True)

                    if complete_btn:
                        comp_res = tracker.complete_intervention_with_reassessment(
                            intervention_id=chosen_id,
                            post_quiz_answers=reassess_answers,
                            quiz_type="POST_INTERVENTION_CHECK",
                        )

                        st.success(f"Intervention #{chosen_id} marked COMPLETED; diagnostic reassessment recorded.")
                        delta_pct = comp_res["improvement_percentage"]
                        delta_color = "#059669" if delta_pct > 0 else "#64748B"
                        arrow = "▲" if delta_pct > 0 else ("▼" if delta_pct < 0 else "")

                        st.markdown(
                            f"""
                            <div class="metric-card" style="border-left:4px solid {delta_color}; margin-top:1rem;">
                                <div class="metric-title">Diagnostic Reassessment Outcome Summary</div>
                                <div class="metric-value" style="color:{delta_color}; font-size:1.4rem;">
                                    Baseline: {comp_res['baseline_score']:.0%} &rarr; Post-Quiz: {comp_res['post_quiz_score']:.0%}
                                    <span style="font-size:1rem; margin-left:0.5rem; background:#F1F5F9; padding:0.2rem 0.5rem; border-radius:4px;">
                                        {arrow} {delta_pct:+.1f}%
                                    </span>
                                </div>
                                <div class="metric-subtitle" style="margin-top:0.5rem;">
                                    Evaluated Mastery Status: <b>{comp_res['new_mastery_status']}</b> &bull;
                                    Assessment: <b>{comp_res['outcome_assessment']}</b>
                                </div>
                                <div style="font-size:11px; color:#64748B; margin-top:8px; border-top:1px dashed #E2E8F0; padding-top:6px;">
                                    * Note: Activity completion is distinct from demonstrated mastery. Demonstrated progress on this diagnostic quiz reflects item performance and does not constitute causal proof of course-level improvement.
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.info(f"Recommended Next Action: {comp_res['next_action']}")
