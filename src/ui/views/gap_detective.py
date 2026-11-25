"""
ExplainAI - UI View: Prerequisite-Aware Knowledge Gap Detective (Signature Feature 2)
Curriculum-guided diagnostic testing, mastery estimation, and DAG prerequisite gap tracing.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from src.module3_explainability_support.knowledge_gap_detective import (
    KnowledgeGapDetective,
    COARSE_ASSESSMENT_LIMITATION,
)
from src.module1_data_knowledge.curriculum_catalog import get_unsupported_subject_message


def render_gap_detective_view(
    gap_detective: KnowledgeGapDetective,
    student: Dict[str, Any],
    subject: str,
):
    student_id = student["student_id"]

    # Subject Restriction Guard
    if not gap_detective.is_subject_supported(subject):
        st.markdown(
            f"""
            <div class="warning-box">
                <strong>UNSUPPORTED SUBJECT FOR CURRICULUM ANALYSIS:</strong><br>
                {get_unsupported_subject_message(subject)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("💡 Please select Mathematics in the sidebar to explore curriculum-guided diagnostic testing and prerequisite gap analysis.")
        return

    st.markdown(
        f"""
        <div class="disclaimer-card">
            <strong>Demonstration Curriculum Notice & Psychometric Scope:</strong><br>
            The knowledge graph represents an academic <strong>Demonstration Curriculum</strong> (10 concepts, 11 prerequisite edges) developed for ExplainAI.
            It is an educational demonstration model and <em>not</em> an officially certified national syllabus.
            The 20 application-created diagnostic questions are formative demonstration questions linked to these Mathematics concepts.<br>
            <em>{COARSE_ASSESSMENT_LIMITATION}</em>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_concepts = gap_detective.kg.get_all_concepts_details()
    concept_map = {c["concept_id"]: f"{c['concept_id']}: {c['name']} ({c['difficulty']})" for c in all_concepts}
    
    st.markdown('<div class="saas-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="saas-card-header">
            <div>
                <h3 class="saas-card-title">Curriculum Target Concept Selection</h3>
                <div class="saas-card-subtitle">Choose a concept node from the Mathematics DAG to diagnose readiness or take a quiz.</div>
            </div>
            <span class="badge-info">Mathematics DAG</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_cid = st.selectbox(
        "Select Curriculum Target Concept to Diagnose / Study:",
        options=list(concept_map.keys()),
        format_func=lambda x: concept_map[x],
        index=4,  # Defaults to MATH_05: Linear Equations
        label_visibility="collapsed",
    )

    gap_analysis = gap_detective.analyze_prerequisite_gaps(student_id, selected_cid, subject=subject)
    t_info = gap_analysis["target_concept"]

    verdict = gap_analysis["readiness_verdict"]
    v_badge = "badge-safe" if "READY_TO_STUDY" in verdict else ("badge-at-risk" if "NOT_READY" in verdict else "badge-warning")

    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1:
        st.markdown(
            f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:12px 14px; border-radius:8px;">
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Target Concept</div>
                <div style="font-size:15px; font-weight:700; color:#14213D; margin-top:4px;">{t_info['name']}</div>
                <div style="font-size:11.5px; color:#64748B;">Domain: <strong>{t_info['domain']}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_meta2:
        st.markdown(
            f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:12px 14px; border-radius:8px;">
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Difficulty & Current Status</div>
                <div style="font-size:15px; font-weight:700; color:#14213D; margin-top:4px;">{t_info['difficulty']}</div>
                <div style="font-size:11.5px; color:#64748B;">Mastery: <strong>{t_info['mastery_status']}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_meta3:
        st.markdown(
            f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:12px 14px; border-radius:8px;">
                <div style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">Curriculum Readiness Verdict</div>
                <div style="margin-top:6px;"><span class="{v_badge}">{verdict}</span></div>
                <div style="font-size:11.5px; color:#64748B; margin-top:4px;">DAG Traversal State</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    tab_prereq, tab_quiz = st.tabs(["Prerequisite Dependency & Gap Analysis", "Interactive Formative Quiz"])

    with tab_prereq:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="saas-card-header">
                <div>
                    <h3 class="saas-card-title">Prerequisite Learning Path (DAG Topological Order)</h3>
                    <div class="saas-card-subtitle">Foundational concepts must be consolidated before attempting downstream dependent topics.</div>
                </div>
                <span class="badge-info">Topological Sequence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prereqs = gap_analysis["prerequisite_learning_path"]
        if not prereqs:
            st.markdown(
                """
                <div class="info-box">
                    <strong>FOUNDATIONAL CONCEPT:</strong> This is a root foundational concept with no prior prerequisites in the curriculum DAG.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            table_rows = []
            for p in prereqs:
                score_str = f"{p['mastery_score']:.0%}" if p['mastery_score'] is not None else "Unassessed"
                table_rows.append({
                    "Concept ID": p["concept_id"],
                    "Concept Name": p["name"],
                    "Difficulty": p["difficulty"],
                    "Mastery Score": score_str,
                    "Mastery Status": p["mastery_status"],
                    "Direct Predecessors": ", ".join(p["direct_prerequisites"]) if p["direct_prerequisites"] else "Root Node",
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            focus = gap_analysis.get("recommended_focus_concept")
            if focus:
                st.markdown(
                    f"""
                    <div class="warning-box">
                        <strong>Recommended Foundational Focus:</strong> Prioritize review of 
                        <strong>{focus['concept_id']}: {focus['name']}</strong> (Status: <em>{focus['mastery_status']}</em>)
                        before attempting {t_info['name']}.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown('</div>', unsafe_allow_html=True)

    with tab_quiz:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="saas-card-header">
                <div>
                    <h3 class="saas-card-title">Diagnostic Quiz: {selected_cid}</h3>
                    <div class="saas-card-subtitle">{concept_map[selected_cid].split(':', 1)[1].strip()}</div>
                </div>
                <span class="badge-info">Formative Assessment</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        quiz_data = gap_detective.get_concept_questions(selected_cid, subject=subject, include_answers=False)
        questions = quiz_data.get("questions", [])

        if not questions:
            st.info("No diagnostic questions currently provisioned for this concept.")
        else:
            with st.form(key=f"quiz_form_{selected_cid}_{student_id}"):
                user_answers = {}
                for idx, q in enumerate(questions, start=1):
                    st.markdown(
                        f"""
                        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:14px 16px; border-radius:8px; margin-bottom:12px;">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                <span style="font-weight:700; font-size:13px; color:#14213D;">Question {idx}</span>
                                <span class="badge-info" style="font-size:10.5px;">Bloom's: {q['blooms_level']}</span>
                            </div>
                            <div style="font-size:14px; color:#1E293B; margin-bottom:10px;">{q['question_text']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    sel = st.radio(
                        label=f"Select answer for Question {idx}:",
                        options=["A", "B", "C", "D"],
                        format_func=lambda x: f"{x}) {q['options'][x]}",
                        key=f"q_{q['question_id']}",
                    )
                    user_answers[q["question_id"]] = sel
                    st.markdown("<div style='margin-bottom:14px;'></div>", unsafe_allow_html=True)

                submitted = st.form_submit_button("Submit Diagnostic Evaluation", type="primary")

            if submitted:
                eval_res = gap_detective.evaluate_quiz_submission(
                    student_id=student_id,
                    concept_id=selected_cid,
                    submitted_answers=user_answers,
                    quiz_type="BASELINE_DIAGNOSTIC",
                    subject=subject,
                )

                score_pct = eval_res["percentage"]
                status = eval_res["mastery_status"]
                stat_badge = "badge-safe" if status == "MASTERED" else ("badge-at-risk" if status == "CRITICAL_GAP" else "badge-info")

                st.markdown(
                    f"""
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:16px; border-radius:10px; margin:16px 0;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-weight:700; color:#14213D;">Diagnostic Outcome Result</span>
                            <span class="{stat_badge}">{status}</span>
                        </div>
                        <div style="font-size:24px; font-weight:800; color:#14213D; margin-top:6px;">{score_pct:.1f}% ({eval_res['correct_answers']} / {eval_res['total_questions']} correct)</div>
                        <div style="font-size:12px; color:#64748B; margin-top:4px;">Result logged in SQLite (date_provenance='ESTIMATED_DEMONSTRATION', is_uci_source=0).</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Question Breakdown & Feedback")
                for item in eval_res["question_results"]:
                    ic = item["is_correct"]
                    badge = "✅ CORRECT" if ic else "❌ INCORRECT"
                    color = "#059669" if ic else "#DC2626"
                    bg = "#ECFDF5" if ic else "#FEF2F2"
                    border_c = "#A7F3D0" if ic else "#FECACA"
                    st.markdown(
                        f"""
                        <div style="background-color:{bg}; border:1px solid {border_c}; padding:14px; border-radius:8px; margin-bottom:10px;">
                            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                                <strong style="color:#1E293B;">{item['question_text']}</strong>
                                <span style="font-weight:700; font-size:12px; color:{color};">{badge}</span>
                            </div>
                            <div style="font-size:12.5px; color:#334155; margin-bottom:4px;">
                                Your Answer: <strong>Option {item['student_answer']}</strong> | Correct Answer: <strong>Option {item['correct_answer']}</strong>
                            </div>
                            <div style="font-size:12px; color:#64748B; background:#FFFFFF; padding:8px 10px; border-radius:6px; border:1px solid {border_c};">
                                <strong>Pedagogical Explanation:</strong> {item['explanation']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.success("Concept mastery state updated successfully in SQLite.")

        st.markdown('</div>', unsafe_allow_html=True)
