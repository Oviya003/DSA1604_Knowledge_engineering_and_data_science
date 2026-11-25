"""
ExplainAI - Main Streamlit Dashboard Application
Explainable AI-Based Student Performance Prediction and Personalized Learning Support System
"""

import streamlit as st
from typing import Dict, Any

from src.config import (
    APP_TITLE,
    APP_SUBTITLE,
    SUPPORTED_SUBJECTS,
)
from src.ui.styles import CUSTOM_CSS
from src.database.db_manager import DatabaseManager
from src.module2_prediction_reasoning.predictor import AcademicRiskPredictor
from src.module2_prediction_reasoning.rule_engine import RuleBasedWarningEngine
from src.module2_prediction_reasoning.what_if_simulator import AcademicWhatIfSimulator
from src.module2_prediction_reasoning.academic_drift import AcademicDriftDetector
from src.module3_explainability_support.shap_explainer import AcademicShapExplainer
from src.module3_explainability_support.knowledge_gap_detective import KnowledgeGapDetective
from src.module3_explainability_support.recommender import PersonalizedLearningRecommender
from src.module3_explainability_support.improvement_tracker import ClosedLoopImprovementTracker

# Import View Components
from src.ui.views.overview import render_overview_view
from src.ui.views.prediction import render_prediction_view
from src.ui.views.explainability import render_explainability_view
from src.ui.views.what_if import render_what_if_view
from src.ui.views.drift import render_drift_view
from src.ui.views.gap_detective import render_gap_detective_view
from src.ui.views.recommendations import render_recommendations_view
from src.ui.views.tracker import render_tracker_view


# ------------------------------------------------------------------------------
# 1. Page Configuration & Service Caching
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_TITLE} - Student Performance Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Initializes and caches core backend service singletons."""
    db = DatabaseManager()
    db.init_database()
    predictor = AcademicRiskPredictor(db_manager=db)
    rule_engine = RuleBasedWarningEngine()
    what_if = AcademicWhatIfSimulator(predictor=predictor, db_manager=db)
    drift = AcademicDriftDetector(db_manager=db)
    shap_service = AcademicShapExplainer()
    gap_detective = KnowledgeGapDetective(db_manager=db)
    recommender = PersonalizedLearningRecommender(db_manager=db)
    tracker = ClosedLoopImprovementTracker(db_manager=db, gap_detective=gap_detective)
    return {
        "db": db,
        "predictor": predictor,
        "rule_engine": rule_engine,
        "what_if": what_if,
        "drift": drift,
        "shap": shap_service,
        "gap_detective": gap_detective,
        "recommender": recommender,
        "tracker": tracker,
    }


def main():
    services = get_services()
    db = services["db"]

    # --------------------------------------------------------------------------
    # 2. Sidebar Controls & Student Selection
    # --------------------------------------------------------------------------
    with st.sidebar:
        # SaaS Brand Box
        st.markdown(
            """
            <div class="brand-box">
                <div class="brand-logo">🎓</div>
                <div class="brand-title-wrap">
                    <span class="brand-title">ExplainAI</span>
                    <span class="brand-subtitle">Student Performance Intelligence</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nav-section-title">Cohort & Student Selection</div>', unsafe_allow_html=True)

        # Subject Selection
        selected_subject = st.selectbox(
            "Academic Subject Cohort:",
            options=SUPPORTED_SUBJECTS,
            index=0,
            help="Mathematics and Portuguese cohorts are analyzed separately under strict multi-subject isolation.",
        )

        # Student Selection
        students = db.get_all_students(subject=selected_subject)
        if not students:
            st.error(f"No student records found in database for {selected_subject}. Please run dataset sync.")
            st.stop()

        student_ids = [s["student_id"] for s in students]
        selected_student_id = st.selectbox(
            f"Select Student ({len(students)} enrolled):",
            options=student_ids,
            index=0,
            help="Choose a student profile to inspect predictions, SHAP values, and diagnostic history.",
        )

        st.markdown('<div class="nav-section-title">Analytics Modules</div>', unsafe_allow_html=True)
        view_choice = st.radio(
            "Select View:",
            options=[
                "📊 Dashboard Overview",
                "🎯 Academic Prediction",
                "🔍 SHAP Explainability",
                "🔮 What-If Simulator (Feature 1)",
                "📉 Academic Drift Monitoring (Feature 3)",
                "🧩 Knowledge Gap Detective (Feature 2)",
                "💡 Personalized Recommendations",
                "🔄 Improvement Tracker (Feature 4)",
            ],
            index=0,
        )

        # Minimalist SaaS Footer
        st.markdown(
            """
            <div class="sidebar-footer">
                <div style="font-weight: 700; color: #14213D; font-size: 11.5px;">ExplainAI Platform</div>
                <div style="color: #64748B; margin-top: 2px;">Multi-Subject ML Risk & XAI Engine</div>
                <div style="margin-top: 8px; display: flex; align-items: center; gap: 6px; color: #059669; font-weight: 600; font-size: 11px;">
                    <span style="font-size: 8px;">●</span> System Online (Localhost)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # 3. Fetch Selected Student Record
    # --------------------------------------------------------------------------
    student = db.get_student_by_id_and_subject(selected_student_id, selected_subject)
    if not student:
        st.error(f"Student '{selected_student_id}' not found for subject '{selected_subject}'.")
        st.stop()

    if "G1" not in student and "G1_raw" in student:
        student["G1"] = student["G1_raw"]

    # --------------------------------------------------------------------------
    # 4. View Header Mapping
    # --------------------------------------------------------------------------
    view_subtitles = {
        "Dashboard Overview": "Comprehensive academic profile, longitudinal trajectory, and active early-warning signals.",
        "Academic Prediction": "Dual-timeline risk evaluation: Intake baseline (Model A) and Post-G1 interim forecast (Model B).",
        "SHAP Explainability": "Game-theoretic feature attributions quantifying factors driving risk predictions.",
        "What-If Simulator (Feature 1)": "Actionable counterfactual scenario simulation across validated academic variables.",
        "Academic Drift Monitoring (Feature 3)": "Chronological performance trajectory monitoring to detect term-over-term score deterioration.",
        "Knowledge Gap Detective (Feature 2)": "Curricular DAG prerequisite tracing to isolate foundational learning deficiencies.",
        "Personalized Recommendations": "Evidence-backed pedagogical interventions synthesized from multi-modal diagnostic signals.",
        "Improvement Tracker (Feature 4)": "Closed-loop intervention tracking and diagnostic reassessment lifecycle.",
    }

    raw_view_name = view_choice.split(" ", 1)[1] if " " in view_choice else view_choice
    view_desc = view_subtitles.get(raw_view_name, APP_SUBTITLE)
    school_name = "Gabriel Pereira (GP)" if student['school'] == 'GP' else "Mousinho da Silveira (MS)"
    intake_status_label = "Historical At-Risk" if student['failures'] > 0 else "Standard Entry"

    st.markdown(
        f"""
        <div class="header-banner">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
                <div>
                    <h1 class="header-title">{raw_view_name}</h1>
                    <div class="header-subtitle">{view_desc}</div>
                </div>
                <div style="text-align:right;">
                    <span class="badge-info" style="font-size:12px;">{selected_subject} Cohort</span>
                </div>
            </div>
            <div class="header-pills">
                <div class="header-pill">👤 Student: <strong>{selected_student_id}</strong></div>
                <div class="header-pill">🏫 School: <strong>{school_name}</strong></div>
                <div class="header-pill">📚 Subject: <strong>{selected_subject}</strong></div>
                <div class="header-pill">📋 Intake Status: <strong>{intake_status_label}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 5. Route to Selected View
    # --------------------------------------------------------------------------
    if "Overview" in view_choice:
        render_overview_view(
            db=db,
            predictor=services["predictor"],
            rule_engine=services["rule_engine"],
            student=student,
            subject=selected_subject,
        )

    elif "Academic Prediction" in view_choice:
        render_prediction_view(
            predictor=services["predictor"],
            rule_engine=services["rule_engine"],
            student=student,
            subject=selected_subject,
        )

    elif "SHAP Explainability" in view_choice:
        render_explainability_view(
            shap_service=services["shap"],
            student=student,
            subject=selected_subject,
        )

    elif "What-If Simulator" in view_choice:
        render_what_if_view(
            simulator=services["what_if"],
            student=student,
            subject=selected_subject,
        )

    elif "Academic Drift" in view_choice:
        render_drift_view(
            drift_detector=services["drift"],
            student=student,
            subject=selected_subject,
        )

    elif "Knowledge Gap" in view_choice:
        render_gap_detective_view(
            gap_detective=services["gap_detective"],
            student=student,
            subject=selected_subject,
        )

    elif "Personalized Recommendations" in view_choice:
        render_recommendations_view(
            db=db,
            recommender=services["recommender"],
            predictor=services["predictor"],
            rule_engine=services["rule_engine"],
            shap_service=services["shap"],
            gap_detective=services["gap_detective"],
            student=student,
            subject=selected_subject,
        )

    elif "Improvement Tracker" in view_choice:
        render_tracker_view(
            db=db,
            tracker=services["tracker"],
            gap_detective=services["gap_detective"],
            student=student,
            subject=selected_subject,
        )


if __name__ == "__main__":
    main()
