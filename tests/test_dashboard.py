"""
ExplainAI - Dashboard Integration Test Suite
Verifies application imports, service caching, view routing,
and graceful handling of edge cases in the Streamlit application.
"""

import pytest
from src.database.db_manager import DatabaseManager
from src.ui.styles import CUSTOM_CSS
from app import get_services


def test_dashboard_styles_and_theming():
    """Verify custom CSS string contains critical theme tokens."""
    assert "#1E3A8A" in CUSTOM_CSS
    assert "metric-card" in CUSTOM_CSS
    assert "badge-at-risk" in CUSTOM_CSS
    assert "badge-safe" in CUSTOM_CSS
    assert "disclaimer-card" in CUSTOM_CSS


def test_dashboard_service_singletons():
    """Verify all backend singletons initialize cleanly via get_services()."""
    services = get_services()
    assert "db" in services
    assert "predictor" in services
    assert "rule_engine" in services
    assert "what_if" in services
    assert "drift" in services
    assert "shap" in services
    assert "gap_detective" in services
    assert "recommender" in services
    assert "tracker" in services

    # Verify database connection and student availability
    db = services["db"]
    students_mat = db.get_all_students("Mathematics")
    students_por = db.get_all_students("Portuguese")
    assert len(students_mat) == 395
    assert len(students_por) == 649


def test_dashboard_view_module_imports():
    """Verify all 8 view components are importable and have render functions."""
    from src.ui.views.overview import render_overview_view
    from src.ui.views.prediction import render_prediction_view
    from src.ui.views.explainability import render_explainability_view
    from src.ui.views.what_if import render_what_if_view
    from src.ui.views.drift import render_drift_view
    from src.ui.views.gap_detective import render_gap_detective_view
    from src.ui.views.recommendations import render_recommendations_view
    from src.ui.views.tracker import render_tracker_view

    assert callable(render_overview_view)
    assert callable(render_prediction_view)
    assert callable(render_explainability_view)
    assert callable(render_what_if_view)
    assert callable(render_drift_view)
    assert callable(render_gap_detective_view)
    assert callable(render_recommendations_view)
    assert callable(render_tracker_view)


def test_dashboard_graceful_missing_data(test_db=None):
    """Verify backend services handle missing or unknown student records gracefully."""
    db = DatabaseManager()
    unknown = db.get_student_by_id_and_subject("NON_EXISTENT", "Mathematics")
    assert unknown is None


def test_streamlit_apptest_navigation_all_views():
    """Execute Streamlit AppTest simulating navigation across all 8 dashboard views."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert len(at.exception) == 0, f"AppTest initial startup exception: {at.exception}"

    radio = at.sidebar.radio[0]
    assert len(radio.options) == 8

    # Cycle through all navigation views and ensure zero unhandled exceptions
    for view_name in radio.options:
        radio.set_value(view_name)
        at.run(timeout=30)
        assert len(at.exception) == 0, f"AppTest threw unhandled exception on view '{view_name}': {at.exception}"


