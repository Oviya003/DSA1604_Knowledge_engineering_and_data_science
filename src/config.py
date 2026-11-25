"""
ExplainAI - Configuration Module
Defines directory paths, feature sets, model hyperparameter baselines,
academic risk thresholds, and database connection settings.
"""

from pathlib import Path

# ==============================================================================
# Path Configurations
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

MATH_DATASET_PATH = RAW_DATA_DIR / "student-mat.csv"
POR_DATASET_PATH = RAW_DATA_DIR / "student-por.csv"
DATABASE_PATH = DATA_DIR / "explainai.db"

# Supported subjects
SUPPORTED_SUBJECTS = ["Mathematics", "Portuguese"]
CURRICULUM_SUPPORTED_SUBJECTS = ["Mathematics"]  # Only Mathematics has the demonstration concept graph

# ==============================================================================
# Machine Learning & Anti-Leakage Feature Definitions
# ==============================================================================
# The target G3 represents the final evaluation grade on a 0-20 scale.
# Generic threshold: G3 < 10 is at-risk (failure), G3 >= 10 is passing.
TARGET_COL = "G3"
BINARY_TARGET_COL = "is_at_risk"
PASSING_THRESHOLD = 10.0

# Base pre-course demographic and behavioral categorical features (17 features)
DEMOGRAPHIC_CATEGORICAL = [
    "school",
    "sex",
    "address",
    "famsize",
    "Pstatus",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
]

# Base pre-course demographic and behavioral numerical/ordinal features (12 features)
# NOTE: 'absences' has been REMOVED from the predictive feature matrix to prevent
# temporal leakage, because the UCI dataset records cumulative annual absences
# rather than timestamped intake or post-G1 absences.
DEMOGRAPHIC_NUMERICAL = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
]

# Descriptive-only attributes (retained for EDA, summary statistics, and student profiling)
# Strictly excluded from predictive feature matrices
DESCRIPTIVE_FEATURES = ["absences"]

# Explicitly excluded features to prevent temporal and target leakage
STRICTLY_EXCLUDED_FROM_FEATURES = [
    "absences",        # Temporal leakage: cumulative annual tally
    "G2",              # Temporal leakage: future midterm grade
    "G3",              # Target leakage: final grade outcome
    BINARY_TARGET_COL, # Target leakage: derived binary label
]

# Experiment A Feature Set: Pre-course features only (29 features)
# Zero assessment or cumulative leakage
FEATURES_MODEL_A = DEMOGRAPHIC_CATEGORICAL + DEMOGRAPHIC_NUMERICAL

# Experiment B Feature Set: Pre-course features + Early Assessment G1 (30 features)
# Evaluates early-term risk prediction once G1 is officially released
EARLY_ASSESSMENT_FEATURES = ["G1"]
FEATURES_MODEL_B = FEATURES_MODEL_A + EARLY_ASSESSMENT_FEATURES

# Actionable counterfactual features for Signature Feature 1 (What-If Simulator)
# Only features realistically adjustable by students or intervention plans
ACTIONABLE_FEATURES = {
    "studytime": {"min": 1, "max": 4, "step": 1, "label": "Weekly Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)"},
    "failures": {"min": 0, "max": 4, "step": 1, "label": "Number of Past Class Failures"},
    "freetime": {"min": 1, "max": 5, "step": 1, "label": "Free Time After School (1: Very Low to 5: Very High)"},
    "goout": {"min": 1, "max": 5, "step": 1, "label": "Going Out with Friends (1: Very Low to 5: Very High)"},
    "Dalc": {"min": 1, "max": 5, "step": 1, "label": "Workday Alcohol Consumption (1-5)"},
    "Walc": {"min": 1, "max": 5, "step": 1, "label": "Weekend Alcohol Consumption (1-5)"},
    "health": {"min": 1, "max": 5, "step": 1, "label": "Current Health Status (1-5)"},
}

# ==============================================================================
# Model Training & Validation Configuration
# ==============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS_CV = 5

# ==============================================================================
# Rule-Based Warning Thresholds
# ==============================================================================
WARNING_THRESHOLDS = {
    "excessive_absences": 10,
    "high_failures": 1,
    "low_study_time": 1,
    "high_alcohol": 4,
    "poor_health": 2,
    "drift_decline_points": 3.0,
}

# ==============================================================================
# UI & Display Themes
# ==============================================================================
APP_TITLE = "ExplainAI"
APP_SUBTITLE = "Explainable AI-Based Academic Intelligence and Personalized Learning Support System"
THEME_COLOR_PRIMARY = "#1E3A8A"   # Academic Navy Blue
THEME_COLOR_SECONDARY = "#0D9488" # Teal Accent
THEME_COLOR_RISK = "#DC2626"      # Coral Red for At-Risk
THEME_COLOR_SAFE = "#16A34A"      # Forest Green for Satisfactory
