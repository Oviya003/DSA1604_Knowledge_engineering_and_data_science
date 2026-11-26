"""
ExplainAI - Module 3: Explainable AI and Personalized Support
Provides SHAP explanations, prerequisite-aware knowledge gap detection (Signature Feature 2),
personalized learning recommendations, and closed-loop improvement tracking (Signature Feature 4).
"""

from src.module3_explainability_support.shap_explainer import AcademicShapExplainer
from src.module3_explainability_support.knowledge_gap_detective import KnowledgeGapDetective
from src.module3_explainability_support.recommender import PersonalizedLearningRecommender
from src.module3_explainability_support.improvement_tracker import ClosedLoopImprovementTracker

__all__ = [
    "AcademicShapExplainer",
    "KnowledgeGapDetective",
    "PersonalizedLearningRecommender",
    "ClosedLoopImprovementTracker",
]

