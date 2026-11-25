"""
ExplainAI - Module 2: Model Training and Evaluation Pipeline
Trains, cross-validates, benchmarks, and persists academic risk prediction models
(DummyClassifier, Logistic Regression, Random Forest, and XGBoost) for Mathematics
and Portuguese cohorts across Experiment A (Intake) and Experiment B (Post-G1).

STRICT TEST SECLUSION & MODEL SELECTION POLICY:
----------------------------------------------
1. Model selection is performed EXCLUSIVELY on training data using 5-Fold Stratified
   Cross-Validation F1-score (with CV balanced accuracy as tie-breaker).
2. The held-out test partition (20%) is NEVER used for model selection, hyperparameter
   tuning, or threshold selection.
3. Preprocessing ColumnTransformer is fitted strictly within training folds.
4. Evaluation metrics report raw predicted at-risk probabilities and Brier score;
   probabilities are NOT claimed to be calibrated.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS_CV,
    TARGET_COL,
    BINARY_TARGET_COL,
    PASSING_THRESHOLD,
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
)
from src.module1_data_knowledge.ingestion import DatasetIngestion
from src.module1_data_knowledge.preprocessing import DataPreprocessor

# Filter harmless solver warning from scipy in Python 3.13
warnings.filterwarnings("ignore", message="Unknown solver options: iprint")

MODEL_VERSION = "1.0.0"


def instantiate_candidate_models(scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """
    Instantiates candidate classifiers with strict reproducibility settings and
    training-derived class imbalance adaptations.
    """
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=0.5,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            min_samples_split=4,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """
    Calculates evaluation metrics treating at-risk (1) as the positive class.
    Includes Brier score to measure probability forecast error.
    """
    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        roc_auc = 0.5

    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except Exception:
        brier = 0.25

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def run_cross_validation(
    pipeline_factory,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = N_SPLITS_CV,
) -> Dict[str, Any]:
    """
    Executes Stratified K-Fold Cross-Validation strictly on the training partition.
    Fits the preprocessing ColumnTransformer inside each fold pipeline to prevent leakage.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_metrics = {
        "accuracy": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "brier_score": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline = pipeline_factory()
        pipeline.fit(X_tr, y_tr)

        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred.astype(float)

        m = compute_metrics(y_val.values, y_pred, y_prob)
        for k in cv_metrics:
            cv_metrics[k].append(m[k])

    summary = {}
    for k, v in cv_metrics.items():
        summary[f"{k}_mean"] = round(float(np.mean(v)), 4)
        summary[f"{k}_std"] = round(float(np.std(v)), 4)

    return summary


def train_single_experiment(
    df: pd.DataFrame,
    subject: str,
    experiment: str,
    output_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    """
    Executes training, cross-validation, model selection, and held-out testing for a specific
    subject and experiment.

    CRITICAL SELECTION RULE:
    -----------------------
    Best model is selected SOLELY using training-set 5-fold cross-validation results:
    score = (cv_f1_mean * 100) + cv_balanced_accuracy_mean.
    The held-out test partition is strictly evaluated post-selection for unbiased reporting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = DataPreprocessor()

    # 1. Feature-Target Extraction (strict anti-leakage)
    X, y, meta = preprocessor.extract_features_and_target(df, model_experiment=experiment)

    # 2. Stratified Train/Test Split (80% Train, 20% Held-Out Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Compute class weight ratio for XGBoost strictly from training partition (neg / pos)
    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    scale_pos_weight = (num_neg / num_pos) if num_pos > 0 else 1.0

    candidates = instantiate_candidate_models(scale_pos_weight=scale_pos_weight)
    experiment_results: Dict[str, Any] = {
        "subject": subject,
        "experiment": f"MODEL_{experiment.upper()}",
        "feature_count": X.shape[1],
        "feature_names": list(X.columns),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_class_distribution": {"passing": num_neg, "at_risk": num_pos},
        "test_class_distribution": {
            "passing": int((y_test == 0).sum()),
            "at_risk": int((y_test == 1).sum()),
        },
        "imbalance_handling": {
            "Dummy": "None (Majority Baseline)",
            "LogisticRegression": "class_weight='balanced' (inverse frequency)",
            "RandomForest": "class_weight='balanced' (inverse frequency)",
            "XGBoost": f"scale_pos_weight={scale_pos_weight:.4f} (training negative/positive ratio)",
        },
        "model_selection_policy": (
            "Training-Only 5-Fold Stratified CV: selected candidate with maximum "
            "(cv_f1_mean * 100 + cv_balanced_accuracy_mean). Zero test-set influence."
        ),
        "models": {},
    }

    best_model_name = None
    best_cv_score = -1.0
    fitted_pipelines = {}

    # Step A: Perform Cross-Validation on Training Data ONLY and identify Best Model
    for name, clf in candidates.items():
        def make_pipeline():
            transformer = preprocessor.build_column_transformer(model_experiment=experiment)
            return Pipeline(steps=[("preprocessor", transformer), ("classifier", clf)])

        # Run 5-fold CV strictly on X_train, y_train
        cv_summary = run_cross_validation(make_pipeline, X_train, y_train, n_splits=N_SPLITS_CV)

        # Fit final pipeline on complete training set for this candidate
        final_pipeline = make_pipeline()
        final_pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = final_pipeline

        # Save individual candidate artifact
        artifact_filename = f"{subject.lower()}_{experiment.lower()}_{name.lower()}.joblib"
        artifact_path = output_dir / artifact_filename
        joblib.dump(final_pipeline, artifact_path)

        # Calculate selection score from CV results ONLY
        cv_score = (cv_summary["f1_mean"] * 100.0) + cv_summary["balanced_accuracy_mean"]
        if cv_score > best_cv_score:
            best_cv_score = cv_score
            best_model_name = name

        experiment_results["models"][name] = {
            "cv_metrics": cv_summary,
            "cv_selection_score": round(cv_score, 4),
            "artifact_file": artifact_filename,
        }

    # Step B: Evaluate candidates on the untouched held-out test set
    for name, pipeline in fitted_pipelines.items():
        y_test_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_test_prob = y_test_pred.astype(float)

        test_metrics = compute_metrics(y_test.values, y_test_pred, y_test_prob)
        experiment_results["models"][name]["test_metrics"] = test_metrics

    # Save designated best model (chosen solely via CV) to standard best artifact path
    experiment_results["best_model"] = best_model_name
    experiment_results["best_model_cv_metrics"] = experiment_results["models"][best_model_name]["cv_metrics"]
    experiment_results["best_model_test_metrics"] = experiment_results["models"][best_model_name]["test_metrics"]
    experiment_results["best_model_metrics"] = experiment_results["models"][best_model_name]["test_metrics"]

    best_artifact_filename = f"{subject.lower()}_{experiment.lower()}_best.joblib"
    best_artifact_path = output_dir / best_artifact_filename
    joblib.dump(fitted_pipelines[best_model_name], best_artifact_path)
    experiment_results["best_artifact_file"] = best_artifact_filename

    return experiment_results


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """
    Coordinates training of all four separate experiments:
    1. Mathematics Model A (Intake)
    2. Mathematics Model B (Post-G1)
    3. Portuguese Model A (Intake)
    4. Portuguese Model B (Post-G1)
    Persists evaluation metrics JSON to models/model_evaluation_metrics.json.
    """
    ingestion = DatasetIngestion()
    all_metrics: Dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_splits_cv": N_SPLITS_CV,
        "experiments": {},
    }

    # Load authentic raw datasets
    df_mat = ingestion.load_raw_dataset("Mathematics")
    df_por = ingestion.load_raw_dataset("Portuguese")

    # Run 4 isolated experiments
    print("Training Mathematics Model A (Intake Baseline, 29 features)...")
    all_metrics["experiments"]["mathematics_model_a"] = train_single_experiment(df_mat, "Mathematics", "A")

    print("Training Mathematics Model B (Post-G1 Interim, 30 features)...")
    all_metrics["experiments"]["mathematics_model_b"] = train_single_experiment(df_mat, "Mathematics", "B")

    print("Training Portuguese Model A (Intake Baseline, 29 features)...")
    all_metrics["experiments"]["portuguese_model_a"] = train_single_experiment(df_por, "Portuguese", "A")

    print("Training Portuguese Model B (Post-G1 Interim, 30 features)...")
    all_metrics["experiments"]["portuguese_model_b"] = train_single_experiment(df_por, "Portuguese", "B")

    # Persist benchmark report
    metrics_path = MODELS_DIR / "model_evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Model training and benchmarking complete. Metrics saved to {metrics_path}")
    return all_metrics


if __name__ == "__main__":
    train_and_evaluate_all_models()
