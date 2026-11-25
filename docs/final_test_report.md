# ExplainAI: An Explainable AI-Based Student Performance Prediction and Personalized Learning Support System for Inclusive and Quality Education

This document records the comprehensive verification, regression testing, and quality assurance results for **ExplainAI: An Explainable AI-Based Student Performance Prediction and Personalized Learning Support System for Inclusive and Quality Education**.

---

## 1. Executive Test Summary

- **Execution Date**: 2026-11-03
- **Test Command**: `python -m pytest tests/ -v`
- **Platform**: Windows 11 / Windows Server x64, Python 3.13.12, pytest 9.1.1, pluggy 1.6.0
- **Total Test Cases**: **45 Collected, 45 Passed, 0 Failed, 0 Skipped** (100% Pass Rate)
- **Deprecation Warnings**: 14 (all benign `PyparsingDeprecationWarning` instances originating from Matplotlib font parsing on Python 3.13)
- **Execution Time**: 23.23 seconds

---

## 2. Test Suite Breakdown by Module

### 2.1 Project Scaffolding & Foundational Integrity (`tests/test_step2_scaffolding.py`)
| Test Case Name | Result | Scope / Verification Objective |
|---|---|---|
| `test_dependencies_imported` | **PASSED** | Verifies all required dependencies (`streamlit`, `pandas`, `numpy`, `sklearn`, `shap`, `networkx`, `plotly`, `xgboost`, `joblib`) import cleanly without version conflicts. |
| `test_uci_datasets_exist_and_valid` | **PASSED** | Validates presence, column structure, and integrity of `student-mat.csv` (395 rows) and `student-por.csv` (649 rows). |
| `test_curriculum_provenance_dag_and_subject_restriction` | **PASSED** | Verifies acyclicity of curricular DAGs, node properties, and ensures Portuguese subject restriction in KG detective is respected. |
| `test_feature_counts_and_leakage_prevention` | **PASSED** | Verifies exact approved feature sets (29 for Model A, 30 for Model B), ensuring strict exclusion of target variable, future grades ($G2, G3$), and post-hoc absences. |
| `test_cross_subject_student_matching_analysis` | **PASSED** | Verifies matching analysis identifying 358 unambiguous 1-to-1 matches and 8 ambiguous keys producing 24 Cartesian product rows (366 distinct matching-key combinations, explicitly confirming these are distinct matching-key combinations, not verified unique individuals, and proving merged rows do not establish 382 unique people) and isolates cohorts. |
| `test_database_multi_subject_isolation_and_foreign_keys` | **PASSED** | Confirms SQLite composite keys `(student_id, subject)` prevent cross-cohort collision and enforce foreign key cascades. |

### 2.2 Module 1: Data Processing & Knowledge Graph (`tests/test_module1_data_kg.py`)
| Test Case Name | Result | Scope / Verification Objective |
|---|---|---|
| `test_dataset_loading_row_and_column_counts` | **PASSED** | Asserts raw and cleaned dataset dimensions match UCI specifications exactly. |
| `test_dataset_columns_and_data_types` | **PASSED** | Confirms numeric and categorical type assignments across all 33 raw attributes. |
| `test_missing_values_and_duplicate_detection` | **PASSED** | Asserts 0 missing values in raw datasets and verifies duplicate detection mechanisms. |
| `test_model_a_feature_count_and_leakage_exclusion` | **PASSED** | Confirms Model A uses exactly 29 features and excludes $G1, G2, G3$, `is_at_risk`, and `absences`. |
| `test_model_b_feature_count_and_early_assessment` | **PASSED** | Confirms Model B uses exactly 30 features (Model A + $G1$) and strictly excludes $G2, G3$. |
| `test_unfitted_column_transformer` | **PASSED** | Ensures scikit-learn transformers are unfitted before cross-validation folds to eliminate pipeline data leakage. |
| `test_knowledge_graph_structure` | **PASSED** | Asserts 10 concepts, 11 prerequisite dependency edges, and valid DAG structure. |
| `test_prerequisite_retrieval_and_transitive_closure` | **PASSED** | Verifies recursive prerequisite retrieval across multiple hierarchical tiers. |
| `test_topological_order_and_paths` | **PASSED** | Verifies topological ordering for curriculum progression without cycles. |
| `test_unsupported_subject_knowledge_graph` | **PASSED** | Verifies graceful error handling for subjects without curriculum definitions. |
| `test_diagnostic_questions_integrity` | **PASSED** | Asserts all 20 diagnostic questions map validly to concepts with verified correct answer keys and Bloom's taxonomy tags. |
| `test_sqlite_student_and_assessment_sync` | **PASSED** | Verifies SQLite synchronization of 1,044 students and 3,132 assessment records ($G1, G2, G3$). |
| `test_post_g1_prediction_cannot_access_future_assessments` | **PASSED** | Rigorously proves that Post-G1 prediction queries cannot read assessment records where `term_period > 1`. |
| `test_sqlite_foreign_key_and_rollback` | **PASSED** | Proves foreign key enforcement and transactional rollback on simulated constraint violation. |

### 2.3 Module 2: Prediction and Rule-Based Reasoning (`tests/test_module2_prediction.py`)
| Test Case Name | Result | Scope / Verification Objective |
|---|---|---|
| `test_feature_counts_and_anti_leakage_in_models` | **PASSED** | Validates feature counts across serialized model artifacts (`models/*.joblib`, including `mathematics_a_best.joblib`, `mathematics_b_best.joblib`, `portuguese_a_best.joblib`, `portuguese_b_best.joblib`). |
| `test_model_artifacts_saved_and_deserializable` | **PASSED** | Verifies all 16 candidate model artifacts and 4 best-model artifacts are deserializable. |
| `test_evaluation_metrics_json_integrity` | **PASSED** | Validates structure, cross-validation metrics, and test evaluations in `models/model_evaluation_metrics.json`. |
| `test_risk_predictor_inference_format` | **PASSED** | Asserts probability output format, risk category assignment, and threshold calibration. |
| `test_model_b_superiority_over_model_a` | **PASSED** | Statistically proves Model B ($G1$ included) achieves superior balanced accuracy and ROC-AUC over Model A across both cohorts. |
| `test_rule_engine_warning_triggers` | **PASSED** | Asserts rule-based heuristics trigger on known risk profiles (e.g., failure history, low study time). |
| `test_rule_engine_configurable_thresholds` | **PASSED** | Verifies warning engine thresholds can be adjusted dynamically. |
| `test_rule_engine_model_a_temporal_suppression` | **PASSED** | Proves assessment-dependent rules are suppressed during intake-only evaluation. |
| `test_what_if_simulator_valid_scenario` | **PASSED** | Simulates valid actionable feature adjustments (`studytime`, `schoolsup`) and verifies probability delta. |
| `test_what_if_simulator_rejection_of_demographics_and_grades` | **PASSED** | Proves simulator rejects immutable demographics (`sex`, `age`), health/substances (`health`, `Dalc`, `Walc`), and historical grades/failures (`failures`, $G1$). |
| `test_academic_drift_detection_trajectory` | **PASSED** | Verifies classification of academic trajectories into Severe Drift, Moderate Drift, Stable, and Improving. |

### 2.4 Module 3: Explainable AI and Personalized Support (`tests/test_module3_explainable_support.py`)
| Test Case Name | Result | Scope / Verification Objective |
|---|---|---|
| `test_shap_explainer_loading_and_instance_explanation` | **PASSED** | Verifies SHAP tree and linear explainers produce additive feature contributions on the log-odds margin scale. |
| `test_shap_feature_mapping_and_anti_leakage` | **PASSED** | Confirms SHAP feature attributions correspond strictly to approved model inputs. |
| `test_shap_global_feature_importance` | **PASSED** | Validates cohort-level mean absolute SHAP value calculation. |
| `test_diagnostic_question_retrieval` | **PASSED** | Confirms diagnostic question retrieval by concept ID from SQLite. |
| `test_diagnostic_quiz_evaluation_and_mastery_classification` | **PASSED** | Validates scoring logic and mastery thresholds (Mastered $\ge 70\%$, Developing $50-69\%$, Needs Remediation $< 50\%$). |
| `test_prerequisite_gap_tracing_and_unassessed_distinction` | **PASSED** | Verifies gap detective distinguishes between confirmed failed concepts and unassessed concepts. |
| `test_unsupported_portuguese_subject_in_gap_detective` | **PASSED** | Verifies informative fallback when querying subjects without diagnostic mastery data. |
| `test_recommender_generation_and_evidence` | **PASSED** | Validates recommendation generation, priority assignment, and empirical evidence citation. |
| `test_closed_loop_intervention_lifecycle` | **PASSED** | Tests complete lifecycle: Prescribing intervention $\rightarrow$ Taking diagnostic quiz $\rightarrow$ Status update to Completed with mastery score. |

### 2.5 Dashboard Integration & Web Interface (`tests/test_dashboard.py`)
| Test Case Name | Result | Scope / Verification Objective |
|---|---|---|
| `test_dashboard_styles_and_theming` | **PASSED** | Confirms presence of key custom CSS tokens, card classes, and badge styling. |
| `test_dashboard_service_singletons` | **PASSED** | Verifies `get_services()` initializes all 9 backend singletons cleanly. |
| `test_dashboard_view_module_imports` | **PASSED** | Confirms all 8 navigation view render functions are importable and callable. |
| `test_dashboard_graceful_missing_data` | **PASSED** | Verifies graceful error handling when querying non-existent student records. |
| `test_streamlit_apptest_navigation_all_views` | **PASSED** | Executes headless Streamlit `AppTest` traversing across all 8 navigation views with zero unhandled exceptions. |

---

## 3. Deprecation Warnings Audit

All 14 warnings emitted during test execution were inspected and confirmed to be third-party deprecation notices:
- **Origin**: `matplotlib._fontconfig_pattern` and `matplotlib._mathtext`.
- **Warning Class**: `PyparsingDeprecationWarning`.
- **Cause**: Matplotlib 3.9.2 invoking deprecated PyParsing functions (`resetCache`, `parseString`, `oneOf`) on Python 3.13.
- **Impact**: Zero impact on core system logic, prediction pipelines, or application stability.

---

## 4. Live Server Health & Localhost Security Verification

1. **Localhost Binding Verification**:
   - Configuration in `.streamlit/config.toml`:
     ```toml
     [server]
     address = "127.0.0.1"
     headless = true
     port = 8501
     enableCORS = true
     enableXsrfProtection = true
     ```
   - Server startup output: Strictly restricted to `URL: http://127.0.0.1:8501`. Network and external URLs suppressed.
2. **HTTP Endpoint Probes**:
   - `GET http://127.0.0.1:8501/_stcore/health` $\rightarrow$ Returned HTTP 200 `ok`.
   - `GET http://127.0.0.1:8501/` $\rightarrow$ Returned HTTP 200.
3. **Browser Rendering & Full-Fidelity Interface Capture**:
   - Playwright headless Chromium successfully navigated across all 8 views, capturing authentic PNG screenshots saved in `docs/screenshots/`.

