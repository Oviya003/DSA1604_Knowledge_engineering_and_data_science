# ExplainAI: An Explainable AI-Based Student Performance Prediction and Personalized Learning Support System for Inclusive and Quality Education

This document provides an end-to-end evaluation script designed for academic examiners, project reviewers, and educators to demonstrate and verify all features of **ExplainAI: An Explainable AI-Based Student Performance Prediction and Personalized Learning Support System for Inclusive and Quality Education**.

---

## 1. System Requirements & Quick Start

### 1.1 Prerequisites
- **Operating System**: Windows 10/11, Linux, or macOS.
- **Python Version**: Python 3.10+ (Developed and verified on Python 3.13.12).
- **Core Dependencies**: Streamlit 1.40.0, Scikit-Learn 1.5.2, SHAP 0.54.1, XGBoost 3.3.0, NetworkX 3.6.1, Plotly 6.9.0.

### 1.2 Verification Commands
Run the automated test suite to verify all 45 unit and integration tests:
```bash
python -m pytest tests/ -v
```
*Expected Output*: `45 passed, 14 warnings in ~15-25s` (all warnings are benign `PyparsingDeprecationWarning` deprecations from Matplotlib).

### 1.3 Launching the Application
Start the secure local Streamlit dashboard:
```bash
python -m streamlit run app.py
```
Open a modern web browser and navigate to:
```
http://127.0.0.1:8501
```
*(The server is configured to bind strictly to localhost `127.0.0.1` for local review security).*

---

## 2. Demonstration Walkthrough: The 8 Navigation Views

ExplainAI includes **three core modules** and **four signature features**, accessible via the left sidebar navigation menu:

```
[Sidebar Navigation Menu]
  ├── 📊 Dashboard Overview
  ├── 🎯 Academic Prediction
  ├── 🔍 SHAP Explainability
  ├── 🔮 What-If Simulator (Signature Feature 1)
  ├── 📉 Academic Drift Monitoring (Signature Feature 3)
  ├── 🧩 Knowledge Gap Detective (Signature Feature 2)
  ├── 💡 Personalized Recommendations
  └── 🔄 Improvement Tracker (Signature Feature 4)
```

---

### Step 1: Sidebar Setup & Cohort Selection
1. Look at the left sidebar:
   - **Academic Subject Cohort**: Choose between `Mathematics` (395 students) and `Portuguese` (649 students).
   - **Student Profile**: Notice that student identifiers are prefixed by subject (e.g., `MAT_001`, `MAT_010`, `POR_005`).
2. Select:
   - Subject: `Mathematics`
   - Student: `MAT_010` (or `MAT_001` for a standard passing profile).

---

### Step 2: 📊 Dashboard Overview
- **Action**: Select `📊 Dashboard Overview` in the sidebar.
- **What to Observe**:
  1. **Executive KPI Cards**: Displays Predicted Risk Tier, At-Risk Probability, Intake Status (`Historical At-Risk` or `Standard`), and Number of Active Interventions.
  2. **Academic Performance Trajectory**: An interactive Plotly chart depicting student term grades across Term 1 ($G1$), Term 2 ($G2$), and Final Exam ($G3$).
  3. **Demographic & Contextual Matrix**: Full summary of the student's study habits, family background, school support, and absences.
  4. **Active Heuristic Alerts**: Rule-based warnings highlighting elevated risk indicators (e.g., historical failures, low study time).

---

### Step 3: 🎯 Academic Prediction (Module 2)
- **Action**: Select `🎯 Academic Prediction` in the sidebar.
- **What to Observe**:
  1. **Dual Model Comparison**:
     - **Model A (Intake Baseline - 29 Features)**: Evaluates risk prior to any term assessments, relying strictly on background, family, and demographic features.
     - **Model B (Post-G1 Early Warning - 30 Features)**: Incorporates the first period grade ($G1$), yielding significantly higher discriminative power (ROC-AUC $\ge 0.90$).
  2. **Model Evaluation Metrics Table**: Live display of test set metrics (Balanced Accuracy, Precision, Recall, $F_1$, ROC-AUC, and Brier Score).
  3. **Anti-Leakage Verification**: Note that $G2$ and $G3$ are strictly omitted from feature inputs, preserving realistic post-G1 early-warning validity.

---

### Step 4: 🔍 SHAP Explainability (Module 3)
- **Action**: Select `🔍 SHAP Explainability` in the sidebar.
- **What to Observe**:
  1. **Local Explanation Waterfall Chart**:
     - Shows the exact additive contributions of individual features pushing the model toward or away from the risk threshold.
     - **Log-Odds Scale Callout**: The visual banner clearly explains that SHAP values are presented on the log-odds (margin) scale before sigmoid conversion.
  2. **Top Risk-Increasing Drivers**: Factors such as low $G1$ score, prior failures, or minimal study time shown in red bars.
  3. **Top Protective Factors**: Factors such as strong parental education, higher education aspirations, or educational support shown in blue bars.
  4. **Global Feature Importance**: Expandable section displaying the overall cohort-level importance of features.

---

### Step 5: 🔮 What-If Simulator (Signature Feature 1)
- **Action**: Select `🔮 What-If Simulator (Feature 1)` in the sidebar.
- **What to Observe**:
  1. **Actionable Scenario Levers**:
     - Sliders for `Weekly Study Time` ($1-4$), `Free Time After School` ($1-5$), and `Going Out with Friends` ($1-5$).
     - Toggles for `Extra School Support` (`schoolsup`), `Family Educational Support` (`famsup`), and `Extra Paid Classes` (`paid`).
  2. **Strict Actionability Constraints**:
     - Notice that immutable demographics (`sex`, `age`, `address`, `guardian`), health/alcohol consumption (`health`, `Dalc`, `Walc`), and historical achievements (`failures`, $G1$) are **not** adjustable.
  3. **Run Simulation**:
     - Increase `studytime` from current level to `4: > 10 hours`.
     - Toggle `schoolsup` to `yes`.
     - Click **"Run Counterfactual Simulation"**.
  4. **Delta Display**:
     - The interface reports the Baseline Probability vs Simulated Probability and highlights the net risk reduction (e.g., $\Delta -14.2\%$).
  5. **Causal Disclaimer Card**:
     - Prominent warning stating: *"What-If simulations illustrate associative model sensitivity under hypothetical feature modifications. They do not constitute guaranteed causal outcomes."*

---

### Step 6: 📉 Academic Drift Monitoring (Signature Feature 3)
- **Action**: Select `📉 Academic Drift Monitoring (Feature 3)` in the sidebar.
- **What to Observe**:
  1. **Longitudinal Trajectory Comparison**:
     - Compares Term 1 ($G1$) score directly against Term 2 ($G2$) score.
  2. **Trajectory Classification**:
     - Demonstrates the four clinical drift categories:
       - **Severe Drift**: Score dropped by $\ge 3.0$ points.
       - **Moderate Drift**: Score dropped by $1.5$ to $2.9$ points.
       - **Stable**: Score changed by $< 1.5$ points.
       - **Improving**: Score increased by $\ge 1.5$ points.
  3. **Drift Warning Banner**:
     - For students exhibiting negative drift, prompts immediate pedagogical intervention before the final examination ($G3$).

---

### Step 7: 🧩 Knowledge Gap Detective (Signature Feature 2)
- **Action**: Select `🧩 Knowledge Gap Detective (Feature 2)` in the sidebar.
- **What to Observe**:
  1. **Curriculum Knowledge Graph DAG**:
     - An interactive NetworkX visualization of curricular concepts (e.g., Linear Equations $\rightarrow$ Quadratic Equations $\rightarrow$ Differential Calculus).
  2. **Curriculum Provenance Notice**:
     - Clear banner identifying the graph as an expert-crafted *demonstration curriculum* mapped to Portuguese secondary curricula.
  3. **Prerequisite Gap Tracing**:
     - Select a target concept requiring remediation (e.g., `MATH_CALC_01 Differential Calculus`).
     - The Gap Detective executes reverse topological traversal across the DAG.
     - Accurately differentiates between:
       - **Verified Gaps**: Pre-requisite concepts where diagnostic score $< 0.70$ (`Needs Remediation`).
       - **Unassessed Prerequisite Nodes**: Concepts not yet formally evaluated, flagged for diagnostic testing rather than assumed failure.

---

### Step 8: 💡 Personalized Recommendations (Module 3)
- **Action**: Select `💡 Personalized Recommendations` in the sidebar.
- **What to Observe**:
  1. **Triaged Recommendation Feed**:
     - Categorized by priority: `🚨 High Priority`, `⚠️ Medium Priority`, and `ℹ️ Low Priority`.
  2. **Evidence-Based Rationale**:
     - Every recommendation cites its empirical justification:
       - *"Justified by SHAP risk driver: low studytime (SHAP +0.48)"*
       - *"Justified by Knowledge Gap Detective: root prerequisite MATH_ALG_01 unmastered"*
       - *"Justified by Longitudinal Drift: student declined by 3.5 points between G1 and G2"*

---

### Step 9: 🔄 Improvement Tracker (Signature Feature 4)
- **Action**: Select `🔄 Improvement Tracker (Feature 4)` in the sidebar.
- **What to Observe**:
  1. **Prescribed Interventions Table**:
     - Displays targeted remediation activities assigned to the student (e.g., *Peer Tutoring for Linear Equations*).
  2. **Interactive Diagnostic Post-Quiz**:
     - Select an intervention in `Prescribed` or `In Progress` status.
     - Answer the diagnostic multiple-choice questions displayed on screen.
     - Submit the quiz.
  3. **Closed-Loop Mastery Verification**:
     - The system scores the submission immediately.
     - If the student achieves mastery ($\ge 70\%$), the intervention status updates to `Completed`, the post-quiz verification score is recorded, and the student's mastery profile in `concept_mastery` is updated.

---

## 3. Scope & Limitations for Evaluators

1. **Dataset Nature**:
   The underlying dataset originates from Cortez & Silva (2008), containing 1,044 secondary course enrollments across two Portuguese schools. Student IDs represent course enrollment records. Cross-course matching across the 13 demographic keys yields 382 merged rows, composed of 358 unambiguous 1-to-1 matches and 8 ambiguous keys producing 24 Cartesian product rows, representing 366 distinct matching-key combinations (clarifying that these are distinct matching-key combinations, not verified unique individuals); cohorts are preserved separately in the database to prevent duplicate identity conflation.
2. **Absences Exclusion**:
   Absences are recorded post-hoc as full-year totals in the original UCI dataset. In this system, `absences` is strictly excluded from predictive model inputs (Model A and Model B) to prevent retrospective target leakage.
3. **Associative vs Causal Nature**:
   All SHAP values and What-If simulations represent associative patterns captured by trained machine learning models, not experimental causal interventions.
4. **Demonstration Curriculum**:
   The 10 curricular concepts and 20 diagnostic questions are expert-crafted demonstration constructs designed to showcase knowledge graph traversal and closed-loop mastery verification.
5. **Statistical Uncertainty & Probability Calibration**:
   Model predictions provide probability estimates from the fitted classifiers. These outputs have not been independently calibrated, and no confidence intervals are provided.

