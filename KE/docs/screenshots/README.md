# ExplainAI: An Explainable AI-Based Student Performance Prediction and Personalized Learning Support System for Inclusive and Quality Education
## Dashboard Interface Screenshots (Modern SaaS Light Theme)

This directory contains authentic, full-resolution interface captures of all eight functional views in the redesigned ExplainAI Streamlit web application. Captured directly from the local running server (`http://127.0.0.1:8501`) using Playwright with Chromium (viewport: 1440 × 950 px).

---

## Screenshot Inventory

| # | Filename | Navigation View | Key Rendered Components | File Size |
|---|---|---|---|---|
| 01 | `01_dashboard_overview.png` | **📊 Dashboard Overview** | Executive KPI metric cards, G1/G2/G3 longitudinal trajectory chart, demographic & contextual profile, rule-based alerts, uncalibrated probability notice. | 148,227 bytes |
| 02 | `02_academic_prediction.png` | **🎯 Academic Prediction** | Model A (Intake) vs Model B (Post-G1) risk category badges, predicted at-risk probabilities, uncalibrated probability notices, rule-based diagnostic warning feed. | 183,027 bytes |
| 03 | `03_shap_explainability.png` | **🔍 SHAP Explainability** | Local SHAP waterfall explanation (log-odds scale), top risk-increasing and protective factors, global feature importance summary bar chart. | 171,247 bytes |
| 04 | `04_what_if_simulator.png` | **🔮 What-If Simulator (Feature 1)** | Actionable scenario sliders (`studytime`, `freetime`, `goout`), intervention support toggles (`schoolsup`, `famsup`), counterfactual risk delta comparison, causal disclaimer, uncalibrated probability notice. | 158,082 bytes |
| 05 | `05_drift_monitoring.png` | **📉 Academic Drift Monitoring (Feature 3)** | Post-G1 to G2 trajectory delta (points and percentage), drift alert classification (`SEVERE_DRIFT`, `MODERATE_DRIFT`, `STABLE`, `IMPROVING`), trajectory visualization. | 163,161 bytes |
| 06 | `06_gap_detective.png` | **🧩 Knowledge Gap Detective (Feature 2)** | Directed Knowledge Graph DAG visualization, concept mastery status cards, prerequisite gap tracing with unassessed concept flagging. | 175,591 bytes |
| 07 | `07_personalized_recommendations.png` | **💡 Personalized Recommendations** | Triaged pedagogical recommendations (High/Medium/Low priority), evidence bases connecting SHAP drivers, prerequisite gaps, and drift status, non-causal advisory notice. | 165,479 bytes |
| 08 | `08_improvement_tracker.png` | **🔄 Improvement Tracker (Feature 4)** | Prescribed learning interventions table, interactive diagnostic quiz verification, score improvement delta, configured 80% mastery benchmark, closed-loop tracking without causal grade claims. | 178,356 bytes |

---

## Provenance & Integrity Statement

1. **Authentic Execution**: Screenshots were generated against genuine student records in `data/explainai.db` and trained machine learning pipelines in `models/`.
2. **Zero Mockups**: All interface widgets, Plotly graphs, tables, and metric badges were rendered live in the Streamlit application runtime.
3. **Design System**: Built with modern SaaS Light Theme tokens (Primary Navy `#14213D`, Accent Blue `#3B82F6`, Background `#F5F7FB`, Card `#FFFFFF`, Borders `#E2E8F0`, and semantic risk badges).
4. **Reproducibility**: Any reviewer can reproduce these exact visual states by executing:
   ```bash
   python -m streamlit run app.py
   ```
   and navigating through the left-hand sidebar menu options.
