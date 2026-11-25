# ExplainAI: An Explainable AI-Based Student Performance Prediction and Personalized Learning Support System for Inclusive and Quality Education

This document provides a comprehensive technical reference for the relational database schema supporting **ExplainAI: An Explainable AI-Based Student Performance Prediction and Personalized Learning Support System for Inclusive and Quality Education**.

---

## 1. Database Overview

- **RDBMS Engine**: SQLite 3 (WAL mode supported, foreign key constraints strictly enforced via `PRAGMA foreign_keys = ON;`).
- **Database File**: `data/explainai.db`
- **Architectural Scope**: Multi-subject isolation across secondary school **Mathematics** (`data/raw/student-mat.csv`) and **Portuguese Language** (`data/raw/student-por.csv`) cohorts, curriculum knowledge graphs, diagnostic assessments, ML prediction logs, personalized learning recommendations, and intervention tracking.

---

## 2. Entity-Relationship Diagram (ERD)

```mermaid
erDiagram
    STUDENTS ||--o{ ASSESSMENTS : "has"
    STUDENTS ||--o{ PREDICTION_RECORDS : "receives"
    STUDENTS ||--o{ CONCEPT_MASTERY : "demonstrates"
    STUDENTS ||--o{ DIAGNOSTIC_QUIZ_ATTEMPTS : "completes"
    STUDENTS ||--o{ LEARNING_RECOMMENDATIONS : "assigned"
    STUDENTS ||--o{ INTERVENTIONS : "undergoes"

    CONCEPTS ||--o{ CONCEPT_PREREQUISITES : "requires"
    CONCEPTS ||--o{ CONCEPT_PREREQUISITES : "is_prerequisite_for"
    CONCEPTS ||--o{ DIAGNOSTIC_QUESTIONS : "evaluated_by"
    CONCEPTS ||--o{ CONCEPT_MASTERY : "tracked_in"
    CONCEPTS ||--o{ DIAGNOSTIC_QUIZ_ATTEMPTS : "tested_in"
    CONCEPTS ||--o{ INTERVENTIONS : "targeted_in"

    STUDENTS {
        string student_id PK
        string subject PK
        string source_dataset
        string school
        string sex
        int age
        string address
        string famsize
        string Pstatus
        int Medu
        int Fedu
        string Mjob
        string Fjob
        string reason
        string guardian
        int traveltime
        int studytime
        int failures
        string schoolsup
        string famsup
        string paid
        string activities
        string nursery
        string higher
        string internet
        string romantic
        int famrel
        int freetime
        int goout
        int Dalc
        int Walc
        int health
        int absences
        float G1_raw
        float G2_raw
        float G3_raw
        int is_at_risk
        string created_at
    }

    ASSESSMENTS {
        int assessment_id PK
        string student_id FK
        string subject FK
        string assessment_name
        int term_period
        float score
        float max_score
        float percentage
        string assessment_date
        string date_provenance
        int is_uci_source
        string notes
    }

    CONCEPTS {
        string concept_id PK
        string subject
        string name
        string domain
        string difficulty
        string description
        string curriculum_type
    }

    CONCEPT_PREREQUISITES {
        int id PK
        string concept_id FK
        string prerequisite_id FK
    }

    DIAGNOSTIC_QUESTIONS {
        string question_id PK
        string concept_id FK
        string question_text
        string option_a
        string option_b
        string option_c
        string option_d
        string correct_option
        string explanation
        string blooms_level
    }

    DIAGNOSTIC_QUIZ_ATTEMPTS {
        int attempt_id PK
        string student_id FK
        string subject FK
        string concept_id FK
        string quiz_type
        float score
        int total_questions
        int correct_answers
        int mastery_achieved
        string attempt_date
        int is_uci_source
    }

    CONCEPT_MASTERY {
        int id PK
        string student_id FK
        string subject FK
        string concept_id FK
        float mastery_score
        string mastery_status
        string last_assessed
    }

    PREDICTION_RECORDS {
        int prediction_id PK
        string student_id FK
        string subject FK
        string model_experiment
        string model_algorithm
        string risk_category
        float at_risk_probability
        string prediction_timestamp
    }

    LEARNING_RECOMMENDATIONS {
        int recommendation_id PK
        string student_id FK
        string subject FK
        string priority
        string recommendation_text
        string rationale
        string created_at
        string status
    }

    INTERVENTIONS {
        int intervention_id PK
        string student_id FK
        string subject FK
        string target_concept_id FK
        string intervention_type
        string prescribed_date
        string completed_date
        string status
        string prescribed_by
        string notes
        float post_quiz_score
    }

    AUDIT_LOGS {
        int log_id PK
        string student_id
        string subject
        string action_type
        string details
        string timestamp
    }
```

---

## 3. Table Catalog & Column Specifications

### 3.1 `students`
Primary repository of student demographic, socioeconomic, academic background, and raw term performance.
- **Primary Key**: `(student_id, subject)` (Composite key ensuring subject cohort isolation).
- **Row Count**: 1,044 records (395 Mathematics, 649 Portuguese).
- **Key Columns**:
  - `student_id` (`TEXT`, NOT NULL): Identifier formatted as `MAT_001`..`MAT_395` or `POR_001`..`POR_649`.
  - `subject` (`TEXT`, NOT NULL): `'Mathematics'` or `'Portuguese'`.
  - `source_dataset` (`TEXT`, NOT NULL): Originating raw CSV filename.
  - Demographic & Family: `school`, `sex`, `age`, `address`, `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `guardian`.
  - Academic & Study Habits: `traveltime`, `studytime`, `failures`, `schoolsup`, `famsup`, `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic`.
  - Social & Health Habits: `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `absences`.
  - Raw Term Grades: `G1_raw`, `G2_raw`, `G3_raw` (`REAL`, scale 0-20).
  - Ground-Truth Target: `is_at_risk` (`INTEGER`, binary: 1 if `G3_raw < 10`, else 0).
  - `created_at` (`TEXT`, NOT NULL): ISO 8601 creation timestamp.

### 3.2 `assessments`
Longitudinal record of term assessments and quiz evaluations.
- **Primary Key**: `assessment_id` (`INTEGER AUTOINCREMENT`).
- **Foreign Keys**: `(student_id, subject)` references `students(student_id, subject)`.
- **Row Count**: 3,132 records (exactly 3 per student record: G1, G2, G3).
- **Key Columns**:
  - `term_period` (`INTEGER`, NOT NULL): Ordered term sequence: `1` (First Period), `2` (Second Period), `3` (Final Examination).
  - `score` (`REAL`, NOT NULL): Raw grade (0-20).
  - `max_score` (`REAL`, NOT NULL): Fixed at 20.0.
  - `percentage` (`REAL`, NOT NULL): Calculated percentage (`score / max_score * 100`).
  - `assessment_date` (`TEXT`, NULL): Calendar date. Set to `NULL` for genuine UCI records because UCI dataset provides term period order but no authentic calendar dates.
  - `date_provenance` (`TEXT`, NOT NULL): Provenance audit flag: `'NOT_PROVIDED_IN_UCI_SOURCE'` for genuine UCI grades, `'ESTIMATED_DEMONSTRATION'` for synthetic demonstration records, or `'VERIFIED_USER_ENTERED'` for verified user entries.
  - `is_uci_source` (`INTEGER`, NOT NULL): `1` for authentic UCI dataset records, `0` for application-generated records.

### 3.3 `concepts`
Curriculum topics and learning objectives structured into domain hierarchies.
- **Primary Key**: `concept_id` (`TEXT`).
- **Row Count**: 10 records (5 Mathematics, 5 Portuguese).
- **Key Columns**:
  - `concept_id`: E.g., `MATH_ALG_01`, `MATH_ALG_02`, `PORT_GRAM_01`.
  - `subject`: Subject area.
  - `name`: Human-readable title (e.g., "Linear Equations & Inequalities").
  - `domain`: Subject domain (e.g., "Algebra", "Calculus", "Morphosyntax", "Literature").
  - `difficulty`: Pedagogical difficulty tier (`'Foundational'`, `'Intermediate'`, `'Advanced'`).
  - `description`: Formal conceptual summary.
  - `curriculum_type` (`TEXT`, NOT NULL): Explicitly designated as `'demonstration_curriculum'` to reflect expert-crafted capstone demonstration alignment.

### 3.4 `concept_prerequisites`
Directed acyclic graph (DAG) edges representing prerequisite dependency relationships.
- **Primary Key**: `id` (`INTEGER AUTOINCREMENT`).
- **Foreign Keys**:
  - `concept_id` references `concepts(concept_id)`.
  - `prerequisite_id` references `concepts(concept_id)`.
- **Row Count**: 11 dependency edges across mathematics and language domains.
- **Integrity Guarantee**: Directed Acyclic Graph topology strictly verified via NetworkX (`nx.is_directed_acyclic_graph(G) == True`).

### 3.5 `diagnostic_questions`
Diagnostic multiple-choice items mapped to curriculum concepts.
- **Primary Key**: `question_id` (`TEXT`).
- **Foreign Key**: `concept_id` references `concepts(concept_id)`.
- **Row Count**: 20 items (exactly 2 questions per concept).
- **Columns**: `question_text`, `option_a`, `option_b`, `option_c`, `option_d`, `correct_option`, `explanation`, `blooms_level` (`'Remember'`, `'Understand'`, `'Apply'`, `'Analyze'`).

### 3.6 `diagnostic_quiz_attempts`
Logs of student submissions on diagnostic concept quizzes.
- **Primary Key**: `attempt_id` (`INTEGER AUTOINCREMENT`).
- **Foreign Keys**: `(student_id, subject)` references `students(student_id, subject)`, `concept_id` references `concepts(concept_id)`.
- **Columns**: `quiz_type` (`'baseline'`, `'post_intervention'`), `score`, `total_questions`, `correct_answers`, `mastery_achieved` (0 or 1), `attempt_date`, `is_uci_source` (always `0`).

### 3.7 `concept_mastery`
Current mastery state of a student for each concept.
- **Primary Key**: `id` (`INTEGER AUTOINCREMENT`).
- **Foreign Keys**: `(student_id, subject)` references `students(student_id, subject)`, `concept_id` references `concepts(concept_id)`.
- **Columns**: `mastery_score` (0.0 to 1.0), `mastery_status` (`'Mastered'`, `'Developing'`, `'Needs Remediation'`), `last_assessed` (ISO timestamp).

### 3.8 `prediction_records`
Audit trail of ML risk predictions generated during application usage.
- **Primary Key**: `prediction_id` (`INTEGER AUTOINCREMENT`).
- **Foreign Keys**: `(student_id, subject)` references `students(student_id, subject)`.
- **Columns**: `model_experiment` (`'MODEL_A'`, `'MODEL_B'`), `model_algorithm` (`'LogisticRegression'`, `'RandomForest'`, `'XGBoost'`), `risk_category` (`'Low Risk'`, `'Moderate Risk'`, `'High Risk'`), `at_risk_probability` (0.0 to 1.0), `prediction_timestamp`.

### 3.9 `learning_recommendations`
Personalized pedagogical recommendations synthesized from ML risk drivers, rule warnings, and prerequisite gap analysis.
- **Primary Key**: `recommendation_id` (`INTEGER AUTOINCREMENT`).
- **Foreign Keys**: `(student_id, subject)` references `students(student_id, subject)`.
- **Columns**: `priority` (`'High'`, `'Medium'`, `'Low'`), `recommendation_text`, `rationale`, `created_at`, `status` (`'Active'`, `'Dismissed'`, `'Completed'`).

### 3.10 `interventions`
Prescribed pedagogical remediation activities supporting Signature Feature 4 (Closed-Loop Improvement Tracker).
- **Primary Key**: `intervention_id` (`INTEGER AUTOINCREMENT`).
- **Foreign Keys**: `(student_id, subject)` references `students(student_id, subject)`, `target_concept_id` references `concepts(concept_id)`.
- **Columns**: `intervention_type` (`'Peer Tutoring'`, `'Targeted Problem Set'`, `'Remedial Workshop'`), `prescribed_date`, `completed_date`, `status` (`'Prescribed'`, `'In Progress'`, `'Completed'`), `prescribed_by`, `notes`, `post_quiz_score` (recorded upon verification).

### 3.11 `audit_logs`
System-wide audit trail recording critical actions and administrative events.
- **Primary Key**: `log_id` (`INTEGER AUTOINCREMENT`).
- **Columns**: `student_id`, `subject`, `action_type`, `details`, `timestamp`.

---

## 4. Date Provenance & Assessment Integrity Rules

1. **UCI Calendar Date Prohibition**:
   The raw UCI Student Performance dataset contains term evaluations denoted by $G1, G2, G3$ but does not contain calendar timestamps. The ExplainAI database schema enforces that `assessment_date` remains `NULL` for all authentic UCI entries (`is_uci_source = 1`).
2. **Date Provenance Flags**:
   Every assessment record carries a mandatory `date_provenance` indicator:
   - `'NOT_PROVIDED_IN_UCI_SOURCE'`: Authentic grade from the original UCI study; chronological ordering is established strictly by `term_period` (1, 2, 3) without calendar dates.
   - `'ESTIMATED_DEMONSTRATION'`: Synthetic demonstration records generated for illustrative purposes in the dashboard.
   - `'VERIFIED_USER_ENTERED'`: Records entered and confirmed with an authentic user-supplied date.
3. **Temporal Anti-Leakage Constraints**:
   In accordance with educational early-warning system principles, Model B inference and Post-G1 evaluations are architecturally restricted from querying assessment records where `term_period > 1` ($G2$ or $G3$).

