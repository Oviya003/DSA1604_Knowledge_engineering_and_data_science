"""
ExplainAI - SQLite Database Schema Definition
Defines tables, relationships, constraints, and indices for persistent
storage of students, assessments, curriculum concepts, prerequisites,
diagnostic quizzes, concept mastery, interventions, recommendations, and audit logs.

Supports multi-subject student enrollments without record contamination,
enforces foreign key cascading, and clearly demarcates UCI source data
from application-generated diagnostic and intervention tracking data.
"""

SCHEMA_SQL = """
-- 1. Students / Enrollments Table
-- Composite Primary Key (student_id, subject) ensures that a student enrolled
-- in both Mathematics and Portuguese maintains completely independent academic
-- records, absence tallies, and risk scores without collisions or overwrites.
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT NOT NULL,
    subject TEXT NOT NULL,                  -- 'Mathematics' or 'Portuguese'
    source_dataset TEXT NOT NULL,           -- 'UCI_MATHEMATICS', 'UCI_PORTUGUESE', or 'SYNTHETIC_DEMO'
    school TEXT NOT NULL,
    sex TEXT NOT NULL,
    age INTEGER NOT NULL,
    address TEXT NOT NULL,
    famsize TEXT NOT NULL,
    Pstatus TEXT NOT NULL,
    Medu INTEGER NOT NULL,
    Fedu INTEGER NOT NULL,
    Mjob TEXT NOT NULL,
    Fjob TEXT NOT NULL,
    reason TEXT NOT NULL,
    guardian TEXT NOT NULL,
    traveltime INTEGER NOT NULL,
    studytime INTEGER NOT NULL,
    failures INTEGER NOT NULL,
    schoolsup TEXT NOT NULL,
    famsup TEXT NOT NULL,
    paid TEXT NOT NULL,
    activities TEXT NOT NULL,
    nursery TEXT NOT NULL,
    higher TEXT NOT NULL,
    internet TEXT NOT NULL,
    romantic TEXT NOT NULL,
    famrel INTEGER NOT NULL,
    freetime INTEGER NOT NULL,
    goout INTEGER NOT NULL,
    Dalc INTEGER NOT NULL,
    Walc INTEGER NOT NULL,
    health INTEGER NOT NULL,
    absences INTEGER NOT NULL,
    G1_raw REAL NOT NULL,
    G2_raw REAL NOT NULL,
    G3_raw REAL NOT NULL,
    is_at_risk INTEGER NOT NULL,            -- 1 if G3_raw < 10 else 0
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (student_id, subject)
);

-- 2. Assessments Table (Chronological history for drift detection)
-- assessment_date is NULLABLE for UCI source assessments because the UCI dataset
-- records term period ordering (G1, G2, G3) but lacks calendar dates.
-- date_provenance explicitly tags whether a calendar date is absent, estimated, or verified.
CREATE TABLE IF NOT EXISTS assessments (
    assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    assessment_name TEXT NOT NULL,
    term_period INTEGER NOT NULL,           -- 1 for Term 1 (G1), 2 for Term 2 (G2), 3 for Final (G3), 4+ for subsequent
    score REAL NOT NULL,                    -- 0.0 to 20.0 scale
    max_score REAL NOT NULL DEFAULT 20.0,
    percentage REAL NOT NULL,
    assessment_date TEXT,                   -- Nullable ISO-8601 YYYY-MM-DD
    date_provenance TEXT NOT NULL DEFAULT 'NOT_PROVIDED_IN_UCI_SOURCE', -- 'NOT_PROVIDED_IN_UCI_SOURCE', 'ESTIMATED_DEMONSTRATION', 'VERIFIED_USER_ENTERED'
    is_uci_source INTEGER NOT NULL,         -- 1: genuine UCI dataset record, 0: application-generated
    notes TEXT,
    FOREIGN KEY (student_id, subject) REFERENCES students(student_id, subject) ON DELETE CASCADE
);

-- 3. Concepts Table (Curriculum knowledge representation)
CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL DEFAULT 'Mathematics',
    name TEXT NOT NULL,
    domain TEXT NOT NULL,
    difficulty TEXT NOT NULL,               -- 'Foundational', 'Intermediate', 'Advanced'
    description TEXT NOT NULL,
    curriculum_type TEXT NOT NULL DEFAULT 'DEMONSTRATION_CURRICULUM' -- Explicitly marked demonstration curriculum
);

-- 4. Concept Prerequisites Table (Directed edges in DAG)
CREATE TABLE IF NOT EXISTS concept_prerequisites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id TEXT NOT NULL,
    prerequisite_id TEXT NOT NULL,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
    FOREIGN KEY (prerequisite_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
    UNIQUE(concept_id, prerequisite_id)
);

-- 5. Diagnostic Questions Table
CREATE TABLE IF NOT EXISTS diagnostic_questions (
    question_id TEXT PRIMARY KEY,
    concept_id TEXT NOT NULL,
    question_text TEXT NOT NULL,
    option_a TEXT NOT NULL,
    option_b TEXT NOT NULL,
    option_c TEXT NOT NULL,
    option_d TEXT NOT NULL,
    correct_option TEXT NOT NULL,
    explanation TEXT NOT NULL,
    blooms_level TEXT NOT NULL,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE
);

-- 6. Diagnostic Quiz Attempts Table
CREATE TABLE IF NOT EXISTS diagnostic_quiz_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT 'Mathematics',
    concept_id TEXT NOT NULL,
    quiz_type TEXT NOT NULL,                -- 'BASELINE_DIAGNOSTIC', 'POST_INTERVENTION_CHECK', 'PRACTICE'
    score REAL NOT NULL,                    -- 0.0 to 1.0 (or 0 to 100%)
    total_questions INTEGER NOT NULL,
    correct_answers INTEGER NOT NULL,
    mastery_achieved INTEGER NOT NULL,      -- 1 if score >= 0.75 else 0
    attempt_date TEXT NOT NULL,
    is_uci_source INTEGER NOT NULL DEFAULT 0, -- Explicitly 0 (application-generated)
    FOREIGN KEY (student_id, subject) REFERENCES students(student_id, subject) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE
);

-- 7. Concept Mastery Table (Student current mastery state per subject & concept)
CREATE TABLE IF NOT EXISTS concept_mastery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT 'Mathematics',
    concept_id TEXT NOT NULL,
    mastery_score REAL NOT NULL,            -- 0.0 to 1.0
    mastery_status TEXT NOT NULL,           -- 'MASTERED' (>=0.75), 'IN_PROGRESS' (0.50-0.74), 'CRITICAL_GAP' (<0.50)
    last_assessed TEXT NOT NULL,
    FOREIGN KEY (student_id, subject) REFERENCES students(student_id, subject) ON DELETE CASCADE,
    FOREIGN KEY (concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE,
    UNIQUE(student_id, subject, concept_id)
);

-- 8. Interventions Table (Signature Feature 4: Closed-loop tracker)
CREATE TABLE IF NOT EXISTS interventions (
    intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT 'Mathematics',
    target_concept_id TEXT NOT NULL,
    intervention_type TEXT NOT NULL,        -- 'PEER_TUTORING', 'REMEDIAL_EXERCISES', 'TEACHER_1ON1', 'CONCEPT_REVISION'
    prescribed_date TEXT NOT NULL,
    completed_date TEXT,
    status TEXT NOT NULL DEFAULT 'PRESCRIBED', -- 'PRESCRIBED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED'
    prescribed_by TEXT NOT NULL DEFAULT 'Academic Support System',
    notes TEXT,
    post_quiz_score REAL,                   -- Populated upon completion
    FOREIGN KEY (student_id, subject) REFERENCES students(student_id, subject) ON DELETE CASCADE,
    FOREIGN KEY (target_concept_id) REFERENCES concepts(concept_id) ON DELETE CASCADE
);

-- 9. Learning Recommendations Table
CREATE TABLE IF NOT EXISTS learning_recommendations (
    recommendation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT 'Mathematics',
    priority TEXT NOT NULL,                 -- 'HIGH', 'MEDIUM', 'LOW'
    recommendation_text TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    status TEXT NOT NULL DEFAULT 'ACTIVE',  -- 'ACTIVE', 'ACKNOWLEDGED', 'COMPLETED'
    FOREIGN KEY (student_id, subject) REFERENCES students(student_id, subject) ON DELETE CASCADE
);

-- 10. Audit Logs Table (Tracks What-If simulations, predictions, and model decisions)
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    subject TEXT,
    action_type TEXT NOT NULL,              -- 'WHAT_IF_SIMULATION', 'PREDICTION_INFERENCE', 'INTERVENTION_UPDATE'
    details TEXT NOT NULL,                  -- JSON string
    timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

-- 11. Prediction Records Table (Stores inference outcomes for student profiles)
CREATE TABLE IF NOT EXISTS prediction_records (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    model_experiment TEXT NOT NULL,         -- 'MODEL_A' or 'MODEL_B'
    model_algorithm TEXT NOT NULL,
    risk_category TEXT NOT NULL,            -- 'At-Risk' or 'Satisfactory'
    at_risk_probability REAL NOT NULL,
    prediction_timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (student_id, subject) REFERENCES students(student_id, subject) ON DELETE CASCADE
);

-- Performance and lookup indices
CREATE INDEX IF NOT EXISTS idx_students_subject ON students(student_id, subject);
CREATE INDEX IF NOT EXISTS idx_assessments_term ON assessments(student_id, subject, term_period);
CREATE INDEX IF NOT EXISTS idx_mastery_student ON concept_mastery(student_id, subject);
CREATE INDEX IF NOT EXISTS idx_interventions_student ON interventions(student_id, subject);
CREATE INDEX IF NOT EXISTS idx_quiz_student_concept ON diagnostic_quiz_attempts(student_id, subject, concept_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_student ON learning_recommendations(student_id, subject);
CREATE INDEX IF NOT EXISTS idx_predictions_student ON prediction_records(student_id, subject);
"""
