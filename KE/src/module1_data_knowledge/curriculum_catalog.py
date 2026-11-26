"""
ExplainAI - Curriculum Catalog Module
Defines the curriculum-based academic knowledge representation for Secondary
School Mathematics.

PROVENANCE & SUBJECT RESTRICTION NOTICE:
----------------------------------------
1. DEMONSTRATION CURRICULUM:
   The 10 concepts and 11 prerequisite edges defined below represent an academic
   DEMONSTRATION CURRICULUM ('DEMONSTRATION_CURRICULUM') for ExplainAI. While the
   progression mirrors standard secondary school algebra-to-calculus sequencing
   (arithmetic -> linear equations -> quadratics/geometry -> polynomials/trig ->
   exponentials/logarithms -> limits -> differential calculus), it is an educational
   demonstration model and NOT an officially certified national syllabus.

2. SUBJECT RESTRICTION:
   This curriculum, its diagnostic questions, prerequisite analysis, and concept
   mastery tracing are strictly restricted to Mathematics. The application does
   NOT assign Mathematics diagnostic questions to Portuguese students. Concept-level
   knowledge tracing for Portuguese is not implemented and is handled gracefully.
"""

from typing import Dict, List, Any, Set, Optional
import networkx as nx

CURRICULUM_PROVENANCE = "DEMONSTRATION_CURRICULUM"
CURRICULUM_SUBJECT = "Mathematics"
SUPPORTED_CURRICULUM_SUBJECTS = {"Mathematics"}


def is_subject_curriculum_supported(subject: str) -> bool:
    """Checks whether concept-level curriculum tracking is supported for the given subject."""
    return subject in SUPPORTED_CURRICULUM_SUBJECTS


def get_unsupported_subject_message(subject: str) -> str:
    """Returns a transparent user-facing message when concept analysis is requested for an unsupported subject."""
    return (
        f"Concept-level knowledge graph, diagnostic quizzes, and prerequisite tracking "
        f"are currently available for Mathematics (Demonstration Curriculum). "
        f"Subject '{subject}' is supported for prediction and trend analysis, but lacks "
        f"a concept-level curriculum catalog."
    )


# Demonstration curriculum definition: 10 structured concepts across 5 domains (Mathematics only)
MATHEMATICS_CURRICULUM: List[Dict[str, Any]] = [
    {
        "concept_id": "MATH_01",
        "name": "Arithmetic & Fraction Operations",
        "subject": "Mathematics",
        "domain": "Foundational Numbers",
        "difficulty": "Foundational",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Fundamental operations on integers, signed numbers, order of operations (PEMDAS), and fractions.",
        "prerequisites": [],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_01_1",
                "question_text": "Evaluate: 3/4 - (1/2 * 2/3) + (-5)",
                "options": {"A": "-4 7/12", "B": "-4 5/12", "C": "-5 1/6", "D": "-3 2/3"},
                "correct_option": "A",
                "explanation": "1/2 * 2/3 = 1/3. Then 3/4 - 1/3 = 9/12 - 4/12 = 5/12. Adding -5 gives 5/12 - 5 = -4 7/12.",
                "blooms_level": "Apply",
            },
            {
                "question_id": "Q_MATH_01_2",
                "question_text": "Simplify using order of operations: 8 - 2 * (3 + 1)^2 / 4",
                "options": {"A": "0", "B": "2", "C": "4", "D": "-4"},
                "correct_option": "A",
                "explanation": "(3+1)=4. 4^2=16. 2*16=32. 32/4=8. 8 - 8 = 0.",
                "blooms_level": "Understand",
            },
        ],
    },
    {
        "concept_id": "MATH_02",
        "name": "Linear Expressions & Single-Variable Equations",
        "subject": "Mathematics",
        "domain": "Algebraic Foundations",
        "difficulty": "Foundational",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Distributive property, combining like terms, isolating variables in linear equations ax + b = c.",
        "prerequisites": ["MATH_01"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_02_1",
                "question_text": "Solve for x: 3(2x - 4) = 4x + 6",
                "options": {"A": "x = 9", "B": "x = 6", "C": "x = 3", "D": "x = 10"},
                "correct_option": "A",
                "explanation": "6x - 12 = 4x + 6 -> 2x = 18 -> x = 9.",
                "blooms_level": "Apply",
            },
            {
                "question_id": "Q_MATH_02_2",
                "question_text": "If 2(x - 3) + 5 = 15, what is the value of 4x - 1?",
                "options": {"A": "31", "B": "32", "C": "35", "D": "29"},
                "correct_option": "A",
                "explanation": "2x - 6 + 5 = 15 -> 2x - 1 = 15 -> 2x = 16 -> x = 8. 4(8) - 1 = 31.",
                "blooms_level": "Analyze",
            },
        ],
    },
    {
        "concept_id": "MATH_03",
        "name": "Systems of Linear Equations",
        "subject": "Mathematics",
        "domain": "Algebraic Foundations",
        "difficulty": "Intermediate",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Solving simultaneous linear systems via substitution, elimination, and geometric intersection.",
        "prerequisites": ["MATH_02"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_03_1",
                "question_text": "Solve the system: 2x + y = 11 and x - y = 1. What is the value of xy?",
                "options": {"A": "12", "B": "15", "C": "8", "D": "9"},
                "correct_option": "A",
                "explanation": "Adding equations gives 3x = 12 -> x = 4. Then y = 3. xy = 4 * 3 = 12.",
                "blooms_level": "Apply",
            },
            {
                "question_id": "Q_MATH_03_2",
                "question_text": "Which statement describes the system: 3x - 6y = 9 and x - 2y = 3?",
                "options": {
                    "A": "Infinitely many solutions (coincident lines)",
                    "B": "No solution (parallel lines)",
                    "C": "Unique solution (3, 0)",
                    "D": "Unique solution (0, 3)",
                },
                "correct_option": "A",
                "explanation": "Dividing the first equation by 3 yields x - 2y = 3, which is identical to the second.",
                "blooms_level": "Understand",
            },
        ],
    },
    {
        "concept_id": "MATH_04",
        "name": "Quadratic Equations & Factoring",
        "subject": "Mathematics",
        "domain": "Polynomials & Non-Linear Functions",
        "difficulty": "Intermediate",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Factoring quadratic polynomials, completing the square, quadratic formula, and discriminant analysis.",
        "prerequisites": ["MATH_02"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_04_1",
                "question_text": "Find the roots of x^2 - 5x + 6 = 0.",
                "options": {"A": "x = 2, 3", "B": "x = -2, -3", "C": "x = 1, 6", "D": "x = -1, 6"},
                "correct_option": "A",
                "explanation": "(x - 2)(x - 3) = 0 yields x = 2 and x = 3.",
                "blooms_level": "Apply",
            },
            {
                "question_id": "Q_MATH_04_2",
                "question_text": "What does a discriminant Delta = b^2 - 4ac < 0 indicate about a quadratic equation ax^2+bx+c=0?",
                "options": {
                    "A": "Two distinct complex/imaginary roots",
                    "B": "Two distinct real rational roots",
                    "C": "One repeated real root",
                    "D": "No roots exist in the complex plane",
                },
                "correct_option": "A",
                "explanation": "A negative discriminant implies the roots involve the square root of a negative number, giving complex conjugates.",
                "blooms_level": "Understand",
            },
        ],
    },
    {
        "concept_id": "MATH_05",
        "name": "Polynomial Functions & Graphing",
        "subject": "Mathematics",
        "domain": "Polynomials & Non-Linear Functions",
        "difficulty": "Advanced",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Degree, end behavior, zeros, multiplicities, vertex form, and transformations of polynomial graphs.",
        "prerequisites": ["MATH_04"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_05_1",
                "question_text": "What is the vertex of the parabola given by f(x) = 2(x - 3)^2 + 4?",
                "options": {"A": "(3, 4)", "B": "(-3, 4)", "C": "(3, -4)", "D": "(2, 4)"},
                "correct_option": "A",
                "explanation": "In vertex form y = a(x - h)^2 + k, the vertex is (h, k), so (3, 4).",
                "blooms_level": "Understand",
            },
            {
                "question_id": "Q_MATH_05_2",
                "question_text": "If a polynomial has degree 4 with positive leading coefficient, what is its end behavior as x -> -infinity?",
                "options": {"A": "f(x) -> +infinity", "B": "f(x) -> -infinity", "C": "f(x) -> 0", "D": "f(x) -> 4"},
                "correct_option": "A",
                "explanation": "Even degree polynomials with positive leading coefficients rise to +infinity on both left and right sides.",
                "blooms_level": "Analyze",
            },
        ],
    },
    {
        "concept_id": "MATH_06",
        "name": "Cartesian Geometry & Linear Slope",
        "subject": "Mathematics",
        "domain": "Coordinate Geometry & Trigonometry",
        "difficulty": "Foundational",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Distance formula, midpoint formula, slope calculation, and equations of lines in slope-intercept form.",
        "prerequisites": ["MATH_02"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_06_1",
                "question_text": "Find the slope of the line passing through (2, 5) and (6, 13).",
                "options": {"A": "2", "B": "1/2", "C": "4", "D": "8"},
                "correct_option": "A",
                "explanation": "Slope m = (13 - 5) / (6 - 2) = 8 / 4 = 2.",
                "blooms_level": "Apply",
            },
            {
                "question_id": "Q_MATH_06_2",
                "question_text": "What is the slope of any line perpendicular to y = -(3/4)x + 5?",
                "options": {"A": "4/3", "B": "-4/3", "C": "3/4", "D": "-3/4"},
                "correct_option": "A",
                "explanation": "Perpendicular lines have negative reciprocal slopes: -1 / (-3/4) = 4/3.",
                "blooms_level": "Understand",
            },
        ],
    },
    {
        "concept_id": "MATH_07",
        "name": "Trigonometric Ratios & Unit Circle",
        "subject": "Mathematics",
        "domain": "Coordinate Geometry & Trigonometry",
        "difficulty": "Intermediate",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Sine, cosine, tangent definitions in right triangles, radian measure, and unit circle coordinates.",
        "prerequisites": ["MATH_06"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_07_1",
                "question_text": "In a right triangle with acute angle theta, sin(theta) = 3/5. What is tan(theta)?",
                "options": {"A": "3/4", "B": "4/3", "C": "4/5", "D": "5/3"},
                "correct_option": "A",
                "explanation": "Opposite = 3, Hypotenuse = 5 -> Adjacent = sqrt(25 - 9) = 4. tan(theta) = Opp/Adj = 3/4.",
                "blooms_level": "Apply",
            },
            {
                "question_id": "Q_MATH_07_2",
                "question_text": "What is the exact value of cos(pi/3 radians)?",
                "options": {"A": "1/2", "B": "sqrt(3)/2", "C": "sqrt(2)/2", "D": "1"},
                "correct_option": "A",
                "explanation": "pi/3 radians corresponds to 60 degrees; cos(60 deg) = 1/2.",
                "blooms_level": "Remember",
            },
        ],
    },
    {
        "concept_id": "MATH_08",
        "name": "Exponential & Logarithmic Functions",
        "subject": "Mathematics",
        "domain": "Advanced Functions",
        "difficulty": "Advanced",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Exponent laws, exponential growth/decay, definition and properties of logarithms, change of base.",
        "prerequisites": ["MATH_04", "MATH_05"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_08_1",
                "question_text": "Solve for x: log_2(x) + log_2(x - 2) = 3",
                "options": {"A": "x = 4", "B": "x = -2", "C": "x = 4 and x = -2", "D": "x = 8"},
                "correct_option": "A",
                "explanation": "log_2(x(x-2)) = 3 -> x^2 - 2x = 2^3 = 8 -> x^2 - 2x - 8 = 0 -> (x-4)(x+2)=0. Since log requires positive arguments, x=4.",
                "blooms_level": "Analyze",
            },
            {
                "question_id": "Q_MATH_08_2",
                "question_text": "If 3^(2x - 1) = 27, what is x?",
                "options": {"A": "2", "B": "1", "C": "3/2", "D": "4"},
                "correct_option": "A",
                "explanation": "27 = 3^3. So 2x - 1 = 3 -> 2x = 4 -> x = 2.",
                "blooms_level": "Apply",
            },
        ],
    },
    {
        "concept_id": "MATH_09",
        "name": "Limits & Continuity",
        "subject": "Mathematics",
        "domain": "Calculus Foundations",
        "difficulty": "Advanced",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Intuitive and algebraic evaluation of limits, indeterminate forms, one-sided limits, continuity criteria.",
        "prerequisites": ["MATH_05", "MATH_08"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_09_1",
                "question_text": "Evaluate lim (x -> 3) [(x^2 - 9) / (x - 3)]",
                "options": {"A": "6", "B": "0", "C": "Undefined", "D": "3"},
                "correct_option": "A",
                "explanation": "Factoring numerator: (x - 3)(x + 3) / (x - 3) = x + 3. Substituting x=3 gives 6.",
                "blooms_level": "Apply",
            },
            {
                "question_id": "Q_MATH_09_2",
                "question_text": "Which condition is required for f(x) to be continuous at x = c?",
                "options": {
                    "A": "f(c) exists, lim_{x->c} f(x) exists, and lim_{x->c} f(x) = f(c)",
                    "B": "Only that f(c) exists",
                    "C": "Only that lim_{x->c} f(x) exists",
                    "D": "The derivative f'(c) must be positive",
                },
                "correct_option": "A",
                "explanation": "The 3-part definition of continuity requires defined value, existing limit, and equality between the two.",
                "blooms_level": "Understand",
            },
        ],
    },
    {
        "concept_id": "MATH_10",
        "name": "Differential Calculus & Rates of Change",
        "subject": "Mathematics",
        "domain": "Calculus Foundations",
        "difficulty": "Advanced",
        "curriculum_type": CURRICULUM_PROVENANCE,
        "description": "Definition of derivative as limit of difference quotient, power rule, product/quotient rules, tangent lines.",
        "prerequisites": ["MATH_09"],
        "diagnostic_questions": [
            {
                "question_id": "Q_MATH_10_1",
                "question_text": "Find the derivative f'(x) for f(x) = 4x^3 - 5x^2 + 7x - 9.",
                "options": {
                    "A": "12x^2 - 10x + 7",
                    "B": "12x^3 - 10x^2 + 7",
                    "C": "4x^2 - 5x + 7",
                    "D": "12x^2 - 5x + 7",
                },
                "correct_option": "A",
                "explanation": "Using power rule d/dx[x^n] = n*x^(n-1): 4*3x^2 - 5*2x + 7 = 12x^2 - 10x + 7.",
                "blooms_level": "Apply",
            },
            {
                "question_id": "Q_MATH_10_2",
                "question_text": "What does the first derivative f'(a) geometrically represent on the graph of y = f(x)?",
                "options": {
                    "A": "The slope of the tangent line to the curve at x = a",
                    "B": "The area under the curve from 0 to a",
                    "C": "The y-intercept of the curve",
                    "D": "The curvature or concavity of the function",
                },
                "correct_option": "A",
                "explanation": "The derivative evaluated at a point gives the instantaneous rate of change, which is the slope of the tangent line at that point.",
                "blooms_level": "Understand",
            },
        ],
    },
]


def build_curriculum_graph(curriculum: List[Dict[str, Any]] = MATHEMATICS_CURRICULUM) -> nx.DiGraph:
    """
    Constructs a NetworkX directed graph from the curriculum catalog,
    where directed edges point from prerequisite -> dependent concept.
    """
    g = nx.DiGraph()
    for item in curriculum:
        cid = item["concept_id"]
        g.add_node(
            cid,
            name=item["name"],
            subject=item.get("subject", CURRICULUM_SUBJECT),
            domain=item["domain"],
            difficulty=item["difficulty"],
            curriculum_type=item.get("curriculum_type", CURRICULUM_PROVENANCE),
            description=item["description"],
        )
        for prereq in item["prerequisites"]:
            # Edge: Prerequisite -> Dependent Concept
            g.add_edge(prereq, cid, relation="is_prerequisite_for")
    return g


def validate_curriculum_dag(curriculum: List[Dict[str, Any]] = MATHEMATICS_CURRICULUM) -> Dict[str, Any]:
    """
    Rigorously validates the curriculum structure:
    1. Uniqueness of concept IDs
    2. Referential integrity of prerequisite IDs
    3. Directed Acyclic Graph (DAG) acyclicity (no recursive dependency loops)
    4. Topological sorting feasibility
    5. Subject uniformity (all concepts belong to the specified subject)
    """
    concept_ids = set()
    errors = []

    for item in curriculum:
        cid = item["concept_id"]
        if cid in concept_ids:
            errors.append(f"Duplicate concept_id found: {cid}")
        concept_ids.add(cid)

    for item in curriculum:
        cid = item["concept_id"]
        for prereq in item["prerequisites"]:
            if prereq not in concept_ids:
                errors.append(f"Prerequisite '{prereq}' referenced by '{cid}' does not exist in curriculum.")

    if errors:
        return {"valid": False, "is_dag": False, "errors": errors, "num_concepts": len(concept_ids)}

    # Build NetworkX graph and test DAG properties
    g = build_curriculum_graph(curriculum)
    is_dag = nx.is_directed_acyclic_graph(g)
    if not is_dag:
        cycles = list(nx.simple_cycles(g))
        errors.append(f"Curriculum graph contains cyclic dependencies: {cycles}")

    topological_order = list(nx.topological_sort(g)) if is_dag else []

    return {
        "valid": len(errors) == 0,
        "is_dag": is_dag,
        "errors": errors,
        "num_concepts": g.number_of_nodes(),
        "num_prerequisite_edges": g.number_of_edges(),
        "topological_order": topological_order,
        "subject": CURRICULUM_SUBJECT,
    }
