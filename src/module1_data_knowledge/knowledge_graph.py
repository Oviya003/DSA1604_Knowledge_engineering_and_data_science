"""
ExplainAI - Module 1: Mathematics Knowledge Graph Engine
Implements the curriculum-based NetworkX knowledge graph representation,
prerequisite traversal, transitive closure resolution, and learning path discovery.

CURRICULUM PROVENANCE:
----------------------
This module operates exclusively on the 10-concept Mathematics DEMONSTRATION_CURRICULUM.
It is an educational representation for prerequisite tracing, NOT an official syllabus.
Portuguese is unsupported for concept-level graph analysis and returns a graceful message.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import networkx as nx

from src.module1_data_knowledge.curriculum_catalog import (
    MATHEMATICS_CURRICULUM,
    CURRICULUM_PROVENANCE,
    CURRICULUM_SUBJECT,
    build_curriculum_graph,
    validate_curriculum_dag,
    is_subject_curriculum_supported,
    get_unsupported_subject_message,
)


class AcademicKnowledgeGraph:
    """
    Manages the directed knowledge graph representing academic concepts,
    prerequisite dependencies, and pedagogical pathways.
    """

    def __init__(self, subject: str = "Mathematics"):
        self.subject = subject
        self.is_supported = is_subject_curriculum_supported(subject)
        if self.is_supported:
            self.graph: nx.DiGraph = build_curriculum_graph(MATHEMATICS_CURRICULUM)
            self.validation_result = validate_curriculum_dag(MATHEMATICS_CURRICULUM)
        else:
            self.graph = nx.DiGraph()
            self.validation_result = {
                "valid": False,
                "is_dag": False,
                "message": get_unsupported_subject_message(subject),
            }

    def get_status(self) -> Dict[str, Any]:
        """Returns the operational status, provenance, and node/edge metrics of the graph."""
        if not self.is_supported:
            return {
                "subject": self.subject,
                "supported": False,
                "message": get_unsupported_subject_message(self.subject),
            }
        return {
            "subject": self.subject,
            "supported": True,
            "curriculum_provenance": CURRICULUM_PROVENANCE,
            "num_concepts": self.graph.number_of_nodes(),
            "num_prerequisite_edges": self.graph.number_of_edges(),
            "is_acyclic": nx.is_directed_acyclic_graph(self.graph),
            "concept_ids": list(self.graph.nodes()),
        }

    def get_concept_details(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves metadata for a specific concept node."""
        if not self.is_supported or concept_id not in self.graph:
            return None
        node_data = dict(self.graph.nodes[concept_id])
        node_data["concept_id"] = concept_id
        node_data["direct_prerequisites"] = self.get_direct_prerequisites(concept_id)
        node_data["direct_dependents"] = self.get_direct_dependents(concept_id)
        return node_data

    def get_all_concepts_details(self) -> List[Dict[str, Any]]:
        """Returns metadata for all concepts in the graph."""
        if not self.is_supported:
            return []
        return [self.get_concept_details(cid) for cid in self.graph.nodes()]

    def get_direct_prerequisites(self, concept_id: str) -> List[str]:
        """
        Retrieves direct predecessors (concepts that must be mastered before concept_id).
        Edges in graph: Prerequisite -> Dependent. Thus, predecessors are prerequisites.
        """
        if not self.is_supported or concept_id not in self.graph:
            return []
        return sorted(list(self.graph.predecessors(concept_id)))

    def get_all_ancestor_prerequisites(self, concept_id: str) -> List[str]:
        """
        Retrieves the complete transitive prerequisite closure (all ancestral prerequisites)
        ordered topologically.
        """
        if not self.is_supported or concept_id not in self.graph:
            return []
        ancestors: Set[str] = nx.ancestors(self.graph, concept_id)
        # Sort ancestors according to the topological learning order
        topo_order = self.get_topological_order()
        return [c for c in topo_order if c in ancestors]

    def get_direct_dependents(self, concept_id: str) -> List[str]:
        """
        Retrieves direct successors (concepts that require concept_id as a prerequisite).
        """
        if not self.is_supported or concept_id not in self.graph:
            return []
        return sorted(list(self.graph.successors(concept_id)))

    def get_all_descendant_dependents(self, concept_id: str) -> List[str]:
        """
        Retrieves all downstream concepts (descendants) that depend directly or
        indirectly upon concept_id.
        """
        if not self.is_supported or concept_id not in self.graph:
            return []
        descendants: Set[str] = nx.descendants(self.graph, concept_id)
        topo_order = self.get_topological_order()
        return [c for c in topo_order if c in descendants]

    def get_topological_order(self) -> List[str]:
        """
        Computes a valid, pedagogical learning sequence where every concept
        appears only after all its prerequisite concepts have been encountered.
        """
        if not self.is_supported:
            return []
        return list(nx.topological_sort(self.graph))

    def find_prerequisite_paths(self, source_concept: str, target_concept: str) -> List[List[str]]:
        """
        Finds all directed dependency paths from a foundational concept (source)
        to a target concept.
        """
        if not self.is_supported:
            return []
        if source_concept not in self.graph or target_concept not in self.graph:
            return []
        try:
            return list(nx.all_simple_paths(self.graph, source=source_concept, target=target_concept))
        except nx.NetworkXNoPath:
            return []

    def get_prerequisite_depth(self, concept_id: str) -> int:
        """
        Computes the maximum path length from any foundational root node to concept_id.
        Foundational concepts have depth 0.
        """
        if not self.is_supported or concept_id not in self.graph:
            return 0
        ancestors = nx.ancestors(self.graph, concept_id)
        if not ancestors:
            return 0
        root_nodes = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        max_depth = 0
        for root in root_nodes:
            if root in ancestors or root == concept_id:
                try:
                    for path in nx.all_simple_paths(self.graph, source=root, target=concept_id):
                        max_depth = max(max_depth, len(path) - 1)
                except nx.NetworkXNoPath:
                    pass
        return max_depth

