"""
Causal Graph Data Structures

Defines CausalEdge and CausalGraph classes for representing causal DAGs.
"""

from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field


@dataclass
class CausalEdge:
    """Represents a causal edge in a DAG"""
    source: str
    target: str
    edge_type: str  # '->', '<-', '--', 'o->', '<->'
    weight: float = 1.0
    p_value: Optional[float] = None


@dataclass
class CausalGraph:
    """
    Represents a causal DAG (Directed Acyclic Graph)

    Supports both directed and undirected edges for representing
    causal discovery results (e.g., from PC algorithm).
    """
    nodes: Set[str] = field(default_factory=set)
    edges: List[CausalEdge] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)

    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.adjacency = {}

    def add_node(self, node: str):
        """Add a node to the graph"""
        self.nodes.add(node)
        if node not in self.adjacency:
            self.adjacency[node] = []

    def add_edge(self, source: str, target: str, edge_type: str = '->',
                 weight: float = 1.0, p_value: float = None):
        """
        Add an edge to the graph

        Args:
            source: Source node name
            target: Target node name
            edge_type: Type of edge ('->', '<-', '--', 'o->', '<->')
            weight: Edge weight (default 1.0)
            p_value: Statistical p-value for the edge (optional)
        """
        self.add_node(source)
        self.add_node(target)
        edge = CausalEdge(source, target, edge_type, weight, p_value)
        self.edges.append(edge)
        self.adjacency[source].append(target)

    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes (direct causes) of a node"""
        parents = []
        for edge in self.edges:
            if edge.target == node and edge.edge_type == '->':
                parents.append(edge.source)
        return parents

    def get_children(self, node: str) -> List[str]:
        """Get child nodes (direct effects) of a node"""
        return self.adjacency.get(node, [])

    def is_ancestor(self, node: str, potential_ancestor: str) -> bool:
        """Check if potential_ancestor is an ancestor of node"""
        visited = set()
        stack = [node]

        while stack:
            current = stack.pop()
            if current == potential_ancestor:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self.get_parents(current))

        return False

    def get_directed_edges(self) -> List[CausalEdge]:
        """Get only directed edges (->)"""
        return [e for e in self.edges if e.edge_type == '->']

    def get_undirected_edges(self) -> List[CausalEdge]:
        """Get only undirected edges (--)"""
        return [e for e in self.edges if e.edge_type == '--']

    def get_bidirected_edges(self) -> List[CausalEdge]:
        """Get only bidirected edges (<->)"""
        return [e for e in self.edges if e.edge_type == '<->']

    def to_adjacency_matrix(self, variable_order: List[str] = None) -> 'np.ndarray':
        """
        Convert graph to adjacency matrix

        Args:
            variable_order: Order of variables (default: sorted node names)

        Returns:
            Numpy array with adjacency matrix
        """
        import numpy as np

        if variable_order is None:
            variable_order = sorted(self.nodes)

        n = len(variable_order)
        var_to_idx = {v: i for i, v in enumerate(variable_order)}

        matrix = np.zeros((n, n))

        for edge in self.edges:
            i = var_to_idx.get(edge.source)
            j = var_to_idx.get(edge.target)
            if i is not None and j is not None:
                if edge.edge_type == '->':
                    matrix[i, j] = 1
                elif edge.edge_type == '--':
                    matrix[i, j] = 1
                    matrix[j, i] = 1
                elif edge.edge_type == '<->':
                    matrix[i, j] = 2
                    matrix[j, i] = 2

        return matrix

    def __repr__(self) -> str:
        return f"CausalGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
