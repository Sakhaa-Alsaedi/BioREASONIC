"""
Structure-Aware Reasoning Module (S)

Implements graph algorithms for biological network analysis:
- DFS (Depth-First Search)
- BFS (Breadth-First Search)
- A* (A-star pathfinding)
- Greedy best-first search

Uses STRING-DB and BioGRID for protein interaction networks.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable
from collections import deque
import heapq
import requests
import logging
from dataclasses import dataclass

from ..schema import NetworkEdge, RiskGene, BenchmarkItem, StructureLabel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PathResult:
    """Result of a path-finding algorithm"""
    path: List[str]
    distance: float
    algorithm: str
    visited_nodes: int
    found: bool


class BiologicalNetwork:
    """Represents a biological interaction network"""

    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()
        self.gene_info: Dict[str, Dict] = {}
        self.edge_sources: Dict[Tuple[str, str], str] = {}

    def add_edge(self, source: str, target: str, weight: float = 1.0,
                 edge_type: str = "interaction", source_db: str = "unknown"):
        """Add an edge to the network"""
        self.graph.add_edge(source, target, weight=weight, type=edge_type)
        self.directed_graph.add_edge(source, target, weight=weight, type=edge_type)
        self.edge_sources[(source, target)] = source_db

    def add_gene_info(self, gene: str, info: Dict):
        """Add gene metadata"""
        self.gene_info[gene] = info

    def get_neighbors(self, gene: str) -> List[str]:
        """Get neighboring genes"""
        if gene in self.graph:
            return list(self.graph.neighbors(gene))
        return []

    def get_edge_weight(self, source: str, target: str) -> float:
        """Get edge weight"""
        if self.graph.has_edge(source, target):
            return self.graph[source][target].get('weight', 1.0)
        return float('inf')

    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def num_edges(self) -> int:
        return self.graph.number_of_edges()


class GraphAlgorithms:
    """Implementation of graph algorithms for biological networks"""

    def __init__(self, network: BiologicalNetwork):
        self.network = network
        self.graph = network.graph

    # ==================== DFS ====================
    def dfs(self, start: str, target: str = None,
            max_depth: int = 10) -> PathResult:
        """
        Depth-First Search

        Args:
            start: Starting gene
            target: Target gene (optional)
            max_depth: Maximum search depth

        Returns:
            PathResult with path if found
        """
        if start not in self.graph:
            return PathResult([], float('inf'), 'dfs', 0, False)

        visited = set()
        path = []
        found = False
        visited_count = 0

        def dfs_recursive(node: str, depth: int) -> bool:
            nonlocal found, visited_count
            if depth > max_depth:
                return False

            visited.add(node)
            visited_count += 1
            path.append(node)

            if target and node == target:
                found = True
                return True

            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    if dfs_recursive(neighbor, depth + 1):
                        return True

            if not found:
                path.pop()
            return False

        dfs_recursive(start, 0)

        return PathResult(
            path=path if found or not target else [],
            distance=len(path) - 1 if path else float('inf'),
            algorithm='dfs',
            visited_nodes=visited_count,
            found=found or (not target and len(path) > 0)
        )

    def dfs_all_paths(self, start: str, target: str,
                      max_depth: int = 5) -> List[List[str]]:
        """Find all paths between two nodes using DFS"""
        if start not in self.graph or target not in self.graph:
            return []

        all_paths = []

        def dfs_paths(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_depth:
                return

            if current == target:
                all_paths.append(path.copy())
                return

            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs_paths(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        dfs_paths(start, [start], {start})
        return all_paths

    # ==================== BFS ====================
    def bfs(self, start: str, target: str = None,
            max_depth: int = 10) -> PathResult:
        """
        Breadth-First Search

        Args:
            start: Starting gene
            target: Target gene (optional)
            max_depth: Maximum search depth

        Returns:
            PathResult with shortest path if found
        """
        if start not in self.graph:
            return PathResult([], float('inf'), 'bfs', 0, False)

        visited = {start}
        queue = deque([(start, [start], 0)])
        visited_count = 0

        while queue:
            node, path, depth = queue.popleft()
            visited_count += 1

            if depth > max_depth:
                continue

            if target and node == target:
                return PathResult(
                    path=path,
                    distance=len(path) - 1,
                    algorithm='bfs',
                    visited_nodes=visited_count,
                    found=True
                )

            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], depth + 1))

        return PathResult(
            path=[],
            distance=float('inf'),
            algorithm='bfs',
            visited_nodes=visited_count,
            found=False
        )

    def bfs_within_distance(self, start: str, max_distance: int) -> List[str]:
        """Find all nodes within a certain distance using BFS"""
        if start not in self.graph:
            return []

        visited = {start}
        queue = deque([(start, 0)])
        result = [start]

        while queue:
            node, dist = queue.popleft()

            if dist >= max_distance:
                continue

            for neighbor in self.graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
                    result.append(neighbor)

        return result

    # ==================== A* ====================
    def astar(self, start: str, target: str,
              heuristic: Callable[[str, str], float] = None) -> PathResult:
        """
        A* Search Algorithm

        Args:
            start: Starting gene
            target: Target gene
            heuristic: Heuristic function (default: 0)

        Returns:
            PathResult with optimal path
        """
        if start not in self.graph or target not in self.graph:
            return PathResult([], float('inf'), 'astar', 0, False)

        if heuristic is None:
            heuristic = lambda a, b: 0  # Uniform cost search if no heuristic

        # Priority queue: (f_score, counter, node, path)
        counter = 0
        open_set = [(0, counter, start, [start])]
        g_score = {start: 0}
        visited_count = 0

        while open_set:
            f, _, current, path = heapq.heappop(open_set)
            visited_count += 1

            if current == target:
                return PathResult(
                    path=path,
                    distance=g_score[current],
                    algorithm='astar',
                    visited_nodes=visited_count,
                    found=True
                )

            for neighbor in self.graph.neighbors(current):
                tentative_g = g_score[current] + self.network.get_edge_weight(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, target)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor, path + [neighbor]))

        return PathResult(
            path=[],
            distance=float('inf'),
            algorithm='astar',
            visited_nodes=visited_count,
            found=False
        )

    # ==================== Greedy Best-First ====================
    def greedy_best_first(self, start: str, target: str,
                          heuristic: Callable[[str, str], float] = None) -> PathResult:
        """
        Greedy Best-First Search

        Args:
            start: Starting gene
            target: Target gene
            heuristic: Heuristic function

        Returns:
            PathResult (may not be optimal)
        """
        if start not in self.graph or target not in self.graph:
            return PathResult([], float('inf'), 'greedy', 0, False)

        if heuristic is None:
            # Default: prefer nodes with more connections to target's neighbors
            target_neighbors = set(self.graph.neighbors(target))
            heuristic = lambda node, t: -len(set(self.graph.neighbors(node)) & target_neighbors)

        visited = set()
        counter = 0
        # Priority queue: (heuristic, counter, node, path)
        open_set = [(heuristic(start, target), counter, start, [start])]
        visited_count = 0

        while open_set:
            _, _, current, path = heapq.heappop(open_set)

            if current in visited:
                continue

            visited.add(current)
            visited_count += 1

            if current == target:
                return PathResult(
                    path=path,
                    distance=len(path) - 1,
                    algorithm='greedy',
                    visited_nodes=visited_count,
                    found=True
                )

            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    counter += 1
                    h = heuristic(neighbor, target)
                    heapq.heappush(open_set, (h, counter, neighbor, path + [neighbor]))

        return PathResult(
            path=[],
            distance=float('inf'),
            algorithm='greedy',
            visited_nodes=visited_count,
            found=False
        )

    # ==================== Utility Methods ====================
    def find_connected_components(self) -> List[Set[str]]:
        """Find all connected components"""
        return [set(c) for c in nx.connected_components(self.graph)]

    def get_shortest_path(self, source: str, target: str) -> List[str]:
        """Get shortest path using NetworkX (Dijkstra)"""
        try:
            return nx.shortest_path(self.graph, source, target, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_all_shortest_paths(self, source: str, target: str) -> List[List[str]]:
        """Get all shortest paths"""
        try:
            return list(nx.all_shortest_paths(self.graph, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []


class STRINGDBClient:
    """Client for STRING-DB protein interaction database"""

    BASE_URL = "https://string-db.org/api"

    def __init__(self, species: int = 9606):  # 9606 = Human
        self.species = species

    def get_interactions(self, genes: List[str],
                        score_threshold: int = 400) -> List[Dict]:
        """Fetch protein interactions from STRING-DB"""
        if not genes:
            return []

        try:
            url = f"{self.BASE_URL}/json/network"
            params = {
                "identifiers": "%0d".join(genes),
                "species": self.species,
                "required_score": score_threshold,
                "caller_identity": "bioreasonc_bench"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.warning(f"STRING-DB request failed: {e}")
            return []

    def get_interaction_partners(self, gene: str, limit: int = 50) -> List[Dict]:
        """Get interaction partners for a gene"""
        try:
            url = f"{self.BASE_URL}/json/interaction_partners"
            params = {
                "identifiers": gene,
                "species": self.species,
                "limit": limit,
                "caller_identity": "bioreasonc_bench"
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.warning(f"STRING-DB request failed: {e}")
            return []


class StructureReasoning:
    """Main class for Structure-Aware reasoning"""

    def __init__(self):
        self.network = BiologicalNetwork()
        self.algorithms = None
        self.string_client = STRINGDBClient()

    def build_network_from_genes(self, genes: List[RiskGene],
                                  fetch_interactions: bool = True) -> BiologicalNetwork:
        """Build network from gene list"""
        logger.info(f"Building network from {len(genes)} genes...")

        # Add genes as nodes
        for gene in genes:
            self.network.graph.add_node(gene.symbol)
            self.network.add_gene_info(gene.symbol, gene.to_dict())

        # Fetch interactions from STRING-DB
        if fetch_interactions:
            gene_symbols = [g.symbol for g in genes]
            interactions = self.string_client.get_interactions(gene_symbols[:100])  # Limit

            for interaction in interactions:
                source = interaction.get('preferredName_A', '')
                target = interaction.get('preferredName_B', '')
                score = interaction.get('score', 0)

                if source and target:
                    self.network.add_edge(
                        source, target,
                        weight=1.0 - score,  # Lower weight = stronger interaction
                        edge_type='protein_interaction',
                        source_db='string'
                    )

        self.algorithms = GraphAlgorithms(self.network)
        logger.info(f"Network built: {self.network.num_nodes()} nodes, "
                   f"{self.network.num_edges()} edges")

        return self.network

    def build_network_from_edges(self, edges: List[Tuple[str, str, float]]):
        """Build network from edge list"""
        for source, target, weight in edges:
            self.network.add_edge(source, target, weight)

        self.algorithms = GraphAlgorithms(self.network)

    def generate_structure_questions(self, genes: List[RiskGene]) -> List[BenchmarkItem]:
        """Generate structure-aware benchmark questions"""
        questions = []
        gene_symbols = [g.symbol for g in genes if g.symbol]

        # Ensure network is built
        if self.algorithms is None:
            self.build_network_from_genes(genes, fetch_interactions=False)

        # S-GENE-MAP questions
        for gene in genes[:20]:  # Limit
            if gene.associated_variants:
                for variant in gene.associated_variants[:2]:
                    q = BenchmarkItem(
                        id=f"S-{len(questions):04d}",
                        taxonomy="S",
                        label=StructureLabel.GENE_MAP.value,
                        template_id="S-GENE-MAP-01",
                        question=f"Which gene is associated with the variant {variant}?",
                        answer=gene.symbol,
                        explanation=f"The variant {variant} is located within or near the {gene.symbol} gene.",
                        source_genes=[gene.symbol],
                        source_variants=[variant],
                        algorithm_used="mapping"
                    )
                    questions.append(q)

        # S-NETWORK-TRAVERSE questions (BFS)
        for gene in genes[:10]:
            if gene.symbol in self.network.graph:
                neighbors = self.network.get_neighbors(gene.symbol)
                if neighbors:
                    q = BenchmarkItem(
                        id=f"S-{len(questions):04d}",
                        taxonomy="S",
                        label=StructureLabel.NETWORK_TRAVERSE.value,
                        template_id="S-NETWORK-TRAVERSE-01",
                        question=f"Using BFS, find all genes within 2 hops of {gene.symbol} in the interaction network.",
                        answer=str(self.algorithms.bfs_within_distance(gene.symbol, 2)),
                        explanation=f"BFS explores nodes level by level, finding all genes within distance 2 from {gene.symbol}.",
                        source_genes=[gene.symbol],
                        algorithm_used="bfs"
                    )
                    questions.append(q)

        # S-SHORTEST-PATH questions
        if len(gene_symbols) >= 2:
            for i in range(min(10, len(gene_symbols) - 1)):
                gene1, gene2 = gene_symbols[i], gene_symbols[i + 1]
                if gene1 in self.network.graph and gene2 in self.network.graph:
                    path_result = self.algorithms.astar(gene1, gene2)
                    if path_result.found:
                        q = BenchmarkItem(
                            id=f"S-{len(questions):04d}",
                            taxonomy="S",
                            label=StructureLabel.SHORTEST_PATH.value,
                            template_id="S-SHORTEST-PATH-01",
                            question=f"What is the shortest path from {gene1} to {gene2} in the protein interaction network?",
                            answer=str(path_result.path),
                            explanation=f"Using A* algorithm, the shortest path has length {path_result.distance}.",
                            source_genes=[gene1, gene2],
                            algorithm_used="astar"
                        )
                        questions.append(q)

        return questions


# Factory function
def create_structure_module() -> StructureReasoning:
    """Create Structure-Aware reasoning module"""
    return StructureReasoning()


if __name__ == "__main__":
    # Test
    module = create_structure_module()

    # Create test network
    edges = [
        ("A", "B", 1.0),
        ("B", "C", 1.0),
        ("A", "C", 2.0),
        ("C", "D", 1.0),
        ("D", "E", 1.0),
    ]
    module.build_network_from_edges(edges)

    # Test algorithms
    print("DFS A->E:", module.algorithms.dfs("A", "E"))
    print("BFS A->E:", module.algorithms.bfs("A", "E"))
    print("A* A->E:", module.algorithms.astar("A", "E"))
    print("Greedy A->E:", module.algorithms.greedy_best_first("A", "E"))
