"""
Adapters for External Causal Libraries

Converts between causal-learn/DoWhy formats and BioREASONC internal formats.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO

import numpy as np
import pandas as pd

# Handle both package and standalone imports
try:
    from .graph import CausalGraph, CausalEdge
except ImportError:
    from graph import CausalGraph, CausalEdge

logger = logging.getLogger(__name__)


class CausalLearnAdapter:
    """
    Adapter for causal-learn library outputs

    Converts causal-learn graph formats to BioREASONC CausalGraph.

    causal-learn edge encoding:
    - cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1: i -> j (directed)
    - cg.G.graph[i,j] = cg.G.graph[j,i] = -1: i -- j (undirected)
    - cg.G.graph[i,j] = cg.G.graph[j,i] = 1: i <-> j (bidirected)
    - cg.G.graph[i,j] = 1 and cg.G.graph[j,i] = 2: i o-> j (partially directed)
    """

    @staticmethod
    def to_causal_graph(cg_result: Any, variable_names: List[str]) -> CausalGraph:
        """
        Convert causal-learn CausalGraph to BioREASONC CausalGraph

        Args:
            cg_result: Result from causal-learn pc() function
            variable_names: List of variable names matching column order

        Returns:
            BioREASONC CausalGraph object
        """
        graph = CausalGraph()
        adj_matrix = cg_result.G.graph  # numpy array

        # Add all nodes
        for var in variable_names:
            graph.add_node(var)

        # Parse edge matrix
        n = len(variable_names)
        for i in range(n):
            for j in range(i + 1, n):
                edge_info = CausalLearnAdapter._decode_edge(
                    adj_matrix[i, j], adj_matrix[j, i]
                )
                if edge_info:
                    src_idx, tgt_idx, edge_type = edge_info
                    src_name = variable_names[i] if src_idx == 'i' else variable_names[j]
                    tgt_name = variable_names[j] if src_idx == 'i' else variable_names[i]
                    graph.add_edge(src_name, tgt_name, edge_type)

        return graph

    @staticmethod
    def _decode_edge(val_ij: int, val_ji: int) -> Optional[Tuple[str, str, str]]:
        """
        Decode causal-learn edge encoding

        Args:
            val_ij: Value at position [i,j] in adjacency matrix
            val_ji: Value at position [j,i] in adjacency matrix

        Returns:
            Tuple of (source_index, target_index, edge_type) or None
        """
        if val_ij == -1 and val_ji == 1:
            return ('i', 'j', '->')  # i -> j
        elif val_ij == 1 and val_ji == -1:
            return ('j', 'i', '->')  # j -> i
        elif val_ij == -1 and val_ji == -1:
            return ('i', 'j', '--')  # i -- j (undirected)
        elif val_ij == 1 and val_ji == 1:
            return ('i', 'j', '<->')  # i <-> j (bidirected)
        elif val_ij == 1 and val_ji == 2:
            return ('i', 'j', 'o->')  # i o-> j
        elif val_ij == 2 and val_ji == 1:
            return ('j', 'i', 'o->')  # j o-> i
        return None

    @staticmethod
    def to_causal_graph_from_fci(g: Any, variable_names: List[str]) -> CausalGraph:
        """
        Convert FCI result to CausalGraph

        FCI produces PAGs (Partial Ancestral Graphs) which may have
        additional edge types for latent confounders.

        Args:
            g: FCI result graph
            variable_names: Variable names

        Returns:
            CausalGraph with PAG edges
        """
        graph = CausalGraph()

        for var in variable_names:
            graph.add_node(var)

        # FCI uses similar encoding to PC
        adj_matrix = g.graph if hasattr(g, 'graph') else g

        n = len(variable_names)
        for i in range(n):
            for j in range(i + 1, n):
                edge_info = CausalLearnAdapter._decode_edge(
                    adj_matrix[i, j], adj_matrix[j, i]
                )
                if edge_info:
                    src_idx, tgt_idx, edge_type = edge_info
                    src_name = variable_names[i] if src_idx == 'i' else variable_names[j]
                    tgt_name = variable_names[j] if src_idx == 'i' else variable_names[i]
                    graph.add_edge(src_name, tgt_name, edge_type)

        return graph

    @staticmethod
    def to_causal_graph_from_ges(record: Dict, variable_names: List[str]) -> CausalGraph:
        """
        Convert GES result to CausalGraph

        Args:
            record: GES result dictionary
            variable_names: Variable names

        Returns:
            CausalGraph
        """
        graph = CausalGraph()

        for var in variable_names:
            graph.add_node(var)

        # GES returns a graph in record['G']
        adj_matrix = record['G'].graph if hasattr(record['G'], 'graph') else record['G']

        n = len(variable_names)
        for i in range(n):
            for j in range(i + 1, n):
                edge_info = CausalLearnAdapter._decode_edge(
                    adj_matrix[i, j], adj_matrix[j, i]
                )
                if edge_info:
                    src_idx, tgt_idx, edge_type = edge_info
                    src_name = variable_names[i] if src_idx == 'i' else variable_names[j]
                    tgt_name = variable_names[j] if src_idx == 'i' else variable_names[i]
                    graph.add_edge(src_name, tgt_name, edge_type)

        return graph

    @staticmethod
    def get_separating_sets(cg_result: Any) -> Dict[Tuple[str, str], set]:
        """
        Extract separating sets from causal-learn result

        Args:
            cg_result: Result from PC/FCI algorithm

        Returns:
            Dictionary mapping variable pairs to their separating sets
        """
        if hasattr(cg_result, 'sepset') and cg_result.sepset is not None:
            return cg_result.sepset
        return {}


class DoWhyAdapter:
    """
    Adapter for DoWhy causal inference library

    Converts BioREASONC data structures to DoWhy formats and vice versa.
    """

    @staticmethod
    def build_causal_model(
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        causal_graph: Optional[CausalGraph] = None,
        common_causes: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None
    ) -> 'CausalModel':
        """
        Build DoWhy CausalModel from data and graph structure

        Args:
            data: DataFrame with all variables
            treatment: Treatment variable name
            outcome: Outcome variable name
            causal_graph: BioREASONC CausalGraph (optional)
            common_causes: List of confounder names
            instruments: List of instrumental variable names

        Returns:
            DoWhy CausalModel object
        """
        from dowhy import CausalModel

        # Build GML graph string if causal_graph provided
        gml_graph = None
        if causal_graph:
            gml_graph = DoWhyAdapter.causal_graph_to_gml(
                causal_graph, treatment, outcome, common_causes, instruments
            )

        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=gml_graph,
            common_causes=common_causes,
            instruments=instruments
        )

        return model

    @staticmethod
    def causal_graph_to_gml(
        graph: CausalGraph,
        treatment: str,
        outcome: str,
        common_causes: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None
    ) -> str:
        """
        Convert CausalGraph to GML string for DoWhy

        Args:
            graph: BioREASONC CausalGraph
            treatment: Treatment variable
            outcome: Outcome variable
            common_causes: Confounders
            instruments: Instrumental variables

        Returns:
            GML format string
        """
        try:
            import networkx as nx
        except ImportError:
            logger.error("NetworkX required for GML conversion")
            return None

        # Build NetworkX DiGraph
        G = nx.DiGraph()

        for node in graph.nodes:
            G.add_node(node)

        for edge in graph.edges:
            if edge.edge_type == '->':
                G.add_edge(edge.source, edge.target)

        # Ensure treatment -> outcome path exists
        if treatment in G.nodes and outcome in G.nodes:
            if not G.has_edge(treatment, outcome):
                G.add_edge(treatment, outcome)

        # Add instrument edges (instrument -> treatment)
        if instruments:
            for inst in instruments:
                if inst not in G.nodes:
                    G.add_node(inst)
                if not G.has_edge(inst, treatment):
                    G.add_edge(inst, treatment)

        # Add common cause edges (confounder -> treatment, confounder -> outcome)
        if common_causes:
            for cc in common_causes:
                if cc not in G.nodes:
                    G.add_node(cc)
                if not G.has_edge(cc, treatment):
                    G.add_edge(cc, treatment)
                if not G.has_edge(cc, outcome):
                    G.add_edge(cc, outcome)

        # Convert to GML
        buffer = BytesIO()
        nx.write_gml(G, buffer)
        return buffer.getvalue().decode('utf-8')

    @staticmethod
    def causal_graph_to_dot(graph: CausalGraph) -> str:
        """
        Convert CausalGraph to DOT format

        Args:
            graph: BioREASONC CausalGraph

        Returns:
            DOT format string
        """
        lines = ["digraph G {"]

        for node in graph.nodes:
            lines.append(f'    "{node}";')

        for edge in graph.edges:
            if edge.edge_type == '->':
                lines.append(f'    "{edge.source}" -> "{edge.target}";')
            elif edge.edge_type == '--':
                lines.append(f'    "{edge.source}" -- "{edge.target}";')
            elif edge.edge_type == '<->':
                lines.append(f'    "{edge.source}" -> "{edge.target}" [dir=both];')

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def dowhy_graph_to_causal_graph(dowhy_graph: Any) -> CausalGraph:
        """
        Convert DoWhy graph back to BioREASONC CausalGraph

        Args:
            dowhy_graph: DoWhy internal graph representation

        Returns:
            CausalGraph
        """
        graph = CausalGraph()

        try:
            import networkx as nx

            if isinstance(dowhy_graph, nx.DiGraph):
                for node in dowhy_graph.nodes():
                    graph.add_node(str(node))
                for source, target in dowhy_graph.edges():
                    graph.add_edge(str(source), str(target), '->')
        except Exception as e:
            logger.warning(f"Failed to convert DoWhy graph: {e}")

        return graph
