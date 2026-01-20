"""
Causal Discovery Module

Wraps causal-learn library for causal structure discovery:
- PC (Peter-Clark) algorithm
- FCI (Fast Causal Inference) for latent confounders
- GES (Greedy Equivalence Search) score-based method
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set

import numpy as np
import pandas as pd

# Handle both package and standalone imports
try:
    from .graph import CausalGraph
    from .adapters import CausalLearnAdapter
except ImportError:
    from graph import CausalGraph
    from adapters import CausalLearnAdapter

logger = logging.getLogger(__name__)


class PCAlgorithmLegacy:
    """
    Legacy PC Algorithm Implementation (for fallback)

    Learns causal structure from observational data using
    conditional independence tests.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.sep_sets: Dict[Tuple[str, str], Set[str]] = {}

    def fit(self, data: pd.DataFrame) -> CausalGraph:
        """
        Learn causal structure from data

        Args:
            data: DataFrame with variables as columns

        Returns:
            CausalGraph with learned structure
        """
        from itertools import combinations

        variables = list(data.columns)
        graph = CausalGraph()

        # Initialize complete undirected graph
        for var in variables:
            graph.add_node(var)

        adjacency = {v: set(variables) - {v} for v in variables}

        # Phase 1: Remove edges based on conditional independence
        depth = 0
        while True:
            removed_any = False

            for x in variables:
                neighbors_x = list(adjacency[x])

                for y in neighbors_x:
                    if y not in adjacency[x]:
                        continue

                    # Get possible conditioning sets
                    other_neighbors = [n for n in adjacency[x] if n != y]

                    if len(other_neighbors) >= depth:
                        for cond_set in combinations(other_neighbors, depth):
                            cond_set = set(cond_set)

                            # Test conditional independence
                            if self._conditional_independence_test(
                                data, x, y, cond_set
                            ):
                                # Remove edge
                                adjacency[x].discard(y)
                                adjacency[y].discard(x)
                                self.sep_sets[(x, y)] = cond_set
                                self.sep_sets[(y, x)] = cond_set
                                removed_any = True
                                break

            if not removed_any:
                break
            depth += 1

        # Phase 2: Orient edges (simplified v-structure detection)
        edges_to_orient = []

        for y in variables:
            neighbors = list(adjacency[y])
            for i, x in enumerate(neighbors):
                for z in neighbors[i+1:]:
                    # Check if x and z are non-adjacent
                    if z not in adjacency[x]:
                        # Check if y is in separating set
                        sep = self.sep_sets.get((x, z), set())
                        if y not in sep:
                            # Orient as v-structure: x -> y <- z
                            edges_to_orient.append((x, y, '->'))
                            edges_to_orient.append((z, y, '->'))

        # Add edges to graph
        for x in variables:
            for y in adjacency[x]:
                # Check if oriented
                oriented = False
                for src, tgt, etype in edges_to_orient:
                    if src == x and tgt == y:
                        graph.add_edge(x, y, '->')
                        oriented = True
                        break

                if not oriented and (y, x, '->') not in [(e[0], e[1], e[2]) for e in edges_to_orient]:
                    # Undirected edge
                    if x < y:  # Avoid duplicates
                        graph.add_edge(x, y, '--')

        return graph

    def _conditional_independence_test(self, data: pd.DataFrame,
                                        x: str, y: str,
                                        cond_set: Set[str]) -> bool:
        """Test conditional independence using partial correlation"""
        if len(cond_set) == 0:
            # Simple correlation test
            corr = data[x].corr(data[y])
            n = len(data)
            # Fisher's z-transform
            z = 0.5 * np.log((1 + corr) / (1 - corr + 1e-10))
            se = 1.0 / np.sqrt(n - 3)
            p_value = 2 * (1 - self._norm_cdf(abs(z) / se))

        else:
            # Partial correlation
            try:
                # Residualize X and Y on conditioning set
                cond_vars = list(cond_set)
                X_cond = data[cond_vars].values
                X_cond = np.column_stack([np.ones(len(data)), X_cond])

                # Residuals of x
                beta_x = np.linalg.lstsq(X_cond, data[x].values, rcond=None)[0]
                resid_x = data[x].values - X_cond @ beta_x

                # Residuals of y
                beta_y = np.linalg.lstsq(X_cond, data[y].values, rcond=None)[0]
                resid_y = data[y].values - X_cond @ beta_y

                # Correlation of residuals
                corr = np.corrcoef(resid_x, resid_y)[0, 1]
                n = len(data)
                k = len(cond_set)

                # Fisher's z-transform
                z = 0.5 * np.log((1 + corr) / (1 - corr + 1e-10))
                se = 1.0 / np.sqrt(n - k - 3)
                p_value = 2 * (1 - self._norm_cdf(abs(z) / se))

            except Exception:
                return False

        return p_value > self.alpha

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF"""
        from math import erf, sqrt
        return 0.5 * (1 + erf(x / sqrt(2)))


class CausalLearnDiscovery:
    """
    Wrapper for causal-learn discovery algorithms

    Provides a unified interface for:
    - PC (constraint-based, assumes no latent confounders)
    - FCI (constraint-based, handles latent confounders)
    - GES (score-based, greedy search)
    """

    def __init__(self, alpha: float = 0.05, indep_test: str = 'fisherz'):
        """
        Initialize discovery engine

        Args:
            alpha: Significance level for independence tests
            indep_test: Test type ('fisherz', 'chisq', 'gsq', 'kci')
        """
        self.alpha = alpha
        self.indep_test = indep_test
        self.adapter = CausalLearnAdapter()
        self.last_result = None
        self._legacy_pc = PCAlgorithmLegacy(alpha=alpha)

    def run_pc(
        self,
        data: pd.DataFrame,
        stable: bool = True,
        uc_rule: int = 0,
        background_knowledge: Optional[Any] = None,
        use_legacy: bool = False
    ) -> CausalGraph:
        """
        Run PC algorithm for causal discovery

        Args:
            data: DataFrame with variables as columns
            stable: Use stable PC variant (recommended)
            uc_rule: Unshielded collider rule (0, 1, or 2)
            background_knowledge: Prior constraints on graph structure
            use_legacy: Use legacy (custom) implementation instead of causal-learn

        Returns:
            CausalGraph with discovered structure
        """
        if use_legacy:
            logger.info("Using legacy PC implementation")
            return self._legacy_pc.fit(data)

        try:
            from causallearn.search.ConstraintBased.PC import pc

            data_array = data.values
            variable_names = list(data.columns)

            logger.info(f"Running PC algorithm on {len(variable_names)} variables...")

            cg = pc(
                data=data_array,
                alpha=self.alpha,
                indep_test=self.indep_test,
                stable=stable,
                uc_rule=uc_rule,
                background_knowledge=background_knowledge,
                verbose=False,
                show_progress=False
            )

            self.last_result = cg
            graph = self.adapter.to_causal_graph(cg, variable_names)

            logger.info(f"Discovered {len(graph.edges)} edges")
            return graph

        except ImportError:
            logger.warning("causal-learn not available, using legacy implementation")
            return self._legacy_pc.fit(data)

        except Exception as e:
            logger.error(f"PC algorithm failed: {e}, falling back to legacy")
            return self._legacy_pc.fit(data)

    def run_fci(
        self,
        data: pd.DataFrame,
        background_knowledge: Optional[Any] = None
    ) -> CausalGraph:
        """
        Run FCI algorithm for causal discovery with latent confounders

        FCI (Fast Causal Inference) can detect the presence of
        latent confounders and produces PAGs (Partial Ancestral Graphs).

        Args:
            data: DataFrame with variables as columns
            background_knowledge: Prior constraints

        Returns:
            CausalGraph (PAG representation)
        """
        try:
            from causallearn.search.ConstraintBased.FCI import fci

            data_array = data.values
            variable_names = list(data.columns)

            logger.info(f"Running FCI algorithm on {len(variable_names)} variables...")

            g, edges = fci(
                data_array,
                self.indep_test,
                self.alpha,
                background_knowledge=background_knowledge,
                verbose=False
            )

            self.last_result = g
            graph = self.adapter.to_causal_graph_from_fci(g, variable_names)

            logger.info(f"Discovered {len(graph.edges)} edges")
            return graph

        except ImportError:
            logger.error("causal-learn required for FCI algorithm")
            raise ImportError("Please install causal-learn: pip install causal-learn")

        except Exception as e:
            logger.error(f"FCI algorithm failed: {e}")
            raise

    def run_ges(
        self,
        data: pd.DataFrame,
        score_func: str = 'local_score_BIC'
    ) -> CausalGraph:
        """
        Run GES (Greedy Equivalence Search) algorithm

        GES is a score-based method that searches the space of
        equivalence classes of DAGs.

        Args:
            data: DataFrame with variables as columns
            score_func: Scoring function ('local_score_BIC', 'local_score_BDeu')

        Returns:
            CausalGraph with discovered structure
        """
        try:
            from causallearn.search.ScoreBased.GES import ges

            data_array = data.values
            variable_names = list(data.columns)

            logger.info(f"Running GES algorithm on {len(variable_names)} variables...")

            record = ges(data_array, score_func=score_func)

            self.last_result = record
            graph = self.adapter.to_causal_graph_from_ges(record, variable_names)

            logger.info(f"Discovered {len(graph.edges)} edges")
            return graph

        except ImportError:
            logger.error("causal-learn required for GES algorithm")
            raise ImportError("Please install causal-learn: pip install causal-learn")

        except Exception as e:
            logger.error(f"GES algorithm failed: {e}")
            raise

    def get_separating_sets(self) -> Dict[Tuple[str, str], Set[str]]:
        """
        Get separating sets from last PC/FCI run

        Returns:
            Dictionary mapping variable pairs to separating sets
        """
        if self.last_result:
            return self.adapter.get_separating_sets(self.last_result)
        return {}

    def get_pvalues(self) -> Dict[Tuple[str, str], float]:
        """
        Get p-values from last independence tests

        Returns:
            Dictionary mapping variable pairs to p-values
        """
        if self.last_result and hasattr(self.last_result, 'p_values'):
            return self.last_result.p_values
        return {}

    def compare_algorithms(
        self,
        data: pd.DataFrame,
        algorithms: List[str] = None
    ) -> Dict[str, CausalGraph]:
        """
        Run multiple algorithms and return results for comparison

        Args:
            data: DataFrame with variables
            algorithms: List of algorithms ('pc', 'fci', 'ges')

        Returns:
            Dictionary mapping algorithm name to discovered graph
        """
        if algorithms is None:
            algorithms = ['pc', 'ges']

        results = {}

        for algo in algorithms:
            try:
                if algo == 'pc':
                    results['pc'] = self.run_pc(data)
                elif algo == 'fci':
                    results['fci'] = self.run_fci(data)
                elif algo == 'ges':
                    results['ges'] = self.run_ges(data)
            except Exception as e:
                logger.warning(f"Algorithm {algo} failed: {e}")

        return results
