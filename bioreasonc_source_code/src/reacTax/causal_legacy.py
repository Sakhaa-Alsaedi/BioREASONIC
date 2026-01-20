"""
Causal-Aware Reasoning Module (C)

Implements causal discovery and inference:
- PC Algorithm for causal structure learning
- Mendelian Randomization (MR) via EpiGraphDB
- Causal DAG construction and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import requests
import logging
from itertools import combinations

from ..schema import CausalRelation, RiskGene, BenchmarkItem, CausalLabel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    """Represents a causal edge"""
    source: str
    target: str
    edge_type: str  # '->', '<-', '--', 'o->', etc.
    weight: float = 1.0
    p_value: Optional[float] = None


@dataclass
class CausalGraph:
    """Represents a causal DAG"""
    nodes: Set[str]
    edges: List[CausalEdge]
    adjacency: Dict[str, List[str]]

    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.adjacency = {}

    def add_node(self, node: str):
        self.nodes.add(node)
        if node not in self.adjacency:
            self.adjacency[node] = []

    def add_edge(self, source: str, target: str, edge_type: str = '->',
                 weight: float = 1.0, p_value: float = None):
        self.add_node(source)
        self.add_node(target)
        edge = CausalEdge(source, target, edge_type, weight, p_value)
        self.edges.append(edge)
        self.adjacency[source].append(target)

    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes (causes)"""
        parents = []
        for edge in self.edges:
            if edge.target == node and edge.edge_type == '->':
                parents.append(edge.source)
        return parents

    def get_children(self, node: str) -> List[str]:
        """Get child nodes (effects)"""
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


class PCAlgorithm:
    """
    PC Algorithm for Causal Discovery

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
        variables = list(data.columns)
        n = len(variables)
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
        """
        Test conditional independence using partial correlation

        H0: X _|_ Y | Z (independent)
        """
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
                from scipy import stats

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


class MendelianRandomization:
    """
    Mendelian Randomization for Causal Inference

    Uses genetic variants as instrumental variables to estimate
    causal effects.
    """

    def __init__(self):
        self.instruments: Dict[str, List[Dict]] = {}
        self.results: Dict[str, Dict] = {}

    def ivw_estimate(self, beta_exposure: np.ndarray,
                     beta_outcome: np.ndarray,
                     se_exposure: np.ndarray,
                     se_outcome: np.ndarray) -> Dict:
        """
        Inverse Variance Weighted (IVW) MR estimate

        Args:
            beta_exposure: Effect sizes on exposure
            beta_outcome: Effect sizes on outcome
            se_exposure: Standard errors for exposure
            se_outcome: Standard errors for outcome

        Returns:
            Dictionary with causal estimate and statistics
        """
        # Wald ratios
        ratio = beta_outcome / beta_exposure

        # Weights (inverse variance)
        weights = 1.0 / (se_outcome**2 / beta_exposure**2)

        # IVW estimate
        causal_estimate = np.sum(weights * ratio) / np.sum(weights)

        # Standard error
        se = np.sqrt(1.0 / np.sum(weights))

        # P-value
        z_stat = causal_estimate / se
        p_value = 2 * (1 - self._norm_cdf(abs(z_stat)))

        return {
            "method": "ivw",
            "estimate": causal_estimate,
            "se": se,
            "p_value": p_value,
            "n_instruments": len(beta_exposure)
        }

    def egger_estimate(self, beta_exposure: np.ndarray,
                       beta_outcome: np.ndarray,
                       se_outcome: np.ndarray) -> Dict:
        """
        MR-Egger regression estimate

        Allows for directional pleiotropy via intercept term.
        """
        # Weighted regression with intercept
        weights = 1.0 / se_outcome**2

        # Design matrix [1, beta_exposure]
        X = np.column_stack([np.ones_like(beta_exposure), beta_exposure])
        W = np.diag(weights)

        # Weighted least squares
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ beta_outcome

        try:
            beta = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        intercept = beta[0]
        causal_estimate = beta[1]

        # Residuals and SE
        residuals = beta_outcome - X @ beta
        mse = np.sum(weights * residuals**2) / (len(beta_exposure) - 2)
        var_beta = mse * np.linalg.inv(XtWX)

        se = np.sqrt(var_beta[1, 1])
        se_intercept = np.sqrt(var_beta[0, 0])

        # P-values
        z_stat = causal_estimate / se
        p_value = 2 * (1 - self._norm_cdf(abs(z_stat)))

        z_intercept = intercept / se_intercept
        p_intercept = 2 * (1 - self._norm_cdf(abs(z_intercept)))

        return {
            "method": "egger",
            "estimate": causal_estimate,
            "se": se,
            "p_value": p_value,
            "intercept": intercept,
            "intercept_se": se_intercept,
            "intercept_p": p_intercept,
            "n_instruments": len(beta_exposure)
        }

    def weighted_median_estimate(self, beta_exposure: np.ndarray,
                                  beta_outcome: np.ndarray,
                                  se_exposure: np.ndarray,
                                  se_outcome: np.ndarray) -> Dict:
        """
        Weighted median MR estimate

        Robust to up to 50% invalid instruments.
        """
        # Wald ratios
        ratio = beta_outcome / beta_exposure

        # Weights
        se_ratio = np.sqrt(se_outcome**2 / beta_exposure**2 +
                          beta_outcome**2 * se_exposure**2 / beta_exposure**4)
        weights = 1.0 / se_ratio**2

        # Sort by ratio
        sorted_idx = np.argsort(ratio)
        sorted_ratio = ratio[sorted_idx]
        sorted_weights = weights[sorted_idx]

        # Cumulative weights
        cum_weights = np.cumsum(sorted_weights)
        cum_weights = cum_weights / cum_weights[-1]

        # Find median
        median_idx = np.searchsorted(cum_weights, 0.5)
        causal_estimate = sorted_ratio[median_idx]

        # Bootstrap SE (simplified)
        se = np.std(ratio) / np.sqrt(len(ratio))

        z_stat = causal_estimate / se
        p_value = 2 * (1 - self._norm_cdf(abs(z_stat)))

        return {
            "method": "weighted_median",
            "estimate": causal_estimate,
            "se": se,
            "p_value": p_value,
            "n_instruments": len(beta_exposure)
        }

    @staticmethod
    def _norm_cdf(x: float) -> float:
        from math import erf, sqrt
        return 0.5 * (1 + erf(x / sqrt(2)))


class EpiGraphDBClient:
    """Client for EpiGraphDB API for MR results"""

    BASE_URL = "https://api.epigraphdb.org"

    def get_mr_results(self, exposure: str, outcome: str) -> List[Dict]:
        """
        Get MR results for exposure-outcome pair

        Args:
            exposure: Exposure trait name
            outcome: Outcome trait name

        Returns:
            List of MR results
        """
        try:
            url = f"{self.BASE_URL}/mr"
            params = {
                "exposure_trait": exposure,
                "outcome_trait": outcome
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return data.get("results", [])

        except requests.RequestException as e:
            logger.warning(f"EpiGraphDB request failed: {e}")
            return []

    def search_traits(self, query: str) -> List[Dict]:
        """Search for traits in EpiGraphDB"""
        try:
            url = f"{self.BASE_URL}/meta/nodes/Gwas/search"
            params = {"name": query}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return data.get("results", [])

        except requests.RequestException as e:
            logger.warning(f"EpiGraphDB request failed: {e}")
            return []

    def get_gene_to_disease(self, gene: str) -> List[Dict]:
        """Get gene-disease associations"""
        try:
            url = f"{self.BASE_URL}/gene/druggability/ppi"
            params = {"gene_name": gene}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return data.get("results", [])

        except requests.RequestException as e:
            logger.warning(f"EpiGraphDB request failed: {e}")
            return []


class CausalReasoning:
    """Main class for Causal-Aware reasoning"""

    def __init__(self):
        self.pc = PCAlgorithm(alpha=0.05)
        self.mr = MendelianRandomization()
        self.epigraphdb = EpiGraphDBClient()
        self.causal_graph: Optional[CausalGraph] = None
        self.causal_relations: List[CausalRelation] = []

    def run_pc_algorithm(self, data: pd.DataFrame) -> CausalGraph:
        """Run PC algorithm on data"""
        logger.info(f"Running PC algorithm on {data.shape[1]} variables...")
        self.causal_graph = self.pc.fit(data)
        logger.info(f"Discovered {len(self.causal_graph.edges)} causal edges")
        return self.causal_graph

    def load_causal_data(self, filepath: str) -> CausalGraph:
        """Load pre-computed causal discovery results"""
        # Placeholder for loading user-provided PC results
        logger.info(f"Loading causal data from {filepath}")
        self.causal_graph = CausalGraph()
        # Parse file and populate graph
        return self.causal_graph

    def run_mr_analysis(self, exposure_betas: np.ndarray,
                        outcome_betas: np.ndarray,
                        exposure_se: np.ndarray,
                        outcome_se: np.ndarray) -> Dict[str, Dict]:
        """Run all MR methods"""
        results = {}

        results['ivw'] = self.mr.ivw_estimate(
            exposure_betas, outcome_betas, exposure_se, outcome_se
        )

        results['egger'] = self.mr.egger_estimate(
            exposure_betas, outcome_betas, outcome_se
        )

        results['weighted_median'] = self.mr.weighted_median_estimate(
            exposure_betas, outcome_betas, exposure_se, outcome_se
        )

        return results

    def query_epigraphdb_mr(self, exposure: str, outcome: str) -> List[Dict]:
        """Query EpiGraphDB for MR evidence"""
        return self.epigraphdb.get_mr_results(exposure, outcome)

    def generate_causal_questions(self, genes: List[RiskGene],
                                   mr_results: Dict = None) -> List[BenchmarkItem]:
        """Generate causal-aware benchmark questions"""
        questions = []

        # C-CAUSAL-VS-ASSOC questions
        for gene in genes[:20]:
            if gene.odds_ratio and gene.p_value:
                is_significant = gene.p_value < 0.05
                answer = "association" if is_significant else "no significant association"

                q = BenchmarkItem(
                    id=f"C-{len(questions):04d}",
                    taxonomy="C",
                    label=CausalLabel.CAUSAL_VS_ASSOC.value,
                    template_id="C-CAUSAL-VS-ASSOC-01",
                    question=f"Is the relationship between {gene.symbol} and COVID-19 severity causal or merely associative based on the genetic evidence?",
                    answer=f"The evidence suggests {answer}. GWAS shows OR={gene.odds_ratio:.2f}, p={gene.p_value:.2e}. Causal inference requires additional evidence from MR or intervention studies.",
                    explanation="Distinguishing causation from association requires: (1) consistent association, (2) temporal precedence, (3) dose-response, and (4) experimental/MR evidence.",
                    source_genes=[gene.symbol],
                    algorithm_used="causal_reasoning"
                )
                questions.append(q)

        # C-MR-INFERENCE questions
        if mr_results:
            for exposure, results in mr_results.items():
                ivw = results.get('ivw', {})
                if ivw:
                    q = BenchmarkItem(
                        id=f"C-{len(questions):04d}",
                        taxonomy="C",
                        label=CausalLabel.MR_INFERENCE.value,
                        template_id="C-MR-INFERENCE-01",
                        question=f"Based on Mendelian Randomization using IVW method, what is the causal effect estimate of {exposure} on COVID-19 severity?",
                        answer=f"IVW estimate: {ivw.get('estimate', 'N/A'):.3f} (SE={ivw.get('se', 'N/A'):.3f}, p={ivw.get('p_value', 'N/A'):.2e})",
                        explanation="IVW MR uses genetic variants as instruments assuming no pleiotropy. The estimate represents the causal effect per unit increase in exposure.",
                        source_genes=[exposure],
                        algorithm_used="mr_ivw"
                    )
                    questions.append(q)

        # C-PC-DISCOVERY questions (if causal graph available)
        if self.causal_graph and self.causal_graph.edges:
            for edge in self.causal_graph.edges[:10]:
                direction = "causes" if edge.edge_type == '->' else "is associated with"
                q = BenchmarkItem(
                    id=f"C-{len(questions):04d}",
                    taxonomy="C",
                    label=CausalLabel.PC_DISCOVERY.value,
                    template_id="C-PC-DISCOVERY-01",
                    question=f"According to PC algorithm causal discovery, what is the causal relationship between {edge.source} and {edge.target}?",
                    answer=f"{edge.source} {direction} {edge.target} (edge type: {edge.edge_type})",
                    explanation="PC algorithm identifies causal structure by testing conditional independence. Directed edges (→) indicate causal direction, undirected (--) indicate association without determined direction.",
                    source_genes=[edge.source, edge.target],
                    algorithm_used="pc_algorithm"
                )
                questions.append(q)

        return questions


# Factory function
def create_causal_module() -> CausalReasoning:
    """Create Causal-Aware reasoning module"""
    return CausalReasoning()


if __name__ == "__main__":
    # Test PC algorithm
    np.random.seed(42)

    # Generate synthetic data with known structure: A -> B -> C
    n = 500
    A = np.random.randn(n)
    B = 0.7 * A + np.random.randn(n) * 0.5
    C = 0.6 * B + np.random.randn(n) * 0.5

    data = pd.DataFrame({'A': A, 'B': B, 'C': C})

    module = create_causal_module()
    graph = module.run_pc_algorithm(data)

    print("Discovered edges:")
    for edge in graph.edges:
        print(f"  {edge.source} {edge.edge_type} {edge.target}")

    # Test MR
    beta_exp = np.array([0.1, 0.15, 0.12, 0.08])
    beta_out = np.array([0.05, 0.07, 0.06, 0.04])
    se_exp = np.array([0.02, 0.02, 0.02, 0.02])
    se_out = np.array([0.01, 0.01, 0.01, 0.01])

    mr_results = module.run_mr_analysis(beta_exp, beta_out, se_exp, se_out)
    print("\nMR Results:")
    for method, result in mr_results.items():
        print(f"  {method}: estimate={result['estimate']:.3f}, p={result['p_value']:.3f}")
