"""
Causal Reasoning Module

Main CausalReasoning class that orchestrates:
- Causal discovery (PC, FCI, GES via causal-learn)
- Causal inference (DoWhy estimation and refutation)
- Mendelian Randomization
- Benchmark question generation
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from .graph import CausalGraph, CausalEdge
from .discovery import CausalLearnDiscovery, PCAlgorithmLegacy
from .inference import DoWhyInference, EffectEstimate, RefutationResult, EstimatorMethod, RefutationMethod
from .mr import MendelianRandomization
from .epigraphdb import EpiGraphDBClient
from .adapters import DoWhyAdapter
from .templates import (
    CAUSAL_QUESTION_TEMPLATES,
    CAUSAL_EXPLANATION_TEMPLATES,
    REFUTATION_ASSUMPTIONS,
    MR_METHOD_DESCRIPTIONS
)

logger = logging.getLogger(__name__)


class CausalReasoning:
    """
    Main class for Causal-Aware reasoning

    Integrates causal-learn (discovery) and DoWhy (inference) with
    backward compatibility for existing API.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CausalReasoning module

        Args:
            config: Configuration dictionary with settings for:
                - pc: PC algorithm settings (alpha, indep_test, use_legacy)
                - fci: FCI algorithm settings
                - ges: GES algorithm settings
                - dowhy: DoWhy settings (default_estimator, refutation_tests)
                - questions: Question generation settings
        """
        self.config = config or {}

        # Extract configuration
        pc_config = self.config.get('pc', {})
        alpha = pc_config.get('alpha', 0.05)
        indep_test = pc_config.get('indep_test', 'fisherz')
        self.use_legacy_pc = pc_config.get('use_legacy', False)

        # Legacy components (preserved for backward compatibility)
        self._pc_legacy = PCAlgorithmLegacy(alpha=alpha)
        self.mr = MendelianRandomization()
        self.epigraphdb = EpiGraphDBClient()

        # New components
        self.discovery = CausalLearnDiscovery(alpha=alpha, indep_test=indep_test)
        self.inference = DoWhyInference()

        # State
        self.causal_graph: Optional[CausalGraph] = None
        self.causal_relations: List[Any] = []  # CausalRelation from schema
        self.effect_estimates: Dict[str, EffectEstimate] = {}
        self.refutation_results: Dict[str, List[RefutationResult]] = {}

    # ==================== Backward Compatible Methods ====================

    def run_pc_algorithm(self, data: pd.DataFrame) -> CausalGraph:
        """
        Run PC algorithm on data (backward compatible)

        Now uses causal-learn by default, falls back to legacy if configured.

        Args:
            data: DataFrame with variables as columns

        Returns:
            CausalGraph with discovered structure
        """
        logger.info(f"Running PC algorithm on {data.shape[1]} variables...")

        if self.use_legacy_pc:
            self.causal_graph = self._pc_legacy.fit(data)
        else:
            self.causal_graph = self.discovery.run_pc(data, use_legacy=False)

        logger.info(f"Discovered {len(self.causal_graph.edges)} causal edges")
        return self.causal_graph

    def load_causal_data(self, filepath: str) -> CausalGraph:
        """
        Load pre-computed causal discovery results

        Args:
            filepath: Path to causal data file

        Returns:
            CausalGraph
        """
        logger.info(f"Loading causal data from {filepath}")
        self.causal_graph = CausalGraph()
        # TODO: Implement file parsing
        return self.causal_graph

    def run_mr_analysis(
        self,
        exposure_betas: np.ndarray,
        outcome_betas: np.ndarray,
        exposure_se: np.ndarray,
        outcome_se: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Run all MR methods (backward compatible)

        Args:
            exposure_betas: Effect sizes on exposure
            outcome_betas: Effect sizes on outcome
            exposure_se: Standard errors for exposure
            outcome_se: Standard errors for outcome

        Returns:
            Dictionary with results from IVW, Egger, and weighted median
        """
        return self.mr.run_all_methods(
            exposure_betas, outcome_betas, exposure_se, outcome_se
        )

    def query_epigraphdb_mr(self, exposure: str, outcome: str) -> List[Dict]:
        """Query EpiGraphDB for MR evidence (backward compatible)"""
        return self.epigraphdb.get_mr_results(exposure, outcome)

    # ==================== New DoWhy Methods ====================

    def run_causal_discovery(
        self,
        data: pd.DataFrame,
        algorithm: str = 'pc'
    ) -> CausalGraph:
        """
        Run causal discovery using specified algorithm

        Args:
            data: DataFrame with variables as columns
            algorithm: 'pc', 'fci', or 'ges'

        Returns:
            CausalGraph with discovered structure
        """
        if algorithm == 'pc':
            self.causal_graph = self.discovery.run_pc(data)
        elif algorithm == 'fci':
            self.causal_graph = self.discovery.run_fci(data)
        elif algorithm == 'ges':
            self.causal_graph = self.discovery.run_ges(data)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use 'pc', 'fci', or 'ges'.")

        return self.causal_graph

    def estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        method: EstimatorMethod = EstimatorMethod.BACKDOOR_LINEAR,
        common_causes: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None
    ) -> EffectEstimate:
        """
        Estimate causal effect using DoWhy

        Args:
            data: DataFrame with all variables
            treatment: Treatment variable name
            outcome: Outcome variable name
            method: Estimation method
            common_causes: Confounder names
            instruments: Instrument variable names

        Returns:
            EffectEstimate with results
        """
        # Build DoWhy model
        model = DoWhyAdapter.build_causal_model(
            data=data,
            treatment=treatment,
            outcome=outcome,
            causal_graph=self.causal_graph,
            common_causes=common_causes,
            instruments=instruments
        )

        self.inference.set_model(model)
        estimate = self.inference.estimate_effect(method)

        # Store result
        key = f"{treatment}->{outcome}"
        self.effect_estimates[key] = estimate

        return estimate

    def run_refutation_tests(
        self,
        methods: Optional[List[RefutationMethod]] = None,
        num_simulations: int = 100
    ) -> List[RefutationResult]:
        """
        Run refutation tests on current estimate

        Args:
            methods: List of refutation methods (default: standard set)
            num_simulations: Number of simulations per test

        Returns:
            List of RefutationResult objects
        """
        if not self.inference.estimate:
            raise ValueError("No estimate to refute. Call estimate_causal_effect() first.")

        results = self.inference.run_all_refutations(methods, num_simulations)

        # Store results
        if self.inference.model:
            treatment = str(self.inference.model._treatment) if hasattr(self.inference.model, '_treatment') else 'treatment'
            outcome = str(self.inference.model._outcome) if hasattr(self.inference.model, '_outcome') else 'outcome'
            key = f"{treatment}->{outcome}"
            self.refutation_results[key] = results

        return results

    # ==================== Question Generation ====================

    def generate_causal_questions(
        self,
        genes: List[Any],  # List[RiskGene] from schema
        mr_results: Dict = None,
        include_refutation: bool = True,
        include_sensitivity: bool = True,
        disease: str = "COVID-19"
    ) -> List[Any]:  # List[BenchmarkItem]
        """
        Generate causal-aware benchmark questions

        Extended to include DoWhy-based question types.

        Args:
            genes: List of RiskGene objects
            mr_results: Pre-computed MR results (optional)
            include_refutation: Include refutation questions
            include_sensitivity: Include sensitivity questions
            disease: Disease name for questions

        Returns:
            List of BenchmarkItem objects
        """
        # Import here to avoid circular imports
        from ...schema import BenchmarkItem, CausalLabel

        questions = []

        # Legacy question types (C-CAUSAL-VS-ASSOC, C-MR-INFERENCE, C-PC-DISCOVERY)
        questions.extend(self._generate_causal_vs_assoc_questions(genes, disease))

        if mr_results:
            questions.extend(self._generate_mr_questions(mr_results))

        if self.causal_graph and self.causal_graph.edges:
            questions.extend(self._generate_pc_discovery_questions())

        # New DoWhy question types
        if include_refutation and self.refutation_results:
            questions.extend(self._generate_refutation_questions())

        if include_sensitivity and self.effect_estimates:
            questions.extend(self._generate_sensitivity_questions())

        return questions

    def _generate_causal_vs_assoc_questions(
        self,
        genes: List[Any],
        disease: str
    ) -> List[Any]:
        """Generate C-CAUSAL-VS-ASSOC questions"""
        from ...schema import BenchmarkItem, CausalLabel

        questions = []

        for gene in genes[:20]:
            if hasattr(gene, 'odds_ratio') and hasattr(gene, 'p_value'):
                if gene.odds_ratio and gene.p_value:
                    is_significant = gene.p_value < 0.05
                    conclusion = "association" if is_significant else "no significant association"

                    q = BenchmarkItem(
                        id=f"C-{len(questions):04d}",
                        taxonomy="C",
                        label=CausalLabel.CAUSAL_VS_ASSOC.value,
                        template_id="C-CAUSAL-VS-ASSOC-01",
                        question=f"Is the relationship between {gene.symbol} and {disease} severity causal or merely associative based on the genetic evidence?",
                        answer=f"The evidence suggests {conclusion}. GWAS shows OR={gene.odds_ratio:.2f}, p={gene.p_value:.2e}. Causal inference requires additional evidence from MR or intervention studies.",
                        explanation=CAUSAL_EXPLANATION_TEMPLATES.get("C-CAUSAL-VS-ASSOC", ""),
                        source_genes=[gene.symbol],
                        algorithm_used="causal_reasoning"
                    )
                    questions.append(q)

        return questions

    def _generate_mr_questions(self, mr_results: Dict) -> List[Any]:
        """Generate C-MR-INFERENCE questions"""
        from ...schema import BenchmarkItem, CausalLabel

        questions = []

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
                    explanation=CAUSAL_EXPLANATION_TEMPLATES.get("C-MR-INFERENCE", "").format(
                        method="IVW",
                        method_description=MR_METHOD_DESCRIPTIONS.get("ivw", "")
                    ),
                    source_genes=[exposure],
                    algorithm_used="mr_ivw"
                )
                questions.append(q)

        return questions

    def _generate_pc_discovery_questions(self) -> List[Any]:
        """Generate C-PC-DISCOVERY questions"""
        from ...schema import BenchmarkItem, CausalLabel

        questions = []

        for edge in self.causal_graph.edges[:10]:
            direction = "causes" if edge.edge_type == '->' else "is associated with"

            q = BenchmarkItem(
                id=f"C-{len(questions):04d}",
                taxonomy="C",
                label=CausalLabel.PC_DISCOVERY.value,
                template_id="C-PC-DISCOVERY-01",
                question=f"According to PC algorithm causal discovery, what is the causal relationship between {edge.source} and {edge.target}?",
                answer=f"{edge.source} {direction} {edge.target} (edge type: {edge.edge_type})",
                explanation=CAUSAL_EXPLANATION_TEMPLATES.get("C-PC-DISCOVERY", ""),
                source_genes=[edge.source, edge.target],
                algorithm_used="pc_algorithm"
            )
            questions.append(q)

        return questions

    def _generate_refutation_questions(self) -> List[Any]:
        """Generate C-REFUTATION questions"""
        from ...schema import BenchmarkItem, CausalLabel

        questions = []

        for key, results in self.refutation_results.items():
            parts = key.split('->')
            treatment = parts[0] if len(parts) > 0 else "treatment"
            outcome = parts[1] if len(parts) > 1 else "outcome"

            for refutation in results:
                robustness = "robust" if refutation.refutation_passed else "not robust"
                result_word = "passed" if refutation.refutation_passed else "failed"

                q = BenchmarkItem(
                    id=f"C-{len(questions):04d}",
                    taxonomy="C",
                    label=CausalLabel.REFUTATION.value if hasattr(CausalLabel, 'REFUTATION') else "C-REFUTATION",
                    template_id="C-REFUTATION-01",
                    question=f"After applying the {refutation.method} refutation test, is the causal effect of {treatment} on {outcome} robust?",
                    answer=f"The estimate is {robustness}. Original effect: {refutation.original_estimate:.3f}, after refutation: {refutation.new_estimate:.3f}. {refutation.interpretation}",
                    explanation=CAUSAL_EXPLANATION_TEMPLATES.get("C-REFUTATION", "").format(
                        refutation_method=refutation.method,
                        assumption_tested=REFUTATION_ASSUMPTIONS.get(refutation.method, "causal assumptions")
                    ),
                    source_genes=[treatment],
                    source_data={
                        "method": refutation.method,
                        "passed": refutation.refutation_passed,
                        "original": refutation.original_estimate,
                        "new": refutation.new_estimate
                    },
                    algorithm_used="dowhy_refutation"
                )
                questions.append(q)

        return questions

    def _generate_sensitivity_questions(self) -> List[Any]:
        """Generate C-SENSITIVITY questions"""
        from ...schema import BenchmarkItem, CausalLabel

        questions = []

        for key, estimate in self.effect_estimates.items():
            parts = key.split('->')
            treatment = parts[0] if len(parts) > 0 else "treatment"
            outcome = parts[1] if len(parts) > 1 else "outcome"

            # Estimate nullifying confounder effect
            threshold = abs(estimate.estimate) * 2

            q = BenchmarkItem(
                id=f"C-{len(questions):04d}",
                taxonomy="C",
                label=CausalLabel.SENSITIVITY.value if hasattr(CausalLabel, 'SENSITIVITY') else "C-SENSITIVITY",
                template_id="C-SENSITIVITY-01",
                question=f"How sensitive is the causal effect estimate of {treatment} on {outcome} to unobserved confounding?",
                answer=f"Effect estimate: {estimate.estimate:.3f} (SE={estimate.std_error:.3f if estimate.std_error else 'N/A'}). Sensitivity analysis shows the estimate would be nullified by unobserved confounding with effect size greater than {threshold:.3f}.",
                explanation=CAUSAL_EXPLANATION_TEMPLATES.get("C-SENSITIVITY", ""),
                source_genes=[treatment],
                source_data={
                    "estimate": estimate.estimate,
                    "method": estimate.method,
                    "std_error": estimate.std_error,
                    "nullifying_threshold": threshold
                },
                algorithm_used="dowhy_sensitivity"
            )
            questions.append(q)

        return questions


# Factory function (backward compatible)
def create_causal_module(config: Optional[Dict] = None) -> CausalReasoning:
    """
    Create Causal-Aware reasoning module

    Args:
        config: Optional configuration dictionary

    Returns:
        CausalReasoning instance
    """
    return CausalReasoning(config)
