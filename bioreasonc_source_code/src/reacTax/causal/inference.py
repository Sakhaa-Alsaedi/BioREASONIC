"""
DoWhy Causal Inference Module

Implements the DoWhy four-step workflow:
1. Model: Define causal graph
2. Identify: Find estimand (causal effect expression)
3. Estimate: Calculate effect using various estimators
4. Refute: Validate robustness of estimates
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EstimatorMethod(str, Enum):
    """Available DoWhy estimation methods"""
    BACKDOOR_LINEAR = "backdoor.linear_regression"
    BACKDOOR_PROPENSITY = "backdoor.propensity_score_matching"
    BACKDOOR_PROPENSITY_STRATIFICATION = "backdoor.propensity_score_stratification"
    BACKDOOR_PROPENSITY_WEIGHTING = "backdoor.propensity_score_weighting"
    IV_REGRESSION = "iv.instrumental_variable"
    IV_TWO_STAGE = "iv.regression_discontinuity"
    FRONTDOOR = "frontdoor.two_stage_regression"


class RefutationMethod(str, Enum):
    """Available DoWhy refutation methods"""
    RANDOM_COMMON_CAUSE = "random_common_cause"
    PLACEBO_TREATMENT = "placebo_treatment_refuter"
    DATA_SUBSET = "data_subset_refuter"
    BOOTSTRAP = "bootstrap_refuter"
    DUMMY_OUTCOME = "dummy_outcome_refuter"


@dataclass
class EffectEstimate:
    """Result from DoWhy effect estimation"""
    method: str
    estimate: float
    std_error: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    estimand: Optional[str] = None
    estimand_type: Optional[str] = None
    treatment: Optional[str] = None
    outcome: Optional[str] = None
    raw_result: Optional[Any] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "method": self.method,
            "estimate": self.estimate,
            "std_error": self.std_error,
            "p_value": self.p_value,
            "confidence_interval": self.confidence_interval,
            "treatment": self.treatment,
            "outcome": self.outcome
        }


@dataclass
class RefutationResult:
    """Result from DoWhy refutation test"""
    method: str
    refutation_passed: bool
    original_estimate: float
    new_estimate: float
    p_value: Optional[float] = None
    effect_diff: Optional[float] = None
    relative_diff: Optional[float] = None
    interpretation: str = ""
    raw_result: Optional[Any] = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "method": self.method,
            "passed": self.refutation_passed,
            "original_estimate": self.original_estimate,
            "new_estimate": self.new_estimate,
            "p_value": self.p_value,
            "effect_diff": self.effect_diff,
            "interpretation": self.interpretation
        }


class DoWhyInference:
    """
    DoWhy-based causal inference engine

    Implements the Model → Identify → Estimate → Refute workflow.
    """

    def __init__(self):
        self.model: Optional[Any] = None  # CausalModel
        self.identified_estimand: Optional[Any] = None
        self.estimate: Optional[Any] = None
        self.effect_estimate: Optional[EffectEstimate] = None
        self.refutations: List[RefutationResult] = []

    def set_model(self, model: 'CausalModel'):
        """
        Set the causal model

        Args:
            model: DoWhy CausalModel object
        """
        self.model = model
        self.identified_estimand = None
        self.estimate = None
        self.effect_estimate = None
        self.refutations = []

    def identify(self) -> str:
        """
        Identify causal estimand

        Finds a valid adjustment formula for the causal effect
        based on the graph structure.

        Returns:
            String representation of identified estimand
        """
        if not self.model:
            raise ValueError("Model not set. Call set_model() first.")

        self.identified_estimand = self.model.identify_effect(
            proceed_when_unidentifiable=True
        )

        return str(self.identified_estimand)

    def estimate_effect(
        self,
        method: EstimatorMethod = EstimatorMethod.BACKDOOR_LINEAR,
        **kwargs
    ) -> EffectEstimate:
        """
        Estimate causal effect

        Args:
            method: Estimation method to use
            **kwargs: Additional arguments for estimator

        Returns:
            EffectEstimate with results
        """
        if not self.model:
            raise ValueError("Model not set. Call set_model() first.")

        if not self.identified_estimand:
            self.identify()

        try:
            estimate_result = self.model.estimate_effect(
                self.identified_estimand,
                method_name=method.value,
                **kwargs
            )

            self.estimate = estimate_result

            # Extract statistics
            std_error = getattr(estimate_result, 'std_error', None)
            p_value = getattr(estimate_result, 'p_value', None)
            ci = getattr(estimate_result, 'confidence_intervals', None)

            self.effect_estimate = EffectEstimate(
                method=method.value,
                estimate=float(estimate_result.value),
                std_error=float(std_error) if std_error is not None else None,
                p_value=float(p_value) if p_value is not None else None,
                confidence_interval=ci,
                estimand=str(self.identified_estimand),
                estimand_type=getattr(self.identified_estimand, 'estimand_type', None),
                treatment=str(self.model._treatment) if hasattr(self.model, '_treatment') else None,
                outcome=str(self.model._outcome) if hasattr(self.model, '_outcome') else None,
                raw_result=estimate_result
            )

            return self.effect_estimate

        except Exception as e:
            logger.error(f"Effect estimation failed: {e}")
            raise

    def refute(
        self,
        method: RefutationMethod = RefutationMethod.RANDOM_COMMON_CAUSE,
        num_simulations: int = 100,
        **kwargs
    ) -> RefutationResult:
        """
        Refute the estimate with a robustness test

        Args:
            method: Refutation method to use
            num_simulations: Number of simulations
            **kwargs: Additional arguments

        Returns:
            RefutationResult with test outcome
        """
        if not self.estimate:
            raise ValueError("No estimate to refute. Call estimate_effect() first.")

        try:
            refutation = self.model.refute_estimate(
                self.identified_estimand,
                self.estimate,
                method_name=method.value,
                num_simulations=num_simulations,
                **kwargs
            )

            # Extract results
            original = float(self.estimate.value)
            new_estimate = float(refutation.new_effect) if hasattr(refutation, 'new_effect') else original

            # Calculate differences
            effect_diff = abs(new_estimate - original)
            relative_diff = effect_diff / (abs(original) + 1e-10)

            # Determine if refutation passed (estimate is robust)
            # Different criteria for different refutation methods
            if method == RefutationMethod.PLACEBO_TREATMENT:
                # Placebo should give near-zero effect
                passed = abs(new_estimate) < 0.1 * abs(original) + 0.01
            elif method == RefutationMethod.RANDOM_COMMON_CAUSE:
                # Adding random confounder shouldn't change estimate much
                passed = relative_diff < 0.15
            elif method == RefutationMethod.DATA_SUBSET:
                # Estimate should be stable across subsets
                passed = relative_diff < 0.20
            else:
                passed = relative_diff < 0.15

            # Get p-value if available
            p_value = None
            if hasattr(refutation, 'refutation_result'):
                p_value = refutation.refutation_result.get('p_value')

            interpretation = self._interpret_refutation(method, passed, relative_diff, new_estimate)

            result = RefutationResult(
                method=method.value,
                refutation_passed=passed,
                original_estimate=original,
                new_estimate=new_estimate,
                p_value=float(p_value) if p_value else None,
                effect_diff=effect_diff,
                relative_diff=relative_diff,
                interpretation=interpretation,
                raw_result=refutation
            )

            self.refutations.append(result)
            return result

        except Exception as e:
            logger.error(f"Refutation {method.value} failed: {e}")
            raise

    def run_all_refutations(
        self,
        methods: List[RefutationMethod] = None,
        num_simulations: int = 100
    ) -> List[RefutationResult]:
        """
        Run multiple refutation tests

        Args:
            methods: List of refutation methods (default: standard set)
            num_simulations: Number of simulations per test

        Returns:
            List of RefutationResult objects
        """
        if methods is None:
            methods = [
                RefutationMethod.RANDOM_COMMON_CAUSE,
                RefutationMethod.PLACEBO_TREATMENT,
                RefutationMethod.DATA_SUBSET
            ]

        results = []
        for method in methods:
            try:
                result = self.refute(method, num_simulations)
                results.append(result)
                logger.info(f"Refutation {method.value}: {'PASSED' if result.refutation_passed else 'FAILED'}")
            except Exception as e:
                logger.warning(f"Refutation {method.value} failed: {e}")

        return results

    def sensitivity_analysis(
        self,
        confounder_effect_on_treatment: float = 0.1,
        confounder_effect_on_outcome: float = 0.1,
        num_simulations: int = 100
    ) -> Dict:
        """
        Perform sensitivity analysis for unobserved confounding

        Estimates how sensitive the causal effect is to
        potential unobserved confounders.

        Args:
            confounder_effect_on_treatment: Effect size of confounder on treatment
            confounder_effect_on_outcome: Effect size of confounder on outcome
            num_simulations: Number of simulations

        Returns:
            Dictionary with sensitivity analysis results
        """
        if not self.estimate:
            raise ValueError("No estimate for sensitivity analysis")

        try:
            # Use random common cause refutation with specific effect sizes
            refutation = self.model.refute_estimate(
                self.identified_estimand,
                self.estimate,
                method_name="add_unobserved_common_cause",
                confounders_effect_on_treatment=confounder_effect_on_treatment,
                confounders_effect_on_outcome=confounder_effect_on_outcome,
                num_simulations=num_simulations
            )

            original = float(self.estimate.value)
            new_estimate = float(refutation.new_effect)

            return {
                "original_estimate": original,
                "adjusted_estimate": new_estimate,
                "confounder_effect_treatment": confounder_effect_on_treatment,
                "confounder_effect_outcome": confounder_effect_on_outcome,
                "estimate_change": new_estimate - original,
                "relative_change": (new_estimate - original) / (abs(original) + 1e-10),
                "nullifying_confounder_effect": self._estimate_nullifying_effect(original)
            }

        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            return {
                "error": str(e),
                "original_estimate": float(self.estimate.value) if self.estimate else None
            }

    def _estimate_nullifying_effect(self, original_estimate: float) -> float:
        """
        Estimate confounder effect needed to nullify the estimate

        This is a rough approximation based on the effect size.
        """
        # Simple heuristic: effect needed is roughly proportional to estimate
        return abs(original_estimate) * 2

    def _interpret_refutation(
        self,
        method: RefutationMethod,
        passed: bool,
        relative_diff: float,
        new_estimate: float
    ) -> str:
        """Generate human-readable interpretation of refutation result"""

        if method == RefutationMethod.RANDOM_COMMON_CAUSE:
            if passed:
                return (
                    f"Adding a random common cause changed the estimate by {relative_diff:.1%}. "
                    "This suggests the estimate is robust to unobserved confounding of similar magnitude."
                )
            else:
                return (
                    f"Adding a random common cause changed the estimate by {relative_diff:.1%}. "
                    "This suggests potential sensitivity to unobserved confounding."
                )

        elif method == RefutationMethod.PLACEBO_TREATMENT:
            if passed:
                return (
                    f"Replacing treatment with placebo yielded effect {new_estimate:.3f}. "
                    "Near-zero placebo effect confirms the treatment-outcome relationship."
                )
            else:
                return (
                    f"Replacing treatment with placebo yielded effect {new_estimate:.3f}. "
                    "Non-zero placebo effect raises concerns about the causal relationship."
                )

        elif method == RefutationMethod.DATA_SUBSET:
            if passed:
                return (
                    f"Estimate changed by {relative_diff:.1%} across data subsets. "
                    "This indicates stability and robustness to sampling variation."
                )
            else:
                return (
                    f"Estimate changed by {relative_diff:.1%} across data subsets. "
                    "High variability suggests potential instability or heterogeneous effects."
                )

        elif method == RefutationMethod.BOOTSTRAP:
            if passed:
                return (
                    f"Bootstrap analysis shows {relative_diff:.1%} variation. "
                    "The estimate appears statistically reliable."
                )
            else:
                return (
                    f"Bootstrap analysis shows {relative_diff:.1%} variation. "
                    "High bootstrap variation suggests statistical uncertainty."
                )

        return f"Refutation {'passed' if passed else 'failed'} with {relative_diff:.1%} change."

    def get_summary(self) -> Dict:
        """
        Get summary of all inference results

        Returns:
            Dictionary with estimate and refutation summaries
        """
        summary = {
            "estimate": self.effect_estimate.to_dict() if self.effect_estimate else None,
            "refutations": [r.to_dict() for r in self.refutations],
            "all_refutations_passed": all(r.refutation_passed for r in self.refutations) if self.refutations else None,
            "num_refutations_passed": sum(1 for r in self.refutations if r.refutation_passed),
            "num_refutations_total": len(self.refutations)
        }
        return summary
