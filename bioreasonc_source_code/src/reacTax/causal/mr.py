"""
Mendelian Randomization Module

Implements MR methods for causal inference using genetic variants
as instrumental variables:
- Inverse Variance Weighted (IVW)
- MR-Egger (accounts for pleiotropy)
- Weighted Median (robust to invalid instruments)
"""

import numpy as np
from typing import Dict, List, Optional
from math import erf, sqrt
import logging

logger = logging.getLogger(__name__)


class MendelianRandomization:
    """
    Mendelian Randomization for Causal Inference

    Uses genetic variants as instrumental variables to estimate
    causal effects between exposures and outcomes.
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

        Assumes all instruments are valid (no pleiotropy).

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
            "estimate": float(causal_estimate),
            "se": float(se),
            "p_value": float(p_value),
            "n_instruments": len(beta_exposure)
        }

    def egger_estimate(self, beta_exposure: np.ndarray,
                       beta_outcome: np.ndarray,
                       se_outcome: np.ndarray) -> Dict:
        """
        MR-Egger regression estimate

        Allows for directional pleiotropy via intercept term.
        A non-zero intercept indicates pleiotropy.

        Args:
            beta_exposure: Effect sizes on exposure
            beta_outcome: Effect sizes on outcome
            se_outcome: Standard errors for outcome

        Returns:
            Dictionary with causal estimate, intercept, and statistics
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
            "estimate": float(causal_estimate),
            "se": float(se),
            "p_value": float(p_value),
            "intercept": float(intercept),
            "intercept_se": float(se_intercept),
            "intercept_p": float(p_intercept),
            "n_instruments": len(beta_exposure)
        }

    def weighted_median_estimate(self, beta_exposure: np.ndarray,
                                  beta_outcome: np.ndarray,
                                  se_exposure: np.ndarray,
                                  se_outcome: np.ndarray) -> Dict:
        """
        Weighted median MR estimate

        Robust to up to 50% invalid instruments.

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
        causal_estimate = sorted_ratio[min(median_idx, len(sorted_ratio) - 1)]

        # Bootstrap SE (simplified)
        se = np.std(ratio) / np.sqrt(len(ratio))

        z_stat = causal_estimate / se
        p_value = 2 * (1 - self._norm_cdf(abs(z_stat)))

        return {
            "method": "weighted_median",
            "estimate": float(causal_estimate),
            "se": float(se),
            "p_value": float(p_value),
            "n_instruments": len(beta_exposure)
        }

    def run_all_methods(self, beta_exposure: np.ndarray,
                        beta_outcome: np.ndarray,
                        se_exposure: np.ndarray,
                        se_outcome: np.ndarray) -> Dict[str, Dict]:
        """
        Run all MR methods and return combined results

        Args:
            beta_exposure: Effect sizes on exposure
            beta_outcome: Effect sizes on outcome
            se_exposure: Standard errors for exposure
            se_outcome: Standard errors for outcome

        Returns:
            Dictionary mapping method name to results
        """
        results = {}

        results['ivw'] = self.ivw_estimate(
            beta_exposure, beta_outcome, se_exposure, se_outcome
        )

        results['egger'] = self.egger_estimate(
            beta_exposure, beta_outcome, se_outcome
        )

        results['weighted_median'] = self.weighted_median_estimate(
            beta_exposure, beta_outcome, se_exposure, se_outcome
        )

        return results

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """Standard normal CDF"""
        return 0.5 * (1 + erf(x / sqrt(2)))
