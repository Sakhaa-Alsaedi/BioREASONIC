"""
CARES: Causal-Aware Reasoning Evaluation Score

Evaluates biomedical LLM performance across the four taxonomy categories
(S, C, R, M) with hallucination penalty and calibration error adjustment.

Reference: Chapter 3, Section 3.4.3

Now includes LLM-as-Judge API integration for proper response evaluation:
- OpenAI GPT-4
- Anthropic Claude
- Local LLMs (Ollama, vLLM, HuggingFace)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math
import json
import logging

logger = logging.getLogger(__name__)


class ReasoningCategory(Enum):
    """Four categories of biomedical reasoning"""
    STRUCTURE = "S"      # Structure-aware
    CAUSAL = "C"         # Causal-aware
    RISK = "R"           # Risk-aware
    SEMANTIC = "M"       # Semantic-aware


class ApplicationDomain(Enum):
    """Application domains with different hallucination tolerance"""
    DRUG_INTERACTION = "drug_interaction"      # HR_max = 5%
    CLINICAL_DECISION = "clinical_decision"    # HR_max = 10%
    LITERATURE_SUMMARY = "literature_summary"  # HR_max = 15%
    RESEARCH_EXPLORATION = "research_exploration"  # HR_max = 25%
    DEFAULT = "default"                        # HR_max = 23%


# Domain-calibrated alpha values (Table 3.8)
# α = -ln(0.5) / HR_max = 0.693 / HR_max
#
# ADJUSTED: Original values at extremes were too harsh/lenient
# - Drug interaction: 13.9 → 10.0 (HR_max: 5% → 7%)
# - Research exploration: 2.8 → 3.5 (HR_max: 25% → 20%)
#
DOMAIN_ALPHA = {
    ApplicationDomain.DRUG_INTERACTION: 10.0,     # HR_max = 7% (adjusted from 5%)
    ApplicationDomain.CLINICAL_DECISION: 6.9,     # HR_max = 10%
    ApplicationDomain.LITERATURE_SUMMARY: 4.6,    # HR_max = 15%
    ApplicationDomain.RESEARCH_EXPLORATION: 3.5,  # HR_max = 20% (adjusted from 25%)
    ApplicationDomain.DEFAULT: 3.5                # HR_max = 20% (adjusted from 23%)
}


@dataclass
class ReasoningScore:
    """
    Score for a single reasoning question

    Score scale (Table 3.7):
    5 = Fully correct, semantically equivalent to gold answer
    4 = Mostly correct, minor imprecisions
    3 = Partially correct, missing >20% of key details
    2 = Safe abstention with expressed uncertainty
    1 = Partial hallucination, mixed correct/incorrect
    0 = Complete hallucination, confidently incorrect
    """
    question_id: str
    category: ReasoningCategory
    score: int  # 0-5 scale
    confidence: float = 0.5  # Model's confidence in response (0-1)

    # Optional metadata
    question_text: Optional[str] = None
    model_response: Optional[str] = None
    gold_answer: Optional[str] = None
    explanation: Optional[str] = None

    def is_hallucination(self) -> bool:
        """
        Check if response is a hallucination

        Hallucination defined as: score <= 1 and confidence > 0.7
        (Eq. 3.20)
        """
        return self.score <= 1 and self.confidence > 0.7

    def normalized_score(self) -> float:
        """Get score normalized to [0, 1]"""
        return self.score / 5.0


@dataclass
class CARESConfig:
    """Configuration for CARES scoring"""
    # Category weights (normalized internally)
    weight_s: float = 0.25  # Structure-aware
    weight_c: float = 0.25  # Causal-aware
    weight_r: float = 0.25  # Risk-aware
    weight_m: float = 0.25  # Semantic-aware

    # Application domain for hallucination penalty
    domain: ApplicationDomain = ApplicationDomain.DEFAULT

    # Custom alpha (overrides domain-based alpha if set)
    custom_alpha: Optional[float] = None

    # ECE calculation settings
    num_bins: int = 10  # Number of bins for calibration error

    def get_normalized_weights(self) -> Dict[ReasoningCategory, float]:
        """Get normalized category weights (sum to 1)"""
        total = self.weight_s + self.weight_c + self.weight_r + self.weight_m
        return {
            ReasoningCategory.STRUCTURE: self.weight_s / total,
            ReasoningCategory.CAUSAL: self.weight_c / total,
            ReasoningCategory.RISK: self.weight_r / total,
            ReasoningCategory.SEMANTIC: self.weight_m / total
        }

    def get_alpha(self) -> float:
        """Get alpha value for hallucination penalty"""
        if self.custom_alpha is not None:
            return self.custom_alpha
        return DOMAIN_ALPHA.get(self.domain, 3.0)


class CARESCalculator:
    """
    CARES Score Calculator

    Computes the Causal-Aware Reasoning Evaluation Score:

    CARES = [Σ w̃_k · CARES-k] × √(Φ(HR) × (1 - ECE))

    where:
    - w̃_k = normalized category weights
    - CARES-k = average score for category k
    - Φ(HR) = exp(-α · HR) = hallucination penalty
    - ECE = expected calibration error
    """

    def __init__(self, config: Optional[CARESConfig] = None):
        """
        Initialize CARES calculator

        Args:
            config: CARES configuration (uses defaults if None)
        """
        self.config = config or CARESConfig()
        self.scores: List[ReasoningScore] = []

    def add_score(
        self,
        question_id: str,
        category: str,
        score: int,
        confidence: float = 0.5,
        question_text: Optional[str] = None,
        model_response: Optional[str] = None,
        gold_answer: Optional[str] = None
    ):
        """
        Add a reasoning score

        Args:
            question_id: Unique question identifier
            category: Reasoning category ('S', 'C', 'R', or 'M')
            score: Score 0-5
            confidence: Model's confidence (0-1)
            question_text: Original question
            model_response: Model's response
            gold_answer: Ground truth answer
        """
        # Parse category
        cat_map = {
            'S': ReasoningCategory.STRUCTURE,
            'C': ReasoningCategory.CAUSAL,
            'R': ReasoningCategory.RISK,
            'M': ReasoningCategory.SEMANTIC
        }
        cat = cat_map.get(category.upper())
        if cat is None:
            raise ValueError(f"Invalid category: {category}")

        # Validate score
        if not 0 <= score <= 5:
            raise ValueError(f"Score must be 0-5, got {score}")

        # Validate confidence
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {confidence}")

        self.scores.append(ReasoningScore(
            question_id=question_id,
            category=cat,
            score=score,
            confidence=confidence,
            question_text=question_text,
            model_response=model_response,
            gold_answer=gold_answer
        ))

    def get_category_scores(self) -> Dict[ReasoningCategory, List[ReasoningScore]]:
        """Group scores by category"""
        by_category = {cat: [] for cat in ReasoningCategory}
        for score in self.scores:
            by_category[score.category].append(score)
        return by_category

    def compute_cares_k(self, category: ReasoningCategory) -> float:
        """
        Compute CARES score for a specific category

        CARES-k = (1/|Q_k|) × Σ(s_i / 5)  (Eq. 3.17)

        Args:
            category: Reasoning category

        Returns:
            Category CARES score in [0, 1]
        """
        cat_scores = [s for s in self.scores if s.category == category]
        if not cat_scores:
            return 0.0

        return sum(s.normalized_score() for s in cat_scores) / len(cat_scores)

    def compute_hallucination_rate(self, category: Optional[ReasoningCategory] = None) -> float:
        """
        Compute hallucination rate

        HR_k = |{q_i ∈ Q_k : s_i ≤ 1 and c_i > 0.7}| / |Q_k|  (Eq. 3.20)

        Args:
            category: Specific category (None for overall)

        Returns:
            Hallucination rate in [0, 1]
        """
        if category:
            scores = [s for s in self.scores if s.category == category]
        else:
            scores = self.scores

        if not scores:
            return 0.0

        hallucinations = sum(1 for s in scores if s.is_hallucination())
        return hallucinations / len(scores)

    def compute_hallucination_penalty(self, hr: Optional[float] = None) -> float:
        """
        Compute hallucination penalty

        Φ(HR) = exp(-α · HR)  (Eq. 3.18)

        Args:
            hr: Hallucination rate (computed if None)

        Returns:
            Penalty factor in (0, 1]
        """
        if hr is None:
            hr = self.compute_hallucination_rate()

        alpha = self.config.get_alpha()
        return math.exp(-alpha * hr)

    def compute_ece(self) -> float:
        """
        Compute Expected Calibration Error

        ECE = Σ (|B_b|/N) × |acc_b - conf_b|  (Eq. 3.19)

        Returns:
            ECE in [0, 1]
        """
        if not self.scores:
            return 0.0

        n_bins = self.config.num_bins
        bins = [[] for _ in range(n_bins)]

        # Assign scores to bins based on confidence
        for score in self.scores:
            bin_idx = min(int(score.confidence * n_bins), n_bins - 1)
            bins[bin_idx].append(score)

        ece = 0.0
        n_total = len(self.scores)

        for bin_scores in bins:
            if not bin_scores:
                continue

            # Average accuracy in bin (normalized score)
            acc_b = sum(s.normalized_score() for s in bin_scores) / len(bin_scores)

            # Average confidence in bin
            conf_b = sum(s.confidence for s in bin_scores) / len(bin_scores)

            # Weighted calibration error
            ece += (len(bin_scores) / n_total) * abs(acc_b - conf_b)

        return ece

    def compute_cares(self) -> float:
        """
        Compute overall CARES score

        CARES = [Σ w̃_k · CARES-k] × √(Φ(HR) × (1 - ECE))  (Eq. 3.15)

        Returns:
            CARES score in [0, 1]
        """
        if not self.scores:
            return 0.0

        # Get normalized weights
        weights = self.config.get_normalized_weights()

        # Compute weighted category scores
        weighted_sum = 0.0
        for cat in ReasoningCategory:
            cares_k = self.compute_cares_k(cat)
            weighted_sum += weights[cat] * cares_k

        # Compute hallucination penalty
        phi_hr = self.compute_hallucination_penalty()

        # Compute calibration error
        ece = self.compute_ece()

        # Final CARES score
        adjustment = math.sqrt(phi_hr * (1 - ece))
        cares = weighted_sum * adjustment

        return cares

    def get_detailed_report(self) -> Dict:
        """
        Get detailed CARES evaluation report

        Returns:
            Dictionary with all scoring components
        """
        weights = self.config.get_normalized_weights()

        # Category breakdown
        category_details = {}
        for cat in ReasoningCategory:
            cat_scores = [s for s in self.scores if s.category == cat]
            if cat_scores:
                category_details[cat.value] = {
                    'count': len(cat_scores),
                    'cares_k': self.compute_cares_k(cat),
                    'weight': weights[cat],
                    'hallucination_rate': self.compute_hallucination_rate(cat),
                    'avg_confidence': sum(s.confidence for s in cat_scores) / len(cat_scores),
                    'score_distribution': {
                        i: sum(1 for s in cat_scores if s.score == i)
                        for i in range(6)
                    }
                }

        # Overall metrics
        hr = self.compute_hallucination_rate()
        phi_hr = self.compute_hallucination_penalty(hr)
        ece = self.compute_ece()
        cares = self.compute_cares()

        return {
            'cares_score': cares,
            'total_questions': len(self.scores),
            'hallucination_rate': hr,
            'hallucination_penalty': phi_hr,
            'calibration_error': ece,
            'alpha': self.config.get_alpha(),
            'domain': self.config.domain.value,
            'category_weights': {cat.value: weights[cat] for cat in ReasoningCategory},
            'category_details': category_details
        }

    def export_scores(self, filepath: str):
        """Export scores to JSON file"""
        data = {
            'config': {
                'weight_s': self.config.weight_s,
                'weight_c': self.config.weight_c,
                'weight_r': self.config.weight_r,
                'weight_m': self.config.weight_m,
                'domain': self.config.domain.value,
                'alpha': self.config.get_alpha()
            },
            'scores': [
                {
                    'question_id': s.question_id,
                    'category': s.category.value,
                    'score': s.score,
                    'confidence': s.confidence,
                    'question_text': s.question_text,
                    'model_response': s.model_response,
                    'gold_answer': s.gold_answer
                }
                for s in self.scores
            ],
            'report': self.get_detailed_report()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_benchmark_results(
        cls,
        results: List[Dict],
        config: Optional[CARESConfig] = None
    ) -> 'CARESCalculator':
        """
        Create calculator from benchmark results

        Args:
            results: List of evaluation results with keys:
                - question_id
                - category (S, C, R, M)
                - score (0-5)
                - confidence (0-1)
            config: CARES configuration

        Returns:
            Initialized CARESCalculator
        """
        calc = cls(config)
        for result in results:
            calc.add_score(
                question_id=result['question_id'],
                category=result['category'],
                score=result['score'],
                confidence=result.get('confidence', 0.5),
                question_text=result.get('question_text'),
                model_response=result.get('model_response'),
                gold_answer=result.get('gold_answer')
            )
        return calc


def score_llm_response(
    response: str,
    gold_answer: str,
    category: str,
    question_text: Optional[str] = None,
    use_llm_judge: bool = False,
    judge_provider: str = "openai",
    api_key: Optional[str] = None
) -> Tuple[int, float]:
    """
    Score an LLM response against gold answer

    This function supports two modes:
    1. Simple heuristic scoring (fallback)
    2. LLM-as-Judge (recommended for production)

    Args:
        response: LLM response text
        gold_answer: Ground truth answer
        category: Reasoning category
        question_text: Original question
        use_llm_judge: Use LLM-as-Judge for evaluation
        judge_provider: "openai", "anthropic", or "local"
        api_key: API key for the judge provider

    Returns:
        Tuple of (score, confidence_estimate)
    """
    if not response or not gold_answer:
        return 0, 0.0

    # Use LLM-as-Judge if enabled
    if use_llm_judge:
        try:
            from .apis import OpenAIJudge, AnthropicJudge, LocalLLMJudge

            if judge_provider == "openai":
                judge = OpenAIJudge(api_key=api_key)
            elif judge_provider == "anthropic":
                judge = AnthropicJudge(api_key=api_key)
            elif judge_provider == "local":
                judge = LocalLLMJudge()
            else:
                raise ValueError(f"Unknown judge provider: {judge_provider}")

            result = judge.score_response(
                question=question_text or "",
                response=response,
                reference_answer=gold_answer,
                category=category
            )

            return result.score, result.confidence

        except ImportError:
            logger.warning("LLM Judge not available, falling back to heuristic scoring")
        except Exception as e:
            logger.error(f"LLM Judge failed: {e}, falling back to heuristic scoring")

    # Fallback: Simple heuristic scoring
    return _heuristic_score(response, gold_answer)


def _heuristic_score(response: str, gold_answer: str) -> Tuple[int, float]:
    """
    Simple heuristic scoring based on term overlap

    WARNING: This is a fallback method. For production use,
    enable LLM-as-Judge for accurate biomedical evaluation.
    """
    response_lower = response.lower()
    gold_lower = gold_answer.lower()

    # Exact match
    if response_lower == gold_lower:
        return 5, 0.95

    # Check for key terms from gold answer
    gold_terms = set(gold_lower.split())
    response_terms = set(response_lower.split())
    overlap = len(gold_terms & response_terms) / max(len(gold_terms), 1)

    # Estimate confidence from response patterns
    confidence = 0.5
    uncertainty_markers = ['maybe', 'possibly', 'uncertain', 'not sure', 'i think']
    confidence_markers = ['definitely', 'certainly', 'clearly', 'obviously']

    if any(marker in response_lower for marker in uncertainty_markers):
        confidence = 0.3
    elif any(marker in response_lower for marker in confidence_markers):
        confidence = 0.85

    # Determine score based on overlap
    if overlap > 0.8:
        score = 4
    elif overlap > 0.6:
        score = 3
    elif overlap > 0.3:
        score = 2
    elif overlap > 0.1:
        score = 1
    else:
        score = 0

    return score, confidence


class APIEnabledCARESCalculator(CARESCalculator):
    """
    CARES Calculator with LLM-as-Judge integration

    Uses LLM APIs to evaluate responses instead of simple heuristics.
    """

    def __init__(
        self,
        config: Optional[CARESConfig] = None,
        judge_provider: str = "openai",
        api_key: Optional[str] = None
    ):
        """
        Initialize API-enabled CARES calculator

        Args:
            config: CARES configuration
            judge_provider: "openai", "anthropic", or "local"
            api_key: API key for the judge
        """
        super().__init__(config)

        self.judge_provider = judge_provider
        self.api_key = api_key
        self._judge = None

    @property
    def judge(self):
        """Lazy-load LLM judge"""
        if self._judge is None:
            try:
                from .apis import OpenAIJudge, AnthropicJudge, LocalLLMJudge

                if self.judge_provider == "openai":
                    self._judge = OpenAIJudge(api_key=self.api_key)
                elif self.judge_provider == "anthropic":
                    self._judge = AnthropicJudge(api_key=self.api_key)
                elif self.judge_provider == "local":
                    self._judge = LocalLLMJudge()
                else:
                    logger.warning(f"Unknown judge provider: {self.judge_provider}")
            except ImportError:
                logger.warning("LLM Judge APIs not available")

        return self._judge

    def evaluate_response(
        self,
        question_id: str,
        category: str,
        question_text: str,
        model_response: str,
        gold_answer: str
    ) -> ReasoningScore:
        """
        Evaluate a response using LLM-as-Judge

        Args:
            question_id: Question identifier
            category: Reasoning category (S, C, R, M)
            question_text: Original question
            model_response: Model's response to evaluate
            gold_answer: Ground truth answer

        Returns:
            ReasoningScore with evaluation
        """
        # Try LLM Judge first
        if self.judge:
            try:
                result = self.judge.score_response(
                    question=question_text,
                    response=model_response,
                    reference_answer=gold_answer,
                    category=category
                )

                score = ReasoningScore(
                    question_id=question_id,
                    category=ReasoningCategory(category),
                    score=result.score,
                    confidence=result.confidence,
                    question_text=question_text,
                    model_response=model_response,
                    gold_answer=gold_answer,
                    explanation=result.reasoning
                )

                self.scores.append(score)
                return score

            except Exception as e:
                logger.warning(f"LLM Judge evaluation failed: {e}")

        # Fallback to heuristic
        heuristic_score, confidence = _heuristic_score(model_response, gold_answer)

        score = ReasoningScore(
            question_id=question_id,
            category=ReasoningCategory(category),
            score=heuristic_score,
            confidence=confidence,
            question_text=question_text,
            model_response=model_response,
            gold_answer=gold_answer,
            explanation="Evaluated using heuristic scoring (LLM Judge unavailable)"
        )

        self.scores.append(score)
        return score

    def evaluate_batch(
        self,
        evaluations: List[Dict]
    ) -> List[ReasoningScore]:
        """
        Evaluate multiple responses

        Args:
            evaluations: List of dicts with keys:
                - question_id
                - category
                - question_text
                - model_response
                - gold_answer

        Returns:
            List of ReasoningScore objects
        """
        results = []
        for ev in evaluations:
            result = self.evaluate_response(
                question_id=ev['question_id'],
                category=ev['category'],
                question_text=ev['question_text'],
                model_response=ev['model_response'],
                gold_answer=ev['gold_answer']
            )
            results.append(result)

        return results


def example_usage():
    """Example usage of CARES calculator"""
    # Configure for clinical decision support
    config = CARESConfig(
        weight_s=0.25,
        weight_c=0.30,  # Higher weight for causal reasoning
        weight_r=0.25,
        weight_m=0.20,
        domain=ApplicationDomain.CLINICAL_DECISION
    )

    calc = CARESCalculator(config)

    # Add some example scores
    example_scores = [
        # Structure-aware questions
        {'id': 'S-001', 'cat': 'S', 'score': 5, 'conf': 0.9},
        {'id': 'S-002', 'cat': 'S', 'score': 4, 'conf': 0.8},
        {'id': 'S-003', 'cat': 'S', 'score': 3, 'conf': 0.6},

        # Causal-aware questions
        {'id': 'C-001', 'cat': 'C', 'score': 4, 'conf': 0.85},
        {'id': 'C-002', 'cat': 'C', 'score': 5, 'conf': 0.95},
        {'id': 'C-003', 'cat': 'C', 'score': 1, 'conf': 0.8},  # Hallucination

        # Risk-aware questions
        {'id': 'R-001', 'cat': 'R', 'score': 5, 'conf': 0.9},
        {'id': 'R-002', 'cat': 'R', 'score': 4, 'conf': 0.75},

        # Semantic-aware questions
        {'id': 'M-001', 'cat': 'M', 'score': 4, 'conf': 0.7},
        {'id': 'M-002', 'cat': 'M', 'score': 3, 'conf': 0.6},
    ]

    for ex in example_scores:
        calc.add_score(ex['id'], ex['cat'], ex['score'], ex['conf'])

    # Compute CARES score
    cares = calc.compute_cares()
    print(f"CARES Score: {cares:.4f}")

    # Get detailed report
    report = calc.get_detailed_report()
    print(f"\nDetailed Report:")
    print(f"  Hallucination Rate: {report['hallucination_rate']:.2%}")
    print(f"  Hallucination Penalty: {report['hallucination_penalty']:.4f}")
    print(f"  Calibration Error: {report['calibration_error']:.4f}")

    print("\nCategory Breakdown:")
    for cat, details in report['category_details'].items():
        print(f"  {cat}: CARES-k={details['cares_k']:.3f}, "
              f"HR={details['hallucination_rate']:.2%}, "
              f"n={details['count']}")


if __name__ == '__main__':
    example_usage()
