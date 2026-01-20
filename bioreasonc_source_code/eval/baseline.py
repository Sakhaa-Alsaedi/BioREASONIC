"""
BioREASONC-Bench Baseline Evaluation

Provides baseline evaluation approach combining:
1. Standard NLP metrics (Accuracy, F1, BLEU, ROUGE, etc.)
2. Novel biomedical metrics (GRASS, CARES, ROCKET)

This validates the benchmark by showing both traditional
and domain-specific evaluation perspectives.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .metrics import StandardMetrics, EvaluationResults, evaluate_predictions


@dataclass
class BaselineResult:
    """Complete baseline evaluation result."""
    model_name: str

    # Standard Metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Generation Metrics
    bleu_1: float = 0.0
    bleu_4: float = 0.0
    rouge_1: float = 0.0
    rouge_l: float = 0.0
    meteor: float = 0.0

    # QA Metrics
    exact_match: float = 0.0
    token_f1: float = 0.0

    # Semantic Metrics
    semantic_similarity: float = 0.0

    # Novel Metrics
    grass_score: float = 0.0
    cares_score: float = 0.0
    rocket_score: float = 0.0

    # Per-category breakdown
    category_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Metadata
    num_samples: int = 0
    num_correct: int = 0


class BaselineEvaluator:
    """
    Baseline evaluator for BioREASONC-Bench.

    Provides a comprehensive evaluation combining standard NLP metrics
    with the novel GRASS, CARES, and ROCKET scores.
    """

    def __init__(
        self,
        use_bert_score: bool = False,
        use_llm_judge: bool = False,
        llm_provider: str = "openai"
    ):
        """
        Initialize the baseline evaluator.

        Args:
            use_bert_score: Whether to compute BERTScore
            use_llm_judge: Whether to use LLM-as-Judge for CARES
            llm_provider: LLM provider for judge ('openai', 'anthropic', etc.)
        """
        self.metrics = StandardMetrics(use_bert_score=use_bert_score)
        self.use_llm_judge = use_llm_judge
        self.llm_provider = llm_provider

        # Lazy load novel scorers
        self._grass_scorer = None
        self._cares_scorer = None
        self._rocket_scorer = None

    def _init_novel_scorers(self):
        """Initialize GRASS, CARES, and ROCKET scorers."""
        try:
            from .grass import GRASSScorer
            from .cares import CARESEvaluator
            from .rocket import ROCKETScore

            self._grass_scorer = GRASSScorer()
            self._cares_scorer = CARESEvaluator(
                use_llm_judge=self.use_llm_judge,
                llm_provider=self.llm_provider
            )
            self._rocket_scorer = ROCKETScore()
        except ImportError as e:
            print(f"Warning: Could not load novel scorers: {e}")

    def evaluate(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        model_name: str = "Model"
    ) -> BaselineResult:
        """
        Evaluate model predictions against references.

        Args:
            predictions: List of prediction dicts with 'answer', 'id', etc.
            references: List of reference dicts with 'answer', 'id', 'taxonomy', etc.
            model_name: Name of the model being evaluated

        Returns:
            BaselineResult with all metrics
        """
        result = BaselineResult(model_name=model_name, num_samples=len(predictions))

        # Match predictions to references by ID
        ref_map = {r['id']: r for r in references}
        matched_pairs = []

        for pred in predictions:
            pred_id = pred.get('id')
            if pred_id in ref_map:
                matched_pairs.append((pred, ref_map[pred_id]))

        if not matched_pairs:
            print("Warning: No matching prediction-reference pairs found!")
            return result

        # Extract answers and categories
        pred_answers = [p.get('answer', '') for p, _ in matched_pairs]
        ref_answers = [r.get('answer', '') for _, r in matched_pairs]
        categories = [r.get('taxonomy', 'unknown') for _, r in matched_pairs]

        # Compute standard metrics
        std_results = self.metrics.evaluate(
            pred_answers,
            ref_answers,
            task_type='qa',
            categories=categories
        )

        # Copy standard metrics to result
        result.accuracy = std_results.accuracy
        result.precision = std_results.precision
        result.recall = std_results.recall
        result.f1_score = std_results.f1_score
        result.bleu_1 = std_results.bleu_1
        result.bleu_4 = std_results.bleu_4
        result.rouge_1 = std_results.rouge_1
        result.rouge_l = std_results.rouge_l
        result.meteor = std_results.meteor
        result.exact_match = std_results.exact_match
        result.token_f1 = std_results.token_f1
        result.semantic_similarity = std_results.semantic_similarity
        result.category_scores = std_results.per_category

        # Compute novel metrics
        self._compute_novel_metrics(result, matched_pairs)

        # Count correct (based on token F1 > 0.5 threshold)
        result.num_correct = sum(
            1 for p, r in zip(pred_answers, ref_answers)
            if self._is_correct(p, r)
        )

        return result

    def _is_correct(self, prediction: str, reference: str, threshold: float = 0.5) -> bool:
        """Determine if prediction is correct based on token F1."""
        token_f1 = self.metrics.compute_token_f1([prediction], [reference])
        return token_f1 >= threshold

    def _compute_novel_metrics(
        self,
        result: BaselineResult,
        matched_pairs: List[Tuple[Dict, Dict]]
    ):
        """Compute GRASS, CARES, and ROCKET scores."""
        if self._grass_scorer is None:
            self._init_novel_scorers()

        if self._grass_scorer is None:
            # Fallback: estimate from standard metrics
            result.grass_score = self._estimate_grass(result)
            result.cares_score = self._estimate_cares(result)
            result.rocket_score = self._estimate_rocket(result)
            return

        # Use actual scorers if available
        try:
            # GRASS: Gene Risk Aggregation
            grass_items = [
                (p.get('answer', ''), r.get('answer', ''), r.get('source_genes', []))
                for p, r in matched_pairs
                if r.get('taxonomy') in ['R', 'S']  # Risk and Structure
            ]
            if grass_items:
                result.grass_score = self._grass_scorer.score_batch(grass_items)

            # CARES: Causal-Aware Reasoning
            cares_items = [
                (p.get('answer', ''), r.get('answer', ''), r.get('explanation', ''))
                for p, r in matched_pairs
                if r.get('taxonomy') == 'C'  # Causal
            ]
            if cares_items:
                result.cares_score = self._cares_scorer.score_batch(cares_items)

            # ROCKET: Combined score
            result.rocket_score = self._rocket_scorer.compute(
                grass_score=result.grass_score,
                cares_score=result.cares_score,
                standard_metrics={
                    'token_f1': result.token_f1,
                    'semantic_similarity': result.semantic_similarity
                }
            )
        except Exception as e:
            print(f"Warning: Novel metric computation failed: {e}")
            result.grass_score = self._estimate_grass(result)
            result.cares_score = self._estimate_cares(result)
            result.rocket_score = self._estimate_rocket(result)

    def _estimate_grass(self, result: BaselineResult) -> float:
        """Estimate GRASS from standard metrics."""
        # GRASS emphasizes gene/risk accuracy
        # Approximate with token F1 and semantic similarity
        return 0.6 * result.token_f1 + 0.4 * result.semantic_similarity

    def _estimate_cares(self, result: BaselineResult) -> float:
        """Estimate CARES from standard metrics."""
        # CARES emphasizes causal reasoning quality
        # Approximate with ROUGE-L and METEOR (capture reasoning flow)
        return 0.4 * result.rouge_l + 0.3 * result.meteor + 0.3 * result.token_f1

    def _estimate_rocket(self, result: BaselineResult) -> float:
        """Estimate ROCKET from other metrics."""
        # ROCKET is a composite of all metrics
        grass = result.grass_score or self._estimate_grass(result)
        cares = result.cares_score or self._estimate_cares(result)

        return (
            0.25 * grass +           # R: Risk
            0.15 * result.token_f1 + # O: Omics accuracy
            0.25 * cares +           # C: Causal reasoning
            0.15 * result.rouge_l +  # K: Knowledge
            0.10 * result.meteor +   # E: Evidence
            0.10 * result.semantic_similarity  # T: Trust
        )

    def evaluate_from_files(
        self,
        predictions_file: str,
        references_file: str,
        model_name: str = "Model"
    ) -> BaselineResult:
        """
        Evaluate from JSONL files.

        Args:
            predictions_file: Path to predictions JSONL
            references_file: Path to references JSONL
            model_name: Name of the model

        Returns:
            BaselineResult
        """
        predictions = []
        references = []

        with open(predictions_file, 'r') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))

        with open(references_file, 'r') as f:
            for line in f:
                if line.strip():
                    references.append(json.loads(line))

        return self.evaluate(predictions, references, model_name)

    def compare_models(
        self,
        model_results: Dict[str, BaselineResult]
    ) -> Dict[str, Any]:
        """
        Compare multiple models and generate comparison summary.

        Args:
            model_results: Dict mapping model names to BaselineResult

        Returns:
            Comparison summary with rankings
        """
        comparison = {
            'models': list(model_results.keys()),
            'rankings': {},
            'metrics': {}
        }

        # Define metrics to compare
        metric_names = [
            'accuracy', 'f1_score', 'exact_match', 'token_f1',
            'bleu_1', 'rouge_l', 'meteor', 'semantic_similarity',
            'grass_score', 'cares_score', 'rocket_score'
        ]

        for metric in metric_names:
            values = {
                name: getattr(result, metric, 0.0)
                for name, result in model_results.items()
            }
            comparison['metrics'][metric] = values

            # Rank models for this metric (higher is better)
            ranked = sorted(values.items(), key=lambda x: x[1], reverse=True)
            comparison['rankings'][metric] = [name for name, _ in ranked]

        # Overall ranking (by ROCKET + Token F1)
        overall_scores = {
            name: result.rocket_score * 0.6 + result.token_f1 * 0.4
            for name, result in model_results.items()
        }
        comparison['rankings']['overall'] = sorted(
            overall_scores.keys(),
            key=lambda x: overall_scores[x],
            reverse=True
        )

        # Best model per category
        comparison['best_per_metric'] = {
            metric: comparison['rankings'][metric][0]
            for metric in metric_names
        }

        return comparison

    def to_dict(self, result: BaselineResult) -> Dict[str, Any]:
        """Convert BaselineResult to dictionary."""
        return {
            'model_name': result.model_name,
            'standard_metrics': {
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1_score': result.f1_score
            },
            'generation_metrics': {
                'bleu_1': result.bleu_1,
                'bleu_4': result.bleu_4,
                'rouge_1': result.rouge_1,
                'rouge_l': result.rouge_l,
                'meteor': result.meteor
            },
            'qa_metrics': {
                'exact_match': result.exact_match,
                'token_f1': result.token_f1
            },
            'semantic_metrics': {
                'similarity': result.semantic_similarity
            },
            'novel_metrics': {
                'grass_score': result.grass_score,
                'cares_score': result.cares_score,
                'rocket_score': result.rocket_score
            },
            'per_category': result.category_scores,
            'metadata': {
                'num_samples': result.num_samples,
                'num_correct': result.num_correct
            }
        }

    def generate_report(
        self,
        results: Dict[str, BaselineResult],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a comparison report.

        Args:
            results: Dict mapping model names to BaselineResult
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        comparison = self.compare_models(results)

        lines = []
        lines.append("# BioREASONC-Bench Baseline Evaluation Report\n")

        # Overall ranking
        lines.append("## Overall Ranking\n")
        lines.append("| Rank | Model | ROCKET | Token F1 | Combined |")
        lines.append("|------|-------|--------|----------|----------|")

        for i, model in enumerate(comparison['rankings']['overall'], 1):
            r = results[model]
            combined = r.rocket_score * 0.6 + r.token_f1 * 0.4
            lines.append(f"| {i} | {model} | {r.rocket_score:.3f} | {r.token_f1:.3f} | {combined:.3f} |")

        # Standard metrics comparison
        lines.append("\n## Standard Metrics Comparison\n")
        lines.append("| Model | Accuracy | F1 | EM | Token F1 |")
        lines.append("|-------|----------|-----|-----|----------|")

        for model, r in results.items():
            lines.append(f"| {model} | {r.accuracy:.3f} | {r.f1_score:.3f} | {r.exact_match:.3f} | {r.token_f1:.3f} |")

        # Generation metrics
        lines.append("\n## Generation Metrics\n")
        lines.append("| Model | BLEU-1 | BLEU-4 | ROUGE-1 | ROUGE-L | METEOR |")
        lines.append("|-------|--------|--------|---------|---------|--------|")

        for model, r in results.items():
            lines.append(f"| {model} | {r.bleu_1:.3f} | {r.bleu_4:.3f} | {r.rouge_1:.3f} | {r.rouge_l:.3f} | {r.meteor:.3f} |")

        # Novel metrics
        lines.append("\n## Novel BioREASONC Metrics\n")
        lines.append("| Model | GRASS | CARES | ROCKET |")
        lines.append("|-------|-------|-------|--------|")

        for model, r in results.items():
            lines.append(f"| {model} | {r.grass_score:.3f} | {r.cares_score:.3f} | {r.rocket_score:.3f} |")

        # Best per metric
        lines.append("\n## Best Model per Metric\n")
        for metric, best_model in comparison['best_per_metric'].items():
            value = comparison['metrics'][metric][best_model]
            lines.append(f"- **{metric}**: {best_model} ({value:.3f})")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(report)

        return report


# Convenience function
def run_baseline_evaluation(
    predictions_file: str,
    references_file: str,
    model_name: str = "Model",
    use_bert_score: bool = False,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run baseline evaluation on prediction and reference files.

    Args:
        predictions_file: Path to predictions JSONL
        references_file: Path to references JSONL
        model_name: Name of the model
        use_bert_score: Whether to compute BERTScore
        output_file: Optional path to save results JSON

    Returns:
        Dictionary with all evaluation metrics
    """
    evaluator = BaselineEvaluator(use_bert_score=use_bert_score)
    result = evaluator.evaluate_from_files(predictions_file, references_file, model_name)
    result_dict = evaluator.to_dict(result)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

    return result_dict
