"""
BioREASONC-Bench Standard Evaluation Metrics

Provides standard NLP and ML metrics for baseline evaluation:
- Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- Text Generation: BLEU, ROUGE, METEOR
- QA-Specific: Exact Match (EM), Token-level F1
- Semantic: BERTScore, Embedding Similarity

These complement the novel GRASS, CARES, and ROCKET scores.
"""

import re
import math
import string
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import Counter
import numpy as np


@dataclass
class MetricResult:
    """Container for metric computation results."""
    name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    support: int = 0  # Number of samples
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """Complete evaluation results with all metrics."""
    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_roc: Optional[float] = None

    # Text generation metrics
    bleu_1: float = 0.0
    bleu_2: float = 0.0
    bleu_4: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0
    meteor: float = 0.0

    # QA-specific metrics
    exact_match: float = 0.0
    token_f1: float = 0.0

    # Semantic metrics
    semantic_similarity: float = 0.0
    bertscore_f1: Optional[float] = None

    # Novel metrics (for integration)
    grass_score: Optional[float] = None
    cares_score: Optional[float] = None
    rocket_score: Optional[float] = None

    # Metadata
    num_samples: int = 0
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)


class StandardMetrics:
    """
    Computes standard NLP evaluation metrics.

    Provides baseline metrics that complement the novel
    GRASS, CARES, and ROCKET scoring systems.
    """

    def __init__(self, use_bert_score: bool = False):
        """
        Initialize metrics calculator.

        Args:
            use_bert_score: Whether to compute BERTScore (requires model)
        """
        self.use_bert_score = use_bert_score
        self._bert_scorer = None

        if use_bert_score:
            self._init_bert_scorer()

    def _init_bert_scorer(self):
        """Initialize BERTScore model."""
        try:
            from bert_score import BERTScorer
            self._bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        except ImportError:
            print("Warning: bert_score not installed. BERTScore will be skipped.")
            self.use_bert_score = False

    # ==================== Text Normalization ====================

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return StandardMetrics.normalize_text(text).split()

    # ==================== Classification Metrics ====================

    def compute_accuracy(
        self,
        predictions: List[Any],
        references: List[Any]
    ) -> float:
        """
        Compute accuracy for classification tasks.

        Args:
            predictions: List of predicted labels
            references: List of ground truth labels

        Returns:
            Accuracy score (0-1)
        """
        if not predictions or not references:
            return 0.0

        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        return correct / len(predictions)

    def compute_precision_recall_f1(
        self,
        predictions: List[Any],
        references: List[Any],
        positive_label: Any = 1,
        average: str = 'binary'
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score.

        Args:
            predictions: List of predicted labels
            references: List of ground truth labels
            positive_label: Label considered positive
            average: 'binary', 'micro', 'macro', or 'weighted'

        Returns:
            Tuple of (precision, recall, f1)
        """
        if not predictions or not references:
            return 0.0, 0.0, 0.0

        if average == 'binary':
            tp = sum(1 for p, r in zip(predictions, references)
                    if p == positive_label and r == positive_label)
            fp = sum(1 for p, r in zip(predictions, references)
                    if p == positive_label and r != positive_label)
            fn = sum(1 for p, r in zip(predictions, references)
                    if p != positive_label and r == positive_label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return precision, recall, f1

        elif average in ['micro', 'macro', 'weighted']:
            labels = set(predictions) | set(references)

            if average == 'micro':
                tp_total = sum(1 for p, r in zip(predictions, references) if p == r)
                total = len(predictions)
                precision = recall = f1 = tp_total / total if total > 0 else 0.0
                return precision, recall, f1

            else:  # macro or weighted
                precisions, recalls, f1s, supports = [], [], [], []

                for label in labels:
                    p, r, f = self.compute_precision_recall_f1(
                        predictions, references, positive_label=label, average='binary'
                    )
                    precisions.append(p)
                    recalls.append(r)
                    f1s.append(f)
                    supports.append(sum(1 for ref in references if ref == label))

                if average == 'macro':
                    return (
                        np.mean(precisions),
                        np.mean(recalls),
                        np.mean(f1s)
                    )
                else:  # weighted
                    total_support = sum(supports)
                    if total_support == 0:
                        return 0.0, 0.0, 0.0

                    weights = [s / total_support for s in supports]
                    return (
                        sum(p * w for p, w in zip(precisions, weights)),
                        sum(r * w for r, w in zip(recalls, weights)),
                        sum(f * w for f, w in zip(f1s, weights))
                    )

        return 0.0, 0.0, 0.0

    def compute_auc_roc(
        self,
        predictions: List[float],
        references: List[int]
    ) -> float:
        """
        Compute AUC-ROC score.

        Args:
            predictions: List of predicted probabilities
            references: List of binary labels (0 or 1)

        Returns:
            AUC-ROC score
        """
        if not predictions or not references:
            return 0.0

        # Sort by prediction score
        sorted_pairs = sorted(zip(predictions, references), key=lambda x: x[0], reverse=True)

        n_pos = sum(references)
        n_neg = len(references) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Compute AUC using trapezoidal rule
        auc = 0.0
        tp = 0
        fp = 0
        prev_tpr = 0.0
        prev_fpr = 0.0

        for _, label in sorted_pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1

            tpr = tp / n_pos
            fpr = fp / n_neg

            # Trapezoidal area
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2

            prev_tpr = tpr
            prev_fpr = fpr

        return auc

    # ==================== Text Generation Metrics ====================

    def compute_bleu(
        self,
        predictions: List[str],
        references: List[str],
        max_n: int = 4,
        weights: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Compute BLEU scores (1 through max_n).

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            max_n: Maximum n-gram order
            weights: Weights for each n-gram (default: uniform)

        Returns:
            Dict with BLEU-1, BLEU-2, etc.
        """
        if not predictions or not references:
            return {f'bleu_{n}': 0.0 for n in range(1, max_n + 1)}

        if weights is None:
            weights = [1.0 / max_n] * max_n

        results = {}

        for n in range(1, max_n + 1):
            scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = self.tokenize(pred)
                ref_tokens = self.tokenize(ref)

                if len(pred_tokens) < n or len(ref_tokens) < n:
                    scores.append(0.0)
                    continue

                # Get n-grams
                pred_ngrams = Counter(
                    tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)
                )
                ref_ngrams = Counter(
                    tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)
                )

                # Clipped counts
                clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
                total = sum(pred_ngrams.values())

                precision = clipped / total if total > 0 else 0.0

                # Brevity penalty
                bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else \
                     math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))

                scores.append(bp * precision)

            results[f'bleu_{n}'] = np.mean(scores) if scores else 0.0

        # Combined BLEU score (geometric mean)
        combined = 0.0
        for n, w in enumerate(weights, 1):
            if results.get(f'bleu_{n}', 0) > 0:
                combined += w * math.log(results[f'bleu_{n}'])
        results['bleu'] = math.exp(combined) if combined > float('-inf') else 0.0

        return results

    def compute_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        if not predictions or not references:
            return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}

        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = self.tokenize(pred)
            ref_tokens = self.tokenize(ref)

            # ROUGE-1 (unigram)
            pred_unigrams = Counter(pred_tokens)
            ref_unigrams = Counter(ref_tokens)
            overlap = sum((pred_unigrams & ref_unigrams).values())

            r1_precision = overlap / len(pred_tokens) if pred_tokens else 0.0
            r1_recall = overlap / len(ref_tokens) if ref_tokens else 0.0
            r1_f1 = 2 * r1_precision * r1_recall / (r1_precision + r1_recall) \
                    if (r1_precision + r1_recall) > 0 else 0.0
            rouge_1_scores.append(r1_f1)

            # ROUGE-2 (bigram)
            if len(pred_tokens) >= 2 and len(ref_tokens) >= 2:
                pred_bigrams = Counter(
                    tuple(pred_tokens[i:i+2]) for i in range(len(pred_tokens) - 1)
                )
                ref_bigrams = Counter(
                    tuple(ref_tokens[i:i+2]) for i in range(len(ref_tokens) - 1)
                )
                overlap = sum((pred_bigrams & ref_bigrams).values())

                r2_precision = overlap / len(pred_bigrams) if pred_bigrams else 0.0
                r2_recall = overlap / len(ref_bigrams) if ref_bigrams else 0.0
                r2_f1 = 2 * r2_precision * r2_recall / (r2_precision + r2_recall) \
                        if (r2_precision + r2_recall) > 0 else 0.0
                rouge_2_scores.append(r2_f1)
            else:
                rouge_2_scores.append(0.0)

            # ROUGE-L (longest common subsequence)
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)
            rl_precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
            rl_recall = lcs_length / len(ref_tokens) if ref_tokens else 0.0
            rl_f1 = 2 * rl_precision * rl_recall / (rl_precision + rl_recall) \
                    if (rl_precision + rl_recall) > 0 else 0.0
            rouge_l_scores.append(rl_f1)

        return {
            'rouge_1': np.mean(rouge_1_scores),
            'rouge_2': np.mean(rouge_2_scores),
            'rouge_l': np.mean(rouge_l_scores)
        }

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        if m == 0 or n == 0:
            return 0

        # Optimize space: only keep two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev

        return prev[n]

    def compute_meteor(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute METEOR score (simplified version).

        Uses unigram matching with stemming approximation.

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            METEOR score
        """
        if not predictions or not references:
            return 0.0

        scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = self.tokenize(pred)
            ref_tokens = self.tokenize(ref)

            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue

            # Exact matches
            pred_set = set(pred_tokens)
            ref_set = set(ref_tokens)
            matches = len(pred_set & ref_set)

            # Precision and recall
            precision = matches / len(pred_tokens)
            recall = matches / len(ref_tokens)

            if precision + recall == 0:
                scores.append(0.0)
                continue

            # F-mean with recall weighted higher (alpha=0.9)
            alpha = 0.9
            f_mean = precision * recall / (alpha * precision + (1 - alpha) * recall)

            # Fragmentation penalty (simplified)
            chunks = self._count_chunks(pred_tokens, ref_tokens)
            frag = chunks / matches if matches > 0 else 1.0
            penalty = 0.5 * (frag ** 3)

            meteor = f_mean * (1 - penalty)
            scores.append(max(0, meteor))

        return np.mean(scores)

    def _count_chunks(self, pred: List[str], ref: List[str]) -> int:
        """Count number of chunks (contiguous matches)."""
        ref_set = set(ref)
        in_chunk = False
        chunks = 0

        for token in pred:
            if token in ref_set:
                if not in_chunk:
                    chunks += 1
                    in_chunk = True
            else:
                in_chunk = False

        return max(chunks, 1)

    # ==================== QA-Specific Metrics ====================

    def compute_exact_match(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute Exact Match (EM) score.

        Args:
            predictions: List of predicted answers
            references: List of reference answers

        Returns:
            EM score (0-1)
        """
        if not predictions or not references:
            return 0.0

        matches = sum(
            1 for pred, ref in zip(predictions, references)
            if self.normalize_text(pred) == self.normalize_text(ref)
        )

        return matches / len(predictions)

    def compute_token_f1(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """
        Compute token-level F1 score (SQuAD-style).

        Args:
            predictions: List of predicted answers
            references: List of reference answers

        Returns:
            Token F1 score
        """
        if not predictions or not references:
            return 0.0

        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = self.tokenize(pred)
            ref_tokens = self.tokenize(ref)

            if not pred_tokens and not ref_tokens:
                f1_scores.append(1.0)
                continue
            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                f1_scores.append(0.0)
                continue

            precision = num_same / len(pred_tokens)
            recall = num_same / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

        return np.mean(f1_scores)

    # ==================== Semantic Metrics ====================

    def compute_semantic_similarity(
        self,
        predictions: List[str],
        references: List[str],
        method: str = 'jaccard'
    ) -> float:
        """
        Compute semantic similarity.

        Args:
            predictions: List of predicted texts
            references: List of reference texts
            method: 'jaccard', 'cosine', or 'embedding'

        Returns:
            Similarity score (0-1)
        """
        if not predictions or not references:
            return 0.0

        if method == 'jaccard':
            similarities = []
            for pred, ref in zip(predictions, references):
                pred_set = set(self.tokenize(pred))
                ref_set = set(self.tokenize(ref))

                if not pred_set and not ref_set:
                    similarities.append(1.0)
                elif not pred_set or not ref_set:
                    similarities.append(0.0)
                else:
                    intersection = len(pred_set & ref_set)
                    union = len(pred_set | ref_set)
                    similarities.append(intersection / union)

            return np.mean(similarities)

        elif method == 'cosine':
            # TF-based cosine similarity
            similarities = []
            for pred, ref in zip(predictions, references):
                pred_tokens = self.tokenize(pred)
                ref_tokens = self.tokenize(ref)

                pred_tf = Counter(pred_tokens)
                ref_tf = Counter(ref_tokens)

                all_tokens = set(pred_tf.keys()) | set(ref_tf.keys())

                if not all_tokens:
                    similarities.append(0.0)
                    continue

                dot_product = sum(pred_tf.get(t, 0) * ref_tf.get(t, 0) for t in all_tokens)
                pred_norm = math.sqrt(sum(v**2 for v in pred_tf.values()))
                ref_norm = math.sqrt(sum(v**2 for v in ref_tf.values()))

                if pred_norm * ref_norm == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(dot_product / (pred_norm * ref_norm))

            return np.mean(similarities)

        return 0.0

    def compute_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Optional[Dict[str, float]]:
        """
        Compute BERTScore (requires bert_score package).

        Args:
            predictions: List of predicted texts
            references: List of reference texts

        Returns:
            Dict with precision, recall, F1 or None if unavailable
        """
        if not self.use_bert_score or self._bert_scorer is None:
            return None

        try:
            P, R, F1 = self._bert_scorer.score(predictions, references)
            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            return None

    # ==================== Combined Evaluation ====================

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        task_type: str = 'qa',
        categories: Optional[List[str]] = None
    ) -> EvaluationResults:
        """
        Compute all relevant metrics for a task.

        Args:
            predictions: List of predicted texts/labels
            references: List of reference texts/labels
            task_type: 'qa', 'classification', 'generation'
            categories: Optional list of category labels for per-category metrics

        Returns:
            EvaluationResults with all computed metrics
        """
        results = EvaluationResults(num_samples=len(predictions))

        # Text generation metrics (always compute for QA)
        if task_type in ['qa', 'generation']:
            bleu_scores = self.compute_bleu(predictions, references)
            results.bleu_1 = bleu_scores.get('bleu_1', 0.0)
            results.bleu_2 = bleu_scores.get('bleu_2', 0.0)
            results.bleu_4 = bleu_scores.get('bleu_4', 0.0)

            rouge_scores = self.compute_rouge(predictions, references)
            results.rouge_1 = rouge_scores.get('rouge_1', 0.0)
            results.rouge_2 = rouge_scores.get('rouge_2', 0.0)
            results.rouge_l = rouge_scores.get('rouge_l', 0.0)

            results.meteor = self.compute_meteor(predictions, references)

        # QA-specific metrics
        if task_type == 'qa':
            results.exact_match = self.compute_exact_match(predictions, references)
            results.token_f1 = self.compute_token_f1(predictions, references)

        # Classification metrics
        if task_type == 'classification':
            results.accuracy = self.compute_accuracy(predictions, references)
            p, r, f1 = self.compute_precision_recall_f1(
                predictions, references, average='macro'
            )
            results.precision = p
            results.recall = r
            results.f1_score = f1

        # Semantic similarity (always compute)
        results.semantic_similarity = self.compute_semantic_similarity(
            predictions, references, method='cosine'
        )

        # BERTScore (if enabled)
        if self.use_bert_score:
            bert_scores = self.compute_bertscore(predictions, references)
            if bert_scores:
                results.bertscore_f1 = bert_scores.get('bertscore_f1')

        # Per-category metrics
        if categories:
            unique_cats = set(categories)
            for cat in unique_cats:
                cat_indices = [i for i, c in enumerate(categories) if c == cat]
                cat_preds = [predictions[i] for i in cat_indices]
                cat_refs = [references[i] for i in cat_indices]

                cat_results = self.evaluate(cat_preds, cat_refs, task_type)
                results.per_category[cat] = {
                    'accuracy': cat_results.accuracy,
                    'f1_score': cat_results.f1_score,
                    'token_f1': cat_results.token_f1,
                    'exact_match': cat_results.exact_match,
                    'bleu_1': cat_results.bleu_1,
                    'rouge_l': cat_results.rouge_l,
                    'num_samples': len(cat_preds)
                }

        return results

    def to_dict(self, results: EvaluationResults) -> Dict[str, Any]:
        """Convert EvaluationResults to dictionary."""
        return {
            'classification': {
                'accuracy': results.accuracy,
                'precision': results.precision,
                'recall': results.recall,
                'f1_score': results.f1_score,
                'auc_roc': results.auc_roc
            },
            'generation': {
                'bleu_1': results.bleu_1,
                'bleu_2': results.bleu_2,
                'bleu_4': results.bleu_4,
                'rouge_1': results.rouge_1,
                'rouge_2': results.rouge_2,
                'rouge_l': results.rouge_l,
                'meteor': results.meteor
            },
            'qa': {
                'exact_match': results.exact_match,
                'token_f1': results.token_f1
            },
            'semantic': {
                'similarity': results.semantic_similarity,
                'bertscore_f1': results.bertscore_f1
            },
            'novel_scores': {
                'grass': results.grass_score,
                'cares': results.cares_score,
                'rocket': results.rocket_score
            },
            'metadata': {
                'num_samples': results.num_samples
            },
            'per_category': results.per_category
        }


# Convenience function
def evaluate_predictions(
    predictions: List[str],
    references: List[str],
    task_type: str = 'qa',
    categories: Optional[List[str]] = None,
    use_bert_score: bool = False
) -> Dict[str, Any]:
    """
    Evaluate predictions against references using standard metrics.

    Args:
        predictions: List of model predictions
        references: List of ground truth references
        task_type: 'qa', 'classification', or 'generation'
        categories: Optional category labels for per-category analysis
        use_bert_score: Whether to compute BERTScore

    Returns:
        Dictionary with all computed metrics
    """
    metrics = StandardMetrics(use_bert_score=use_bert_score)
    results = metrics.evaluate(predictions, references, task_type, categories)
    return metrics.to_dict(results)
