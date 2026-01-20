"""
Human Expert Evaluator Module for BioREASONC-Creator

This module provides a human-in-the-loop evaluation system:
1. Takes validation results from MultiLLMValidator
2. Exports to CSV for human expert review
3. Collects human agreement/disagreement with LLM judgments
4. Provides feedback to generator for improving question generation

Focus: "Does the model tell the truth about causality when explaining biomedical research?"

Pipeline Flow:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  BIOREASONC PIPELINE                                                  │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │   1. Generator (generator.py)                                         │
    │      └─── Creates Q&A pairs from GWAS data                           │
    │              │                                                        │
    │              ▼                                                        │
    │   2. Paraphraser (paraphraser.py)                                    │
    │      └─── Generates 2-3 diverse paraphrases per question             │
    │              │                                                        │
    │              ▼                                                        │
    │   3. Explainer (explainer.py)                                        │
    │      └─── Adds ~35 word explanations for each Q&A                    │
    │              │                                                        │
    │              ▼                                                        │
    │   4. Validator (validator.py)                                        │
    │      └─── Multi-LLM validation with consensus scoring                │
    │              │                                                        │
    │              ▼                                                        │
    │   5. Human Expert Evaluator (human_exp_evaluator.py)  ◀── THIS FILE  │
    │      └─── Human reviews LLM judgments (Agree=1 / Disagree=0)         │
    │              │                                                        │
    │              ▼                                                        │
    │   6. Feedback Loop                                                   │
    │      └─── Disagreements fed back to Generator for improvement        │
    │                                                                       │
    └──────────────────────────────────────────────────────────────────────┘
"""

import csv
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Import from other modules
from .validator import ValidationResult


@dataclass
class HumanEvaluation:
    """Human expert evaluation record."""
    item_id: str
    question: str
    answer: str
    taxonomy: str
    label: str
    # LLM Validator results
    llm_avg_score: float
    llm_consensus_judgment: str  # ASSOCIATIVE | CAUSAL | NOT_APPLICABLE
    llm_overclaim_consensus: bool
    llm_agreement_score: int  # 1 = LLMs agree, 0 = LLMs disagree
    llm_is_valid: bool
    # Human expert evaluation
    human_judgment: Optional[str] = None  # ASSOCIATIVE | CAUSAL | NOT_APPLICABLE
    human_is_overclaim: Optional[bool] = None
    human_agrees_with_llm: Optional[int] = None  # 1 = Agree, 0 = Disagree
    human_feedback: Optional[str] = None
    human_evaluator_id: Optional[str] = None
    evaluation_timestamp: Optional[str] = None


@dataclass
class FeedbackRecord:
    """Feedback record for generator improvement."""
    item_id: str
    taxonomy: str
    label: str
    issue_type: str  # OVERCLAIM | UNDERCLAIM | ENTITY_ERROR | OTHER
    original_answer: str
    corrected_judgment: str
    human_feedback: str
    should_regenerate: bool


class HumanExpertEvaluator:
    """
    Human-in-the-loop evaluation system for validating LLM judgments
    and providing feedback to the generator.
    """

    def __init__(
        self,
        output_dir: str = "./human_evaluation",
        evaluator_id: Optional[str] = None
    ):
        """
        Initialize the human expert evaluator.

        Args:
            output_dir: Directory for evaluation CSV files
            evaluator_id: Optional identifier for the human evaluator
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator_id = evaluator_id or "anonymous"

        # CSV column headers
        self.export_columns = [
            'item_id', 'question', 'answer', 'explanation', 'taxonomy', 'label',
            'llm_avg_score', 'llm_consensus_judgment', 'llm_overclaim_consensus',
            'llm_agreement_score', 'llm_is_valid',
            'llm_causal_judgments', 'llm_evidence',
            # Human evaluation columns (to be filled by expert)
            'human_judgment', 'human_is_overclaim', 'human_agrees_with_llm',
            'human_feedback', 'human_evaluator_id', 'evaluation_timestamp'
        ]

        self.feedback_columns = [
            'item_id', 'taxonomy', 'label', 'issue_type',
            'original_answer', 'corrected_judgment', 'human_feedback',
            'should_regenerate'
        ]

    def export_for_human_review(
        self,
        items: List[Dict[str, Any]],
        validation_results: List[ValidationResult],
        filename: Optional[str] = None
    ) -> str:
        """
        Export validation results to CSV for human expert review.

        Args:
            items: Original Q&A items
            validation_results: Results from MultiLLMValidator
            filename: Optional custom filename

        Returns:
            Path to the exported CSV file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"human_review_{timestamp}.csv"

        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.export_columns)
            writer.writeheader()

            for item, result in zip(items, validation_results):
                row = {
                    'item_id': result.item_id,
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'explanation': item.get('explanation', ''),
                    'taxonomy': item.get('taxonomy', ''),
                    'label': item.get('label', ''),
                    'llm_avg_score': result.avg_score,
                    'llm_consensus_judgment': result.consensus_judgment,
                    'llm_overclaim_consensus': result.overclaim_consensus,
                    'llm_agreement_score': result.agreement_score,
                    'llm_is_valid': result.is_valid,
                    'llm_causal_judgments': json.dumps(result.causal_judgments or {}),
                    'llm_evidence': json.dumps(result.evidence or {}),
                    # Empty columns for human to fill
                    'human_judgment': '',
                    'human_is_overclaim': '',
                    'human_agrees_with_llm': '',
                    'human_feedback': '',
                    'human_evaluator_id': '',
                    'evaluation_timestamp': ''
                }
                writer.writerow(row)

        print(f"Exported {len(items)} items for human review: {filepath}")
        return str(filepath)

    def load_human_evaluations(self, filepath: str) -> List[HumanEvaluation]:
        """
        Load human-completed evaluations from CSV.

        Args:
            filepath: Path to the completed CSV file

        Returns:
            List of HumanEvaluation records
        """
        evaluations = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Parse human agreement (1 = Agree, 0 = Disagree, empty = not evaluated)
                human_agrees = row.get('human_agrees_with_llm', '').strip()
                if human_agrees:
                    human_agrees = int(human_agrees) if human_agrees.isdigit() else None
                else:
                    human_agrees = None

                # Parse human overclaim judgment
                human_overclaim = row.get('human_is_overclaim', '').strip().lower()
                if human_overclaim in ['true', '1', 'yes']:
                    human_overclaim = True
                elif human_overclaim in ['false', '0', 'no']:
                    human_overclaim = False
                else:
                    human_overclaim = None

                eval_record = HumanEvaluation(
                    item_id=row['item_id'],
                    question=row['question'],
                    answer=row['answer'],
                    taxonomy=row['taxonomy'],
                    label=row['label'],
                    llm_avg_score=float(row['llm_avg_score']),
                    llm_consensus_judgment=row['llm_consensus_judgment'],
                    llm_overclaim_consensus=row['llm_overclaim_consensus'].lower() == 'true',
                    llm_agreement_score=int(row['llm_agreement_score']),
                    llm_is_valid=row['llm_is_valid'].lower() == 'true',
                    human_judgment=row.get('human_judgment', '').strip() or None,
                    human_is_overclaim=human_overclaim,
                    human_agrees_with_llm=human_agrees,
                    human_feedback=row.get('human_feedback', '').strip() or None,
                    human_evaluator_id=row.get('human_evaluator_id', '').strip() or None,
                    evaluation_timestamp=row.get('evaluation_timestamp', '').strip() or None
                )
                evaluations.append(eval_record)

        print(f"Loaded {len(evaluations)} human evaluations from: {filepath}")
        return evaluations

    def calculate_agreement_stats(
        self,
        evaluations: List[HumanEvaluation]
    ) -> Dict[str, Any]:
        """
        Calculate agreement statistics between human experts and LLM validators.

        Args:
            evaluations: List of human evaluations

        Returns:
            Dictionary with agreement statistics
        """
        # Filter evaluations that have human judgments
        evaluated = [e for e in evaluations if e.human_agrees_with_llm is not None]

        if not evaluated:
            return {'error': 'No human evaluations found'}

        total = len(evaluated)
        agreements = sum(1 for e in evaluated if e.human_agrees_with_llm == 1)
        disagreements = total - agreements

        # Calculate by taxonomy
        by_taxonomy = {}
        for taxonomy in ['S', 'C', 'R', 'M']:
            tax_evals = [e for e in evaluated if e.taxonomy == taxonomy]
            if tax_evals:
                tax_agreements = sum(1 for e in tax_evals if e.human_agrees_with_llm == 1)
                by_taxonomy[taxonomy] = {
                    'total': len(tax_evals),
                    'agreements': tax_agreements,
                    'disagreements': len(tax_evals) - tax_agreements,
                    'agreement_rate': tax_agreements / len(tax_evals)
                }

        # Calculate overclaim detection accuracy
        overclaim_evals = [e for e in evaluated if e.human_is_overclaim is not None]
        if overclaim_evals:
            overclaim_correct = sum(
                1 for e in overclaim_evals
                if e.llm_overclaim_consensus == e.human_is_overclaim
            )
            overclaim_accuracy = overclaim_correct / len(overclaim_evals)
        else:
            overclaim_accuracy = None

        return {
            'total_evaluated': total,
            'agreements': agreements,
            'disagreements': disagreements,
            'agreement_rate': agreements / total,
            'by_taxonomy': by_taxonomy,
            'overclaim_detection_accuracy': overclaim_accuracy,
            'human_evaluators': list(set(
                e.human_evaluator_id for e in evaluated if e.human_evaluator_id
            ))
        }

    def generate_feedback_for_generator(
        self,
        evaluations: List[HumanEvaluation]
    ) -> List[FeedbackRecord]:
        """
        Generate feedback records for the generator based on human disagreements.

        Items where human disagrees with LLM should be flagged for regeneration
        or used to improve prompt templates.

        Args:
            evaluations: List of human evaluations

        Returns:
            List of FeedbackRecord for generator improvement
        """
        feedback_records = []

        for eval_record in evaluations:
            # Only process where human disagrees with LLM
            if eval_record.human_agrees_with_llm == 0:
                # Determine issue type
                if eval_record.human_is_overclaim and not eval_record.llm_overclaim_consensus:
                    issue_type = "OVERCLAIM_MISSED"  # LLM missed an overclaim
                elif not eval_record.human_is_overclaim and eval_record.llm_overclaim_consensus:
                    issue_type = "FALSE_POSITIVE_OVERCLAIM"  # LLM wrongly flagged overclaim
                elif eval_record.human_judgment != eval_record.llm_consensus_judgment:
                    issue_type = "WRONG_CAUSAL_JUDGMENT"
                else:
                    issue_type = "OTHER"

                feedback = FeedbackRecord(
                    item_id=eval_record.item_id,
                    taxonomy=eval_record.taxonomy,
                    label=eval_record.label,
                    issue_type=issue_type,
                    original_answer=eval_record.answer,
                    corrected_judgment=eval_record.human_judgment or "UNKNOWN",
                    human_feedback=eval_record.human_feedback or "",
                    should_regenerate=(issue_type in ["OVERCLAIM_MISSED", "WRONG_CAUSAL_JUDGMENT"])
                )
                feedback_records.append(feedback)

        return feedback_records

    def export_feedback_for_generator(
        self,
        feedback_records: List[FeedbackRecord],
        filename: Optional[str] = None
    ) -> str:
        """
        Export feedback to CSV for generator improvement.

        Args:
            feedback_records: List of feedback records
            filename: Optional custom filename

        Returns:
            Path to the exported feedback CSV
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generator_feedback_{timestamp}.csv"

        filepath = self.output_dir / filename

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.feedback_columns)
            writer.writeheader()

            for record in feedback_records:
                writer.writerow(asdict(record))

        print(f"Exported {len(feedback_records)} feedback records: {filepath}")
        return str(filepath)

    def get_items_to_regenerate(
        self,
        feedback_records: List[FeedbackRecord]
    ) -> List[Dict[str, Any]]:
        """
        Get list of items that should be regenerated based on human feedback.

        Args:
            feedback_records: List of feedback records

        Returns:
            List of items to regenerate with context
        """
        to_regenerate = []

        for record in feedback_records:
            if record.should_regenerate:
                to_regenerate.append({
                    'item_id': record.item_id,
                    'taxonomy': record.taxonomy,
                    'label': record.label,
                    'issue_type': record.issue_type,
                    'corrected_judgment': record.corrected_judgment,
                    'human_feedback': record.human_feedback,
                    'improvement_hint': self._get_improvement_hint(record)
                })

        return to_regenerate

    def _get_improvement_hint(self, record: FeedbackRecord) -> str:
        """Generate improvement hint based on issue type."""
        hints = {
            "OVERCLAIM_MISSED": (
                "The answer overclaims causation. Ensure answer uses 'associated with' "
                "instead of 'causes' for GWAS-only evidence."
            ),
            "FALSE_POSITIVE_OVERCLAIM": (
                "The answer was incorrectly flagged as overclaim. "
                "Review if causal language is appropriate for the evidence type."
            ),
            "WRONG_CAUSAL_JUDGMENT": (
                f"LLM misjudged causal relationship. Correct judgment: {record.corrected_judgment}. "
                "Review answer for causal language accuracy."
            ),
            "OTHER": (
                f"Human feedback: {record.human_feedback}"
            )
        }
        return hints.get(record.issue_type, "Review and improve answer quality.")

    def create_summary_report(
        self,
        evaluations: List[HumanEvaluation],
        feedback_records: List[FeedbackRecord]
    ) -> str:
        """
        Create a summary report of human evaluation results.

        Args:
            evaluations: List of human evaluations
            feedback_records: List of feedback records

        Returns:
            Summary report as string
        """
        stats = self.calculate_agreement_stats(evaluations)

        report = []
        report.append("=" * 70)
        report.append("HUMAN EXPERT EVALUATION SUMMARY REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report.append("\n## Overall Agreement Statistics")
        report.append(f"Total items evaluated: {stats.get('total_evaluated', 0)}")
        report.append(f"Human-LLM Agreements: {stats.get('agreements', 0)}")
        report.append(f"Human-LLM Disagreements: {stats.get('disagreements', 0)}")
        report.append(f"Agreement Rate: {stats.get('agreement_rate', 0):.2%}")

        if stats.get('overclaim_detection_accuracy') is not None:
            report.append(f"Overclaim Detection Accuracy: {stats['overclaim_detection_accuracy']:.2%}")

        report.append("\n## Agreement by Taxonomy")
        for taxonomy, tax_stats in stats.get('by_taxonomy', {}).items():
            report.append(f"  {taxonomy}: {tax_stats['agreement_rate']:.2%} "
                         f"({tax_stats['agreements']}/{tax_stats['total']})")

        report.append("\n## Feedback for Generator")
        report.append(f"Total feedback items: {len(feedback_records)}")
        report.append(f"Items to regenerate: {sum(1 for r in feedback_records if r.should_regenerate)}")

        # Issue type breakdown
        issue_counts = {}
        for record in feedback_records:
            issue_counts[record.issue_type] = issue_counts.get(record.issue_type, 0) + 1

        report.append("\n## Issue Type Breakdown")
        for issue_type, count in issue_counts.items():
            report.append(f"  {issue_type}: {count}")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


def run_human_evaluation_workflow(
    items: List[Dict[str, Any]],
    validation_results: List[ValidationResult],
    output_dir: str = "./human_evaluation"
) -> Tuple[str, str]:
    """
    Run the complete human evaluation workflow.

    Step 1: Export validation results to CSV for human review
    Step 2: (Manual) Human expert fills in the evaluation columns
    Step 3: Load completed evaluations and generate feedback

    Args:
        items: Original Q&A items
        validation_results: Results from MultiLLMValidator
        output_dir: Directory for evaluation files

    Returns:
        Tuple of (review_csv_path, instructions)
    """
    evaluator = HumanExpertEvaluator(output_dir=output_dir)

    # Step 1: Export for human review
    review_path = evaluator.export_for_human_review(items, validation_results)

    instructions = f"""
================================================================================
HUMAN EXPERT EVALUATION WORKFLOW
================================================================================

Step 1: COMPLETE - Validation results exported to:
        {review_path}

Step 2: MANUAL - Open the CSV file and fill in these columns:

        | Column                | Values                              |
        |-----------------------|-------------------------------------|
        | human_judgment        | ASSOCIATIVE / CAUSAL / NOT_APPLICABLE |
        | human_is_overclaim    | TRUE / FALSE                        |
        | human_agrees_with_llm | 1 (Agree) / 0 (Disagree)            |
        | human_feedback        | Free text explanation               |
        | human_evaluator_id    | Your identifier                     |
        | evaluation_timestamp  | YYYY-MM-DD HH:MM:SS                 |

Step 3: After completing the review, run:

        from bioreasonc_creator.human_exp_evaluator import HumanExpertEvaluator

        evaluator = HumanExpertEvaluator(output_dir="{output_dir}")
        evaluations = evaluator.load_human_evaluations("{review_path}")
        feedback = evaluator.generate_feedback_for_generator(evaluations)
        evaluator.export_feedback_for_generator(feedback)

        # Get items to regenerate
        to_regenerate = evaluator.get_items_to_regenerate(feedback)

        # Print summary report
        report = evaluator.create_summary_report(evaluations, feedback)
        print(report)

================================================================================
"""

    print(instructions)
    return review_path, instructions


# Example usage and CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Human Expert Evaluator for BioREASONC")
    parser.add_argument("--action", choices=["export", "load", "feedback", "report"],
                       required=True, help="Action to perform")
    parser.add_argument("--input", help="Input CSV file path")
    parser.add_argument("--output-dir", default="./human_evaluation",
                       help="Output directory for evaluation files")

    args = parser.parse_args()

    evaluator = HumanExpertEvaluator(output_dir=args.output_dir)

    if args.action == "load" and args.input:
        evaluations = evaluator.load_human_evaluations(args.input)
        stats = evaluator.calculate_agreement_stats(evaluations)
        print(json.dumps(stats, indent=2))

    elif args.action == "feedback" and args.input:
        evaluations = evaluator.load_human_evaluations(args.input)
        feedback = evaluator.generate_feedback_for_generator(evaluations)
        evaluator.export_feedback_for_generator(feedback)

    elif args.action == "report" and args.input:
        evaluations = evaluator.load_human_evaluations(args.input)
        feedback = evaluator.generate_feedback_for_generator(evaluations)
        report = evaluator.create_summary_report(evaluations, feedback)
        print(report)

    else:
        print("Invalid action or missing input file. Use --help for usage.")
