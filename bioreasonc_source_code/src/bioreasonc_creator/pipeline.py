"""
BioREASONC Pipeline Orchestrator

================================================================================
MODULE OVERVIEW
================================================================================

Complete pipeline for generating, validating, and evaluating biomedical Q&A pairs.
This module orchestrates ALL components into a unified workflow.

Focus: "Does the model tell the truth about causality when explaining biomedical research?"

================================================================================
PIPELINE ARCHITECTURE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          BioREASONC PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│   │   CAUSALdb2 │ ──▶ │  GENERATOR  │ ──▶ │  VALIDATOR  │ ──▶ │  EXPLAINER  │  │
│   │     KG      │     │  (Step 1)   │     │  (Step 2)   │     │  (Step 3)   │  │
│   └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘  │
│                                                  │                  │          │
│                                                  ▼                  ▼          │
│                                           ┌─────────────┐     ┌─────────────┐  │
│                                           │   INVALID   │     │ PARAPHRASER │  │
│                                           │  (Rejected) │     │  (Step 4)   │  │
│                                           └─────────────┘     └─────────────┘  │
│                                                                     │          │
│                                                                     ▼          │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐  │
│   │   METRICS   │ ◀── │  EVALUATOR  │ ◀── │  EXPORTER   │ ◀── │   FINAL     │  │
│   │   (Step 7)  │     │  (Step 6)   │     │  (Step 5)   │     │  BENCHMARK  │  │
│   └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

STEP-BY-STEP FLOW:
┌──────┬───────────────────────────────────────────────────────────────────────────┐
│ Step │ Description                                                              │
├──────┼───────────────────────────────────────────────────────────────────────────┤
│  1   │ GENERATOR: Creates Q&A pairs from CAUSALdb2 gene-disease relationships   │
│      │ Input: KG with 66,057 pairs, evidence scores                            │
│      │ Output: KGGeneratedItem with question, answer, ground_truth              │
├──────┼───────────────────────────────────────────────────────────────────────────┤
│  2   │ VALIDATOR: Multi-LLM quality check (OpenAI + Anthropic + Gemini)        │
│      │ Input: Generated items                                                   │
│      │ Output: Valid items (score >= 4.0), rejected items discarded            │
│      │ Special: Overclaim detection for C taxonomy                             │
├──────┼───────────────────────────────────────────────────────────────────────────┤
│  3   │ EXPLAINER: Adds educational context (~35 words)                          │
│      │ Input: Valid items                                                       │
│      │ Output: Items with explanation field                                     │
├──────┼───────────────────────────────────────────────────────────────────────────┤
│  4   │ PARAPHRASER: Generates 2-3 diverse question phrasings                   │
│      │ Input: Explained items                                                   │
│      │ Output: Original + paraphrased versions (3x expansion)                   │
│      │ Special: Entity preservation, causal framing maintained                  │
├──────┼───────────────────────────────────────────────────────────────────────────┤
│  5   │ EXPORTER: Splits into train/dev/test, saves to HuggingFace format       │
│      │ Input: All processed items                                               │
│      │ Output: train.json, dev.json, test.json, metadata.json                   │
├──────┼───────────────────────────────────────────────────────────────────────────┤
│  6   │ EVALUATOR: Runs models on benchmark, computes CFS scores                │
│      │ Input: test.json + model predictions                                     │
│      │ Output: Evaluation metrics, overclaim rates, CFS scores                  │
├──────┼───────────────────────────────────────────────────────────────────────────┤
│  7   │ METRICS: Aggregates results, generates reports                          │
│      │ Input: Evaluation results                                                │
│      │ Output: Benchmark leaderboard, analysis reports                          │
└──────┴───────────────────────────────────────────────────────────────────────────┘

================================================================================
CONFIGURATION
================================================================================

PipelineConfig Parameters:
┌──────────────────────────────┬─────────────────┬─────────────────────────────────┐
│ Parameter                    │ Default         │ Description                     │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ API KEYS                                                                         │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ openai_api_key               │ None            │ For validation/explanation      │
│ anthropic_api_key            │ None            │ For validation/explanation      │
│ gemini_api_key               │ None            │ For validation                  │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ GENERATOR SETTINGS                                                               │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ use_cot_answers              │ True            │ Chain-of-Thought answers        │
│ kg_path                      │ None            │ Path to CAUSALdb2 CSV           │
│ use_kg_generator             │ True            │ Use KG-based generation         │
│ n_per_taxonomy               │ 100             │ Items per taxonomy (S,C,R,M)    │
│ stratify_by_evidence         │ True            │ Balance by evidence level       │
│ include_comparisons          │ True            │ Include comparison questions    │
│ include_mr_focused           │ True            │ Include MR-focused items        │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ PARAPHRASER SETTINGS                                                             │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ num_paraphrases              │ 2               │ Paraphrases per question (1-3)  │
│ use_llm_paraphrase           │ True            │ Use LLM for paraphrasing        │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ EXPLAINER SETTINGS                                                               │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ target_explanation_words     │ 35              │ Target word count               │
│ use_llm_explanation          │ True            │ Use LLM for explanation         │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ VALIDATOR SETTINGS                                                               │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ passing_threshold            │ 4.0             │ Min score to pass (1-5)         │
│ require_majority             │ True            │ Majority must pass              │
│ min_validators               │ 2               │ Minimum validators needed       │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ OUTPUT SETTINGS                                                                  │
├──────────────────────────────┼─────────────────┼─────────────────────────────────┤
│ output_dir                   │ ./bioreasonc_.. │ Output directory                │
└──────────────────────────────┴─────────────────┴─────────────────────────────────┘

================================================================================
OUTPUT FILES
================================================================================

The pipeline generates the following outputs:

┌─────────────────────────────────┬────────────────────────────────────────────────┐
│ File                            │ Description                                    │
├─────────────────────────────────┼────────────────────────────────────────────────┤
│ train.json                      │ 70% of items, for model training               │
│ dev.json                        │ 15% of items, for validation/tuning            │
│ test.json                       │ 15% of items, for final evaluation             │
│ metadata.json                   │ Benchmark statistics, generation config        │
│ pipeline_report.json            │ Full pipeline execution report                 │
│ validation_stats.json           │ Validation statistics per taxonomy             │
│ rejected_items.json             │ Items that failed validation (for analysis)    │
└─────────────────────────────────┴────────────────────────────────────────────────┘

================================================================================
USAGE EXAMPLES
================================================================================

Example 1: Full Pipeline Execution
```python
from bioreasonc_creator.pipeline import BioREASONCPipeline, PipelineConfig
import os

config = PipelineConfig(
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    kg_path="data/causaldb2_gene_disease.csv",
    n_per_taxonomy=200,
    output_dir="./benchmark_output"
)

pipeline = BioREASONCPipeline(config)
report = pipeline.run()

print(f"Generated: {report['total_generated']}")
print(f"Validated: {report['total_validated']}")
print(f"Exported: {report['total_exported']}")
```

Example 2: Step-by-Step Execution
```python
pipeline = BioREASONCPipeline(config)

# Step 1: Generate
generated = pipeline.generate()

# Step 2: Validate
valid, invalid = pipeline.validate(generated)

# Step 3: Explain
explained = pipeline.explain(valid)

# Step 4: Paraphrase
paraphrased = pipeline.paraphrase(explained)

# Step 5: Export
pipeline.export(paraphrased)
```

Example 3: Resume from Checkpoint
```python
pipeline = BioREASONCPipeline(config)
pipeline.load_checkpoint("checkpoint_validated.json")
report = pipeline.run(skip_until="explain")
```

================================================================================
BEST PRACTICES
================================================================================

★ API KEYS:
  • Provide at least 2 LLM keys for validation consensus
  • All 3 (OpenAI + Anthropic + Gemini) recommended

★ KNOWLEDGE GRAPH:
  • Use official CAUSALdb2 data for reproducibility
  • Enable stratify_by_evidence for balanced benchmark

★ QUALITY CONTROL:
  • Review rejected_items.json to understand validation failures
  • Adjust passing_threshold if too many items rejected

★ REPRODUCIBILITY:
  • Save pipeline_report.json with all configuration
  • Use fixed random seed for deterministic splits

================================================================================
Author: Sakhaa Alsaedi, sakhaa.alsaedi@kaust.edu.sa
================================================================================
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import os
from datetime import datetime

# Import pipeline components
from .generator import QuestionGenerator, GeneratedItem
from .kg_ingest import (
    CAUSALdbKnowledgeGraph,
    KGQuestionGenerator,
    KGGeneratedItem,
    EvidenceLevel,
    load_causaldb_kg
)
from .paraphraser import QuestionParaphraser
from .explainer import ExplanationGenerator
from .validator import MultiLLMValidator, ValidationResult, ValidatorConfig
from .human_exp_evaluator import HumanExpertEvaluator, FeedbackRecord
from .benchmark_exporter import BenchmarkExporter


@dataclass
class PipelineConfig:
    """Configuration for the BioREASONC pipeline."""
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

    # Generator settings
    use_cot_answers: bool = True  # Use Chain-of-Thought answers

    # Knowledge Graph settings
    kg_path: Optional[str] = None  # Path to CAUSALdb2 KG CSV
    use_kg_generator: bool = True  # Use KG-based generation
    n_per_taxonomy: int = 100  # Number of items per taxonomy (S, C, R, M)
    stratify_by_evidence: bool = True  # Stratify by evidence level
    include_comparisons: bool = True  # Include comparison questions
    include_mr_focused: bool = True  # Include MR-focused benchmark items

    # Paraphraser settings
    num_paraphrases: int = 2  # Number of paraphrases per question
    use_llm_paraphrase: bool = True

    # Explainer settings
    target_explanation_words: int = 35
    use_llm_explanation: bool = True

    # Validator settings
    passing_threshold: float = 4.0
    require_majority: bool = True
    min_validators: int = 2

    # Output settings
    output_dir: str = "./bioreasonc_output"


class BioREASONCPipeline:
    """
    Orchestrates the complete BioREASONC pipeline.

    Pipeline:
        Generator → Paraphraser → Explainer → Validator → Human Expert → Feedback
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components (lazy loading)
        self._generator = None
        self._kg = None
        self._kg_generator = None
        self._paraphraser = None
        self._explainer = None
        self._validator = None
        self._human_evaluator = None

    @property
    def kg(self) -> CAUSALdbKnowledgeGraph:
        """Get or create the knowledge graph."""
        if self._kg is None:
            kg_path = self.config.kg_path
            if kg_path is None:
                # Default CAUSALdb2 path
                kg_path = "/ibex/user/alsaedsb/ROCKET/Data/CAUSALdb2/v2.1/kg/gene_disease_kg_corrected.csv"
            self._kg = load_causaldb_kg(kg_path)
        return self._kg

    @property
    def kg_generator(self) -> KGQuestionGenerator:
        """Get or create the KG question generator."""
        if self._kg_generator is None:
            self._kg_generator = KGQuestionGenerator(self.kg)
        return self._kg_generator

    @property
    def generator(self) -> QuestionGenerator:
        """Get or create the question generator."""
        if self._generator is None:
            self._generator = QuestionGenerator()
        return self._generator

    @property
    def paraphraser(self) -> QuestionParaphraser:
        """Get or create the paraphraser."""
        if self._paraphraser is None:
            self._paraphraser = QuestionParaphraser(
                openai_api_key=self.config.openai_api_key,
                anthropic_api_key=self.config.anthropic_api_key,
                num_paraphrases=self.config.num_paraphrases,
                use_llm=self.config.use_llm_paraphrase
            )
        return self._paraphraser

    @property
    def explainer(self) -> ExplanationGenerator:
        """Get or create the explainer."""
        if self._explainer is None:
            self._explainer = ExplanationGenerator(
                openai_api_key=self.config.openai_api_key,
                anthropic_api_key=self.config.anthropic_api_key,
                target_words=self.config.target_explanation_words,
                use_llm=self.config.use_llm_explanation
            )
        return self._explainer

    @property
    def validator(self) -> MultiLLMValidator:
        """Get or create the validator."""
        if self._validator is None:
            validator_config = ValidatorConfig(
                passing_threshold=self.config.passing_threshold,
                require_majority=self.config.require_majority,
                min_validators=self.config.min_validators
            )
            self._validator = MultiLLMValidator(
                openai_api_key=self.config.openai_api_key,
                anthropic_api_key=self.config.anthropic_api_key,
                gemini_api_key=self.config.gemini_api_key,
                config=validator_config
            )
        return self._validator

    @property
    def human_evaluator(self) -> HumanExpertEvaluator:
        """Get or create the human evaluator."""
        if self._human_evaluator is None:
            self._human_evaluator = HumanExpertEvaluator(
                output_dir=str(self.output_dir / "human_evaluation")
            )
        return self._human_evaluator

    def step0_generate_from_kg(
        self,
        n_per_taxonomy: Optional[int] = None,
        stratify_by_evidence: Optional[bool] = None,
        include_comparisons: Optional[bool] = None,
        include_mr_focused: Optional[bool] = None,
        seed: int = 42,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Step 0: Generate Q&A pairs from CAUSALdb2 Knowledge Graph.

        This is the PRIMARY generation method using the 66K+ gene-disease pairs
        from CAUSALdb2 with evidence scores (MR, causal confidence, GO, etc.).

        Args:
            n_per_taxonomy: Items per taxonomy (default: config.n_per_taxonomy)
            stratify_by_evidence: Stratify by evidence level (default: config value)
            include_comparisons: Include gene comparison questions (default: config value)
            include_mr_focused: Include MR-focused items (default: config value)
            seed: Random seed for reproducibility
            progress_callback: Optional progress callback

        Returns:
            List of generated Q&A items in pipeline format
        """
        print("\n" + "=" * 60)
        print("STEP 0: KG GENERATOR - Creating Q&A from CAUSALdb2")
        print("=" * 60)

        # Use config values as defaults
        n_per_taxonomy = n_per_taxonomy or self.config.n_per_taxonomy
        stratify_by_evidence = stratify_by_evidence if stratify_by_evidence is not None else self.config.stratify_by_evidence
        include_comparisons = include_comparisons if include_comparisons is not None else self.config.include_comparisons
        include_mr_focused = include_mr_focused if include_mr_focused is not None else self.config.include_mr_focused

        # Show KG statistics
        stats = self.kg.get_statistics()
        print(f"\nKnowledge Graph Statistics:")
        print(f"  Total gene-disease pairs: {stats['total_pairs']:,}")
        print(f"  Unique genes: {stats['unique_genes']:,}")
        print(f"  Unique diseases: {stats['unique_diseases']:,}")
        print(f"  MR-validated pairs: {stats['mr_validated_pairs']:,}")
        print(f"\nEvidence distribution:")
        for level, count in stats['evidence_distribution'].items():
            pct = count / stats['total_pairs'] * 100
            print(f"    {level}: {count:,} ({pct:.1f}%)")

        # Generate benchmark items
        print(f"\nGenerating {n_per_taxonomy} items per taxonomy...")
        kg_items = self.kg_generator.generate_benchmark(
            n_per_taxonomy=n_per_taxonomy,
            stratify_by_evidence=stratify_by_evidence,
            include_comparisons=include_comparisons,
            seed=seed
        )

        # Add MR-focused items if requested
        if include_mr_focused:
            print("Adding MR-focused items...")
            mr_items = self.kg_generator.generate_mr_focused_benchmark(
                n=min(100, n_per_taxonomy // 2)
            )
            kg_items.extend(mr_items)

        # Get generation statistics
        gen_stats = self.kg_generator.get_statistics(kg_items)
        print(f"\nGeneration Statistics:")
        print(f"  Total items: {gen_stats['total']}")
        print(f"  By taxonomy: {gen_stats['by_taxonomy']}")
        print(f"  By evidence level: {gen_stats['by_evidence_level']}")
        print(f"  By difficulty: {gen_stats['by_difficulty']}")

        # Convert to pipeline format
        items = []
        for kg_item in kg_items:
            item_dict = {
                'id': kg_item.id,
                'question': kg_item.question,
                'answer': kg_item.answer,
                'taxonomy': kg_item.taxonomy,
                'label': kg_item.label,
                'template_id': kg_item.template_id,
                'answer_type': kg_item.answer_type,
                'entities': kg_item.entities,
                'ground_truth': kg_item.ground_truth,
                'difficulty': kg_item.difficulty,
                'evidence_level': kg_item.evidence_level,
                'source_data': kg_item.source_pair,
                'explanation': '',  # Will be filled by explainer
                'metadata': {
                    'data_source': 'CAUSALdb2',
                    'kg_version': 'v2.1'
                }
            }
            items.append(item_dict)

            if progress_callback and len(items) % 100 == 0:
                progress_callback(len(items), gen_stats['total'], "Generating from KG")

        print(f"\nGenerated {len(items)} Q&A items from KG")
        return items

    def step1_generate(
        self,
        gwas_data: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Generate Q&A pairs from GWAS data.

        Args:
            gwas_data: List of GWAS records
            progress_callback: Optional progress callback

        Returns:
            List of generated Q&A items
        """
        print("\n" + "=" * 60)
        print("STEP 1: GENERATOR - Creating Q&A pairs")
        print("=" * 60)

        import pandas as pd
        items = []
        total = len(gwas_data)

        for idx, record in enumerate(gwas_data):
            # Convert record to DataFrame format expected by generator
            single_df = pd.DataFrame([{
                'rsid': record.get('rsid', record.get('SNPS', '')),
                'gene': record.get('gene', record.get('MAPPED_GENE', '')),
                'chromosome': record.get('chromosome', record.get('CHR_ID', '')),
                'OR': record.get('or_value', record.get('OR or BETA', 1.0)),
                'P-Value': record.get('p_value', record.get('P-VALUE', 0.05)),
            }])

            disease = record.get('disease', record.get('DISEASE/TRAIT', 'Unknown Disease'))
            generated = self.generator.generate_from_dataframe(single_df, disease=disease)

            for item in generated:
                # Handle ground_truth - could be dict, object with __dict__, or None
                if item.ground_truth is None:
                    ground_truth = None
                elif isinstance(item.ground_truth, dict):
                    ground_truth = item.ground_truth
                elif hasattr(item.ground_truth, '__dict__'):
                    ground_truth = item.ground_truth.__dict__
                else:
                    ground_truth = str(item.ground_truth)

                items.append({
                    'id': item.id,
                    'question': item.question,
                    'answer': item.answer,
                    'taxonomy': item.taxonomy.value if hasattr(item.taxonomy, 'value') else item.taxonomy,
                    'label': item.label,
                    'ground_truth': ground_truth,
                    'metadata': item.metadata if hasattr(item, 'metadata') else {},
                    'explanation': ''
                })

            if progress_callback:
                progress_callback(idx + 1, total, "Generating")

        print(f"Generated {len(items)} Q&A items from {total} GWAS records")
        return items

    def step2_paraphrase(
        self,
        items: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Step 2: Generate paraphrases for questions.

        Args:
            items: List of Q&A items
            progress_callback: Optional progress callback

        Returns:
            List of items with paraphrases added
        """
        print("\n" + "=" * 60)
        print("STEP 2: PARAPHRASER - Generating question variations")
        print("=" * 60)

        paraphrased_items = self.paraphraser.paraphrase_batch(
            items,
            progress_callback=progress_callback
        )

        original_count = sum(1 for i in paraphrased_items if not i.get('paraphrased', False))
        para_count = sum(1 for i in paraphrased_items if i.get('paraphrased', False))

        print(f"Original items: {original_count}, Paraphrased items: {para_count}")
        print(f"Total items: {len(paraphrased_items)}")
        return paraphrased_items

    def step3_explain(
        self,
        items: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Step 3: Add explanations to Q&A pairs.

        Args:
            items: List of Q&A items
            progress_callback: Optional progress callback

        Returns:
            List of items with explanations added
        """
        print("\n" + "=" * 60)
        print("STEP 3: EXPLAINER - Adding explanations")
        print("=" * 60)

        explained_items = self.explainer.add_explanations(
            items,
            progress_callback=progress_callback
        )

        stats = self.explainer.get_stats(explained_items)
        print(f"Average explanation length: {stats['avg_word_count']:.1f} words")
        print(f"Method distribution: {stats['method_distribution']}")
        return explained_items

    def step4_validate(
        self,
        items: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[ValidationResult]]:
        """
        Step 4: Validate items using multi-LLM consensus.

        Args:
            items: List of Q&A items with explanations
            progress_callback: Optional progress callback

        Returns:
            Tuple of (valid_items, invalid_items, validation_results)
        """
        print("\n" + "=" * 60)
        print("STEP 4: VALIDATOR - Multi-LLM consensus validation")
        print("=" * 60)

        valid_items, invalid_items = self.validator.validate_batch(
            items,
            progress_callback=progress_callback
        )

        # Collect validation results for human review
        validation_results = []
        for item in valid_items + invalid_items:
            result = ValidationResult(
                item_id=item.get('id', ''),
                scores={p: item.get('validation_scores', [])[i]
                       for i, p in enumerate(item.get('validation_providers', []))},
                avg_score=item.get('avg_score', 0),
                is_valid=item.get('is_valid', False),
                feedback=item.get('validation_feedback', {}),
                ground_truth_match=item.get('ground_truth_match'),
                agreement_score=item.get('agreement_score', 0),
                consensus_judgment=item.get('consensus_judgment', 'NOT_APPLICABLE'),
                overclaim_consensus=item.get('overclaim_consensus', False),
                causal_judgments=item.get('causal_judgments', {}),
                evidence=item.get('evidence', {})
            )
            validation_results.append(result)

        stats = self.validator.get_stats(valid_items, invalid_items)
        print(f"Valid items: {stats['valid_items']}/{stats['total_items']} ({stats['validation_rate']:.1%})")
        print(f"Average score: {stats['avg_score']:.2f}")

        return valid_items, invalid_items, validation_results

    def step5_export_for_human_review(
        self,
        items: List[Dict[str, Any]],
        validation_results: List[ValidationResult],
        filename: Optional[str] = None
    ) -> str:
        """
        Step 5: Export for human expert review.

        Args:
            items: All items (valid and invalid)
            validation_results: Validation results
            filename: Optional custom filename

        Returns:
            Path to exported CSV file
        """
        print("\n" + "=" * 60)
        print("STEP 5: HUMAN EXPERT EVALUATOR - Export for review")
        print("=" * 60)

        csv_path = self.human_evaluator.export_for_human_review(
            items, validation_results, filename
        )

        print(f"\nExported to: {csv_path}")
        print("\n*** MANUAL STEP REQUIRED ***")
        print("Human expert must review the CSV and fill in:")
        print("  - human_judgment: ASSOCIATIVE / CAUSAL / NOT_APPLICABLE")
        print("  - human_is_overclaim: TRUE / FALSE")
        print("  - human_agrees_with_llm: 1 (Agree) / 0 (Disagree)")
        print("  - human_feedback: Free text explanation")

        return csv_path

    def step6_process_human_feedback(
        self,
        human_review_csv: str
    ) -> Tuple[List[FeedbackRecord], List[Dict[str, Any]]]:
        """
        Step 6: Process human feedback and generate improvement suggestions.

        Args:
            human_review_csv: Path to completed human review CSV

        Returns:
            Tuple of (feedback_records, items_to_regenerate)
        """
        print("\n" + "=" * 60)
        print("STEP 6: FEEDBACK LOOP - Processing human evaluations")
        print("=" * 60)

        # Load human evaluations
        evaluations = self.human_evaluator.load_human_evaluations(human_review_csv)

        # Calculate agreement statistics
        stats = self.human_evaluator.calculate_agreement_stats(evaluations)
        print(f"\nHuman-LLM Agreement Rate: {stats.get('agreement_rate', 0):.1%}")

        # Generate feedback for generator
        feedback = self.human_evaluator.generate_feedback_for_generator(evaluations)
        print(f"Feedback items: {len(feedback)}")

        # Export feedback
        self.human_evaluator.export_feedback_for_generator(feedback)

        # Get items to regenerate
        to_regenerate = self.human_evaluator.get_items_to_regenerate(feedback)
        print(f"Items to regenerate: {len(to_regenerate)}")

        # Print summary report
        report = self.human_evaluator.create_summary_report(evaluations, feedback)
        print(report)

        return feedback, to_regenerate

    def step7_regenerate_with_feedback(
        self,
        feedback_items: List[Dict[str, Any]],
        original_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Step 7: Regenerate items based on human feedback.

        This step takes the feedback from human experts and regenerates
        items that were incorrectly generated (overclaims, wrong judgments, etc.).

        Args:
            feedback_items: Items to regenerate from step6
            original_items: Original items for reference

        Returns:
            List of regenerated items
        """
        print("\n" + "=" * 60)
        print("STEP 7: REGENERATION - Applying human feedback")
        print("=" * 60)

        if not feedback_items:
            print("No items to regenerate.")
            return []

        # Use generator's regenerate_with_feedback method
        regenerated = self.generator.regenerate_with_feedback(
            feedback_items, original_items
        )

        print(f"Regenerated {len(regenerated)} items with feedback corrections")

        # Convert to dict format
        regenerated_dicts = []
        for item in regenerated:
            regenerated_dicts.append({
                'id': item.id,
                'question': item.question,
                'answer': item.answer,
                'taxonomy': item.taxonomy,
                'label': item.label,
                'explanation': '',  # Will be re-explained
                'source_data': item.source_data
            })

        # Get feedback summary
        summary = self.generator.get_feedback_summary(feedback_items)
        print("\nFeedback Summary:")
        print(f"  Total feedback items: {summary['total_feedback']}")
        for issue, count in summary['by_issue_type'].items():
            print(f"  - {issue}: {count}")

        if summary['recommendations']:
            print("\nRecommendations for template improvement:")
            for rec in summary['recommendations']:
                print(f"  • {rec}")

        return regenerated_dicts

    def run_feedback_loop(
        self,
        human_review_csv: str,
        original_items: List[Dict[str, Any]],
        iteration: int = 1
    ) -> Dict[str, Any]:
        """
        Run ONE iteration of the feedback loop after human review.

        Args:
            human_review_csv: Path to completed human review CSV
            original_items: Original items that were reviewed
            iteration: Current iteration number (for tracking)

        Returns:
            Dictionary with feedback loop results
        """
        print("\n" + "=" * 70)
        print(f"BIOREASONC FEEDBACK LOOP - ITERATION {iteration}")
        print("=" * 70)

        # Step 6: Process human feedback
        feedback, to_regenerate = self.step6_process_human_feedback(human_review_csv)

        # Step 7: Regenerate with feedback
        regenerated_items = self.step7_regenerate_with_feedback(
            to_regenerate, original_items
        )

        if not regenerated_items:
            print("\nNo items to regenerate - feedback loop complete.")
            return {
                'iteration': iteration,
                'feedback_count': len(feedback),
                'regenerated_count': 0,
                'status': 'complete',
                'needs_another_round': False
            }

        # Re-run explanation on regenerated items
        print("\n" + "-" * 40)
        print("Re-explaining regenerated items...")
        regenerated_items = self.step3_explain(regenerated_items)

        # Re-validate regenerated items
        print("\n" + "-" * 40)
        print("Re-validating regenerated items...")
        valid_regen, invalid_regen, validation_results = self.step4_validate(
            regenerated_items
        )

        # Export for next human review if needed
        if invalid_regen:
            next_csv = self.step5_export_for_human_review(
                invalid_regen,
                [r for r in validation_results if not r.is_valid],
                f"human_review_iteration_{iteration + 1}.csv"
            )
        else:
            next_csv = None

        results = {
            'iteration': iteration,
            'feedback_count': len(feedback),
            'regenerated_count': len(regenerated_items),
            'valid_after_regen': len(valid_regen),
            'invalid_after_regen': len(invalid_regen),
            'improvement_rate': len(valid_regen) / len(regenerated_items) if regenerated_items else 0,
            'status': 'complete' if not invalid_regen else 'needs_review',
            'needs_another_round': len(invalid_regen) > 0,
            'next_review_csv': next_csv,
            'regenerated_items': regenerated_items
        }

        print("\n" + "=" * 70)
        print(f"FEEDBACK LOOP ITERATION {iteration} COMPLETE")
        print("=" * 70)
        print(f"Items regenerated: {results['regenerated_count']}")
        print(f"Valid after regeneration: {results['valid_after_regen']}")
        print(f"Still invalid: {results['invalid_after_regen']}")

        return results

    def run_iterative_feedback_loop(
        self,
        initial_human_review_csv: str,
        original_items: List[Dict[str, Any]],
        max_iterations: int = 3,
        target_agreement_rate: float = 0.95
    ) -> Dict[str, Any]:
        """
        Run the complete iterative feedback loop with stopping conditions.

        The loop will stop when:
        1. Maximum iterations reached (default: 3)
        2. Target agreement rate achieved (default: 95%)
        3. No more items need regeneration

        Args:
            initial_human_review_csv: Path to first human review CSV
            original_items: Original items for reference
            max_iterations: Maximum number of feedback iterations (default: 3)
            target_agreement_rate: Stop when this agreement rate is reached (default: 0.95)

        Returns:
            Dictionary with complete loop results
        """
        print("\n" + "=" * 70)
        print("BIOREASONC ITERATIVE FEEDBACK LOOP")
        print("=" * 70)
        print(f"Max iterations: {max_iterations}")
        print(f"Target agreement rate: {target_agreement_rate:.0%}")
        print("=" * 70)

        all_iterations = []
        current_csv = initial_human_review_csv
        current_items = original_items
        already_regenerated = set()  # Track items to avoid infinite loops

        for iteration in range(1, max_iterations + 1):
            print(f"\n>>> ITERATION {iteration}/{max_iterations}")

            # Check if CSV exists (manual step required)
            if not os.path.exists(current_csv):
                print(f"\n⚠️  MANUAL STEP REQUIRED")
                print(f"Human expert must review: {current_csv}")
                print("Run this method again after completing the review.")
                return {
                    'status': 'waiting_for_human_review',
                    'current_iteration': iteration,
                    'pending_review_csv': current_csv,
                    'completed_iterations': all_iterations
                }

            # Run one feedback iteration
            result = self.run_feedback_loop(
                current_csv, current_items, iteration
            )
            all_iterations.append(result)

            # Track regenerated items
            for item in result.get('regenerated_items', []):
                item_id = item.get('id', '').replace('-REGEN', '')
                if item_id in already_regenerated:
                    print(f"⚠️  Item {item_id} already regenerated - skipping to avoid loop")
                    continue
                already_regenerated.add(item_id)

            # Check stopping conditions
            if not result['needs_another_round']:
                print(f"\n✓ All items valid - stopping at iteration {iteration}")
                break

            # Calculate current agreement rate
            total_items = result['valid_after_regen'] + result['invalid_after_regen']
            if total_items > 0:
                current_rate = result['valid_after_regen'] / total_items
                if current_rate >= target_agreement_rate:
                    print(f"\n✓ Target agreement rate {target_agreement_rate:.0%} achieved")
                    break

            # Prepare for next iteration
            current_csv = result.get('next_review_csv')
            current_items = result.get('regenerated_items', [])

            if iteration < max_iterations:
                print(f"\n⚠️  MANUAL STEP REQUIRED for iteration {iteration + 1}")
                print(f"Human expert must review: {current_csv}")

        # Final summary
        total_regenerated = sum(r['regenerated_count'] for r in all_iterations)
        final_valid = all_iterations[-1]['valid_after_regen'] if all_iterations else 0

        final_results = {
            'status': 'complete',
            'total_iterations': len(all_iterations),
            'total_regenerated': total_regenerated,
            'final_valid_count': final_valid,
            'items_regenerated_ids': list(already_regenerated),
            'iterations': all_iterations
        }

        print("\n" + "=" * 70)
        print("ITERATIVE FEEDBACK LOOP COMPLETE")
        print("=" * 70)
        print(f"Total iterations: {final_results['total_iterations']}")
        print(f"Total items regenerated: {final_results['total_regenerated']}")
        print(f"Final valid count: {final_results['final_valid_count']}")

        return final_results

    def run_full_pipeline(
        self,
        gwas_data: List[Dict[str, Any]],
        skip_human_review: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline (Steps 1-5).

        Note: Step 6 requires manual human review before running.

        Args:
            gwas_data: List of GWAS records
            skip_human_review: If True, skip human review export

        Returns:
            Dictionary with pipeline results
        """
        print("\n" + "=" * 70)
        print("BIOREASONC PIPELINE - STARTING")
        print("=" * 70)
        print(f"Input: {len(gwas_data)} GWAS records")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Generate
        items = self.step1_generate(gwas_data)

        # Step 2: Paraphrase
        items = self.step2_paraphrase(items)

        # Step 3: Explain
        items = self.step3_explain(items)

        # Step 4: Validate
        valid_items, invalid_items, validation_results = self.step4_validate(items)

        # Step 5: Export for human review (optional)
        human_review_csv = None
        if not skip_human_review:
            all_items = valid_items + invalid_items
            human_review_csv = self.step5_export_for_human_review(
                all_items, validation_results
            )

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'input_records': len(gwas_data),
            'generated_items': len(items),
            'valid_items': len(valid_items),
            'invalid_items': len(invalid_items),
            'validation_rate': len(valid_items) / len(items) if items else 0,
            'human_review_csv': human_review_csv,
            'output_dir': str(self.output_dir)
        }

        # Save results summary
        summary_path = self.output_dir / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {summary_path}")

        # Store valid items for later export
        results['_valid_items'] = valid_items

        return results

    def step8_export_benchmark(
        self,
        items: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        dataset_name: str = "BioREASONC-Bench",
        version: str = "1.0.0",
        min_score: float = 4.0,
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """
        Step 8: Export benchmark to HuggingFace format.

        Creates train.json, dev.json, test.json splits with metadata.

        Args:
            items: Validated items to export
            output_dir: Output directory (default: self.output_dir/benchmark)
            dataset_name: Name of the dataset
            version: Version string
            min_score: Minimum validation score to include
            train_ratio: Proportion for training set
            dev_ratio: Proportion for development set
            test_ratio: Proportion for test set

        Returns:
            Dictionary with export results
        """
        print("\n" + "=" * 60)
        print("STEP 8: EXPORT BENCHMARK - Creating HuggingFace dataset")
        print("=" * 60)

        export_dir = output_dir or str(self.output_dir / "benchmark")

        exporter = BenchmarkExporter(
            output_dir=export_dir,
            dataset_name=dataset_name,
            version=version,
            min_score=min_score
        )

        results = exporter.export_benchmark(
            items,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
            test_ratio=test_ratio
        )

        return results

    def run_complete_pipeline(
        self,
        gwas_data: List[Dict[str, Any]],
        export_benchmark: bool = True,
        skip_human_review: bool = True,
        benchmark_output_dir: Optional[str] = None,
        dataset_name: str = "BioREASONC-Bench",
        version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline including benchmark export.

        This runs Steps 1-5 and Step 8 (skipping human review loop).
        For production use with human review, use run_full_pipeline() first,
        then run_feedback_loop(), then step8_export_benchmark().

        Args:
            gwas_data: List of GWAS records
            export_benchmark: If True, export to HuggingFace format
            skip_human_review: If True, skip human review step
            benchmark_output_dir: Output directory for benchmark files
            dataset_name: Name of the dataset
            version: Version string

        Returns:
            Dictionary with complete pipeline results
        """
        # Run main pipeline
        results = self.run_full_pipeline(gwas_data, skip_human_review=skip_human_review)

        # Export benchmark if requested
        if export_benchmark and results.get('_valid_items'):
            export_results = self.step8_export_benchmark(
                items=results['_valid_items'],
                output_dir=benchmark_output_dir,
                dataset_name=dataset_name,
                version=version
            )
            results['benchmark_export'] = export_results

        return results


    def run_kg_pipeline(
        self,
        n_per_taxonomy: int = 100,
        skip_human_review: bool = False,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run the pipeline using CAUSALdb2 Knowledge Graph as data source.

        This is the PRIMARY pipeline method for generating the BioREASONC benchmark.

        Pipeline flow:
            KG → Generate → Paraphrase → Explain → Validate → Human Review

        Args:
            n_per_taxonomy: Number of items per taxonomy (S, C, R, M)
            skip_human_review: If True, skip human review export
            seed: Random seed for reproducibility

        Returns:
            Dictionary with pipeline results
        """
        print("\n" + "=" * 70)
        print("BIOREASONC KG PIPELINE - STARTING")
        print("=" * 70)
        print(f"Data source: CAUSALdb2 Knowledge Graph")
        print(f"Items per taxonomy: {n_per_taxonomy}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 0: Generate from KG
        items = self.step0_generate_from_kg(
            n_per_taxonomy=n_per_taxonomy,
            seed=seed
        )

        # Step 2: Paraphrase
        items = self.step2_paraphrase(items)

        # Step 3: Explain
        items = self.step3_explain(items)

        # Step 4: Validate
        valid_items, invalid_items, validation_results = self.step4_validate(items)

        # Step 5: Export for human review (optional)
        human_review_csv = None
        if not skip_human_review:
            all_items = valid_items + invalid_items
            human_review_csv = self.step5_export_for_human_review(
                all_items, validation_results
            )

        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'CAUSALdb2',
            'n_per_taxonomy': n_per_taxonomy,
            'generated_items': len(items),
            'valid_items': len(valid_items),
            'invalid_items': len(invalid_items),
            'validation_rate': len(valid_items) / len(items) if items else 0,
            'human_review_csv': human_review_csv,
            'output_dir': str(self.output_dir)
        }

        # Save results summary
        summary_path = self.output_dir / f"kg_pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 70)
        print("KG PIPELINE COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {summary_path}")

        # Store valid items for later export
        results['_valid_items'] = valid_items

        return results

    def run_complete_kg_pipeline(
        self,
        n_per_taxonomy: int = 100,
        export_benchmark: bool = True,
        skip_human_review: bool = True,
        benchmark_output_dir: Optional[str] = None,
        dataset_name: str = "BioREASONC-Bench",
        version: str = "1.0.0",
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Run the complete KG pipeline including benchmark export.

        This is the recommended method for generating the full BioREASONC benchmark
        from CAUSALdb2 Knowledge Graph data.

        Args:
            n_per_taxonomy: Number of items per taxonomy
            export_benchmark: If True, export to HuggingFace format
            skip_human_review: If True, skip human review step
            benchmark_output_dir: Output directory for benchmark files
            dataset_name: Name of the dataset
            version: Version string
            seed: Random seed for reproducibility

        Returns:
            Dictionary with complete pipeline results
        """
        # Run main KG pipeline
        results = self.run_kg_pipeline(
            n_per_taxonomy=n_per_taxonomy,
            skip_human_review=skip_human_review,
            seed=seed
        )

        # Export benchmark if requested
        if export_benchmark and results.get('_valid_items'):
            export_results = self.step8_export_benchmark(
                items=results['_valid_items'],
                output_dir=benchmark_output_dir,
                dataset_name=dataset_name,
                version=version
            )
            results['benchmark_export'] = export_results

        return results


def print_pipeline_diagram():
    """Print the pipeline flow diagram."""
    diagram = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          BIOREASONC PIPELINE FLOW                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   ┌─────────────────┐                                                        ║
║   │   GWAS Data     │                                                        ║
║   │   (Input)       │                                                        ║
║   └────────┬────────┘                                                        ║
║            │                                                                 ║
║            ▼                                                                 ║
║   ┌─────────────────┐     Creates Q&A pairs with CoT answers                 ║
║   │ 1. Generator    │     for each taxonomy (S, C, R, M)                     ║
║   │  generator.py   │                                                        ║
║   └────────┬────────┘                                                        ║
║            │                                                                 ║
║            ▼                                                                 ║
║   ┌─────────────────┐     Generates 2-3 paraphrases per question             ║
║   │ 2. Paraphraser  │     while preserving entities                          ║
║   │ paraphraser.py  │                                                        ║
║   └────────┬────────┘                                                        ║
║            │                                                                 ║
║            ▼                                                                 ║
║   ┌─────────────────┐     Adds ~35 word explanations                         ║
║   │ 3. Explainer    │     for each Q&A pair                                  ║
║   │  explainer.py   │                                                        ║
║   └────────┬────────┘                                                        ║
║            │                                                                 ║
║            ▼                                                                 ║
║   ┌─────────────────┐     Multi-LLM validation:                              ║
║   │ 4. Validator    │     - OpenAI, Anthropic, Gemini                        ║
║   │  validator.py   │     - Consensus scoring (Agree=1, Disagree=0)          ║
║   └────────┬────────┘     - Overclaim detection                              ║
║            │                                                                 ║
║            ▼                                                                 ║
║   ┌─────────────────┐     Exports CSV for human review:                      ║
║   │ 5. Human Expert │     - human_judgment                                   ║
║   │ human_exp_eval  │     - human_is_overclaim                               ║
║   │     .py         │     - human_agrees_with_llm (1/0)                      ║
║   └────────┬────────┘                                                        ║
║            │                                                                 ║
║            ▼                                                                 ║
║   ┌─────────────────┐     Processes disagreements:                           ║
║   │ 6. Feedback     │     - Generates improvement hints                      ║
║   │    Loop         │     - Identifies items to regenerate                   ║
║   └────────┬────────┘                                                        ║
║            │                                                                 ║
║            └──────────────────────┐                                          ║
║                                   │                                          ║
║                                   ▼                                          ║
║                          ┌─────────────────┐                                 ║
║                          │   Back to       │                                 ║
║                          │   Generator     │                                 ║
║                          └─────────────────┘                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(diagram)


if __name__ == "__main__":
    print_pipeline_diagram()
