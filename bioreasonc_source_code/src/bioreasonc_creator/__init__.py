"""
BioREASONC-Creator: Biomedical Benchmark Generation Pipeline

A complete pipeline for generating high-quality biomedical reasoning questions:
- generator.py: Generate Q&A from SNP→Gene→Disease data with ground truth
- kg_ingest.py: CAUSALdb2 Knowledge Graph ingestion and question generation
- paraphraser.py: Create diverse paraphrases while preserving entities
- explainer.py: Add concise biomedical explanations
- validator.py: Multi-LLM quality validation
- human_exp_evaluator.py: Human expert evaluation interface
- pipeline.py: Pipeline orchestrator with feedback loop
- benchmark_exporter.py: Export to HuggingFace format (train/dev/test)

Core Question: "Does the model tell the truth about causality when explaining biomedical research?"

Data Sources:
- CAUSALdb2 v2.1: 66,057 gene-disease pairs with evidence scores
- Evidence levels: very_strong, strong, moderate, suggestive, weak
- MR (Mendelian Randomization) provides strong causal evidence
"""

from .generator import QuestionGenerator, GeneratedItem
from .kg_ingest import (
    CAUSALdbKnowledgeGraph,
    KGQuestionGenerator,
    KGGeneratedItem,
    GeneDiseasePair,
    EvidenceLevel,
    load_causaldb_kg
)
from .paraphraser import QuestionParaphraser
from .explainer import ExplanationGenerator
from .validator import MultiLLMValidator, ValidationResult
from .human_exp_evaluator import HumanExpertEvaluator, HumanEvaluation, FeedbackRecord
from .pipeline import BioREASONCPipeline, PipelineConfig
from .benchmark_exporter import BenchmarkExporter, export_to_huggingface

__all__ = [
    # Core components
    'QuestionGenerator',
    'GeneratedItem',
    'QuestionParaphraser',
    'ExplanationGenerator',
    'MultiLLMValidator',
    'ValidationResult',
    # Knowledge Graph
    'CAUSALdbKnowledgeGraph',
    'KGQuestionGenerator',
    'KGGeneratedItem',
    'GeneDiseasePair',
    'EvidenceLevel',
    'load_causaldb_kg',
    # Human evaluation
    'HumanExpertEvaluator',
    'HumanEvaluation',
    'FeedbackRecord',
    # Pipeline
    'BioREASONCPipeline',
    'PipelineConfig',
    # Export
    'BenchmarkExporter',
    'export_to_huggingface',
]

__version__ = "1.0.0"
