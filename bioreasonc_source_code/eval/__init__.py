"""
BioREASONC-Bench Evaluation Module

Provides scoring frameworks for biomedical reasoning evaluation:

Standard Metrics:
- Accuracy, Precision, Recall, F1 Score
- BLEU, ROUGE, METEOR (text generation)
- Exact Match, Token F1 (QA-specific)
- Semantic Similarity, BERTScore

Novel Biomedical Metrics:
- GRASS: Genetic Risk Aggregate Score (WGRS)
- CARES: Causal-Aware Reasoning Evaluation Score
- ROCKET: Risk-Omics Causal Knowledge Enrichment Trust Score

API Integrations:
- ClinVar: Clinical variant significance
- Open Targets: GWAS associations
- Ensembl: Gene annotations
- STRING-DB: Protein interactions
- Enrichr: Pathway enrichment
- EpiGraphDB: Mendelian randomization evidence
- LLM-as-Judge: Response evaluation (OpenAI, Anthropic, Local)
"""

# Standard metrics
from .metrics import (
    StandardMetrics, MetricResult, EvaluationResults,
    evaluate_predictions
)

# Baseline evaluator
from .baseline import (
    BaselineEvaluator, BaselineResult,
    run_baseline_evaluation
)

# Core calculators
from .grass import GRASSCalculator, SNPRisk, GeneScore, APIEnabledGRASSCalculator
from .cares import (
    CARESCalculator, CARESConfig, ReasoningScore,
    APIEnabledCARESCalculator, score_llm_response
)
from .rocket import ROCKETCalculator, ComponentScores, APIEnabledROCKETCalculator

# API clients (imported lazily to avoid dependency issues)
def get_api_clients():
    """Get API client classes (lazy import)"""
    from .apis import (
        UnifiedBiomedicalClient,
        ClinVarClient,
        OpenTargetsClient,
        EnsemblClient,
        StringDBClient,
        EnrichrClient,
        EpiGraphDBClient,
        OpenAIJudge,
        AnthropicJudge,
        LocalLLMJudge
    )
    return {
        'UnifiedBiomedicalClient': UnifiedBiomedicalClient,
        'ClinVarClient': ClinVarClient,
        'OpenTargetsClient': OpenTargetsClient,
        'EnsemblClient': EnsemblClient,
        'StringDBClient': StringDBClient,
        'EnrichrClient': EnrichrClient,
        'EpiGraphDBClient': EpiGraphDBClient,
        'OpenAIJudge': OpenAIJudge,
        'AnthropicJudge': AnthropicJudge,
        'LocalLLMJudge': LocalLLMJudge
    }

__all__ = [
    # Standard Metrics
    'StandardMetrics',
    'MetricResult',
    'EvaluationResults',
    'evaluate_predictions',
    # Baseline Evaluator
    'BaselineEvaluator',
    'BaselineResult',
    'run_baseline_evaluation',
    # Core GRASS
    'GRASSCalculator',
    'SNPRisk',
    'GeneScore',
    'APIEnabledGRASSCalculator',
    # Core CARES
    'CARESCalculator',
    'CARESConfig',
    'ReasoningScore',
    'APIEnabledCARESCalculator',
    'score_llm_response',
    # Core ROCKET
    'ROCKETCalculator',
    'ComponentScores',
    'APIEnabledROCKETCalculator',
    # API clients accessor
    'get_api_clients'
]

__version__ = '1.2.0'
