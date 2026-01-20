"""
BioREASONC-Bench API Integrations

LLM Judge APIs for model evaluation.
Database clients have moved to src/Causal_KG/databases/

For backwards compatibility, database clients are re-exported here.
New code should import from src.Causal_KG.databases directly.
"""

# LLM Judge (stays here)
from .llm_judge import (
    LLMJudge, OpenAIJudge, AnthropicJudge, GeminiJudge, TogetherJudge,
    LocalLLMJudge, JudgeResult, create_judge
)

# Re-export database clients for backwards compatibility
# New code should use: from src.Causal_KG.databases import ...
from src.Causal_KG.databases import (
    ClinVarClient, ClinVarVariant, ClinicalSignificance,
    OpenTargetsClient, GeneDiseaseAssociation, GWASAssociation, DrugTarget,
    EnsemblClient, GeneInfo, VariantInfo,
    StringDBClient, NetworkCentrality, ProteinInteraction,
    EnrichrClient, EnrichmentResult,
    EpiGraphDBClient, MREvidence, PheWASResult,
    UnifiedBiomedicalClient, VariantAnnotation, GeneAnnotation,
)

__all__ = [
    # LLM Judge
    'LLMJudge',
    'OpenAIJudge',
    'AnthropicJudge',
    'GeminiJudge',
    'TogetherJudge',
    'LocalLLMJudge',
    'JudgeResult',
    'create_judge',
    # Database clients (backwards compatibility)
    'ClinVarClient',
    'ClinVarVariant',
    'ClinicalSignificance',
    'OpenTargetsClient',
    'GeneDiseaseAssociation',
    'GWASAssociation',
    'DrugTarget',
    'EnsemblClient',
    'GeneInfo',
    'VariantInfo',
    'StringDBClient',
    'NetworkCentrality',
    'ProteinInteraction',
    'EnrichrClient',
    'EnrichmentResult',
    'EpiGraphDBClient',
    'MREvidence',
    'PheWASResult',
    'UnifiedBiomedicalClient',
    'VariantAnnotation',
    'GeneAnnotation',
]
