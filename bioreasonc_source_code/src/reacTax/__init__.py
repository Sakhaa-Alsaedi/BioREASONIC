"""
BioREASONC-Bench ReacTax Module

Taxonomy-based reasoning modules for biomedical QA generation:
- S (Structure): Network and pathway structure analysis
- C (Causal): Causal inference and reasoning
- R (Risk): Genetic risk scoring and aggregation
- M (seMantic): Entity recognition and relation extraction

ReacTax = Reasoning + Taxonomy
"""

from .structure import (
    StructureReasoning,
    create_structure_module,
    BiologicalNetwork,
    GraphAlgorithms,
    STRINGDBClient
)

# Import from new causal package
from .causal import (
    CausalReasoning,
    create_causal_module,
    CausalGraph,
    CausalEdge,
    MendelianRandomization,
    EpiGraphDBClient,
    # New exports for DoWhy/causal-learn integration
    CausalLearnDiscovery,
    DoWhyInference,
    EffectEstimate,
    RefutationResult,
    EstimatorMethod,
    RefutationMethod,
    CausalLearnAdapter,
    DoWhyAdapter,
)

# Backward compatibility alias
PCAlgorithm = CausalLearnDiscovery

from .risk import (
    RiskReasoning,
    create_risk_module,
    RiskClassifier,
    RiskScoreCalculator,
    GWASStatistics
)

from .semantic import (
    SemanticReasoning,
    create_semantic_module,
    EntityRecognizer,
    RuleBasedNER,
    RelationExtractor,
    BiomedicalTextMiner
)

__all__ = [
    # Structure (S)
    'StructureReasoning',
    'create_structure_module',
    'BiologicalNetwork',
    'GraphAlgorithms',
    'STRINGDBClient',
    # Causal (C)
    'CausalReasoning',
    'create_causal_module',
    'CausalGraph',
    'CausalEdge',
    'PCAlgorithm',  # Alias for backward compatibility
    'MendelianRandomization',
    'EpiGraphDBClient',
    # New Causal exports (DoWhy/causal-learn)
    'CausalLearnDiscovery',
    'DoWhyInference',
    'EffectEstimate',
    'RefutationResult',
    'EstimatorMethod',
    'RefutationMethod',
    'CausalLearnAdapter',
    'DoWhyAdapter',
    # Risk (R)
    'RiskReasoning',
    'create_risk_module',
    'RiskClassifier',
    'RiskScoreCalculator',
    'GWASStatistics',
    # Semantic (M)
    'SemanticReasoning',
    'create_semantic_module',
    'EntityRecognizer',
    'RuleBasedNER',
    'RelationExtractor',
    'BiomedicalTextMiner',
]
