"""
Causal-Aware Reasoning Module (C)

Implements causal discovery and inference:
- PC Algorithm for causal structure learning (via causal-learn)
- FCI for handling latent confounders (via causal-learn)
- GES for score-based discovery (via causal-learn)
- DoWhy for causal effect estimation and refutation
- Mendelian Randomization (MR) via EpiGraphDB and custom implementations

Usage:
    from src.reacTax.causal import create_causal_module, CausalReasoning

    # Create module
    module = create_causal_module()

    # Run PC algorithm
    graph = module.run_pc_algorithm(data)

    # Run MR analysis
    mr_results = module.run_mr_analysis(beta_exp, beta_out, se_exp, se_out)

    # Generate benchmark questions
    questions = module.generate_causal_questions(genes)

    # NEW: Run DoWhy inference
    estimate = module.estimate_causal_effect(data, treatment, outcome)
    refutations = module.run_refutation_tests()
"""

# Graph structures
from .graph import CausalEdge, CausalGraph

# Discovery
from .discovery import CausalLearnDiscovery, PCAlgorithmLegacy

# Inference
from .inference import (
    DoWhyInference,
    EffectEstimate,
    RefutationResult,
    EstimatorMethod,
    RefutationMethod
)

# MR and external APIs
from .mr import MendelianRandomization
from .epigraphdb import EpiGraphDBClient

# Adapters
from .adapters import CausalLearnAdapter, DoWhyAdapter

# Main class and factory
from .reasoning import CausalReasoning, create_causal_module

__all__ = [
    # Graph
    'CausalEdge',
    'CausalGraph',

    # Discovery
    'CausalLearnDiscovery',
    'PCAlgorithmLegacy',

    # Inference
    'DoWhyInference',
    'EffectEstimate',
    'RefutationResult',
    'EstimatorMethod',
    'RefutationMethod',

    # MR
    'MendelianRandomization',
    'EpiGraphDBClient',

    # Adapters
    'CausalLearnAdapter',
    'DoWhyAdapter',

    # Main
    'CausalReasoning',
    'create_causal_module',
]
