# BioREASONC-Bench Source Package
"""
BioREASONC-Bench: Biomedical REASONing-Centric Benchmark Generation Pipeline

Modules:
- schema: Data structures and validation
- ingest: Data loading and preprocessing (wrapper for Causal_KG)
- Causal_KG/: Knowledge Graph and database integration
  - data_loader: Disease risk data loading
  - kg_loader: CAUSALdb2 KG loading
  - databases/: External database clients (ClinVar, Ensembl, etc.)
- reacTax/: Taxonomy-based reasoning modules (ReacTax = Reasoning + Taxonomy)
  - structure: Structure-Aware reasoning (S) - Graph algorithms
  - causal: Causal-Aware reasoning (C) - PC, MR algorithms
  - risk: Risk-Aware reasoning (R) - GWAS scores
  - semantic: Semantic-Aware reasoning (M) - Text mining
- generator: Benchmark question generation
"""

from .schema import BenchmarkItem, BenchmarkDataset, RiskGene, GeneticVariant
from .ingest import DataIngestor

# Import ReacTax modules
from .reacTax import (
    # Structure (S)
    StructureReasoning,
    create_structure_module,
    # Causal (C)
    CausalReasoning,
    create_causal_module,
    # Risk (R)
    RiskReasoning,
    create_risk_module,
    # Semantic (M)
    SemanticReasoning,
    create_semantic_module,
)

# Import generator after reacTax (it depends on reacTax)
from .generator import BenchmarkGenerator

__version__ = "1.0.0"
__author__ = "BioREASONC-Bench Team"

__all__ = [
    # Core
    'BenchmarkItem',
    'BenchmarkDataset',
    'RiskGene',
    'GeneticVariant',
    'DataIngestor',
    'BenchmarkGenerator',
    # ReacTax modules
    'StructureReasoning',
    'create_structure_module',
    'CausalReasoning',
    'create_causal_module',
    'RiskReasoning',
    'create_risk_module',
    'SemanticReasoning',
    'create_semantic_module',
]
