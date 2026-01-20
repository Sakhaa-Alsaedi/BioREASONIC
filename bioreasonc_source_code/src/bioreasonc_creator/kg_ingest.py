"""
Knowledge Graph Ingestion Module for BioREASONC

DEPRECATED: This module has moved to src/Causal_KG/kg_loader.py
This file is kept for backwards compatibility.

New code should use:
    from src.Causal_KG import CAUSALdb2Loader, GeneDiseasePair, EvidenceLevel
    from src.Causal_KG.kg_loader import (
        CAUSALdbKnowledgeGraph,
        KGQuestionGenerator,
        load_causaldb_kg
    )
"""

# Re-export from new location for backwards compatibility
from src.Causal_KG.kg_loader import (
    EvidenceLevel,
    AnswerFormat,
    GeneDiseasePair,
    CAUSALdbKnowledgeGraph,
    load_causaldb_kg,
    KGGeneratedItem,
    KGQuestionGenerator,
)

# Alias for backwards compatibility
CAUSALdb2Loader = CAUSALdbKnowledgeGraph

__all__ = [
    'EvidenceLevel',
    'AnswerFormat',
    'GeneDiseasePair',
    'CAUSALdbKnowledgeGraph',
    'CAUSALdb2Loader',
    'load_causaldb_kg',
    'KGGeneratedItem',
    'KGQuestionGenerator',
]
