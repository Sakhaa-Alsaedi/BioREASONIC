"""
Causal Knowledge Graph Module for BioREASONC-Bench

This module provides the data foundation for causal reasoning benchmarks:

Components:
-----------
1. Data Loaders:
   - DataIngestor: Load disease risk data (COVID-19, RA)
   - CAUSALdb2Loader: Load gene-disease KG from CAUSALdb2

2. Database Integrations (databases/):
   - ClinVar: Clinical variant significance
   - Ensembl: Gene/variant annotations
   - Open Targets: GWAS associations
   - STRING-DB: Protein interactions
   - Enrichr: Pathway enrichment
   - EpiGraphDB: Mendelian randomization

3. Knowledge Graph:
   - Gene-Disease relationships with evidence scores
   - SNP annotations with fine-mapping PIPs
   - Causal inference support (MR scores)

Usage:
------
```python
from src.Causal_KG import DataIngestor, CAUSALdb2Loader
from src.Causal_KG.databases import ClinVarClient, EnsemblClient

# Load disease risk data
ingestor = DataIngestor()
ingestor.load_all()

# Load KG
loader = CAUSALdb2Loader(data_path="path/to/causaldb2.csv")
pairs = loader.load_gene_disease_pairs()

# Annotate variants
clinvar = ClinVarClient()
variant = clinvar.get_variant("rs12345")
```
"""

from .data_loader import DataIngestor, load_data
from .kg_loader import (
    CAUSALdbKnowledgeGraph,
    GeneDiseasePair,
    EvidenceLevel,
    load_causaldb_kg,
    KGQuestionGenerator,
    KGGeneratedItem,
)

# Aliases for convenience
CAUSALdb2Loader = CAUSALdbKnowledgeGraph
load_causaldb2_kg = load_causaldb_kg

# Re-export database clients
from .databases import (
    ClinVarClient,
    ClinVarVariant,
    ClinicalSignificance,
    EnsemblClient,
    GeneInfo,
    VariantInfo,
    OpenTargetsClient,
    GeneDiseaseAssociation,
    GWASAssociation,
    DrugTarget,
    StringDBClient,
    NetworkCentrality,
    ProteinInteraction,
    EnrichrClient,
    EnrichmentResult,
    EpiGraphDBClient,
    MREvidence,
    PheWASResult,
    UnifiedBiomedicalClient,
    VariantAnnotation,
    GeneAnnotation,
)

__all__ = [
    # Data Loaders
    'DataIngestor',
    'load_data',
    'CAUSALdbKnowledgeGraph',
    'CAUSALdb2Loader',  # Alias
    'GeneDiseasePair',
    'EvidenceLevel',
    'load_causaldb_kg',
    'load_causaldb2_kg',  # Alias
    'KGQuestionGenerator',
    'KGGeneratedItem',
    # Database Clients
    'ClinVarClient',
    'ClinVarVariant',
    'ClinicalSignificance',
    'EnsemblClient',
    'GeneInfo',
    'VariantInfo',
    'OpenTargetsClient',
    'GeneDiseaseAssociation',
    'GWASAssociation',
    'DrugTarget',
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
