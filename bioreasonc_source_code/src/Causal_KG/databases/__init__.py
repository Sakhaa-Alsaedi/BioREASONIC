"""
Causal KG Database Integration Module

External biomedical databases for gene/SNP annotation and causal evidence:
- ClinVar: Clinical variant significance
- Ensembl: Gene annotations and variant effects
- Open Targets: GWAS associations and drug targets
- STRING-DB: Protein-protein interactions
- EpiGraphDB: Mendelian randomization evidence
- UnifiedClient: Combined annotation pipeline

Note: Enrichr has moved to enrichment/ module
"""

from .clinvar import ClinVarClient, ClinVarVariant, ClinicalSignificance
from .ensembl import EnsemblClient, GeneInfo, VariantInfo
from .open_targets import OpenTargetsClient, GeneDiseaseAssociation, GWASAssociation, DrugTarget
from .string_db import StringDBClient, NetworkCentrality, ProteinInteraction
from .epigraphdb import EpiGraphDBClient, MREvidence, PheWASResult
from .unified_client import UnifiedBiomedicalClient, VariantAnnotation, GeneAnnotation

# Re-export Enrichr from new location for backwards compatibility
from enrichment import EnrichrClient, EnrichmentResult

__all__ = [
    # ClinVar
    'ClinVarClient',
    'ClinVarVariant',
    'ClinicalSignificance',
    # Ensembl
    'EnsemblClient',
    'GeneInfo',
    'VariantInfo',
    # Open Targets
    'OpenTargetsClient',
    'GeneDiseaseAssociation',
    'GWASAssociation',
    'DrugTarget',
    # STRING-DB
    'StringDBClient',
    'NetworkCentrality',
    'ProteinInteraction',
    # Enrichr
    'EnrichrClient',
    'EnrichmentResult',
    # EpiGraphDB
    'EpiGraphDBClient',
    'MREvidence',
    'PheWASResult',
    # Unified Client
    'UnifiedBiomedicalClient',
    'VariantAnnotation',
    'GeneAnnotation',
]
