#!/usr/bin/env python3
"""
GRASS Pipeline Configuration
============================
Central configuration for all pipeline steps.

GRASS: Genetic Risk Aggregation Scoring System
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = BASE_DIR / "pipeline"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
CACHE_DIR = ANNOTATIONS_DIR / "cache"
SEEDEXP_DIR = BASE_DIR / "seedExp"

# Create output directory if not exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Input data files (pre-processed from CAUSALdb2)
AD_INPUT_FILE = DATA_DIR / "AD_snps.csv"
T2D_INPUT_FILE = DATA_DIR / "t2d_snps.csv"

# Gene annotation file (Ensembl GRCh37)
GENE_ANNOTATION_FILE = ANNOTATIONS_DIR / "grch37_genes.csv"

# =============================================================================
# DISEASE META IDS (for reference - data already extracted)
# =============================================================================
# AD meta_ids from CAUSALdb2 (Alzheimer's Disease studies)
AD_META_IDS = [
    "GCST90027158",  # Bellenguez 2022 - largest AD GWAS
    "OT344",         # Jansen 2018
    "F900093",       # FinnGen AD - contains rs429358 fine-mapped
    "F900750",       # FinnGen AD related
    "GD00445",       # Another AD study
    "CA183",         # AD study
]

# T2D meta_ids from CAUSALdb2 (Type 2 Diabetes studies)
T2D_META_IDS = [
    "OT332",   # T2D study
    "OT333",   # T2D study
    "GD00829", # T2D study
    "GD09426", # T2D study
]

# =============================================================================
# FINE-MAPPING METHODS
# =============================================================================
# 7 fine-mapping methods from CAUSALdb2
PIP_COLUMNS = [
    'abf',              # Approximate Bayes Factor
    'finemap',          # FINEMAP
    'paintor',          # PAINTOR
    'caviarbf',         # CAVIARBF
    'susie',            # SuSiE
    'polyfun_finemap',  # PolyFun + FINEMAP
    'polyfun_susie',    # PolyFun + SuSiE
]

# =============================================================================
# GRASS FORMULA PARAMETERS
# =============================================================================
# Gene-level GRASS Score:
#   GRASS = MAX + (1 - MAX) × GRASS_BETA × support
#   Where:
#     MAX     = max(extended_score) for all SNPs mapped to gene
#     support = log(1 + n_sig) / log(1 + N_max)
#     n_sig   = count of significant SNPs (p < GWAS_SIGNIFICANCE)

GRASS_BETA = 0.3  # Support weight in GRASS formula

# =============================================================================
# SNP SCORING FORMULA
# =============================================================================
# SNP Risk Score = ALPHA * evidence_score + BETA * causal_score
ALPHA = 0.3  # Weight for evidence (GWAS significance)
BETA = 0.7   # Weight for causality (fine-mapping)

# Evidence score normalization: evidence = min(-log10(p) / EVIDENCE_SCALE, 1.0)
EVIDENCE_SCALE = 10.0  # p=1e-10 gives evidence_score=1.0

# =============================================================================
# EXTENDED SCORE FORMULA (4-COMPONENT)
# =============================================================================
# extended_score = W_CAUSAL × causal_score      (fine-mapping PIPs)
#                + W_VEP × vep_score            (functional impact)
#                + W_EVIDENCE × evidence_score  (GWAS p-value)
#                + W_EFFECT × effect_score      (GWAS effect size |beta|)
#
# Weights sum to 1.0

W_CAUSAL = 0.65    # Weight for causality (fine-mapping PIPs) - 65%
W_CLINVAR = 0.00   # Weight for ClinVar (disabled) - 0%
W_VEP = 0.05       # Weight for VEP functional impact - 5%
W_EVIDENCE = 0.10  # Weight for GWAS significance - 10%
W_EFFECT = 0.20    # Weight for GWAS effect size - 20%

# =============================================================================
# CLINVAR SCORE MAPPING
# =============================================================================
CLINVAR_SCORES = {
    'pathogenic': 1.0,
    'likely_pathogenic': 0.8,
    'uncertain_significance': 0.3,
    'likely_benign': 0.1,
    'benign': 0.0,
    'not_in_clinvar': 0.0,
    'not_annotated': 0.0,
}

# =============================================================================
# VEP IMPACT SCORE MAPPING
# =============================================================================
VEP_IMPACT_SCORES = {
    # HIGH impact (1.0)
    'transcript_ablation': 1.0,
    'splice_acceptor_variant': 1.0,
    'splice_donor_variant': 1.0,
    'stop_gained': 1.0,
    'frameshift_variant': 1.0,
    'stop_lost': 1.0,
    'start_lost': 1.0,
    # MODERATE impact (0.6-0.7)
    'missense_variant': 0.7,
    'inframe_insertion': 0.6,
    'inframe_deletion': 0.6,
    'protein_altering_variant': 0.6,
    # LOW impact (0.3)
    'splice_region_variant': 0.3,
    'splice_donor_5th_base_variant': 0.3,
    'splice_polypyrimidine_tract_variant': 0.3,
    'synonymous_variant': 0.3,
    # MODIFIER impact (0.1)
    'intron_variant': 0.1,
    'intergenic_variant': 0.1,
    'upstream_gene_variant': 0.1,
    'downstream_gene_variant': 0.1,
    '3_prime_UTR_variant': 0.1,
    '5_prime_UTR_variant': 0.1,
    'non_coding_transcript_exon_variant': 0.1,
    'regulatory_region_variant': 0.1,
    'NMD_transcript_variant': 0.1,
    # Default
    'not_annotated': 0.0,
    '': 0.0,
}

# =============================================================================
# QUALITY CONTROL THRESHOLDS
# =============================================================================
GWAS_SIGNIFICANCE = 5e-8  # Genome-wide significance
MAF_MIN = 0.001           # Minimum minor allele frequency
MAF_MAX = 0.5             # Maximum MAF

# =============================================================================
# GENE MAPPING PARAMETERS
# =============================================================================
WINDOW_SIZE = 500000  # 500kb window for SNP-gene mapping
MAPPING_TYPES = ['intronic', 'exonic', 'upstream', 'downstream', 'intergenic']

# =============================================================================
# API ENDPOINTS (GRCh37)
# =============================================================================
ENSEMBL_REST = "https://grch37.rest.ensembl.org"
CLINVAR_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
CLINVAR_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Rate limiting
API_DELAY = 0.15  # seconds between API requests
MAX_RETRIES = 3

# =============================================================================
# OUTPUT FILES
# =============================================================================
OUTPUT_FILES = {
    "step0": {
        "ad_scored": "step0_ad_snps_scored.csv",
        "t2d_scored": "step0_t2d_snps_scored.csv",
        "summary": "step0_data_loading_summary.json",
    },
    "step1": {
        "ad_scored": "step1_ad_snps_scored.csv",
        "t2d_scored": "step1_t2d_snps_scored.csv",
        "summary": "step1_snp_scoring_summary.json",
    },
    "step2": {
        "ad_annotated": "step2_ad_snps_annotated.csv",
        "t2d_annotated": "step2_t2d_snps_annotated.csv",
        "summary": "step2_annotation_summary.json",
    },
    "step3": {
        "ad_evidence": "step3_ad_evidence_scores.csv",
        "t2d_evidence": "step3_t2d_evidence_scores.csv",
        "summary": "step3_evidence_summary.json",
    },
    "step4": {
        "ad_mapped": "step4_ad_snp_gene_mapping.csv",
        "t2d_mapped": "step4_t2d_snp_gene_mapping.csv",
        "summary": "step4_mapping_summary.json",
    },
    "step5": {
        "ad_genes": "GRASS_AD_gene_scores.csv",
        "t2d_genes": "GRASS_T2D_gene_scores.csv",
        "master": "GRASS_master_scores.csv",
        "summary": "step5_grass_summary.json",
    },
    "step6": {
        "ad_ranking": "ad_genes_normalized_ranking.csv",
        "t2d_ranking": "t2d_genes_normalized_ranking.csv",
        "shared_ranking": "shared_genes_normalized_ranking.csv",
        "summary": "step6_enrichment_prep_summary.json",
    },
}


def get_output_path(step, key):
    """Get output file path for a given step and key."""
    return OUTPUT_DIR / OUTPUT_FILES[step][key]


def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("GRASS PIPELINE CONFIGURATION")
    print("=" * 70)
    print(f"\nPaths:")
    print(f"  BASE_DIR:     {BASE_DIR}")
    print(f"  OUTPUT_DIR:   {OUTPUT_DIR}")
    print(f"  DATA_DIR:     {DATA_DIR}")

    print(f"\nGRASS Formula:")
    print(f"  GRASS = MAX + (1 - MAX) × {GRASS_BETA} × support")
    print(f"  support = log(1 + n_sig) / log(1 + N_max)")

    print(f"\nExtended Score Weights:")
    print(f"  Causal:   {W_CAUSAL:.0%}")
    print(f"  VEP:      {W_VEP:.0%}")
    print(f"  Evidence: {W_EVIDENCE:.0%}")
    print(f"  Effect:   {W_EFFECT:.0%}")

    print(f"\nInput Files:")
    print(f"  AD:  {AD_INPUT_FILE}")
    print(f"  T2D: {T2D_INPUT_FILE}")


if __name__ == "__main__":
    print_config()
