#!/usr/bin/env python3
"""
STEP 5: Calculate GRASS Gene Scores (Extended)
===============================================
Aggregate SNP-level extended scores to gene-level GRASS scores.

Input:  SNP-Gene mappings from Step 4 + Extended scores from Step 3b
Output: Gene scores + MASTER file with all scores

Extended Score Formula (5-component, NO REDUNDANCY):
    extended_score = 0.40 × causal_score      (fine-mapping PIPs)
                   + 0.05 × clinvar_score     (clinical significance)
                   + 0.15 × vep_score         (functional impact)
                   + 0.20 × evidence_score    (GWAS p-value: -log10(p))
                   + 0.20 × effect_score      (GWAS effect size: |beta|)

Gene Score Formula:
- gene_score = MAX(extended_score) of all RISK SNPs mapped to gene
- This follows the "best evidence" principle

Master File Contains:
- All genes with GRASS scores
- Best SNP per gene (by extended_score)
- Component scores (causal, clinvar, vep, evidence, effect)
- Full scoring provenance
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, PIP_COLUMNS, ALPHA, BETA, EVIDENCE_SCALE,
    W_CAUSAL, W_CLINVAR, W_VEP, W_EVIDENCE, W_EFFECT,
    get_output_path
)


def print_step(msg, char="="):
    """Print formatted step message."""
    print(f"\n{char * 70}")
    print(f"  {msg}")
    print(f"{char * 70}")


def load_data():
    """Load SNP-gene mappings and extended scores."""
    print_step("Loading Data", "-")

    # Step 4: SNP-gene mappings
    ad_map_path = get_output_path("step4", "ad_mapped")
    t2d_map_path = get_output_path("step4", "t2d_mapped")

    if not ad_map_path.exists() or not t2d_map_path.exists():
        raise FileNotFoundError("Run Step 4 first to create SNP-gene mappings")

    ad_mapped = pd.read_csv(ad_map_path)
    t2d_mapped = pd.read_csv(t2d_map_path)

    print(f"  AD SNP-gene pairs:  {len(ad_mapped):,}")
    print(f"  T2D SNP-gene pairs: {len(t2d_mapped):,}")

    # Step 3: Evidence scores
    ad_ext_path = get_output_path("step3", "ad_evidence")
    t2d_ext_path = get_output_path("step3", "t2d_evidence")

    if not ad_ext_path.exists() or not t2d_ext_path.exists():
        raise FileNotFoundError("Run Step 3 first to calculate evidence scores")

    ad_ext = pd.read_csv(ad_ext_path)
    t2d_ext = pd.read_csv(t2d_ext_path)

    print(f"  AD extended scores:  {len(ad_ext):,}")
    print(f"  T2D extended scores: {len(t2d_ext):,}")

    # Merge mappings with extended scores
    ext_cols = ['rsid', 'extended_score', 'clinvar_score', 'vep_score', 'effect_score']
    ad_merged = ad_mapped.merge(ad_ext[ext_cols], on='rsid', how='left')
    t2d_merged = t2d_mapped.merge(t2d_ext[ext_cols], on='rsid', how='left')

    print(f"\n  Merged AD:  {len(ad_merged):,} pairs")
    print(f"  Merged T2D: {len(t2d_merged):,} pairs")

    return ad_merged, t2d_merged


def calculate_gene_scores(mapped_df, disease_name):
    """Calculate gene-level GRASS scores using extended_score."""
    print_step(f"Calculating {disease_name} Gene Scores", "-")

    print(f"  Aggregation method: MAX (best evidence)")
    print(f"  gene_score = MAX(extended_score) for all RISK SNPs in gene")

    # Find best SNP per gene by extended_score (for provenance)
    best_snp_idx = mapped_df.groupby('gene_name')['extended_score'].idxmax()
    best_snps = mapped_df.loc[best_snp_idx][[
        'gene_name', 'rsid', 'extended_score', 'risk_score',
        'causal_score', 'evidence_score', 'clinvar_score', 'vep_score', 'effect_score',
        'beta', 'p', 'position',
        'clinical_significance', 'most_severe_consequence'
    ]]
    best_snps = best_snps.rename(columns={
        'rsid': 'best_snp',
        'extended_score': 'best_snp_extended',
        'risk_score': 'best_snp_risk',
        'causal_score': 'best_snp_causal',
        'evidence_score': 'best_snp_evidence',
        'clinvar_score': 'best_snp_clinvar_score',
        'vep_score': 'best_snp_vep_score',
        'effect_score': 'best_snp_effect_score',
        'beta': 'best_snp_beta',
        'p': 'best_snp_p',
        'position': 'best_snp_position',
        'clinical_significance': 'best_snp_clinvar',
        'most_severe_consequence': 'best_snp_consequence'
    })

    # Aggregate by gene
    gene_agg = mapped_df.groupby(['gene_name', 'gene_id']).agg({
        'extended_score': ['max', 'mean'],
        'risk_score': ['max', 'mean', 'count'],
        'causal_score': 'max',
        'evidence_score': 'max',
        'clinvar_score': 'max',
        'vep_score': 'max',
        'effect_score': 'max',
        'beta': 'max',
        'p': 'min',
        'n_valid_pips': 'max',
        'rsid': lambda x: ','.join(x.unique()[:5])  # Top SNPs
    }).reset_index()

    # Flatten column names
    gene_agg.columns = [
        'gene_name', 'gene_id',
        'gene_score', 'mean_extended_score',
        'max_risk_score', 'mean_risk_score', 'snp_count',
        'max_causal', 'max_evidence', 'max_clinvar', 'max_vep', 'max_effect',
        'max_beta', 'min_p',
        'max_valid_pips', 'snps'
    ]

    # Merge with best SNP info
    gene_scores = gene_agg.merge(best_snps, on='gene_name', how='left')

    # Sort by gene score (extended_score)
    gene_scores = gene_scores.sort_values('gene_score', ascending=False).reset_index(drop=True)

    # Add rank
    gene_scores['rank'] = range(1, len(gene_scores) + 1)

    print(f"\n  Results:")
    print(f"    Total genes: {len(gene_scores):,}")
    print(f"    Extended score range: [{gene_scores['gene_score'].min():.6f}, {gene_scores['gene_score'].max():.6f}]")
    print(f"    Mean extended score:  {gene_scores['gene_score'].mean():.6f}")

    # Component contribution summary
    print(f"\n  Component Contribution (average):")
    avg_causal = gene_scores['max_causal'].mean()
    avg_clinvar = gene_scores['max_clinvar'].mean()
    avg_vep = gene_scores['max_vep'].mean()
    avg_evidence = gene_scores['max_evidence'].mean()
    avg_effect = gene_scores['max_effect'].mean()

    print(f"    Causal   ({W_CAUSAL:.0%}): {avg_causal:.4f} -> {W_CAUSAL * avg_causal:.4f}")
    print(f"    ClinVar  ({W_CLINVAR:.0%}): {avg_clinvar:.4f} -> {W_CLINVAR * avg_clinvar:.4f}")
    print(f"    VEP      ({W_VEP:.0%}): {avg_vep:.4f} -> {W_VEP * avg_vep:.4f}")
    print(f"    Evidence ({W_EVIDENCE:.0%}): {avg_evidence:.4f} -> {W_EVIDENCE * avg_evidence:.4f}")
    print(f"    Effect   ({W_EFFECT:.0%}): {avg_effect:.4f} -> {W_EFFECT * avg_effect:.4f}")

    return gene_scores


def show_top_genes(gene_scores, disease_name, n=20):
    """Display top scoring genes."""
    print_step(f"Top {n} {disease_name} GRASS Risk Genes (by Extended Score)", "-")

    print(f"{'Rank':<6}{'Gene':<15}{'ExtScore':<10}{'RiskScore':<10}{'VEP':<8}{'Effect':<8}{'SNPs':<6}{'Best SNP':<15}{'Consequence'}")
    print("-" * 110)

    for _, row in gene_scores.head(n).iterrows():
        consequence = str(row.get('best_snp_consequence', 'N/A'))[:15]
        print(f"{int(row['rank']):<6}{row['gene_name']:<15}{row['gene_score']:<10.4f}"
              f"{row['max_risk_score']:<10.4f}{row['max_vep']:<8.2f}{row['max_effect']:<8.4f}"
              f"{int(row['snp_count']):<6}{row['best_snp']:<15}{consequence}")


def create_master_file(ad_genes, t2d_genes):
    """Create master file combining AD and T2D results."""
    print_step("Creating MASTER File", "-")

    # Add disease column
    ad_genes = ad_genes.copy()
    t2d_genes = t2d_genes.copy()
    ad_genes['disease'] = 'AD'
    t2d_genes['disease'] = 'T2D'

    # Combine
    master = pd.concat([ad_genes, t2d_genes], ignore_index=True)

    # Add scoring formula information
    master['extended_formula'] = f"{W_CAUSAL}*causal + {W_CLINVAR}*clinvar + {W_VEP}*vep + {W_EVIDENCE}*evidence + {W_EFFECT}*effect"
    master['risk_formula'] = f"{ALPHA}*evidence + {BETA}*causal"
    master['aggregation'] = "MAX(extended_score)"

    # Reorder columns
    col_order = [
        'disease', 'rank', 'gene_name', 'gene_id', 'gene_score',
        'max_risk_score', 'max_causal', 'max_clinvar', 'max_vep', 'max_evidence', 'max_effect',
        'max_beta', 'min_p',
        'snp_count', 'max_valid_pips',
        'best_snp', 'best_snp_extended', 'best_snp_risk',
        'best_snp_causal', 'best_snp_clinvar_score', 'best_snp_vep_score',
        'best_snp_evidence', 'best_snp_effect_score',
        'best_snp_beta', 'best_snp_p', 'best_snp_position',
        'best_snp_clinvar', 'best_snp_consequence',
        'snps',
        'extended_formula', 'risk_formula', 'aggregation'
    ]
    col_order = [c for c in col_order if c in master.columns]
    master = master[col_order]

    print(f"  Master file contains:")
    print(f"    Total rows: {len(master):,}")
    print(f"    AD genes:   {len(ad_genes):,}")
    print(f"    T2D genes:  {len(t2d_genes):,}")

    # Check for shared genes
    ad_gene_set = set(ad_genes['gene_name'])
    t2d_gene_set = set(t2d_genes['gene_name'])
    shared = ad_gene_set & t2d_gene_set

    print(f"    Shared genes: {len(shared)}")
    if shared:
        print(f"    Examples: {list(shared)[:5]}")

    return master


def save_results(ad_genes, t2d_genes, master):
    """Save all results."""
    print_step("Saving Results", "-")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual disease gene scores
    ad_path = get_output_path("step5", "ad_genes")
    t2d_path = get_output_path("step5", "t2d_genes")
    master_path = get_output_path("step5", "master")

    ad_genes.to_csv(ad_path, index=False)
    t2d_genes.to_csv(t2d_path, index=False)
    master.to_csv(master_path, index=False)

    print(f"  Saved: {ad_path}")
    print(f"  Saved: {t2d_path}")
    print(f"  Saved: {master_path}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "formula": {
            "extended_score": f"{W_CAUSAL}*causal + {W_CLINVAR}*clinvar + {W_VEP}*vep + {W_EVIDENCE}*evidence + {W_EFFECT}*effect",
            "weights": {
                "causal": W_CAUSAL,
                "clinvar": W_CLINVAR,
                "vep": W_VEP,
                "evidence": W_EVIDENCE,
                "effect": W_EFFECT
            },
            "risk_score": f"{ALPHA}*evidence + {BETA}*causal (legacy)",
            "gene_score": "MAX(extended_score) of RISK SNPs (beta > 0)"
        },
        "pip_methods": PIP_COLUMNS,
        "ad": {
            "total_genes": len(ad_genes),
            "score_range": [float(ad_genes['gene_score'].min()), float(ad_genes['gene_score'].max())],
            "top_10_genes": ad_genes.head(10)['gene_name'].tolist(),
        },
        "t2d": {
            "total_genes": len(t2d_genes),
            "score_range": [float(t2d_genes['gene_score'].min()), float(t2d_genes['gene_score'].max())],
            "top_10_genes": t2d_genes.head(10)['gene_name'].tolist(),
        },
        "shared_genes": int(len(set(ad_genes['gene_name']) & set(t2d_genes['gene_name']))),
        "master_file": str(master_path)
    }

    summary_path = get_output_path("step5", "summary")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {summary_path}")

    return summary


def main():
    """Run Step 5: Calculate GRASS Gene Scores (Extended)."""
    print_step("STEP 5: CALCULATE GRASS GENE SCORES (EXTENDED)")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"\n  Extended Score Formula (5-component):")
    print(f"    extended_score = {W_CAUSAL:.2f} × causal")
    print(f"                   + {W_CLINVAR:.2f} × clinvar")
    print(f"                   + {W_VEP:.2f} × vep")
    print(f"                   + {W_EVIDENCE:.2f} × evidence")
    print(f"                   + {W_EFFECT:.2f} × effect")
    print(f"\n  Gene Aggregation: MAX(extended_score)")

    # Load data
    ad_merged, t2d_merged = load_data()

    # Calculate AD gene scores
    print_step("PROCESSING AD", "=")
    ad_genes = calculate_gene_scores(ad_merged, "AD")
    show_top_genes(ad_genes, "AD")

    # Calculate T2D gene scores
    print_step("PROCESSING T2D", "=")
    t2d_genes = calculate_gene_scores(t2d_merged, "T2D")
    show_top_genes(t2d_genes, "T2D")

    # Create master file
    master = create_master_file(ad_genes, t2d_genes)

    # Save
    summary = save_results(ad_genes, t2d_genes, master)

    print_step("STEP 5 COMPLETE")
    print(f"  AD genes:    {len(ad_genes):,}")
    print(f"  T2D genes:   {len(t2d_genes):,}")
    print(f"  Master file: {get_output_path('step5', 'master')}")
    print(f"\n  Top AD gene:  {ad_genes.iloc[0]['gene_name']} (score: {ad_genes.iloc[0]['gene_score']:.4f})")
    print(f"  Top T2D gene: {t2d_genes.iloc[0]['gene_name']} (score: {t2d_genes.iloc[0]['gene_score']:.4f})")

    return ad_genes, t2d_genes, master, summary


if __name__ == "__main__":
    ad_genes, t2d_genes, master, summary = main()
