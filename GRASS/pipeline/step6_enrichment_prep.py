#!/usr/bin/env python3
"""
STEP 6: Prepare Files for Enrichment Analysis
==============================================
Generate normalized ranking files required by enrichment scripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

def print_step(msg, char="="):
    print(f"\n{char * 70}")
    print(f"  {msg}")
    print(f"{char * 70}")


def normalize_column(series):
    """Normalize to [0, 1] range."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val > 0:
        return (series - min_val) / (max_val - min_val)
    return series * 0


def prepare_gene_rankings():
    """Create normalized ranking files for enrichment analysis."""
    print_step("STEP 6: PREPARE ENRICHMENT FILES")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Load GRASS scores
    ad_df = pd.read_csv(OUTPUT_DIR / "GRASS_AD_gene_scores.csv")
    t2d_df = pd.read_csv(OUTPUT_DIR / "GRASS_T2D_gene_scores.csv")

    print(f"\n  AD genes loaded: {len(ad_df)}")
    print(f"  T2D genes loaded: {len(t2d_df)}")

    # Add normalized columns
    for df, name in [(ad_df, 'AD'), (t2d_df, 'T2D')]:
        df['causal_norm'] = normalize_column(df['max_score'])
        df['p_norm'] = normalize_column(-np.log10(df['min_p'].clip(lower=1e-300)))
        df['risk_norm'] = normalize_column(df['GRASS_score'])
        df['vep_norm'] = 0.1  # Default VEP score
        df['clinvar_norm'] = 0.0  # Default ClinVar
        df['grass_score'] = df['GRASS_score']

        # Rename columns for compatibility
        df['n_snps'] = df['snp_count']
        df['causal_score'] = df['max_score']
        df['risk_score'] = df['GRASS_score']
        df['extended_score'] = df['max_score']
        df['beta_norm'] = normalize_column(df['avg_score'])

    # Rename rank column
    ad_df = ad_df.rename(columns={'rank': 'rank'})
    t2d_df = t2d_df.rename(columns={'rank': 'rank'})

    # Select output columns
    output_cols = ['gene_name', 'gene_id', 'n_snps', 'n_sig',
                   'causal_score', 'risk_score', 'extended_score',
                   'p_norm', 'beta_norm', 'causal_norm', 'risk_norm',
                   'vep_norm', 'clinvar_norm', 'support', 'grass_score', 'rank']

    # Ensure all columns exist
    for col in output_cols:
        if col not in ad_df.columns:
            ad_df[col] = 0
        if col not in t2d_df.columns:
            t2d_df[col] = 0

    # Save individual rankings
    ad_out = ad_df[output_cols].copy()
    t2d_out = t2d_df[output_cols].copy()

    ad_path = OUTPUT_DIR / "ad_genes_normalized_ranking.csv"
    t2d_path = OUTPUT_DIR / "t2d_genes_normalized_ranking.csv"

    ad_out.to_csv(ad_path, index=False)
    t2d_out.to_csv(t2d_path, index=False)

    print(f"\n  Saved: {ad_path}")
    print(f"  Saved: {t2d_path}")

    # Find shared genes
    ad_genes = set(ad_df['gene_name'])
    t2d_genes = set(t2d_df['gene_name'])
    shared_genes = ad_genes & t2d_genes

    print(f"\n  AD-only genes: {len(ad_genes - t2d_genes)}")
    print(f"  T2D-only genes: {len(t2d_genes - ad_genes)}")
    print(f"  Shared genes: {len(shared_genes)}")

    if len(shared_genes) > 0:
        # Create shared genes dataframe by averaging scores
        shared_rows = []
        for gene in shared_genes:
            ad_row = ad_df[ad_df['gene_name'] == gene].iloc[0]
            t2d_row = t2d_df[t2d_df['gene_name'] == gene].iloc[0]

            shared_rows.append({
                'gene_name': gene,
                'grass_score': (ad_row['GRASS_score'] + t2d_row['GRASS_score']) / 2,
                'causal_norm': (ad_row['causal_norm'] + t2d_row['causal_norm']) / 2,
                'p_norm': (ad_row['p_norm'] + t2d_row['p_norm']) / 2,
                'beta_norm': (ad_row['beta_norm'] + t2d_row['beta_norm']) / 2,
                'risk_norm': (ad_row['risk_norm'] + t2d_row['risk_norm']) / 2,
                'vep_norm': 0.1,
                'clinvar_norm': 0.0,
            })

        shared_df = pd.DataFrame(shared_rows)
        shared_df = shared_df.sort_values('grass_score', ascending=False).reset_index(drop=True)
        shared_df['rank'] = range(1, len(shared_df) + 1)

        shared_path = OUTPUT_DIR / "shared_genes_normalized_ranking.csv"
        shared_df.to_csv(shared_path, index=False)
        print(f"  Saved: {shared_path}")
        print(f"\n  Top 5 shared genes: {', '.join(shared_df.head(5)['gene_name'].tolist())}")
    else:
        # Create empty shared file
        shared_df = pd.DataFrame(columns=['gene_name', 'grass_score', 'causal_norm',
                                          'p_norm', 'beta_norm', 'risk_norm',
                                          'vep_norm', 'clinvar_norm', 'rank'])
        shared_path = OUTPUT_DIR / "shared_genes_normalized_ranking.csv"
        shared_df.to_csv(shared_path, index=False)
        print(f"  No shared genes found - created empty file")

    print_step("STEP 6 COMPLETE")
    print(f"""
  Files created for enrichment analysis:
    - ad_genes_normalized_ranking.csv ({len(ad_df)} genes)
    - t2d_genes_normalized_ranking.csv ({len(t2d_df)} genes)
    - shared_genes_normalized_ranking.csv ({len(shared_genes)} genes)
""")

    return ad_out, t2d_out, shared_df


if __name__ == "__main__":
    ad_df, t2d_df, shared_df = prepare_gene_rankings()
