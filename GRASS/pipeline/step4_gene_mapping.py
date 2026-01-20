#!/usr/bin/env python3
"""
STEP 4: Map SNPs to Genes
=========================
Map annotated SNPs to nearby genes.

Input:  Annotated SNPs from Step 3
Output: SNP-Gene mappings with all scores and annotations

Mapping Strategy:
- Use existing SNP-gene mappings from seedExp
- Integrate with newly calculated scores from Step 2/3
- One SNP can map to multiple genes (within window)
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
    OUTPUT_DIR, BASE_DIR, WINDOW_SIZE, get_output_path
)

SEEDEXP_DIR = BASE_DIR / "seedExp"


def print_step(msg, char="="):
    """Print formatted step message."""
    print(f"\n{char * 70}")
    print(f"  {msg}")
    print(f"{char * 70}")


def load_step3_data():
    """Load annotated SNP data from Step 3 (or Step 2 if Step 3 was skipped)."""
    print_step("Loading SNP Data", "-")

    # Try Step 3 output first
    ad_path = get_output_path("step3", "ad_annotated")
    t2d_path = get_output_path("step3", "t2d_annotated")

    if ad_path.exists() and t2d_path.exists():
        print(f"  Loading from Step 3 (annotated SNPs)")
        ad_df = pd.read_csv(ad_path)
        t2d_df = pd.read_csv(t2d_path)
    else:
        # Fall back to Step 2 output
        print(f"  Step 3 output not found, loading from Step 2 (scored SNPs)")
        ad_path = get_output_path("step2", "ad_scored")
        t2d_path = get_output_path("step2", "t2d_scored")

        if not ad_path.exists() or not t2d_path.exists():
            raise FileNotFoundError("Run Step 2 first to calculate scores")

        ad_df = pd.read_csv(ad_path)
        t2d_df = pd.read_csv(t2d_path)

        # Add placeholder annotation columns
        ad_df['clinical_significance'] = 'not_annotated'
        ad_df['most_severe_consequence'] = 'not_annotated'
        ad_df['conditions'] = ''
        t2d_df['clinical_significance'] = 'not_annotated'
        t2d_df['most_severe_consequence'] = 'not_annotated'
        t2d_df['conditions'] = ''

    print(f"  AD SNPs loaded:  {len(ad_df):,}")
    print(f"  T2D SNPs loaded: {len(t2d_df):,}")

    return ad_df, t2d_df


def load_existing_mappings():
    """Load existing SNP-gene mappings."""
    print_step("Loading Existing Mappings", "-")

    ad_map_path = SEEDEXP_DIR / "ad_snp_gene_mapping.csv"
    t2d_map_path = SEEDEXP_DIR / "t2d_snp_gene_mapping.csv"

    if not ad_map_path.exists() or not t2d_map_path.exists():
        raise FileNotFoundError("SNP-gene mapping files not found in seedExp")

    ad_mapping = pd.read_csv(ad_map_path)
    t2d_mapping = pd.read_csv(t2d_map_path)

    # Keep only mapping columns (ignore old scores)
    mapping_cols = ['snp_id', 'gene_name', 'gene_id', 'distance', 'position', 'mapping_type']

    ad_mapping = ad_mapping[mapping_cols].drop_duplicates()
    t2d_mapping = t2d_mapping[mapping_cols].drop_duplicates()

    print(f"  AD mappings:  {len(ad_mapping):,} unique SNP-gene pairs")
    print(f"    Unique SNPs:  {ad_mapping['snp_id'].nunique():,}")
    print(f"    Unique genes: {ad_mapping['gene_name'].nunique():,}")

    print(f"  T2D mappings: {len(t2d_mapping):,} unique SNP-gene pairs")
    print(f"    Unique SNPs:  {t2d_mapping['snp_id'].nunique():,}")
    print(f"    Unique genes: {t2d_mapping['gene_name'].nunique():,}")

    return ad_mapping, t2d_mapping


def merge_snps_with_mappings(snp_df, mapping_df, disease_name):
    """Merge SNP scores/annotations with gene mappings."""
    print_step(f"Merging {disease_name} Data", "-")

    # Columns from SNP data to include
    snp_cols = ['rsid', 'chr', 'bp', 'beta', 'p', 'maf',
                'causal_score', 'evidence_score', 'risk_score', 'n_valid_pips',
                'clinical_significance', 'most_severe_consequence', 'conditions',
                'meta_id', 'lead_snp']
    snp_cols = [c for c in snp_cols if c in snp_df.columns]

    # CRITICAL FIX: Filter mapping to prioritize intronic/exonic over window mappings
    # For SNPs inside a gene (distance=0), keep ONLY that gene, not nearby genes
    print(f"  Applying SNP-gene mapping priority filter...")

    # Separate intronic mappings (distance=0) from window mappings
    intronic_map = mapping_df[mapping_df['distance'] == 0].copy()
    window_map = mapping_df[mapping_df['distance'] > 0].copy()

    # Get SNPs that have intronic mappings
    snps_with_intronic = set(intronic_map['snp_id'].unique())

    # For SNPs with intronic mappings, exclude window mappings
    window_map_filtered = window_map[~window_map['snp_id'].isin(snps_with_intronic)]

    # Combine: intronic + window (for SNPs without intronic)
    mapping_filtered = pd.concat([intronic_map, window_map_filtered], ignore_index=True)

    print(f"    Original mappings: {len(mapping_df):,}")
    print(f"    Intronic mappings: {len(intronic_map):,}")
    print(f"    Window mappings (SNPs without intronic): {len(window_map_filtered):,}")
    print(f"    Filtered mappings: {len(mapping_filtered):,}")
    print(f"    Removed window mappings: {len(mapping_df) - len(mapping_filtered):,}")

    # Merge on rsid <-> snp_id
    merged = mapping_filtered.merge(
        snp_df[snp_cols],
        left_on='snp_id',
        right_on='rsid',
        how='inner'
    )

    # Remove duplicate rsid column if exists
    if 'rsid' in merged.columns and 'snp_id' in merged.columns:
        merged = merged.drop(columns=['rsid'])
        merged = merged.rename(columns={'snp_id': 'rsid'})

    before_dedup = len(merged)
    print(f"  After merge: {before_dedup:,} SNP-Gene pairs")

    # Remove duplicate SNP-gene pairs (keep highest risk_score)
    merged = merged.sort_values('risk_score', ascending=False)
    merged = merged.drop_duplicates(subset=['rsid', 'gene_name'], keep='first')
    after_dedup = len(merged)

    print(f"\n  Duplicate SNP-Gene pair removal:")
    print(f"    Before: {before_dedup:,}")
    print(f"    After:  {after_dedup:,}")
    print(f"    Removed: {before_dedup - after_dedup:,} duplicates")

    print(f"\n  Final statistics:")
    print(f"    Input SNPs: {snp_df['rsid'].nunique():,}")
    print(f"    Mapped SNPs: {merged['rsid'].nunique():,}")
    print(f"    SNP-Gene pairs: {len(merged):,}")
    print(f"    Genes with SNPs: {merged['gene_name'].nunique():,}")

    # Match rate
    match_rate = 100 * merged['rsid'].nunique() / snp_df['rsid'].nunique()
    print(f"    Match rate: {match_rate:.1f}%")

    return merged


def analyze_mapping_quality(merged_df, disease_name):
    """Analyze quality of SNP-gene mappings."""
    print_step(f"Mapping Quality ({disease_name})", "-")

    # Position distribution
    print(f"  Position distribution:")
    pos_counts = merged_df['position'].value_counts()
    for pos, count in pos_counts.items():
        pct = 100 * count / len(merged_df)
        print(f"    {pos}: {count:,} ({pct:.1f}%)")

    # Mapping type distribution
    print(f"\n  Mapping type distribution:")
    type_counts = merged_df['mapping_type'].value_counts()
    for mtype, count in type_counts.items():
        pct = 100 * count / len(merged_df)
        print(f"    {mtype}: {count:,} ({pct:.1f}%)")

    # Distance statistics
    print(f"\n  Distance to gene:")
    print(f"    Min:    {merged_df['distance'].min():,.0f} bp")
    print(f"    Median: {merged_df['distance'].median():,.0f} bp")
    print(f"    Max:    {merged_df['distance'].max():,.0f} bp")

    # Clinical significance by position
    if 'clinical_significance' in merged_df.columns:
        print(f"\n  Clinical variants by position:")
        pathogenic = merged_df[merged_df['clinical_significance'].str.contains('athogenic', na=False)]
        if len(pathogenic) > 0:
            path_pos = pathogenic['position'].value_counts()
            for pos, count in path_pos.items():
                print(f"    {pos}: {count} pathogenic variants")


def show_top_mappings(merged_df, disease_name, n=10):
    """Show top SNP-gene mappings by risk score."""
    print_step(f"Top {n} {disease_name} SNP-Gene Pairs", "-")

    top = merged_df.nlargest(n, 'risk_score')

    print(f"{'SNP':<15}{'Gene':<15}{'Risk':<10}{'Causal':<10}{'Position':<12}{'Dist':<10}{'ClinVar'}")
    print("-" * 90)

    for _, row in top.iterrows():
        clinvar = str(row.get('clinical_significance', 'N/A'))[:15]
        print(f"{row['rsid']:<15}{row['gene_name']:<15}{row['risk_score']:<10.4f}"
              f"{row['causal_score']:<10.4f}{row['position']:<12}{int(row['distance']):<10}{clinvar}")


def save_results(ad_df, t2d_df):
    """Save SNP-gene mapping results."""
    print_step("Saving Results", "-")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Columns to save
    output_cols = [
        'rsid', 'chr', 'bp', 'gene_name', 'gene_id',
        'distance', 'position', 'mapping_type',
        'beta', 'p', 'maf',
        'causal_score', 'evidence_score', 'risk_score', 'n_valid_pips',
        'clinical_significance', 'most_severe_consequence', 'conditions',
        'meta_id', 'lead_snp'
    ]
    ad_cols = [c for c in output_cols if c in ad_df.columns]
    t2d_cols = [c for c in output_cols if c in t2d_df.columns]

    # Save
    ad_path = get_output_path("step4", "ad_mapped")
    t2d_path = get_output_path("step4", "t2d_mapped")

    ad_df[ad_cols].to_csv(ad_path, index=False)
    t2d_df[t2d_cols].to_csv(t2d_path, index=False)

    print(f"  Saved: {ad_path}")
    print(f"  Saved: {t2d_path}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "mapping_source": "seedExp/snp_gene_mapping.csv",
        "ad": {
            "snp_gene_pairs": len(ad_df),
            "unique_snps": int(ad_df['rsid'].nunique()),
            "unique_genes": int(ad_df['gene_name'].nunique()),
            "intronic_pairs": int((ad_df['position'] == 'intronic').sum()),
            "top_gene": ad_df.groupby('gene_name')['risk_score'].max().idxmax(),
        },
        "t2d": {
            "snp_gene_pairs": len(t2d_df),
            "unique_snps": int(t2d_df['rsid'].nunique()),
            "unique_genes": int(t2d_df['gene_name'].nunique()),
            "intronic_pairs": int((t2d_df['position'] == 'intronic').sum()),
            "top_gene": t2d_df.groupby('gene_name')['risk_score'].max().idxmax(),
        }
    }

    summary_path = get_output_path("step4", "summary")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {summary_path}")

    return summary


def main():
    """Run Step 4: Map SNPs to Genes."""
    print_step("STEP 4: MAP SNPs TO GENES")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Load data
    ad_snps, t2d_snps = load_step3_data()
    ad_mapping, t2d_mapping = load_existing_mappings()

    # Merge AD
    print_step("PROCESSING AD", "=")
    ad_merged = merge_snps_with_mappings(ad_snps, ad_mapping, "AD")
    analyze_mapping_quality(ad_merged, "AD")
    show_top_mappings(ad_merged, "AD")

    # Merge T2D
    print_step("PROCESSING T2D", "=")
    t2d_merged = merge_snps_with_mappings(t2d_snps, t2d_mapping, "T2D")
    analyze_mapping_quality(t2d_merged, "T2D")
    show_top_mappings(t2d_merged, "T2D")

    # Save
    summary = save_results(ad_merged, t2d_merged)

    print_step("STEP 4 COMPLETE")
    print(f"  AD SNP-Gene pairs:  {len(ad_merged):,}")
    print(f"  T2D SNP-Gene pairs: {len(t2d_merged):,}")

    return ad_merged, t2d_merged, summary


if __name__ == "__main__":
    ad_merged, t2d_merged, summary = main()
