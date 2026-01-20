#!/usr/bin/env python3
"""
STEP 0: Load Pre-processed Input Data
======================================
Load AD and T2D SNP data that was pre-processed from CAUSALdb2.

This replaces Step 1 (raw extraction) and Step 2 (scoring) since the
input data already contains calculated scores.

Input:  data/AD_snps.csv, data/t2d_snps.csv (pre-processed)
Output: output/step2_ad_snps_scored.csv, output/step2_t2d_snps_scored.csv
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
    DATA_DIR, OUTPUT_DIR, AD_INPUT_FILE, T2D_INPUT_FILE,
    GWAS_SIGNIFICANCE, MAF_MIN, MAF_MAX
)


def print_step(msg, char="="):
    """Print formatted step message."""
    print(f"\n{char * 70}")
    print(f"  {msg}")
    print(f"{char * 70}")


def load_and_process_data(input_file, disease_name):
    """Load pre-processed SNP data and standardize column names."""
    print_step(f"Loading {disease_name} Data", "-")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)
    print(f"  Input file: {input_file}")
    print(f"  Raw rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Standardize column names
    column_mapping = {
        'x_id': 'rsid',
        'position': 'bp',
        'pvalue': 'p',
        'effect_allele': 'ea',
    }

    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # Ensure required columns exist
    required_cols = ['rsid', 'chr', 'bp', 'beta', 'p', 'maf',
                     'causal_score', 'evidence_score', 'risk_score']

    for col in required_cols:
        if col not in df.columns:
            print(f"  WARNING: Missing column '{col}'")

    # Apply QC filters
    print(f"\n  Applying QC filters:")
    initial_count = len(df)

    # Filter 1: P-value significance
    df = df[df['p'] < GWAS_SIGNIFICANCE]
    print(f"    P < {GWAS_SIGNIFICANCE}: {len(df):,} ({len(df)/initial_count*100:.1f}%)")

    # Filter 2: MAF range
    if 'maf' in df.columns:
        df = df[(df['maf'] >= MAF_MIN) & (df['maf'] <= MAF_MAX)]
        print(f"    MAF in [{MAF_MIN}, {MAF_MAX}]: {len(df):,}")

    # Filter 3: Risk direction (beta > 0)
    df = df[df['beta'] > 0]
    print(f"    Beta > 0 (risk SNPs): {len(df):,}")

    # Filter 4: Remove duplicates (keep highest risk_score)
    if 'risk_score' in df.columns:
        df = df.sort_values('risk_score', ascending=False)
        df = df.drop_duplicates(subset=['rsid'], keep='first')
        print(f"    After deduplication: {len(df):,}")

    print(f"\n  Final {disease_name} SNPs: {len(df):,}")
    print(f"  Unique rsIDs: {df['rsid'].nunique():,}")

    # Show score distributions
    if 'causal_score' in df.columns:
        print(f"  Causal score range: [{df['causal_score'].min():.4f}, {df['causal_score'].max():.4f}]")
    if 'risk_score' in df.columns:
        print(f"  Risk score range: [{df['risk_score'].min():.4f}, {df['risk_score'].max():.4f}]")

    return df


def save_results(ad_df, t2d_df):
    """Save processed data in format expected by subsequent steps."""
    print_step("Saving Results", "-")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save in step2 output format (expected by step3)
    ad_path = OUTPUT_DIR / "step2_ad_snps_scored.csv"
    t2d_path = OUTPUT_DIR / "step2_t2d_snps_scored.csv"

    ad_df.to_csv(ad_path, index=False)
    t2d_df.to_csv(t2d_path, index=False)

    print(f"  Saved: {ad_path}")
    print(f"  Saved: {t2d_path}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "step": "0 (data loading)",
        "description": "Loaded pre-processed CAUSALdb2 data with QC filtering",
        "qc_filters": {
            "p_threshold": GWAS_SIGNIFICANCE,
            "maf_range": [MAF_MIN, MAF_MAX],
            "risk_direction": "beta > 0"
        },
        "ad": {
            "input_file": str(AD_INPUT_FILE),
            "final_snps": len(ad_df),
            "unique_rsids": int(ad_df['rsid'].nunique()),
        },
        "t2d": {
            "input_file": str(T2D_INPUT_FILE),
            "final_snps": len(t2d_df),
            "unique_rsids": int(t2d_df['rsid'].nunique()),
        }
    }

    summary_path = OUTPUT_DIR / "step0_data_loading_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {summary_path}")

    return summary


def main():
    """Run Step 0: Load pre-processed data."""
    print_step("STEP 0: LOAD PRE-PROCESSED INPUT DATA")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Load and process AD data
    ad_df = load_and_process_data(AD_INPUT_FILE, "AD")

    # Load and process T2D data
    t2d_df = load_and_process_data(T2D_INPUT_FILE, "T2D")

    # Save results
    summary = save_results(ad_df, t2d_df)

    print_step("STEP 0 COMPLETE")
    print(f"  AD SNPs:  {len(ad_df):,}")
    print(f"  T2D SNPs: {len(t2d_df):,}")
    print(f"\n  Ready for Step 3 (annotation)")

    return ad_df, t2d_df, summary


if __name__ == "__main__":
    ad_df, t2d_df, summary = main()
