#!/usr/bin/env python3
"""
STEP 1: Extract Raw PIPs from CAUSALdb2 (OPTIONAL)
==================================================
Extract SNP data with 7 fine-mapping posterior inclusion probabilities.

NOTE: This is a ONE-TIME extraction step. If data/AD_snps.csv and
data/t2d_snps.csv already exist, this step can be skipped.

Input:  CAUSALdb2 credible_set.txt (36M rows)
Output: AD and T2D SNPs with raw PIPs

Fine-mapping methods:
1. ABF (Approximate Bayes Factor)
2. FINEMAP
3. PAINTOR
4. CAVIARBF
5. SuSiE
6. PolyFun + FINEMAP
7. PolyFun + SuSiE

PIP values: [0, 1] = probability SNP is causal
            -1 = method failed/not available
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
    OUTPUT_DIR, PIP_COLUMNS, DATA_DIR,
    AD_META_IDS, T2D_META_IDS, AD_INPUT_FILE, T2D_INPUT_FILE
)

# Optional: Path to CAUSALdb2 raw file (only needed for initial extraction)
CAUSALDB_FILE = DATA_DIR.parent.parent / "CAUSALdb2" / "credible_set.txt"


def print_step(msg, char="="):
    """Print formatted step message."""
    print(f"\n{char * 70}")
    print(f"  {msg}")
    print(f"{char * 70}")


def validate_source_file():
    """Validate CAUSALdb2 source file exists."""
    print_step("Validating Source File", "-")

    if not CAUSALDB_FILE.exists():
        raise FileNotFoundError(f"CAUSALdb2 file not found: {CAUSALDB_FILE}")

    # Check file size
    size_gb = CAUSALDB_FILE.stat().st_size / (1024**3)
    print(f"  Source: {CAUSALDB_FILE}")
    print(f"  Size:   {size_gb:.2f} GB")

    return True


def extract_disease_snps(meta_ids, disease_name):
    """Extract SNPs for a specific disease from CAUSALdb2."""
    print_step(f"Extracting {disease_name} SNPs", "-")

    print(f"  Meta IDs: {meta_ids}")

    # Read header to get column names
    header_df = pd.read_csv(CAUSALDB_FILE, sep='\t', nrows=0)
    columns = header_df.columns.tolist()
    print(f"  Available columns: {len(columns)}")

    # Columns to extract
    extract_cols = [
        'chr', 'bp', 'rsid', 'maf', 'ea', 'nea', 'beta', 'se', 'p',
        'abf', 'finemap', 'paintor', 'caviarbf', 'susie',
        'polyfun_finemap', 'polyfun_susie',
        'meta_id', 'lead_snp'
    ]

    # Read in chunks and filter by meta_id
    chunk_size = 500000
    all_snps = []
    total_rows = 0

    print(f"  Reading in chunks of {chunk_size:,}...")

    for i, chunk in enumerate(pd.read_csv(CAUSALDB_FILE, sep='\t',
                                           usecols=extract_cols,
                                           chunksize=chunk_size)):
        total_rows += len(chunk)

        # Filter by meta_id
        matched = chunk[chunk['meta_id'].isin(meta_ids)]

        if len(matched) > 0:
            all_snps.append(matched)

        if (i + 1) % 20 == 0:
            print(f"    Processed {total_rows:,} rows, found {sum(len(df) for df in all_snps):,} {disease_name} SNPs")

    # Combine all chunks
    if all_snps:
        result = pd.concat(all_snps, ignore_index=True)
    else:
        result = pd.DataFrame()

    print(f"\n  Total rows scanned: {total_rows:,}")
    print(f"  {disease_name} SNPs found: {len(result):,}")
    print(f"  Unique rsIDs: {result['rsid'].nunique():,}")
    print(f"  Studies matched: {result['meta_id'].nunique()}")

    # Show PIP coverage
    print(f"\n  PIP method coverage:")
    for col in PIP_COLUMNS:
        valid = (result[col] >= 0).sum()
        pct = 100 * valid / len(result) if len(result) > 0 else 0
        print(f"    {col:18s}: {valid:,} ({pct:.1f}%)")

    return result


def validate_extracted_data(df, disease_name):
    """Validate extracted data quality."""
    print_step(f"Validating {disease_name} Data", "-")

    issues = []

    # Check for required columns
    required = ['rsid', 'chr', 'bp', 'beta', 'p'] + PIP_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Check for null rsids
    null_rsids = df['rsid'].isna().sum()
    if null_rsids > 0:
        issues.append(f"Null rsIDs: {null_rsids}")

    # Check p-value range
    invalid_p = ((df['p'] <= 0) | (df['p'] > 1)).sum()
    if invalid_p > 0:
        issues.append(f"Invalid p-values: {invalid_p}")

    # Check chromosome values
    valid_chrs = set(range(1, 23))
    invalid_chr = (~df['chr'].isin(valid_chrs)).sum()
    if invalid_chr > 0:
        issues.append(f"Invalid chromosomes: {invalid_chr}")

    if issues:
        print(f"  Issues found:")
        for issue in issues:
            print(f"    - {issue}")
        return False

    print(f"  All validations passed")
    print(f"  Rows:        {len(df):,}")
    print(f"  Chromosomes: {sorted(df['chr'].unique())}")
    print(f"  p-value range: [{df['p'].min():.2e}, {df['p'].max():.2e}]")
    print(f"  beta range: [{df['beta'].min():.4f}, {df['beta'].max():.4f}]")

    return True


def save_results(ad_df, t2d_df):
    """Save extracted data."""
    print_step("Saving Results", "-")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    ad_path = OUTPUT_DIR / "step1_ad_raw_pips.csv"
    t2d_path = OUTPUT_DIR / "step1_t2d_raw_pips.csv"

    ad_df.to_csv(ad_path, index=False)
    t2d_df.to_csv(t2d_path, index=False)

    print(f"  Saved: {ad_path}")
    print(f"  Saved: {t2d_path}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "source": str(CAUSALDB_FILE),
        "pip_methods": PIP_COLUMNS,
        "ad": {
            "meta_ids": AD_META_IDS,
            "total_snps": len(ad_df),
            "unique_rsids": int(ad_df['rsid'].nunique()),
            "studies": int(ad_df['meta_id'].nunique()),
        },
        "t2d": {
            "meta_ids": T2D_META_IDS,
            "total_snps": len(t2d_df),
            "unique_rsids": int(t2d_df['rsid'].nunique()),
            "studies": int(t2d_df['meta_id'].nunique()),
        }
    }

    summary_path = OUTPUT_DIR / "step1_extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {summary_path}")

    return summary


def main():
    """Run Step 1: Extract PIPs from CAUSALdb2."""
    print_step("STEP 1: EXTRACT RAW PIPs FROM CAUSALdb2")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Check if pre-processed data already exists
    if AD_INPUT_FILE.exists() and T2D_INPUT_FILE.exists():
        print(f"\n  Pre-processed data already exists:")
        print(f"    - {AD_INPUT_FILE}")
        print(f"    - {T2D_INPUT_FILE}")
        print(f"\n  SKIPPING extraction - use Step 0 to load existing data.")
        print_step("STEP 1 SKIPPED (data exists)")
        return None, None, {"status": "skipped", "reason": "pre-processed data exists"}

    # Check if CAUSALdb2 source file exists
    if not CAUSALDB_FILE.exists():
        print(f"\n  ERROR: CAUSALdb2 source file not found:")
        print(f"    Expected: {CAUSALDB_FILE}")
        print(f"\n  This step requires the raw CAUSALdb2 credible_set.txt file.")
        print(f"  If you have pre-processed data, place it in data/ directory.")
        print_step("STEP 1 FAILED")
        return None, None, {"status": "failed", "reason": "source file not found"}

    # Validate source
    validate_source_file()

    # Extract AD SNPs
    ad_df = extract_disease_snps(AD_META_IDS, "AD")
    validate_extracted_data(ad_df, "AD")

    # Extract T2D SNPs
    t2d_df = extract_disease_snps(T2D_META_IDS, "T2D")
    validate_extracted_data(t2d_df, "T2D")

    # Save results
    summary = save_results(ad_df, t2d_df)

    print_step("STEP 1 COMPLETE")
    print(f"  AD SNPs:  {len(ad_df):,}")
    print(f"  T2D SNPs: {len(t2d_df):,}")

    return ad_df, t2d_df, summary


if __name__ == "__main__":
    ad_df, t2d_df, summary = main()
