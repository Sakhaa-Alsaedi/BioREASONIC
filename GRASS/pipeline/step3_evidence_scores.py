#!/usr/bin/env python3
"""
STEP 3B: Calculate Extended Scores with ClinVar and VEP
========================================================
Calculate 5-component extended_score using annotations from Step 3.

Input:  Annotated SNPs from Step 3 (with evidence_score, effect_score from Step 2)
Output: SNPs with extended_score (5-component formula)

Extended Score Formula (NO REDUNDANCY):
---------------------------------------
extended_score = W_CAUSAL   × causal_score      (0.40)
               + W_CLINVAR  × clinvar_score     (0.05)
               + W_VEP      × vep_score         (0.15)
               + W_EVIDENCE × evidence_score    (0.20)
               + W_EFFECT   × effect_score      (0.20)

Components:
-----------
- causal_score:   Mean of fine-mapping PIPs [0, 1]
- clinvar_score:  ClinVar clinical significance mapped to [0, 1]
- vep_score:      VEP functional impact mapped to [0, 1]
- evidence_score: -log10(p) min-max normalized [0, 1] (from Step 2)
- effect_score:   |beta| min-max normalized [0, 1] (from Step 2)

NOTE: evidence_score and effect_score are kept SEPARATE to avoid
      double-counting (p-value already depends on beta).

Note: Original risk_score is preserved unchanged.
"""

import pandas as pd
import numpy as np
import json
import math
from datetime import datetime
from pathlib import Path

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, get_output_path,
    W_CAUSAL, W_CLINVAR, W_VEP, W_EVIDENCE, W_EFFECT,
    CLINVAR_SCORES, VEP_IMPACT_SCORES
)


def print_step(msg, char="="):
    """Print formatted step message."""
    print(f"\n{char * 70}")
    print(f"  {msg}")
    print(f"{char * 70}")


def load_step3_data():
    """Load annotated SNP data from Step 3."""
    print_step("Loading Step 3 Annotated Data", "-")

    ad_path = get_output_path("step3", "ad_annotated")
    t2d_path = get_output_path("step3", "t2d_annotated")

    if not ad_path.exists() or not t2d_path.exists():
        raise FileNotFoundError("Run Step 3 first to annotate SNPs")

    ad_df = pd.read_csv(ad_path)
    t2d_df = pd.read_csv(t2d_path)

    print(f"  AD SNPs loaded:  {len(ad_df):,}")
    print(f"  T2D SNPs loaded: {len(t2d_df):,}")

    return ad_df, t2d_df


def calculate_clinvar_score(df):
    """
    Map ClinVar clinical significance to clinvar_score [0, 1].

    Mapping:
        pathogenic:             1.0
        likely_pathogenic:      0.8
        uncertain_significance: 0.3
        likely_benign:          0.1
        benign:                 0.0
        not_in_clinvar:         0.0 (default)
    """
    print_step("Calculating ClinVar Score", "-")

    df = df.copy()

    # Map clinical significance to score
    df['clinvar_score'] = df['clinical_significance'].map(CLINVAR_SCORES).fillna(0.0)

    # Stats
    has_clinvar = df['clinvar_score'] > 0
    print(f"  SNPs with ClinVar data: {has_clinvar.sum():,} ({100*has_clinvar.mean():.1f}%)")
    print(f"  ClinVar score range: [{df['clinvar_score'].min():.2f}, {df['clinvar_score'].max():.2f}]")
    print(f"  ClinVar score mean:  {df['clinvar_score'].mean():.4f}")

    # Distribution
    print(f"\n  ClinVar Distribution:")
    for sig, score in sorted(CLINVAR_SCORES.items(), key=lambda x: -x[1]):
        count = (df['clinical_significance'] == sig).sum()
        if count > 0:
            print(f"    {sig:30s}: {count:5d} -> score {score:.1f}")

    return df


def calculate_vep_score(df):
    """
    Map VEP consequence to vep_score [0, 1].

    Impact levels:
        HIGH:     1.0 (stop_gained, frameshift, splice_donor/acceptor)
        MODERATE: 0.6 (missense, inframe_indel)
        LOW:      0.3 (synonymous, splice_region)
        MODIFIER: 0.1 (intron, intergenic, UTR)
    """
    print_step("Calculating VEP Score", "-")

    df = df.copy()

    # Map consequence to score
    df['vep_score'] = df['most_severe_consequence'].map(VEP_IMPACT_SCORES).fillna(0.0)

    # Stats
    has_vep = df['vep_score'] > 0
    print(f"  SNPs with VEP data: {has_vep.sum():,} ({100*has_vep.mean():.1f}%)")
    print(f"  VEP score range: [{df['vep_score'].min():.2f}, {df['vep_score'].max():.2f}]")
    print(f"  VEP score mean:  {df['vep_score'].mean():.4f}")

    # Impact distribution
    print(f"\n  VEP Impact Distribution:")
    for impact, score in [(1.0, 'HIGH'), (0.6, 'MODERATE'), (0.3, 'LOW'), (0.1, 'MODIFIER'), (0.0, 'NONE')]:
        count = (df['vep_score'] == impact).sum()
        if count > 0:
            print(f"    {score:10s} (score={impact:.1f}): {count:5d} SNPs")

    return df


def calculate_extended_score(df):
    """
    Calculate 5-component extended_score (NO REDUNDANCY).

    Formula:
        extended_score = W_CAUSAL   × causal_score
                       + W_CLINVAR  × clinvar_score
                       + W_VEP      × vep_score
                       + W_EVIDENCE × evidence_score
                       + W_EFFECT   × effect_score

    NOTE: Uses effect_score SEPARATELY from evidence_score
          to avoid double-counting (p-value depends on beta).
    """
    print_step("Calculating Extended Score", "-")

    print(f"  Formula: extended_score = ")
    print(f"           {W_CAUSAL:.2f} × causal_score")
    print(f"         + {W_CLINVAR:.2f} × clinvar_score")
    print(f"         + {W_VEP:.2f} × vep_score")
    print(f"         + {W_EVIDENCE:.2f} × evidence_score")
    print(f"         + {W_EFFECT:.2f} × effect_score")
    print(f"  Weights sum: {W_CAUSAL + W_CLINVAR + W_VEP + W_EVIDENCE + W_EFFECT:.2f}")

    df = df.copy()

    # Calculate extended score (NO gwas_score - use effect_score directly)
    df['extended_score'] = (
        W_CAUSAL * df['causal_score'] +
        W_CLINVAR * df['clinvar_score'] +
        W_VEP * df['vep_score'] +
        W_EVIDENCE * df['evidence_score'] +
        W_EFFECT * df['effect_score']
    )

    # Stats
    print(f"\n  Extended Score Statistics:")
    print(f"    Range: [{df['extended_score'].min():.6f}, {df['extended_score'].max():.6f}]")
    print(f"    Mean:  {df['extended_score'].mean():.6f}")
    print(f"    Std:   {df['extended_score'].std():.6f}")

    # Compare with original risk_score
    print(f"\n  Comparison with Original risk_score:")
    print(f"    risk_score range:     [{df['risk_score'].min():.6f}, {df['risk_score'].max():.6f}]")
    print(f"    extended_score range: [{df['extended_score'].min():.6f}, {df['extended_score'].max():.6f}]")

    corr = df['risk_score'].corr(df['extended_score'])
    print(f"    Correlation:          {corr:.4f}")

    return df


def show_component_contributions(df, disease_name):
    """Show average contribution of each component to extended_score."""
    print_step(f"Component Contributions ({disease_name})", "-")

    avg_causal = df['causal_score'].mean()
    avg_clinvar = df['clinvar_score'].mean()
    avg_vep = df['vep_score'].mean()
    avg_evidence = df['evidence_score'].mean()
    avg_effect = df['effect_score'].mean()

    contrib_causal = W_CAUSAL * avg_causal
    contrib_clinvar = W_CLINVAR * avg_clinvar
    contrib_vep = W_VEP * avg_vep
    contrib_evidence = W_EVIDENCE * avg_evidence
    contrib_effect = W_EFFECT * avg_effect

    total = contrib_causal + contrib_clinvar + contrib_vep + contrib_evidence + contrib_effect

    print(f"  Component     | Weight | Avg Score | Contribution | % of Total")
    print(f"  {'-'*60}")
    print(f"  Causal        | {W_CAUSAL:.2f}   | {avg_causal:.4f}    | {contrib_causal:.4f}       | {100*contrib_causal/total:.1f}%")
    print(f"  ClinVar       | {W_CLINVAR:.2f}   | {avg_clinvar:.4f}    | {contrib_clinvar:.4f}       | {100*contrib_clinvar/total:.1f}%")
    print(f"  VEP           | {W_VEP:.2f}   | {avg_vep:.4f}    | {contrib_vep:.4f}       | {100*contrib_vep/total:.1f}%")
    print(f"  Evidence      | {W_EVIDENCE:.2f}   | {avg_evidence:.4f}    | {contrib_evidence:.4f}       | {100*contrib_evidence/total:.1f}%")
    print(f"  Effect        | {W_EFFECT:.2f}   | {avg_effect:.4f}    | {contrib_effect:.4f}       | {100*contrib_effect/total:.1f}%")
    print(f"  {'-'*60}")
    print(f"  Total         | 1.00   |           | {total:.4f}       | 100.0%")


def show_example(df, rsid, disease_name):
    """Show detailed calculation for an example SNP."""
    print_step(f"Example Calculation: {rsid} ({disease_name})", "-")

    snp = df[df['rsid'] == rsid]
    if len(snp) == 0:
        print(f"  SNP {rsid} not found")
        return

    row = snp.iloc[0]

    print(f"\n  SNP: {row['rsid']} (chr{int(row['chr'])}:{int(row['bp'])})")

    print(f"\n  Component Scores:")
    print(f"    causal_score:   {row['causal_score']:.6f} (from {int(row['n_valid_pips'])} PIPs)")
    print(f"    clinvar_score:  {row['clinvar_score']:.6f} ({row['clinical_significance']})")
    print(f"    vep_score:      {row['vep_score']:.6f} ({row['most_severe_consequence']})")
    print(f"    evidence_score: {row['evidence_score']:.6f} (p={row['p']:.2e})")
    print(f"    effect_score:   {row['effect_score']:.6f} (beta={row['beta']:.4f})")

    print(f"\n  Extended Score Calculation:")
    print(f"    = {W_CAUSAL:.2f} × {row['causal_score']:.4f}")
    print(f"    + {W_CLINVAR:.2f} × {row['clinvar_score']:.4f}")
    print(f"    + {W_VEP:.2f} × {row['vep_score']:.4f}")
    print(f"    + {W_EVIDENCE:.2f} × {row['evidence_score']:.4f}")
    print(f"    + {W_EFFECT:.2f} × {row['effect_score']:.4f}")

    contrib_causal = W_CAUSAL * row['causal_score']
    contrib_clinvar = W_CLINVAR * row['clinvar_score']
    contrib_vep = W_VEP * row['vep_score']
    contrib_evidence = W_EVIDENCE * row['evidence_score']
    contrib_effect = W_EFFECT * row['effect_score']

    print(f"\n    = {contrib_causal:.4f} + {contrib_clinvar:.4f} + {contrib_vep:.4f} + {contrib_evidence:.4f} + {contrib_effect:.4f}")
    print(f"    = {row['extended_score']:.6f}")

    print(f"\n  Comparison:")
    print(f"    Original risk_score: {row['risk_score']:.6f}")
    print(f"    Extended score:      {row['extended_score']:.6f}")
    print(f"    Difference:          {row['extended_score'] - row['risk_score']:+.6f}")


def save_results(ad_df, t2d_df):
    """Save extended score data."""
    print_step("Saving Results", "-")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    ad_path = get_output_path("step3", "ad_evidence")
    t2d_path = get_output_path("step3", "t2d_evidence")

    ad_df.to_csv(ad_path, index=False)
    t2d_df.to_csv(t2d_path, index=False)

    print(f"  Saved: {ad_path}")
    print(f"  Saved: {t2d_path}")

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
                "effect": W_EFFECT,
            }
        },
        "ad": {
            "snp_count": len(ad_df),
            "extended_score_range": [float(ad_df['extended_score'].min()), float(ad_df['extended_score'].max())],
            "extended_score_mean": float(ad_df['extended_score'].mean()),
            "risk_score_correlation": float(ad_df['risk_score'].corr(ad_df['extended_score'])),
            "clinvar_coverage": float((ad_df['clinvar_score'] > 0).mean()),
            "vep_coverage": float((ad_df['vep_score'] > 0).mean()),
            "top_snp": ad_df.loc[ad_df['extended_score'].idxmax(), 'rsid'],
        },
        "t2d": {
            "snp_count": len(t2d_df),
            "extended_score_range": [float(t2d_df['extended_score'].min()), float(t2d_df['extended_score'].max())],
            "extended_score_mean": float(t2d_df['extended_score'].mean()),
            "risk_score_correlation": float(t2d_df['risk_score'].corr(t2d_df['extended_score'])),
            "clinvar_coverage": float((t2d_df['clinvar_score'] > 0).mean()),
            "vep_coverage": float((t2d_df['vep_score'] > 0).mean()),
            "top_snp": t2d_df.loc[t2d_df['extended_score'].idxmax(), 'rsid'],
        }
    }

    summary_path = get_output_path("step3", "summary")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {summary_path}")

    return summary


def main():
    """Run Step 3B: Calculate Extended Scores."""
    print_step("STEP 3B: CALCULATE EXTENDED SCORES (NO REDUNDANCY)")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print(f"\n  Extended Score = {W_CAUSAL}×causal + {W_CLINVAR}×clinvar + {W_VEP}×vep + {W_EVIDENCE}×evidence + {W_EFFECT}×effect")
    print(f"  NOTE: evidence_score and effect_score kept SEPARATE (no double-counting)")

    # Load data
    ad_df, t2d_df = load_step3_data()

    # =========================================================================
    # PROCESS AD
    # =========================================================================
    print_step("PROCESSING AD", "=")

    # Verify effect_score exists from Step 2
    print(f"  effect_score from Step 2: [{ad_df['effect_score'].min():.4f}, {ad_df['effect_score'].max():.4f}]")
    print(f"  evidence_score from Step 2: [{ad_df['evidence_score'].min():.4f}, {ad_df['evidence_score'].max():.4f}]")

    ad_df = calculate_clinvar_score(ad_df)
    ad_df = calculate_vep_score(ad_df)
    # NO calculate_gwas_score - use effect_score directly from Step 2
    ad_df = calculate_extended_score(ad_df)

    show_component_contributions(ad_df, "AD")
    show_example(ad_df, "rs429358", "AD")  # APOE SNP

    # =========================================================================
    # PROCESS T2D
    # =========================================================================
    print_step("PROCESSING T2D", "=")

    # Verify effect_score exists from Step 2
    print(f"  effect_score from Step 2: [{t2d_df['effect_score'].min():.4f}, {t2d_df['effect_score'].max():.4f}]")
    print(f"  evidence_score from Step 2: [{t2d_df['evidence_score'].min():.4f}, {t2d_df['evidence_score'].max():.4f}]")

    t2d_df = calculate_clinvar_score(t2d_df)
    t2d_df = calculate_vep_score(t2d_df)
    # NO calculate_gwas_score - use effect_score directly from Step 2
    t2d_df = calculate_extended_score(t2d_df)

    show_component_contributions(t2d_df, "T2D")

    # Save
    summary = save_results(ad_df, t2d_df)

    print_step("STEP 3B COMPLETE")
    print(f"  AD:  {len(ad_df):,} SNPs with extended_score")
    print(f"  T2D: {len(t2d_df):,} SNPs with extended_score")
    print(f"\n  Extended score correlation with risk_score:")
    print(f"    AD:  {summary['ad']['risk_score_correlation']:.4f}")
    print(f"    T2D: {summary['t2d']['risk_score_correlation']:.4f}")

    return ad_df, t2d_df, summary


if __name__ == "__main__":
    ad_extended, t2d_extended, summary = main()
