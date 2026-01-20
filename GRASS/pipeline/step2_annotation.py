#!/usr/bin/env python3
"""
STEP 3: Annotate SNPs with ClinVar and VEP
==========================================
Add clinical and functional annotations to scored SNPs.

Input:  Scored SNPs from Step 2
Output: Annotated SNPs with ClinVar and VEP data

Annotation Sources:
1. ClinVar - Clinical significance (pathogenic, benign, etc.)
2. VEP - Variant effect (missense, synonymous, etc.)

Note: Uses cached results to avoid repeated API calls.
"""

import pandas as pd
import numpy as np
import json
import requests
import time
from datetime import datetime
from pathlib import Path

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    OUTPUT_DIR, CACHE_DIR, get_output_path,
    CLINVAR_ESEARCH, CLINVAR_ESUMMARY, ENSEMBL_REST,
    API_DELAY, MAX_RETRIES
)


def print_step(msg, char="="):
    """Print formatted step message."""
    print(f"\n{char * 70}")
    print(f"  {msg}")
    print(f"{char * 70}")


def load_step2_data():
    """Load scored SNP data from Step 2."""
    print_step("Loading Step 2 Data", "-")

    ad_path = get_output_path("step2", "ad_scored")
    t2d_path = get_output_path("step2", "t2d_scored")

    if not ad_path.exists() or not t2d_path.exists():
        raise FileNotFoundError("Run Step 2 first to calculate scores")

    ad_df = pd.read_csv(ad_path)
    t2d_df = pd.read_csv(t2d_path)

    print(f"  AD SNPs loaded:  {len(ad_df):,}")
    print(f"  T2D SNPs loaded: {len(t2d_df):,}")

    return ad_df, t2d_df


def api_request(url, params=None, headers=None, method='GET', data=None):
    """Make API request with retry logic."""
    if headers is None:
        headers = {"Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(API_DELAY)

            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=30)
            else:
                response = requests.post(url, params=params, headers=headers,
                                         data=data, timeout=30)

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'json' in content_type:
                    return response.json()
                return response.text

            elif response.status_code == 429:  # Rate limited
                wait_time = int(response.headers.get('Retry-After', 60))
                print(f"    Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)

        except requests.exceptions.Timeout:
            pass
        except requests.exceptions.RequestException:
            pass

        time.sleep(2 ** attempt)  # Exponential backoff

    return None


# =============================================================================
# CLINVAR ANNOTATION
# =============================================================================

def get_clinvar_annotation(rsid, cache):
    """Get ClinVar annotation for a single rsID."""
    if rsid in cache:
        return cache[rsid]

    # Search ClinVar
    search_params = {
        'db': 'clinvar',
        'term': f'{rsid}[RS]',
        'retmode': 'json',
        'retmax': 5
    }

    search_result = api_request(CLINVAR_ESEARCH, params=search_params)

    if not search_result or 'esearchresult' not in search_result:
        result = {'clinvar_id': '', 'clinical_significance': 'not_in_clinvar', 'conditions': ''}
        cache[rsid] = result
        return result

    id_list = search_result.get('esearchresult', {}).get('idlist', [])

    if not id_list:
        result = {'clinvar_id': '', 'clinical_significance': 'not_in_clinvar', 'conditions': ''}
        cache[rsid] = result
        return result

    # Get summary
    summary_params = {
        'db': 'clinvar',
        'id': ','.join(id_list[:3]),
        'retmode': 'json'
    }

    summary_result = api_request(CLINVAR_ESUMMARY, params=summary_params)

    if not summary_result or 'result' not in summary_result:
        result = {'clinvar_id': '', 'clinical_significance': 'api_error', 'conditions': ''}
        cache[rsid] = result
        return result

    # Extract first result
    for uid in id_list[:1]:
        if uid in summary_result['result']:
            entry = summary_result['result'][uid]
            result = {
                'clinvar_id': uid,
                'clinical_significance': entry.get('clinical_significance', {}).get('description', ''),
                'review_status': entry.get('clinical_significance', {}).get('review_status', ''),
                'conditions': ','.join([t.get('trait_name', '') for t in entry.get('trait_set', [])[:3]])
            }
            cache[rsid] = result
            return result

    result = {'clinvar_id': '', 'clinical_significance': 'not_found', 'conditions': ''}
    cache[rsid] = result
    return result


def annotate_clinvar(df, disease_name, max_snps=500):
    """Annotate SNPs with ClinVar data."""
    print_step(f"ClinVar Annotation ({disease_name})", "-")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{disease_name.lower()}_clinvar_cache.json"

    # Load cache
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached annotations")

    # Get unique SNPs, prioritized by risk_score
    unique_snps = df.drop_duplicates('rsid').nlargest(max_snps, 'risk_score')['rsid'].tolist()

    print(f"  Annotating top {len(unique_snps)} SNPs...")

    annotations = {}
    cached_count = 0
    api_count = 0

    for i, rsid in enumerate(unique_snps):
        if rsid in cache:
            cached_count += 1
        else:
            api_count += 1

        annotations[rsid] = get_clinvar_annotation(rsid, cache)

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(unique_snps)} (cached: {cached_count}, API: {api_count})")

    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    # Convert to DataFrame
    ann_df = pd.DataFrame([
        {'rsid': rsid, **ann} for rsid, ann in annotations.items()
    ])

    # Show distribution
    if 'clinical_significance' in ann_df.columns:
        print(f"\n  Clinical significance distribution:")
        sig_counts = ann_df['clinical_significance'].value_counts()
        for sig, count in sig_counts.head(8).items():
            print(f"    {sig}: {count}")

    return ann_df


# =============================================================================
# VEP ANNOTATION
# =============================================================================

def get_vep_annotation(rsid, cache):
    """Get VEP annotation for a single rsID."""
    if rsid in cache:
        return cache[rsid]

    url = f"{ENSEMBL_REST}/vep/human/id/{rsid}"
    headers = {"Content-Type": "application/json"}

    data = api_request(url, headers=headers)

    if data and isinstance(data, list) and len(data) > 0:
        variant = data[0]

        # Extract consequences
        consequences = []
        genes_affected = []

        for tc in variant.get('transcript_consequences', []):
            consequences.extend(tc.get('consequence_terms', []))
            if tc.get('gene_symbol'):
                genes_affected.append(tc['gene_symbol'])

        result = {
            'most_severe_consequence': variant.get('most_severe_consequence', ''),
            'all_consequences': ','.join(set(consequences))[:200],
            'vep_genes': ','.join(set(genes_affected))[:100],
            'allele_string': variant.get('allele_string', ''),
        }
        cache[rsid] = result
        return result

    result = {'most_severe_consequence': 'not_found', 'all_consequences': '', 'vep_genes': '', 'allele_string': ''}
    cache[rsid] = result
    return result


def annotate_vep(df, disease_name, max_snps=200):
    """Annotate SNPs with VEP data."""
    print_step(f"VEP Annotation ({disease_name})", "-")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{disease_name.lower()}_vep_cache.json"

    # Load cache
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached annotations")

    # Get unique SNPs, prioritized by risk_score
    unique_snps = df.drop_duplicates('rsid').nlargest(max_snps, 'risk_score')['rsid'].tolist()

    print(f"  Annotating top {len(unique_snps)} SNPs...")

    annotations = {}
    cached_count = 0
    api_count = 0

    for i, rsid in enumerate(unique_snps):
        if rsid in cache:
            cached_count += 1
        else:
            api_count += 1

        annotations[rsid] = get_vep_annotation(rsid, cache)

        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(unique_snps)} (cached: {cached_count}, API: {api_count})")

    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    # Convert to DataFrame
    ann_df = pd.DataFrame([
        {'rsid': rsid, **ann} for rsid, ann in annotations.items()
    ])

    # Show distribution
    if 'most_severe_consequence' in ann_df.columns:
        print(f"\n  Consequence distribution:")
        conseq_counts = ann_df['most_severe_consequence'].value_counts()
        for conseq, count in conseq_counts.head(8).items():
            print(f"    {conseq}: {count}")

    return ann_df


# =============================================================================
# MERGE ANNOTATIONS
# =============================================================================

def merge_annotations(snp_df, clinvar_df, vep_df):
    """Merge all annotations into SNP DataFrame."""
    result = snp_df.copy()

    # Merge ClinVar
    if len(clinvar_df) > 0:
        clinvar_cols = [c for c in ['rsid', 'clinvar_id', 'clinical_significance', 'conditions', 'review_status']
                        if c in clinvar_df.columns]
        result = result.merge(clinvar_df[clinvar_cols], on='rsid', how='left')

    # Merge VEP
    if len(vep_df) > 0:
        vep_cols = [c for c in ['rsid', 'most_severe_consequence', 'all_consequences', 'vep_genes', 'allele_string']
                    if c in vep_df.columns]
        result = result.merge(vep_df[vep_cols], on='rsid', how='left')

    # Fill NaN for non-annotated SNPs
    fill_cols = ['clinical_significance', 'most_severe_consequence']
    for col in fill_cols:
        if col in result.columns:
            result[col] = result[col].fillna('not_annotated')

    return result


def save_results(ad_df, t2d_df, ad_clinvar, t2d_clinvar, ad_vep, t2d_vep):
    """Save annotated SNP data."""
    print_step("Saving Results", "-")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save annotated SNPs
    ad_path = get_output_path("step3", "ad_annotated")
    t2d_path = get_output_path("step3", "t2d_annotated")

    ad_df.to_csv(ad_path, index=False)
    t2d_df.to_csv(t2d_path, index=False)

    print(f"  Saved: {ad_path}")
    print(f"  Saved: {t2d_path}")

    # Calculate annotation stats
    def count_pathogenic(df):
        if 'clinical_significance' not in df.columns:
            return 0
        return df['clinical_significance'].str.contains('athogenic', na=False).sum()

    def count_coding(df):
        if 'most_severe_consequence' not in df.columns:
            return 0
        coding = ['missense_variant', 'frameshift_variant', 'stop_gained',
                  'start_lost', 'stop_lost', 'splice_donor_variant', 'splice_acceptor_variant']
        return df['most_severe_consequence'].isin(coding).sum()

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "sources": ["ClinVar", "Ensembl_VEP_GRCh37"],
        "ad": {
            "total_snps": len(ad_df),
            "clinvar_annotated": len(ad_clinvar),
            "vep_annotated": len(ad_vep),
            "pathogenic_variants": int(count_pathogenic(ad_df)),
            "coding_variants": int(count_coding(ad_df)),
        },
        "t2d": {
            "total_snps": len(t2d_df),
            "clinvar_annotated": len(t2d_clinvar),
            "vep_annotated": len(t2d_vep),
            "pathogenic_variants": int(count_pathogenic(t2d_df)),
            "coding_variants": int(count_coding(t2d_df)),
        }
    }

    summary_path = get_output_path("step3", "summary")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {summary_path}")

    return summary


def main():
    """Run Step 3: Annotate SNPs."""
    print_step("STEP 3: ANNOTATE SNPs (ClinVar + VEP)")
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Load data
    ad_df, t2d_df = load_step2_data()

    # Annotate AD
    print_step("ANNOTATING AD SNPs", "=")
    ad_clinvar = annotate_clinvar(ad_df, "AD", max_snps=500)
    ad_vep = annotate_vep(ad_df, "AD", max_snps=200)
    ad_annotated = merge_annotations(ad_df, ad_clinvar, ad_vep)

    # Annotate T2D
    print_step("ANNOTATING T2D SNPs", "=")
    t2d_clinvar = annotate_clinvar(t2d_df, "T2D", max_snps=500)
    t2d_vep = annotate_vep(t2d_df, "T2D", max_snps=200)
    t2d_annotated = merge_annotations(t2d_df, t2d_clinvar, t2d_vep)

    # Save
    summary = save_results(ad_annotated, t2d_annotated, ad_clinvar, t2d_clinvar, ad_vep, t2d_vep)

    print_step("STEP 3 COMPLETE")
    print(f"  AD annotated SNPs:  {len(ad_annotated):,}")
    print(f"  T2D annotated SNPs: {len(t2d_annotated):,}")

    return ad_annotated, t2d_annotated, summary


if __name__ == "__main__":
    ad_annotated, t2d_annotated, summary = main()
