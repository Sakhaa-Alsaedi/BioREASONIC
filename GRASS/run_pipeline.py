#!/usr/bin/env python3
"""
GRASS Pipeline Runner
=====================
Run the complete GRASS pipeline from data loading to enrichment analysis.

Usage:
    python run_pipeline.py [options]

Options:
    --skip-annotations    Skip Step 2 (use cached annotations)
    --skip-enrichment     Skip Step 7 (enrichment analysis)
    --steps STEPS         Run specific steps (comma-separated, e.g., "0,3,5")

Pipeline Steps:
    Step 0: Data Loading        - Load and QC filter SNP data
    Step 1: SNP Scoring         - Extract PIPs and calculate risk scores
    Step 2: Annotation          - Annotate with ClinVar + VEP
    Step 3: Evidence Scores     - Calculate 4-component evidence scores
    Step 4: Gene Mapping        - Map SNPs to genes (500kb window)
    Step 5: GRASS Scoring       - Calculate GRASS gene scores
    Step 6: Enrichment Prep     - Prepare normalized rankings
    Step 7: Enrichment Analysis - Run Enrichr + STRING PPI
"""

import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent / "pipeline"

STEPS = [
    ("step0_data_loading.py", "Step 0: Data Loading", "0"),
    ("step1_snp_scoring.py", "Step 1: SNP Scoring", "1"),
    ("step2_annotation.py", "Step 2: Annotation (ClinVar + VEP)", "2"),
    ("step3_evidence_scores.py", "Step 3: Evidence Scores", "3"),
    ("step4_gene_mapping.py", "Step 4: Gene Mapping", "4"),
    ("step5_grass_scoring.py", "Step 5: GRASS Scoring", "5"),
    ("step6_enrichment_prep.py", "Step 6: Enrichment Prep", "6"),
    ("step7_enrichment_analysis.py", "Step 7: Enrichment Analysis", "7"),
]


def run_step(script, description, skip_annotations=False, skip_enrichment=False):
    """Run a pipeline step."""
    if skip_annotations and "step2_annotation" in script:
        print(f"\n{'='*70}")
        print(f"  SKIPPING: {description}")
        print(f"{'='*70}")
        return True

    if skip_enrichment and "step7" in script:
        print(f"\n{'='*70}")
        print(f"  SKIPPING: {description}")
        print(f"{'='*70}")
        return True

    print(f"\n{'='*70}")
    print(f"  RUNNING: {description}")
    print(f"{'='*70}\n")

    script_path = PIPELINE_DIR / script

    if not script_path.exists():
        print(f"  WARNING: Script not found: {script_path}")
        return True  # Continue with other steps

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(Path(__file__).parent)
    )

    if result.returncode != 0:
        print(f"\n  ERROR: {description} failed!")
        return False

    return True


def main():
    skip_annotations = "--skip-annotations" in sys.argv
    skip_enrichment = "--skip-enrichment" in sys.argv

    # Parse --steps argument
    specific_steps = None
    for i, arg in enumerate(sys.argv):
        if arg == "--steps" and i + 1 < len(sys.argv):
            specific_steps = sys.argv[i + 1].split(",")

    print("="*70)
    print("  GRASS PIPELINE RUNNER")
    print("  Genetic Risk Aggregation Scoring System")
    print("="*70)
    print("""
  Pipeline Steps:
    Step 0: Data Loading        - Load and QC filter SNP data
    Step 1: SNP Scoring         - Extract PIPs and calculate risk scores
    Step 2: Annotation          - Annotate with ClinVar + VEP
    Step 3: Evidence Scores     - Calculate 4-component evidence scores
    Step 4: Gene Mapping        - Map SNPs to genes (500kb window)
    Step 5: GRASS Scoring       - Calculate GRASS gene scores
    Step 6: Enrichment Prep     - Prepare normalized rankings
    Step 7: Enrichment Analysis - Run Enrichr + STRING PPI

  Options:
    --skip-annotations  Skip Step 2 (use cached annotations)
    --skip-enrichment   Skip Step 7 (enrichment analysis)
    --steps 0,3,5       Run only specific steps
""")

    for script, description, step_num in STEPS:
        # Check if we should run this step
        if specific_steps is not None:
            if step_num not in specific_steps:
                continue

        success = run_step(script, description, skip_annotations, skip_enrichment)
        if not success:
            print(f"\nPipeline failed at: {description}")
            sys.exit(1)

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    print("""
  Results saved to output/ directory:

  Gene Scores:
    - GRASS_master_scores.csv     : Combined master file
    - GRASS_AD_gene_scores.csv    : Alzheimer's Disease rankings
    - GRASS_T2D_gene_scores.csv   : Type 2 Diabetes rankings
    - ad_genes_normalized_ranking.csv
    - t2d_genes_normalized_ranking.csv
    - shared_genes_normalized_ranking.csv

  Enrichment (in enrichment/ directory):
    - enrichr_all_41_genes.csv    : Full enrichment results
    - ppi_network.graphml         : STRING PPI network
    - *.png                       : Visualization plots

  View top genes:
    head output/GRASS_master_scores.csv
""")


if __name__ == "__main__":
    main()
