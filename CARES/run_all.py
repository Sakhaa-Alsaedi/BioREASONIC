#!/usr/bin/env python3
"""
CARES Workflow - Run All Steps

This script runs the complete CARES workflow:
1. Generate taxonomy scores from raw results
2. Generate all plots (radar, bar, heatmap, line charts)

Usage:
    python run_all.py
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / 'src'
PLOTS_DIR = BASE_DIR / 'plots'


def run_script(script_name: str, description: str):
    """Run a Python script and handle errors."""
    script_path = SRC_DIR / script_name
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SRC_DIR),
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"WARNING: Script exited with code {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    print("="*70)
    print("CARES WORKFLOW - Complete Pipeline")
    print("="*70)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Source Directory: {SRC_DIR}")
    print(f"Plots Directory: {PLOTS_DIR}")

    # Ensure plots directory exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate taxonomy scores
    print("\n" + "="*70)
    print("STEP 1: Generate CARES Taxonomy Scores")
    print("="*70)
    run_script('generate_cares_taxonomy_scores.py', 'Generate taxonomy scores from raw results')

    # Step 2: Generate all plots
    plot_scripts = [
        ('generate_cares_radar.py', 'Radar Plot - Strategy Comparison'),
        ('generate_cares_bar.py', 'Bar Chart - CARES Scores'),
        ('generate_cares_heatmap.py', 'Heatmap - Model vs Strategy'),
        ('generate_cares_line.py', 'Line Chart - Strategy Progression'),
        ('generate_cares_grouped_bar.py', 'Grouped Bar - Category Breakdown'),
        ('generate_cares_models_comparison.py', 'Models Comparison - Radar & Variance'),
    ]

    print("\n" + "="*70)
    print("STEP 2: Generate All Plots")
    print("="*70)

    for script, description in plot_scripts:
        if (SRC_DIR / script).exists():
            run_script(script, description)
        else:
            print(f"Skipping {script} (not found)")

    # Summary
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("\n[Data]")
    for f in sorted((BASE_DIR / 'data' / 'processed').glob('*.csv')):
        print(f"  - {f.relative_to(BASE_DIR)}")
    print("\n[Plots]")
    for f in sorted(PLOTS_DIR.glob('*.png')):
        print(f"  - {f.relative_to(BASE_DIR)}")


if __name__ == '__main__':
    main()
