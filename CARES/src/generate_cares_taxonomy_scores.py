#!/usr/bin/env python3
"""
Generate CARES Taxonomy Scores from Raw Results

This script processes all_results.csv to create cares_taxonomy_scores.csv
which contains aggregated CARES scores by model, strategy, and taxonomy (S, C, R, M).

Input: data/input/all_results.csv
Output: data/processed/cares_taxonomy_scores.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / 'data' / 'input' / 'all_results.csv'
OUTPUT_FILE = BASE_DIR / 'data' / 'processed' / 'cares_taxonomy_scores.csv'


def compute_taxonomy_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CARES scores aggregated by model, strategy, and taxonomy.

    Args:
        df: DataFrame with columns: model, strategy, source, taxonomy, correct, score

    Returns:
        DataFrame with columns: model, strategy, C_Causal, M_Mechanism, R_Risk, S_Structure, Avg_SCRM
    """
    # Filter only BioREASONC data (has taxonomy)
    bio_df = df[df['source'].str.contains('BioREASONC', na=False)].copy()

    if len(bio_df) == 0:
        raise ValueError("No BioREASONC data found in input file")

    print(f"Processing {len(bio_df)} BioREASONC records...")

    # Taxonomy mapping
    taxonomy_names = {
        'C': 'C_Causal',
        'M': 'M_Mechanism',
        'R': 'R_Risk',
        'S': 'S_Structure'
    }

    taxonomy_scores = []

    for (model, strategy), group in bio_df.groupby(['model', 'strategy']):
        row = {'model': model, 'strategy': strategy}

        for tax, col_name in taxonomy_names.items():
            tax_data = group[group['taxonomy'] == tax]
            if len(tax_data) > 0:
                # Score is accuracy (correct=True -> 1, correct=False -> 0)
                if 'correct' in tax_data.columns:
                    avg_score = tax_data['correct'].astype(int).mean()
                elif 'score' in tax_data.columns:
                    avg_score = tax_data['score'].mean()
                else:
                    avg_score = 0.0
                row[col_name] = round(avg_score, 2)
            else:
                row[col_name] = 0.0

        # Compute average across SCRM categories
        scores = [row.get('C_Causal', 0), row.get('M_Mechanism', 0),
                  row.get('R_Risk', 0), row.get('S_Structure', 0)]
        row['Avg_SCRM'] = round(np.mean(scores), 3)

        taxonomy_scores.append(row)

    # Create DataFrame
    result_df = pd.DataFrame(taxonomy_scores)

    # Sort by strategy then by Avg_SCRM descending
    strategy_order = ['zero-shot', 'few-shot', 'cot', 'structured-cot']
    result_df['strategy_order'] = result_df['strategy'].map(
        {s: i for i, s in enumerate(strategy_order)}
    )
    result_df = result_df.sort_values(['strategy_order', 'Avg_SCRM'],
                                       ascending=[True, False])
    result_df = result_df.drop('strategy_order', axis=1)

    # Reorder columns
    columns = ['model', 'strategy', 'C_Causal', 'M_Mechanism', 'R_Risk', 'S_Structure', 'Avg_SCRM']
    result_df = result_df[columns]

    return result_df


def main():
    print("=" * 60)
    print("Generate CARES Taxonomy Scores")
    print("=" * 60)

    # Load input data
    print(f"\n[1] Loading input data from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"    Loaded {len(df)} records")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Models: {df['model'].unique().tolist()}")
    print(f"    Strategies: {df['strategy'].unique().tolist()}")

    # Compute taxonomy scores
    print(f"\n[2] Computing taxonomy scores...")
    result_df = compute_taxonomy_scores(df)
    print(f"    Generated {len(result_df)} model-strategy combinations")

    # Save output
    print(f"\n[3] Saving output to: {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_FILE, index=False)

    # Display results
    print(f"\n[4] Results Preview:")
    print(result_df.to_string(index=False))

    # Summary statistics
    print(f"\n[5] Summary by Strategy:")
    summary = result_df.groupby('strategy')[['C_Causal', 'M_Mechanism', 'R_Risk', 'S_Structure', 'Avg_SCRM']].mean()
    print(summary.round(3).to_string())

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
