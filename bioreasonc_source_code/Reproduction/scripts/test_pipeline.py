#!/usr/bin/env python3
"""
BioREASONIC-Bench Reproduction Test Script
==========================================

This script tests the complete question generation pipeline with sample data.
It demonstrates:
1. Loading GWAS data
2. Generating questions across all taxonomies (S, C, R, M)
3. Using expert-style answers for C and R taxonomies
4. Outputting results in JSON format

Usage:
    python scripts/test_pipeline.py

Author: Sakhaa Alsaedi
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.bioreasonc_creator.generator import QuestionGenerator
from src.bioreasonc_creator import generator as gen_module
USE_EXPERT_PROMPTS = gen_module.USE_EXPERT_PROMPTS


def main():
    """Run the complete pipeline test."""
    print("=" * 70)
    print("BioREASONIC-Bench Pipeline Test")
    print("=" * 70)

    # Configuration
    data_file = Path(__file__).parent.parent / "data" / "sample_gwas.csv"
    output_file = Path(__file__).parent.parent / "output" / "generated_questions.json"
    disease = "Type 2 Diabetes"

    print(f"\nConfiguration:")
    print(f"  - Input file: {data_file}")
    print(f"  - Disease: {disease}")
    print(f"  - Expert prompts enabled: {USE_EXPERT_PROMPTS}")
    print(f"  - Output file: {output_file}")

    # Check input file exists
    if not data_file.exists():
        print(f"\nError: Input file not found: {data_file}")
        return 1

    # Initialize generator
    print("\n" + "-" * 70)
    print("Step 1: Initializing Question Generator...")
    print("-" * 70)
    generator = QuestionGenerator()
    print(f"  - Loaded {len(generator.templates)} question templates")

    # Generate questions
    print("\n" + "-" * 70)
    print("Step 2: Generating Questions from GWAS Data...")
    print("-" * 70)
    items = generator.generate_from_csv(
        str(data_file),
        disease=disease,
        max_per_template=10
    )
    print(f"  - Generated {len(items)} questions")

    # Get statistics
    print("\n" + "-" * 70)
    print("Step 3: Pipeline Statistics")
    print("-" * 70)
    stats = generator.get_statistics(items)
    print(f"\n  Total items: {stats['total']}")
    print(f"\n  By Taxonomy:")
    for tax, count in sorted(stats['by_taxonomy'].items()):
        taxonomy_names = {
            'S': 'Structure-Aware',
            'C': 'Causal-Aware',
            'R': 'Risk-Aware',
            'M': 'Mechanism-Aware'
        }
        print(f"    {tax} ({taxonomy_names.get(tax, 'Unknown')}): {count}")

    print(f"\n  By Difficulty:")
    for diff, count in sorted(stats['by_difficulty'].items()):
        print(f"    {diff}: {count}")

    print(f"\n  By Answer Type:")
    for atype, count in sorted(stats['by_answer_type'].items()):
        print(f"    {atype}: {count}")

    # Display sample outputs
    print("\n" + "-" * 70)
    print("Step 4: Sample Generated Questions")
    print("-" * 70)

    # Show one example from each taxonomy
    shown_taxonomies = set()
    for item in items:
        if item.taxonomy not in shown_taxonomies:
            shown_taxonomies.add(item.taxonomy)
            print(f"\n{'='*60}")
            print(f"Taxonomy: {item.taxonomy} | Template: {item.label}")
            print(f"ID: {item.id} | Difficulty: {item.difficulty}")
            print(f"{'='*60}")
            print(f"\nQUESTION:")
            print(f"  {item.question}")
            print(f"\nANSWER:")
            # Truncate long answers for display
            answer_preview = item.answer[:500] + "..." if len(item.answer) > 500 else item.answer
            for line in answer_preview.split('\n'):
                print(f"  {line}")
            print(f"\nEntities: {item.entities}")

            # Show if expert prompts were used
            if item.source_data.get('expert_prompts_used'):
                print(f"\n[Expert-style answer generated]")

        if len(shown_taxonomies) == 4:
            break

    # Save output
    print("\n" + "-" * 70)
    print("Step 5: Saving Output...")
    print("-" * 70)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        'metadata': {
            'disease': disease,
            'input_file': str(data_file),
            'expert_prompts_enabled': USE_EXPERT_PROMPTS,
            'total_items': len(items)
        },
        'statistics': stats,
        'items': [item.to_dict() for item in items]
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  - Saved {len(items)} items to {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Test Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_file}")
    print("\nTo view the full output, run:")
    print(f"  cat {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
