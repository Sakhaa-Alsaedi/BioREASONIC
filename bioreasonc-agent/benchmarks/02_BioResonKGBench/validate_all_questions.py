#!/usr/bin/env python3
"""
Validate all CKGQA questions against the KG.
"""

import json
import yaml
from pathlib import Path
from neo4j import GraphDatabase

SCRIPT_DIR = Path(__file__).parent

def load_config():
    with open(SCRIPT_DIR / 'config' / 'kg_config.yml') as f:
        return yaml.safe_load(f)

def get_driver(cfg):
    return GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

def validate_question(session, q):
    """Validate a single question."""
    try:
        result = session.run(q['cypher'])
        data = list(result)
        if not data:
            return False, "No results"

        answer_key = q.get('answer_key', 'answer')
        val = data[0].get(answer_key)

        if val is None:
            return False, f"No value for key '{answer_key}'"

        if isinstance(val, list) and len(val) == 0:
            return False, "Empty list result"

        return True, val
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 80)
    print("VALIDATING ALL CKGQA QUESTIONS")
    print("=" * 80)
    print()

    cfg = load_config()
    driver = get_driver(cfg)
    data_dir = SCRIPT_DIR / 'data'

    # Load combined files
    with open(data_dir / 'combined_CKGQA_dev_matched.json') as f:
        dev_questions = json.load(f)
    with open(data_dir / 'combined_CKGQA_test_matched.json') as f:
        test_questions = json.load(f)

    print(f"Dev questions: {len(dev_questions)}")
    print(f"Test questions: {len(test_questions)}")
    print()

    # Validate
    results = {'dev': {'valid': 0, 'invalid': 0, 'by_taxonomy': {}},
               'test': {'valid': 0, 'invalid': 0, 'by_taxonomy': {}}}

    with driver.session() as session:
        for split, questions in [('dev', dev_questions), ('test', test_questions)]:
            print(f"Validating {split}...")

            for q in questions:
                taxonomy = q.get('taxonomy', 'Unknown')
                if taxonomy not in results[split]['by_taxonomy']:
                    results[split]['by_taxonomy'][taxonomy] = {'valid': 0, 'invalid': 0}

                valid, _ = validate_question(session, q)
                if valid:
                    results[split]['valid'] += 1
                    results[split]['by_taxonomy'][taxonomy]['valid'] += 1
                else:
                    results[split]['invalid'] += 1
                    results[split]['by_taxonomy'][taxonomy]['invalid'] += 1

    driver.close()

    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    for split in ['dev', 'test']:
        total = results[split]['valid'] + results[split]['invalid']
        pct = 100 * results[split]['valid'] / total if total > 0 else 0
        print(f"\n{split.upper()}: {results[split]['valid']}/{total} valid ({pct:.1f}%)")

        for taxonomy in sorted(results[split]['by_taxonomy'].keys()):
            stats = results[split]['by_taxonomy'][taxonomy]
            total_tax = stats['valid'] + stats['invalid']
            pct_tax = 100 * stats['valid'] / total_tax if total_tax > 0 else 0
            print(f"  {taxonomy}: {stats['valid']}/{total_tax} valid ({pct_tax:.1f}%)")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
