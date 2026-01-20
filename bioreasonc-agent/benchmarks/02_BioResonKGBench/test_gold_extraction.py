#!/usr/bin/env python3
"""Test gold answer extraction from Cypher queries."""

import sys
import json
import yaml
from pathlib import Path
from neo4j import GraphDatabase

# Setup paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent / '01_BioKGBench' / 'src'))

def normalize_answer(answer):
    if answer is None:
        return ''
    return str(answer).strip().lower()

def main():
    print("="*60)
    print("GOLD ANSWER EXTRACTION TEST")
    print("="*60)

    # Load config
    config_path = SCRIPT_DIR / 'config' / 'kg_config.yml'
    with open(config_path) as f:
        neo4j_config = yaml.safe_load(f)

    uri = f"bolt://{neo4j_config['db_url']}:{neo4j_config['db_port']}"
    print(f"Connecting to: {uri}")

    driver = GraphDatabase.driver(
        uri,
        auth=(neo4j_config['db_user'], neo4j_config['db_password']),
        encrypted=False
    )

    # Test connection
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
        count = result.single()['count']
        print(f"Connected! Node count: {count:,}")

    # Load questions
    with open(SCRIPT_DIR / 'data' / 'combined_dev.json') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Test gold answer extraction
    print("\n" + "="*60)
    print("TESTING GOLD ANSWER EXTRACTION ON 5 QUESTIONS")
    print("="*60)

    success_count = 0
    for i, q in enumerate(questions[:5]):
        print(f"\nQ{i+1}: {q['question'][:55]}...")
        print(f"    Task: {q['task_id']}, Type: {q.get('type', 'N/A')}")
        print(f"    Answer Key: {q.get('answer_key', 'N/A')}")

        cypher = q.get('cypher', '')
        answer_key = q.get('answer_key', '')

        if cypher and answer_key:
            try:
                with driver.session() as session:
                    result = session.run(cypher)
                    records = list(result)

                    gold_answers = set()
                    for record in records:
                        if answer_key in record.keys():
                            value = record[answer_key]
                            if value is not None:
                                gold_answers.add(normalize_answer(str(value)))

                    if gold_answers:
                        print(f"    ✓ Gold Answers: {list(gold_answers)[:3]}{'...' if len(gold_answers)>3 else ''}")
                        print(f"    ✓ Count: {len(gold_answers)} answers")
                        success_count += 1
                    else:
                        print(f"    ✗ No answers found (query returned {len(records)} records)")
                        # Show what columns are available
                        if records:
                            print(f"    Available columns: {list(records[0].keys())}")

            except Exception as e:
                print(f"    ✗ Error: {e}")
        else:
            print(f"    ✗ Missing cypher or answer_key")

    driver.close()

    print("\n" + "="*60)
    print(f"RESULT: {success_count}/5 questions had extractable gold answers")
    print("="*60)

    return success_count

if __name__ == "__main__":
    main()
