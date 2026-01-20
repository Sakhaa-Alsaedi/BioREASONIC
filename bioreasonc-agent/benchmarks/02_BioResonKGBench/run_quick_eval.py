#!/usr/bin/env python3
"""Quick evaluation test for BioResonKGBench with fixes."""

import sys
import json
import yaml
import time
from pathlib import Path
from collections import defaultdict
from neo4j import GraphDatabase

# Setup paths
SCRIPT_DIR = Path(__file__).parent
BIOKGBENCH_DIR = SCRIPT_DIR.parent / '01_BioKGBench'
sys.path.insert(0, str(BIOKGBENCH_DIR / 'src'))

from kg_qa_system_v2 import KnowledgeGraphQAv2, QAMode

def normalize_answer(answer):
    if answer is None:
        return ''
    return str(answer).strip().lower()

def get_neo4j_driver():
    config_path = SCRIPT_DIR / 'config' / 'kg_config.yml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

def extract_gold_answers(q, driver):
    answers = set()
    cypher = q.get('cypher', '')
    answer_key = q.get('answer_key', '')

    if not cypher or not answer_key:
        return answers

    try:
        with driver.session() as session:
            result = session.run(cypher)
            for record in result:
                if answer_key in record.keys():
                    value = record[answer_key]
                    if value is not None:
                        answers.add(normalize_answer(str(value)))
    except Exception as e:
        pass

    return answers

def main():
    print("="*60)
    print("QUICK EVALUATION TEST")
    print("="*60)

    # Load questions
    with open(SCRIPT_DIR / 'data' / 'combined_dev.json') as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions")

    # Pre-filter valid questions
    print("\nPre-filtering questions with valid gold answers...")
    driver = get_neo4j_driver()

    valid_questions = []
    for q in questions:
        gold = extract_gold_answers(q, driver)
        if gold:
            q['_gold_answers'] = gold
            valid_questions.append(q)

    driver.close()
    print(f"Valid questions: {len(valid_questions)}/{len(questions)}")

    # Test with one model
    model_name = 'gpt-4o-mini'
    max_questions = 10

    print(f"\nEvaluating {model_name} on {max_questions} questions...")

    # Load config
    config_path = BIOKGBENCH_DIR / 'config' / 'config.local.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config['llm']['provider'] = model_name

    temp_config = '/tmp/quick_eval_config.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    # Initialize QA system
    qa = KnowledgeGraphQAv2(config_path=temp_config, mode=QAMode.LLM)
    qa.connect()

    # Evaluate
    results = []
    start_time = time.time()

    for i, q in enumerate(valid_questions[:max_questions]):
        print(f"\n--- Question {i+1}/{max_questions} ---")
        print(f"Q: {q['question'][:60]}...")

        gold_answers = q['_gold_answers']
        print(f"Gold: {list(gold_answers)[:3]}")

        try:
            result = qa.answer(q['question'], q)
            predicted = []

            if result.success and result.answers:
                for ans in result.answers:
                    if ans.name:
                        predicted.append(normalize_answer(ans.name))
                    if ans.id:
                        predicted.append(normalize_answer(ans.id))

            print(f"Pred: {predicted[:3]}")

            # Check correctness
            correct = False
            for p in predicted:
                if p in gold_answers:
                    correct = True
                    break
                for g in gold_answers:
                    if g in p or p in g:
                        correct = True
                        break

            print(f"Correct: {correct}")
            results.append({'correct': correct, 'success': result.success})

        except Exception as e:
            print(f"Error: {e}")
            results.append({'correct': False, 'success': False})

    qa.close()
    total_time = time.time() - start_time

    # Summary
    correct = sum(1 for r in results if r['correct'])
    success = sum(1 for r in results if r['success'])

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Questions: {len(results)}")
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
    print(f"Executability: {success}/{len(results)} = {success/len(results)*100:.1f}%")
    print(f"Time: {total_time:.1f}s")
    print("="*60)

if __name__ == "__main__":
    main()
