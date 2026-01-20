#!/usr/bin/env python3
"""
Evaluate LLMs with KG Access on both BioKGBench and BioResonKGBench.
LLMs generate Cypher queries that are executed against the KG.
"""

import json
import yaml
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Setup paths - use absolute paths
SCRIPT_DIR = Path('/ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/benchmarks/02_BioResonKGBench')
BIOKGBENCH_DIR = Path('/ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/benchmarks/01_BioKGBench')
SRC_DIR = BIOKGBENCH_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from neo4j import GraphDatabase

# Import KG QA system
from kg_qa_system_v2 import KnowledgeGraphQAv2, QAMode

def load_config():
    """Load configuration and set API keys."""
    with open(SCRIPT_DIR / 'config' / 'config.local.yaml') as f:
        config = yaml.safe_load(f)

    llm_config = config.get('llm', {})
    if llm_config.get('openai', {}).get('api_key'):
        os.environ['OPENAI_API_KEY'] = llm_config['openai']['api_key']
    if llm_config.get('claude', {}).get('api_key'):
        os.environ['ANTHROPIC_API_KEY'] = llm_config['claude']['api_key']
    if llm_config.get('together', {}).get('api_key'):
        os.environ['TOGETHER_API_KEY'] = llm_config['together']['api_key']

    return config

def get_neo4j_driver():
    """Get Neo4j driver."""
    with open(SCRIPT_DIR / 'config' / 'kg_config.yml') as f:
        cfg = yaml.safe_load(f)

    return GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    return str(answer).strip().lower()

# ============================================================================
# Load Questions
# ============================================================================

def load_biokgbench_dev(n_samples=5):
    """Load BioKGBench dev questions."""
    with open(BIOKGBENCH_DIR / 'data' / 'dev.json') as f:
        questions = json.load(f)

    samples = []
    types_seen = set()
    for q in questions:
        qtype = q.get('type', 'unknown')
        if qtype not in types_seen and len(samples) < n_samples:
            samples.append(q)
            types_seen.add(qtype)

    for q in questions:
        if q not in samples and len(samples) < n_samples:
            samples.append(q)

    return samples[:n_samples]

def load_bioresonkgbench_dev(n_samples=5):
    """Load BioResonKGBench dev questions."""
    with open(SCRIPT_DIR / 'data' / 'combined_CKGQA_dev_matched.json') as f:
        questions = json.load(f)

    # Get balanced samples across taxonomies
    samples = []
    per_tax = max(1, n_samples // 4)

    for tax in ['S', 'R', 'C', 'M']:
        tax_qs = [q for q in questions if q['taxonomy'] == tax]
        for q in tax_qs[:per_tax]:
            if len(samples) < n_samples:
                samples.append(q)

    # Fill remaining with any questions
    for q in questions:
        if q not in samples and len(samples) < n_samples:
            samples.append(q)

    return samples[:n_samples]

# ============================================================================
# Gold Answer Extraction
# ============================================================================

def get_biokgbench_gold(question):
    """Extract gold answers from BioKGBench question."""
    answers = set()
    answer_field = question.get('answer', [])

    if isinstance(answer_field, list):
        for ans in answer_field:
            if isinstance(ans, dict):
                if 'answer' in ans:
                    answers.add(normalize_answer(ans['answer']))
                if 'name' in ans:
                    answers.add(normalize_answer(ans['name']))
            else:
                answers.add(normalize_answer(str(ans)))
    else:
        answers.add(normalize_answer(str(answer_field)))

    return answers

def get_bioresonkgbench_gold(question, driver):
    """Extract gold answers by running Cypher query."""
    cypher = question.get('cypher', '')
    answer_key = question.get('answer_key', '')

    if not cypher or not answer_key:
        return set()

    answers = set()
    try:
        with driver.session() as session:
            result = session.run(cypher)
            for record in result:
                if answer_key in record.keys():
                    value = record[answer_key]
                    if value is not None:
                        if isinstance(value, list):
                            for v in value[:5]:
                                answers.add(normalize_answer(str(v)))
                        else:
                            answers.add(normalize_answer(str(value)))
    except:
        pass

    return answers

# ============================================================================
# Check Correctness
# ============================================================================

def check_correctness(predicted_answers, gold_answers):
    """Check if any predicted answer matches gold."""
    if not predicted_answers or not gold_answers:
        return False

    for pred in predicted_answers:
        pred_norm = normalize_answer(pred)
        if not pred_norm:
            continue

        for gold in gold_answers:
            if not gold:
                continue
            if gold in pred_norm or pred_norm in gold:
                return True
            # Check partial matches
            pred_words = set(pred_norm.replace(',', ' ').replace('.', ' ').split())
            gold_words = set(gold.replace(',', ' ').replace('.', ' ').split())
            if gold_words & pred_words:
                return True

    return False

# ============================================================================
# Create Config for Each Model
# ============================================================================

def create_model_config(model_name, base_config):
    """Create a config file for a specific model."""
    config = base_config.copy()
    config['llm']['provider'] = model_name

    temp_path = f'/tmp/kg_eval_config_{model_name.replace("/", "_")}.yaml'
    with open(temp_path, 'w') as f:
        yaml.dump(config, f)

    return temp_path

# ============================================================================
# Main Evaluation with KG Access
# ============================================================================

def evaluate_with_kg_access(benchmark_name, questions, gold_func, models, base_config, driver=None):
    """Evaluate using KG-connected LLM approach."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {benchmark_name} (with KG Access)")
    print(f"{'='*80}")
    print(f"Questions: {len(questions)}")

    results = {}

    for model_name in models:
        print(f"\n--- Model: {model_name} ---")

        # Create config for this model
        config_path = create_model_config(model_name, base_config)

        try:
            # Initialize KG QA system with LLM mode
            qa = KnowledgeGraphQAv2(config_path=config_path, mode=QAMode.LLM)
            qa.connect()
        except Exception as e:
            print(f"  Failed to initialize: {e}")
            results[model_name] = {'correct': 0, 'total': len(questions), 'accuracy': 0, 'error': str(e)}
            continue

        model_results = []
        correct = 0

        for i, q in enumerate(questions):
            question_text = q.get('question', '')
            qtype = q.get('taxonomy', q.get('type', 'unknown'))

            # Get gold answers
            if driver:
                gold = gold_func(q, driver)
            else:
                gold = gold_func(q)

            print(f"  Q{i+1} [{qtype}]: {question_text[:50]}...")
            print(f"    Gold: {list(gold)[:3]}")

            # Get LLM prediction via KG
            try:
                result = qa.answer(question_text, q)

                predicted = []
                if result.success and result.answers:
                    for ans in result.answers:
                        if ans.name:
                            predicted.append(ans.name)
                        if ans.id:
                            predicted.append(ans.id)

                cypher_generated = result.cypher_query or "None"
                error = result.error

            except Exception as e:
                predicted = []
                cypher_generated = "ERROR"
                error = str(e)

            print(f"    Cypher: {cypher_generated[:60]}..." if cypher_generated else "    Cypher: None")
            print(f"    Pred: {predicted[:3]}")

            # Check correctness
            is_correct = check_correctness(predicted, gold)
            if is_correct:
                correct += 1
                print(f"    Result: ✓ CORRECT")
            else:
                print(f"    Result: ✗ WRONG")

            model_results.append({
                'question': question_text,
                'type': qtype,
                'gold': list(gold)[:5],
                'predicted': predicted[:5],
                'cypher': cypher_generated[:200] if cypher_generated else None,
                'correct': is_correct,
                'error': error
            })

            time.sleep(0.5)

        qa.close()

        accuracy = correct / len(questions) * 100 if questions else 0
        results[model_name] = {
            'correct': correct,
            'total': len(questions),
            'accuracy': accuracy,
            'details': model_results
        }
        print(f"\n  {model_name}: {correct}/{len(questions)} ({accuracy:.1f}%)")

    return results

def main():
    print("=" * 80)
    print("DUAL BENCHMARK EVALUATION WITH KG ACCESS")
    print("LLMs generate Cypher → Execute on Neo4j → Get Answers")
    print("=" * 80)
    print()

    # Load config
    base_config = load_config()
    print("API keys loaded")

    # Models to test
    models = [
        'gpt-4o-mini',
        'gpt-4o',
        'claude-3-haiku',
    ]

    N_SAMPLES = 10

    # Get Neo4j driver
    driver = get_neo4j_driver()

    # ========================================================================
    # Evaluate BioKGBench with KG Access
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading BioKGBench...")
    biokgbench_questions = load_biokgbench_dev(N_SAMPLES)
    print(f"Loaded {len(biokgbench_questions)} questions")

    biokgbench_results = evaluate_with_kg_access(
        "BioKGBench (01)",
        biokgbench_questions,
        get_biokgbench_gold,
        models,
        base_config
    )

    # ========================================================================
    # Evaluate BioResonKGBench with KG Access
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading BioResonKGBench...")
    bioresonkgbench_questions = load_bioresonkgbench_dev(N_SAMPLES)
    print(f"Loaded {len(bioresonkgbench_questions)} questions")

    bioresonkgbench_results = evaluate_with_kg_access(
        "BioResonKGBench (02)",
        bioresonkgbench_questions,
        get_bioresonkgbench_gold,
        models,
        base_config,
        driver=driver
    )

    driver.close()

    # ========================================================================
    # Summary Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (WITH KG ACCESS)")
    print("=" * 80)
    print()
    print(f"{'Model':<20} {'BioKGBench':>15} {'BioResonKGBench':>18}")
    print("-" * 60)

    for model_name in models:
        bio_acc = biokgbench_results.get(model_name, {}).get('accuracy', 0)
        biores_acc = bioresonkgbench_results.get(model_name, {}).get('accuracy', 0)
        print(f"{model_name:<20} {bio_acc:>14.1f}% {biores_acc:>17.1f}%")

    print("-" * 60)

    # Average
    bio_avg = sum(r.get('accuracy', 0) for r in biokgbench_results.values()) / len(models) if biokgbench_results else 0
    biores_avg = sum(r.get('accuracy', 0) for r in bioresonkgbench_results.values()) / len(models) if bioresonkgbench_results else 0
    print(f"{'AVERAGE':<20} {bio_avg:>14.1f}% {biores_avg:>17.1f}%")
    print("=" * 60)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': timestamp,
        'mode': 'KG_ACCESS',
        'n_samples': N_SAMPLES,
        'models': models,
        'BioKGBench': biokgbench_results,
        'BioResonKGBench': bioresonkgbench_results
    }

    output_path = SCRIPT_DIR / 'results' / f'kg_access_eval_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
