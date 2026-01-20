#!/usr/bin/env python3
"""
Evaluate LLMs on both BioKGBench and BioResonKGBench dev sets.
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
    """Get Neo4j driver for BioResonKGBench."""
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
# BioKGBench Functions
# ============================================================================

def load_biokgbench_dev(n_samples=5):
    """Load BioKGBench dev questions."""
    with open(BIOKGBENCH_DIR / 'data' / 'dev.json') as f:
        questions = json.load(f)

    # Select diverse samples by type
    samples = []
    types_seen = set()
    for q in questions:
        qtype = q.get('type', 'unknown')
        if qtype not in types_seen and len(samples) < n_samples:
            samples.append(q)
            types_seen.add(qtype)

    # Fill remaining
    for q in questions:
        if q not in samples and len(samples) < n_samples:
            samples.append(q)

    return samples[:n_samples]

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

# ============================================================================
# BioResonKGBench Functions
# ============================================================================

def load_bioresonkgbench_dev(n_samples=5):
    """Load BioResonKGBench dev questions."""
    with open(SCRIPT_DIR / 'data' / 'combined_CKGQA_dev_matched.json') as f:
        questions = json.load(f)

    # Select diverse samples by taxonomy
    samples = []
    taxonomies = ['S', 'R', 'C', 'M']
    for tax in taxonomies:
        q = next((q for q in questions if q['taxonomy'] == tax), None)
        if q and len(samples) < n_samples:
            samples.append(q)

    # Fill remaining
    for q in questions:
        if q not in samples and len(samples) < n_samples:
            samples.append(q)

    return samples[:n_samples]

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
    except Exception as e:
        pass

    return answers

# ============================================================================
# LLM Evaluation Functions
# ============================================================================

def evaluate_with_openai(question_text, model_name):
    """Evaluate question using OpenAI models."""
    from openai import OpenAI

    client = OpenAI()

    prompt = f"""You are a biomedical knowledge graph expert. Answer the following question based on your knowledge of genetics, proteins, diseases, and biological pathways.

Question: {question_text}

Provide a concise answer. If it's a list of items, separate them with commas. Just give the answer, no explanation needed."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

def evaluate_with_anthropic(question_text, model_name):
    """Evaluate question using Anthropic models."""
    from anthropic import Anthropic

    client = Anthropic()

    prompt = f"""You are a biomedical knowledge graph expert. Answer the following question based on your knowledge of genetics, proteins, diseases, and biological pathways.

Question: {question_text}

Provide a concise answer. If it's a list of items, separate them with commas. Just give the answer, no explanation needed."""

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def check_correctness(predicted, gold_answers):
    """Check if prediction matches any gold answer."""
    if not predicted or not gold_answers or predicted.startswith("ERROR"):
        return False

    pred_normalized = normalize_answer(predicted)

    for gold in gold_answers:
        if not gold:
            continue
        if gold in pred_normalized or pred_normalized in gold:
            return True
        # Check individual words
        pred_words = set(pred_normalized.replace(',', ' ').replace('.', ' ').split())
        if gold in pred_words:
            return True

    return False

def evaluate_benchmark(name, questions, gold_func, models, driver=None):
    """Evaluate a benchmark with multiple models."""
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*80}")
    print(f"Questions: {len(questions)}")

    results = {}

    for model_name, provider in models:
        print(f"\n--- Model: {model_name} ---")
        model_results = []
        correct = 0

        for i, q in enumerate(questions):
            question_text = q.get('question', '')

            # Get gold answers
            if driver:
                gold = gold_func(q, driver)
            else:
                gold = gold_func(q)

            # Get question type/taxonomy
            qtype = q.get('taxonomy', q.get('type', 'unknown'))

            print(f"  Q{i+1} [{qtype}]: {question_text[:50]}...")
            print(f"    Gold: {list(gold)[:3]}")

            # Get prediction
            if provider == 'openai':
                pred = evaluate_with_openai(question_text, model_name)
            elif provider == 'anthropic':
                pred = evaluate_with_anthropic(question_text, model_name)
            else:
                pred = "ERROR: Unknown provider"

            print(f"    Pred: {pred[:60]}...")

            # Check correctness
            is_correct = check_correctness(pred, gold)
            if is_correct:
                correct += 1
                print(f"    Result: ✓ CORRECT")
            else:
                print(f"    Result: ✗ WRONG")

            model_results.append({
                'question': question_text,
                'type': qtype,
                'gold': list(gold)[:5],
                'predicted': pred[:200],
                'correct': is_correct
            })

            time.sleep(0.5)  # Rate limiting

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
    print("DUAL BENCHMARK EVALUATION")
    print("BioKGBench vs BioResonKGBench")
    print("=" * 80)
    print()

    # Load config
    config = load_config()
    print("API keys loaded")

    # Models to test
    models = [
        ('gpt-4o-mini', 'openai'),
        ('gpt-4o', 'openai'),
        ('claude-3-haiku-20240307', 'anthropic'),
    ]

    N_SAMPLES = 5

    # ========================================================================
    # Evaluate BioKGBench
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading BioKGBench...")
    biokgbench_questions = load_biokgbench_dev(N_SAMPLES)
    print(f"Loaded {len(biokgbench_questions)} questions")

    biokgbench_results = evaluate_benchmark(
        "BioKGBench (01)",
        biokgbench_questions,
        get_biokgbench_gold,
        models
    )

    # ========================================================================
    # Evaluate BioResonKGBench
    # ========================================================================
    print("\n" + "=" * 80)
    print("Loading BioResonKGBench...")
    bioresonkgbench_questions = load_bioresonkgbench_dev(N_SAMPLES)
    print(f"Loaded {len(bioresonkgbench_questions)} questions")

    driver = get_neo4j_driver()
    bioresonkgbench_results = evaluate_benchmark(
        "BioResonKGBench (02)",
        bioresonkgbench_questions,
        get_bioresonkgbench_gold,
        models,
        driver=driver
    )
    driver.close()

    # ========================================================================
    # Summary Comparison
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Model':<30} {'BioKGBench':>15} {'BioResonKGBench':>18}")
    print("-" * 80)

    for model_name, _ in models:
        bio_acc = biokgbench_results.get(model_name, {}).get('accuracy', 0)
        biores_acc = bioresonkgbench_results.get(model_name, {}).get('accuracy', 0)
        print(f"{model_name:<30} {bio_acc:>14.1f}% {biores_acc:>17.1f}%")

    print("-" * 80)

    # Average
    bio_avg = sum(r.get('accuracy', 0) for r in biokgbench_results.values()) / len(models)
    biores_avg = sum(r.get('accuracy', 0) for r in bioresonkgbench_results.values()) / len(models)
    print(f"{'AVERAGE':<30} {bio_avg:>14.1f}% {biores_avg:>17.1f}%")
    print("=" * 80)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = {
        'timestamp': timestamp,
        'n_samples': N_SAMPLES,
        'models': [m[0] for m in models],
        'BioKGBench': biokgbench_results,
        'BioResonKGBench': bioresonkgbench_results
    }

    output_path = SCRIPT_DIR / 'results' / f'dual_benchmark_eval_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
