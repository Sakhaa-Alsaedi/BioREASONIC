#!/usr/bin/env python3
"""
Quick LLM Evaluation on BioResonKGBench (5 samples from dev set)
"""

import json
import yaml
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Setup paths
SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent / '01_BioKGBench' / 'src'
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

def extract_gold_answer(question, driver):
    """Extract gold answer by running Cypher query."""
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
                            for v in value:
                                answers.add(normalize_answer(str(v)))
                        else:
                            answers.add(normalize_answer(str(value)))
    except Exception as e:
        print(f"  Cypher error: {e}")

    return answers

def evaluate_with_openai(question, model_name):
    """Evaluate question using OpenAI models."""
    from openai import OpenAI

    client = OpenAI()

    prompt = f"""You are a biomedical knowledge graph expert. Answer the following question based on your knowledge of genetics, diseases, and biological pathways.

Question: {question['question']}

Provide a concise answer. If it's a list, separate items with commas."""

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

def evaluate_with_anthropic(question, model_name):
    """Evaluate question using Anthropic models."""
    from anthropic import Anthropic

    client = Anthropic()

    prompt = f"""You are a biomedical knowledge graph expert. Answer the following question based on your knowledge of genetics, diseases, and biological pathways.

Question: {question['question']}

Provide a concise answer. If it's a list, separate items with commas."""

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"ERROR: {e}"

def evaluate_with_together(question, model_name):
    """Evaluate question using Together AI models."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get('TOGETHER_API_KEY'),
        base_url="https://api.together.xyz/v1"
    )

    prompt = f"""You are a biomedical knowledge graph expert. Answer the following question based on your knowledge of genetics, diseases, and biological pathways.

Question: {question['question']}

Provide a concise answer. If it's a list, separate items with commas."""

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

def check_correctness(predicted, gold_answers):
    """Check if prediction matches any gold answer."""
    if not predicted or not gold_answers:
        return False

    pred_normalized = normalize_answer(predicted)

    for gold in gold_answers:
        if gold in pred_normalized or pred_normalized in gold:
            return True
        # Check individual words for list answers
        pred_words = set(pred_normalized.replace(',', ' ').split())
        if gold in pred_words:
            return True

    return False

def main():
    print("=" * 80)
    print("BioResonKGBench Quick Evaluation (5 samples from dev set)")
    print("=" * 80)
    print()

    # Load config
    config = load_config()
    print("API keys loaded")

    # Load questions
    with open(SCRIPT_DIR / 'data' / 'combined_CKGQA_dev_matched.json') as f:
        all_questions = json.load(f)

    # Select 5 diverse samples (one from each taxonomy + 1 extra)
    samples = []
    for taxonomy in ['S', 'R', 'C', 'M']:
        q = next((q for q in all_questions if q['taxonomy'] == taxonomy), None)
        if q:
            samples.append(q)
    # Add one more
    if len(samples) < 5:
        for q in all_questions:
            if q not in samples:
                samples.append(q)
                break

    print(f"Selected {len(samples)} samples")
    print()

    # Get gold answers
    driver = get_neo4j_driver()
    for q in samples:
        q['_gold'] = extract_gold_answer(q, driver)
    driver.close()

    # Define models to test
    models = [
        ('gpt-4o-mini', 'openai'),
        ('gpt-4o', 'openai'),
        ('claude-3-haiku-20240307', 'anthropic'),
        ('meta-llama/Llama-3.1-8B-Instruct-Turbo', 'together'),
    ]

    # Results storage
    results = defaultdict(list)

    # Evaluate each model
    for model_name, provider in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        correct = 0
        total = len(samples)

        for i, q in enumerate(samples):
            print(f"\nQ{i+1} [{q['taxonomy']}]: {q['question'][:60]}...")
            print(f"  Gold: {list(q['_gold'])[:3]}")

            # Get prediction
            if provider == 'openai':
                pred = evaluate_with_openai(q, model_name)
            elif provider == 'anthropic':
                pred = evaluate_with_anthropic(q, model_name)
            elif provider == 'together':
                pred = evaluate_with_together(q, model_name)
            else:
                pred = "ERROR: Unknown provider"

            print(f"  Pred: {pred[:80]}...")

            # Check correctness
            is_correct = check_correctness(pred, q['_gold'])
            if is_correct:
                correct += 1
                print(f"  Result: ✓ CORRECT")
            else:
                print(f"  Result: ✗ WRONG")

            results[model_name].append({
                'question': q['question'],
                'taxonomy': q['taxonomy'],
                'gold': list(q['_gold'])[:5],
                'predicted': pred[:200],
                'correct': is_correct
            })

            time.sleep(0.5)  # Rate limiting

        accuracy = correct / total * 100
        print(f"\n{model_name}: {correct}/{total} ({accuracy:.1f}%)")

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<45} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 80)

    for model_name, model_results in results.items():
        correct = sum(1 for r in model_results if r['correct'])
        total = len(model_results)
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"{model_name:<45} {correct:>8} {total:>8} {accuracy:>9.1f}%")

    print("=" * 80)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = SCRIPT_DIR / 'results' / f'quick_eval_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump(dict(results), f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
