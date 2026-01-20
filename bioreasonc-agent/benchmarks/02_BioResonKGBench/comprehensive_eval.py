#!/usr/bin/env python3
"""
Comprehensive Evaluation of LLMs on BioKGBench and BioResonKGBench.
Tests with and without KG access across 8 models.

Metrics:
- F1: Token-level F1 score
- EM: Exact Match accuracy
- Exec: Query executability rate (KG mode only)
- Hits@1: Top-1 hit rate
- MRR: Mean Reciprocal Rank
"""

import json
import yaml
import os
import sys
import time
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import traceback

# Setup paths
SCRIPT_DIR = Path('/ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/benchmarks/02_BioResonKGBench')
BIOKGBENCH_DIR = Path('/ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/benchmarks/01_BioKGBench')
SRC_DIR = BIOKGBENCH_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from neo4j import GraphDatabase

# Try to import KG QA system
try:
    from kg_qa_system_v2 import KnowledgeGraphQAv2, QAMode
    KG_QA_AVAILABLE = True
except:
    KG_QA_AVAILABLE = False
    print("Warning: KG QA system not available")

# ============================================================================
# Configuration
# ============================================================================

MODELS = [
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-4-turbo',      # gpt-4.1
    'gpt-4-turbo-mini', # gpt-4.1-mini (fallback to gpt-4o-mini if not available)
    'claude-3-haiku',
    'deepseek-v3',
    'llama-3.1-8b',
    'qwen-2.5-7b',
]

# Model mappings for display
MODEL_DISPLAY_NAMES = {
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4o': 'gpt-4o',
    'gpt-4-turbo': 'gpt-4.1',
    'gpt-4-turbo-mini': 'gpt-4.1-mini',
    'claude-3-haiku': 'claude-3-haiku',
    'deepseek-v3': 'deepseek-v3',
    'llama-3.1-8b': 'llama-3.1-8b',
    'qwen-2.5-7b': 'qwen-2.5-7b',
}

N_SAMPLES = 20

# ============================================================================
# Load Config and Setup
# ============================================================================

def load_config():
    """Load configuration and set API keys."""
    config_path = SCRIPT_DIR / 'config' / 'config.local.yaml'
    if not config_path.exists():
        config_path = SCRIPT_DIR / 'config' / 'kg_config.yml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set API keys from environment or config
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

# ============================================================================
# LLM Clients
# ============================================================================

def get_llm_client(model_name):
    """Get appropriate LLM client for model."""
    if model_name.startswith('gpt'):
        from openai import OpenAI
        return OpenAI(), 'openai'
    elif model_name.startswith('claude'):
        from anthropic import Anthropic
        return Anthropic(), 'anthropic'
    else:
        # Together AI for open models
        from openai import OpenAI
        return OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get('TOGETHER_API_KEY', '')
        ), 'together'

def get_model_id(model_name):
    """Map model name to API model ID."""
    mapping = {
        'gpt-4o-mini': 'gpt-4o-mini',
        'gpt-4o': 'gpt-4o',
        'gpt-4-turbo': 'gpt-4-turbo',
        'gpt-4-turbo-mini': 'gpt-4o-mini',  # Fallback
        'claude-3-haiku': 'claude-3-haiku-20240307',
        'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
        'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
        'qwen-2.5-7b': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
    }
    return mapping.get(model_name, model_name)

def call_llm(client, client_type, model_id, prompt, max_tokens=500):
    """Call LLM and get response."""
    try:
        if client_type == 'anthropic':
            response = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        else:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0
            )
            return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

# ============================================================================
# Data Loading
# ============================================================================

def load_biokgbench_questions(n_samples):
    """Load BioKGBench questions."""
    with open(BIOKGBENCH_DIR / 'data' / 'dev.json') as f:
        questions = json.load(f)

    # Balance by question type
    samples = []
    by_type = defaultdict(list)
    for q in questions:
        by_type[q.get('type', 'unknown')].append(q)

    per_type = max(1, n_samples // len(by_type))
    for qtype, qs in by_type.items():
        samples.extend(qs[:per_type])

    # Fill remaining
    for q in questions:
        if q not in samples and len(samples) < n_samples:
            samples.append(q)

    return samples[:n_samples]

def load_bioresonkgbench_questions(n_samples):
    """Load BioResonKGBench questions."""
    with open(SCRIPT_DIR / 'data' / 'combined_CKGQA_dev_matched.json') as f:
        questions = json.load(f)

    # Balance by taxonomy
    samples = []
    per_tax = max(1, n_samples // 4)

    for tax in ['S', 'R', 'C', 'M']:
        tax_qs = [q for q in questions if q.get('taxonomy') == tax]
        samples.extend(tax_qs[:per_tax])

    # Fill remaining
    for q in questions:
        if q not in samples and len(samples) < n_samples:
            samples.append(q)

    return samples[:n_samples]

# ============================================================================
# Gold Answer Extraction
# ============================================================================

def normalize_answer(answer):
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    text = str(answer).strip().lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def get_biokgbench_gold(question):
    """Extract gold answers from BioKGBench question."""
    answers = set()
    answer_field = question.get('answer', [])

    if isinstance(answer_field, list):
        for ans in answer_field:
            if isinstance(ans, dict):
                for key in ['answer', 'name', 'id']:
                    if key in ans:
                        answers.add(normalize_answer(ans[key]))
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
                            for v in value[:10]:
                                answers.add(normalize_answer(str(v)))
                        else:
                            answers.add(normalize_answer(str(value)))
    except:
        pass

    return answers

# ============================================================================
# Metrics Calculation
# ============================================================================

def tokenize(text):
    """Simple tokenization."""
    return normalize_answer(text).split()

def compute_f1(pred_tokens, gold_tokens):
    """Compute token-level F1."""
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def compute_exact_match(pred, gold_set):
    """Check if prediction exactly matches any gold answer."""
    pred_norm = normalize_answer(pred)
    return any(pred_norm == g or g in pred_norm or pred_norm in g for g in gold_set if g)

def compute_hits_at_k(pred_list, gold_set, k=1):
    """Compute Hits@K."""
    for pred in pred_list[:k]:
        if compute_exact_match(pred, gold_set):
            return 1.0
    return 0.0

def compute_mrr(pred_list, gold_set):
    """Compute Mean Reciprocal Rank."""
    for i, pred in enumerate(pred_list):
        if compute_exact_match(pred, gold_set):
            return 1.0 / (i + 1)
    return 0.0

def compute_metrics(predictions, gold_answers):
    """Compute all metrics for a single question."""
    if not predictions:
        predictions = ['']
    if not gold_answers:
        return {'f1': 0, 'em': 0, 'hits1': 0, 'mrr': 0}

    # Best F1 across all gold answers
    best_f1 = 0
    pred_tokens = tokenize(predictions[0]) if predictions else []
    for gold in gold_answers:
        gold_tokens = tokenize(gold)
        f1 = compute_f1(pred_tokens, gold_tokens)
        best_f1 = max(best_f1, f1)

    # Exact match
    em = 1.0 if compute_exact_match(predictions[0], gold_answers) else 0.0

    # Hits@1
    hits1 = compute_hits_at_k(predictions, gold_answers, k=1)

    # MRR
    mrr = compute_mrr(predictions, gold_answers)

    return {'f1': best_f1, 'em': em, 'hits1': hits1, 'mrr': mrr}

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_no_kg(questions, gold_func, model_name, benchmark_name, driver=None):
    """Evaluate model without KG access (direct answering)."""
    print(f"\n  [{model_name}] No KG mode...")

    client, client_type = get_llm_client(model_name)
    model_id = get_model_id(model_name)

    results = []
    metrics_sum = {'f1': 0, 'em': 0, 'hits1': 0, 'mrr': 0}

    for i, q in enumerate(questions):
        question_text = q.get('question', '')

        # Get gold
        if driver:
            gold = gold_func(q, driver)
        else:
            gold = gold_func(q)

        # Create prompt for direct answering
        prompt = f"""Answer the following biomedical question concisely.
Provide only the answer, no explanation.

Question: {question_text}

Answer:"""

        # Get prediction
        try:
            response = call_llm(client, client_type, model_id, prompt)
            predictions = [response.strip()]
        except Exception as e:
            predictions = [f"ERROR: {e}"]

        # Compute metrics
        m = compute_metrics(predictions, gold)
        for k in metrics_sum:
            metrics_sum[k] += m[k]

        results.append({
            'question': question_text,
            'gold': list(gold)[:5],
            'predicted': predictions[:3],
            'metrics': m
        })

        # Progress
        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(questions)} done")

        time.sleep(0.3)

    # Average metrics
    n = len(questions)
    avg_metrics = {k: v / n * 100 for k, v in metrics_sum.items()}

    return {
        'model': model_name,
        'benchmark': benchmark_name,
        'mode': 'no_kg',
        'n_questions': n,
        'metrics': avg_metrics,
        'details': results
    }

def evaluate_with_kg(questions, gold_func, model_name, benchmark_name, base_config, driver):
    """Evaluate model with KG access (Cypher generation)."""
    print(f"\n  [{model_name}] With KG mode...")

    if not KG_QA_AVAILABLE:
        print("    KG QA system not available, skipping...")
        return None

    # Create config for model
    config = base_config.copy()
    config['llm'] = config.get('llm', {})
    config['llm']['provider'] = model_name

    temp_config = f'/tmp/eval_config_{model_name.replace("/", "_")}.yaml'
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    try:
        qa = KnowledgeGraphQAv2(config_path=temp_config, mode=QAMode.LLM)
        qa.connect()
    except Exception as e:
        print(f"    Failed to init KG QA: {e}")
        return None

    results = []
    metrics_sum = {'f1': 0, 'em': 0, 'hits1': 0, 'mrr': 0, 'exec': 0}

    for i, q in enumerate(questions):
        question_text = q.get('question', '')

        # Get gold
        gold = gold_func(q, driver) if driver else gold_func(q)

        # Get prediction via KG
        try:
            result = qa.answer(question_text, q)
            predictions = []
            if result.success and result.answers:
                for ans in result.answers[:5]:
                    if ans.name:
                        predictions.append(ans.name)
                    if ans.id:
                        predictions.append(ans.id)

            exec_success = 1.0 if result.success else 0.0
            cypher = result.cypher_query or ""
        except Exception as e:
            predictions = []
            exec_success = 0.0
            cypher = f"ERROR: {e}"

        # Compute metrics
        m = compute_metrics(predictions, gold)
        m['exec'] = exec_success
        for k in metrics_sum:
            metrics_sum[k] += m.get(k, 0)

        results.append({
            'question': question_text,
            'gold': list(gold)[:5],
            'predicted': predictions[:5],
            'cypher': cypher[:200] if cypher else None,
            'metrics': m
        })

        if (i + 1) % 5 == 0:
            print(f"    {i+1}/{len(questions)} done")

        time.sleep(0.3)

    qa.close()

    # Average metrics
    n = len(questions)
    avg_metrics = {k: v / n * 100 for k, v in metrics_sum.items()}

    return {
        'model': model_name,
        'benchmark': benchmark_name,
        'mode': 'with_kg',
        'n_questions': n,
        'metrics': avg_metrics,
        'details': results
    }

# ============================================================================
# Main Evaluation
# ============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE LLM EVALUATION")
    print("BioKGBench vs BioResonKGBench - With and Without KG Access")
    print("=" * 80)
    print(f"\nModels: {len(MODELS)}")
    print(f"Samples per benchmark: {N_SAMPLES}")
    print(f"Total evaluations: {len(MODELS)} x 4 conditions = {len(MODELS) * 4}")
    print()

    # Load config
    base_config = load_config()
    driver = get_neo4j_driver()

    # Load questions
    print("Loading questions...")
    biokgbench_qs = load_biokgbench_questions(N_SAMPLES)
    bioresonkgbench_qs = load_bioresonkgbench_questions(N_SAMPLES)
    print(f"  BioKGBench: {len(biokgbench_qs)} questions")
    print(f"  BioResonKGBench: {len(bioresonkgbench_qs)} questions")

    # Results storage
    all_results = {
        'biokgbench_no_kg': {},
        'biokgbench_kg': {},
        'bioresonkgbench_no_kg': {},
        'bioresonkgbench_kg': {},
    }

    # Evaluate each model
    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"MODEL: {MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
        print(f"{'='*60}")

        start_time = time.time()

        # 1. BioKGBench - No KG
        try:
            result = evaluate_no_kg(
                biokgbench_qs,
                get_biokgbench_gold,
                model_name,
                'BioKGBench'
            )
            all_results['biokgbench_no_kg'][model_name] = result
        except Exception as e:
            print(f"  Error in BioKGBench No KG: {e}")
            all_results['biokgbench_no_kg'][model_name] = {'error': str(e)}

        # 2. BioKGBench - With KG
        try:
            result = evaluate_with_kg(
                biokgbench_qs,
                get_biokgbench_gold,
                model_name,
                'BioKGBench',
                base_config,
                None  # BioKGBench doesn't need driver for gold
            )
            if result:
                all_results['biokgbench_kg'][model_name] = result
        except Exception as e:
            print(f"  Error in BioKGBench KG: {e}")
            all_results['biokgbench_kg'][model_name] = {'error': str(e)}

        # 3. BioResonKGBench - No KG
        try:
            result = evaluate_no_kg(
                bioresonkgbench_qs,
                get_bioresonkgbench_gold,
                model_name,
                'BioResonKGBench',
                driver
            )
            all_results['bioresonkgbench_no_kg'][model_name] = result
        except Exception as e:
            print(f"  Error in BioResonKGBench No KG: {e}")
            all_results['bioresonkgbench_no_kg'][model_name] = {'error': str(e)}

        # 4. BioResonKGBench - With KG
        try:
            result = evaluate_with_kg(
                bioresonkgbench_qs,
                get_bioresonkgbench_gold,
                model_name,
                'BioResonKGBench',
                base_config,
                driver
            )
            if result:
                all_results['bioresonkgbench_kg'][model_name] = result
        except Exception as e:
            print(f"  Error in BioResonKGBench KG: {e}")
            all_results['bioresonkgbench_kg'][model_name] = {'error': str(e)}

        elapsed = time.time() - start_time
        print(f"\n  Model completed in {elapsed/60:.1f} minutes")

    driver.close()

    # ========================================================================
    # Print Summary Table
    # ========================================================================
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    # Header
    print(f"\n{'Model':<16} | {'BioKGBench (No KG)':<20} | {'BioKGBench (KG)':<20} | {'BioResonKGBench (No KG)':<24} | {'BioResonKGBench (KG)':<20}")
    print(f"{'':<16} | {'EM%':>6} {'F1%':>6} {'H@1':>6} | {'EM%':>6} {'F1%':>6} {'H@1':>6} | {'EM%':>6} {'F1%':>6} {'H@1':>6} | {'EM%':>6} {'F1%':>6} {'H@1':>6}")
    print("-" * 120)

    for model_name in MODELS:
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)[:15]

        row = f"{display_name:<16} |"

        for condition in ['biokgbench_no_kg', 'biokgbench_kg', 'bioresonkgbench_no_kg', 'bioresonkgbench_kg']:
            result = all_results[condition].get(model_name, {})
            if 'error' in result or not result:
                row += f" {'--':>6} {'--':>6} {'--':>6} |"
            else:
                m = result.get('metrics', {})
                row += f" {m.get('em', 0):>6.1f} {m.get('f1', 0):>6.1f} {m.get('hits1', 0):>6.1f} |"

        print(row)

    print("-" * 120)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = SCRIPT_DIR / 'results' / f'comprehensive_eval_{timestamp}.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_samples': N_SAMPLES,
            'models': MODELS,
            'results': all_results
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return all_results

if __name__ == "__main__":
    main()
