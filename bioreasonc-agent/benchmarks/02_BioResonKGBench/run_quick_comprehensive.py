#!/usr/bin/env python3
"""
Quick comprehensive evaluation with fewer models for testing.
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

# Setup paths
SCRIPT_DIR = Path('/ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/benchmarks/02_BioResonKGBench')
BIOKGBENCH_DIR = Path('/ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/benchmarks/01_BioKGBench')
SRC_DIR = BIOKGBENCH_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

from neo4j import GraphDatabase

# Try KG QA
try:
    from kg_qa_system_v2 import KnowledgeGraphQAv2, QAMode
    KG_QA_AVAILABLE = True
except:
    KG_QA_AVAILABLE = False

# ============================================================================
# Config
# ============================================================================

MODELS = [
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-4-turbo',      # gpt-4.1
    'claude-3-haiku',
    'deepseek-v3',
    'llama-3.1-8b',
    'qwen-2.5-7b',
]

MODEL_DISPLAY = {
    'gpt-4o-mini': 'GPT-4o-mini',
    'gpt-4o': 'GPT-4o',
    'gpt-4-turbo': 'GPT-4.1',
    'claude-3-haiku': 'Claude-3-Haiku',
    'deepseek-v3': 'DeepSeek-V3',
    'llama-3.1-8b': 'Llama-3.1-8B',
    'qwen-2.5-7b': 'Qwen-2.5-7B',
}

MODEL_IDS = {
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4o': 'gpt-4o',
    'gpt-4-turbo': 'gpt-4-turbo',
    'claude-3-haiku': 'claude-3-haiku-20240307',
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'qwen-2.5-7b': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
}

N_SAMPLES = 20

def load_config():
    with open(SCRIPT_DIR / 'config' / 'config.local.yaml') as f:
        config = yaml.safe_load(f)
    llm = config.get('llm', {})
    if llm.get('openai', {}).get('api_key'):
        os.environ['OPENAI_API_KEY'] = llm['openai']['api_key']
    if llm.get('claude', {}).get('api_key'):
        os.environ['ANTHROPIC_API_KEY'] = llm['claude']['api_key']
    if llm.get('together', {}).get('api_key'):
        os.environ['TOGETHER_API_KEY'] = llm['together']['api_key']
    return config

def get_neo4j_driver():
    with open(SCRIPT_DIR / 'config' / 'kg_config.yml') as f:
        cfg = yaml.safe_load(f)
    return GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

def get_llm_client(model_name):
    if model_name.startswith('gpt'):
        from openai import OpenAI
        return OpenAI(), 'openai'
    elif model_name.startswith('claude'):
        from anthropic import Anthropic
        return Anthropic(), 'anthropic'
    else:
        from openai import OpenAI
        return OpenAI(base_url="https://api.together.xyz/v1", api_key=os.environ.get('TOGETHER_API_KEY', '')), 'together'

def call_llm(client, client_type, model_id, prompt, max_tokens=500):
    try:
        if client_type == 'anthropic':
            response = client.messages.create(model=model_id, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
            return response.content[0].text
        else:
            response = client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0)
            return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

def normalize_answer(answer):
    if answer is None: return ""
    text = str(answer).strip().lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def tokenize(text):
    return normalize_answer(text).split()

def compute_f1(pred_tokens, gold_tokens):
    if not pred_tokens or not gold_tokens: return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common: return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)

def compute_exact_match(pred, gold_set):
    pred_norm = normalize_answer(pred)
    return any(pred_norm == g or g in pred_norm or pred_norm in g for g in gold_set if g)

def compute_metrics(predictions, gold_answers):
    if not predictions: predictions = ['']
    if not gold_answers: return {'f1': 0, 'em': 0, 'hits1': 0, 'mrr': 0}

    best_f1 = 0
    pred_tokens = tokenize(predictions[0]) if predictions else []
    for gold in gold_answers:
        f1 = compute_f1(pred_tokens, tokenize(gold))
        best_f1 = max(best_f1, f1)

    em = 1.0 if compute_exact_match(predictions[0], gold_answers) else 0.0
    hits1 = 1.0 if any(compute_exact_match(p, gold_answers) for p in predictions[:1]) else 0.0
    mrr = 0.0
    for i, p in enumerate(predictions):
        if compute_exact_match(p, gold_answers):
            mrr = 1.0 / (i + 1)
            break

    return {'f1': best_f1, 'em': em, 'hits1': hits1, 'mrr': mrr}

def load_biokgbench_questions(n):
    with open(BIOKGBENCH_DIR / 'data' / 'dev.json') as f:
        questions = json.load(f)
    by_type = defaultdict(list)
    for q in questions: by_type[q.get('type', 'unknown')].append(q)
    samples = []
    for qs in by_type.values(): samples.extend(qs[:max(1, n//len(by_type))])
    for q in questions:
        if q not in samples and len(samples) < n: samples.append(q)
    return samples[:n]

def load_bioresonkgbench_questions(n):
    with open(SCRIPT_DIR / 'data' / 'combined_CKGQA_dev_matched.json') as f:
        questions = json.load(f)
    samples = []
    for tax in ['S', 'R', 'C', 'M']:
        for q in questions:
            if q.get('taxonomy') == tax and len(samples) < (n // 4) * (list('SRCM').index(tax) + 1):
                if q not in samples: samples.append(q)
    for q in questions:
        if q not in samples and len(samples) < n: samples.append(q)
    return samples[:n]

def get_biokgbench_gold(q):
    answers = set()
    af = q.get('answer', [])
    if isinstance(af, list):
        for a in af:
            if isinstance(a, dict):
                for k in ['answer', 'name', 'id']:
                    if k in a: answers.add(normalize_answer(a[k]))
            else: answers.add(normalize_answer(str(a)))
    else: answers.add(normalize_answer(str(af)))
    return answers

def get_bioresonkgbench_gold(q, driver):
    cypher, answer_key = q.get('cypher', ''), q.get('answer_key', '')
    if not cypher or not answer_key: return set()
    answers = set()
    try:
        with driver.session() as s:
            for r in s.run(cypher):
                if answer_key in r.keys():
                    v = r[answer_key]
                    if v is not None:
                        if isinstance(v, list):
                            for x in v[:10]: answers.add(normalize_answer(str(x)))
                        else: answers.add(normalize_answer(str(v)))
    except: pass
    return answers

def evaluate_no_kg(questions, gold_func, model_name, driver=None):
    client, client_type = get_llm_client(model_name)
    model_id = MODEL_IDS.get(model_name, model_name)
    metrics_sum = {'f1': 0, 'em': 0, 'hits1': 0, 'mrr': 0}

    for q in questions:
        gold = gold_func(q, driver) if driver else gold_func(q)
        prompt = f"Answer concisely. No explanation.\n\nQuestion: {q.get('question', '')}\n\nAnswer:"
        try:
            response = call_llm(client, client_type, model_id, prompt)
            predictions = [response.strip()]
        except: predictions = ['']

        m = compute_metrics(predictions, gold)
        for k in metrics_sum: metrics_sum[k] += m[k]
        time.sleep(0.2)

    n = len(questions)
    return {k: v / n * 100 for k, v in metrics_sum.items()}

def evaluate_with_kg(questions, gold_func, model_name, base_config, driver):
    if not KG_QA_AVAILABLE: return None

    config = base_config.copy()
    config['llm'] = config.get('llm', {})
    config['llm']['provider'] = model_name
    temp = f'/tmp/eval_{model_name}.yaml'
    with open(temp, 'w') as f: yaml.dump(config, f)

    try:
        qa = KnowledgeGraphQAv2(config_path=temp, mode=QAMode.LLM)
        qa.connect()
    except Exception as e:
        return None

    metrics_sum = {'f1': 0, 'em': 0, 'hits1': 0, 'mrr': 0, 'exec': 0}

    for q in questions:
        gold = gold_func(q, driver) if driver else gold_func(q)
        try:
            result = qa.answer(q.get('question', ''), q)
            predictions = []
            if result.success and result.answers:
                for ans in result.answers[:5]:
                    if ans.name: predictions.append(ans.name)
                    if ans.id: predictions.append(ans.id)
            exec_success = 1.0 if result.success else 0.0
        except:
            predictions = []
            exec_success = 0.0

        m = compute_metrics(predictions, gold)
        for k in ['f1', 'em', 'hits1', 'mrr']: metrics_sum[k] += m[k]
        metrics_sum['exec'] += exec_success
        time.sleep(0.2)

    qa.close()
    n = len(questions)
    return {k: v / n * 100 for k, v in metrics_sum.items()}

def main():
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION (Quick Mode)")
    print("=" * 80)

    config = load_config()
    driver = get_neo4j_driver()

    bio_qs = load_biokgbench_questions(N_SAMPLES)
    biores_qs = load_bioresonkgbench_questions(N_SAMPLES)
    print(f"BioKGBench: {len(bio_qs)}, BioResonKGBench: {len(biores_qs)}")

    results = {}

    for model in MODELS:
        print(f"\n{'='*50}")
        print(f"MODEL: {MODEL_DISPLAY[model]}")
        print(f"{'='*50}")

        results[model] = {}

        # No KG
        print("  BioKGBench (No KG)...")
        results[model]['biokgbench_no_kg'] = evaluate_no_kg(bio_qs, get_biokgbench_gold, model)
        print(f"    EM: {results[model]['biokgbench_no_kg']['em']:.1f}%")

        print("  BioResonKGBench (No KG)...")
        results[model]['bioresonkgbench_no_kg'] = evaluate_no_kg(biores_qs, get_bioresonkgbench_gold, model, driver)
        print(f"    EM: {results[model]['bioresonkgbench_no_kg']['em']:.1f}%")

        # With KG
        print("  BioKGBench (KG)...")
        r = evaluate_with_kg(bio_qs, get_biokgbench_gold, model, config, None)
        results[model]['biokgbench_kg'] = r if r else {'em': 0, 'f1': 0, 'hits1': 0, 'mrr': 0, 'exec': 0}
        if r: print(f"    EM: {r['em']:.1f}%, Exec: {r['exec']:.1f}%")

        print("  BioResonKGBench (KG)...")
        r = evaluate_with_kg(biores_qs, get_bioresonkgbench_gold, model, config, driver)
        results[model]['bioresonkgbench_kg'] = r if r else {'em': 0, 'f1': 0, 'hits1': 0, 'mrr': 0, 'exec': 0}
        if r: print(f"    EM: {r['em']:.1f}%, Exec: {r['exec']:.1f}%")

    driver.close()

    # Print summary table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"\n{'Model':<16} | {'BioKGBench (No KG)':<18} | {'BioKGBench (KG)':<18} | {'BioResonKGBench (No KG)':<22} | {'BioResonKGBench (KG)':<18}")
    print(f"{'':<16} | {'EM%':>5} {'F1%':>5} {'H@1':>5} | {'EM%':>5} {'F1%':>5} {'Exec':>5} | {'EM%':>5} {'F1%':>5} {'H@1':>5} | {'EM%':>5} {'F1%':>5} {'Exec':>5}")
    print("-" * 115)

    for model in MODELS:
        r = results[model]
        bn = r['biokgbench_no_kg']
        bk = r['biokgbench_kg']
        rn = r['bioresonkgbench_no_kg']
        rk = r['bioresonkgbench_kg']

        print(f"{MODEL_DISPLAY[model]:<16} | {bn['em']:>5.1f} {bn['f1']:>5.1f} {bn['hits1']:>5.1f} | {bk['em']:>5.1f} {bk['f1']:>5.1f} {bk.get('exec', 0):>5.1f} | {rn['em']:>5.1f} {rn['f1']:>5.1f} {rn['hits1']:>5.1f} | {rk['em']:>5.1f} {rk['f1']:>5.1f} {rk.get('exec', 0):>5.1f}")

    print("-" * 115)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output = SCRIPT_DIR / 'results' / f'quick_comprehensive_{timestamp}.json'
    output.parent.mkdir(exist_ok=True)
    with open(output, 'w') as f:
        json.dump({'timestamp': timestamp, 'n_samples': N_SAMPLES, 'models': MODELS, 'results': results}, f, indent=2)
    print(f"\nSaved: {output}")

if __name__ == "__main__":
    main()
