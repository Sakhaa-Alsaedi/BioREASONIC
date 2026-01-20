#!/usr/bin/env python3
"""
Run BioKGBench evaluation on all specified LLM models.
Saves results progressively to avoid losing data on failures.
"""

import json
import os
import sys
import yaml
import time
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Set
from huggingface_hub import snapshot_download
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kg_qa_system_v2 import KnowledgeGraphQAv2, QAMode, QAResult

# Models to evaluate
MODELS_TO_EVAL = [
    'claude-3-haiku',
    'deepseek-v3',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4o',
    'gpt-4o-mini',
    'llama-3.1-8b',
    'qwen-2.5-7b',
]


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    return str(answer).strip().lower()


def extract_gold_answers(gold: List) -> Set[str]:
    answers = set()
    for item in gold:
        if isinstance(item, dict):
            for key in ['answer', 'id', 'name']:
                if key in item and item[key]:
                    answers.add(normalize_answer(item[key]))
        elif isinstance(item, str):
            answers.add(normalize_answer(item))
    return answers


def compute_f1(predicted_set: Set[str], gold_set: Set[str]) -> float:
    if not predicted_set or not gold_set:
        return 0.0
    matches = 0
    for pred in predicted_set:
        for gold in gold_set:
            if pred == gold or pred in gold or gold in pred:
                matches += 1
                break
    precision = matches / len(predicted_set) if predicted_set else 0
    recall = matches / len(gold_set) if gold_set else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_exact_match(predicted_set: Set[str], gold_set: Set[str]) -> bool:
    if not predicted_set or not gold_set:
        return False
    for pred in predicted_set:
        if pred in gold_set:
            return True
        for gold in gold_set:
            if pred in gold or gold in pred:
                return True
    return False


def compute_metrics(results: List[Dict]) -> Dict:
    total = len(results)
    hits_1 = hits_5 = hits_10 = 0
    mrr_sum = f1_sum = 0.0
    answered = em_count = executable_count = 0

    by_type = defaultdict(lambda: {
        'total': 0, 'answered': 0, 'hits_1': 0, 'hits_5': 0, 'mrr': 0.0,
        'f1_sum': 0.0, 'em_count': 0, 'executable': 0
    })

    for r in results:
        qtype = r.get('question_type', 'unknown')
        by_type[qtype]['total'] += 1

        if r['success']:
            executable_count += 1
            by_type[qtype]['executable'] += 1

        if not r['success'] or not r['predicted']:
            continue

        answered += 1
        by_type[qtype]['answered'] += 1

        gold_answers = r['gold_answers']
        predicted = [normalize_answer(a) for a in r['predicted']]
        predicted_set = set(predicted)

        f1 = compute_f1(predicted_set, gold_answers)
        f1_sum += f1
        by_type[qtype]['f1_sum'] += f1

        if compute_exact_match(predicted_set, gold_answers):
            em_count += 1
            by_type[qtype]['em_count'] += 1

        best_rank = None
        for i, pred in enumerate(predicted[:10]):
            if pred in gold_answers:
                best_rank = i + 1
                break
            for gold in gold_answers:
                if gold in pred or pred in gold:
                    best_rank = i + 1
                    break
            if best_rank:
                break

        if best_rank:
            if best_rank <= 1:
                hits_1 += 1
                by_type[qtype]['hits_1'] += 1
            if best_rank <= 5:
                hits_5 += 1
                by_type[qtype]['hits_5'] += 1
            if best_rank <= 10:
                hits_10 += 1
            mrr_sum += 1.0 / best_rank
            by_type[qtype]['mrr'] += 1.0 / best_rank

    metrics = {
        'total': total,
        'answered': answered,
        'coverage': answered / total if total > 0 else 0,
        'executability': executable_count / total if total > 0 else 0,
        'hits_1': hits_1 / answered if answered > 0 else 0,
        'hits_5': hits_5 / answered if answered > 0 else 0,
        'hits_10': hits_10 / answered if answered > 0 else 0,
        'mrr': mrr_sum / answered if answered > 0 else 0,
        'f1': f1_sum / total if total > 0 else 0,
        'em': em_count / total if total > 0 else 0,
    }

    metrics['by_type'] = {}
    for qtype, stats in by_type.items():
        ans = stats['answered']
        t = stats['total']
        metrics['by_type'][qtype] = {
            'total': t,
            'answered': ans,
            'coverage': ans / t if t > 0 else 0,
            'executability': stats['executable'] / t if t > 0 else 0,
            'hits_1': stats['hits_1'] / ans if ans > 0 else 0,
            'hits_5': stats['hits_5'] / ans if ans > 0 else 0,
            'mrr': stats['mrr'] / ans if ans > 0 else 0,
            'f1': stats['f1_sum'] / t if t > 0 else 0,
            'em': stats['em_count'] / t if t > 0 else 0,
        }

    return metrics


def evaluate_model(model_name: str, questions: List[Dict], config_path: str,
                   max_questions: int = None) -> Dict:
    """Evaluate a single model on the questions."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print("="*70)

    # Load config and modify for this model
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['llm']['provider'] = model_name
    temp_config_path = f'/tmp/kgqa_config_{model_name}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    try:
        qa = KnowledgeGraphQAv2(config_path=temp_config_path, mode=QAMode.LLM)
        qa.connect()
    except Exception as e:
        print(f"Error connecting: {e}")
        return None

    results = []
    questions_to_eval = questions[:max_questions] if max_questions else questions
    start_time = time.time()

    for q in tqdm(questions_to_eval, desc=f"{model_name}"):
        question_text = q.get('question', '')
        gold = q.get('answer', [])
        qtype = q.get('type', 'unknown')

        try:
            result = qa.answer(question_text, q)
            predicted = []
            if result.success and result.answers:
                for ans in result.answers:
                    if ans.name:
                        predicted.append(ans.name)
                    if ans.id:
                        predicted.append(ans.id)

            results.append({
                'question': question_text,
                'question_type': qtype,
                'gold_answers': extract_gold_answers(gold),
                'predicted': predicted,
                'success': result.success,
                'error': result.error,
            })
        except Exception as e:
            results.append({
                'question': question_text,
                'question_type': qtype,
                'gold_answers': extract_gold_answers(gold),
                'predicted': [],
                'success': False,
                'error': str(e),
            })

    qa.close()
    total_time = time.time() - start_time

    metrics = compute_metrics(results)
    metrics['total_time_sec'] = total_time
    metrics['model'] = model_name

    print(f"\n{model_name} Results:")
    print(f"  F1:            {metrics['f1']*100:.1f}%")
    print(f"  EM:            {metrics['em']*100:.1f}%")
    print(f"  Executability: {metrics['executability']*100:.1f}%")
    print(f"  Hits@1:        {metrics['hits_1']*100:.1f}%")
    print(f"  MRR:           {metrics['mrr']:.3f}")
    print(f"  Time:          {total_time:.1f}s")

    return {'metrics': metrics, 'results': results}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all models evaluation')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Max questions per model (for testing)')
    parser.add_argument('--models', nargs='+', default=MODELS_TO_EVAL,
                       help='Models to evaluate')
    parser.add_argument('--dataset', choices=['dev', 'test'], default='test',
                       help='Dataset to evaluate on')
    args = parser.parse_args()

    print("="*70)
    print("BioKGBench Multi-Model Evaluation")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Models: {args.models}")
    print(f"Dataset: {args.dataset}")
    if args.max_questions:
        print(f"Max questions: {args.max_questions}")

    # Load config
    config_path = os.path.join(SCRIPT_DIR, 'config.local.yaml')
    if not os.path.exists(config_path):
        config_path = os.path.join(SCRIPT_DIR, 'config.yaml')
    print(f"Config: {config_path}")

    # Load benchmark data
    print("\nLoading BioKGBench dataset...")
    DATA_PATH = snapshot_download(
        repo_id="AutoLab-Westlake/BioKGBench-Dataset",
        repo_type="dataset"
    )

    with open(f"{DATA_PATH}/kgqa/{args.dataset}.json", 'r') as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} questions from {args.dataset} set")

    # Run evaluations
    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for model_name in args.models:
        try:
            result = evaluate_model(model_name, eval_data, config_path, args.max_questions)
            if result:
                all_results[model_name] = result

                # Save intermediate results
                intermediate_file = f'eval_results_{model_name}_{args.dataset}_{timestamp}.json'
                with open(intermediate_file, 'w') as f:
                    json.dump({
                        'model': model_name,
                        'dataset': args.dataset,
                        'timestamp': timestamp,
                        'metrics': result['metrics'],
                        'num_results': len(result['results'])
                    }, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))
                print(f"Saved: {intermediate_file}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary table
    print("\n" + "="*90)
    print("SUMMARY - ALL MODELS")
    print("="*90)
    print(f"{'Model':<18} {'F1':>8} {'EM':>8} {'Exec':>8} {'Hits@1':>8} {'MRR':>8} {'Time':>10}")
    print("-"*90)

    summary = []
    for model_name, result in all_results.items():
        m = result['metrics']
        print(f"{model_name:<18} {m['f1']*100:>7.1f}% {m['em']*100:>7.1f}% "
              f"{m['executability']*100:>7.1f}% {m['hits_1']*100:>7.1f}% "
              f"{m['mrr']:>8.3f} {m.get('total_time_sec', 0):>9.1f}s")
        summary.append({
            'model': model_name,
            'f1': m['f1'],
            'em': m['em'],
            'executability': m['executability'],
            'hits_1': m['hits_1'],
            'mrr': m['mrr'],
            'total_time_sec': m.get('total_time_sec', 0)
        })

    print("="*90)

    # Save final summary
    output = {
        'timestamp': timestamp,
        'dataset': args.dataset,
        'num_questions': len(eval_data) if not args.max_questions else args.max_questions,
        'summary': summary,
        'detailed_metrics': {m: r['metrics'] for m, r in all_results.items()}
    }

    output_file = f'biokgbench_eval_all_models_{args.dataset}_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))
    print(f"\nFinal results saved to: {output_file}")


if __name__ == '__main__':
    main()
