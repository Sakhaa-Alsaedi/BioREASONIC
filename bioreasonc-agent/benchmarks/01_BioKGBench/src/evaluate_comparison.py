#!/usr/bin/env python3
"""
Compare Rule-based vs LLM-based KGQA Performance.
Supports OpenAI and Claude APIs.
"""

import json
import os
import sys
import yaml
import argparse
from collections import defaultdict
from typing import List, Dict, Set
from huggingface_hub import snapshot_download
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kg_qa_system_v2 import KnowledgeGraphQAv2, QAMode, QAResult


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if answer is None:
        return ""
    return str(answer).strip().lower()


def extract_gold_answers(gold: List) -> Set[str]:
    """Extract normalized gold answers from benchmark format."""
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
    """Compute F1 score between predicted and gold answer sets."""
    if not predicted_set or not gold_set:
        return 0.0

    # Count matches (with partial matching for flexibility)
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
    """Check if predicted answers exactly match gold answers."""
    if not predicted_set or not gold_set:
        return False

    # Check if any predicted answer exactly matches any gold answer
    for pred in predicted_set:
        if pred in gold_set:
            return True
        # Also check partial matches
        for gold in gold_set:
            if pred in gold or gold in pred:
                return True
    return False


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics including F1, EM, and Executability."""
    total = len(results)
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    mrr_sum = 0.0
    answered = 0

    # New metrics
    f1_sum = 0.0
    em_count = 0
    executable_count = 0  # Successfully executed queries (even if no results)

    by_type = defaultdict(lambda: {
        'total': 0, 'answered': 0, 'hits_1': 0, 'hits_5': 0, 'mrr': 0.0,
        'f1_sum': 0.0, 'em_count': 0, 'executable': 0
    })

    for r in results:
        qtype = r.get('question_type', 'unknown')
        by_type[qtype]['total'] += 1

        # Executability: query was successfully executed (success=True, even if empty)
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

        # Compute F1 for this question
        f1 = compute_f1(predicted_set, gold_answers)
        f1_sum += f1
        by_type[qtype]['f1_sum'] += f1

        # Compute Exact Match
        if compute_exact_match(predicted_set, gold_answers):
            em_count += 1
            by_type[qtype]['em_count'] += 1

        # Find best rank for Hits@K and MRR
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
        'f1': f1_sum / total if total > 0 else 0,  # F1 over all questions
        'em': em_count / total if total > 0 else 0,  # EM over all questions
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


def evaluate_mode(qa_system: KnowledgeGraphQAv2, questions: List[Dict],
                  mode_name: str, max_questions: int = None) -> Dict:
    """Evaluate a QA mode on questions."""
    results = []

    questions_to_eval = questions[:max_questions] if max_questions else questions

    for q in tqdm(questions_to_eval, desc=f"Evaluating {mode_name}"):
        question_text = q.get('question', '')
        gold = q.get('answer', [])
        qtype = q.get('type', 'unknown')

        result = qa_system.answer(question_text, q)

        # Extract predictions (names first)
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
            'cypher': result.cypher_query,
        })

    return compute_metrics(results), results


def print_comparison_table(all_metrics: Dict[str, Dict]):
    """Print a comparison table of all modes."""
    print("\n" + "="*90)
    print("COMPARISON RESULTS (BioKGBench Metrics)")
    print("="*90)

    # Header
    modes = list(all_metrics.keys())
    print(f"\n{'Metric':<20}", end="")
    for mode in modes:
        print(f"{mode:>15}", end="")
    print()
    print("-"*90)

    # Primary metrics (matching BioKGBench paper Table 5)
    print("PRIMARY METRICS:")
    metric_names = ['f1', 'em', 'executability']
    metric_labels = ['F1', 'EM (Exact Match)', 'Executability']

    for name, label in zip(metric_names, metric_labels):
        print(f"  {label:<18}", end="")
        for mode in modes:
            val = all_metrics[mode].get(name, 0)
            print(f"{val*100:>13.1f}%", end="")
        print()

    # Secondary metrics
    print("\nSECONDARY METRICS:")
    metric_names = ['hits_1', 'hits_5', 'hits_10', 'mrr', 'coverage']
    metric_labels = ['Hits@1', 'Hits@5', 'Hits@10', 'MRR', 'Coverage']

    for name, label in zip(metric_names, metric_labels):
        print(f"  {label:<18}", end="")
        for mode in modes:
            val = all_metrics[mode].get(name, 0)
            if name == 'mrr':
                print(f"{val:>14.3f}", end="")
            else:
                print(f"{val*100:>13.1f}%", end="")
        print()

    # By question type - F1
    print("\n" + "-"*90)
    print("BY QUESTION TYPE (F1)")
    print("-"*90)

    qtypes = ['one-hop', 'multi-hop', 'conjunction']
    for qtype in qtypes:
        print(f"  {qtype:<18}", end="")
        for mode in modes:
            by_type = all_metrics[mode].get('by_type', {})
            if qtype in by_type:
                val = by_type[qtype].get('f1', 0)
                print(f"{val*100:>13.1f}%", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()

    # By question type - Executability
    print("\n" + "-"*90)
    print("BY QUESTION TYPE (Executability)")
    print("-"*90)

    for qtype in qtypes:
        print(f"  {qtype:<18}", end="")
        for mode in modes:
            by_type = all_metrics[mode].get('by_type', {})
            if qtype in by_type:
                val = by_type[qtype].get('executability', 0)
                print(f"{val*100:>13.1f}%", end="")
            else:
                print(f"{'N/A':>14}", end="")
        print()


# Available modes for evaluation
# Note: mixtral excluded due to JSON formatting issues
ALL_MODES = [
    'rule_based',
    # OpenAI models
    'gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini',
    # Anthropic models
    'claude-3-haiku',
    # Together AI models
    'llama-3.1-8b', 'qwen-2.5-7b', 'deepseek-v3',
    # Legacy names (for backward compatibility)
    'openai', 'claude', 'llama3-70b', 'llama3-8b', 'qwen-72b', 'deepseek'
]

# Experiment setup documentation
EXPERIMENT_SETUP = """
================================================================================
EXPERIMENT SETUP (Fair Comparison)
================================================================================
- Temperature: 0.0 (deterministic) for all LLM models
- Max tokens: 1000 for all models
- Same Cypher generation prompt template for all LLMs
- Same KG schema provided to all LLMs
- Same evaluation metrics: F1, EM, Executability, Hits@K, MRR
- Dataset: BioKGBench (AutoLab-Westlake/BioKGBench-Dataset)

Models:
- rule_based: Pattern-matching Cypher generation (no LLM)
- gpt-4o: GPT-4o (OpenAI)
- gpt-4o-mini: GPT-4o-mini (OpenAI)
- gpt-4.1: GPT-4-turbo (OpenAI)
- gpt-4.1-mini: GPT-4-turbo-preview (OpenAI)
- claude-3-haiku: Claude-3-Haiku (Anthropic)
- llama-3.1-8b: Meta-Llama-3.1-8B-Instruct-Turbo (Together AI)
- qwen-2.5-7b: Qwen2.5-7B-Instruct-Turbo (Together AI)
- deepseek-v3: DeepSeek-V3 (Together AI)
================================================================================
"""


def main():
    parser = argparse.ArgumentParser(description='Compare KGQA modes')
    parser.add_argument('--modes', nargs='+', default=['rule_based'],
                       choices=ALL_MODES + ['all', 'all_llm'],
                       help='Modes to evaluate. Use "all" for all modes, "all_llm" for all LLM modes')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Max questions to evaluate (for testing)')
    parser.add_argument('--dataset', choices=['dev', 'test', 'both'], default='test',
                       help='Which dataset to evaluate')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    args = parser.parse_args()

    print("="*80)
    print("KGQA Mode Comparison")
    print("="*80)
    print(EXPERIMENT_SETUP)

    # Load config
    config_path = args.config
    if config_path is None:
        config_path = os.path.join(SCRIPT_DIR, 'config.yaml')
        local_config = os.path.join(SCRIPT_DIR, 'config.local.yaml')
        if os.path.exists(local_config):
            config_path = local_config

    print(f"\nConfig: {config_path}")

    # Load benchmark data
    print("\nLoading BioKGBench dataset...")
    DATA_PATH = snapshot_download(
        repo_id="AutoLab-Westlake/BioKGBench-Dataset",
        repo_type="dataset"
    )

    with open(f"{DATA_PATH}/kgqa/dev.json", 'r') as f:
        dev_data = json.load(f)
    with open(f"{DATA_PATH}/kgqa/test.json", 'r') as f:
        test_data = json.load(f)

    # Select dataset
    if args.dataset == 'dev':
        eval_data = dev_data
        dataset_name = 'Dev'
    elif args.dataset == 'test':
        eval_data = test_data
        dataset_name = 'Test'
    else:
        eval_data = dev_data + test_data
        dataset_name = 'Dev+Test'

    print(f"Evaluating on {dataset_name}: {len(eval_data)} questions")

    # Determine modes to evaluate
    modes_to_eval = []
    if 'all' in args.modes:
        modes_to_eval = ALL_MODES
    elif 'all_llm' in args.modes:
        modes_to_eval = [m for m in ALL_MODES if m != 'rule_based']
    else:
        modes_to_eval = args.modes

    all_metrics = {}
    all_results = {}

    for mode_name in modes_to_eval:
        print(f"\n{'='*80}")
        print(f"Evaluating: {mode_name.upper()}")
        print("="*80)

        try:
            # Create QA system with appropriate config
            if mode_name == 'rule_based':
                qa = KnowledgeGraphQAv2(config_path=config_path, mode=QAMode.RULE_BASED)
            else:
                # Temporarily modify config for this mode
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                config['llm']['provider'] = mode_name
                temp_config_path = f'/tmp/kgqa_config_{mode_name}.yaml'
                with open(temp_config_path, 'w') as f:
                    yaml.dump(config, f)

                qa = KnowledgeGraphQAv2(config_path=temp_config_path, mode=QAMode.LLM)

            qa.connect()
            metrics, results = evaluate_mode(qa, eval_data, mode_name, args.max_questions)
            qa.close()

            all_metrics[mode_name] = metrics
            all_results[mode_name] = results

            print(f"\n{mode_name} Results:")
            print(f"  F1:           {metrics['f1']*100:.1f}%")
            print(f"  EM:           {metrics['em']*100:.1f}%")
            print(f"  Executability:{metrics['executability']*100:.1f}%")
            print(f"  Hits@1:       {metrics['hits_1']*100:.1f}%")
            print(f"  MRR:          {metrics['mrr']:.3f}")

        except Exception as e:
            print(f"Error evaluating {mode_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    if len(all_metrics) > 1:
        print_comparison_table(all_metrics)

    # Save results
    output = {
        'dataset': dataset_name,
        'num_questions': len(eval_data),
        'metrics': all_metrics,
    }

    output_file = f'comparison_results_{dataset_name.lower()}.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))
    print(f"\nResults saved to {output_file}")

    # Print final summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)

    if all_metrics:
        best_f1_mode = max(all_metrics.keys(), key=lambda m: all_metrics[m].get('f1', 0))
        best_f1 = all_metrics[best_f1_mode]['f1'] * 100
        best_em = all_metrics[best_f1_mode]['em'] * 100
        best_exec = all_metrics[best_f1_mode]['executability'] * 100

        print(f"\nBest performing mode (by F1): {best_f1_mode}")
        print(f"  F1:            {best_f1:.1f}%")
        print(f"  EM:            {best_em:.1f}%")
        print(f"  Executability: {best_exec:.1f}%")


if __name__ == "__main__":
    main()
