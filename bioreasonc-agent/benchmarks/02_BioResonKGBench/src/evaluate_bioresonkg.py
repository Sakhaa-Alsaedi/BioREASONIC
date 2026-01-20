#!/usr/bin/env python3
"""
BioResonKGBench Evaluation Script

Evaluates LLMs on the Biomedical Reasoning Knowledge Graph Benchmark.
Supports both knowledge and reasoning tracks.

Usage:
    python evaluate_bioresonkg.py --model gpt-4o --task knowledge --split test
    python evaluate_bioresonkg.py --model gpt-4o --task reasoning --split test
    python evaluate_bioresonkg.py --model gpt-4o --task all --split test
"""

import json
import os
import sys
import argparse
import time
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "01_BioKGBench" / "src"))

try:
    from llm_client import create_llm_client as get_llm_client
except ImportError:
    print("Warning: llm_client not found. Make sure to run from correct directory.")
    get_llm_client = None


# =============================================================================
# Configuration
# =============================================================================

QUESTION_TYPES = ['S', 'R', 'C', 'M']
TASK_TYPES = ['knowledge', 'reasoning']
SPLITS = ['dev', 'test']

DEFAULT_MODELS = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4.1',
    'gpt-4.1-mini',
    'claude-3-haiku',
    'deepseek-v3',
    'llama-3.1-8b',
    'qwen-2.5-7b',
]


# =============================================================================
# Metrics Functions
# =============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    if answer is None:
        return ""
    return str(answer).strip().lower()


def extract_answers(gold: List) -> Set[str]:
    """Extract normalized answers from various formats."""
    answers = set()
    for item in gold:
        if isinstance(item, dict):
            for key in ['answer', 'id', 'name', 'symbol']:
                if key in item and item[key]:
                    answers.add(normalize_answer(item[key]))
        elif isinstance(item, str):
            answers.add(normalize_answer(item))
    return answers


def compute_f1(predicted: Set[str], gold: Set[str]) -> float:
    """Compute F1 score with partial matching."""
    if not predicted or not gold:
        return 0.0

    matches = 0
    for pred in predicted:
        for g in gold:
            if pred == g or pred in g or g in pred:
                matches += 1
                break

    precision = matches / len(predicted) if predicted else 0
    recall = matches / len(gold) if gold else 0

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_exact_match(predicted: Set[str], gold: Set[str]) -> bool:
    """Check if any prediction matches any gold answer."""
    if not predicted or not gold:
        return False

    for pred in predicted:
        if pred in gold:
            return True
        for g in gold:
            if pred in g or g in pred:
                return True
    return False


def compute_accuracy(predicted: Set[str], gold: Set[str]) -> float:
    """Compute accuracy (1 if any match, 0 otherwise)."""
    return 1.0 if compute_exact_match(predicted, gold) else 0.0


# =============================================================================
# Data Loading
# =============================================================================

def load_questions(data_dir: str, task: str, split: str, q_type: Optional[str] = None) -> List[Dict]:
    """
    Load questions from CKG_Bench data files.

    Args:
        data_dir: Path to data directory
        task: 'knowledge', 'reasoning', or 'all'
        split: 'dev' or 'test'
        q_type: Optional question type filter ('S', 'R', 'C', 'M')

    Returns:
        List of question dictionaries
    """
    questions = []

    tasks_to_load = TASK_TYPES if task == 'all' else [task]
    types_to_load = [q_type] if q_type else QUESTION_TYPES

    for t in tasks_to_load:
        for qt in types_to_load:
            filepath = os.path.join(data_dir, t, f"{qt}_{t}_{split}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for q in data:
                            q['task'] = t
                            q['type'] = qt
                        questions.extend(data)
                    elif isinstance(data, dict) and 'questions' in data:
                        for q in data['questions']:
                            q['task'] = t
                            q['type'] = qt
                        questions.extend(data['questions'])

    return questions


def load_combined_questions(data_dir: str, split: str) -> List[Dict]:
    """Load questions from combined file."""
    filepath = os.path.join(data_dir, f"combined_{split}.json")
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []


# =============================================================================
# LLM Evaluation
# =============================================================================

def create_prompt(question: Dict, task: str) -> str:
    """Create prompt for LLM based on task type."""
    q_text = question.get('question', '')

    if task == 'knowledge':
        prompt = f"""You are a biomedical knowledge graph expert. Answer the following question based on your knowledge of genes, diseases, and their causal relationships.

Question: {q_text}

Provide a concise answer. If there are multiple answers, list them separated by commas.
Answer:"""

    elif task == 'reasoning':
        prompt = f"""You are a biomedical reasoning expert. Answer the following question by reasoning through the causal relationships between genes, SNPs, and diseases.

Question: {q_text}

Think step by step:
1. Identify the key entities mentioned
2. Consider the causal relationships
3. Derive the answer

Provide your final answer concisely. If there are multiple answers, list them separated by commas.
Answer:"""

    else:
        prompt = f"""Answer the following biomedical question:

Question: {q_text}

Answer:"""

    return prompt


def parse_llm_response(response: str) -> List[str]:
    """Parse LLM response to extract answers."""
    if not response:
        return []

    # Clean response
    response = response.strip()

    # Handle common response formats
    # Remove "Answer:" prefix if present
    if response.lower().startswith('answer:'):
        response = response[7:].strip()

    # Split by common delimiters
    answers = []
    for delimiter in [',', '\n', ';', ' and ']:
        if delimiter in response:
            parts = response.split(delimiter)
            answers = [p.strip() for p in parts if p.strip()]
            break

    if not answers:
        answers = [response]

    # Clean each answer
    cleaned = []
    for ans in answers:
        # Remove numbering like "1.", "2.", etc.
        ans = ans.lstrip('0123456789.-) ')
        if ans:
            cleaned.append(ans)

    return cleaned


def evaluate_question(
    question: Dict,
    llm_client,
    model: str
) -> Dict:
    """Evaluate a single question."""
    task = question.get('task', 'knowledge')
    prompt = create_prompt(question, task)

    try:
        response = llm_client.generate(prompt, model=model)
        predicted = parse_llm_response(response)
        success = True
        error = None
    except Exception as e:
        response = ""
        predicted = []
        success = False
        error = str(e)

    gold = question.get('answer', [])
    if isinstance(gold, str):
        gold = [gold]

    gold_set = extract_answers(gold)
    pred_set = set(normalize_answer(p) for p in predicted)

    return {
        'question_id': question.get('id', ''),
        'question': question.get('question', ''),
        'type': question.get('type', ''),
        'task': task,
        'gold_answers': list(gold_set),
        'predicted': predicted,
        'response': response,
        'success': success,
        'error': error,
        'f1': compute_f1(pred_set, gold_set),
        'accuracy': compute_accuracy(pred_set, gold_set),
        'exact_match': compute_exact_match(pred_set, gold_set),
    }


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate_model(
    model: str,
    questions: List[Dict],
    config_path: str,
    output_dir: str
) -> Dict:
    """
    Evaluate a model on CKG_Bench questions.

    Args:
        model: Model name
        questions: List of questions
        config_path: Path to config file
        output_dir: Directory for output files

    Returns:
        Dictionary with metrics and results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model}")
    print(f"Questions: {len(questions)}")
    print(f"{'='*60}")

    # Load config and initialize client
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llm_client = get_llm_client(config)

    # Evaluate each question
    results = []
    start_time = time.time()

    for q in tqdm(questions, desc=model):
        result = evaluate_question(q, llm_client, model)
        results.append(result)

    total_time = time.time() - start_time

    # Compute metrics
    metrics = compute_metrics(results)
    metrics['model'] = model
    metrics['total_time_sec'] = total_time
    metrics['total_questions'] = len(questions)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"bioresonkg_eval_{model}_{timestamp}.json")

    output_data = {
        'model': model,
        'timestamp': timestamp,
        'metrics': metrics,
        'results': results
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print_metrics(metrics)

    return output_data


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute aggregate metrics from results."""
    total = len(results)
    if total == 0:
        return {}

    # Overall metrics
    f1_sum = sum(r['f1'] for r in results)
    acc_sum = sum(r['accuracy'] for r in results)
    em_count = sum(1 for r in results if r['exact_match'])
    success_count = sum(1 for r in results if r['success'])

    metrics = {
        'total': total,
        'success_rate': success_count / total,
        'f1': f1_sum / total,
        'accuracy': acc_sum / total,
        'exact_match': em_count / total,
    }

    # Per-type metrics
    by_type = defaultdict(list)
    for r in results:
        by_type[r['type']].append(r)

    metrics['by_type'] = {}
    for qtype, type_results in by_type.items():
        n = len(type_results)
        metrics['by_type'][qtype] = {
            'total': n,
            'f1': sum(r['f1'] for r in type_results) / n,
            'accuracy': sum(r['accuracy'] for r in type_results) / n,
            'exact_match': sum(1 for r in type_results if r['exact_match']) / n,
        }

    # Per-task metrics
    by_task = defaultdict(list)
    for r in results:
        by_task[r['task']].append(r)

    metrics['by_task'] = {}
    for task, task_results in by_task.items():
        n = len(task_results)
        metrics['by_task'][task] = {
            'total': n,
            'f1': sum(r['f1'] for r in task_results) / n,
            'accuracy': sum(r['accuracy'] for r in task_results) / n,
            'exact_match': sum(1 for r in task_results if r['exact_match']) / n,
        }

    return metrics


def print_metrics(metrics: Dict):
    """Print metrics in formatted table."""
    print(f"\n{'='*60}")
    print("OVERALL METRICS")
    print(f"{'='*60}")
    print(f"  F1 Score:     {metrics.get('f1', 0)*100:.2f}%")
    print(f"  Accuracy:     {metrics.get('accuracy', 0)*100:.2f}%")
    print(f"  Exact Match:  {metrics.get('exact_match', 0)*100:.2f}%")
    print(f"  Success Rate: {metrics.get('success_rate', 0)*100:.2f}%")

    if 'by_type' in metrics:
        print(f"\n{'='*60}")
        print("BY QUESTION TYPE")
        print(f"{'='*60}")
        print(f"  {'Type':<10} {'F1':>10} {'Accuracy':>10} {'EM':>10}")
        print(f"  {'-'*40}")
        for qtype in QUESTION_TYPES:
            if qtype in metrics['by_type']:
                m = metrics['by_type'][qtype]
                print(f"  {qtype:<10} {m['f1']*100:>9.1f}% {m['accuracy']*100:>9.1f}% {m['exact_match']*100:>9.1f}%")

    if 'by_task' in metrics:
        print(f"\n{'='*60}")
        print("BY TASK TYPE")
        print(f"{'='*60}")
        print(f"  {'Task':<12} {'F1':>10} {'Accuracy':>10} {'EM':>10}")
        print(f"  {'-'*42}")
        for task in TASK_TYPES:
            if task in metrics['by_task']:
                m = metrics['by_task'][task]
                print(f"  {task:<12} {m['f1']*100:>9.1f}% {m['accuracy']*100:>9.1f}% {m['exact_match']*100:>9.1f}%")

    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BioResonKGBench Evaluation Script")
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='Model to evaluate')
    parser.add_argument('--task', type=str, default='all',
                        choices=['knowledge', 'reasoning', 'all'],
                        help='Task type to evaluate')
    parser.add_argument('--split', type=str, default='test',
                        choices=['dev', 'test'],
                        help='Data split to use')
    parser.add_argument('--type', type=str, default=None,
                        choices=['S', 'R', 'C', 'M'],
                        help='Question type filter')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--config', type=str, default='../01_BioKGBench/config/config.local.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--all-models', action='store_true',
                        help='Evaluate all default models')

    args = parser.parse_args()

    # Load questions
    print(f"Loading questions from {args.data_dir}...")
    questions = load_questions(args.data_dir, args.task, args.split, args.type)

    if not questions:
        # Try combined file
        questions = load_combined_questions(args.data_dir, args.split)
        if args.task != 'all':
            questions = [q for q in questions if q.get('task') == args.task]
        if args.type:
            questions = [q for q in questions if q.get('type') == args.type]

    if not questions:
        print(f"No questions found for task={args.task}, split={args.split}")
        return

    print(f"Loaded {len(questions)} questions")

    # Evaluate
    models = DEFAULT_MODELS if args.all_models else [args.model]

    for model in models:
        try:
            evaluate_model(model, questions, args.config, args.output_dir)
        except Exception as e:
            print(f"Error evaluating {model}: {e}")


if __name__ == '__main__':
    main()
