#!/usr/bin/env python3
"""
Evaluate the KG QA System on BioKGBench KGQA Benchmark.

Metrics:
- Hits@1, Hits@5, Hits@10: Answer in top K results
- MRR: Mean Reciprocal Rank
- Accuracy: Exact match
- Coverage: % of questions that can be answered
"""

import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Set
from huggingface_hub import snapshot_download
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from kg_qa_system import KnowledgeGraphQA, QAResult

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
            # Check different possible keys
            for key in ['answer', 'id', 'name']:
                if key in item and item[key]:
                    answers.add(normalize_answer(item[key]))
        elif isinstance(item, str):
            answers.add(normalize_answer(item))
    return answers

def compute_metrics(results: List[Dict]) -> Dict:
    """Compute evaluation metrics."""
    total = len(results)
    hits_1 = 0
    hits_5 = 0
    hits_10 = 0
    mrr_sum = 0.0
    answered = 0
    correct = 0

    by_type = defaultdict(lambda: {
        'total': 0, 'answered': 0, 'hits_1': 0, 'hits_5': 0, 'hits_10': 0, 'mrr': 0.0
    })

    for r in results:
        qtype = r.get('question_type', 'unknown')
        by_type[qtype]['total'] += 1

        if not r['success'] or not r['predicted']:
            continue

        answered += 1
        by_type[qtype]['answered'] += 1

        gold_answers = r['gold_answers']
        predicted = [normalize_answer(a) for a in r['predicted']]

        # Find best rank
        best_rank = None
        for i, pred in enumerate(predicted):
            if pred in gold_answers:
                best_rank = i + 1
                break
            # Also check if gold answer is contained in prediction (for names)
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
                by_type[qtype]['hits_10'] += 1

            mrr_sum += 1.0 / best_rank
            by_type[qtype]['mrr'] += 1.0 / best_rank
            correct += 1

    metrics = {
        'total': total,
        'answered': answered,
        'coverage': answered / total if total > 0 else 0,
        'hits_1': hits_1 / answered if answered > 0 else 0,
        'hits_5': hits_5 / answered if answered > 0 else 0,
        'hits_10': hits_10 / answered if answered > 0 else 0,
        'mrr': mrr_sum / answered if answered > 0 else 0,
        'accuracy': correct / total if total > 0 else 0,
    }

    # Per-type metrics
    metrics['by_type'] = {}
    for qtype, stats in by_type.items():
        ans = stats['answered']
        metrics['by_type'][qtype] = {
            'total': stats['total'],
            'answered': ans,
            'coverage': ans / stats['total'] if stats['total'] > 0 else 0,
            'hits_1': stats['hits_1'] / ans if ans > 0 else 0,
            'hits_5': stats['hits_5'] / ans if ans > 0 else 0,
            'hits_10': stats['hits_10'] / ans if ans > 0 else 0,
            'mrr': stats['mrr'] / ans if ans > 0 else 0,
        }

    return metrics

def evaluate(qa_system: KnowledgeGraphQA, questions: List[Dict], verbose: bool = False) -> Dict:
    """Evaluate the QA system on a set of questions."""
    results = []

    for q in tqdm(questions, desc="Evaluating"):
        question_text = q.get('question', '')
        gold = q.get('answer', [])
        qtype = q.get('type', 'unknown')

        # Get QA result
        result = qa_system.answer(question_text, q)

        # Extract predicted answers - prioritize names over IDs
        predicted_names = []
        predicted_ids = []
        if result.success and result.answers:
            for ans in result.answers:
                if ans.name:
                    predicted_names.append(ans.name)
                if ans.id:
                    predicted_ids.append(ans.id)

        # Names first (most answers are expected as names), then IDs
        all_predicted = predicted_names + predicted_ids

        results.append({
            'question': question_text,
            'question_type': qtype,
            'gold_answers': extract_gold_answers(gold),
            'predicted': all_predicted,
            'success': result.success,
            'error': result.error,
        })

        if verbose and not result.success:
            print(f"\nFailed: {question_text[:80]}...")
            print(f"  Error: {result.error}")

    return compute_metrics(results), results

def main():
    print("="*70)
    print("BioKGBench KGQA Evaluation")
    print("="*70)

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

    print(f"Loaded {len(dev_data)} dev, {len(test_data)} test questions")

    # Initialize QA system
    print("\nInitializing QA system...")
    qa = KnowledgeGraphQA()
    qa.connect()
    print("Connected to Neo4j!")

    # Evaluate on dev set
    print("\n" + "="*70)
    print("Evaluating on DEV set")
    print("="*70)

    dev_metrics, dev_results = evaluate(qa, dev_data, verbose=False)

    print(f"\nDev Set Results:")
    print(f"  Coverage: {dev_metrics['coverage']*100:.1f}%")
    print(f"  Hits@1:   {dev_metrics['hits_1']*100:.1f}%")
    print(f"  Hits@5:   {dev_metrics['hits_5']*100:.1f}%")
    print(f"  Hits@10:  {dev_metrics['hits_10']*100:.1f}%")
    print(f"  MRR:      {dev_metrics['mrr']:.3f}")

    print("\n  By Question Type:")
    for qtype, stats in dev_metrics['by_type'].items():
        print(f"    {qtype}:")
        print(f"      Coverage: {stats['coverage']*100:.1f}%")
        print(f"      Hits@1:   {stats['hits_1']*100:.1f}%")

    # Evaluate on test set
    print("\n" + "="*70)
    print("Evaluating on TEST set")
    print("="*70)

    test_metrics, test_results = evaluate(qa, test_data, verbose=False)

    print(f"\nTest Set Results:")
    print(f"  Coverage: {test_metrics['coverage']*100:.1f}%")
    print(f"  Hits@1:   {test_metrics['hits_1']*100:.1f}%")
    print(f"  Hits@5:   {test_metrics['hits_5']*100:.1f}%")
    print(f"  Hits@10:  {test_metrics['hits_10']*100:.1f}%")
    print(f"  MRR:      {test_metrics['mrr']:.3f}")

    print("\n  By Question Type:")
    for qtype, stats in test_metrics['by_type'].items():
        print(f"    {qtype}:")
        print(f"      Coverage: {stats['coverage']*100:.1f}%")
        print(f"      Hits@1:   {stats['hits_1']*100:.1f}%")
        print(f"      Hits@5:   {stats['hits_5']*100:.1f}%")

    # Save detailed results
    output = {
        'dev': {
            'metrics': dev_metrics,
            'num_questions': len(dev_data)
        },
        'test': {
            'metrics': test_metrics,
            'num_questions': len(test_data)
        }
    }

    with open('evaluation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to evaluation_results.json")

    # Save detailed predictions
    with open('dev_predictions.json', 'w') as f:
        json.dump(dev_results, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))

    with open('test_predictions.json', 'w') as f:
        json.dump(test_results, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))
    print("Detailed predictions saved")

    qa.close()

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"""
┌─────────────────────────────────────────────────────────┐
│                    KGQA Benchmark Results               │
├─────────────────────────────────────────────────────────┤
│ Dataset       │ Coverage │ Hits@1  │ Hits@5  │   MRR   │
├───────────────┼──────────┼─────────┼─────────┼─────────┤
│ Dev (n={len(dev_data):3d})   │  {dev_metrics['coverage']*100:5.1f}%  │ {dev_metrics['hits_1']*100:5.1f}%  │ {dev_metrics['hits_5']*100:5.1f}%  │ {dev_metrics['mrr']:.3f}   │
│ Test (n={len(test_data):3d})  │  {test_metrics['coverage']*100:5.1f}%  │ {test_metrics['hits_1']*100:5.1f}%  │ {test_metrics['hits_5']*100:5.1f}%  │ {test_metrics['mrr']:.3f}   │
└─────────────────────────────────────────────────────────┘
""")

if __name__ == "__main__":
    main()
