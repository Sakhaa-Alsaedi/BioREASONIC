#!/usr/bin/env python3
"""
KGQA Evaluation Script - BioKGBench & BioResonKGBench
======================================================
This script evaluates LLMs on Knowledge Graph Question Answering tasks.

Usage:
    python run_kgqa_evaluation.py --dataset biokgbench --samples 50
    python run_kgqa_evaluation.py --dataset bioresonkgbench --samples 50
    python run_kgqa_evaluation.py --dataset both --samples 50

Author: ROCKET KG Team
"""

import argparse
import json
import pandas as pd
import time
import sys
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='KGQA Evaluation Script')
    parser.add_argument('--dataset', type=str, default='both', 
                        choices=['biokgbench', 'bioresonkgbench', 'both'],
                        help='Dataset to evaluate: biokgbench, bioresonkgbench, or both')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples per model per approach (default: 50)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to evaluate (default: all)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV filename (default: auto-generated)')
    return parser.parse_args()

def patch_function_source(code):
    """Inject code to capture detailed results per sample."""
    if 'm = compute_metrics(' in code and 'detailed_results.append' not in code:
        patch_str = """
m = compute_metrics(predictions, gold, raw_response=locals().get('raw_response', locals().get('output_text', '')), question_metadata=q, kg_entities=locals().get('kg_entities'))

# PATCH: Capture detailed row with ALL metrics
_d_row = m.copy()
_d_row['question'] = q.get('question', '')
_d_row['response'] = locals().get('output_text') or locals().get('raw_response') or ''
_d_row['gold_answers'] = gold
_d_row['latency_ms'] = locals().get('latency', 0)

_resp = locals().get('response')
if _resp and hasattr(_resp, 'usage'):
    _d_row['input_tokens'] = getattr(_resp.usage, 'input_tokens', getattr(_resp.usage, 'prompt_tokens', 0))
    _d_row['output_tokens'] = getattr(_resp.usage, 'output_tokens', getattr(_resp.usage, 'completion_tokens', 0))
    _d_row['total_tokens'] = _d_row['input_tokens'] + _d_row['output_tokens']

detailed_results.append(_d_row)
"""
        lines = code.split('\n')
        new_lines = []
        skip_mode = False
        
        for line in lines:
            if skip_mode:
                if ')' in line:
                    skip_mode = False
                continue
                
            if 'm = compute_metrics(' in line:
                indent = line[:line.find('m =')]
                dedented = textwrap.dedent(patch_str).strip()
                for p_line in dedented.split('\n'):
                    new_lines.append(indent + p_line)
                if ')' not in line:
                    skip_mode = True
            else:
                new_lines.append(line)
        code = '\n'.join(new_lines)
        
    if 'def evaluate_' in code and 'detailed_results = []' not in code:
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if line.strip().startswith('def evaluate_') and ':' in line:
                match = re.match(r'^(\s*)', line)
                indent = match.group(1) if match else ''
                func_indent = indent + '    '
                new_lines.append(f"{func_indent}detailed_results = []")
        code = '\n'.join(new_lines)
    
    if 'def evaluate_' in code and "result['results'] = detailed_results" not in code:
        code = code.replace("return result", "result['results'] = detailed_results\n    return result")
    
    return code

def main():
    args = parse_args()
    
    print("=" * 100)
    print("🚀 KGQA EVALUATION SCRIPT")
    print(f"   Dataset: {args.dataset.upper()}")
    print(f"   Samples: {args.samples}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # Load notebook
    print("\n📦 Loading evaluation framework...")
    NOTEBOOK_PATH = Path(__file__).parent / 'Bioreasonic.ipynb'
    
    with open(NOTEBOOK_PATH) as f:
        nb = json.load(f)

    code_cells = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
    exec_globals = {'__name__': '__main__', '__doc__': None}

    for i, cell in enumerate(code_cells[:20]):
        source = ''.join(cell.get('source', []))
        if 'def evaluate_' in source:
            source = patch_function_source(source)
        try:
            exec(source, exec_globals)
        except Exception as e:
            print(f"❌ Error executing cell {i}: {e}")

    print("✅ Framework loaded!")

    # Get functions
    evaluate_no_kg = exec_globals.get('evaluate_no_kg')
    evaluate_with_kg = exec_globals.get('evaluate_with_kg')
    evaluate_react_cot_with_kg = exec_globals.get('evaluate_react_cot_with_kg')
    evaluate_multiagent_with_kg = exec_globals.get('evaluate_multiagent_with_kg')
    get_biokgbench_gold = exec_globals.get('get_biokgbench_gold')
    get_bioresonkgbench_gold = exec_globals.get('get_bioresonkgbench_gold')
    load_biokgbench = exec_globals.get('load_biokgbench')
    load_bioresonkgbench = exec_globals.get('load_bioresonkgbench')
    MODELS = exec_globals.get('MODELS', [])
    MODEL_DISPLAY = exec_globals.get('MODEL_DISPLAY', {})
    config = exec_globals.get('config', {})
    driver = exec_globals.get('driver')

    # Filter models if specified
    if args.models:
        MODELS = [m for m in MODELS if any(name.lower() in m.lower() for name in args.models)]
        print(f"   Filtered models: {len(MODELS)}")

    # Load datasets
    datasets = {}
    if args.dataset in ['biokgbench', 'both']:
        datasets['BioKGBench'] = {
            'data': load_biokgbench('test', args.samples),
            'gold_fn': get_biokgbench_gold
        }
    if args.dataset in ['bioresonkgbench', 'both']:
        datasets['BioResonKGBench'] = {
            'data': load_bioresonkgbench('test', args.samples),
            'gold_fn': get_bioresonkgbench_gold
        }

    print(f"\n📊 Loaded datasets: {list(datasets.keys())}")

    # Output setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = args.output or f'detailed_results_{args.dataset}_{timestamp}.csv'
    
    all_rows = []

    def extract_sample_row(sample, model, approach, dataset_name):
        return {
            'Dataset': dataset_name,
            'Model': MODEL_DISPLAY.get(model, model),
            'Approach': approach,
            'Question': sample.get('question', ''),
            'Model_Answer': sample.get('response', ''),
            'Ground_Truth': str(sample.get('gold_answers', '')),
            'EM': sample.get('em', 0),
            'F1': sample.get('f1', 0) * 100 if sample.get('f1', 0) <= 1 else sample.get('f1', 0),
            'Accuracy': sample.get('em', 0),
            'Success': sample.get('em', 0) > 0,
            'Precision': sample.get('precision', 0),
            'Recall': sample.get('recall', 0),
            'Hits@1': sample.get('hits1', 0),
            'Hits@5': sample.get('hits5', 0),
            'Hits@10': sample.get('hits10', 0),
            'MRR': sample.get('mrr', 0),
            'Containment': sample.get('containment', 0),
            'ECE': sample.get('ece', 0),
            'BLEU': sample.get('bleu', 0),
            'ROUGE-1': sample.get('rouge1', 0),
            'ROUGE-L': sample.get('rougeL', 0),
            'Semantic_Sim': sample.get('semantic_sim', 0),
            'Latency_ms': sample.get('latency_ms', 0),
            'Input_Tokens': sample.get('input_tokens', 0),
            'Output_Tokens': sample.get('output_tokens', 0),
            'Total_Tokens': sample.get('input_tokens', 0) + sample.get('output_tokens', 0),
            'LLM_Calls': 1,
            'Path_Validity': sample.get('path_validity', 0),
            'Reasoning_Depth': sample.get('reasoning_depth', 0),
            'Steps': sample.get('iterations', 0),
            'Hallucination': sample.get('hallucination', 0),
            'Confidence': sample.get('confidence', 0.5),
            'Abstention': sample.get('abstention', 0),
            'Type': sample.get('question_type', 'N/A'),
            'Error': '',
        }

    def process_result(r, name, model, dataset_name):
        samples = r.get('results', []) if r else []
        if samples:
            avg_latency = r.get('avg_latency_ms', 0)
            avg_input_tok = r.get('avg_input_tokens', 0)
            avg_output_tok = r.get('avg_output_tokens', 0)
            avg_llm_calls = r.get('avg_llm_calls', 1)
            
            for sample in samples:
                row = extract_sample_row(sample, model, name, dataset_name)
                if row['Latency_ms'] == 0 and avg_latency > 0:
                    row['Latency_ms'] = avg_latency
                if row['Input_Tokens'] == 0 and avg_input_tok > 0:
                    row['Input_Tokens'] = avg_input_tok
                if row['Output_Tokens'] == 0 and avg_output_tok > 0:
                    row['Output_Tokens'] = avg_output_tok
                if row['Total_Tokens'] == 0:
                    row['Total_Tokens'] = avg_input_tok + avg_output_tok
                if name == 'Multi-Agent':
                    row['LLM_Calls'] = int(avg_llm_calls) if avg_llm_calls > 0 else 4
                all_rows.append(row)
            
            df = pd.DataFrame(all_rows)
            df.to_csv(output_csv, index=False)
            print(f"      ✅ {len(samples)} samples (EM: {r.get('em', 0):.1f}%)")

    # Run evaluation
    for dataset_name, ds in datasets.items():
        print(f"\n{'='*80}")
        print(f"📊 EVALUATING: {dataset_name}")
        print('='*80)
        
        for model in MODELS:
            print(f"\n  🔄 MODEL: {MODEL_DISPLAY.get(model, model)}")
            
            for approach_name, eval_fn in [
                ('Direct (No KG)', lambda: evaluate_no_kg(ds['data'], ds['gold_fn'], model, driver)),
                ('Cypher KG', lambda: evaluate_with_kg(ds['data'], ds['gold_fn'], model, config, driver)),
                ('ReAct-COT', lambda: evaluate_react_cot_with_kg(ds['data'], ds['gold_fn'], model, config, driver)),
                ('Multi-Agent', lambda: evaluate_multiagent_with_kg(ds['data'], ds['gold_fn'], model, config, driver)),
            ]:
                print(f"    [{approach_name}] Running...", end=' ')
                try:
                    r = eval_fn()
                    process_result(r, approach_name, model, dataset_name)
                except Exception as e:
                    print(f"❌ Error: {e}")

    # Final save
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 100)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"   Total rows: {len(all_rows)}")
    print(f"   Output: {output_csv}")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    if driver:
        driver.close()

if __name__ == '__main__':
    main()
