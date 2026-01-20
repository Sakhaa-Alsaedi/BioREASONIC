#!/usr/bin/env python3
"""
Detailed analysis of BioKGBench KGQA dataset.
"""

import json
import os
from collections import Counter, defaultdict
from huggingface_hub import snapshot_download

print("="*70)
print("BioKGBench KGQA - Detailed Analysis")
print("="*70)

# Download/load dataset
DATA_PATH = snapshot_download(
    repo_id="AutoLab-Westlake/BioKGBench-Dataset",
    repo_type="dataset"
)

KGQA_PATH = f"{DATA_PATH}/kgqa"

# Load data
with open(f"{KGQA_PATH}/dev.json", 'r') as f:
    kgqa_dev = json.load(f)
with open(f"{KGQA_PATH}/test.json", 'r') as f:
    kgqa_test = json.load(f)

print(f"\nDataset Size:")
print(f"  Dev set:  {len(kgqa_dev)} questions")
print(f"  Test set: {len(kgqa_test)} questions")
print(f"  Total:    {len(kgqa_dev) + len(kgqa_test)} questions")

# Analyze structure
print("\n" + "="*70)
print("Question Structure")
print("="*70)
sample = kgqa_dev[0]
print(f"\nKeys in each sample: {list(sample.keys())}")
for key, value in sample.items():
    print(f"\n  {key}:")
    print(f"    Type: {type(value).__name__}")
    if isinstance(value, str):
        print(f"    Example: {value[:100]}...")
    elif isinstance(value, list):
        print(f"    Length: {len(value)}")
        if value:
            print(f"    First item: {value[0]}")
    elif isinstance(value, dict):
        print(f"    Keys: {list(value.keys())}")

# Analyze question types
print("\n" + "="*70)
print("Question Types Distribution")
print("="*70)

all_data = kgqa_dev + kgqa_test
type_counts = Counter(q.get('type', 'unknown') for q in all_data)
print("\nAll data:")
for t, count in type_counts.most_common():
    pct = count / len(all_data) * 100
    print(f"  {t}: {count} ({pct:.1f}%)")

print("\nDev set:")
dev_types = Counter(q.get('type', 'unknown') for q in kgqa_dev)
for t, count in dev_types.most_common():
    print(f"  {t}: {count}")

print("\nTest set:")
test_types = Counter(q.get('type', 'unknown') for q in kgqa_test)
for t, count in test_types.most_common():
    print(f"  {t}: {count}")

# Analyze entity types
print("\n" + "="*70)
print("Entity Types in Questions")
print("="*70)

entity_types = Counter()
entity_examples = defaultdict(list)
for q in all_data:
    entities = q.get('entities', {})
    for etype, eid in entities.items():
        entity_types[etype] += 1
        if len(entity_examples[etype]) < 3:
            entity_examples[etype].append(eid)

print("\nEntity types used:")
for etype, count in entity_types.most_common():
    examples = entity_examples[etype]
    print(f"  {etype}: {count} occurrences")
    print(f"    Examples: {examples}")

# Analyze answer types
print("\n" + "="*70)
print("Answer Analysis")
print("="*70)

answer_counts = []
answer_types = Counter()
for q in all_data:
    answers = q.get('answer', [])
    if isinstance(answers, list):
        answer_counts.append(len(answers))
        if answers and isinstance(answers[0], dict):
            # Check what the answer refers to
            for ans in answers:
                if 'answer' in ans:
                    aid = ans['answer']
                    if aid is None:
                        answer_types['null'] += 1
                    elif isinstance(aid, str):
                        if aid.startswith('P') and len(aid) > 1 and aid[1:].replace('_', '').replace('-', '').isalnum():
                            answer_types['protein'] += 1
                        elif aid.startswith('DOID:'):
                            answer_types['disease'] += 1
                        elif aid.startswith('GO:'):
                            answer_types['GO_term'] += 1
                        elif aid.startswith('BTO:'):
                            answer_types['tissue'] += 1
                        elif aid.startswith('R-HSA-'):
                            answer_types['pathway'] += 1
                        elif aid.startswith('SMP'):
                            answer_types['pathway'] += 1
                        else:
                            answer_types['other'] += 1
                    else:
                        answer_types['other'] += 1
    else:
        answer_counts.append(1)

print(f"\nAnswer count statistics:")
print(f"  Min answers: {min(answer_counts)}")
print(f"  Max answers: {max(answer_counts)}")
print(f"  Avg answers: {sum(answer_counts)/len(answer_counts):.2f}")

print(f"\nAnswer entity types:")
for atype, count in answer_types.most_common():
    print(f"  {atype}: {count}")

# Analyze question patterns
print("\n" + "="*70)
print("Question Patterns")
print("="*70)

question_starts = Counter()
for q in all_data:
    question = q.get('question', '')
    words = question.split()[:3]
    start = ' '.join(words)
    question_starts[start] += 1

print("\nCommon question starts:")
for start, count in question_starts.most_common(15):
    print(f"  '{start}...': {count}")

# Show examples by type
print("\n" + "="*70)
print("Examples by Question Type")
print("="*70)

for qtype in ['one-hop', 'multi-hop', 'conjunction']:
    print(f"\n--- {qtype.upper()} ---")
    examples = [q for q in all_data if q.get('type') == qtype][:3]
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"  Q: {ex['question']}")
        print(f"  Source: {ex.get('source', 'N/A')}")
        print(f"  Entities: {ex.get('entities', {})}")
        answers = ex.get('answer', [])
        if isinstance(answers, list) and answers:
            if len(answers) <= 3:
                print(f"  Answers: {answers}")
            else:
                print(f"  Answers ({len(answers)} total): {answers[:2]}...")

# Analyze relationship patterns in questions
print("\n" + "="*70)
print("Relationship Patterns (inferred from questions)")
print("="*70)

relationship_keywords = {
    'interact': 'protein-protein interaction',
    'pathway': 'protein-pathway',
    'disease': 'protein-disease',
    'tissue': 'protein-tissue',
    'biological process': 'protein-GO_BP',
    'cellular component': 'protein-GO_CC',
    'molecular function': 'protein-GO_MF',
    'gene': 'gene-protein',
}

rel_counts = Counter()
for q in all_data:
    question = q.get('question', '').lower()
    for keyword, rel in relationship_keywords.items():
        if keyword in question:
            rel_counts[rel] += 1

print("\nRelationship types in questions:")
for rel, count in rel_counts.most_common():
    print(f"  {rel}: {count}")

# Save summary to file
print("\n" + "="*70)
print("Saving detailed data...")
print("="*70)

summary = {
    'total_questions': len(all_data),
    'dev_size': len(kgqa_dev),
    'test_size': len(kgqa_test),
    'question_types': dict(type_counts),
    'entity_types': dict(entity_types),
    'answer_types': dict(answer_types),
    'relationship_patterns': dict(rel_counts),
}

with open('kgqa_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Summary saved to kgqa_summary.json")

# Save all questions for reference
with open('kgqa_all_questions.json', 'w') as f:
    json.dump(all_data, f, indent=2)
print("All questions saved to kgqa_all_questions.json")

print("\n" + "="*70)
print("Done!")
print("="*70)
