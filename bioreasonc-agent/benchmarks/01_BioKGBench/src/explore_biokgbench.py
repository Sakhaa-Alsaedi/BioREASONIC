#!/usr/bin/env python3
"""
Explore BioKGBench Dataset from HuggingFace.
"""

import json
from huggingface_hub import snapshot_download

print("="*70)
print("Downloading BioKGBench Dataset")
print("="*70)

# Download dataset
DATA_PATH = snapshot_download(
    repo_id="AutoLab-Westlake/BioKGBench-Dataset",
    repo_type="dataset"
)

print(f"\nDataset downloaded to: {DATA_PATH}")

# List contents
import os
print("\n" + "="*70)
print("Dataset Structure")
print("="*70)
for root, dirs, files in os.walk(DATA_PATH):
    level = root.replace(DATA_PATH, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = '  ' * (level + 1)
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        if size > 1024*1024:
            size_str = f"{size/(1024*1024):.1f}MB"
        elif size > 1024:
            size_str = f"{size/1024:.1f}KB"
        else:
            size_str = f"{size}B"
        print(f"{subindent}{file} ({size_str})")

# Load KGQA data
print("\n" + "="*70)
print("KGQA Data")
print("="*70)

kgqa_path = f"{DATA_PATH}/kgqa"
if os.path.exists(kgqa_path):
    with open(f"{kgqa_path}/dev.json", 'r') as f:
        kgqa_dev = json.load(f)
    with open(f"{kgqa_path}/test.json", 'r') as f:
        kgqa_test = json.load(f)

    print(f"KGQA Data loaded:")
    print(f"  Dev: {len(kgqa_dev)} samples")
    print(f"  Test: {len(kgqa_test)} samples")

    # Show sample
    print(f"\nSample question (dev[0]):")
    print(json.dumps(kgqa_dev[0], indent=2))

    # Analyze question types
    print("\n" + "="*70)
    print("Question Analysis")
    print("="*70)

    # Check structure of samples
    if kgqa_dev:
        sample = kgqa_dev[0]
        print(f"\nSample keys: {list(sample.keys())}")

        # Count by type if available
        if 'type' in sample:
            types = {}
            for item in kgqa_dev:
                t = item.get('type', 'unknown')
                types[t] = types.get(t, 0) + 1
            print("\nQuestion types in dev set:")
            for t, count in sorted(types.items(), key=lambda x: -x[1]):
                print(f"  {t}: {count}")

    # Show more samples
    print("\n" + "="*70)
    print("More Sample Questions")
    print("="*70)
    for i, sample in enumerate(kgqa_dev[:5]):
        print(f"\n--- Sample {i+1} ---")
        if 'question' in sample:
            print(f"Q: {sample['question']}")
        if 'answer' in sample:
            ans = sample['answer']
            if isinstance(ans, list):
                print(f"A: {ans[:3]}..." if len(ans) > 3 else f"A: {ans}")
            else:
                print(f"A: {ans}")
        if 'type' in sample:
            print(f"Type: {sample['type']}")
else:
    print("KGQA directory not found!")

# Check for other data types
print("\n" + "="*70)
print("Other Data Types")
print("="*70)

for subdir in os.listdir(DATA_PATH):
    subpath = os.path.join(DATA_PATH, subdir)
    if os.path.isdir(subpath) and subdir != 'kgqa':
        print(f"\n{subdir}/")
        for f in os.listdir(subpath)[:5]:
            print(f"  {f}")

        # Try to load and show sample
        json_files = [f for f in os.listdir(subpath) if f.endswith('.json')]
        if json_files:
            sample_file = os.path.join(subpath, json_files[0])
            try:
                with open(sample_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    print(f"  Sample from {json_files[0]}:")
                    print(f"    {json.dumps(data[0], indent=4)[:500]}...")
                elif isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:10]}")
            except Exception as e:
                print(f"  Error loading: {e}")

print("\n" + "="*70)
print("Done!")
print("="*70)
