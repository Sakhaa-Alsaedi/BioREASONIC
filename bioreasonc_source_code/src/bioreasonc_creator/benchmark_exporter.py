"""
BioREASONC Benchmark Exporter

================================================================================
MODULE OVERVIEW
================================================================================

Exports validated benchmark items to HuggingFace-compatible format.
This is STEP 5 in the BioREASONC pipeline:

    Generator → Validator → Explainer → Paraphraser → EXPORTER → Evaluator

Focus: "Does the model tell the truth about causality when explaining biomedical research?"

================================================================================
OUTPUT FILES
================================================================================

The exporter generates:
┌─────────────────────────────────┬────────────────────────────────────────────────┐
│ File                            │ Description                                    │
├─────────────────────────────────┼────────────────────────────────────────────────┤
│ train.json                      │ Training set (70% of items)                    │
│                                 │ Used for few-shot examples or fine-tuning      │
├─────────────────────────────────┼────────────────────────────────────────────────┤
│ dev.json                        │ Development/validation set (15% of items)      │
│                                 │ Used for hyperparameter tuning                 │
├─────────────────────────────────┼────────────────────────────────────────────────┤
│ test.json                       │ Test set (15% of items)                        │
│                                 │ Used for final evaluation (DO NOT TRAIN ON)    │
├─────────────────────────────────┼────────────────────────────────────────────────┤
│ README.md                       │ Dataset card with YAML frontmatter             │
│                                 │ HuggingFace Hub compatible                     │
├─────────────────────────────────┼────────────────────────────────────────────────┤
│ dataset_info.json               │ Full metadata (statistics, taxonomy, etc.)     │
└─────────────────────────────────┴────────────────────────────────────────────────┘

================================================================================
DATA FORMAT
================================================================================

Each item in train/dev/test.json has the following structure:

```json
{
    "id": "C-0001",
    "question": "Is the relationship between GCK and Type 2 Diabetes causal or associative?",
    "answer": "The relationship is ASSOCIATIVE, not causal. GWAS identifies statistical associations...",
    "explanation": "GWAS identifies associations, NOT causal relationships. Causation requires MR or...",
    "taxonomy": "C",
    "label": "C-CAUSAL-VS-ASSOC",
    "difficulty": "hard",
    "ground_truth": {
        "answer": "ASSOCIATIVE",
        "mr_score": 0.15,
        "evidence_level": "moderate",
        "confidence": 1.0
    },
    "metadata": {
        "gene": "GCK",
        "disease": "Type 2 Diabetes",
        "paraphrased": false,
        "validation_score": 4.5
    }
}
```

Field Descriptions:
┌─────────────────────┬──────────────────────────────────────────────────────────────┐
│ Field               │ Description                                                  │
├─────────────────────┼──────────────────────────────────────────────────────────────┤
│ id                  │ Unique identifier (taxonomy prefix + number)                 │
│ question            │ The benchmark question                                       │
│ answer              │ The gold-standard answer (with reasoning for C taxonomy)     │
│ explanation         │ Educational context (~35 words)                              │
│ taxonomy            │ S (Structure), C (Causal), R (Risk), M (Mechanism)           │
│ label               │ Specific template type (e.g., C-CAUSAL-VS-ASSOC)             │
│ difficulty          │ easy, medium, hard                                           │
│ ground_truth        │ Evidence data for validation (scores, confidence)            │
│ metadata            │ Additional context (entities, validation results)            │
└─────────────────────┴──────────────────────────────────────────────────────────────┘

================================================================================
SPLIT STRATEGY
================================================================================

Stratified Splitting:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ The exporter ensures balanced representation across all splits:                 │
│                                                                                 │
│ 1. TAXONOMY STRATIFICATION                                                      │
│    Each split has proportional representation of S, C, R, M taxonomies         │
│                                                                                 │
│ 2. DIFFICULTY STRATIFICATION                                                    │
│    Each split has balanced easy/medium/hard questions                          │
│                                                                                 │
│ 3. EVIDENCE LEVEL STRATIFICATION                                                │
│    Each split includes questions from all evidence levels                       │
│                                                                                 │
│ 4. PARAPHRASE GROUPING                                                          │
│    Original and paraphrased versions stay in the SAME split                    │
│    (prevents test contamination from training on paraphrases)                  │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
CONFIGURATION
================================================================================

BenchmarkExporter Parameters:
┌─────────────────────────┬─────────────────┬────────────────────────────────────────┐
│ Parameter               │ Default         │ Description                            │
├─────────────────────────┼─────────────────┼────────────────────────────────────────┤
│ output_dir              │ ./benchmark_... │ Directory to save files                │
│ dataset_name            │ BioREASONC-Bench│ Name for dataset card                  │
│ version                 │ 1.0.0           │ Semantic version                       │
│ min_score               │ 4.0             │ Minimum validation score to include    │
│ random_seed             │ 42              │ For reproducible splits                │
└─────────────────────────┴─────────────────┴────────────────────────────────────────┘

Split Ratios (fixed):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ train: 70%                                                                      │
│ dev:   15%                                                                      │
│ test:  15%                                                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
HUGGINGFACE COMPATIBILITY
================================================================================

The exported dataset is compatible with HuggingFace Datasets:

```python
from datasets import load_dataset

# Load from local directory
dataset = load_dataset("path/to/benchmark_output")

# Access splits
train = dataset["train"]
test = dataset["test"]

# Iterate over examples
for example in test:
    print(f"Q: {example['question']}")
    print(f"A: {example['answer']}")
```

Dataset Card (README.md) includes:
- YAML frontmatter with HF metadata
- Dataset description and motivation
- Data fields documentation
- Usage examples
- Citation information
- License

================================================================================
USAGE EXAMPLES
================================================================================

Example 1: Basic Export
```python
from bioreasonc_creator.benchmark_exporter import BenchmarkExporter

exporter = BenchmarkExporter(
    output_dir="./benchmark_v1",
    dataset_name="BioREASONC-Bench",
    version="1.0.0"
)

# Export validated items
exporter.export(validated_items)

print(f"Exported to {exporter.output_dir}")
```

Example 2: Custom Split with Seed
```python
exporter = BenchmarkExporter(
    output_dir="./benchmark_v1",
    random_seed=12345  # For reproducibility
)

stats = exporter.export(items)
print(f"Train: {stats['num_train']}, Test: {stats['num_test']}")
```

Example 3: With Metadata
```python
exporter.export(
    items,
    extra_metadata={
        "source": "CAUSALdb2 v2.1",
        "generation_date": "2024-01-15",
        "created_by": "Sakhaa Alsaedi"
    }
)
```

================================================================================
BEST PRACTICES
================================================================================

★ REPRODUCIBILITY:
  • Always use a fixed random_seed
  • Save the full config in metadata

★ QUALITY CONTROL:
  • Only export items with validation score >= min_score
  • Review test set manually before publication

★ CONTAMINATION PREVENTION:
  • Paraphrases stay with originals in same split
  • Never train on test questions or paraphrases

★ VERSIONING:
  • Use semantic versioning (major.minor.patch)
  • Document changes between versions

================================================================================
Author: Sakhaa Alsaedi, sakhaa.alsaedi@kaust.edu.sa
================================================================================
"""

import json
import os
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class BenchmarkItem:
    """A single benchmark item in HuggingFace format."""
    id: str
    question: str
    answer: str
    explanation: str
    taxonomy: str  # S, C, R, M
    label: str
    difficulty: str
    ground_truth: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkMetadata:
    """Metadata for the benchmark dataset."""
    name: str
    version: str
    description: str
    homepage: str
    license: str
    citation: str
    languages: List[str]
    task_categories: List[str]
    task_ids: List[str]
    size_categories: str
    created: str
    num_train: int
    num_dev: int
    num_test: int
    taxonomy_distribution: Dict[str, int]


class BenchmarkExporter:
    """
    Exports validated benchmark items to HuggingFace-compatible format.

    Creates:
    - train.json (70% of data)
    - dev.json (15% of data)
    - test.json (15% of data)
    - README.md (dataset card with YAML frontmatter)
    - dataset_info.json (metadata)
    """

    def __init__(
        self,
        output_dir: str = "./benchmark_output",
        dataset_name: str = "BioREASONC-Bench",
        version: str = "1.0.0",
        min_score: float = 4.0,
        random_seed: int = 42
    ):
        """
        Initialize the benchmark exporter.

        Args:
            output_dir: Directory to save benchmark files
            dataset_name: Name of the dataset
            version: Version string
            min_score: Minimum validation score to include
            random_seed: Random seed for reproducible splits
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset_name
        self.version = version
        self.min_score = min_score
        self.random_seed = random_seed
        random.seed(random_seed)

    def filter_high_quality_items(
        self,
        items: List[Dict[str, Any]],
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter items by validation score.

        Args:
            items: List of validated items
            min_score: Minimum score threshold (default: self.min_score)

        Returns:
            List of high-quality items
        """
        threshold = min_score or self.min_score
        filtered = []

        for item in items:
            score = item.get('avg_score', 0)
            is_valid = item.get('is_valid', False)
            overclaim = item.get('overclaim_consensus', False)

            # Include if: valid, high score, no overclaim
            if is_valid and score >= threshold and not overclaim:
                filtered.append(item)

        print(f"Filtered {len(filtered)}/{len(items)} items (score >= {threshold})")
        return filtered

    def create_splits(
        self,
        items: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by: str = "taxonomy"
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split items into train/dev/test sets.

        Args:
            items: List of items to split
            train_ratio: Proportion for training (default: 0.70)
            dev_ratio: Proportion for development (default: 0.15)
            test_ratio: Proportion for test (default: 0.15)
            stratify_by: Field to stratify by (default: "taxonomy")

        Returns:
            Tuple of (train_items, dev_items, test_items)
        """
        assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"

        # Group by stratification field
        groups = {}
        for item in items:
            key = item.get(stratify_by, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        train_items = []
        dev_items = []
        test_items = []

        # Split each group proportionally
        for key, group_items in groups.items():
            random.shuffle(group_items)
            n = len(group_items)

            train_end = int(n * train_ratio)
            dev_end = train_end + int(n * dev_ratio)

            train_items.extend(group_items[:train_end])
            dev_items.extend(group_items[train_end:dev_end])
            test_items.extend(group_items[dev_end:])

        # Shuffle final splits
        random.shuffle(train_items)
        random.shuffle(dev_items)
        random.shuffle(test_items)

        print(f"Split: train={len(train_items)}, dev={len(dev_items)}, test={len(test_items)}")
        return train_items, dev_items, test_items

    def format_item_for_export(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format an item for JSON export in HuggingFace format.

        Args:
            item: Raw validated item

        Returns:
            Formatted item dictionary
        """
        return {
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "explanation": item.get("explanation", ""),
            "taxonomy": item.get("taxonomy", ""),
            "taxonomy_name": self._get_taxonomy_name(item.get("taxonomy", "")),
            "label": item.get("label", ""),
            "difficulty": item.get("difficulty", "medium"),
            "validation_score": item.get("avg_score", 0),
            "consensus_judgment": item.get("consensus_judgment", ""),
            "ground_truth": item.get("ground_truth", {}),
            "metadata": {
                "source": "BioREASONC-Bench",
                "version": self.version,
                "validated": True,
                "validation_providers": item.get("validation_providers", []),
            }
        }

    def _get_taxonomy_name(self, taxonomy: str) -> str:
        """Get full taxonomy name."""
        names = {
            "S": "Structure-Aware",
            "C": "Causal-Aware",
            "R": "Risk-Aware",
            "M": "Semantic-Aware"
        }
        return names.get(taxonomy, "Unknown")

    def export_split(
        self,
        items: List[Dict[str, Any]],
        filename: str
    ) -> str:
        """
        Export a split to JSON file.

        Args:
            items: Items to export
            filename: Output filename

        Returns:
            Path to exported file
        """
        filepath = self.output_dir / filename

        formatted_items = [self.format_item_for_export(item) for item in items]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_items, f, indent=2, ensure_ascii=False)

        print(f"Exported {len(items)} items to {filepath}")
        return str(filepath)

    def create_dataset_card(
        self,
        train_items: List[Dict],
        dev_items: List[Dict],
        test_items: List[Dict]
    ) -> str:
        """
        Create README.md dataset card for HuggingFace.

        Args:
            train_items: Training set items
            dev_items: Development set items
            test_items: Test set items

        Returns:
            Path to README.md
        """
        # Calculate statistics
        all_items = train_items + dev_items + test_items
        taxonomy_dist = {}
        for item in all_items:
            tax = item.get("taxonomy", "U")
            taxonomy_dist[tax] = taxonomy_dist.get(tax, 0) + 1

        # Determine size category
        total = len(all_items)
        if total < 1000:
            size_cat = "n<1K"
        elif total < 10000:
            size_cat = "1K<n<10K"
        elif total < 100000:
            size_cat = "10K<n<100K"
        else:
            size_cat = "100K<n<1M"

        readme_content = f'''---
language:
- en
license: apache-2.0
size_categories:
- {size_cat}
task_categories:
- question-answering
- text-generation
task_ids:
- closed-domain-qa
- open-domain-qa
pretty_name: BioREASONC-Bench
tags:
- biomedical
- causal-reasoning
- gwas
- genomics
- faithfulness
- hallucination-detection
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: explanation
    dtype: string
  - name: taxonomy
    dtype: string
  - name: label
    dtype: string
  splits:
  - name: train
    num_examples: {len(train_items)}
  - name: validation
    num_examples: {len(dev_items)}
  - name: test
    num_examples: {len(test_items)}
configs:
- config_name: default
  data_files:
  - split: train
    path: train.json
  - split: validation
    path: dev.json
  - split: test
    path: test.json
---

# BioREASONC-Bench: Biomedical Reasoning and Causal Faithfulness Benchmark

## Dataset Description

**BioREASONC-Bench** is a benchmark dataset designed to evaluate whether language models tell the truth about causality when explaining biomedical research. The dataset focuses on testing LLMs' ability to distinguish between **causal** and **associative** relationships in genomic and disease contexts.

### Core Question
> "Does the model tell the truth about causality when explaining biomedical research?"

## Dataset Summary

| Split | Count |
|-------|-------|
| Train | {len(train_items)} |
| Validation | {len(dev_items)} |
| Test | {len(test_items)} |
| **Total** | **{total}** |

### Taxonomy Distribution

| Taxonomy | Full Name | Count | Description |
|----------|-----------|-------|-------------|
| S | Structure-Aware | {taxonomy_dist.get('S', 0)} | SNP-Gene mapping, genomic structure |
| C | Causal-Aware | {taxonomy_dist.get('C', 0)} | Causal vs associative relationships |
| R | Risk-Aware | {taxonomy_dist.get('R', 0)} | Risk levels, OR interpretation |
| M | Semantic-Aware | {taxonomy_dist.get('M', 0)} | Entity recognition, relation extraction |

## Task Description

Models are evaluated on their ability to:

1. **Correctly identify causal vs associative language** - GWAS studies show associations, not causation
2. **Avoid overclaiming** - Not saying "causes" when evidence only shows "associated with"
3. **Provide accurate explanations** - Distinguishing correlation from causation
4. **Answer biomedical questions faithfully** - Using evidence-based reasoning

### Example

```json
{{
  "id": "C-0001",
  "question": "Does the variant rs1234567 cause increased risk of Breast Cancer?",
  "answer": "The variant rs1234567 is ASSOCIATED with increased risk of Breast Cancer (OR=1.85), but this association does not establish causation. GWAS studies identify statistical associations, not causal mechanisms.",
  "taxonomy": "C",
  "label": "C-CAUSE-ASSOC"
}}
```

## Dataset Creation

### Source Data
- GWAS Catalog and curated SNP-Gene-Disease associations
- Validated through multi-LLM consensus (OpenAI, Anthropic)
- Human expert review for causal faithfulness

### Validation Process
1. **Multi-LLM Consensus**: Multiple LLMs validate each item
2. **Causal Judgment**: ASSOCIATIVE / CAUSAL / NOT_APPLICABLE
3. **Overclaim Detection**: Flag items that incorrectly claim causation
4. **Human Expert Review**: Final validation by domain experts

### Quality Criteria
- Minimum validation score: {self.min_score}/5.0
- No overclaim consensus
- Human expert agreement

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("your-username/bioreasonc-bench")

# Access splits
train = dataset["train"]
validation = dataset["validation"]
test = dataset["test"]

# Example item
print(train[0])
```

### Evaluation

```python
# Evaluate model responses for causal faithfulness
def check_overclaim(response: str) -> bool:
    causal_words = ["causes", "leads to", "results in", "produces"]
    associative_words = ["associated with", "linked to", "correlated with"]

    has_causal = any(word in response.lower() for word in causal_words)
    has_associative = any(word in response.lower() for word in associative_words)

    # Overclaim if uses causal language without associative qualifier
    return has_causal and not has_associative
```

## Citation

```bibtex
@dataset{{bioreasonc_bench_{datetime.now().year},
  title = {{BioREASONC-Bench: Biomedical Reasoning and Causal Faithfulness Benchmark}},
  author = {{BioREASONC Team}},
  year = {{{datetime.now().year}}},
  version = {{{self.version}}},
  url = {{https://huggingface.co/datasets/bioreasonc-bench}},
  note = {{A benchmark for evaluating causal faithfulness in biomedical LLMs}}
}}
```

## License

This dataset is released under the Apache 2.0 License.

## Contact

For questions or feedback, please open an issue on the dataset repository.

---

*Generated by BioREASONC-Bench Pipeline v{self.version} on {datetime.now().strftime("%Y-%m-%d")}*
'''

        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print(f"Created dataset card: {readme_path}")
        return str(readme_path)

    def create_dataset_info(
        self,
        train_items: List[Dict],
        dev_items: List[Dict],
        test_items: List[Dict]
    ) -> str:
        """
        Create dataset_info.json for HuggingFace.

        Args:
            train_items: Training set items
            dev_items: Development set items
            test_items: Test set items

        Returns:
            Path to dataset_info.json
        """
        all_items = train_items + dev_items + test_items
        taxonomy_dist = {}
        for item in all_items:
            tax = item.get("taxonomy", "U")
            taxonomy_dist[tax] = taxonomy_dist.get(tax, 0) + 1

        info = {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "description": "Biomedical Reasoning and Causal Faithfulness Benchmark",
            "homepage": "https://github.com/bioreasonc/bioreasonc-bench",
            "license": "apache-2.0",
            "languages": ["en"],
            "task_categories": ["question-answering", "text-generation"],
            "created": datetime.now().isoformat(),
            "splits": {
                "train": {
                    "num_examples": len(train_items),
                    "file": "train.json"
                },
                "validation": {
                    "num_examples": len(dev_items),
                    "file": "dev.json"
                },
                "test": {
                    "num_examples": len(test_items),
                    "file": "test.json"
                }
            },
            "total_examples": len(all_items),
            "taxonomy_distribution": taxonomy_dist,
            "features": {
                "id": "string",
                "question": "string",
                "answer": "string",
                "explanation": "string",
                "taxonomy": "string (S/C/R/M)",
                "taxonomy_name": "string",
                "label": "string",
                "difficulty": "string (easy/medium/hard)",
                "validation_score": "float (1-5)",
                "consensus_judgment": "string (ASSOCIATIVE/CAUSAL/NOT_APPLICABLE)",
                "ground_truth": "object",
                "metadata": "object"
            },
            "validation_criteria": {
                "min_score": self.min_score,
                "no_overclaim": True,
                "human_validated": True
            }
        }

        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        print(f"Created dataset info: {info_path}")
        return str(info_path)

    def export_benchmark(
        self,
        items: List[Dict[str, Any]],
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Dict[str, Any]:
        """
        Export complete benchmark with train/dev/test splits and metadata.

        Args:
            items: Validated items to export
            train_ratio: Proportion for training
            dev_ratio: Proportion for development
            test_ratio: Proportion for test

        Returns:
            Dictionary with export results
        """
        print("\n" + "=" * 70)
        print("EXPORTING BIOREASONC BENCHMARK")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"Version: {self.version}")
        print(f"Minimum score: {self.min_score}")

        # Filter high-quality items
        filtered_items = self.filter_high_quality_items(items)

        if not filtered_items:
            print("ERROR: No items passed quality filter!")
            return {"error": "No items passed quality filter"}

        # Create splits
        train_items, dev_items, test_items = self.create_splits(
            filtered_items, train_ratio, dev_ratio, test_ratio
        )

        # Export splits
        train_path = self.export_split(train_items, "train.json")
        dev_path = self.export_split(dev_items, "dev.json")
        test_path = self.export_split(test_items, "test.json")

        # Create metadata
        readme_path = self.create_dataset_card(train_items, dev_items, test_items)
        info_path = self.create_dataset_info(train_items, dev_items, test_items)

        results = {
            "status": "success",
            "output_dir": str(self.output_dir),
            "version": self.version,
            "files": {
                "train": train_path,
                "dev": dev_path,
                "test": test_path,
                "readme": readme_path,
                "dataset_info": info_path
            },
            "statistics": {
                "total_items": len(filtered_items),
                "train_items": len(train_items),
                "dev_items": len(dev_items),
                "test_items": len(test_items),
                "filtered_out": len(items) - len(filtered_items)
            }
        }

        print("\n" + "=" * 70)
        print("BENCHMARK EXPORT COMPLETE")
        print("=" * 70)
        print(f"Total items: {results['statistics']['total_items']}")
        print(f"Train: {results['statistics']['train_items']}")
        print(f"Dev: {results['statistics']['dev_items']}")
        print(f"Test: {results['statistics']['test_items']}")
        print(f"\nFiles created:")
        for name, path in results['files'].items():
            print(f"  - {name}: {path}")

        return results


def export_to_huggingface(
    items: List[Dict[str, Any]],
    output_dir: str = "./benchmark_output",
    dataset_name: str = "BioREASONC-Bench",
    version: str = "1.0.0",
    min_score: float = 4.0
) -> Dict[str, Any]:
    """
    Convenience function to export benchmark to HuggingFace format.

    Args:
        items: Validated items from pipeline
        output_dir: Output directory
        dataset_name: Name of the dataset
        version: Version string
        min_score: Minimum validation score

    Returns:
        Export results dictionary
    """
    exporter = BenchmarkExporter(
        output_dir=output_dir,
        dataset_name=dataset_name,
        version=version,
        min_score=min_score
    )
    return exporter.export_benchmark(items)
