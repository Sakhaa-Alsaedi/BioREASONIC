# BioREASONIC-Bench

**A Benchmark for Explainable Causal Reasoning in Biomedical Data**

BioREASONIC-Bench evaluates whether Large Language Models (LLMs) can correctly distinguish between **causal** and **associative** relationships when explaining biomedical research findings.

---

## Core Question

> *"Does the model tell the truth about causality when explaining biomedical research?"*

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Taxonomy Structure](#taxonomy-structure)
- [Pipeline Architecture](#pipeline-architecture)
- [Evaluation Metrics](#evaluation-metrics)
- [Data Sources](#data-sources)
- [Usage Examples](#usage-examples)
- [Reproduction Guide](#reproduction-guide)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contact](#contact)

---

## Overview

BioREASONIC-Bench addresses a critical gap in biomedical AI: the tendency of LLMs to **overclaim causation** when the evidence only supports association. This is particularly dangerous in healthcare contexts where incorrect causal claims can lead to:

- Inappropriate clinical decisions
- Misguided drug development
- Public health misinformation

### What Makes This Benchmark Different?

| Feature | Traditional Benchmarks | BioREASONIC-Bench |
|---------|----------------------|-------------------|
| **Focus** | Factual recall | Causal reasoning |
| **Answers** | Short/factual | Expert-style biological explanations |
| **Evaluation** | Accuracy only | Multi-dimensional (CARES score) |
| **Ground Truth** | Static | Evidence-based with confidence levels |

---

## Key Features

- **Four Taxonomy Categories**: Structure (S), Causal (C), Risk (R), Mechanism (M)
- **Expert-Style Answers**: Biological pathway explanations, not mechanical templates
- **CARES Evaluation**: Causal-Aware Reasoning Evaluation Score
- **GRASS Scoring**: Gene Risk Association Scoring System
- **Chain-of-Thought**: Built-in CoT reasoning for causal questions
- **Deterministic Ground Truth**: Every answer traceable to source evidence

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Sakhaa-Alsaedi/BioREASONIC.git
cd BioREASONIC/bioreasonc-bench

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib (for visualizations)

---

## Quick Start

```python
from src.bioreasonc_creator.generator import QuestionGenerator
import pandas as pd

# Initialize generator
generator = QuestionGenerator()

# Load your GWAS data
df = pd.DataFrame({
    'rsid': ['rs7903146', 'rs1801282'],
    'gene': ['TCF7L2', 'PPARG'],
    'chromosome': ['10', '3'],
    'OR': [1.37, 1.14],
    'P-Value': [2e-50, 5.2e-12]
})

# Generate questions
items = generator.generate_from_dataframe(df, disease="Type 2 Diabetes")

# View statistics
stats = generator.get_statistics(items)
print(f"Generated {stats['total']} questions")
print(f"By taxonomy: {stats['by_taxonomy']}")
```

---

## Taxonomy Structure

### S - Structure-Aware (Genomic Architecture)

Tests understanding of SNP-gene mapping and chromosomal organization.

| Template | Question Type | Difficulty |
|----------|--------------|------------|
| S-GENE-MAP | Which gene contains variant X? | Easy |
| S-SNP-GENE | Is variant X in gene Y? | Easy |
| S-CHROM-LOC | What chromosome is gene X on? | Easy |

**Example:**
```
Q: Which gene is the variant rs7903146 located in?
A: TCF7L2
```

---

### C - Causal-Aware (Critical for Benchmark)

Tests ability to distinguish GWAS association from causation. **This is the most important taxonomy.**

| Template | Question Type | Difficulty |
|----------|--------------|------------|
| C-CAUSAL-VS-ASSOC | Is relationship causal or associative? | Hard |
| C-MR-EVIDENCE | What evidence strengthens causal claims? | Hard |

**Example:**
```
Q: Is the relationship between TCF7L2 and Type 2 Diabetes causal or
   associative based on GWAS evidence?

A: No—and this distinction is crucial for interpreting genetic studies
   correctly.

   GWAS tells us that people carrying TCF7L2 variants have higher Type 2
   Diabetes rates. The association is robust: OR of 1.37, replicated across
   studies. But association is not causation.

   Consider the alternatives:

   **Confounding:** Perhaps TCF7L2 variants are more common in populations
   with lifestyle factors that independently increase Type 2 Diabetes risk.

   **Linkage disequilibrium:** The associated variant might simply tag the
   true causal variant nearby.

   **Reverse causation:** Less likely for germline variants, but disease-
   related metabolic changes could theoretically affect TCF7L2 regulation.

   TCF7L2 encodes a transcription factor in the Wnt signaling pathway that
   regulates beta cell proliferation, survival, and incretin (GLP-1)
   responses. The biological pathway to Type 2 Diabetes involves insulin
   resistance in peripheral tissues and progressive beta cell dysfunction.

   The correct statement is: "TCF7L2 variants are ASSOCIATED with increased
   Type 2 Diabetes risk." Claiming causation requires MR, functional studies,
   or ideally both.
```

---

### R - Risk-Aware (Clinical Interpretation)

Tests interpretation of odds ratios and genetic risk.

| Template | Question Type | Difficulty |
|----------|--------------|------------|
| R-RISK-LEVEL | What is the risk level for OR=X? | Medium |
| R-OR-INTERPRET | How does carrier risk compare? | Medium |
| R-PVALUE-SIG | Is association genome-wide significant? | Easy |

**Example:**
```
Q: What is the risk level conferred by rs7903146 for Type 2 Diabetes
   given OR=1.37?

A: TCF7L2 is a well-supported risk factor for Type 2 Diabetes with
   moderate evidence.

   The statistical evidence is moderate. TCF7L2 variants show OR of 1.37,
   which is somewhat larger than typical GWAS findings (OR 1.05-1.20) but
   not unusual for well-established disease genes.

   Why does TCF7L2 affect Type 2 Diabetes risk? TCF7L2 encodes a
   transcription factor in the Wnt signaling pathway that regulates beta
   cell proliferation, survival, and incretin (GLP-1) responses.

   For individual risk interpretation: an OR of 1.37 means carriers have
   1.37x the odds of developing Type 2 Diabetes compared to non-carriers.
   However, OR describes relative risk—the absolute risk increase depends
   on baseline population risk.
```

---

### M - Mechanism-Aware (Biological Understanding)

Tests entity recognition and relationship extraction.

| Template | Question Type | Difficulty |
|----------|--------------|------------|
| M-ENTITY-RECOGNIZE | Extract gene/SNP from text | Easy |
| M-REL-EXTRACT | Extract biomedical relationships | Medium |

**Example:**
```
Q: Identify the gene symbol mentioned in: 'TCF7L2 variant rs7903146 is
   associated with Type 2 Diabetes susceptibility.'
A: TCF7L2
```

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      BioREASONIC-Bench Pipeline                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌────────────┐    ┌───────────┐    ┌───────────┐    ┌─────────────┐    │
│   │ GWAS       │───▶│ Generator │───▶│ Validator │───▶│ Benchmark   │    │
│   │ Data /GRAP │    │           │    │           │   │     Export   │    │
│   └────────────┘    └───────────┘    └───────────┘    └─────────────┘    │
│        │               │                 │                  │            │
│        │               ▼                 ▼                  ▼            │
│        │         ┌───────────┐    ┌───────────┐    ┌─────────────┐       │
│        │         │ Expert    │    │ Multi-LLM │    │ HuggingFace │       │
│        │         │ Prompts   │    │ QA Check  │    │ Format      │       │
│        │         └───────────┘    └───────────┘    └─────────────┘       │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐    │
│   │                       Data Sources                              │    │
│   │  • CAUSALdb2 v2.1: 66,057 gene-disease pairs                    │    │
│   │  • Evidence levels: very_strong, strong, moderate, weak         │    │
│   │  • MR (Mendelian Randomization) causal evidence                 │    │
│   └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Metrics

### CARES Score (Causal-Aware Reasoning Evaluation Score)

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 0.25 | Factual accuracy of the answer |
| Causal Faithfulness | 0.30 | Correct causal vs associative language |
| Completeness | 0.20 | Coverage of required information |
| Coherence | 0.15 | Logical flow and consistency |
| Uncertainty | 0.10 | Appropriate hedging and limitations |

**Score Scale (0-5):**
| Score | Meaning |
|-------|---------|
| 5 | Fully correct, semantically equivalent |
| 4 | Mostly correct, minor imprecisions |
| 3 | Partially correct, missing >20% details |
| 2 | Safe abstention with uncertainty |
| 1 | Partial hallucination |
| 0 | Complete hallucination |

**Domain-specific α values:**

| Application | HR_max | α |
|-------------|--------|---|
| Drug interaction | 5% | 13.9 |
| Clinical decision | 10% | 6.9 |
| Literature summary | 15% | 4.6 |
| Research exploration | 25% | 2.8 |


## Data Sources

| Source | Description | Size |
|--------|-------------|------|
| CAUSALdb2 v2.1 | Gene-disease associations with MR evidence | 66,057 pairs |
| GWAS Catalog | Genome-wide association studies | Variable |
| OpenTargets | Drug targets and disease associations | Supplementary |
| STRING | Protein-protein interactions | Supplementary |

---

## Usage Examples

### Generate Benchmark from CSV

```python
from src.bioreasonc_creator.generator import QuestionGenerator

generator = QuestionGenerator()
items = generator.generate_from_csv(
    "data/gwas_results.csv",
    disease="Alzheimer's Disease",
    max_per_template=50
)

# Export to JSON
import json
with open("benchmark.json", "w") as f:
    json.dump([item.to_dict() for item in items], f, indent=2)
```

### Evaluate LLM Responses

```python
from eval.cares import CARESEvaluator

evaluator = CARESEvaluator()
results = evaluator.evaluate(
    predictions=model_responses,
    references=ground_truth,
    taxonomy="C"  # Causal taxonomy
)

print(f"CARES Score: {results['cares_score']:.3f}")
print(f"Overclaim Rate: {results['overclaim_rate']:.2%}")
```

### Run Complete Evaluation

```bash
# Create sample responses for testing
python eval/evaluate.py create-sample \
    -b outputs/bioreasonc_bench_v1.jsonl \
    -o sample_responses.jsonl

# Evaluate model responses
python eval/evaluate.py evaluate \
    -b outputs/bioreasonc_bench_v1.jsonl \
    -r model_responses.jsonl \
    -o evaluation_report.json \
    -d clinical_decision
```

---

## Reproduction Guide

A complete reproduction guide is available in the `Reproduction/` directory:

```bash
cd Reproduction
python scripts/test_pipeline.py
```

**Expected Output:**
```
Total items: 75
By Taxonomy:
  S (Structure-Aware): 20
  C (Causal-Aware): 15
  R (Risk-Aware): 20
  M (Mechanism-Aware): 20
```

See `Reproduction/README.md` for detailed step-by-step instructions.

---

## Project Structure

```
bioreasonc-bench/
├── src/
│   ├── bioreasonc_creator/          # Main question generation
│   │   ├── generator.py             # Question generator with expert prompts
│   │   ├── prompts.py               # Expert prompts and biological helpers
│   │   ├── validator.py             # Multi-LLM validation
│   │   ├── pipeline.py              # Pipeline orchestrator
│   │   ├── kg_ingest.py             # Knowledge graph ingestion
│   │   ├── human_exp_evaluator.py   # Human expert evaluation
│   │   └── benchmark_exporter.py    # Export to HuggingFace format
│   ├── reacTax/                     # Taxonomy modules
│   │   ├── structure.py             # S taxonomy
│   │   ├── causal/                  # C taxonomy (critical)
│   │   │   ├── discovery.py         # Causal discovery algorithms
│   │   │   ├── inference.py         # Causal inference
│   │   │   ├── mr.py                # Mendelian Randomization
│   │   │   └── reasoning.py         # Causal reasoning
│   │   ├── risk.py                  # R taxonomy
│   │   └── semantic.py              # M taxonomy
│   └── Causal_KG/                   # Knowledge graph modules
│       ├── databases/               # Database connectors
│       │   ├── epigraphdb.py        # EpiGraphDB integration
│       │   ├── open_targets.py      # OpenTargets integration
│       │   └── string_db.py         # STRING-DB integration
│       └── kg_loader.py             # KG loading utilities
├── eval/
│   ├── cares.py                     # CARES evaluation metric
│   ├── grass.py                     # GRASS gene scoring
│   ├── metrics.py                   # Additional metrics
│   └── evaluate.py                  # Evaluation runner
├── config/
│   ├── config.py                    # Configuration settings
│   └── apis/                        # API configurations
│       └── llm_judge.py             # LLM judge settings
├── visuals/
│   ├── plots.py                     # Visualization functions
│   ├── dashboard.py                 # Interactive dashboard
│   └── summary.py                   # Summary generation
├── docs/
│   ├── methodology_detailed.tex     # Full methodology
│   ├── expert_generation_prompts.tex# Expert prompt templates
│   ├── ideal_expert_answers.tex     # Example expert answers
│   └── equations.tex                # Mathematical formulations
├── Reproduction/                    # Reproduction guide
│   ├── README.md                    # Step-by-step instructions
│   ├── data/sample_gwas.csv         # Sample data (5 genes)
│   ├── scripts/test_pipeline.py     # Test script
│   └── output/                      # Generated output
├── bioreasonc_output/               # Generated benchmark files
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── requirements.txt
└── README.md
```

---

## Configuration

Edit `src/bioreasonc_creator/generator.py`:

```python
# Enable Chain-of-Thought reasoning for Causal taxonomy
USE_COT_ANSWERS = True

# Enable expert-style biological explanations
USE_EXPERT_PROMPTS = True
```

---

## Output Format

```json
{
  "id": "C-0001",
  "taxonomy": "C",
  "label": "C-CAUSAL-VS-ASSOC",
  "template_id": "C-CAUSAL-VS-ASSOC-01",
  "question": "Is the relationship between TCF7L2 and Type 2 Diabetes causal or associative?",
  "answer": "No—and this distinction is crucial...[expert explanation]",
  "answer_type": "explanation",
  "entities": {
    "gene": "TCF7L2",
    "disease": "Type 2 Diabetes",
    "snp": "rs7903146",
    "odds_ratio": "1.37"
  },
  "ground_truth": {
    "answer": "...",
    "answer_normalized": "...",
    "confidence": 1.0
  },
  "difficulty": "hard",
  "source_data": {
    "expert_prompts_used": true,
    "original_template_answer": "..."
  }
}
```
