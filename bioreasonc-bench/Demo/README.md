# BioREASONIC-Bench Reproduction Guide

This directory contains everything needed to reproduce the BioREASONIC-Bench question generation pipeline.

## Overview

BioREASONIC-Bench generates expert-level biomedical questions from GWAS data with four taxonomy categories:

| Taxonomy | Name | Description |
|----------|------|-------------|
| **S** | Structure-Aware | SNP-Gene mapping, chromosomal location |
| **C** | Causal-Aware | Causal vs associative distinction (critical) |
| **R** | Risk-Aware | Odds ratio interpretation, risk classification |
| **M** | Mechanism-Aware | Entity recognition, relation extraction |

## Directory Structure

```
Reproduction/
├── README.md              # This file
├── data/
│   └── sample_gwas.csv    # Sample GWAS data (5 genes)
├── output/
│   └── generated_questions.json  # Generated output
└── scripts/
    └── test_pipeline.py   # Test script
```

## Prerequisites

1. Python 3.8+
2. Required packages:
   ```bash
   pip install pandas
   ```

## Step-by-Step Reproduction

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd bioreasonc-bench
```

### Step 2: Verify Sample Data

The sample GWAS data is in `Reproduction/data/sample_gwas.csv`:

| rsid | gene | chromosome | OR | P-Value |
|------|------|------------|-----|---------|
| rs7903146 | TCF7L2 | 10 | 1.37 | 2.0e-50 |
| rs1801282 | PPARG | 3 | 1.14 | 5.2e-12 |
| rs5219 | KCNJ11 | 11 | 1.15 | 3.1e-15 |
| rs4402960 | IGF2BP2 | 3 | 1.17 | 8.6e-16 |
| rs1111875 | HHEX | 10 | 1.13 | 5.7e-10 |

### Step 3: Run the Pipeline Test

```bash
cd bioreasonc-bench
python Reproduction/scripts/test_pipeline.py
```

### Step 4: View Generated Output

```bash
# View summary
cat Reproduction/output/generated_questions.json | python -m json.tool | head -100

# Or use jq for better formatting
jq '.statistics' Reproduction/output/generated_questions.json
```

## Expected Output

Running the pipeline with 5 sample genes generates **75 questions**:

```
Total items: 75

By Taxonomy:
  S (Structure-Aware): 20
  C (Causal-Aware): 15
  R (Risk-Aware): 20
  M (Mechanism-Aware): 20

By Difficulty:
  easy: 35
  medium: 25
  hard: 15
```

## Example Generated Questions

### S (Structure) - Easy
**Question:** Which gene is the variant rs7903146 located in or associated with?
**Answer:** TCF7L2

### C (Causal) - Hard
**Question:** Is the relationship between TCF7L2 and Type 2 Diabetes causal or associative based on GWAS evidence?

**Answer:**
> No—and this distinction is crucial for interpreting genetic studies correctly.
>
> GWAS tells us that people carrying TCF7L2 variants have higher Type 2 Diabetes rates. The association is robust: OR of 1.37, replicated across studies. But association is not causation.
>
> Consider the alternatives:
> - **Confounding:** Perhaps TCF7L2 variants are more common in populations with lifestyle factors that independently increase Type 2 Diabetes risk.
> - **Linkage disequilibrium:** The associated variant might simply tag the true causal variant nearby.
> - **Reverse causation:** Less likely for germline variants, but disease-related metabolic changes could theoretically affect TCF7L2 regulation.
>
> TCF7L2 encodes a transcription factor in the Wnt signaling pathway that regulates beta cell development and insulin secretion. The biological pathway to Type 2 Diabetes involves insulin resistance in peripheral tissues combined with beta cell dysfunction...
>
> The correct statement is: "TCF7L2 variants are associated with increased Type 2 Diabetes risk." Claiming causation requires MR, functional studies, or ideally both.

### R (Risk) - Medium
**Question:** What is the risk level conferred by rs7903146 for Type 2 Diabetes given OR=1.37?

**Answer:**
> Yes, TCF7L2 may contribute to disease risk with moderate evidence.
>
> The statistical evidence is moderate. TCF7L2 variants show OR of 1.37. This is a moderate effect, somewhat larger than typical GWAS findings (OR 1.05-1.20).
>
> Why does TCF7L2 affect Type 2 Diabetes risk? TCF7L2 encodes a transcription factor in the Wnt signaling pathway that regulates beta cell development and insulin secretion...
>
> For individual risk interpretation: an OR of 1.37 means carriers have 1.37x the odds of developing Type 2 Diabetes compared to non-carriers.

### M (Mechanism) - Easy
**Question:** Identify the gene symbol mentioned in: 'TCF7L2 variant rs7903146 is associated with Type 2 Diabetes susceptibility.'
**Answer:** TCF7L2

## Configuration Options

Edit `src/bioreasonc_creator/generator.py` to modify:

```python
# Line 279: Enable/disable CoT reasoning for Causal taxonomy
USE_COT_ANSWERS = True

# Line 282: Enable/disable expert-style biological explanations
USE_EXPERT_PROMPTS = True
```

## Using Your Own Data

1. Prepare a CSV file with these columns:
   - `rsid` or `SNP`: SNP identifier (e.g., "rs7903146")
   - `gene` or `Gene`: Gene symbol (e.g., "TCF7L2")
   - `chromosome` or `chr`: Chromosome number
   - `OR` or `odds_ratio`: Odds ratio
   - `P-Value` or `p_value`: P-value

2. Modify the test script or use the generator directly:

```python
from src.bioreasonc_creator.generator import QuestionGenerator

generator = QuestionGenerator()
items = generator.generate_from_csv(
    "path/to/your/data.csv",
    disease="Your Disease Name",
    max_per_template=50
)

# Get statistics
stats = generator.get_statistics(items)
print(f"Generated {stats['total']} items")
```

## Key Features

1. **Expert-Style Answers**: C and R taxonomies generate biologically-informed explanations, not mechanical templates
2. **Causal Faithfulness**: C taxonomy answers correctly distinguish association from causation
3. **Biological Context**: Answers include gene function and disease mechanism
4. **Ground Truth**: Each item has deterministic ground truth for LLM evaluation

## Troubleshooting

### Import Error
If you see `ImportError: cannot import name 'KG_QUESTION_TEMPLATES'`:
```bash
# The __init__.py may have stale imports. Run from project root:
python -c "from src.bioreasonc_creator.generator import QuestionGenerator; print('OK')"
```

### Missing Pandas
```bash
pip install pandas
```

## Citation

If you use BioREASONIC-Bench, please cite:

```bibtex
@article{bioreasonicbench2024,
  title={BioREASONIC-Bench: A Benchmark for Explainable Causal Reasoning in Biomedical Data},
  author={Alsaedi, Sakhaa},
  year={2024}
}
```

## Contact

For questions or issues:
- Email: sakhaa.alsaedi@kaust.edu.sa
- GitHub Issues: [repository-url]/issues
