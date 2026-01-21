# CARES Workflow

Complete workflow for generating CARES (Causal-Aware Reasoning Evaluation Score) taxonomy scores and visualizations.

## Overview

This workflow processes LLM evaluation results to compute CARES scores across:
- **8 Models**: GPT-4o, GPT-4o-Mini, GPT-4.1, GPT-4.1-Mini, Claude-3-Haiku, Llama-3.1-8B, DeepSeek-V3, Qwen-2.5-7B
- **4 Prompting Strategies**: Zero-shot, Few-shot, CoT, Structured-CoT
- **4 SCRM Taxonomy Categories**: Structure (S), Causal (C), Risk (R), Mechanism (M)

## Directory Structure

```
CARES/
├── README.md                 # This file
├── run_all.py               # Main script to run complete workflow
├── data/
│   ├── input/
│   │   └── all_results.csv   # Raw LLM predictions (INPUT)
│   └── processed/
│       ├── cares_taxonomy_scores.csv    # Aggregated CARES scores
│       └── metrics_summary_final.csv    # Detailed metrics
├── src/
│   ├── classification_benchmark_eval.py # Full evaluation pipeline
│   ├── generate_cares_taxonomy_scores.py # Process raw -> taxonomy scores
│   ├── generate_cares_radar.py          # Radar plot generation
│   ├── generate_cares_bar.py            # Bar chart generation
│   ├── generate_cares_heatmap.py        # Heatmap generation
│   ├── generate_cares_line.py           # Line chart generation
│   ├── generate_cares_grouped_bar.py    # Grouped bar generation
│   └── generate_cares_models_comparison.py # Model comparison plots
├── plots/
│   ├── cares_taxonomy_radar_pastel.png  # Main radar plot
│   ├── cares_bar_chart.png
│   ├── cares_heatmap.png
│   ├── cares_line_chart.png
│   ├── cares_grouped_bar.png
│   ├── cares_models_radar.png
│   └── cares_models_variance.png
└── results/
    └── (generated results will be saved here)
```

## Quick Start

### Run Complete Workflow
```bash
cd /ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/Task/CARES
python run_all.py
```

### Run Individual Steps

1. **Generate taxonomy scores from raw data:**
```bash
python src/generate_cares_taxonomy_scores.py
```

2. **Generate radar plot:**
```bash
python src/generate_cares_radar.py
```

3. **Generate other plots:**
```bash
python src/generate_cares_bar.py
python src/generate_cares_heatmap.py
python src/generate_cares_line.py
python src/generate_cares_grouped_bar.py
python src/generate_cares_models_comparison.py
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT: data/input/all_results.csv                          │
│  - Raw LLM predictions                                       │
│  - Columns: model, strategy, source, taxonomy, correct, etc.│
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PROCESS: src/generate_cares_taxonomy_scores.py             │
│  - Aggregates scores by model, strategy, taxonomy           │
│  - Computes average SCRM scores                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: data/processed/cares_taxonomy_scores.csv           │
│  - Columns: model, strategy, C_Causal, M_Mechanism,         │
│             R_Risk, S_Structure, Avg_SCRM                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PLOTS: src/generate_cares_*.py                             │
│  - Radar, Bar, Heatmap, Line, Grouped Bar charts            │
│  - Output to plots/ directory                                │
└─────────────────────────────────────────────────────────────┘
```

## CARES Score Formula

```
CARES = [Σ w̃_k · CARES-k] × √(Φ(HR) × (1 - ECE))

Where:
- w̃_k = normalized category weights (S=0.25, C=0.30, R=0.25, M=0.20)
- CARES-k = average score for category k
- Φ(HR) = exp(-α · HR) = hallucination penalty (α=6.9 for clinical)
- ECE = expected calibration error
```

## Key Results

### Best Performance by Strategy (Average across all models)

| Strategy | C (Causal) | R (Risk) | M (Mechanism) | S (Structure) | Avg |
|----------|------------|----------|---------------|---------------|-----|
| Zero-shot | 0.31 | 0.32 | 0.37 | 0.40 | 0.35 |
| Few-shot | 0.29 | 0.24 | 0.28 | 0.26 | 0.27 |
| CoT | 0.35 | 0.38 | 0.40 | 0.41 | 0.39 |
| **Structured-CoT** | **0.63** | **0.60** | **0.65** | **0.65** | **0.63** |

### Best Model-Strategy Combination
- **GPT-4o-Mini + Structured-CoT**: Avg_SCRM = 0.715

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn
```

## Author

BioREASONIC Project
