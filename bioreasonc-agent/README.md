# BioREASONIC Reviewer Package

This package allows you to verify the results of the "BioREASONIC: Logic-Enhanced Biomedical Knowledge Graph Question Answering" paper.

## Contents
- `Bioreasonic.ipynb`: Main evaluation notebook.
- `run_kgqa_evaluation.py`: Automated evaluation script.
- `benchmarks/`: Data for BioKGBench and BioResonKGBench.

## Configuration Setup (Required)
Before running, you must add your **LLM API keys**. The Database connection is pre-configured.

1. **LLM Keys:** Edit `benchmarks/01_BioKGBench/config/config.local.yaml` and `benchmarks/02_BioResonKGBench/config/config.local.yaml`.
   - Add your OpenAI/Claude/Together API keys.
   - Replace "YOUR_..._KEY_HERE".

2. **Database:** (Pre-configured)
   - The configs are already set to connect to the hosted Neo4j instance at `34.31.148.159`. No action needed.

## How to Run

You have two options to run the evaluation:

### Option 1: Python Script (Automated) - RECOMMENDED
Run the evaluation directly from the terminal. This is the fastest and most robust way to verify results for batch evaluations.

```bash
# Evaluate both datasets (BioKGBench + BioResonKGBench)
python3 run_kgqa_evaluation.py --dataset both --samples 10

# Evaluate only BioResonKGBench
python3 run_kgqa_evaluation.py --dataset bioresonkgbench --samples 50
```

### Option 2: Jupyter Notebook (Visual)
Use this option if you want to inspect the code, view inline visualizations, or run cells interactively.

1. Open `Bioreasonic.ipynb` in Jupyter Lab or VS Code.
2. Ensure you have the required dependencies installed (`pip install -r requirements.txt`).
3. Run all cells.
4. The notebook will:
   - Load the datasets.
   - Run the 4 approaches (Direct, Cypher, ReAct-COT, Multi-Agent).
   - Display visualizations (Radar Plots, Heatmaps).
   - Save detailed results to a CSV file.

## Output
Both methods produce:
1. **CSV File:** `detailed_results_<dataset>_<timestamp>.csv` containing 30+ metrics (EM, BLEU, ROUGE, Latency, etc.) per sample.
2. **Visualizations:** PNG files for radar charts, heatmaps, and impact analysis.
3. **Logs:** Execution progress and verification logs.

## Requirements
- Python 3.10+
- Dependencies: `pip install -r requirements.txt`
- Neo4j Instance (configured in `benchmarks/config/config.local.yaml`)
