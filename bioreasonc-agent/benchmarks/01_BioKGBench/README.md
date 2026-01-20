# 01_BioKGBench

## Biomedical Knowledge Graph Question Answering Benchmark

---

## Overview

BioKGBench is an external benchmark for evaluating LLMs on biomedical KGQA tasks. The dataset tests models' ability to:

1. **Understand** natural language biomedical questions
2. **Generate** accurate Cypher queries for Neo4j
3. **Retrieve** correct answers through knowledge graph traversal

---

## Dataset

**Source**: [HuggingFace - AutoLab-Westlake/BioKGBench-Dataset](https://huggingface.co/datasets/AutoLab-Westlake/BioKGBench-Dataset)

| Split | Questions |
|-------|-----------|
| Test  | 638       |
| Dev   | 60        |

### Question Types

| Type | Count | Description |
|------|-------|-------------|
| one-hop | 355 (55.6%) | Single relationship traversal |
| multi-hop | 184 (28.8%) | Multiple relationship traversals |
| conjunction | 99 (15.5%) | Questions requiring AND/OR logic |

---

## Directory Structure

```
01_BioKGBench/
├── README.md
├── config/
│   ├── config.yaml              # Main configuration
│   └── config.local.yaml        # Local config with API keys
├── src/
│   ├── kg_qa_system_v2.py       # Main KGQA system
│   ├── llm_client.py            # LLM API clients
│   ├── llm_cypher_generator.py  # Cypher query generation
│   ├── question_parser.py       # Question parsing
│   ├── cypher_generator.py      # Rule-based Cypher generation
│   ├── evaluate_kgqa.py         # Evaluation script
│   └── run_all_models_eval.py   # Multi-model evaluation
├── results/
│   ├── eval_results_*.json      # Per-model results
│   ├── biokgbench_*.csv         # Summary tables
│   └── biokgbench_*.png         # Visualizations
└── tutorials/
    └── BioKGBench_Evaluation_Tutorial.ipynb  # ISMB 2026 Tutorial
```

---

## Evaluation Results (8 LLMs)

| Model | F1 (%) | EM (%) | Exec (%) | Hits@1 (%) | Hits@5 (%) | MRR |
|-------|--------|--------|----------|------------|------------|-----|
| gpt-4.1 | 65.4 | 76.2 | 99.5 | 79.1 | 81.1 | 0.80 |
| gpt-4.1-mini | 62.4 | 75.2 | 99.8 | 76.9 | 79.1 | 0.78 |
| gpt-4o | 56.4 | 69.1 | 90.6 | 74.2 | 77.8 | 0.75 |
| claude-3-haiku | 45.5 | 62.1 | 78.5 | 75.3 | 80.0 | 0.77 |
| gpt-4o-mini | 39.4 | 61.8 | 85.6 | 66.6 | 70.7 | 0.68 |
| deepseek-v3 | 35.5 | 54.9 | 78.5 | 75.2 | 78.5 | 0.76 |
| qwen-2.5-7b | 27.1 | 48.7 | 74.0 | 59.9 | 67.2 | 0.62 |
| llama-3.1-8b | 27.0 | 48.7 | 72.7 | 64.8 | 75.0 | 0.68 |

---

## Metrics

| Metric | Description |
|--------|-------------|
| **F1** | Harmonic mean of precision and recall |
| **EM** | Exact Match - any prediction matches gold |
| **Executability** | % of queries that execute without error |
| **Coverage** | % of questions with non-empty answers |
| **Hits@K** | Correct answer in top-K results |
| **MRR** | Mean Reciprocal Rank |

---

## Usage

### Run Evaluation

```bash
# Single model
python src/run_all_models_eval.py --model gpt-4o --dataset test

# All models
python src/run_all_models_eval.py --all --dataset test
```

### Configuration

Edit `config/config.local.yaml`:

```yaml
neo4j:
  uri: "bolt://10.73.107.108:7687"
  user: "neo4j"
  password: "password123"

llm:
  provider: "gpt-4o"
  temperature: 0.0
```

---

## Tutorial

See `tutorials/BioKGBench_Evaluation_Tutorial.ipynb` for the ISMB 2026 tutorial notebook.

---

## Citation

```bibtex
@dataset{biokgbench,
  title={BioKGBench: A Benchmark for Biomedical KGQA},
  author={AutoLab-Westlake},
  year={2024},
  publisher={HuggingFace}
}
```
