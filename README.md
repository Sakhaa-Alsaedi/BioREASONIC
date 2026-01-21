<div align="center">

<img width="5985" height="1072" alt="Image" src="https://github.com/user-attachments/assets/64a268ca-9fe2-4dda-a012-e6430a559704" />

# BioREASONIC Agnatic Reasoning System

### A Causal-Oriented GraphRAG System for Multi-Aware Biomedical Reasoning

![Image](https://github.com/user-attachments/assets/f7456ef1-818e-49fb-92be-e247b708cc21)

[![Paper](https://img.shields.io/badge/Paper-ISMB%202026-blue.svg)](https://doi.org/XXXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1.svg)](https://neo4j.com/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/Sakhaa-Alsaedi/BioREASONIC/graphs/commit-activity)

<!--
[Paper](#citation) | [Web Demo](https://bioreasonicexplainer.kaust.edu.sa) | [Benchmark](#bioreasonicbench) | [Installation](#installation)
-->

</div>

---

## Overview

**BioREASONIC** is an agentic causal-oriented GraphRAG system designed to perform multi-aware biomedical reasoning. It bridges the gap between large language models (LLMs) and trustworthy biomedical AI by integrating causal knowledge graphs, genetic risk scoring, and explainable reasoning.




> **Figure 1:** Overview of the BioREASONIC agentic causal-oriented GraphRAG system showing (A) Genetic Risk Aggregation Scoring System (GRASS), (B) Causal Biomedical Agentic Reasoning with latency-aware dual-path design and causal GraphRAG inference, and (C) BioREASONIC Explainer interface with conversational and omics-enrichment modes.

---

## Key Features

| Feature | Repository | Description |
|--------|------------|-------------|
| **A) GRASS** | [GRASS](https://github.com/Sakhaa-Alsaedi/BioREASONIC/tree/main/GRASS) | Genetic Risk Aggregation Scoring System for quantifying gene-level disease risk |
| **BioREASONIC Agent** | [bioreasonc-agent](https://github.com/Sakhaa-Alsaedi/BioREASONIC/tree/main/bioreasonc-agent)| Single-agent causal GraphRAG system, achieving 3–20× higher efficiency than multi-agent baselines |
| **BioREASONIC-Bench** |[Benchmarks](https://github.com/Sakhaa-Alsaedi/BioREASONIC/tree/main/bioreasonc-agent/benchmarks/02_BioResonKGBench)| Multi-aware biomedical reasoning benchmark based on the S–R–C–M taxonomy |
| **Experiments** | [Notebook](https://github.com/Sakhaa-Alsaedi/BioREASONIC/tree/main/notebooks) | Executable notebooks reproducing all experiments reported in the manuscript |
<!-- | **Web Interface** | — | Interactive BioREASONIC Explainer for causal risk graph analysis | -->



---

## System Architecture

### Causal-Risk Knowledge Graph Schema

The graph comprises **1,006,535 nodes** across 13 entity types:
- 521,419 Causal SNPs
- 43,842 Risk Genes  
- 11,698 Diseases
- 240,877 Proteins

And **20,398,953 relationships** across 13 relationship categories.

---


## Installation

### Prerequisites

- Python 3.9+
- Neo4j 5.x
- CUDA 11.8+ (for GPU acceleration, optional)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Sakhaa-Alsaedi/BioREASONIC.git
cd BioREASONIC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Download data and set up Neo4j
bash scripts/download_data.sh
bash scripts/setup_neo4j.sh
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the BioREASONIC Explainer at http://localhost:8080
```

---

## Usage

### Basic Usage

```python
from bioreasonc import BioREASONICAgent
from bioreasonc.grass import GRASSScorer

# Initialize the agent
agent = BioREASONICAgent(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    llm_model="gpt-4.1"
)

# Query the system
response = agent.query(
    "Is APOE a causal risk gene shared by T2D and Alzheimer's disease?"
)

print(response.answer)
print(response.causal_paths)
print(response.risk_scores)
```

### GRASS Scoring

```python
from bioreasonc.grass import GRASSScorer

# Initialize GRASS scorer
scorer = GRASSScorer()

# Calculate risk scores for a gene-disease pair
score = scorer.calculate(
    gene="APOE",
    disease="Alzheimer's disease",
    top_n=5  # Top-n SNP aggregation
)

print(f"GRASS Score: {score.value:.4f}")
print(f"Evidence breakdown: {score.evidence_scores}")
```

### CARES Evaluation

```python
from bioreasonc.evaluation import CARESEvaluator

# Initialize evaluator
evaluator = CARESEvaluator()

# Evaluate model response
cares_score = evaluator.evaluate(
    question="What is the causal relationship between APOE and T2D?",
    predicted_answer=model_output,
    ground_truth=reference_answer,
    causal_paths=kg_evidence
)

print(f"CARES Score: {cares_score.overall:.2f}")
print(f"  Structure: {cares_score.structure:.2f}")
print(f"  Risk: {cares_score.risk:.2f}")
print(f"  Causal: {cares_score.causal:.2f}")
print(f"  Mechanism: {cares_score.mechanism:.2f}")
```

---

## BioREASONIC-Bench

Our benchmark evaluates LLMs across four complementary reasoning taxonomies:

| Taxonomy | Abbr. | Description | Questions |
|----------|-------|-------------|-----------|
| **Structure-aware** | S | Graph topology understanding | 300 |
| **Risk-aware** | R | Genetic risk interpretation | 300 |
| **Causal-aware** | C | Causal inference capabilities | 300 |
| **Mechanism-aware** | M | Biological mechanism reasoning | 300 |

### Benchmark Generation Pipeline


### Running the Benchmark

```bash
# Run full benchmark evaluation
python -m bioreasonc.benchmark.run \
    --model gpt-4.1 \
    --benchmark all \
    --output results/

# Run specific taxonomy
python -m bioreasonc.benchmark.run \
    --model gpt-4.1 \
    --benchmark causal_aware \
    --strategy structured-cot
```

### Benchmark Results

**Table 1:** Average accuracy (%) across prompting strategies

| Strategy | BR-QA | PubMedQA | MedQA | BR-MCQ | MedMCQA | MMLU-Med |
|----------|-------|----------|-------|--------|---------|----------|
| Zero-shot | 58.3 | 57.4 | 72.0 | 11.6 | 64.8 | 80.6 |
| Few-shot | 40.1 | 52.4 | 18.6 | 13.1 | 43.6 | 80.1 |
| CoT | 46.6 | 56.9 | 79.7 | 33.9 | 69.8 | 69.6 |
| **Structured-CoT** | **64.4** | 55.9 | 77.0 | **64.5** | 66.8 | 73.2 |

---

## Experiments

### Single-Agent vs Multi-Agent Performance

| Model | BioREASONIC (Single) | BKGAgent (Multi) |
|-------|---------------------|------------------|
| GPT-4.1 | **96.84%** | 56.32% |
| GPT-4.1-mini | **95.26%** | 56.84% |
| DeepSeek-v3 | **94.21%** | 64.74% |
| Llama-3.1-8B | **80.00%** | 46.32% |

---

## Case Study: AD-T2D Biomarkers

BioREASONIC identified novel candidate genes for Alzheimer's Disease (AD) and Type 2 Diabetes (T2D) comorbidity:

| Rank | Gene | Category | Novelty |
|------|------|----------|---------|
| 1 | BTNL2 | Shared | 🔴 High |
| 2 | KIF11 | T2D | 🔴 High |
| 3 | HLA-DQB2 | Shared | 🔴 High |
| 4 | QPCTL | Shared | 🔴 High |
| 5 | WFS1 | T2D | 🟡 Moderate |
| 6 | JAZF1 | Shared | 🟡 Moderate |
| 7 | GIPR | Shared | 🟡 Moderate |
| 8 | TP53INP1 | Shared | 🟢 Low |

---

## 📧 Contact
 
**Sakhaa Alsaedi** - [sakhaa.alsaedi@kaust.edu.sa](mailto:sakhaa.alsaedi@kaust.edu.sa) and **Mohammed Saif** - [Mohammed.Saif@kaust.edu.sa](mailto:Mohammed.Saif@kaust.edu.sa)

King Abdullah University of Science and Technology (KAUST)  
Thuwal, Saudi Arabia

---


