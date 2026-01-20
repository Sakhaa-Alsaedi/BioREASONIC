# BioResonKGBench

## Causal Knowledge Graph Question Answering (CKGQA) Benchmark

---

## Overview

BioResonKGBench is a benchmark for evaluating LLMs on **Causal Knowledge Graph Question Answering (CKGQA)** tasks in the biomedical domain. The benchmark focuses on:

1. **Causal Relationships** - Understanding SNP → Gene → Disease causal pathways
2. **Multi-step Reasoning** - Complex inference chains across biological entities
3. **Knowledge vs Reasoning** - Separate evaluation tracks

---

## Dataset Size

### Overall

| Split | Questions |
|-------|-----------|
| Dev   | 192       |
| Test  | 1,088     |
| **Total** | **1,280** |

### By Track & Question Type

**Knowledge Track (760 questions)**

| Type | Dev | Test | Description |
|------|-----|------|-------------|
| **S** (Structure) | 24 | 136 | Graph topology and navigation |
| **R** (Risk) | 30 | 170 | Quantitative risk assessment |
| **C** (Causal) | 30 | 170 | Causal evidence evaluation |
| **M** (Mechanism) | 30 | 170 | Pathway and biological mechanisms |
| **Subtotal** | 114 | 646 | |

**Reasoning Track (520 questions)**

| Type | Dev | Test | Description |
|------|-----|------|-------------|
| **S** (Structure) | 18 | 102 | Multi-hop graph traversal |
| **R** (Risk) | 18 | 102 | Comparative risk analysis |
| **C** (Causal) | 18 | 102 | Causal inference reasoning |
| **M** (Mechanism) | 24 | 136 | Pathway integration |
| **Subtotal** | 78 | 442 | |

---

## Knowledge Graph Schema

```
                                    ┌─────────────────┐
                                    │     Tissue      │
                                    └────────▲────────┘
                                             │ ASSOCIATED_WITH
                                             │
┌───────┐  MAPS_TO   ┌───────┐  TRANSLATED   ┌─────────┐  ASSOCIATED   ┌──────────────────┐
│  SNP  │───────────>│ Gene  │──────INTO────>│ Protein │──────WITH────>│ Biological_proc  │
└───────┘            └───────┘               └─────────┘               │ Molecular_func   │
    │                    │                        │                    │ Cellular_comp    │
    │ PUTATIVE_          │ INCREASES_             │ ANNOTATED_         └──────────────────┘
    │ CAUSAL_            │ RISK_OF                │ IN_PATHWAY
    │ EFFECT             │                        │
    ▼                    ▼                        ▼
┌─────────┐         ┌─────────┐              ┌─────────┐
│ Disease │<────────│ Disease │              │ Pathway │
└─────────┘         └─────────┘              └─────────┘
```

### Node Types

| Node | Count | Properties |
|------|-------|------------|
| Gene | 118,314 | id, name, node_type (risk_gene) |
| Protein | 118,202 | id, name, node_type (risk_protein) |
| SNP | 528,789 | id, node_type (causal_snp, risk_snp) |
| Disease | 2,962 | id (DOID format), name |
| Tissue | 239 | id, name |
| Pathway | 2,442 | id, name |
| Biological_process | 12,432 | id, name |

### Relationship Types

| Relationship | Description |
|--------------|-------------|
| MAPS_TO | SNP → Gene mapping |
| TRANSLATED_INTO | Gene → Protein translation |
| INCREASES_RISK_OF | Gene → Disease risk association |
| PUTATIVE_CAUSAL_EFFECT | SNP → Disease causal link |
| ASSOCIATED_WITH | Protein → Tissue/GO associations |
| ANNOTATED_IN_PATHWAY | Protein → Pathway annotation |

---

## Directory Structure

```
02_BioResonKGBench/
├── README.md
├── config/
│   ├── config.local.yaml           # API keys
│   └── kg_config.yml               # Neo4j connection
│
├── data/
│   ├── combined_CKGQA_dev_matched.json    # All dev questions (192)
│   ├── combined_CKGQA_test_matched.json   # All test questions (1,088)
│   ├── knowledge/                   # Knowledge track by taxonomy
│   │   ├── S_knowledge_dev.json
│   │   ├── S_knowledge_test.json
│   │   ├── R_knowledge_dev.json
│   │   ├── R_knowledge_test.json
│   │   ├── C_knowledge_dev.json
│   │   ├── C_knowledge_test.json
│   │   ├── M_knowledge_dev.json
│   │   └── M_knowledge_test.json
│   └── reasoning/                   # Reasoning track by taxonomy
│       ├── S_reasoning_dev.json
│       ├── S_reasoning_test.json
│       ├── R_reasoning_dev.json
│       ├── R_reasoning_test.json
│       ├── C_reasoning_dev.json
│       ├── C_reasoning_test.json
│       ├── M_reasoning_dev.json
│       └── M_reasoning_test.json
│
├── src/
│   ├── generate_questions.py
│   └── evaluate_bioresonkg.py
│
├── results/                         # Evaluation outputs
│
├── tutorials/
│   ├── BioResonKGBench_Evaluation_Tutorial.ipynb
│   └── BioResonKGBench_MultiModel_Eval.ipynb
│
└── *.py                             # Utility scripts
```

---

## Data Format

### Question Format (JSON)

```json
{
  "question": "What genes are associated with increased risk for asthma?",
  "cypher": "MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d:Disease {id: \"DOID:2841\"}) RETURN g.id AS gene",
  "task_id": "S-DISEASE-GENES",
  "taxonomy": "S",
  "type": "one-hop",
  "answer_key": "gene",
  "parameters": {
    "disease": "asthma",
    "disease_id": "DOID:2841"
  },
  "category": "knowledge"
}
```

### Fields

| Field | Description |
|-------|-------------|
| `question` | Natural language question |
| `cypher` | Gold standard Cypher query |
| `task_id` | Task identifier (e.g., S-DISEASE-GENES) |
| `taxonomy` | Question type: S, R, C, or M |
| `type` | one-hop, multi-hop, classification, reasoning |
| `answer_key` | Column name in Cypher result containing the answer |
| `parameters` | Entity parameters used in the question |
| `category` | knowledge or reasoning |

---

## Task Types

### S (Structure) Questions
- Graph navigation and topology
- Entity lookup and retrieval
- Example: "What genes map to SNP rs12345?"

### R (Risk) Questions
- Quantitative risk assessment
- P-values, risk scores, evidence scores
- Example: "What is the p-value for SNP rs12345's effect on diabetes?"

### C (Causal) Questions
- Causal evidence evaluation
- Evidence levels, causal scores
- Example: "What is the evidence level for gene BRCA1 affecting breast cancer?"

### M (Mechanism) Questions
- Biological pathway understanding
- Protein function, GO terms, pathways
- Example: "What biological processes involve gene TP53?"

---

## Usage

### Load Data

```python
import json

# Load combined dataset
with open('data/combined_CKGQA_dev_matched.json') as f:
    dev_questions = json.load(f)

with open('data/combined_CKGQA_test_matched.json') as f:
    test_questions = json.load(f)

print(f"Dev: {len(dev_questions)} questions")
print(f"Test: {len(test_questions)} questions")
```

### Run Evaluation

```python
# See tutorials/BioResonKGBench_MultiModel_Eval.ipynb
```

---

## Validation

All 1,280 questions have been validated to return results from the Neo4j Knowledge Graph:

| Track | Valid | Total | Percentage |
|-------|-------|-------|------------|
| Knowledge | 760 | 760 | 100% |
| Reasoning | 520 | 520 | 100% |
| **Total** | **1,280** | **1,280** | **100%** |

---

## Comparison with BioKGBench

| Feature | BioKGBench | BioResonKGBench |
|---------|------------|-----------------|
| Total Questions | 698 | **1,280** |
| Question Types | 3 (hop-based) | 4 (SRMC taxonomy) |
| Task Tracks | 1 | 2 (Knowledge + Reasoning) |
| Causal Focus | No | **Yes** |
| Disease IDs | Mixed | DOID format |
| Validation | Partial | **100% validated** |

BioResonKGBench is ~1.8x larger than BioKGBench.

---

## Citation

```bibtex
@inproceedings{bioresonkgbench2026,
  title={BioResonKGBench: A Benchmark for Biomedical Causal Knowledge Graph Question Answering},
  author={KAUST CBRC},
  booktitle={ISMB 2026},
  year={2026}
}
```
