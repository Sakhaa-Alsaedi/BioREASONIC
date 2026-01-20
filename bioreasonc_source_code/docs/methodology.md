# BioREASONC-Bench: A Benchmark for Evaluating LLM Causal Reasoning in Biomedicine

## Methodology and Implementation Documentation

---

## Abstract

BioREASONC-Bench represents a systematic framework for evaluating Large Language Models' capacity for causal reasoning in biomedical contexts. This document presents the methodology and technical implementation of our benchmark, which leverages the CAUSALdb2 knowledge graph containing 66,057 gene-disease causal relationships derived through Mendelian Randomization analysis. Our framework implements a four-taxonomy categorization system (Structure, Risk, Causal, Mechanism) with five answer formats to comprehensively assess LLM reasoning capabilities.

---

## 1. Introduction

### 1.1 Background and Motivation

The emergence of Large Language Models (LLMs) has catalyzed significant interest in their application to biomedical research. However, evaluating their capacity for genuine causal reasoning—as opposed to pattern matching or memorization—remains challenging. Existing benchmarks often conflate correlation with causation or fail to ground evaluations in experimentally validated causal relationships.

BioREASONC-Bench addresses these limitations by grounding all benchmark items in Mendelian Randomization evidence from CAUSALdb2, which provides gold-standard causal inference by leveraging genetic variants as instrumental variables immune to confounding.

### 1.2 Benchmark Design Principles

- **Evidence-Grounded:** All causal claims derived from MR analysis with quantified effect sizes
- **Multi-Dimensional:** Four taxonomies testing distinct aspects of causal understanding  
- **Format-Diverse:** Five answer formats from yes/no to complex reasoning chains
- **Reproducible:** Deterministic generation with complete provenance tracking

---

## 2. Knowledge Graph Structure

### 2.1 CAUSALdb2 Overview

CAUSALdb2 serves as the foundation knowledge graph, containing curated causal relationships derived from large-scale GWAS studies and Mendelian Randomization analyses.

| Metric | Value | Description |
|--------|-------|-------------|
| Gene-Disease Pairs | 66,057 | Unique causal relationships |
| Unique Genes | ~15,000 | Genes with MR evidence |
| Unique Diseases | ~2,000 | Disease/trait phenotypes |
| Total SNPs | ~500,000 | Instrumental variables |
| Evidence Levels | 5 | From weak to very strong |

### 2.2 Evidence Strength Classification

| Evidence Level | P-value Threshold | Description |
|---------------|-------------------|-------------|
| Very Strong | p < 5×10⁻⁸ | Genome-wide significance |
| Strong | 5×10⁻⁸ ≤ p < 1×10⁻⁵ | Strong statistical support |
| Moderate | 1×10⁻⁵ ≤ p < 1×10⁻³ | Moderate evidence |
| Suggestive | 1×10⁻³ ≤ p < 0.01 | Suggestive association |
| Weak | p ≥ 0.01 | Weak or nominal evidence |

---

## 3. Taxonomy Framework

BioREASONC-Bench implements a four-taxonomy classification system, each targeting distinct aspects of causal reasoning capability.

### 3.1 S-Taxonomy: Structure

The Structure taxonomy evaluates understanding of basic causal graph structure and relationships between entities.

- **S-GENE-DISEASE:** Maps genes to their associated diseases
- **S-SNP-COUNT:** Quantifies instrumental variable support

### 3.2 R-Taxonomy: Risk

The Risk taxonomy assesses comprehension of risk factors and effect magnitudes.

- **R-RISK-FACTOR:** Identifies genetic risk factors for diseases
- **R-RISK-LEVEL:** Evaluates risk magnitude interpretation

### 3.3 C-Taxonomy: Causal

The Causal taxonomy directly probes causal inference understanding.

- **C-MR-EVIDENCE:** Interprets Mendelian Randomization results
- **C-CAUSAL-STRENGTH:** Evaluates causal effect magnitude

### 3.4 M-Taxonomy: Mechanism

The Mechanism taxonomy tests mechanistic pathway understanding.

- **M-PATHWAY:** Identifies biological pathway involvement
- **M-MECHANISM:** Explains molecular mechanisms

---

## 4. Answer Format Specifications

| Format | Description | Evaluation Method |
|--------|-------------|-------------------|
| yes_no | Binary true/false questions | Exact match accuracy |
| mcq | Multiple choice (4 options) | Option selection accuracy |
| short | Brief factual responses | Token F1 / semantic similarity |
| long | Extended explanations | ROUGE / semantic evaluation |
| reasoning | Chain-of-thought responses | Step validity + conclusion |

---

## 5. Implementation Architecture

### 5.1 System Components

- **kg_ingest.py:** Knowledge graph loading and question generation (~1,400 lines)
- **prompts.py:** All prompt templates and taxonomies (~3,400 lines)
- **validator.py:** Multi-LLM validation pipeline
- **explainer.py:** Answer explanation generation
- **paraphraser.py:** Question paraphrasing for robustness testing

### 5.2 Generation Pipeline

1. Load and parse CAUSALdb2 knowledge graph from JSON
2. Build indices for gene→diseases and disease→genes lookup
3. Calculate evidence levels based on p-values and effect sizes
4. Sample representative gene-disease pairs across evidence levels
5. Generate questions using taxonomy-specific templates
6. Apply answer format transformations
7. Validate generated items for completeness and consistency
8. Export benchmark in multiple formats (JSON, JSONL, CSV)

---

## 6. Quality Assurance

### 6.1 Template Validation

All generated questions undergo multi-stage validation:

- **Structural Validation:** Verify all template placeholders are filled
- **Semantic Validation:** Ensure question-answer coherence
- **Factual Validation:** Cross-check against knowledge graph
- **Format Validation:** Verify answer format compliance

### 6.2 Multi-LLM Validation

A subset of generated items undergoes validation by multiple LLMs to ensure questions are answerable and answers are derivable from provided context.

---

## 7. Benchmark Scaling

| Configuration | Items per Taxonomy | Total Items | Use Case |
|---------------|-------------------|-------------|----------|
| Development | 50 | ~1,000 | Quick testing |
| Standard | 200 | ~4,000 | Regular evaluation |
| Full | 500 | ~10,000 | Comprehensive assessment |
| Extended | 1,000 | ~20,000 | Research-scale analysis |

---

## 8. Usage Example

```python
from bioreasonc_creator import kg_ingest

# Initialize with CAUSALdb2
kg = kg_ingest.BioReasonCKG("data/causaldb2.json")

# Generate benchmark (200 items per taxonomy)
benchmark = kg.generate_benchmark_all_formats(
    n_per_taxonomy=200,
    formats=['yes_no', 'mcq', 'short', 'long', 'reasoning']
)

# Export to multiple formats
kg.export_benchmark(benchmark, "output/", formats=['json', 'jsonl', 'csv'])
```

---

## 9. Evaluation Metrics

### 9.1 Per-Format Metrics

| Format | Primary Metric | Secondary Metrics |
|--------|---------------|-------------------|
| yes_no | Accuracy | F1, Precision, Recall |
| mcq | Accuracy | Per-option analysis |
| short | Token F1 | Exact match, BERTScore |
| long | ROUGE-L | BERTScore, Human eval |
| reasoning | Step accuracy | Chain validity, Conclusion match |

### 9.2 Aggregate Analysis

Results are aggregated across multiple dimensions: by taxonomy (S, R, C, M), by evidence level (very_strong to weak), by answer format, and by disease category. This enables fine-grained analysis of model strengths and weaknesses.

---

## 10. Conclusion

BioREASONC-Bench provides a rigorous, evidence-grounded framework for evaluating LLM causal reasoning in biomedicine. By leveraging Mendelian Randomization evidence from CAUSALdb2 and implementing a comprehensive taxonomy system, the benchmark enables nuanced assessment of model capabilities across multiple dimensions of causal understanding.

Key contributions include:
1. A novel four-taxonomy framework targeting distinct causal reasoning aspects
2. Five answer formats enabling multi-faceted evaluation
3. Grounding in experimentally-validated causal relationships
4. A scalable generation pipeline with comprehensive quality assurance

---

## References

1. CAUSALdb2: An integrative database for gene-trait causal relationships
2. Mendelian Randomization: Methods for using genetic variants in causal estimation
3. UK Biobank: A large prospective cohort study
4. FinnGen: Genomic analyses of Finnish population cohorts
