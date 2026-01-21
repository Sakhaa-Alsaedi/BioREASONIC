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
|--------|----------------------|-------------|
| **GRASS** | [GRASS](https://github.com/Sakhaa-Alsaedi/BioREASONIC/tree/main/GRASS) | Genetic Risk Aggregation Scoring System for quantifying gene-level disease risk |
| **BioREASONIC Agent** | [bioreasonc-agent](https://github.com/Sakhaa-Alsaedi/BioREASONIC/tree/main/bioreasonc-agent)| Single-agent causal GraphRAG system, achieving 3–20× higher efficiency than multi-agent baselines |
| **BioREASONIC-Bench** |[Benchmarks](https://github.com/Sakhaa-Alsaedi/BioREASONIC/tree/main/bioreasonc-agent/benchmarks/02_BioResonKGBench)| Multi-aware biomedical reasoning benchmark based on the S–R–C–M taxonomy |
| **Experiments** | [Notebook](https://github.com/Sakhaa-Alsaedi/BioREASONIC/tree/main/notebooks) | Executable notebooks reproducing all experiments reported in the manuscript |
<!-- | **Web Interface** | — | Interactive BioREASONIC Explainer for causal risk graph analysis | -->



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


