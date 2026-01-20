#!/usr/bin/env python3
"""
CKG_Bench Question Generator

Generates questions for the Causal Knowledge Graph Benchmark with:
- Knowledge-based questions (graph traversal)
- Reasoning evaluation tasks (critical thinking)

Each question includes:
- Natural language question
- Cypher query for Neo4j
- Expected answer
- Task metadata
"""

import os
import sys
import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / 'data'
KG_DATA_DIR = Path('./Neo4j_KG/data')

random.seed(42)

# ============================================================
# QUESTION TEMPLATES WITH CYPHER QUERIES
# ============================================================

# --- KNOWLEDGE-BASED TEMPLATES ---

KNOWLEDGE_TEMPLATES = {
    # S-Taxonomy: Structure
    'S-SNP-GENE': {
        'questions': [
            "Which gene is SNP {snp_id} mapped to?",
            "What gene does the SNP {snp_id} map to?",
            "Can you identify the gene that SNP {snp_id} is associated with?",
            "Which gene is linked to SNP {snp_id}?",
        ],
        'cypher': """
            MATCH (s:SNP {{id: '{snp_id}'}})-[r:MAPS_TO]->(g:Gene)
            RETURN g.id AS gene_symbol, g.name AS gene_name, r.link_score AS link_score
            ORDER BY r.link_score DESC
        """,
        'answer_key': 'gene_symbol',
        'type': 'one-hop',
        'taxonomy': 'S',
        'task_id': 'S-SNP-GENE'
    },

    'S-GENE-DISEASE': {
        'questions': [
            "Which diseases does gene {gene} increase risk of?",
            "What diseases are associated with increased risk from gene {gene}?",
            "Can you list the diseases that gene {gene} contributes to?",
            "Which diseases has gene {gene} been linked to as a risk factor?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            RETURN d.name AS disease, d.id AS disease_id, r.risk_score AS risk_score
            ORDER BY r.risk_score DESC
        """,
        'answer_key': 'disease',
        'type': 'one-hop',
        'taxonomy': 'S',
        'task_id': 'S-GENE-DISEASE'
    },

    'S-DISEASE-GENES': {
        'questions': [
            "Which genes increase risk of {disease}?",
            "What genes are risk factors for {disease}?",
            "Can you identify genes that contribute to {disease} risk?",
            "Which genes have been implicated in {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, r.risk_score AS risk_score, r.snp_count AS snp_count
            ORDER BY r.risk_score DESC
            LIMIT 20
        """,
        'answer_key': 'gene',
        'type': 'one-hop',
        'taxonomy': 'S',
        'task_id': 'S-DISEASE-GENES'
    },

    'S-SNP-COUNT': {
        'questions': [
            "How many SNPs are mapped to gene {gene}?",
            "What is the count of SNPs associated with gene {gene}?",
            "Can you tell me the number of SNPs linked to gene {gene}?",
        ],
        'cypher': """
            MATCH (s:SNP)-[r:MAPS_TO]->(g:Gene {{id: '{gene}'}})
            RETURN count(s) AS snp_count
        """,
        'answer_key': 'snp_count',
        'type': 'aggregation',
        'taxonomy': 'S',
        'task_id': 'S-SNP-COUNT'
    },

    'S-SNP-DISEASE': {
        'questions': [
            "Which diseases does SNP {snp_id} have a putative causal effect on?",
            "What diseases are causally linked to SNP {snp_id}?",
            "Can you identify diseases with causal associations to SNP {snp_id}?",
        ],
        'cypher': """
            MATCH (s:SNP {{id: '{snp_id}'}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            RETURN d.name AS disease, r.causal_score AS causal_score, r.pvalue AS pvalue
            ORDER BY r.causal_score DESC
        """,
        'answer_key': 'disease',
        'type': 'one-hop',
        'taxonomy': 'S',
        'task_id': 'S-SNP-DISEASE'
    },

    'S-PATH': {
        'questions': [
            "What is the path from SNP {snp_id} to gene and then to disease?",
            "Can you trace the causal path from SNP {snp_id} through its gene to diseases?",
            "Show the complete path: SNP {snp_id} → Gene → Disease",
        ],
        'cypher': """
            MATCH (s:SNP {{id: '{snp_id}'}})-[r1:MAPS_TO]->(g:Gene)-[r2:INCREASES_RISK_OF]->(d:Disease)
            RETURN s.id AS snp, g.id AS gene, d.name AS disease,
                   r1.link_score AS snp_gene_score, r2.risk_score AS gene_disease_score
            LIMIT 10
        """,
        'answer_key': ['snp', 'gene', 'disease'],
        'type': 'multi-hop',
        'taxonomy': 'S',
        'task_id': 'S-PATH'
    },

    # R-Taxonomy: Risk
    'R-RISK-SCORE': {
        'questions': [
            "What is the risk score of gene {gene} for {disease}?",
            "How strong is the risk association between gene {gene} and {disease}?",
            "Can you provide the risk score for gene {gene}'s effect on {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease, r.risk_score AS risk_score,
                   r.evidence_score AS evidence_score
        """,
        'answer_key': 'risk_score',
        'type': 'one-hop',
        'taxonomy': 'R',
        'task_id': 'R-RISK-SCORE'
    },

    'R-TOP-RISK': {
        'questions': [
            "Which gene has the highest risk score for {disease}?",
            "What is the top risk gene for {disease}?",
            "Can you identify the gene with strongest risk association for {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, r.risk_score AS risk_score
            ORDER BY r.risk_score DESC
            LIMIT 1
        """,
        'answer_key': 'gene',
        'type': 'ranking',
        'taxonomy': 'R',
        'task_id': 'R-TOP-RISK'
    },

    'R-COMPARE-RISK': {
        'questions': [
            "Which gene has higher risk for {disease}: {gene1} or {gene2}?",
            "Between {gene1} and {gene2}, which has stronger risk association with {disease}?",
            "Compare the risk scores of {gene1} and {gene2} for {disease}.",
        ],
        'cypher': """
            MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE g.id IN ['{gene1}', '{gene2}'] AND (d.name CONTAINS '{disease}' OR d.id = '{disease_id}')
            RETURN g.id AS gene, r.risk_score AS risk_score
            ORDER BY r.risk_score DESC
        """,
        'answer_key': 'gene',
        'type': 'conjunction',
        'taxonomy': 'R',
        'task_id': 'R-COMPARE-RISK'
    },

    'R-EFFECT-SIZE': {
        'questions': [
            "What is the effect size (beta) of SNP {snp_id} on {disease}?",
            "Can you provide the beta value for SNP {snp_id}'s effect on {disease}?",
            "What is the magnitude of effect for SNP {snp_id} on {disease}?",
        ],
        'cypher': """
            MATCH (s:SNP {{id: '{snp_id}'}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN s.id AS snp, d.name AS disease, r.beta AS beta, r.pvalue AS pvalue
        """,
        'answer_key': 'beta',
        'type': 'one-hop',
        'taxonomy': 'R',
        'task_id': 'R-EFFECT-SIZE'
    },

    'R-PVALUE': {
        'questions': [
            "What is the p-value for SNP {snp_id}'s effect on {disease}?",
            "How statistically significant is the association between SNP {snp_id} and {disease}?",
        ],
        'cypher': """
            MATCH (s:SNP {{id: '{snp_id}'}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN s.id AS snp, d.name AS disease, r.pvalue AS pvalue, r.causal_score AS causal_score
        """,
        'answer_key': 'pvalue',
        'type': 'one-hop',
        'taxonomy': 'R',
        'task_id': 'R-PVALUE'
    },

    # C-Taxonomy: Causal
    'C-CAUSAL-SNP': {
        'questions': [
            "Which SNPs have a putative causal effect on {disease}?",
            "What SNPs are causally linked to {disease}?",
            "Can you list the causal SNPs for {disease}?",
        ],
        'cypher': """
            MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN s.id AS snp, r.causal_score AS causal_score, r.pvalue AS pvalue
            ORDER BY r.causal_score DESC
            LIMIT 20
        """,
        'answer_key': 'snp',
        'type': 'one-hop',
        'taxonomy': 'C',
        'task_id': 'C-CAUSAL-SNP'
    },

    'C-CAUSAL-SCORE': {
        'questions': [
            "What is the causal score of SNP {snp_id} for {disease}?",
            "How strong is the causal evidence for SNP {snp_id} affecting {disease}?",
        ],
        'cypher': """
            MATCH (s:SNP {{id: '{snp_id}'}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN s.id AS snp, d.name AS disease, r.causal_score AS causal_score
        """,
        'answer_key': 'causal_score',
        'type': 'one-hop',
        'taxonomy': 'C',
        'task_id': 'C-CAUSAL-SCORE'
    },

    'C-MR-VALIDATED': {
        'questions': [
            "Is gene {gene} MR-validated for {disease}?",
            "Does gene {gene} have Mendelian Randomization support for {disease}?",
            "Is there MR evidence for gene {gene} causing {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE (d.name CONTAINS '{disease}' OR d.id = '{disease_id}') AND r.mr_score > 0
            RETURN g.id AS gene, d.name AS disease, r.mr_score AS mr_score,
                   CASE WHEN r.mr_score >= 0.8 THEN 'Yes' ELSE 'Partial' END AS mr_validated
        """,
        'answer_key': 'mr_validated',
        'type': 'classification',
        'taxonomy': 'C',
        'task_id': 'C-MR-VALIDATED'
    },

    'C-EVIDENCE-LEVEL': {
        'questions': [
            "What is the evidence level for gene {gene} affecting {disease}?",
            "How strong is the causal evidence for gene {gene} → {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease,
                   r.evidence_score AS evidence_score,
                   CASE
                       WHEN r.evidence_score >= 0.8 THEN 'very_strong'
                       WHEN r.evidence_score >= 0.6 THEN 'strong'
                       WHEN r.evidence_score >= 0.4 THEN 'moderate'
                       WHEN r.evidence_score >= 0.2 THEN 'suggestive'
                       ELSE 'weak'
                   END AS evidence_level
        """,
        'answer_key': 'evidence_level',
        'type': 'classification',
        'taxonomy': 'C',
        'task_id': 'C-EVIDENCE-LEVEL'
    },

    'C-TOP-CAUSAL': {
        'questions': [
            "Which SNP has the strongest causal effect on {disease}?",
            "What is the top causal SNP for {disease}?",
        ],
        'cypher': """
            MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN s.id AS snp, r.causal_score AS causal_score
            ORDER BY r.causal_score DESC
            LIMIT 1
        """,
        'answer_key': 'snp',
        'type': 'ranking',
        'taxonomy': 'C',
        'task_id': 'C-TOP-CAUSAL'
    },

    # M-Taxonomy: Mechanism
    'M-PATHWAY': {
        'questions': [
            "Which pathways involve genes that increase risk of {disease}?",
            "What biological pathways are enriched in {disease} risk genes?",
        ],
        'cypher': """
            MATCH (g:Gene)-[r1:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            MATCH (g)-[:TRANSLATED_INTO]->(p:Protein)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
            RETURN pw.name AS pathway, count(DISTINCT g) AS gene_count
            ORDER BY gene_count DESC
            LIMIT 10
        """,
        'answer_key': 'pathway',
        'type': 'multi-hop',
        'taxonomy': 'M',
        'task_id': 'M-PATHWAY'
    },

    'M-GO-PROCESS': {
        'questions': [
            "What biological processes are associated with genes that increase risk of {disease}?",
            "Which GO biological processes involve {disease} risk genes?",
        ],
        'cypher': """
            MATCH (g:Gene)-[r1:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            MATCH (g)-[:TRANSLATED_INTO]->(p:Protein)-[:ASSOCIATED_WITH]->(bp:Biological_process)
            RETURN bp.name AS biological_process, count(DISTINCT g) AS gene_count
            ORDER BY gene_count DESC
            LIMIT 10
        """,
        'answer_key': 'biological_process',
        'type': 'multi-hop',
        'taxonomy': 'M',
        'task_id': 'M-GO-PROCESS'
    },

    'M-PROTEIN': {
        'questions': [
            "Which protein is encoded by the risk gene {gene}?",
            "What protein does gene {gene} translate into?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[:TRANSLATED_INTO]->(p:Protein)
            RETURN g.id AS gene, p.id AS protein_id, p.name AS protein_name
        """,
        'answer_key': 'protein_name',
        'type': 'one-hop',
        'taxonomy': 'M',
        'task_id': 'M-PROTEIN'
    },

    'M-SHARED-PATHWAY': {
        'questions': [
            "Which pathways are shared by risk genes for both {disease1} and {disease2}?",
            "What pathways do {disease1} and {disease2} risk genes have in common?",
        ],
        'cypher': """
            MATCH (g1:Gene)-[:INCREASES_RISK_OF]->(d1:Disease)
            WHERE d1.name CONTAINS '{disease1}'
            MATCH (g1)-[:TRANSLATED_INTO]->(p1:Protein)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
            MATCH (g2:Gene)-[:INCREASES_RISK_OF]->(d2:Disease)
            WHERE d2.name CONTAINS '{disease2}'
            MATCH (g2)-[:TRANSLATED_INTO]->(p2:Protein)-[:ANNOTATED_IN_PATHWAY]->(pw)
            RETURN pw.name AS shared_pathway,
                   collect(DISTINCT g1.id) AS genes_disease1,
                   collect(DISTINCT g2.id) AS genes_disease2
            LIMIT 10
        """,
        'answer_key': 'shared_pathway',
        'type': 'conjunction',
        'taxonomy': 'M',
        'task_id': 'M-SHARED-PATHWAY'
    },

    'M-DRUG-TARGET': {
        'questions': [
            "Which drugs target proteins from genes that increase risk of {disease}?",
            "What drugs could potentially treat {disease} based on risk gene targets?",
        ],
        'cypher': """
            MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            MATCH (g)-[:TRANSLATED_INTO]->(p:Protein)<-[:TARGETS]-(dr:Drug)
            RETURN dr.name AS drug, p.name AS target_protein, g.id AS risk_gene
            ORDER BY r.risk_score DESC
            LIMIT 10
        """,
        'answer_key': 'drug',
        'type': 'multi-hop',
        'taxonomy': 'M',
        'task_id': 'M-DRUG-TARGET'
    },
}

# --- REASONING TASK TEMPLATES ---

REASONING_TEMPLATES = {
    # S: Structure-Aware Reasoning
    'S1': {
        'questions': [
            "Is the Gene {gene}–{disease} relationship direct or mediated by another node?",
            "Does gene {gene} have a direct relationship to {disease}, or is it indirect?",
        ],
        'cypher': """
            // Check for direct relationship
            OPTIONAL MATCH direct = (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            // Check for indirect through SNP
            OPTIONAL MATCH indirect = (g:Gene {{id: '{gene}'}})<-[:MAPS_TO]-(s:SNP)-[:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN
                CASE WHEN direct IS NOT NULL THEN 'direct' ELSE 'indirect' END AS relationship_type,
                CASE WHEN direct IS NOT NULL THEN true ELSE false END AS has_direct,
                CASE WHEN indirect IS NOT NULL THEN true ELSE false END AS has_indirect
        """,
        'answer_format': 'yes_no',
        'reasoning_type': 'Structure-aware',
        'evaluation_focus': 'Direct vs indirect relations',
        'taxonomy': 'S',
        'task_id': 'S1'
    },

    'S2': {
        'questions': [
            "Which genes linked to {disease} have more than {threshold} supporting SNPs?",
            "List genes associated with {disease} that have at least {threshold} mapped SNPs.",
        ],
        'cypher': """
            MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            WITH g, r
            MATCH (s:SNP)-[:MAPS_TO]->(g)
            WITH g, r, count(s) AS snp_count
            WHERE snp_count > {threshold}
            RETURN g.id AS gene, snp_count, r.risk_score AS risk_score
            ORDER BY snp_count DESC
        """,
        'answer_format': 'list',
        'reasoning_type': 'Structure-aware',
        'evaluation_focus': 'Structural constraint handling',
        'taxonomy': 'S',
        'task_id': 'S2'
    },

    'S3': {
        'questions': [
            "Does finding pathways for {disease} risk genes require one-hop or multi-hop traversal?",
            "Is the query 'pathways for {disease} risk genes' a one-hop or multi-hop query?",
        ],
        'cypher': """
            // This is a meta-query - the answer depends on schema understanding
            // Gene -[:INCREASES_RISK_OF]-> Disease (1 hop)
            // Gene -[:TRANSLATED_INTO]-> Protein -[:ANNOTATED_IN_PATHWAY]-> Pathway (2 hops from gene)
            // So Gene -> Disease -> Pathway requires: Gene->Protein->Pathway = multi-hop
            RETURN 'multi-hop' AS answer,
                   'Gene->Protein->Pathway requires 2 hops' AS explanation
        """,
        'answer_format': 'mcq',
        'options': ['one-hop', 'multi-hop'],
        'correct_answer': 'multi-hop',
        'reasoning_type': 'Structure-aware',
        'evaluation_focus': 'Hop complexity identification',
        'taxonomy': 'S',
        'task_id': 'S3'
    },

    # R: Risk-Aware Reasoning
    'R1': {
        'questions': [
            "Does Gene {gene} increase or reduce the risk of {disease}?",
            "What is the direction of effect for Gene {gene} on {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease,
                   r.causal_score AS causal_score,
                   CASE WHEN r.causal_score > 0 THEN 'increases' ELSE 'decreases' END AS risk_direction
        """,
        'answer_format': 'mcq',
        'options': ['increases risk', 'decreases risk', 'no effect'],
        'reasoning_type': 'Risk-aware',
        'evaluation_focus': 'Risk polarity interpretation',
        'taxonomy': 'R',
        'task_id': 'R1'
    },

    'R2': {
        'questions': [
            "Is the risk effect of Gene {gene1} stronger than Gene {gene2} for {disease}?",
            "Compare: Does {gene1} have a greater risk contribution to {disease} than {gene2}?",
        ],
        'cypher': """
            MATCH (g1:Gene {{id: '{gene1}'}})-[r1:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            MATCH (g2:Gene {{id: '{gene2}'}})-[r2:INCREASES_RISK_OF]->(d)
            RETURN g1.id AS gene1, r1.risk_score AS risk_score1,
                   g2.id AS gene2, r2.risk_score AS risk_score2,
                   CASE WHEN r1.risk_score > r2.risk_score THEN 'Yes' ELSE 'No' END AS gene1_stronger
        """,
        'answer_format': 'yes_no',
        'reasoning_type': 'Risk-aware',
        'evaluation_focus': 'Effect size comparison',
        'taxonomy': 'R',
        'task_id': 'R2'
    },

    'R3': {
        'questions': [
            "Is the effect size of Gene {gene} for {disease} clinically meaningful (risk_score > {threshold})?",
            "Does Gene {gene} have a clinically significant effect on {disease} (threshold: {threshold})?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease, r.risk_score AS risk_score,
                   CASE WHEN r.risk_score > {threshold} THEN 'Yes' ELSE 'No' END AS clinically_meaningful
        """,
        'answer_format': 'yes_no',
        'reasoning_type': 'Risk-aware',
        'evaluation_focus': 'Threshold-based reasoning',
        'taxonomy': 'R',
        'task_id': 'R3'
    },

    # C: Causal-Aware Reasoning
    'C1': {
        'questions': [
            "Is Gene {gene} causally linked to {disease} or only statistically associated?",
            "Does the relationship between Gene {gene} and {disease} imply causation or just correlation?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease,
                   r.mr_score AS mr_score,
                   r.causal_score AS causal_score,
                   CASE
                       WHEN r.mr_score >= 0.8 AND r.causal_score > 0.5 THEN 'causal'
                       WHEN r.mr_score >= 0.5 THEN 'likely_causal'
                       ELSE 'associated'
                   END AS relationship_type
        """,
        'answer_format': 'mcq',
        'options': ['causally linked', 'statistically associated', 'insufficient evidence'],
        'reasoning_type': 'Causal-aware',
        'evaluation_focus': 'Association vs causation',
        'taxonomy': 'C',
        'task_id': 'C1'
    },

    'C2': {
        'questions': [
            "Does MR evidence support a causal role for Gene {gene} in {disease}?",
            "Is there Mendelian Randomization support for Gene {gene} causing {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease,
                   r.mr_score AS mr_score,
                   CASE WHEN r.mr_score >= 0.8 THEN 'Yes'
                        WHEN r.mr_score >= 0.5 THEN 'Partial'
                        ELSE 'No' END AS mr_support
        """,
        'answer_format': 'yes_no',
        'reasoning_type': 'Causal-aware',
        'evaluation_focus': 'MR evidence validation',
        'taxonomy': 'C',
        'task_id': 'C2'
    },

    'C3': {
        'questions': [
            "Is there enough evidence to claim Gene {gene} causes {disease}?",
            "Can we confidently say Gene {gene} has a causal effect on {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease,
                   r.mr_score AS mr_score,
                   r.evidence_score AS evidence_score,
                   r.snp_count AS supporting_snps,
                   CASE
                       WHEN r.mr_score >= 0.8 AND r.evidence_score >= 0.6 AND r.snp_count >= 5 THEN 'sufficient'
                       WHEN r.mr_score >= 0.5 OR r.evidence_score >= 0.4 THEN 'partial'
                       ELSE 'insufficient'
                   END AS evidence_sufficiency
        """,
        'answer_format': 'yes_no',
        'reasoning_type': 'Causal-aware',
        'evaluation_focus': 'Insufficient evidence recognition',
        'taxonomy': 'C',
        'task_id': 'C3'
    },

    # M: Semantic-Aware Reasoning
    'M1': {
        'questions': [
            "Gene {gene} leads to increased susceptibility to {disease}—what does this imply?",
            "If Gene {gene} increases susceptibility to {disease}, what is the interpretation?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease,
                   'Carriers of risk variants in this gene have elevated probability of developing the disease' AS implication,
                   r.risk_score AS risk_score
        """,
        'answer_format': 'short',
        'reasoning_type': 'Semantic-aware',
        'evaluation_focus': 'Risk-related language understanding',
        'taxonomy': 'M',
        'task_id': 'M1'
    },

    'M2': {
        'questions': [
            "Does 'Gene {gene} is linked to {disease}' imply causation?",
            "If we say Gene {gene} is linked to {disease}, does this mean causation?",
        ],
        'cypher': """
            // This is a semantic reasoning question
            // "Linked to" does not necessarily imply causation - it could be correlation
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN 'No' AS implies_causation,
                   'Linked to indicates association, not necessarily causation. Causation requires MR or experimental evidence.' AS explanation,
                   r.mr_score AS mr_score
        """,
        'answer_format': 'yes_no',
        'correct_answer': 'No',
        'reasoning_type': 'Semantic-aware',
        'evaluation_focus': 'False causal inference avoidance',
        'taxonomy': 'M',
        'task_id': 'M2'
    },

    'M3': {
        'questions': [
            "What does the term 'risk factor' indicate in the context of Gene {gene} and {disease}?",
            "How should we interpret 'risk factor' for Gene {gene} regarding {disease}?",
        ],
        'cypher': """
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN g.id AS gene, d.name AS disease,
                   'A risk factor is a variable associated with increased probability of disease, not a deterministic cause' AS definition,
                   r.risk_score AS risk_score
        """,
        'answer_format': 'short',
        'reasoning_type': 'Semantic-aware',
        'evaluation_focus': 'Ontology-based interpretation',
        'taxonomy': 'M',
        'task_id': 'M3'
    },

    'M4': {
        'questions': [
            "Do 'increases risk' and 'elevates susceptibility' mean the same relation for Gene {gene} and {disease}?",
            "Are 'increases risk of' and 'elevates susceptibility to' semantically equivalent?",
        ],
        'cypher': """
            // Semantic equivalence check
            MATCH (g:Gene {{id: '{gene}'}})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE d.name CONTAINS '{disease}' OR d.id = '{disease_id}'
            RETURN 'Yes' AS semantic_equivalence,
                   'Both phrases indicate the gene variant is associated with higher disease probability' AS explanation
        """,
        'answer_format': 'yes_no',
        'correct_answer': 'Yes',
        'reasoning_type': 'Semantic-aware',
        'evaluation_focus': 'Semantic consistency',
        'taxonomy': 'M',
        'task_id': 'M4'
    },
}


# ============================================================
# DATA LOADING
# ============================================================

def load_kg_data():
    """Load knowledge graph data from TSV files."""
    print("Loading KG data...")

    # Load genes
    genes_df = pd.read_csv(KG_DATA_DIR / 'Gene.tsv', sep='\t')
    genes = genes_df['name'].tolist()
    print(f"  Loaded {len(genes)} genes")

    # Load diseases
    diseases_df = pd.read_csv(KG_DATA_DIR / 'Disease.tsv', sep='\t')
    diseases = diseases_df[['ID', 'name']].to_dict('records')
    print(f"  Loaded {len(diseases)} diseases")

    # Load SNPs
    snps_df = pd.read_csv(KG_DATA_DIR / 'SNP.tsv', sep='\t')
    snps = snps_df['ID'].tolist()
    print(f"  Loaded {len(snps)} SNPs")

    # Load Gene-Disease relationships
    gene_disease_df = pd.read_csv(KG_DATA_DIR / 'Gene_INCREASES_RISK_OF_Disease.tsv', sep='\t')
    gene_disease = gene_disease_df.to_dict('records')
    print(f"  Loaded {len(gene_disease)} gene-disease relationships")

    # Load SNP-Gene relationships
    snp_gene_df = pd.read_csv(KG_DATA_DIR / 'SNP_MAPS_TO_Gene.tsv', sep='\t')
    snp_gene = snp_gene_df.to_dict('records')
    print(f"  Loaded {len(snp_gene)} SNP-gene relationships")

    # Load SNP-Disease relationships
    snp_disease_df = pd.read_csv(KG_DATA_DIR / 'SNP_PUTATIVE_CAUSAL_EFFECT_Disease.tsv', sep='\t')
    snp_disease = snp_disease_df.head(100000).to_dict('records')  # Limit for memory
    print(f"  Loaded {len(snp_disease)} SNP-disease relationships")

    # Create lookup for gene names
    gene_id_to_name = dict(zip(genes_df['ID'], genes_df['name']))

    return {
        'genes': genes,
        'diseases': diseases,
        'snps': snps,
        'gene_disease': gene_disease,
        'snp_gene': snp_gene,
        'snp_disease': snp_disease,
        'gene_id_to_name': gene_id_to_name
    }


# ============================================================
# QUESTION GENERATION
# ============================================================

def generate_knowledge_questions(kg_data: Dict, n_per_task: int = 50) -> List[Dict]:
    """Generate knowledge-based questions."""
    questions = []

    gene_disease = kg_data['gene_disease']
    snp_gene = kg_data['snp_gene']
    snp_disease = kg_data['snp_disease']
    diseases = kg_data['diseases']
    gene_id_to_name = kg_data['gene_id_to_name']

    # Sample gene-disease pairs
    gd_sample = random.sample(gene_disease, min(n_per_task * 10, len(gene_disease)))

    for task_id, template in KNOWLEDGE_TEMPLATES.items():
        print(f"  Generating {task_id} questions...")
        task_questions = []

        for i in range(n_per_task):
            try:
                if task_id.startswith('S-SNP') or task_id in ['R-EFFECT-SIZE', 'R-PVALUE', 'C-CAUSAL-SNP', 'C-CAUSAL-SCORE', 'C-TOP-CAUSAL']:
                    # SNP-based questions
                    if task_id == 'S-SNP-GENE':
                        record = random.choice(snp_gene)
                        params = {
                            'snp_id': record['START_ID'],
                        }
                    elif task_id == 'S-SNP-DISEASE':
                        record = random.choice(snp_disease)
                        disease_info = next((d for d in diseases if d['ID'] == record['END_ID']), None)
                        params = {
                            'snp_id': record['START_ID'],
                            'disease': disease_info['name'] if disease_info else record['END_ID'],
                            'disease_id': record['END_ID']
                        }
                    elif task_id == 'S-PATH':
                        # Find SNP with both gene and disease links
                        snp_rec = random.choice(snp_gene)
                        params = {'snp_id': snp_rec['START_ID']}
                    else:
                        record = random.choice(snp_disease)
                        disease_info = next((d for d in diseases if d['ID'] == record['END_ID']), None)
                        params = {
                            'snp_id': record['START_ID'],
                            'disease': disease_info['name'] if disease_info else record['END_ID'],
                            'disease_id': record['END_ID']
                        }
                elif task_id in ['S-SNP-COUNT', 'M-PROTEIN']:
                    # Gene-only questions
                    record = random.choice(gd_sample)
                    gene_name = gene_id_to_name.get(record['START_ID'], record['START_ID'])
                    params = {'gene': gene_name}
                elif task_id in ['R-COMPARE-RISK']:
                    # Two-gene comparison
                    records = random.sample(gd_sample, 2)
                    gene1 = gene_id_to_name.get(records[0]['START_ID'], records[0]['START_ID'])
                    gene2 = gene_id_to_name.get(records[1]['START_ID'], records[1]['START_ID'])
                    disease_info = next((d for d in diseases if d['ID'] == records[0]['END_ID']), None)
                    params = {
                        'gene1': gene1,
                        'gene2': gene2,
                        'disease': disease_info['name'] if disease_info else records[0]['END_ID'],
                        'disease_id': records[0]['END_ID']
                    }
                elif task_id == 'M-SHARED-PATHWAY':
                    # Two diseases
                    disease_samples = random.sample(diseases, 2)
                    params = {
                        'disease1': disease_samples[0]['name'],
                        'disease2': disease_samples[1]['name']
                    }
                else:
                    # Gene-disease questions
                    record = random.choice(gd_sample)
                    gene_name = gene_id_to_name.get(record['START_ID'], record['START_ID'])
                    disease_info = next((d for d in diseases if d['ID'] == record['END_ID']), None)
                    params = {
                        'gene': gene_name,
                        'disease': disease_info['name'] if disease_info else record['END_ID'],
                        'disease_id': record['END_ID']
                    }

                # Generate question
                question_template = random.choice(template['questions'])
                question = question_template.format(**params)
                cypher = template['cypher'].format(**params)

                q = {
                    'question': question,
                    'cypher': cypher.strip(),
                    'task_id': task_id,
                    'taxonomy': template['taxonomy'],
                    'type': template['type'],
                    'answer_key': template['answer_key'],
                    'parameters': params,
                    'category': 'knowledge'
                }
                task_questions.append(q)

            except Exception as e:
                continue

        questions.extend(task_questions[:n_per_task])

    return questions


def generate_reasoning_questions(kg_data: Dict, n_per_task: int = 50) -> List[Dict]:
    """Generate reasoning evaluation questions."""
    questions = []

    gene_disease = kg_data['gene_disease']
    diseases = kg_data['diseases']
    gene_id_to_name = kg_data['gene_id_to_name']

    # Sample gene-disease pairs with good scores
    gd_sample = [r for r in gene_disease if r.get('risk_score', 0) > 0.3]
    gd_sample = random.sample(gd_sample, min(n_per_task * 10, len(gd_sample)))

    for task_id, template in REASONING_TEMPLATES.items():
        print(f"  Generating {task_id} reasoning questions...")
        task_questions = []

        for i in range(n_per_task):
            try:
                if task_id in ['R2']:
                    # Two-gene comparison
                    records = random.sample(gd_sample, 2)
                    gene1 = gene_id_to_name.get(records[0]['START_ID'], records[0]['START_ID'])
                    gene2 = gene_id_to_name.get(records[1]['START_ID'], records[1]['START_ID'])
                    disease_info = next((d for d in diseases if d['ID'] == records[0]['END_ID']), None)
                    params = {
                        'gene1': gene1,
                        'gene2': gene2,
                        'disease': disease_info['name'] if disease_info else records[0]['END_ID'],
                        'disease_id': records[0]['END_ID']
                    }
                elif task_id in ['S2']:
                    record = random.choice(gd_sample)
                    disease_info = next((d for d in diseases if d['ID'] == record['END_ID']), None)
                    params = {
                        'disease': disease_info['name'] if disease_info else record['END_ID'],
                        'disease_id': record['END_ID'],
                        'threshold': random.choice([3, 5, 10])
                    }
                elif task_id in ['S3']:
                    disease_info = random.choice(diseases)
                    params = {
                        'disease': disease_info['name'],
                        'disease_id': disease_info['ID']
                    }
                elif task_id in ['R3']:
                    record = random.choice(gd_sample)
                    gene_name = gene_id_to_name.get(record['START_ID'], record['START_ID'])
                    disease_info = next((d for d in diseases if d['ID'] == record['END_ID']), None)
                    params = {
                        'gene': gene_name,
                        'disease': disease_info['name'] if disease_info else record['END_ID'],
                        'disease_id': record['END_ID'],
                        'threshold': random.choice([0.3, 0.5, 0.7])
                    }
                else:
                    record = random.choice(gd_sample)
                    gene_name = gene_id_to_name.get(record['START_ID'], record['START_ID'])
                    disease_info = next((d for d in diseases if d['ID'] == record['END_ID']), None)
                    params = {
                        'gene': gene_name,
                        'disease': disease_info['name'] if disease_info else record['END_ID'],
                        'disease_id': record['END_ID']
                    }

                # Generate question
                question_template = random.choice(template['questions'])
                question = question_template.format(**params)
                cypher = template['cypher'].format(**params)

                q = {
                    'question': question,
                    'cypher': cypher.strip(),
                    'task_id': task_id,
                    'taxonomy': template['taxonomy'],
                    'reasoning_type': template['reasoning_type'],
                    'evaluation_focus': template['evaluation_focus'],
                    'answer_format': template['answer_format'],
                    'parameters': params,
                    'category': 'reasoning'
                }

                if 'options' in template:
                    q['options'] = template['options']
                if 'correct_answer' in template:
                    q['correct_answer'] = template['correct_answer']

                task_questions.append(q)

            except Exception as e:
                continue

        questions.extend(task_questions[:n_per_task])

    return questions


def split_dev_test(questions: List[Dict], dev_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict]]:
    """Split questions into dev and test sets."""
    random.shuffle(questions)

    # Group by task_id
    by_task = {}
    for q in questions:
        task_id = q['task_id']
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(q)

    dev = []
    test = []

    for task_id, task_questions in by_task.items():
        n_dev = max(1, int(len(task_questions) * dev_ratio))
        dev.extend(task_questions[:n_dev])
        test.extend(task_questions[n_dev:])

    random.shuffle(dev)
    random.shuffle(test)

    return dev, test


def save_questions(questions: List[Dict], filepath: Path):
    """Save questions to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(questions, f, indent=2)
    print(f"  Saved {len(questions)} questions to {filepath}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("CKG_Bench Question Generator")
    print("="*70)
    print(f"Start time: {datetime.now()}")

    # Load KG data
    kg_data = load_kg_data()

    # Generate knowledge questions
    print("\nGenerating knowledge-based questions...")
    knowledge_questions = generate_knowledge_questions(kg_data, n_per_task=40)
    print(f"Generated {len(knowledge_questions)} knowledge questions")

    # Generate reasoning questions
    print("\nGenerating reasoning questions...")
    reasoning_questions = generate_reasoning_questions(kg_data, n_per_task=40)
    print(f"Generated {len(reasoning_questions)} reasoning questions")

    # Split into dev/test
    print("\nSplitting into dev/test...")
    knowledge_dev, knowledge_test = split_dev_test(knowledge_questions)
    reasoning_dev, reasoning_test = split_dev_test(reasoning_questions)

    # Save knowledge questions
    print("\nSaving knowledge questions...")
    for taxonomy in ['S', 'R', 'C', 'M']:
        tax_dev = [q for q in knowledge_dev if q['taxonomy'] == taxonomy]
        tax_test = [q for q in knowledge_test if q['taxonomy'] == taxonomy]
        save_questions(tax_dev, DATA_DIR / 'knowledge' / f'{taxonomy}_knowledge_dev.json')
        save_questions(tax_test, DATA_DIR / 'knowledge' / f'{taxonomy}_knowledge_test.json')

    # Save reasoning questions
    print("\nSaving reasoning questions...")
    for taxonomy in ['S', 'R', 'C', 'M']:
        tax_dev = [q for q in reasoning_dev if q['taxonomy'] == taxonomy]
        tax_test = [q for q in reasoning_test if q['taxonomy'] == taxonomy]
        save_questions(tax_dev, DATA_DIR / 'reasoning' / f'{taxonomy}_reasoning_dev.json')
        save_questions(tax_test, DATA_DIR / 'reasoning' / f'{taxonomy}_reasoning_test.json')

    # Save combined files
    print("\nSaving combined files...")
    all_dev = knowledge_dev + reasoning_dev
    all_test = knowledge_test + reasoning_test
    save_questions(all_dev, DATA_DIR / 'combined_dev.json')
    save_questions(all_test, DATA_DIR / 'combined_test.json')

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Knowledge questions: {len(knowledge_questions)}")
    print(f"  Dev: {len(knowledge_dev)}, Test: {len(knowledge_test)}")
    print(f"Reasoning questions: {len(reasoning_questions)}")
    print(f"  Dev: {len(reasoning_dev)}, Test: {len(reasoning_test)}")
    print(f"Total: {len(all_dev) + len(all_test)}")
    print(f"  Dev: {len(all_dev)}, Test: {len(all_test)}")

    print(f"\nEnd time: {datetime.now()}")
    print("="*70)


if __name__ == "__main__":
    main()
