#!/usr/bin/env python3
"""
Regenerate BioResonKGBench questions to match actual KG schema.
Fixes the disease ID mismatch (MeSH -> DOID) and ensures all queries return valid data.
"""

import json
import yaml
import random
from pathlib import Path
from neo4j import GraphDatabase
from typing import List, Dict, Tuple

SCRIPT_DIR = Path(__file__).parent
random.seed(42)  # For reproducibility

def get_driver():
    with open(SCRIPT_DIR / 'config' / 'kg_config.yml') as f:
        cfg = yaml.safe_load(f)
    return GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

def get_valid_entities(driver) -> Dict:
    """Extract valid entities from the KG for question generation."""
    entities = {}

    with driver.session() as s:
        # Get diseases with gene associations
        r = s.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[r:INCREASES_RISK_OF]->(d:Disease)
            WITH d, count(DISTINCT g) as gene_count, avg(r.risk_score) as avg_risk
            WHERE gene_count >= 5
            RETURN d.id as id, d.name as name, gene_count, avg_risk
            ORDER BY gene_count DESC
            LIMIT 100
        ''')
        entities['diseases'] = [dict(rec) for rec in r]

        # Get genes with disease and protein associations
        r = s.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[:INCREASES_RISK_OF]->(d:Disease)
            MATCH (g)-[:TRANSLATED_INTO]->(p:Protein)
            WITH g, collect(DISTINCT d.name)[0..3] as diseases, count(DISTINCT p) as protein_count
            WHERE protein_count >= 1
            RETURN g.id as id, g.name as name, diseases, protein_count
            ORDER BY protein_count DESC
            LIMIT 200
        ''')
        entities['genes'] = [dict(rec) for rec in r]

        # Get SNPs with gene and disease associations
        r = s.run('''
            MATCH (s:SNP)-[:MAPS_TO]->(g:Gene {node_type: "risk_gene"})
            MATCH (s)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WITH s, g, d, r.pvalue as pvalue, r.causal_score as causal_score
            WHERE pvalue IS NOT NULL OR causal_score IS NOT NULL
            RETURN s.id as id, g.id as gene, d.name as disease, d.id as disease_id,
                   pvalue, causal_score
            ORDER BY causal_score DESC
            LIMIT 200
        ''')
        entities['snps'] = [dict(rec) for rec in r]

        # Get genes with GO annotations
        r = s.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ASSOCIATED_WITH]->(bp:Biological_process)
            WITH g, collect(DISTINCT bp.name)[0..3] as processes
            WHERE size(processes) >= 1
            RETURN g.id as gene, processes
            LIMIT 100
        ''')
        entities['gene_go'] = [dict(rec) for rec in r]

        # Get genes with pathway annotations
        r = s.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
            WITH g, collect(DISTINCT pw.name)[0..3] as pathways
            WHERE size(pathways) >= 1
            RETURN g.id as gene, pathways
            LIMIT 100
        ''')
        entities['gene_pathway'] = [dict(rec) for rec in r]

    return entities


def generate_structure_questions(entities: Dict) -> List[Dict]:
    """Generate Structure (S) taxonomy questions."""
    questions = []

    # S-DISEASE-GENES: What genes are risk factors for [Disease]?
    for d in random.sample(entities['diseases'], min(10, len(entities['diseases']))):
        q = {
            "question": f"What genes are risk factors for {d['name']}?",
            "cypher": f'''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{d['id']}"}})
            RETURN g.id AS gene, r.risk_score AS risk_score
            ORDER BY r.risk_score DESC
            LIMIT 20''',
            "task_id": "S-DISEASE-GENES",
            "taxonomy": "S",
            "type": "one-hop",
            "answer_key": "gene",
            "parameters": {"disease": d['name'], "disease_id": d['id']},
            "category": "knowledge"
        }
        questions.append(q)

    # S-GENE-DISEASE: What diseases are associated with gene [Gene]?
    for g in random.sample(entities['genes'], min(10, len(entities['genes']))):
        q = {
            "question": f"What diseases are associated with increased risk from gene {g['id']}?",
            "cypher": f'''MATCH (g:Gene {{id: "{g['id']}"}})-[r:INCREASES_RISK_OF]->(d:Disease)
            RETURN d.name AS disease, r.risk_score AS risk_score
            ORDER BY r.risk_score DESC''',
            "task_id": "S-GENE-DISEASE",
            "taxonomy": "S",
            "type": "one-hop",
            "answer_key": "disease",
            "parameters": {"gene": g['id']},
            "category": "knowledge"
        }
        questions.append(q)

    # S-SNP-GENE: Which gene is linked to SNP [rs_id]?
    for snp in random.sample(entities['snps'], min(10, len(entities['snps']))):
        q = {
            "question": f"Which gene is linked to SNP {snp['id']}?",
            "cypher": f'''MATCH (s:SNP {{id: "{snp['id']}"}})-[:MAPS_TO]->(g:Gene)
            RETURN g.id AS gene, g.name AS gene_name''',
            "task_id": "S-SNP-GENE",
            "taxonomy": "S",
            "type": "one-hop",
            "answer_key": "gene",
            "parameters": {"snp": snp['id']},
            "category": "knowledge"
        }
        questions.append(q)

    # S-SNP-DISEASE: Which diseases does SNP [rs_id] have a putative causal effect on?
    for snp in random.sample(entities['snps'], min(10, len(entities['snps']))):
        q = {
            "question": f"Which diseases does SNP {snp['id']} have a putative causal effect on?",
            "cypher": f'''MATCH (s:SNP {{id: "{snp['id']}"}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            RETURN d.name AS disease, r.causal_score AS causal_score''',
            "task_id": "S-SNP-DISEASE",
            "taxonomy": "S",
            "type": "one-hop",
            "answer_key": "disease",
            "parameters": {"snp": snp['id']},
            "category": "knowledge"
        }
        questions.append(q)

    return questions


def generate_risk_questions(entities: Dict) -> List[Dict]:
    """Generate Risk (R) taxonomy questions."""
    questions = []

    # R-PVALUE: What is the p-value for SNP-Disease association?
    snps_with_pvalue = [s for s in entities['snps'] if s.get('pvalue')]
    for snp in random.sample(snps_with_pvalue, min(10, len(snps_with_pvalue))):
        q = {
            "question": f"How statistically significant is the association between SNP {snp['id']} and {snp['disease']}?",
            "cypher": f'''MATCH (s:SNP {{id: "{snp['id']}"}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{snp['disease_id']}"}})
            RETURN s.id AS snp, d.name AS disease, r.pvalue AS pvalue, r.causal_score AS causal_score''',
            "task_id": "R-PVALUE",
            "taxonomy": "R",
            "type": "one-hop",
            "answer_key": "pvalue",
            "parameters": {"snp": snp['id'], "disease": snp['disease'], "disease_id": snp['disease_id']},
            "category": "knowledge"
        }
        questions.append(q)

    # R-RISK-SCORE: What is the risk score for gene-disease?
    for g in random.sample(entities['genes'], min(10, len(entities['genes']))):
        if g['diseases']:
            disease = g['diseases'][0]
            q = {
                "question": f"What is the risk score for gene {g['id']} affecting {disease}?",
                "cypher": f'''MATCH (g:Gene {{id: "{g['id']}"}})-[r:INCREASES_RISK_OF]->(d:Disease)
                WHERE d.name = "{disease}"
                RETURN g.id AS gene, d.name AS disease, r.risk_score AS risk_score''',
                "task_id": "R-RISK-SCORE",
                "taxonomy": "R",
                "type": "one-hop",
                "answer_key": "risk_score",
                "parameters": {"gene": g['id'], "disease": disease},
                "category": "knowledge"
            }
            questions.append(q)

    # R-TOP-RISK: Which gene has the highest risk score for [Disease]?
    for d in random.sample(entities['diseases'], min(10, len(entities['diseases']))):
        q = {
            "question": f"Which gene has the highest risk score for {d['name']}?",
            "cypher": f'''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{d['id']}"}})
            RETURN g.id AS gene, r.risk_score AS risk_score
            ORDER BY r.risk_score DESC
            LIMIT 1''',
            "task_id": "R-TOP-RISK",
            "taxonomy": "R",
            "type": "ranking",
            "answer_key": "gene",
            "parameters": {"disease": d['name'], "disease_id": d['id']},
            "category": "knowledge"
        }
        questions.append(q)

    return questions


def generate_causal_questions(entities: Dict) -> List[Dict]:
    """Generate Causal (C) taxonomy questions."""
    questions = []

    # C-EVIDENCE-LEVEL: What is the evidence level for gene-disease?
    for g in random.sample(entities['genes'], min(10, len(entities['genes']))):
        if g['diseases']:
            disease = g['diseases'][0]
            q = {
                "question": f"What is the evidence level for gene {g['id']} affecting {disease}?",
                "cypher": f'''MATCH (g:Gene {{id: "{g['id']}"}})-[r:INCREASES_RISK_OF]->(d:Disease)
                WHERE d.name = "{disease}"
                RETURN g.id AS gene, d.name AS disease, r.evidence_score AS evidence_score,
                       CASE
                           WHEN r.evidence_score >= 0.8 THEN 'very_strong'
                           WHEN r.evidence_score >= 0.6 THEN 'strong'
                           WHEN r.evidence_score >= 0.4 THEN 'moderate'
                           WHEN r.evidence_score >= 0.2 THEN 'suggestive'
                           ELSE 'weak'
                       END AS evidence_level''',
                "task_id": "C-EVIDENCE-LEVEL",
                "taxonomy": "C",
                "type": "classification",
                "answer_key": "evidence_level",
                "parameters": {"gene": g['id'], "disease": disease},
                "category": "knowledge"
            }
            questions.append(q)

    # C-CAUSAL-SCORE: What is the causal score for SNP-Disease?
    snps_with_score = [s for s in entities['snps'] if s.get('causal_score')]
    for snp in random.sample(snps_with_score, min(10, len(snps_with_score))):
        q = {
            "question": f"What is the causal score for SNP {snp['id']} on {snp['disease']}?",
            "cypher": f'''MATCH (s:SNP {{id: "{snp['id']}"}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{snp['disease_id']}"}})
            RETURN s.id AS snp, d.name AS disease, r.causal_score AS causal_score''',
            "task_id": "C-CAUSAL-SCORE",
            "taxonomy": "C",
            "type": "one-hop",
            "answer_key": "causal_score",
            "parameters": {"snp": snp['id'], "disease": snp['disease'], "disease_id": snp['disease_id']},
            "category": "knowledge"
        }
        questions.append(q)

    # C-TOP-CAUSAL: Which SNP has the strongest causal effect on [Disease]?
    for d in random.sample(entities['diseases'], min(10, len(entities['diseases']))):
        q = {
            "question": f"Which SNP has the strongest causal effect on {d['name']}?",
            "cypher": f'''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{d['id']}"}})
            RETURN s.id AS snp, r.causal_score AS causal_score
            ORDER BY r.causal_score DESC
            LIMIT 1''',
            "task_id": "C-TOP-CAUSAL",
            "taxonomy": "C",
            "type": "ranking",
            "answer_key": "snp",
            "parameters": {"disease": d['name'], "disease_id": d['id']},
            "category": "knowledge"
        }
        questions.append(q)

    # C-MR-VALIDATED: Is there MR evidence for gene causing disease?
    for g in random.sample(entities['genes'], min(10, len(entities['genes']))):
        if g['diseases']:
            disease = g['diseases'][0]
            q = {
                "question": f"Is there Mendelian Randomization evidence for gene {g['id']} causing {disease}?",
                "cypher": f'''MATCH (g:Gene {{id: "{g['id']}"}})-[r:INCREASES_RISK_OF]->(d:Disease)
                WHERE d.name = "{disease}"
                RETURN g.id AS gene, d.name AS disease, r.mr_score AS mr_score,
                       CASE WHEN r.mr_score >= 0.5 THEN 'Yes' ELSE 'Weak/No' END AS mr_validated''',
                "task_id": "C-MR-VALIDATED",
                "taxonomy": "C",
                "type": "classification",
                "answer_key": "mr_validated",
                "parameters": {"gene": g['id'], "disease": disease},
                "category": "knowledge"
            }
            questions.append(q)

    return questions


def generate_mechanism_questions(entities: Dict) -> List[Dict]:
    """Generate Mechanism (M) taxonomy questions."""
    questions = []

    # M-PROTEIN: What protein does gene translate into?
    for g in random.sample(entities['genes'], min(10, len(entities['genes']))):
        q = {
            "question": f"What protein does gene {g['id']} translate into?",
            "cypher": f'''MATCH (g:Gene {{id: "{g['id']}"}})-[:TRANSLATED_INTO]->(p:Protein)
            RETURN g.id AS gene, p.id AS protein_id, p.name AS protein_name''',
            "task_id": "M-PROTEIN",
            "taxonomy": "M",
            "type": "one-hop",
            "answer_key": "protein_name",
            "parameters": {"gene": g['id']},
            "category": "knowledge"
        }
        questions.append(q)

    # M-GO-PROCESS: What biological processes involve gene?
    for g_go in random.sample(entities['gene_go'], min(10, len(entities['gene_go']))):
        q = {
            "question": f"Which GO biological processes involve gene {g_go['gene']}?",
            "cypher": f'''MATCH (g:Gene {{id: "{g_go['gene']}"}})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ASSOCIATED_WITH]->(bp:Biological_process)
            RETURN DISTINCT bp.name AS process, bp.id AS process_id
            LIMIT 10''',
            "task_id": "M-GO-PROCESS",
            "taxonomy": "M",
            "type": "one-hop",
            "answer_key": "process",
            "parameters": {"gene": g_go['gene']},
            "category": "knowledge"
        }
        questions.append(q)

    # M-PATHWAY: What pathways involve gene?
    for g_pw in random.sample(entities['gene_pathway'], min(10, len(entities['gene_pathway']))):
        q = {
            "question": f"What pathways involve gene {g_pw['gene']}?",
            "cypher": f'''MATCH (g:Gene {{id: "{g_pw['gene']}"}})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
            RETURN DISTINCT pw.name AS pathway
            LIMIT 10''',
            "task_id": "M-PATHWAY",
            "taxonomy": "M",
            "type": "one-hop",
            "answer_key": "pathway",
            "parameters": {"gene": g_pw['gene']},
            "category": "knowledge"
        }
        questions.append(q)

    return questions


def generate_reasoning_questions(entities: Dict) -> List[Dict]:
    """Generate reasoning questions (multi-hop)."""
    questions = []

    # Reasoning: SNP -> Gene -> Disease path
    for snp in random.sample(entities['snps'], min(10, len(entities['snps']))):
        q = {
            "question": f"Through which gene does SNP {snp['id']} influence disease risk?",
            "cypher": f'''MATCH (s:SNP {{id: "{snp['id']}"}})-[:MAPS_TO]->(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            RETURN s.id AS snp, g.id AS gene, d.name AS disease''',
            "task_id": "S-REASONING-PATH",
            "taxonomy": "S",
            "type": "multi-hop",
            "answer_key": "gene",
            "parameters": {"snp": snp['id']},
            "category": "reasoning"
        }
        questions.append(q)

    # Reasoning: Gene -> Protein -> Function
    for g_go in random.sample(entities['gene_go'], min(10, len(entities['gene_go']))):
        q = {
            "question": f"What biological functions are performed by the protein encoded by gene {g_go['gene']}?",
            "cypher": f'''MATCH (g:Gene {{id: "{g_go['gene']}"}})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ASSOCIATED_WITH]->(bp:Biological_process)
            RETURN g.id AS gene, p.name AS protein, collect(DISTINCT bp.name)[0..5] AS processes''',
            "task_id": "M-REASONING-FUNCTION",
            "taxonomy": "M",
            "type": "multi-hop",
            "answer_key": "processes",
            "parameters": {"gene": g_go['gene']},
            "category": "reasoning"
        }
        questions.append(q)

    # Reasoning: Compare risk between diseases
    disease_pairs = []
    for i, d1 in enumerate(entities['diseases'][:20]):
        for d2 in entities['diseases'][i+1:21]:
            disease_pairs.append((d1, d2))

    for d1, d2 in random.sample(disease_pairs, min(10, len(disease_pairs))):
        q = {
            "question": f"Which genes are risk factors for both {d1['name']} and {d2['name']}?",
            "cypher": f'''MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d1:Disease {{id: "{d1['id']}"}})
            MATCH (g)-[:INCREASES_RISK_OF]->(d2:Disease {{id: "{d2['id']}"}})
            RETURN g.id AS gene, d1.name AS disease1, d2.name AS disease2''',
            "task_id": "R-REASONING-SHARED",
            "taxonomy": "R",
            "type": "multi-hop",
            "answer_key": "gene",
            "parameters": {"disease1": d1['name'], "disease2": d2['name']},
            "category": "reasoning"
        }
        questions.append(q)

    return questions


def validate_questions(questions: List[Dict], driver) -> Tuple[List[Dict], List[Dict]]:
    """Validate that all questions return data."""
    valid = []
    invalid = []

    with driver.session() as s:
        for q in questions:
            try:
                r = s.run(q['cypher'])
                records = list(r)
                if len(records) > 0:
                    valid.append(q)
                else:
                    invalid.append(q)
            except Exception as e:
                q['error'] = str(e)
                invalid.append(q)

    return valid, invalid


def main():
    print('='*80)
    print('REGENERATING BIORESONKGBENCH QUESTIONS')
    print('='*80)

    driver = get_driver()

    # Step 1: Extract valid entities
    print('\n1. Extracting valid entities from KG...')
    entities = get_valid_entities(driver)
    print(f'   Diseases: {len(entities["diseases"])}')
    print(f'   Genes: {len(entities["genes"])}')
    print(f'   SNPs: {len(entities["snps"])}')
    print(f'   Gene-GO: {len(entities["gene_go"])}')
    print(f'   Gene-Pathway: {len(entities["gene_pathway"])}')

    # Step 2: Generate questions
    print('\n2. Generating questions...')
    all_questions = []

    s_questions = generate_structure_questions(entities)
    print(f'   Structure (S): {len(s_questions)}')
    all_questions.extend(s_questions)

    r_questions = generate_risk_questions(entities)
    print(f'   Risk (R): {len(r_questions)}')
    all_questions.extend(r_questions)

    c_questions = generate_causal_questions(entities)
    print(f'   Causal (C): {len(c_questions)}')
    all_questions.extend(c_questions)

    m_questions = generate_mechanism_questions(entities)
    print(f'   Mechanism (M): {len(m_questions)}')
    all_questions.extend(m_questions)

    reasoning_questions = generate_reasoning_questions(entities)
    print(f'   Reasoning: {len(reasoning_questions)}')
    all_questions.extend(reasoning_questions)

    print(f'\n   Total generated: {len(all_questions)}')

    # Step 3: Validate questions
    print('\n3. Validating questions...')
    valid, invalid = validate_questions(all_questions, driver)
    print(f'   Valid: {len(valid)}')
    print(f'   Invalid: {len(invalid)}')

    if invalid:
        print(f'\n   Sample invalid questions:')
        for q in invalid[:3]:
            print(f'     - {q["task_id"]}: {q.get("error", "No results")}')

    # Step 4: Split into dev and test
    print('\n4. Splitting into dev/test sets...')
    random.shuffle(valid)
    split_idx = int(len(valid) * 0.15)  # 15% dev, 85% test
    dev_questions = valid[:split_idx]
    test_questions = valid[split_idx:]

    print(f'   Dev set: {len(dev_questions)}')
    print(f'   Test set: {len(test_questions)}')

    # Step 5: Save questions
    print('\n5. Saving questions...')

    # Backup original files
    import shutil
    for fname in ['combined_dev.json', 'combined_test.json']:
        src = SCRIPT_DIR / 'data' / fname
        dst = SCRIPT_DIR / 'data' / f'{fname}.backup'
        if src.exists():
            shutil.copy(src, dst)
            print(f'   Backed up {fname}')

    # Save new files
    with open(SCRIPT_DIR / 'data' / 'combined_dev.json', 'w') as f:
        json.dump(dev_questions, f, indent=2)
    print(f'   Saved combined_dev.json ({len(dev_questions)} questions)')

    with open(SCRIPT_DIR / 'data' / 'combined_test.json', 'w') as f:
        json.dump(test_questions, f, indent=2)
    print(f'   Saved combined_test.json ({len(test_questions)} questions)')

    driver.close()

    # Summary
    print('\n' + '='*80)
    print('REGENERATION COMPLETE')
    print('='*80)
    print(f'\nTotal valid questions: {len(valid)}')
    print(f'Dev set: {len(dev_questions)}')
    print(f'Test set: {len(test_questions)}')

    # Distribution summary
    from collections import Counter
    taxonomy_dist = Counter(q['taxonomy'] for q in valid)
    category_dist = Counter(q['category'] for q in valid)

    print(f'\nTaxonomy distribution:')
    for t, c in taxonomy_dist.most_common():
        print(f'  {t}: {c}')

    print(f'\nCategory distribution:')
    for cat, c in category_dist.most_common():
        print(f'  {cat}: {c}')


if __name__ == "__main__":
    main()
