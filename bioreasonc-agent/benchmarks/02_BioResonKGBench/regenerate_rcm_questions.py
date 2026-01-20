#!/usr/bin/env python3
"""
Hybrid Regeneration: Keep valid S questions, regenerate R/C/M with valid KG relationships.
Ensures every question has an answer in the KG.
"""

import json
import yaml
import random
from pathlib import Path
from neo4j import GraphDatabase
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).parent
random.seed(42)

def load_config():
    with open(SCRIPT_DIR / 'config' / 'kg_config.yml') as f:
        return yaml.safe_load(f)

def get_valid_s_questions(driver) -> Dict[str, List[dict]]:
    """Extract valid S questions from existing dataset."""
    valid_s = {
        'knowledge_dev': [],
        'knowledge_test': [],
        'reasoning_dev': [],
        'reasoning_test': []
    }

    with driver.session() as session:
        for track in ['knowledge', 'reasoning']:
            for split in ['dev', 'test']:
                file_path = SCRIPT_DIR / 'data' / track / f'S_{track}_{split}.json'
                with open(file_path) as f:
                    questions = json.load(f)

                for q in questions:
                    cypher = q.get('cypher', '')
                    if not cypher:
                        continue
                    try:
                        result = session.run(cypher)
                        data = list(result)
                        if len(data) > 0:
                            valid_s[f'{track}_{split}'].append(q)
                    except:
                        pass

    return valid_s

def get_kg_relationships(driver) -> Dict:
    """Extract actual relationships from KG for R/C/M question generation."""
    data = {
        'gene_disease': [],      # For R questions
        'snp_disease': [],       # For R/C questions
        'gene_protein': [],      # For M questions
        'protein_go': [],        # For M questions
        'protein_pathway': [],   # For M questions
        'protein_tissue': [],    # For M questions
    }

    with driver.session() as session:
        # Gene-Disease relationships (for R questions)
        print("  Fetching gene-disease relationships...")
        result = session.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE r.risk_score IS NOT NULL AND r.evidence_score IS NOT NULL
            RETURN g.id AS gene, d.id AS disease_id, d.name AS disease_name,
                   r.risk_score AS risk_score, r.evidence_score AS evidence_score
            ORDER BY r.risk_score DESC
            LIMIT 500
        ''')
        data['gene_disease'] = [dict(rec) for rec in result]

        # SNP-Disease relationships (for R/C questions)
        print("  Fetching SNP-disease relationships...")
        result = session.run('''
            MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WHERE r.pvalue IS NOT NULL AND r.causal_score IS NOT NULL
            RETURN s.id AS snp, d.id AS disease_id, d.name AS disease_name,
                   r.pvalue AS pvalue, r.causal_score AS causal_score,
                   r.beta AS beta
            ORDER BY r.causal_score DESC
            LIMIT 500
        ''')
        data['snp_disease'] = [dict(rec) for rec in result]

        # Gene-Protein relationships (for M questions)
        print("  Fetching gene-protein relationships...")
        result = session.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein)
            RETURN g.id AS gene, p.name AS protein, p.id AS protein_id
            LIMIT 300
        ''')
        data['gene_protein'] = [dict(rec) for rec in result]

        # Protein-GO relationships (for M questions)
        print("  Fetching protein-GO relationships...")
        result = session.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ASSOCIATED_WITH]->(bp:Biological_process)
            RETURN g.id AS gene, p.name AS protein, collect(DISTINCT bp.name)[0..5] AS processes
            LIMIT 300
        ''')
        data['protein_go'] = [dict(rec) for rec in result]

        # Protein-Pathway relationships (for M questions)
        print("  Fetching protein-pathway relationships...")
        result = session.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
            RETURN g.id AS gene, p.name AS protein, collect(DISTINCT pw.name)[0..5] AS pathways
            LIMIT 300
        ''')
        data['protein_pathway'] = [dict(rec) for rec in result]

        # Protein-Tissue relationships (for M questions)
        print("  Fetching protein-tissue relationships...")
        result = session.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ASSOCIATED_WITH]->(t:Tissue)
            RETURN g.id AS gene, p.name AS protein, collect(DISTINCT t.name)[0..5] AS tissues
            LIMIT 300
        ''')
        data['protein_tissue'] = [dict(rec) for rec in result]

    return data

def generate_r_questions(kg_data: Dict, count: int, category: str) -> List[dict]:
    """Generate R (Risk) questions using actual KG relationships."""
    questions = []
    gene_disease = kg_data['gene_disease']
    snp_disease = kg_data['snp_disease']

    # Shuffle for variety
    random.shuffle(gene_disease)
    random.shuffle(snp_disease)

    templates = [
        # Gene risk score questions
        {
            'template': "What is the risk score for gene {gene} affecting {disease}?",
            'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN r.risk_score AS risk_score''',
            'task_id': 'R-RISK-SCORE',
            'answer_key': 'risk_score',
            'data_source': 'gene_disease'
        },
        # SNP pvalue questions
        {
            'template': "What is the p-value for SNP {snp}'s association with {disease}?",
            'cypher': '''MATCH (s:SNP {{id: "{snp}"}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            RETURN r.pvalue AS pvalue''',
            'task_id': 'R-PVALUE',
            'answer_key': 'pvalue',
            'data_source': 'snp_disease'
        },
        # Top risk gene for disease
        {
            'template': "Which gene has the highest risk score for {disease}?",
            'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN g.id AS gene, r.risk_score AS risk_score
            ORDER BY r.risk_score DESC LIMIT 1''',
            'task_id': 'R-TOP-RISK',
            'answer_key': 'gene',
            'data_source': 'gene_disease'
        },
        # Gene evidence score
        {
            'template': "What is the evidence score for gene {gene}'s association with {disease}?",
            'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN r.evidence_score AS evidence_score''',
            'task_id': 'R-EVIDENCE',
            'answer_key': 'evidence_score',
            'data_source': 'gene_disease'
        },
    ]

    # Add reasoning templates if needed
    if category == 'reasoning':
        templates.append({
            'template': "Which genes are risk factors for both {disease1} and {disease2}?",
            'cypher': '''MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d1:Disease {{id: "{disease_id1}"}})
            MATCH (g)-[:INCREASES_RISK_OF]->(d2:Disease {{id: "{disease_id2}"}})
            RETURN DISTINCT g.id AS gene LIMIT 10''',
            'task_id': 'R-REASONING-SHARED',
            'answer_key': 'gene',
            'data_source': 'gene_disease_pair',
            'type': 'multi-hop'
        })

    idx = 0
    template_idx = 0
    used_combinations = set()

    while len(questions) < count and idx < 1000:
        template = templates[template_idx % len(templates)]
        template_idx += 1

        if template['data_source'] == 'gene_disease':
            if idx >= len(gene_disease):
                idx = 0
                random.shuffle(gene_disease)
            rec = gene_disease[idx % len(gene_disease)]
            combo_key = f"{template['task_id']}_{rec['gene']}_{rec['disease_id']}"
            if combo_key in used_combinations:
                idx += 1
                continue
            used_combinations.add(combo_key)

            q = {
                'question': template['template'].format(
                    gene=rec['gene'],
                    disease=rec['disease_name']
                ),
                'cypher': template['cypher'].format(
                    gene=rec['gene'],
                    disease_id=rec['disease_id']
                ),
                'task_id': template['task_id'],
                'taxonomy': 'R',
                'type': template.get('type', 'one-hop'),
                'answer_key': template['answer_key'],
                'parameters': {
                    'gene': rec['gene'],
                    'disease': rec['disease_name'],
                    'disease_id': rec['disease_id']
                },
                'category': category
            }
            questions.append(q)

        elif template['data_source'] == 'snp_disease':
            if idx >= len(snp_disease):
                idx = 0
                random.shuffle(snp_disease)
            rec = snp_disease[idx % len(snp_disease)]
            combo_key = f"{template['task_id']}_{rec['snp']}_{rec['disease_id']}"
            if combo_key in used_combinations:
                idx += 1
                continue
            used_combinations.add(combo_key)

            q = {
                'question': template['template'].format(
                    snp=rec['snp'],
                    disease=rec['disease_name']
                ),
                'cypher': template['cypher'].format(
                    snp=rec['snp'],
                    disease_id=rec['disease_id']
                ),
                'task_id': template['task_id'],
                'taxonomy': 'R',
                'type': 'one-hop',
                'answer_key': template['answer_key'],
                'parameters': {
                    'snp': rec['snp'],
                    'disease': rec['disease_name'],
                    'disease_id': rec['disease_id']
                },
                'category': category
            }
            questions.append(q)

        elif template['data_source'] == 'gene_disease_pair':
            # For reasoning questions - find genes with multiple diseases
            diseases = list(set([r['disease_id'] for r in gene_disease[:100]]))
            if len(diseases) >= 2:
                d1, d2 = random.sample(diseases, 2)
                d1_name = next((r['disease_name'] for r in gene_disease if r['disease_id'] == d1), d1)
                d2_name = next((r['disease_name'] for r in gene_disease if r['disease_id'] == d2), d2)

                combo_key = f"{template['task_id']}_{d1}_{d2}"
                if combo_key not in used_combinations:
                    used_combinations.add(combo_key)
                    q = {
                        'question': template['template'].format(
                            disease1=d1_name,
                            disease2=d2_name
                        ),
                        'cypher': template['cypher'].format(
                            disease_id1=d1,
                            disease_id2=d2
                        ),
                        'task_id': template['task_id'],
                        'taxonomy': 'R',
                        'type': 'multi-hop',
                        'answer_key': template['answer_key'],
                        'parameters': {
                            'disease1': d1_name,
                            'disease2': d2_name
                        },
                        'category': category
                    }
                    questions.append(q)

        idx += 1

    return questions[:count]

def generate_c_questions(kg_data: Dict, count: int, category: str) -> List[dict]:
    """Generate C (Causal) questions using actual KG relationships."""
    questions = []
    snp_disease = kg_data['snp_disease']
    gene_disease = kg_data['gene_disease']

    random.shuffle(snp_disease)
    random.shuffle(gene_disease)

    templates = [
        # Causal score
        {
            'template': "What is the causal score for SNP {snp}'s effect on {disease}?",
            'cypher': '''MATCH (s:SNP {{id: "{snp}"}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            RETURN r.causal_score AS causal_score''',
            'task_id': 'C-CAUSAL-SCORE',
            'answer_key': 'causal_score',
            'data_source': 'snp_disease'
        },
        # Evidence level
        {
            'template': "What is the evidence level for gene {gene} affecting {disease}?",
            'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN CASE
                WHEN r.evidence_score >= 0.8 THEN "very_strong"
                WHEN r.evidence_score >= 0.6 THEN "strong"
                WHEN r.evidence_score >= 0.4 THEN "moderate"
                WHEN r.evidence_score >= 0.2 THEN "suggestive"
                ELSE "weak" END AS evidence_level''',
            'task_id': 'C-EVIDENCE-LEVEL',
            'answer_key': 'evidence_level',
            'data_source': 'gene_disease'
        },
        # Top causal SNP
        {
            'template': "Which SNP has the strongest causal effect on {disease}?",
            'cypher': '''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            RETURN s.id AS snp, r.causal_score AS causal_score
            ORDER BY r.causal_score DESC LIMIT 1''',
            'task_id': 'C-TOP-CAUSAL',
            'answer_key': 'snp',
            'data_source': 'snp_disease'
        },
        # Beta effect
        {
            'template': "What is the effect size (beta) of SNP {snp} on {disease}?",
            'cypher': '''MATCH (s:SNP {{id: "{snp}"}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            RETURN r.beta AS beta''',
            'task_id': 'C-BETA',
            'answer_key': 'beta',
            'data_source': 'snp_disease'
        },
    ]

    if category == 'reasoning':
        templates.append({
            'template': "Is the causal evidence for {snp} affecting {disease} stronger than suggestive level?",
            'cypher': '''MATCH (s:SNP {{id: "{snp}"}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            RETURN CASE WHEN r.causal_score > 0.5 THEN "yes" ELSE "no" END AS answer''',
            'task_id': 'C-REASONING-STRENGTH',
            'answer_key': 'answer',
            'data_source': 'snp_disease',
            'type': 'reasoning'
        })

    idx = 0
    template_idx = 0
    used_combinations = set()

    while len(questions) < count and idx < 1000:
        template = templates[template_idx % len(templates)]
        template_idx += 1

        if template['data_source'] == 'snp_disease':
            if idx >= len(snp_disease):
                idx = 0
                random.shuffle(snp_disease)
            rec = snp_disease[idx % len(snp_disease)]
            combo_key = f"{template['task_id']}_{rec['snp']}_{rec['disease_id']}"
            if combo_key in used_combinations:
                idx += 1
                continue
            used_combinations.add(combo_key)

            q = {
                'question': template['template'].format(
                    snp=rec['snp'],
                    disease=rec['disease_name']
                ),
                'cypher': template['cypher'].format(
                    snp=rec['snp'],
                    disease_id=rec['disease_id']
                ),
                'task_id': template['task_id'],
                'taxonomy': 'C',
                'type': template.get('type', 'one-hop'),
                'answer_key': template['answer_key'],
                'parameters': {
                    'snp': rec['snp'],
                    'disease': rec['disease_name'],
                    'disease_id': rec['disease_id']
                },
                'category': category
            }
            questions.append(q)

        elif template['data_source'] == 'gene_disease':
            if idx >= len(gene_disease):
                idx = 0
                random.shuffle(gene_disease)
            rec = gene_disease[idx % len(gene_disease)]
            combo_key = f"{template['task_id']}_{rec['gene']}_{rec['disease_id']}"
            if combo_key in used_combinations:
                idx += 1
                continue
            used_combinations.add(combo_key)

            q = {
                'question': template['template'].format(
                    gene=rec['gene'],
                    disease=rec['disease_name']
                ),
                'cypher': template['cypher'].format(
                    gene=rec['gene'],
                    disease_id=rec['disease_id']
                ),
                'task_id': template['task_id'],
                'taxonomy': 'C',
                'type': 'one-hop',
                'answer_key': template['answer_key'],
                'parameters': {
                    'gene': rec['gene'],
                    'disease': rec['disease_name'],
                    'disease_id': rec['disease_id']
                },
                'category': category
            }
            questions.append(q)

        idx += 1

    return questions[:count]

def generate_m_questions(kg_data: Dict, count: int, category: str) -> List[dict]:
    """Generate M (Mechanism) questions using actual KG relationships."""
    questions = []
    gene_protein = kg_data['gene_protein']
    protein_go = kg_data['protein_go']
    protein_pathway = kg_data['protein_pathway']
    protein_tissue = kg_data['protein_tissue']

    random.shuffle(gene_protein)
    random.shuffle(protein_go)
    random.shuffle(protein_pathway)

    templates = [
        # Gene to protein
        {
            'template': "What protein does gene {gene} translate into?",
            'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[:TRANSLATED_INTO]->(p:Protein)
            RETURN p.name AS protein''',
            'task_id': 'M-PROTEIN',
            'answer_key': 'protein',
            'data_source': 'gene_protein'
        },
        # GO processes
        {
            'template': "What biological processes involve the protein encoded by gene {gene}?",
            'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ASSOCIATED_WITH]->(bp:Biological_process)
            RETURN collect(DISTINCT bp.name)[0..5] AS processes''',
            'task_id': 'M-GO-PROCESS',
            'answer_key': 'processes',
            'data_source': 'protein_go'
        },
        # Pathways
        {
            'template': "What pathways involve gene {gene}?",
            'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
            RETURN collect(DISTINCT pw.name)[0..5] AS pathways''',
            'task_id': 'M-PATHWAY',
            'answer_key': 'pathways',
            'data_source': 'protein_pathway'
        },
        # Tissues
        {
            'template': "In which tissues is gene {gene}'s protein expressed?",
            'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ASSOCIATED_WITH]->(t:Tissue)
            RETURN collect(DISTINCT t.name)[0..5] AS tissues''',
            'task_id': 'M-TISSUE',
            'answer_key': 'tissues',
            'data_source': 'protein_tissue'
        },
    ]

    if category == 'reasoning':
        templates.append({
            'template': "What is the functional pathway of gene {gene} from protein to biological process?",
            'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[:TRANSLATED_INTO]->(p:Protein)
            MATCH (p)-[:ASSOCIATED_WITH]->(bp:Biological_process)
            RETURN p.name AS protein, collect(DISTINCT bp.name)[0..3] AS processes''',
            'task_id': 'M-REASONING-FUNCTION',
            'answer_key': 'protein',
            'data_source': 'protein_go',
            'type': 'multi-hop'
        })

    idx = 0
    template_idx = 0
    used_combinations = set()

    while len(questions) < count and idx < 1000:
        template = templates[template_idx % len(templates)]
        template_idx += 1

        data_source = template['data_source']
        source_data = {
            'gene_protein': gene_protein,
            'protein_go': protein_go,
            'protein_pathway': protein_pathway,
            'protein_tissue': protein_tissue
        }.get(data_source, gene_protein)

        if not source_data:
            idx += 1
            continue

        if idx >= len(source_data):
            idx = 0
            random.shuffle(source_data)

        rec = source_data[idx % len(source_data)]
        combo_key = f"{template['task_id']}_{rec['gene']}"
        if combo_key in used_combinations:
            idx += 1
            continue
        used_combinations.add(combo_key)

        q = {
            'question': template['template'].format(gene=rec['gene']),
            'cypher': template['cypher'].format(gene=rec['gene']),
            'task_id': template['task_id'],
            'taxonomy': 'M',
            'type': template.get('type', 'one-hop'),
            'answer_key': template['answer_key'],
            'parameters': {'gene': rec['gene']},
            'category': category
        }
        questions.append(q)
        idx += 1

    return questions[:count]

def validate_questions(questions: List[dict], driver) -> Tuple[List[dict], List[dict]]:
    """Validate questions return data from KG."""
    valid = []
    invalid = []

    with driver.session() as session:
        for q in questions:
            try:
                result = session.run(q['cypher'])
                data = list(result)
                if len(data) > 0:
                    valid.append(q)
                else:
                    invalid.append(q)
            except Exception as e:
                invalid.append(q)

    return valid, invalid

def main():
    print('=' * 80)
    print('HYBRID REGENERATION: Keep valid S, regenerate R/C/M')
    print('=' * 80)
    print()

    cfg = load_config()
    driver = GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

    # Target distribution (original)
    targets = {
        'knowledge': {'S': (24, 136), 'R': (30, 170), 'C': (30, 170), 'M': (30, 170)},
        'reasoning': {'S': (18, 102), 'R': (18, 102), 'C': (18, 102), 'M': (24, 136)}
    }

    # Step 1: Extract valid S questions
    print("1. Extracting valid S questions from existing dataset...")
    valid_s = get_valid_s_questions(driver)
    for key, qs in valid_s.items():
        print(f"   {key}: {len(qs)} valid")

    # Step 2: Get KG relationships for R/C/M generation
    print("\n2. Fetching KG relationships for R/C/M generation...")
    kg_data = get_kg_relationships(driver)
    for key, data in kg_data.items():
        print(f"   {key}: {len(data)} records")

    # Step 3: Generate R/C/M questions
    print("\n3. Generating R/C/M questions...")

    results = {}
    for track in ['knowledge', 'reasoning']:
        results[track] = {}

        # Keep valid S
        results[track]['S'] = {
            'dev': valid_s[f'{track}_dev'],
            'test': valid_s[f'{track}_test']
        }

        # Generate R
        dev_count, test_count = targets[track]['R']
        print(f"   Generating R_{track}: {dev_count} dev, {test_count} test")
        r_questions = generate_r_questions(kg_data, dev_count + test_count, track)
        valid_r, _ = validate_questions(r_questions, driver)
        results[track]['R'] = {
            'dev': valid_r[:dev_count],
            'test': valid_r[dev_count:dev_count + test_count]
        }

        # Generate C
        dev_count, test_count = targets[track]['C']
        print(f"   Generating C_{track}: {dev_count} dev, {test_count} test")
        c_questions = generate_c_questions(kg_data, dev_count + test_count, track)
        valid_c, _ = validate_questions(c_questions, driver)
        results[track]['C'] = {
            'dev': valid_c[:dev_count],
            'test': valid_c[dev_count:dev_count + test_count]
        }

        # Generate M
        dev_count, test_count = targets[track]['M']
        print(f"   Generating M_{track}: {dev_count} dev, {test_count} test")
        m_questions = generate_m_questions(kg_data, dev_count + test_count, track)
        valid_m, _ = validate_questions(m_questions, driver)
        results[track]['M'] = {
            'dev': valid_m[:dev_count],
            'test': valid_m[dev_count:dev_count + test_count]
        }

    # Step 4: Save to files
    print("\n4. Saving regenerated questions...")
    data_dir = SCRIPT_DIR / 'data'

    for track in ['knowledge', 'reasoning']:
        track_dir = data_dir / track
        for taxonomy in ['S', 'R', 'C', 'M']:
            for split in ['dev', 'test']:
                questions = results[track][taxonomy][split]
                file_path = track_dir / f'{taxonomy}_{track}_{split}.json'

                # Backup original
                if file_path.exists():
                    backup_path = track_dir / f'{taxonomy}_{track}_{split}.json.backup'
                    if not backup_path.exists():
                        import shutil
                        shutil.copy(file_path, backup_path)

                with open(file_path, 'w') as f:
                    json.dump(questions, f, indent=2)
                print(f"   Saved {file_path.name}: {len(questions)} questions")

    # Step 5: Final validation
    print("\n5. Final validation...")
    total_valid = 0
    total_questions = 0

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    for track in ['knowledge', 'reasoning']:
        print(f"\n{track.upper()} TRACK:")
        for taxonomy in ['S', 'R', 'C', 'M']:
            dev_count = len(results[track][taxonomy]['dev'])
            test_count = len(results[track][taxonomy]['test'])
            target_dev, target_test = targets[track][taxonomy]
            total_valid += dev_count + test_count
            total_questions += target_dev + target_test

            status = "✓" if dev_count >= target_dev and test_count >= target_test else "!"
            print(f"  {taxonomy}: dev={dev_count}/{target_dev}, test={test_count}/{target_test} {status}")

    print(f"\nTotal: {total_valid} questions generated")

    driver.close()
    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
