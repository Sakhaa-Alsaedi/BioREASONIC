#!/usr/bin/env python3
"""
Regenerate R and C questions to use scores for RANKING/FILTERING, not as answers.

OLD Pattern (problematic):
  Q: "What is the risk score for gene X?"
  A: 0.632 (LLMs can't match exact numbers)

NEW Pattern (better):
  Q: "Which gene has the highest risk score for disease Y?"
  A: GENE_NAME (LLMs can match entity names)

The scores are used to ORDER BY or filter, but the answer is an entity.
"""

import json
import yaml
import random
from pathlib import Path
from neo4j import GraphDatabase

SCRIPT_DIR = Path(__file__).parent
random.seed(42)

def load_config():
    with open(SCRIPT_DIR / 'config' / 'kg_config.yml') as f:
        return yaml.safe_load(f)

def get_driver(cfg):
    return GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

# ============================================================================
# NEW R (Risk) Question Templates - Entity-based answers
# ============================================================================

R_TEMPLATES = [
    # Top risk gene for disease
    {
        'template': "Which gene has the highest risk for {disease}?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN g.id AS gene
            ORDER BY r.risk_score DESC LIMIT 1''',
        'task_id': 'R-TOP-RISK-GENE',
        'answer_key': 'gene',
        'type': 'one-hop'
    },
    # Top 3 risk genes
    {
        'template': "What are the top 3 risk genes for {disease}?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN collect(g.id)[0..3] AS genes
            ORDER BY r.risk_score DESC LIMIT 3''',
        'task_id': 'R-TOP3-RISK-GENES',
        'answer_key': 'genes',
        'type': 'one-hop'
    },
    # High risk genes
    {
        'template': "Which genes have high risk for {disease}?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.risk_score > 0.5
            RETURN collect(DISTINCT g.id)[0..5] AS genes''',
        'task_id': 'R-HIGH-RISK-GENES',
        'answer_key': 'genes',
        'type': 'one-hop'
    },
    # Most significant SNP
    {
        'template': "Which SNP has the most significant association with {disease}?",
        'cypher': '''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.pvalue IS NOT NULL
            RETURN s.id AS snp
            ORDER BY r.pvalue ASC LIMIT 1''',
        'task_id': 'R-SIGNIFICANT-SNP',
        'answer_key': 'snp',
        'type': 'one-hop'
    },
    # Disease most affected by gene
    {
        'template': "Which disease is most affected by gene {gene}?",
        'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[r:INCREASES_RISK_OF]->(d:Disease)
            RETURN d.name AS disease
            ORDER BY r.risk_score DESC LIMIT 1''',
        'task_id': 'R-TOP-DISEASE-FROM-GENE',
        'answer_key': 'disease',
        'type': 'one-hop'
    },
    # Genes with strong evidence
    {
        'template': "Which genes have strong evidence for affecting {disease}?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.evidence_score > 0.6
            RETURN collect(DISTINCT g.id)[0..5] AS genes''',
        'task_id': 'R-STRONG-EVIDENCE-GENES',
        'answer_key': 'genes',
        'type': 'one-hop'
    },
]

R_REASONING_TEMPLATES = [
    # Shared high-risk genes
    {
        'template': "Which genes have high risk for both {disease1} and {disease2}?",
        'cypher': '''MATCH (g:Gene)-[r1:INCREASES_RISK_OF]->(d1:Disease {{id: "{disease_id1}"}})
            WHERE r1.risk_score > 0.4
            MATCH (g)-[r2:INCREASES_RISK_OF]->(d2:Disease {{id: "{disease_id2}"}})
            WHERE r2.risk_score > 0.4
            RETURN collect(DISTINCT g.id)[0..5] AS genes''',
        'task_id': 'R-REASONING-SHARED-HIGH-RISK',
        'answer_key': 'genes',
        'type': 'multi-hop'
    },
    # Compare risk
    {
        'template': "Does gene {gene} have higher risk for {disease1} or {disease2}?",
        'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[r1:INCREASES_RISK_OF]->(d1:Disease {{id: "{disease_id1}"}})
            MATCH (g)-[r2:INCREASES_RISK_OF]->(d2:Disease {{id: "{disease_id2}"}})
            RETURN CASE WHEN r1.risk_score > r2.risk_score THEN d1.name ELSE d2.name END AS disease''',
        'task_id': 'R-REASONING-COMPARE-RISK',
        'answer_key': 'disease',
        'type': 'multi-hop'
    },
]

# ============================================================================
# NEW C (Causal) Question Templates - Entity-based answers
# ============================================================================

C_TEMPLATES = [
    # Top causal SNP
    {
        'template': "Which SNP has the strongest causal effect on {disease}?",
        'cypher': '''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            RETURN s.id AS snp
            ORDER BY r.causal_score DESC LIMIT 1''',
        'task_id': 'C-TOP-CAUSAL-SNP',
        'answer_key': 'snp',
        'type': 'one-hop'
    },
    # Top 3 causal SNPs
    {
        'template': "What are the top 3 SNPs with strongest causal effects on {disease}?",
        'cypher': '''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            RETURN collect(s.id)[0..3] AS snps
            ORDER BY r.causal_score DESC LIMIT 3''',
        'task_id': 'C-TOP3-CAUSAL-SNPS',
        'answer_key': 'snps',
        'type': 'one-hop'
    },
    # High causal SNPs
    {
        'template': "Which SNPs have strong causal effects on {disease}?",
        'cypher': '''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.causal_score > 0.8
            RETURN collect(DISTINCT s.id)[0..5] AS snps''',
        'task_id': 'C-STRONG-CAUSAL-SNPS',
        'answer_key': 'snps',
        'type': 'one-hop'
    },
    # Genes with very strong evidence
    {
        'template': "Which genes have very strong evidence for causing {disease}?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.evidence_score > 0.8
            RETURN collect(DISTINCT g.id)[0..5] AS genes''',
        'task_id': 'C-VERY-STRONG-EVIDENCE',
        'answer_key': 'genes',
        'type': 'one-hop'
    },
    # Disease most affected by SNP
    {
        'template': "Which disease is most strongly affected by SNP {snp}?",
        'cypher': '''MATCH (s:SNP {{id: "{snp}"}})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            RETURN d.name AS disease
            ORDER BY r.causal_score DESC LIMIT 1''',
        'task_id': 'C-TOP-DISEASE-FROM-SNP',
        'answer_key': 'disease',
        'type': 'one-hop'
    },
    # Validated causal SNPs
    {
        'template': "Which SNPs have validated causal effects on {disease}?",
        'cypher': '''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.causal_score = 1.0
            RETURN collect(DISTINCT s.id)[0..5] AS snps''',
        'task_id': 'C-VALIDATED-CAUSAL',
        'answer_key': 'snps',
        'type': 'one-hop'
    },
]

C_REASONING_TEMPLATES = [
    # SNPs affecting multiple diseases
    {
        'template': "Which SNPs have causal effects on both {disease1} and {disease2}?",
        'cypher': '''MATCH (s:SNP)-[:PUTATIVE_CAUSAL_EFFECT]->(d1:Disease {{id: "{disease_id1}"}})
            MATCH (s)-[:PUTATIVE_CAUSAL_EFFECT]->(d2:Disease {{id: "{disease_id2}"}})
            RETURN collect(DISTINCT s.id)[0..5] AS snps''',
        'task_id': 'C-REASONING-SHARED-CAUSAL',
        'answer_key': 'snps',
        'type': 'multi-hop'
    },
    # Causal chain: SNP -> Gene -> Disease
    {
        'template': "Which gene connects SNP {snp} to its causal effect on diseases?",
        'cypher': '''MATCH (s:SNP {{id: "{snp}"}})-[:MAPS_TO]->(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            RETURN DISTINCT g.id AS gene LIMIT 1''',
        'task_id': 'C-REASONING-CAUSAL-CHAIN',
        'answer_key': 'gene',
        'type': 'multi-hop'
    },
]

def get_kg_data(driver):
    """Fetch data from KG for question generation."""
    data = {}

    with driver.session() as session:
        # Diseases with high-risk genes
        print("  Fetching diseases with risk genes...")
        result = session.run('''
            MATCH (d:Disease)<-[r:INCREASES_RISK_OF]-(g:Gene)
            WHERE r.risk_score > 0.3
            WITH d, collect({gene: g.id, score: r.risk_score})[0..10] AS genes
            WHERE size(genes) >= 3
            RETURN d.id AS disease_id, d.name AS disease_name, genes
            LIMIT 100
        ''')
        data['disease_genes'] = [dict(r) for r in result]

        # Diseases with causal SNPs
        print("  Fetching diseases with causal SNPs...")
        result = session.run('''
            MATCH (d:Disease)<-[r:PUTATIVE_CAUSAL_EFFECT]-(s:SNP)
            WHERE r.causal_score > 0.5
            WITH d, collect({snp: s.id, score: r.causal_score})[0..10] AS snps
            WHERE size(snps) >= 3
            RETURN d.id AS disease_id, d.name AS disease_name, snps
            LIMIT 100
        ''')
        data['disease_snps'] = [dict(r) for r in result]

        # Genes with multiple diseases
        print("  Fetching genes with multiple diseases...")
        result = session.run('''
            MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            WHERE r.risk_score > 0.3
            WITH g, collect({disease_id: d.id, disease_name: d.name, score: r.risk_score}) AS diseases
            WHERE size(diseases) >= 2
            RETURN g.id AS gene, diseases[0..5] AS diseases
            LIMIT 100
        ''')
        data['gene_diseases'] = [dict(r) for r in result]

        # SNPs with causal effects
        print("  Fetching SNPs with causal effects...")
        result = session.run('''
            MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WHERE r.causal_score > 0.5
            WITH s, collect({disease_id: d.id, disease_name: d.name, score: r.causal_score}) AS diseases
            WHERE size(diseases) >= 1
            RETURN s.id AS snp, diseases[0..5] AS diseases
            LIMIT 100
        ''')
        data['snp_diseases'] = [dict(r) for r in result]

    return data

def generate_questions(templates, kg_data, count, category):
    """Generate questions from templates using KG data."""
    questions = []
    used = set()

    random.shuffle(templates)

    for _ in range(count * 3):  # Generate extra to account for failures
        if len(questions) >= count:
            break

        template = random.choice(templates)

        try:
            if '{disease}' in template['template'] and '{gene}' not in template['template']:
                # Disease-based question
                if not kg_data.get('disease_genes'):
                    continue
                rec = random.choice(kg_data['disease_genes'])

                key = f"{template['task_id']}_{rec['disease_id']}"
                if key in used:
                    continue
                used.add(key)

                q = {
                    'question': template['template'].format(disease=rec['disease_name']),
                    'cypher': template['cypher'].format(disease_id=rec['disease_id']),
                    'task_id': template['task_id'],
                    'taxonomy': 'R' if 'R-' in template['task_id'] else 'C',
                    'type': template['type'],
                    'answer_key': template['answer_key'],
                    'parameters': {'disease': rec['disease_name'], 'disease_id': rec['disease_id']},
                    'category': category
                }
                questions.append(q)

            elif '{gene}' in template['template'] and '{disease1}' not in template['template']:
                # Gene-based question
                if not kg_data.get('gene_diseases'):
                    continue
                rec = random.choice(kg_data['gene_diseases'])

                key = f"{template['task_id']}_{rec['gene']}"
                if key in used:
                    continue
                used.add(key)

                q = {
                    'question': template['template'].format(gene=rec['gene']),
                    'cypher': template['cypher'].format(gene=rec['gene']),
                    'task_id': template['task_id'],
                    'taxonomy': 'R' if 'R-' in template['task_id'] else 'C',
                    'type': template['type'],
                    'answer_key': template['answer_key'],
                    'parameters': {'gene': rec['gene']},
                    'category': category
                }
                questions.append(q)

            elif '{snp}' in template['template']:
                # SNP-based question
                if not kg_data.get('snp_diseases'):
                    continue
                rec = random.choice(kg_data['snp_diseases'])

                key = f"{template['task_id']}_{rec['snp']}"
                if key in used:
                    continue
                used.add(key)

                q = {
                    'question': template['template'].format(snp=rec['snp']),
                    'cypher': template['cypher'].format(snp=rec['snp']),
                    'task_id': template['task_id'],
                    'taxonomy': 'C',
                    'type': template['type'],
                    'answer_key': template['answer_key'],
                    'parameters': {'snp': rec['snp']},
                    'category': category
                }
                questions.append(q)

            elif '{disease1}' in template['template']:
                # Multi-disease reasoning question
                if not kg_data.get('disease_genes') or len(kg_data['disease_genes']) < 2:
                    continue

                recs = random.sample(kg_data['disease_genes'], 2)
                d1, d2 = recs[0], recs[1]

                key = f"{template['task_id']}_{d1['disease_id']}_{d2['disease_id']}"
                if key in used:
                    continue
                used.add(key)

                if '{gene}' in template['template']:
                    # Need a gene that affects both
                    genes = kg_data.get('gene_diseases', [])
                    if not genes:
                        continue
                    gene_rec = random.choice(genes)

                    q = {
                        'question': template['template'].format(
                            gene=gene_rec['gene'],
                            disease1=d1['disease_name'],
                            disease2=d2['disease_name']
                        ),
                        'cypher': template['cypher'].format(
                            gene=gene_rec['gene'],
                            disease_id1=d1['disease_id'],
                            disease_id2=d2['disease_id']
                        ),
                        'task_id': template['task_id'],
                        'taxonomy': 'R' if 'R-' in template['task_id'] else 'C',
                        'type': template['type'],
                        'answer_key': template['answer_key'],
                        'parameters': {
                            'gene': gene_rec['gene'],
                            'disease1': d1['disease_name'],
                            'disease2': d2['disease_name']
                        },
                        'category': category
                    }
                else:
                    q = {
                        'question': template['template'].format(
                            disease1=d1['disease_name'],
                            disease2=d2['disease_name']
                        ),
                        'cypher': template['cypher'].format(
                            disease_id1=d1['disease_id'],
                            disease_id2=d2['disease_id']
                        ),
                        'task_id': template['task_id'],
                        'taxonomy': 'R' if 'R-' in template['task_id'] else 'C',
                        'type': template['type'],
                        'answer_key': template['answer_key'],
                        'parameters': {
                            'disease1': d1['disease_name'],
                            'disease2': d2['disease_name']
                        },
                        'category': category
                    }
                questions.append(q)

        except Exception as e:
            continue

    return questions[:count]

def validate_questions(questions, driver):
    """Validate questions return data."""
    valid = []

    with driver.session() as session:
        for q in questions:
            try:
                result = session.run(q['cypher'])
                data = list(result)
                if data and data[0].get(q['answer_key']):
                    valid.append(q)
            except:
                pass

    return valid

def main():
    print("=" * 80)
    print("REGENERATING R/C QUESTIONS - ENTITY-BASED ANSWERS")
    print("Using scores for ranking/filtering, returning entities")
    print("=" * 80)
    print()

    cfg = load_config()
    driver = get_driver(cfg)

    # Target counts
    targets = {
        'knowledge': {'R': (30, 170), 'C': (30, 170)},
        'reasoning': {'R': (18, 102), 'C': (18, 102)}
    }

    # Get KG data
    print("1. Fetching KG data...")
    kg_data = get_kg_data(driver)
    for key, data in kg_data.items():
        print(f"   {key}: {len(data)} records")

    # Generate questions
    print("\n2. Generating questions...")

    results = {}
    for track in ['knowledge', 'reasoning']:
        results[track] = {}

        # R questions
        dev_count, test_count = targets[track]['R']
        templates = R_TEMPLATES + (R_REASONING_TEMPLATES if track == 'reasoning' else [])

        r_questions = generate_questions(templates, kg_data, dev_count + test_count + 50, track)
        valid_r = validate_questions(r_questions, driver)

        results[track]['R'] = {
            'dev': valid_r[:dev_count],
            'test': valid_r[dev_count:dev_count + test_count]
        }
        print(f"   R_{track}: {len(valid_r[:dev_count])} dev, {len(valid_r[dev_count:dev_count+test_count])} test")

        # C questions
        dev_count, test_count = targets[track]['C']
        templates = C_TEMPLATES + (C_REASONING_TEMPLATES if track == 'reasoning' else [])

        c_questions = generate_questions(templates, kg_data, dev_count + test_count + 50, track)
        valid_c = validate_questions(c_questions, driver)

        results[track]['C'] = {
            'dev': valid_c[:dev_count],
            'test': valid_c[dev_count:dev_count + test_count]
        }
        print(f"   C_{track}: {len(valid_c[:dev_count])} dev, {len(valid_c[dev_count:dev_count+test_count])} test")

    # Save to files
    print("\n3. Saving questions...")
    data_dir = SCRIPT_DIR / 'data'

    for track in ['knowledge', 'reasoning']:
        track_dir = data_dir / track
        for taxonomy in ['R', 'C']:
            for split in ['dev', 'test']:
                questions = results[track][taxonomy][split]
                file_path = track_dir / f'{taxonomy}_{track}_{split}.json'

                with open(file_path, 'w') as f:
                    json.dump(questions, f, indent=2)
                print(f"   Saved {file_path.name}: {len(questions)} questions")

    # Update combined files
    print("\n4. Updating combined files...")

    # Load S and M (unchanged)
    all_dev = []
    all_test = []

    for track in ['knowledge', 'reasoning']:
        for taxonomy in ['S', 'R', 'C', 'M']:
            dev_file = data_dir / track / f'{taxonomy}_{track}_dev.json'
            test_file = data_dir / track / f'{taxonomy}_{track}_test.json'

            with open(dev_file) as f:
                all_dev.extend(json.load(f))
            with open(test_file) as f:
                all_test.extend(json.load(f))

    # Save combined
    with open(data_dir / 'combined_CKGQA_dev_matched.json', 'w') as f:
        json.dump(all_dev, f, indent=2)
    with open(data_dir / 'combined_CKGQA_test_matched.json', 'w') as f:
        json.dump(all_test, f, indent=2)

    print(f"   Combined dev: {len(all_dev)} questions")
    print(f"   Combined test: {len(all_test)} questions")

    driver.close()

    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    print("\nNEW R/C Question Pattern:")
    print("  OLD: 'What is the risk score?' → Answer: 0.632")
    print("  NEW: 'Which gene has highest risk?' → Answer: GENE_NAME")
    print("\nScores used for: ORDER BY, WHERE filtering")
    print("Answers are: Entity names (genes, SNPs, diseases)")

if __name__ == "__main__":
    main()
