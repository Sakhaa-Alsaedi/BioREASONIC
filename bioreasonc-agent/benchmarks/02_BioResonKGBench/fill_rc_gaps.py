#!/usr/bin/env python3
"""
Fill gaps in R/C questions to meet target counts.
Uses additional templates and more data from KG.
"""

import json
import yaml
import random
from pathlib import Path
from neo4j import GraphDatabase

SCRIPT_DIR = Path(__file__).parent
random.seed(44)  # Different seed for variety

def load_config():
    with open(SCRIPT_DIR / 'config' / 'kg_config.yml') as f:
        return yaml.safe_load(f)

def get_driver(cfg):
    return GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

# Additional R templates
R_TEMPLATES_EXTRA = [
    {
        'template': "Which protein has the highest risk for {disease}?",
        'cypher': '''MATCH (p:Protein)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN p.id AS protein
            ORDER BY r.risk_score DESC LIMIT 1''',
        'task_id': 'R-TOP-RISK-PROTEIN',
        'answer_key': 'protein',
        'type': 'one-hop'
    },
    {
        'template': "Which proteins are associated with high risk for {disease}?",
        'cypher': '''MATCH (p:Protein)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.risk_score > 0.5
            RETURN collect(DISTINCT p.id)[0..5] AS proteins''',
        'task_id': 'R-HIGH-RISK-PROTEINS',
        'answer_key': 'proteins',
        'type': 'one-hop'
    },
    {
        'template': "Which SNPs are in genes associated with {disease}?",
        'cypher': '''MATCH (s:SNP)-[:MAPS_TO]->(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN collect(DISTINCT s.id)[0..5] AS snps''',
        'task_id': 'R-SNPS-IN-RISK-GENES',
        'answer_key': 'snps',
        'type': 'multi-hop'
    },
    {
        'template': "Which gene has the lowest p-value association with {disease}?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.pvalue IS NOT NULL
            RETURN g.id AS gene
            ORDER BY r.pvalue ASC LIMIT 1''',
        'task_id': 'R-LOWEST-PVALUE-GENE',
        'answer_key': 'gene',
        'type': 'one-hop'
    },
    {
        'template': "What diseases are associated with the highest risk gene {gene}?",
        'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[r:INCREASES_RISK_OF]->(d:Disease)
            RETURN collect(DISTINCT d.name)[0..3] AS diseases
            ORDER BY r.risk_score DESC''',
        'task_id': 'R-DISEASES-FROM-GENE',
        'answer_key': 'diseases',
        'type': 'one-hop'
    },
    {
        'template': "How many genes increase risk for {disease}?",
        'cypher': '''MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            RETURN count(DISTINCT g) AS count''',
        'task_id': 'R-COUNT-RISK-GENES',
        'answer_key': 'count',
        'type': 'aggregation'
    },
    {
        'template': "Which genes have moderate risk for {disease}?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.risk_score > 0.3 AND r.risk_score < 0.7
            RETURN collect(DISTINCT g.id)[0..5] AS genes''',
        'task_id': 'R-MODERATE-RISK-GENES',
        'answer_key': 'genes',
        'type': 'one-hop'
    },
    {
        'template': "Which genes increase risk for {disease} with strong evidence?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.evidence_score > 0.7
            RETURN collect(DISTINCT g.id)[0..5] AS genes''',
        'task_id': 'R-STRONG-EVIDENCE-GENES-2',
        'answer_key': 'genes',
        'type': 'one-hop'
    },
    {
        'template': "Which gene has the second highest risk for {disease}?",
        'cypher': '''MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease {{id: "{disease_id}"}})
            WITH g, r ORDER BY r.risk_score DESC SKIP 1 LIMIT 1
            RETURN g.id AS gene''',
        'task_id': 'R-SECOND-RISK-GENE',
        'answer_key': 'gene',
        'type': 'one-hop'
    },
]

# Additional C templates
C_TEMPLATES_EXTRA = [
    {
        'template': "Which SNPs are mapped to gene {gene}?",
        'cypher': '''MATCH (s:SNP)-[:MAPS_TO]->(g:Gene {{id: "{gene}"}})
            RETURN collect(DISTINCT s.id)[0..5] AS snps''',
        'task_id': 'C-SNPS-IN-GENE',
        'answer_key': 'snps',
        'type': 'one-hop'
    },
    {
        'template': "Which disease has the strongest causal link from SNPs?",
        'cypher': '''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WITH d, avg(r.causal_score) AS avg_score
            RETURN d.name AS disease
            ORDER BY avg_score DESC LIMIT 1''',
        'task_id': 'C-TOP-CAUSAL-DISEASE',
        'answer_key': 'disease',
        'type': 'aggregation'
    },
    {
        'template': "Which genes contain SNPs with causal effects on {disease}?",
        'cypher': '''MATCH (s:SNP)-[:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            MATCH (s)-[:MAPS_TO]->(g:Gene)
            RETURN collect(DISTINCT g.id)[0..5] AS genes''',
        'task_id': 'C-GENES-WITH-CAUSAL-SNPS',
        'answer_key': 'genes',
        'type': 'multi-hop'
    },
    {
        'template': "How many SNPs have causal effects on {disease}?",
        'cypher': '''MATCH (s:SNP)-[:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            RETURN count(DISTINCT s) AS count''',
        'task_id': 'C-COUNT-CAUSAL-SNPS',
        'answer_key': 'count',
        'type': 'aggregation'
    },
    {
        'template': "Which SNPs have moderate causal effects on {disease}?",
        'cypher': '''MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease {{id: "{disease_id}"}})
            WHERE r.causal_score > 0.3 AND r.causal_score < 0.7
            RETURN collect(DISTINCT s.id)[0..5] AS snps''',
        'task_id': 'C-MODERATE-CAUSAL-SNPS',
        'answer_key': 'snps',
        'type': 'one-hop'
    },
]

# Additional reasoning templates
R_REASONING_EXTRA = [
    {
        'template': "Which diseases share risk genes with {disease}?",
        'cypher': '''MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d1:Disease {{id: "{disease_id}"}})
            MATCH (g)-[:INCREASES_RISK_OF]->(d2:Disease)
            WHERE d2.id <> d1.id
            RETURN collect(DISTINCT d2.name)[0..5] AS diseases''',
        'task_id': 'R-SHARED-RISK-DISEASES',
        'answer_key': 'diseases',
        'type': 'multi-hop'
    },
    {
        'template': "Which gene increases risk for the most diseases?",
        'cypher': '''MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            WITH g, count(DISTINCT d) AS disease_count
            RETURN g.id AS gene
            ORDER BY disease_count DESC LIMIT 1''',
        'task_id': 'R-MOST-CONNECTED-GENE',
        'answer_key': 'gene',
        'type': 'aggregation'
    },
    {
        'template': "How many diseases does gene {gene} increase risk for?",
        'cypher': '''MATCH (g:Gene {{id: "{gene}"}})-[:INCREASES_RISK_OF]->(d:Disease)
            RETURN count(DISTINCT d) AS count''',
        'task_id': 'R-GENE-DISEASE-COUNT',
        'answer_key': 'count',
        'type': 'aggregation'
    },
    {
        'template': "Which genes are common risk factors for diseases?",
        'cypher': '''MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            WITH g, count(DISTINCT d) AS disease_count
            WHERE disease_count >= 2
            RETURN collect(g.id)[0..5] AS genes''',
        'task_id': 'R-COMMON-RISK-GENES',
        'answer_key': 'genes',
        'type': 'aggregation'
    },
    {
        'template': "Which disease has the most risk genes?",
        'cypher': '''MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            WITH d, count(DISTINCT g) AS gene_count
            RETURN d.name AS disease
            ORDER BY gene_count DESC LIMIT 1''',
        'task_id': 'R-DISEASE-MOST-GENES',
        'answer_key': 'disease',
        'type': 'aggregation'
    },
]

C_REASONING_EXTRA = [
    {
        'template': "Which diseases share causal SNPs?",
        'cypher': '''MATCH (s:SNP)-[:PUTATIVE_CAUSAL_EFFECT]->(d1:Disease)
            MATCH (s)-[:PUTATIVE_CAUSAL_EFFECT]->(d2:Disease)
            WHERE d1.id < d2.id
            RETURN DISTINCT d1.name + ' and ' + d2.name AS disease_pair
            LIMIT 5''',
        'task_id': 'C-SHARED-CAUSAL-PAIRS',
        'answer_key': 'disease_pair',
        'type': 'multi-hop'
    },
    {
        'template': "Through which gene does SNP {snp} affect diseases?",
        'cypher': '''MATCH (s:SNP {{id: "{snp}"}})-[:MAPS_TO]->(g:Gene)
            RETURN g.id AS gene LIMIT 1''',
        'task_id': 'C-SNP-TO-GENE',
        'answer_key': 'gene',
        'type': 'one-hop'
    },
]

def get_more_kg_data(driver):
    """Fetch additional data from KG for question generation."""
    data = {}

    with driver.session() as session:
        # More diseases with risk genes (lower threshold)
        print("  Fetching more diseases with risk genes...")
        result = session.run('''
            MATCH (d:Disease)<-[r:INCREASES_RISK_OF]-(g:Gene)
            WITH d, collect({gene: g.id, score: r.risk_score})[0..20] AS genes
            WHERE size(genes) >= 1
            RETURN d.id AS disease_id, d.name AS disease_name, genes
            LIMIT 200
        ''')
        data['disease_genes'] = [dict(r) for r in result]

        # More diseases with causal SNPs (lower threshold)
        print("  Fetching more diseases with causal SNPs...")
        result = session.run('''
            MATCH (d:Disease)<-[r:PUTATIVE_CAUSAL_EFFECT]-(s:SNP)
            WITH d, collect({snp: s.id, score: r.causal_score})[0..20] AS snps
            WHERE size(snps) >= 1
            RETURN d.id AS disease_id, d.name AS disease_name, snps
            LIMIT 200
        ''')
        data['disease_snps'] = [dict(r) for r in result]

        # More genes with diseases
        print("  Fetching more genes with diseases...")
        result = session.run('''
            MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            WITH g, collect({disease_id: d.id, disease_name: d.name, score: r.risk_score}) AS diseases
            WHERE size(diseases) >= 1
            RETURN g.id AS gene, diseases[0..10] AS diseases
            LIMIT 200
        ''')
        data['gene_diseases'] = [dict(r) for r in result]

        # More SNPs with causal effects
        print("  Fetching more SNPs with causal effects...")
        result = session.run('''
            MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            WITH s, collect({disease_id: d.id, disease_name: d.name, score: r.causal_score}) AS diseases
            WHERE size(diseases) >= 1
            RETURN s.id AS snp, diseases[0..10] AS diseases
            LIMIT 200
        ''')
        data['snp_diseases'] = [dict(r) for r in result]

        # Diseases with proteins
        print("  Fetching diseases with proteins...")
        result = session.run('''
            MATCH (d:Disease)<-[r:INCREASES_RISK_OF]-(p:Protein)
            WITH d, collect({protein: p.id, score: r.risk_score})[0..20] AS proteins
            WHERE size(proteins) >= 1
            RETURN d.id AS disease_id, d.name AS disease_name, proteins
            LIMIT 100
        ''')
        data['disease_proteins'] = [dict(r) for r in result]

    return data

def generate_questions(templates, kg_data, count, category):
    """Generate questions from templates using KG data."""
    questions = []
    used = set()

    for _ in range(count * 5):
        if len(questions) >= count:
            break

        template = random.choice(templates)

        try:
            if '{disease}' in template['template'] and '{gene}' not in template['template'] and '{disease1}' not in template['template']:
                # Disease-based question
                if 'protein' in template['answer_key']:
                    if not kg_data.get('disease_proteins'):
                        continue
                    rec = random.choice(kg_data['disease_proteins'])
                else:
                    if not kg_data.get('disease_genes') and not kg_data.get('disease_snps'):
                        continue
                    pool = kg_data.get('disease_genes', []) + kg_data.get('disease_snps', [])
                    rec = random.choice(pool)

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

            else:
                # Template without parameters
                key = template['task_id']
                if key in used:
                    continue
                used.add(key)

                q = {
                    'question': template['template'],
                    'cypher': template['cypher'],
                    'task_id': template['task_id'],
                    'taxonomy': 'R' if 'R-' in template['task_id'] else 'C',
                    'type': template['type'],
                    'answer_key': template['answer_key'],
                    'parameters': {},
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
                if data and data[0].get(q['answer_key']) is not None:
                    val = data[0].get(q['answer_key'])
                    # Ensure non-empty result
                    if val and (not isinstance(val, list) or len(val) > 0):
                        valid.append(q)
            except:
                pass

    return valid

def main():
    print("=" * 80)
    print("FILLING R/C QUESTION GAPS")
    print("=" * 80)
    print()

    cfg = load_config()
    driver = get_driver(cfg)

    # Load existing questions
    data_dir = SCRIPT_DIR / 'data'

    # Check gaps
    targets = {
        'knowledge': {'R': 170, 'C': 170},
        'reasoning': {'R': 102, 'C': 102}
    }

    print("Current question counts vs targets:")
    gaps = {}
    for track in ['knowledge', 'reasoning']:
        gaps[track] = {}
        for taxonomy in ['R', 'C']:
            test_file = data_dir / track / f'{taxonomy}_{track}_test.json'
            with open(test_file) as f:
                current = len(json.load(f))
            target = targets[track][taxonomy]
            gap = target - current
            gaps[track][taxonomy] = gap
            print(f"  {taxonomy}_{track}_test: {current}/{target} (gap: {gap})")

    # Get more KG data
    print("\n1. Fetching more KG data...")
    kg_data = get_more_kg_data(driver)
    for key, data in kg_data.items():
        print(f"   {key}: {len(data)} records")

    # Generate extra questions for each gap
    print("\n2. Generating additional questions...")

    extra_questions = {}
    for track in ['knowledge', 'reasoning']:
        extra_questions[track] = {}

        for taxonomy in ['R', 'C']:
            gap = gaps[track][taxonomy]
            if gap <= 0:
                extra_questions[track][taxonomy] = []
                continue

            # Select templates
            if taxonomy == 'R':
                templates = R_TEMPLATES_EXTRA + (R_REASONING_EXTRA if track == 'reasoning' else [])
            else:
                templates = C_TEMPLATES_EXTRA + (C_REASONING_EXTRA if track == 'reasoning' else [])

            # Generate
            new_qs = generate_questions(templates, kg_data, gap + 20, track)
            valid = validate_questions(new_qs, driver)
            extra_questions[track][taxonomy] = valid[:gap]
            print(f"   Generated {len(valid[:gap])} extra {taxonomy}_{track}_test questions")

    # Merge with existing files
    print("\n3. Merging with existing files...")

    for track in ['knowledge', 'reasoning']:
        for taxonomy in ['R', 'C']:
            test_file = data_dir / track / f'{taxonomy}_{track}_test.json'

            with open(test_file) as f:
                existing = json.load(f)

            # Add extra questions
            extra = extra_questions[track][taxonomy]
            merged = existing + extra

            # Ensure no duplicates based on cypher
            seen = set()
            unique = []
            for q in merged:
                if q['cypher'] not in seen:
                    seen.add(q['cypher'])
                    unique.append(q)

            with open(test_file, 'w') as f:
                json.dump(unique, f, indent=2)

            print(f"   {taxonomy}_{track}_test: {len(existing)} -> {len(unique)}")

    # Update combined files
    print("\n4. Updating combined files...")

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

    with open(data_dir / 'combined_CKGQA_dev_matched.json', 'w') as f:
        json.dump(all_dev, f, indent=2)
    with open(data_dir / 'combined_CKGQA_test_matched.json', 'w') as f:
        json.dump(all_test, f, indent=2)

    print(f"   Combined dev: {len(all_dev)} questions")
    print(f"   Combined test: {len(all_test)} questions")

    driver.close()

    print("\n" + "=" * 80)
    print("GAP FILLING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
