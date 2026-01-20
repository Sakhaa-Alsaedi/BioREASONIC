#!/usr/bin/env python3
"""
Check how many KGQA questions can be answered by CAUSALdb2+bioKG.
"""

import json
import os
import sys
import yaml
from collections import Counter, defaultdict
from huggingface_hub import snapshot_download

# Add parent directory to path for config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

def read_config():
    config_path = os.path.join(PARENT_DIR, 'config', 'kg_config.yml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_driver(config):
    import neo4j
    uri = f"bolt://{config['db_url']}:{config['db_port']}"
    driver = neo4j.GraphDatabase.driver(
        uri,
        auth=(config['db_user'], config['db_password']),
        encrypted=False
    )
    return driver

print("="*70)
print("KGQA Coverage Analysis: CAUSALdb2 + bioKG")
print("="*70)

# Load KGQA data
DATA_PATH = snapshot_download(
    repo_id="AutoLab-Westlake/BioKGBench-Dataset",
    repo_type="dataset"
)

with open(f"{DATA_PATH}/kgqa/dev.json", 'r') as f:
    kgqa_dev = json.load(f)
with open(f"{DATA_PATH}/kgqa/test.json", 'r') as f:
    kgqa_test = json.load(f)

all_questions = kgqa_dev + kgqa_test
print(f"\nLoaded {len(all_questions)} KGQA questions")

# Connect to Neo4j
config = read_config()
print(f"Connecting to Neo4j at {config['db_url']}:{config['db_port']}...")
driver = get_driver(config)

# First, check what's in our KG
print("\n" + "="*70)
print("Step 1: Analyze Our KG Schema")
print("="*70)

with driver.session() as session:
    # Get node types
    result = session.run("""
        CALL db.labels() YIELD label
        RETURN label
    """)
    labels = [r['label'] for r in result]
    print(f"\nNode types in KG: {labels}")

    # Get relationship types
    result = session.run("""
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
    """)
    rel_types = [r['relationshipType'] for r in result]
    print(f"\nRelationship types: {rel_types}")

    # Count key entities
    print("\nEntity counts:")
    for label in ['Protein', 'Gene', 'Disease', 'Pathway', 'SNP']:
        if label in labels:
            result = session.run(f"MATCH (n:{label}) RETURN count(n) as cnt")
            cnt = result.single()['cnt']
            print(f"  {label}: {cnt:,}")

# Check entity coverage
print("\n" + "="*70)
print("Step 2: Check Entity Coverage")
print("="*70)

# Extract all proteins mentioned in questions
proteins_in_questions = set()
genes_in_questions = set()

for q in all_questions:
    # From entities field
    entities = q.get('entities', {})
    if 'protein' in entities:
        proteins_in_questions.add(entities['protein'])

    # From source field
    source = q.get('source', '')
    if source and source.startswith('P') or source.startswith('Q'):
        proteins_in_questions.add(source)

    # From question text (extract gene names for multi-hop)
    question = q.get('question', '')
    if 'gene' in question.lower():
        # Try to extract gene name
        import re
        gene_match = re.search(r'gene\s+([A-Z][A-Z0-9]+)', question)
        if gene_match:
            genes_in_questions.add(gene_match.group(1))

print(f"\nProteins in KGQA questions: {len(proteins_in_questions)}")
print(f"Sample proteins: {list(proteins_in_questions)[:10]}")

print(f"\nGenes in KGQA questions: {len(genes_in_questions)}")
print(f"Sample genes: {list(genes_in_questions)[:10]}")

# Check how many proteins exist in our KG
with driver.session() as session:
    proteins_found = 0
    proteins_missing = []

    for protein_id in proteins_in_questions:
        result = session.run("""
            MATCH (p:Protein {id: $pid})
            RETURN count(p) as cnt
        """, pid=protein_id)
        if result.single()['cnt'] > 0:
            proteins_found += 1
        else:
            proteins_missing.append(protein_id)

    print(f"\nProtein coverage:")
    print(f"  Found: {proteins_found}/{len(proteins_in_questions)} ({100*proteins_found/len(proteins_in_questions):.1f}%)")
    print(f"  Missing: {len(proteins_missing)}")
    if proteins_missing[:5]:
        print(f"  Sample missing: {proteins_missing[:5]}")

    # Check genes
    genes_found = 0
    genes_missing = []

    for gene_id in genes_in_questions:
        result = session.run("""
            MATCH (g:Gene {id: $gid})
            RETURN count(g) as cnt
        """, gid=gene_id)
        if result.single()['cnt'] > 0:
            genes_found += 1
        else:
            genes_missing.append(gene_id)

    if genes_in_questions:
        print(f"\nGene coverage:")
        print(f"  Found: {genes_found}/{len(genes_in_questions)} ({100*genes_found/len(genes_in_questions) if genes_in_questions else 0:.1f}%)")

# Check relationship coverage
print("\n" + "="*70)
print("Step 3: Check Relationship Coverage")
print("="*70)

with driver.session() as session:
    # Relationships needed for KGQA
    relationship_checks = {
        'Protein-Protein Interaction': """
            MATCH (p1:Protein)-[r]->(p2:Protein)
            WHERE type(r) IN ['INTERACTS_WITH', 'ACTS_ON']
            RETURN count(r) as cnt
        """,
        'Gene-Protein (TRANSLATED_INTO)': """
            MATCH (g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
            RETURN count(*) as cnt
        """,
        'Protein-Disease': """
            MATCH (p:Protein)-[r]->(d:Disease)
            RETURN count(r) as cnt
        """,
        'Protein-Pathway': """
            MATCH (p:Protein)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
            RETURN count(*) as cnt
        """,
        'Protein-GO (Biological Process)': """
            MATCH (p:Protein)-[:ASSOCIATED_WITH]->(bp:Biological_process)
            RETURN count(*) as cnt
        """,
        'Protein-GO (Molecular Function)': """
            MATCH (p:Protein)-[:ASSOCIATED_WITH]->(mf:Molecular_function)
            RETURN count(*) as cnt
        """,
        'Protein-GO (Cellular Component)': """
            MATCH (p:Protein)-[:ASSOCIATED_WITH]->(cc:Cellular_component)
            RETURN count(*) as cnt
        """,
    }

    print("\nRelationship availability:")
    for name, query in relationship_checks.items():
        try:
            result = session.run(query)
            cnt = result.single()['cnt']
            status = "✓" if cnt > 0 else "✗"
            print(f"  {status} {name}: {cnt:,}")
        except Exception as e:
            print(f"  ✗ {name}: Error - {str(e)[:50]}")

# Now check actual question answering capability
print("\n" + "="*70)
print("Step 4: Test Question Answering Capability")
print("="*70)

def can_answer_question(session, question_data):
    """Check if we can answer a specific question."""
    qtype = question_data.get('type', '')
    question = question_data.get('question', '').lower()
    entities = question_data.get('entities', {})
    source = question_data.get('source', '')

    protein_id = entities.get('protein') or source

    # One-hop: protein interactions
    if 'interact' in question and protein_id:
        result = session.run("""
            MATCH (p:Protein {id: $pid})-[r]-(p2:Protein)
            WHERE type(r) IN ['INTERACTS_WITH', 'ACTS_ON']
            RETURN count(p2) as cnt
        """, pid=protein_id)
        return result.single()['cnt'] > 0

    # One-hop: protein-disease
    if 'disease' in question and protein_id:
        result = session.run("""
            MATCH (p:Protein {id: $pid})-[:ASSOCIATED_WITH]->(d:Disease)
            RETURN count(d) as cnt
        """, pid=protein_id)
        return result.single()['cnt'] > 0

    # One-hop: protein-pathway
    if 'pathway' in question and protein_id:
        result = session.run("""
            MATCH (p:Protein {id: $pid})-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
            RETURN count(pw) as cnt
        """, pid=protein_id)
        return result.single()['cnt'] > 0

    # One-hop: biological process
    if 'biological process' in question and protein_id:
        result = session.run("""
            MATCH (p:Protein {id: $pid})-[:ASSOCIATED_WITH]->(bp:Biological_process)
            RETURN count(bp) as cnt
        """, pid=protein_id)
        return result.single()['cnt'] > 0

    # One-hop: molecular function
    if 'molecular function' in question and protein_id:
        result = session.run("""
            MATCH (p:Protein {id: $pid})-[:ASSOCIATED_WITH]->(mf:Molecular_function)
            RETURN count(mf) as cnt
        """, pid=protein_id)
        return result.single()['cnt'] > 0

    # Multi-hop: gene -> protein -> function
    if 'gene' in question and ('function' in question or 'process' in question):
        import re
        gene_match = re.search(r'gene\s+([A-Z][A-Z0-9]+)', question_data.get('question', ''))
        if gene_match:
            gene_name = gene_match.group(1)
            result = session.run("""
                MATCH (g:Gene {id: $gid})-[:TRANSLATED_INTO]->(p:Protein)
                RETURN count(p) as cnt
            """, gid=gene_name)
            return result.single()['cnt'] > 0

    return None  # Unknown question type

# Test on all questions
answerable = {'yes': 0, 'no': 0, 'unknown': 0}
by_type = defaultdict(lambda: {'yes': 0, 'no': 0, 'unknown': 0})

with driver.session() as session:
    for q in all_questions:
        result = can_answer_question(session, q)
        qtype = q.get('type', 'unknown')

        if result is True:
            answerable['yes'] += 1
            by_type[qtype]['yes'] += 1
        elif result is False:
            answerable['no'] += 1
            by_type[qtype]['no'] += 1
        else:
            answerable['unknown'] += 1
            by_type[qtype]['unknown'] += 1

print("\nOverall Answering Capability:")
total = sum(answerable.values())
print(f"  Answerable: {answerable['yes']}/{total} ({100*answerable['yes']/total:.1f}%)")
print(f"  Not answerable: {answerable['no']}/{total} ({100*answerable['no']/total:.1f}%)")
print(f"  Unknown pattern: {answerable['unknown']}/{total} ({100*answerable['unknown']/total:.1f}%)")

print("\nBy Question Type:")
for qtype, counts in by_type.items():
    type_total = sum(counts.values())
    print(f"\n  {qtype}:")
    print(f"    Answerable: {counts['yes']}/{type_total} ({100*counts['yes']/type_total:.1f}%)")
    print(f"    Not answerable: {counts['no']}/{type_total}")
    print(f"    Unknown: {counts['unknown']}/{type_total}")

# Check what's missing
print("\n" + "="*70)
print("Step 5: Gap Analysis")
print("="*70)

with driver.session() as session:
    # Check if we have GO terms
    result = session.run("""
        MATCH (n)
        WHERE n:Biological_process OR n:Molecular_function OR n:Cellular_component
        RETURN labels(n)[0] as label, count(n) as cnt
    """)
    print("\nGO Term nodes:")
    for r in result:
        print(f"  {r['label']}: {r['cnt']:,}")

    # Check protein-GO associations
    result = session.run("""
        MATCH (p:Protein)-[r:ASSOCIATED_WITH]->(go)
        WHERE go:Biological_process OR go:Molecular_function OR go:Cellular_component
        RETURN labels(go)[0] as go_type, count(r) as cnt
    """)
    print("\nProtein-GO associations:")
    for r in result:
        print(f"  Protein -> {r['go_type']}: {r['cnt']:,}")

driver.close()

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"""
KGQA Benchmark Coverage:
- Total questions: {len(all_questions)}
- Protein coverage: {proteins_found}/{len(proteins_in_questions)} ({100*proteins_found/len(proteins_in_questions):.1f}%)
- Answerable: {answerable['yes']}/{total} ({100*answerable['yes']/total:.1f}%)

Key gaps:
- Questions about GO terms (biological process, molecular function)
- Protein-protein interactions
- Multi-hop queries through gene->protein->function
""")

# Save results
results = {
    'total_questions': len(all_questions),
    'protein_coverage': {
        'found': proteins_found,
        'total': len(proteins_in_questions),
        'percentage': 100*proteins_found/len(proteins_in_questions)
    },
    'answerable': answerable,
    'by_type': dict(by_type)
}

with open('kgqa_coverage_report.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to kgqa_coverage_report.json")
