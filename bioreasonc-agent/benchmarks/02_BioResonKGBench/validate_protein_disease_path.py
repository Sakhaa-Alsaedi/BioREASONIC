#!/usr/bin/env python3
"""
Validate that INCREASES_RISK_OF links to Proteins through Genes.
Path: Protein <-[TRANSLATED_INTO]- Gene -[INCREASES_RISK_OF]-> Disease
"""

import yaml
from neo4j import GraphDatabase
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

def main():
    # Load config
    with open(SCRIPT_DIR / 'config' / 'kg_config.yml') as f:
        cfg = yaml.safe_load(f)

    driver = GraphDatabase.driver(
        f"bolt://{cfg['db_url']}:{cfg['db_port']}",
        auth=(cfg['db_user'], cfg['db_password']),
        encrypted=False
    )

    print('='*80)
    print('PROTEIN-GENE-DISEASE PATH VALIDATION')
    print('='*80)
    print()
    print('Expected Path: Protein <-[TRANSLATED_INTO]- Gene -[INCREASES_RISK_OF]-> Disease')
    print()

    with driver.session() as s:

        # 1. Basic stats for INCREASES_RISK_OF
        print('--- 1. INCREASES_RISK_OF Relationship Stats ---')
        r = s.run('''
            MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            RETURN count(r) as total_links,
                   count(DISTINCT g) as unique_genes,
                   count(DISTINCT d) as unique_diseases
        ''')
        rec = r.single()
        print(f'Total Gene->Disease links: {rec["total_links"]:,}')
        print(f'Unique genes with disease risk: {rec["unique_genes"]:,}')
        print(f'Unique diseases: {rec["unique_diseases"]:,}')

        # 2. Check how many risk genes have proteins
        print('\n--- 2. Risk Genes WITH Protein Links ---')
        r = s.run('''
            MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            WITH DISTINCT g
            OPTIONAL MATCH (g)-[:TRANSLATED_INTO]->(p:Protein)
            WITH g, count(p) as protein_count
            RETURN
                count(g) as total_risk_genes,
                sum(CASE WHEN protein_count > 0 THEN 1 ELSE 0 END) as genes_with_proteins,
                sum(CASE WHEN protein_count = 0 THEN 1 ELSE 0 END) as genes_without_proteins
        ''')
        rec = r.single()
        total = rec["total_risk_genes"]
        with_p = rec["genes_with_proteins"]
        without_p = rec["genes_without_proteins"]
        print(f'Total risk genes: {total:,}')
        print(f'Risk genes WITH proteins: {with_p:,} ({with_p/total*100:.1f}%)')
        print(f'Risk genes WITHOUT proteins: {without_p:,} ({without_p/total*100:.1f}%)')

        # 3. Full path: Protein <- Gene -> Disease
        print('\n--- 3. Complete Protein-Gene-Disease Paths ---')
        r = s.run('''
            MATCH (p:Protein)<-[:TRANSLATED_INTO]-(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            RETURN count(*) as total_paths,
                   count(DISTINCT p) as unique_proteins,
                   count(DISTINCT g) as unique_genes,
                   count(DISTINCT d) as unique_diseases
        ''')
        rec = r.single()
        print(f'Total Protein-Gene-Disease paths: {rec["total_paths"]:,}')
        print(f'Unique proteins in paths: {rec["unique_proteins"]:,}')
        print(f'Unique genes in paths: {rec["unique_genes"]:,}')
        print(f'Unique diseases in paths: {rec["unique_diseases"]:,}')

        # 4. Sample complete paths
        print('\n--- 4. Sample Protein-Gene-Disease Paths ---')
        r = s.run('''
            MATCH (p:Protein)<-[:TRANSLATED_INTO]-(g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
            RETURN p.name as protein, g.id as gene, d.name as disease,
                   r.risk_score as risk_score
            ORDER BY r.risk_score DESC
            LIMIT 15
        ''')
        print(f'{"Protein":<15} {"Gene":<15} {"Disease":<30} {"Risk Score":>10}')
        print('-'*75)
        for rec in r:
            disease = str(rec["disease"])[:30] if rec["disease"] else "N/A"
            score = f'{rec["risk_score"]:.3f}' if rec["risk_score"] else "N/A"
            print(f'{rec["protein"]:<15} {rec["gene"]:<15} {disease:<30} {score:>10}')

        # 5. Check if protein names match gene names in risk paths
        print('\n--- 5. Name Matching in Risk Paths ---')
        r = s.run('''
            MATCH (p:Protein)<-[:TRANSLATED_INTO]-(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            WITH p, g, d,
                 toLower(g.id) as gene_id,
                 toLower(p.name) as protein_name
            RETURN
                count(*) as total,
                sum(CASE WHEN gene_id = protein_name THEN 1 ELSE 0 END) as matching
        ''')
        rec = r.single()
        total = rec["total"]
        matching = rec["matching"]
        print(f'Total paths: {total:,}')
        print(f'Gene ID = Protein Name: {matching:,} ({matching/total*100:.1f}%)')

        # 6. Top diseases by protein coverage
        print('\n--- 6. Top Diseases by Protein-Linked Genes ---')
        r = s.run('''
            MATCH (p:Protein)<-[:TRANSLATED_INTO]-(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            WITH d, count(DISTINCT g) as gene_count, count(DISTINCT p) as protein_count
            RETURN d.name as disease, gene_count, protein_count
            ORDER BY protein_count DESC
            LIMIT 10
        ''')
        print(f'{"Disease":<40} {"Genes":>10} {"Proteins":>10}')
        print('-'*65)
        for rec in r:
            disease = str(rec["disease"])[:40] if rec["disease"] else "N/A"
            print(f'{disease:<40} {rec["gene_count"]:>10} {rec["protein_count"]:>10}')

        # 7. Risk genes without proteins (potential gaps)
        print('\n--- 7. Sample Risk Genes WITHOUT Proteins (Potential Gaps) ---')
        r = s.run('''
            MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
            WHERE NOT (g)-[:TRANSLATED_INTO]->(:Protein)
            RETURN g.id as gene, g.name as gene_name, collect(DISTINCT d.name)[0..2] as diseases
            LIMIT 10
        ''')
        print(f'{"Gene ID":<15} {"Gene Name":<35} {"Sample Diseases"}')
        print('-'*80)
        for rec in r:
            gname = str(rec["gene_name"])[:35] if rec["gene_name"] else "N/A"
            diseases = ", ".join(rec["diseases"][:2]) if rec["diseases"] else "N/A"
            print(f'{rec["gene"]:<15} {gname:<35} {diseases[:30]}')

    driver.close()
    print('\n' + '='*80)
    print('VALIDATION COMPLETE')
    print('='*80)

if __name__ == "__main__":
    main()
