#!/usr/bin/env python3
"""
Validate Gene-Protein linkage in the Knowledge Graph.
Checks that gene names and protein names are properly linked.
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
    print('GENE-PROTEIN LINKAGE VALIDATION')
    print('='*80)

    with driver.session() as s:

        # 1. Total Gene->Protein relationships
        print('\n--- 1. TRANSLATED_INTO Relationship Stats ---')
        r = s.run('''
            MATCH (g:Gene)-[r:TRANSLATED_INTO]->(p:Protein)
            RETURN count(r) as total_links,
                   count(DISTINCT g) as unique_genes,
                   count(DISTINCT p) as unique_proteins
        ''')
        rec = r.single()
        print(f'Total Gene->Protein links: {rec["total_links"]:,}')
        print(f'Unique genes with proteins: {rec["unique_genes"]:,}')
        print(f'Unique proteins linked: {rec["unique_proteins"]:,}')

        # 2. Check name matching
        print('\n--- 2. Gene-Protein Name Matching ---')
        r = s.run('''
            MATCH (g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
            WITH g, p,
                 toLower(g.id) as gene_id,
                 toLower(p.name) as protein_name
            RETURN
                count(*) as total,
                sum(CASE WHEN gene_id = protein_name THEN 1 ELSE 0 END) as exact_match
        ''')
        rec = r.single()
        total = rec["total"]
        exact = rec["exact_match"]
        print(f'Total links: {total:,}')
        print(f'Gene ID = Protein Name (exact): {exact:,} ({exact/total*100:.1f}%)')

        # 3. Sample matching pairs
        print('\n--- 3. Sample MATCHING Gene-Protein Pairs ---')
        r = s.run('''
            MATCH (g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
            WHERE toLower(g.id) = toLower(p.name)
            RETURN g.id as gene_id, g.name as gene_name,
                   p.id as protein_id, p.name as protein_name
            LIMIT 10
        ''')
        print(f'{"Gene ID":<15} {"Gene Name":<30} {"Protein ID":<15} {"Protein Name":<15}')
        print('-'*80)
        for rec in r:
            gname = str(rec["gene_name"])[:30] if rec["gene_name"] else "N/A"
            print(f'{rec["gene_id"]:<15} {gname:<30} {rec["protein_id"]:<15} {rec["protein_name"]:<15}')

        # 4. Sample non-matching pairs
        print('\n--- 4. Sample NON-MATCHING Gene-Protein Pairs (to check) ---')
        r = s.run('''
            MATCH (g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
            WHERE toLower(g.id) <> toLower(p.name)
            RETURN g.id as gene_id, g.name as gene_name,
                   p.id as protein_id, p.name as protein_name
            LIMIT 10
        ''')
        print(f'{"Gene ID":<15} {"Gene Name":<30} {"Protein ID":<15} {"Protein Name":<15}')
        print('-'*80)
        for rec in r:
            gname = str(rec["gene_name"])[:30] if rec["gene_name"] else "N/A"
            print(f'{rec["gene_id"]:<15} {gname:<30} {rec["protein_id"]:<15} {rec["protein_name"]:<15}')

        # 5. Orphan analysis
        print('\n--- 5. Orphan Nodes Analysis ---')

        # Genes without proteins
        r = s.run('''
            MATCH (g:Gene)
            OPTIONAL MATCH (g)-[:TRANSLATED_INTO]->(p:Protein)
            WITH g, count(p) as pcount
            RETURN sum(CASE WHEN pcount = 0 THEN 1 ELSE 0 END) as orphan_genes,
                   count(g) as total_genes
        ''')
        rec = r.single()
        print(f'Genes without proteins: {rec["orphan_genes"]:,} / {rec["total_genes"]:,}')

        # Proteins without genes
        r = s.run('''
            MATCH (p:Protein)
            OPTIONAL MATCH (g:Gene)-[:TRANSLATED_INTO]->(p)
            WITH p, count(g) as gcount
            RETURN sum(CASE WHEN gcount = 0 THEN 1 ELSE 0 END) as orphan_proteins,
                   count(p) as total_proteins
        ''')
        rec = r.single()
        print(f'Proteins without genes: {rec["orphan_proteins"]:,} / {rec["total_proteins"]:,}')

        # 6. Multi-protein genes (isoforms)
        print('\n--- 6. Genes with Multiple Proteins (Top 10) ---')
        r = s.run('''
            MATCH (g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
            WITH g, collect(p.name) as proteins, count(p) as cnt
            WHERE cnt > 1
            RETURN g.id as gene, cnt as protein_count, proteins[0..3] as sample_proteins
            ORDER BY cnt DESC
            LIMIT 10
        ''')
        print(f'{"Gene":<15} {"Count":>8} {"Sample Proteins"}')
        print('-'*60)
        for rec in r:
            prots = ", ".join(rec["sample_proteins"][:3])
            print(f'{rec["gene"]:<15} {rec["protein_count"]:>8} {prots}')

    driver.close()
    print('\n' + '='*80)
    print('VALIDATION COMPLETE')
    print('='*80)

if __name__ == "__main__":
    main()
