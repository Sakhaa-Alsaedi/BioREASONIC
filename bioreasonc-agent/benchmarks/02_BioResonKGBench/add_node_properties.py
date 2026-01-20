#!/usr/bin/env python3
"""
Add descriptive properties to KG nodes without renaming them.
This preserves all existing code while adding semantic context.

Properties added:
- Gene: node_type = "risk_gene" (for genes with INCREASES_RISK_OF)
- Protein: node_type = "risk_protein" (for proteins linked to risk genes)
- SNP: node_type = "causal_snp" (for SNPs with PUTATIVE_CAUSAL_EFFECT)
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
    print('ADDING DESCRIPTIVE PROPERTIES TO KG NODES')
    print('='*80)
    print()
    print('This adds properties WITHOUT renaming nodes (preserves all existing code)')
    print()

    with driver.session() as s:

        # 1. Add node_type to Risk Genes
        print('--- 1. Adding node_type="risk_gene" to genes with INCREASES_RISK_OF ---')
        result = s.run('''
            MATCH (g:Gene)-[:INCREASES_RISK_OF]->(:Disease)
            WITH DISTINCT g
            SET g.node_type = "risk_gene"
            RETURN count(g) as updated
        ''')
        count = result.single()['updated']
        print(f'Updated {count:,} genes with node_type="risk_gene"')

        # 2. Add node_type to Risk Proteins (linked to risk genes)
        print('\n--- 2. Adding node_type="risk_protein" to proteins linked to risk genes ---')
        result = s.run('''
            MATCH (p:Protein)<-[:TRANSLATED_INTO]-(g:Gene)-[:INCREASES_RISK_OF]->(:Disease)
            WITH DISTINCT p
            SET p.node_type = "risk_protein"
            RETURN count(p) as updated
        ''')
        count = result.single()['updated']
        print(f'Updated {count:,} proteins with node_type="risk_protein"')

        # 3. Add node_type to Causal SNPs
        print('\n--- 3. Adding node_type="causal_snp" to SNPs with PUTATIVE_CAUSAL_EFFECT ---')
        result = s.run('''
            MATCH (s:SNP)-[:PUTATIVE_CAUSAL_EFFECT]->(:Disease)
            WITH DISTINCT s
            SET s.node_type = "causal_snp"
            RETURN count(s) as updated
        ''')
        count = result.single()['updated']
        print(f'Updated {count:,} SNPs with node_type="causal_snp"')

        # 4. Add node_type to SNPs that map to risk genes
        print('\n--- 4. Adding node_type="risk_snp" to SNPs mapping to risk genes ---')
        result = s.run('''
            MATCH (s:SNP)-[:MAPS_TO]->(g:Gene)-[:INCREASES_RISK_OF]->(:Disease)
            WITH DISTINCT s
            SET s.node_type = COALESCE(s.node_type, "") +
                CASE WHEN s.node_type IS NULL OR s.node_type = "" THEN "risk_snp"
                     WHEN NOT s.node_type CONTAINS "risk_snp" THEN ",risk_snp"
                     ELSE "" END
            RETURN count(s) as updated
        ''')
        count = result.single()['updated']
        print(f'Updated {count:,} SNPs with risk_snp designation')

        # 5. Verify the updates
        print('\n' + '='*80)
        print('VERIFICATION')
        print('='*80)

        print('\n--- Node Type Distribution ---')

        # Genes
        result = s.run('''
            MATCH (g:Gene)
            RETURN g.node_type as type, count(g) as count
            ORDER BY count DESC
        ''')
        print('\nGenes:')
        for rec in result:
            t = rec['type'] if rec['type'] else '(no type)'
            print(f'  {t}: {rec["count"]:,}')

        # Proteins
        result = s.run('''
            MATCH (p:Protein)
            RETURN p.node_type as type, count(p) as count
            ORDER BY count DESC
        ''')
        print('\nProteins:')
        for rec in result:
            t = rec['type'] if rec['type'] else '(no type)'
            print(f'  {t}: {rec["count"]:,}')

        # SNPs
        result = s.run('''
            MATCH (s:SNP)
            RETURN s.node_type as type, count(s) as count
            ORDER BY count DESC
        ''')
        print('\nSNPs:')
        for rec in result:
            t = rec['type'] if rec['type'] else '(no type)'
            print(f'  {t}: {rec["count"]:,}')

        # 6. Sample queries showing the new properties
        print('\n' + '='*80)
        print('SAMPLE QUERIES WITH NEW PROPERTIES')
        print('='*80)

        print('\n--- Risk Genes with their Proteins ---')
        result = s.run('''
            MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein {node_type: "risk_protein"})
            RETURN g.id as gene, p.name as protein, g.node_type as gene_type, p.node_type as protein_type
            LIMIT 5
        ''')
        print(f'{"Gene":<15} {"Protein":<15} {"Gene Type":<15} {"Protein Type":<15}')
        print('-'*60)
        for rec in result:
            print(f'{rec["gene"]:<15} {rec["protein"]:<15} {rec["gene_type"]:<15} {rec["protein_type"]:<15}')

        print('\n--- Causal SNPs with Diseases ---')
        result = s.run('''
            MATCH (s:SNP {node_type: "causal_snp"})-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
            RETURN s.id as snp, d.name as disease, s.node_type as snp_type, r.causal_score as score
            ORDER BY r.causal_score DESC
            LIMIT 5
        ''')
        print(f'{"SNP":<15} {"Disease":<30} {"SNP Type":<15} {"Score":>10}')
        print('-'*75)
        for rec in result:
            disease = str(rec["disease"])[:30] if rec["disease"] else "N/A"
            score = f'{rec["score"]:.3f}' if rec["score"] else "N/A"
            print(f'{rec["snp"]:<15} {disease:<30} {rec["snp_type"]:<15} {score:>10}')

    driver.close()
    print('\n' + '='*80)
    print('PROPERTY ADDITION COMPLETE')
    print('='*80)
    print()
    print('New properties added:')
    print('  - Gene.node_type = "risk_gene"')
    print('  - Protein.node_type = "risk_protein"')
    print('  - SNP.node_type = "causal_snp" or "risk_snp"')
    print()
    print('All existing code continues to work unchanged!')
    print('='*80)

if __name__ == "__main__":
    main()
