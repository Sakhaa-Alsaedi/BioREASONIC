#!/usr/bin/env python3
"""
BioREASONIC Knowledge Graph Connectivity Test
Tests that all nodes are properly linked in the Neo4j database.
"""

import yaml
from pathlib import Path
from neo4j import GraphDatabase

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / 'config' / 'kg_config.yml'

# Load config
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

NEO4J_URI = f"bolt://{config['db_url']}:{config['db_port']}"
NEO4J_USER = config['db_user']
NEO4J_PASSWORD = config['db_password']

# =============================================================================
# Test Queries
# =============================================================================

QUERIES = {
    "1. Node Counts": """
        MATCH (n)
        RETURN labels(n)[0] AS NodeType, count(n) AS Count
        ORDER BY Count DESC
    """,

    "2. Relationship Counts": """
        MATCH ()-[r]->()
        RETURN type(r) AS RelationshipType, count(r) AS Count
        ORDER BY Count DESC
    """,

    "3a. SNP->Gene (MAPS_TO)": """
        MATCH (s:SNP)-[r:MAPS_TO]->(g:Gene)
        RETURN 'SNP->Gene' AS Path, count(r) AS Count
    """,

    "3b. SNP->Disease (PUTATIVE_CAUSAL_EFFECT)": """
        MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
        RETURN 'SNP->Disease' AS Path, count(r) AS Count
    """,

    "3c. Gene->Disease (INCREASES_RISK_OF)": """
        MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
        RETURN 'Gene->Disease' AS Path, count(r) AS Count
    """,

    "3d. Gene->Protein (TRANSLATED_INTO)": """
        MATCH (g:Gene)-[r:TRANSLATED_INTO]->(p:Protein)
        RETURN 'Gene->Protein' AS Path, count(r) AS Count
    """,

    "4. Isolated Nodes (No Relationships)": """
        MATCH (n)
        WHERE NOT (n)--()
        RETURN labels(n)[0] AS NodeType, count(n) AS IsolatedCount
    """,

    "5. Sample SNP->Gene->Disease Paths": """
        MATCH path = (s:SNP)-[:MAPS_TO]->(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
        RETURN s.id AS SNP, g.symbol AS Gene, d.name AS Disease
        LIMIT 5
    """,

    "6. Sample SNP->Gene->Protein Paths": """
        MATCH path = (s:SNP)-[:MAPS_TO]->(g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
        RETURN s.id AS SNP, g.symbol AS Gene, p.name AS Protein
        LIMIT 5
    """,

    "7. Top Diseases by Gene Count": """
        MATCH (d:Disease)<-[:INCREASES_RISK_OF]-(g:Gene)
        WITH d, count(g) AS geneCount
        ORDER BY geneCount DESC
        RETURN d.name AS Disease, geneCount AS Genes
        LIMIT 5
    """,

    "8. Top Genes by Connections": """
        MATCH (g:Gene)
        OPTIONAL MATCH (g)-[r]-()
        WITH g, count(r) AS totalRels
        ORDER BY totalRels DESC
        RETURN g.symbol AS Gene, totalRels AS Connections
        LIMIT 5
    """,

    "9. Graph Health Summary": """
        MATCH (n) WITH count(n) AS nodes
        MATCH ()-[r]->() WITH nodes, count(r) AS rels
        RETURN nodes AS TotalNodes, rels AS TotalRelationships,
               round(toFloat(rels)/nodes, 2) AS AvgRelsPerNode
    """,
}


def run_test(driver, name, query):
    """Run a single test query and display results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)

    try:
        with driver.session() as session:
            result = session.run(query)
            records = list(result)

            if not records:
                print("  No results (empty)")
                return False

            # Get column names
            keys = records[0].keys()

            # Print header
            header = " | ".join(f"{k:<20}" for k in keys)
            print(header)
            print("-" * len(header))

            # Print rows
            for record in records:
                row = " | ".join(f"{str(record[k]):<20}" for k in keys)
                print(row)

            return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    print("="*60)
    print("BioREASONIC Knowledge Graph Connectivity Test")
    print("="*60)
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"User: {NEO4J_USER}")

    # Connect to Neo4j
    print("\nConnecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            encrypted=False
        )

        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            result.single()
        print("Connected successfully!")

    except Exception as e:
        print(f"Connection FAILED: {e}")
        return

    # Run all tests
    passed = 0
    failed = 0

    for name, query in QUERIES.items():
        if run_test(driver, name, query):
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")

    if failed == 0:
        print("\n✓ All connectivity tests passed!")
    else:
        print(f"\n✗ {failed} tests failed - check graph structure")

    driver.close()


if __name__ == "__main__":
    main()
