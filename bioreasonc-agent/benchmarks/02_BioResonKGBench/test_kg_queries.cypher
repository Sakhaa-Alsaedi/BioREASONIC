// =============================================================================
// BioREASONIC Knowledge Graph Validation Queries
// Run these in Neo4j Browser or via Python to validate graph connectivity
// =============================================================================

// -----------------------------------------------------------------------------
// 1. BASIC NODE COUNTS - Check all node types exist
// -----------------------------------------------------------------------------

// Count all node types
MATCH (n)
RETURN labels(n)[0] AS NodeType, count(n) AS Count
ORDER BY Count DESC;

// -----------------------------------------------------------------------------
// 2. RELATIONSHIP COUNTS - Check all relationship types exist
// -----------------------------------------------------------------------------

// Count all relationship types
MATCH ()-[r]->()
RETURN type(r) AS RelationshipType, count(r) AS Count
ORDER BY Count DESC;

// -----------------------------------------------------------------------------
// 3. SCHEMA VALIDATION - Verify expected relationships exist
// -----------------------------------------------------------------------------

// 3a. SNP -> Gene (MAPS_TO)
MATCH (s:SNP)-[r:MAPS_TO]->(g:Gene)
RETURN 'SNP->Gene (MAPS_TO)' AS Relationship, count(r) AS Count;

// 3b. SNP -> Disease (PUTATIVE_CAUSAL_EFFECT)
MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
RETURN 'SNP->Disease (PUTATIVE_CAUSAL_EFFECT)' AS Relationship, count(r) AS Count;

// 3c. Gene -> Disease (INCREASES_RISK_OF)
MATCH (g:Gene)-[r:INCREASES_RISK_OF]->(d:Disease)
RETURN 'Gene->Disease (INCREASES_RISK_OF)' AS Relationship, count(r) AS Count;

// 3d. Gene -> Protein (TRANSLATED_INTO)
MATCH (g:Gene)-[r:TRANSLATED_INTO]->(p:Protein)
RETURN 'Gene->Protein (TRANSLATED_INTO)' AS Relationship, count(r) AS Count;

// -----------------------------------------------------------------------------
// 4. CONNECTIVITY CHECK - Find isolated nodes (no relationships)
// -----------------------------------------------------------------------------

// Find nodes with no relationships (should be minimal or zero)
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] AS NodeType, count(n) AS IsolatedCount;

// -----------------------------------------------------------------------------
// 5. SAMPLE DATA - View example nodes and relationships
// -----------------------------------------------------------------------------

// 5a. Sample Genes
MATCH (g:Gene)
RETURN g.id AS GeneID, g.name AS GeneName, g.symbol AS Symbol
LIMIT 5;

// 5b. Sample Diseases
MATCH (d:Disease)
RETURN d.id AS DiseaseID, d.name AS DiseaseName
LIMIT 5;

// 5c. Sample SNPs
MATCH (s:SNP)
RETURN s.id AS SNPID, s.rsid AS rsID
LIMIT 5;

// 5d. Sample Proteins
MATCH (p:Protein)
RETURN p.id AS ProteinID, p.name AS ProteinName
LIMIT 5;

// -----------------------------------------------------------------------------
// 6. PATH VALIDATION - Test multi-hop connections
// -----------------------------------------------------------------------------

// 6a. Find SNP -> Gene -> Disease paths
MATCH path = (s:SNP)-[:MAPS_TO]->(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
RETURN s.id AS SNP, g.symbol AS Gene, d.name AS Disease
LIMIT 10;

// 6b. Find SNP -> Gene -> Protein paths
MATCH path = (s:SNP)-[:MAPS_TO]->(g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
RETURN s.id AS SNP, g.symbol AS Gene, p.name AS Protein
LIMIT 10;

// 6c. Complete path: SNP -> Gene -> Disease with causal effect
MATCH (s:SNP)-[:MAPS_TO]->(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
WHERE (s)-[:PUTATIVE_CAUSAL_EFFECT]->(d)
RETURN s.id AS SNP, g.symbol AS Gene, d.name AS Disease,
       'Direct + Gene-mediated' AS PathType
LIMIT 10;

// -----------------------------------------------------------------------------
// 7. GRAPH STATISTICS - Overall connectivity metrics
// -----------------------------------------------------------------------------

// Average connections per node type
MATCH (n)
WITH labels(n)[0] AS NodeType, n
OPTIONAL MATCH (n)-[r]-()
WITH NodeType, n, count(r) AS connections
RETURN NodeType,
       count(n) AS TotalNodes,
       avg(connections) AS AvgConnections,
       max(connections) AS MaxConnections,
       min(connections) AS MinConnections;

// -----------------------------------------------------------------------------
// 8. DISEASE-CENTRIC QUERIES - Test disease connectivity
// -----------------------------------------------------------------------------

// Find diseases with most associated genes
MATCH (d:Disease)<-[:INCREASES_RISK_OF]-(g:Gene)
WITH d, count(g) AS geneCount
ORDER BY geneCount DESC
RETURN d.name AS Disease, geneCount AS AssociatedGenes
LIMIT 10;

// Find diseases with most associated SNPs
MATCH (d:Disease)<-[:PUTATIVE_CAUSAL_EFFECT]-(s:SNP)
WITH d, count(s) AS snpCount
ORDER BY snpCount DESC
RETURN d.name AS Disease, snpCount AS AssociatedSNPs
LIMIT 10;

// -----------------------------------------------------------------------------
// 9. GENE-CENTRIC QUERIES - Test gene connectivity
// -----------------------------------------------------------------------------

// Find genes with most relationships
MATCH (g:Gene)
OPTIONAL MATCH (g)-[r]-()
WITH g, count(r) AS totalRels
ORDER BY totalRels DESC
RETURN g.symbol AS Gene, g.name AS GeneName, totalRels AS TotalRelationships
LIMIT 10;

// Find genes connected to both diseases and proteins
MATCH (g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
MATCH (g)-[:TRANSLATED_INTO]->(p:Protein)
RETURN g.symbol AS Gene,
       collect(DISTINCT d.name)[0..3] AS Diseases,
       collect(DISTINCT p.name)[0..3] AS Proteins
LIMIT 10;

// -----------------------------------------------------------------------------
// 10. DATA QUALITY CHECKS
// -----------------------------------------------------------------------------

// 10a. Check for nodes missing required properties
MATCH (g:Gene) WHERE g.symbol IS NULL
RETURN 'Genes without symbol' AS Issue, count(g) AS Count
UNION
MATCH (d:Disease) WHERE d.name IS NULL
RETURN 'Diseases without name' AS Issue, count(d) AS Count
UNION
MATCH (s:SNP) WHERE s.id IS NULL AND s.rsid IS NULL
RETURN 'SNPs without ID' AS Issue, count(s) AS Count;

// 10b. Check for duplicate nodes
MATCH (g:Gene)
WITH g.symbol AS symbol, count(g) AS cnt
WHERE cnt > 1
RETURN 'Duplicate Gene symbols' AS Issue, symbol, cnt
LIMIT 5;

// -----------------------------------------------------------------------------
// 11. QUICK HEALTH CHECK - Single comprehensive query
// -----------------------------------------------------------------------------

CALL {
    MATCH (n) RETURN 'Total Nodes' AS Metric, count(n) AS Value
    UNION ALL
    MATCH ()-[r]->() RETURN 'Total Relationships' AS Metric, count(r) AS Value
    UNION ALL
    MATCH (g:Gene) RETURN 'Genes' AS Metric, count(g) AS Value
    UNION ALL
    MATCH (d:Disease) RETURN 'Diseases' AS Metric, count(d) AS Value
    UNION ALL
    MATCH (s:SNP) RETURN 'SNPs' AS Metric, count(s) AS Value
    UNION ALL
    MATCH (p:Protein) RETURN 'Proteins' AS Metric, count(p) AS Value
}
RETURN Metric, Value;
