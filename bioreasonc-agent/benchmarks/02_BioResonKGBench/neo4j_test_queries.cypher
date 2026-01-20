// =============================================================================
// BioREASONIC Knowledge Graph - Comprehensive Test Queries
// Run these in Neo4j Browser at http://10.73.107.108:7474
// =============================================================================

// =============================================================================
// SECTION 1: GRAPH OVERVIEW
// =============================================================================

// 1.1 - Total node and relationship counts
MATCH (n)
WITH count(n) as nodes
MATCH ()-[r]->()
RETURN nodes as TotalNodes, count(r) as TotalRelationships;

// 1.2 - Node counts by type
MATCH (n)
RETURN labels(n)[0] as NodeType, count(n) as Count
ORDER BY Count DESC;

// 1.3 - Relationship counts by type
MATCH ()-[r]->()
RETURN type(r) as RelationType, count(r) as Count
ORDER BY Count DESC;

// 1.4 - Node type property distribution
MATCH (n)
WHERE n.node_type IS NOT NULL
RETURN labels(n)[0] as NodeLabel, n.node_type as NodeType, count(n) as Count
ORDER BY NodeLabel, Count DESC;

// =============================================================================
// SECTION 2: VISUALIZE GRAPH SCHEMA (Small Sample)
// =============================================================================

// 2.1 - Sample graph showing ALL relationship types (limit for visualization)
MATCH (g:Gene)-[r1:INCREASES_RISK_OF]->(d:Disease)
WITH g, d, r1 LIMIT 1
MATCH (s:SNP)-[r2:MAPS_TO]->(g)
WITH g, d, s, r1, r2 LIMIT 1
MATCH (s)-[r3:PUTATIVE_CAUSAL_EFFECT]->(d2:Disease)
WITH g, d, s, r1, r2, r3, d2 LIMIT 1
MATCH (g)-[r4:TRANSLATED_INTO]->(p:Protein)
WITH g, d, s, p, r1, r2, r3, r4, d2 LIMIT 1
RETURN g, d, s, p, d2, r1, r2, r3, r4;

// 2.2 - Complete path visualization: SNP -> Gene -> Protein -> Disease
MATCH path = (s:SNP)-[:MAPS_TO]->(g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
WHERE (g)-[:INCREASES_RISK_OF]->(:Disease)
WITH s, g, p, path LIMIT 5
MATCH (g)-[r:INCREASES_RISK_OF]->(d:Disease)
RETURN s, g, p, d, r LIMIT 10;

// =============================================================================
// SECTION 3: TEST RISK GENES (node_type = "risk_gene")
// =============================================================================

// 3.1 - Count risk genes
MATCH (g:Gene {node_type: "risk_gene"})
RETURN count(g) as RiskGeneCount;

// 3.2 - Sample risk genes with their diseases
MATCH (g:Gene {node_type: "risk_gene"})-[r:INCREASES_RISK_OF]->(d:Disease)
RETURN g.id as Gene, g.name as GeneName, d.name as Disease,
       r.risk_score as RiskScore, r.evidence_score as EvidenceScore
ORDER BY r.risk_score DESC
LIMIT 20;

// 3.3 - Visualize risk genes network (small sample)
MATCH (g:Gene {node_type: "risk_gene"})-[r:INCREASES_RISK_OF]->(d:Disease)
WHERE r.risk_score > 0.8
RETURN g, d, r
LIMIT 50;

// 3.4 - Risk genes with most disease associations
MATCH (g:Gene {node_type: "risk_gene"})-[:INCREASES_RISK_OF]->(d:Disease)
WITH g, count(d) as diseaseCount, collect(d.name)[0..5] as sampleDiseases
ORDER BY diseaseCount DESC
RETURN g.id as Gene, diseaseCount as DiseaseCount, sampleDiseases
LIMIT 15;

// =============================================================================
// SECTION 4: TEST RISK PROTEINS (node_type = "risk_protein")
// =============================================================================

// 4.1 - Count risk proteins
MATCH (p:Protein {node_type: "risk_protein"})
RETURN count(p) as RiskProteinCount;

// 4.2 - Risk proteins linked to risk genes
MATCH (p:Protein {node_type: "risk_protein"})<-[:TRANSLATED_INTO]-(g:Gene {node_type: "risk_gene"})
RETURN p.id as ProteinID, p.name as ProteinName, g.id as GeneID
LIMIT 20;

// 4.3 - Visualize Gene-Protein links for risk genes
MATCH (g:Gene {node_type: "risk_gene"})-[r:TRANSLATED_INTO]->(p:Protein {node_type: "risk_protein"})
RETURN g, p, r
LIMIT 30;

// 4.4 - Verify gene name = protein name matching
MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein {node_type: "risk_protein"})
WHERE toLower(g.id) = toLower(p.name)
RETURN g.id as Gene, p.name as Protein, "MATCH" as Status
LIMIT 10
UNION
MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein {node_type: "risk_protein"})
WHERE toLower(g.id) <> toLower(p.name)
RETURN g.id as Gene, p.name as Protein, "NO MATCH" as Status
LIMIT 10;

// =============================================================================
// SECTION 5: TEST CAUSAL SNPs (node_type contains "causal_snp")
// =============================================================================

// 5.1 - Count causal SNPs
MATCH (s:SNP)
WHERE s.node_type CONTAINS "causal_snp"
RETURN count(s) as CausalSNPCount;

// 5.2 - Causal SNPs with disease associations
MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
WHERE s.node_type CONTAINS "causal_snp"
RETURN s.id as SNP, d.name as Disease, r.pvalue as PValue,
       r.causal_score as CausalScore, s.node_type as SNPType
ORDER BY r.causal_score DESC
LIMIT 20;

// 5.3 - Visualize causal SNP-Disease network
MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
WHERE s.node_type CONTAINS "causal_snp" AND r.causal_score > 0.9
RETURN s, d, r
LIMIT 50;

// 5.4 - SNPs that are both causal AND map to risk genes
MATCH (s:SNP)-[:MAPS_TO]->(g:Gene {node_type: "risk_gene"})
WHERE s.node_type CONTAINS "causal_snp" AND s.node_type CONTAINS "risk_snp"
MATCH (s)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
RETURN s.id as SNP, g.id as Gene, d.name as Disease, s.node_type as SNPType
LIMIT 20;

// =============================================================================
// SECTION 6: COMPLETE PATH TESTS
// =============================================================================

// 6.1 - Full path: SNP -> Gene -> Disease (via MAPS_TO + INCREASES_RISK_OF)
MATCH path = (s:SNP)-[:MAPS_TO]->(g:Gene)-[:INCREASES_RISK_OF]->(d:Disease)
WHERE g.node_type = "risk_gene"
RETURN s.id as SNP, g.id as Gene, d.name as Disease
LIMIT 20;

// 6.2 - Full path: SNP -> Gene -> Protein
MATCH path = (s:SNP)-[:MAPS_TO]->(g:Gene)-[:TRANSLATED_INTO]->(p:Protein)
WHERE g.node_type = "risk_gene" AND p.node_type = "risk_protein"
RETURN s.id as SNP, g.id as Gene, p.name as Protein
LIMIT 20;

// 6.3 - Complete chain: SNP -> Gene -> Protein, Gene -> Disease
MATCH (s:SNP)-[:MAPS_TO]->(g:Gene {node_type: "risk_gene"})
MATCH (g)-[:TRANSLATED_INTO]->(p:Protein {node_type: "risk_protein"})
MATCH (g)-[r:INCREASES_RISK_OF]->(d:Disease)
RETURN s.id as SNP, g.id as Gene, p.name as Protein, d.name as Disease, r.risk_score as RiskScore
ORDER BY r.risk_score DESC
LIMIT 20;

// 6.4 - Visualize complete chain (for Neo4j Browser graph view)
MATCH (s:SNP)-[r1:MAPS_TO]->(g:Gene {node_type: "risk_gene"})
MATCH (g)-[r2:TRANSLATED_INTO]->(p:Protein {node_type: "risk_protein"})
MATCH (g)-[r3:INCREASES_RISK_OF]->(d:Disease)
WHERE r3.risk_score > 0.8
RETURN s, g, p, d, r1, r2, r3
LIMIT 30;

// =============================================================================
// SECTION 7: DISEASE-CENTRIC QUERIES
// =============================================================================

// 7.1 - Diseases with most risk genes
MATCH (d:Disease)<-[:INCREASES_RISK_OF]-(g:Gene {node_type: "risk_gene"})
WITH d, count(g) as geneCount
ORDER BY geneCount DESC
RETURN d.id as DiseaseID, d.name as DiseaseName, geneCount as RiskGeneCount
LIMIT 15;

// 7.2 - Diseases with most causal SNPs
MATCH (d:Disease)<-[:PUTATIVE_CAUSAL_EFFECT]-(s:SNP)
WHERE s.node_type CONTAINS "causal_snp"
WITH d, count(s) as snpCount
ORDER BY snpCount DESC
RETURN d.id as DiseaseID, d.name as DiseaseName, snpCount as CausalSNPCount
LIMIT 15;

// 7.3 - Visualize a specific disease network (e.g., diabetes)
MATCH (d:Disease)
WHERE toLower(d.name) CONTAINS "diabetes"
WITH d LIMIT 1
MATCH (g:Gene)-[r1:INCREASES_RISK_OF]->(d)
OPTIONAL MATCH (s:SNP)-[r2:PUTATIVE_CAUSAL_EFFECT]->(d)
OPTIONAL MATCH (g)-[r3:TRANSLATED_INTO]->(p:Protein)
RETURN d, g, s, p, r1, r2, r3
LIMIT 50;

// =============================================================================
// SECTION 8: VALIDATION QUERIES
// =============================================================================

// 8.1 - Verify all risk genes have INCREASES_RISK_OF relationship
MATCH (g:Gene {node_type: "risk_gene"})
WHERE NOT (g)-[:INCREASES_RISK_OF]->(:Disease)
RETURN count(g) as RiskGenesWithoutDisease;
// Should return 0

// 8.2 - Verify all risk proteins are linked to risk genes
MATCH (p:Protein {node_type: "risk_protein"})
WHERE NOT (:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p)
RETURN count(p) as OrphanRiskProteins;
// Should return 0

// 8.3 - Verify all causal SNPs have PUTATIVE_CAUSAL_EFFECT
MATCH (s:SNP)
WHERE s.node_type CONTAINS "causal_snp"
  AND NOT (s)-[:PUTATIVE_CAUSAL_EFFECT]->(:Disease)
RETURN count(s) as CausalSNPsWithoutEffect;
// Should return 0

// 8.4 - Check for orphan nodes (no relationships)
MATCH (n)
WHERE NOT (n)--()
RETURN labels(n)[0] as NodeType, count(n) as OrphanCount;

// =============================================================================
// SECTION 9: STATISTICS SUMMARY
// =============================================================================

// 9.1 - Complete KG Statistics
CALL {
    MATCH (g:Gene) RETURN "Genes" as Type, count(g) as Total,
        sum(CASE WHEN g.node_type = "risk_gene" THEN 1 ELSE 0 END) as WithType
    UNION ALL
    MATCH (p:Protein) RETURN "Proteins" as Type, count(p) as Total,
        sum(CASE WHEN p.node_type = "risk_protein" THEN 1 ELSE 0 END) as WithType
    UNION ALL
    MATCH (s:SNP) RETURN "SNPs" as Type, count(s) as Total,
        sum(CASE WHEN s.node_type IS NOT NULL THEN 1 ELSE 0 END) as WithType
    UNION ALL
    MATCH (d:Disease) RETURN "Diseases" as Type, count(d) as Total, 0 as WithType
}
RETURN Type, Total, WithType,
       round(toFloat(WithType)/Total * 100, 1) as PercentWithType
ORDER BY Total DESC;

// =============================================================================
// SECTION 10: SAMPLE CYPHER FOR BENCHMARK QUESTIONS
// =============================================================================

// 10.1 - Structure (S): What genes are risk factors for a disease?
MATCH (g:Gene {node_type: "risk_gene"})-[r:INCREASES_RISK_OF]->(d:Disease)
WHERE toLower(d.name) CONTAINS "diabetes"
RETURN g.id as Gene, d.name as Disease, r.risk_score as RiskScore
ORDER BY r.risk_score DESC
LIMIT 10;

// 10.2 - Risk (R): What is the p-value for SNP-Disease association?
MATCH (s:SNP)-[r:PUTATIVE_CAUSAL_EFFECT]->(d:Disease)
WHERE s.node_type CONTAINS "causal_snp"
RETURN s.id as SNP, d.name as Disease, r.pvalue as PValue, r.causal_score as CausalScore
ORDER BY r.pvalue ASC
LIMIT 10;

// 10.3 - Causal (C): What is the evidence level for Gene-Disease?
MATCH (g:Gene {node_type: "risk_gene"})-[r:INCREASES_RISK_OF]->(d:Disease)
RETURN g.id as Gene, d.name as Disease, r.evidence_score as EvidenceScore,
       CASE
           WHEN r.evidence_score >= 0.8 THEN "very_strong"
           WHEN r.evidence_score >= 0.6 THEN "strong"
           WHEN r.evidence_score >= 0.4 THEN "moderate"
           WHEN r.evidence_score >= 0.2 THEN "suggestive"
           ELSE "weak"
       END as EvidenceLevel
ORDER BY r.evidence_score DESC
LIMIT 10;

// 10.4 - Mechanism (M): What protein does a gene translate into?
MATCH (g:Gene {node_type: "risk_gene"})-[:TRANSLATED_INTO]->(p:Protein {node_type: "risk_protein"})
RETURN g.id as Gene, g.name as GeneName, p.id as ProteinID, p.name as ProteinName
LIMIT 10;

// =============================================================================
// END OF TEST QUERIES
// =============================================================================
