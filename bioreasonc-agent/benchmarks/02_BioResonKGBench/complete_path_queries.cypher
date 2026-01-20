// =============================================================================
// COMPLETE PATH QUERIES: SNP -> Gene -> Protein -> Disease + Tissue + GO
// Run in Neo4j Browser at http://10.73.107.108:7474
// =============================================================================

// =============================================================================
// GRAPH SCHEMA VISUALIZATION
// =============================================================================
/*
                                    ┌─────────────────┐
                                    │     Tissue      │
                                    └────────▲────────┘
                                             │ ASSOCIATED_WITH
                                             │
┌───────┐  MAPS_TO   ┌───────┐  TRANSLATED   ┌─────────┐  ASSOCIATED   ┌──────────────────┐
│  SNP  │───────────>│ Gene  │──────INTO────>│ Protein │──────WITH────>│ Biological_proc  │
└───────┘            └───────┘               └─────────┘               │ Molecular_func   │
    │                    │                        │                    │ Cellular_comp    │
    │ PUTATIVE_          │ INCREASES_             │ ANNOTATED_         └──────────────────┘
    │ CAUSAL_            │ RISK_OF                │ IN_PATHWAY
    │ EFFECT             │                        │
    ▼                    ▼                        ▼
┌─────────┐         ┌─────────┐              ┌─────────┐
│ Disease │<────────│ Disease │              │ Pathway │
└─────────┘         └─────────┘              └─────────┘
*/

// =============================================================================
// QUERY 1: Complete path for a specific gene (THRB example)
// =============================================================================

// 1a. View all connections for gene THRB
MATCH (snp:SNP)-[r1:MAPS_TO]->(gene:Gene {id: "THRB"})
MATCH (gene)-[r2:TRANSLATED_INTO]->(protein:Protein)
OPTIONAL MATCH (gene)-[r3:INCREASES_RISK_OF]->(disease:Disease)
OPTIONAL MATCH (protein)-[r4:ASSOCIATED_WITH]->(tissue:Tissue)
OPTIONAL MATCH (protein)-[r5:ASSOCIATED_WITH]->(bp:Biological_process)
OPTIONAL MATCH (protein)-[r6:ANNOTATED_IN_PATHWAY]->(pathway:Pathway)
RETURN snp, gene, protein, disease, tissue, bp, pathway,
       r1, r2, r3, r4, r5, r6
LIMIT 50;

// =============================================================================
// QUERY 2: Tabular view of complete path
// =============================================================================

// 2a. Get SNP -> Gene -> Disease -> Protein -> Tissue/GO for high-risk genes
MATCH (snp:SNP)-[:MAPS_TO]->(gene:Gene {node_type: "risk_gene"})
MATCH (gene)-[:TRANSLATED_INTO]->(protein:Protein {node_type: "risk_protein"})
MATCH (gene)-[risk:INCREASES_RISK_OF]->(disease:Disease)
WHERE risk.risk_score > 0.4
WITH snp, gene, protein, disease, risk
LIMIT 20
OPTIONAL MATCH (protein)-[:ASSOCIATED_WITH]->(tissue:Tissue)
WITH snp, gene, protein, disease, risk, collect(DISTINCT tissue.name)[0] as tissue
OPTIONAL MATCH (protein)-[:ASSOCIATED_WITH]->(bp:Biological_process)
WITH snp, gene, protein, disease, risk, tissue, collect(DISTINCT bp.name)[0] as bioProcess
OPTIONAL MATCH (protein)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
RETURN snp.id as SNP,
       gene.id as Gene,
       protein.name as Protein,
       disease.name as Disease,
       round(risk.risk_score, 3) as RiskScore,
       tissue as Tissue,
       bioProcess as GO_Process,
       collect(DISTINCT pw.name)[0] as Pathway
LIMIT 20;

// =============================================================================
// QUERY 3: Visualize complete network for a disease
// =============================================================================

// 3a. All connections for "asthma"
MATCH (disease:Disease {name: "asthma"})
MATCH (gene:Gene)-[r1:INCREASES_RISK_OF]->(disease)
WITH disease, gene, r1
LIMIT 10
OPTIONAL MATCH (snp:SNP)-[r2:MAPS_TO]->(gene)
WITH disease, gene, snp, r1, r2
LIMIT 30
OPTIONAL MATCH (gene)-[r3:TRANSLATED_INTO]->(protein:Protein)
WITH disease, gene, snp, protein, r1, r2, r3
LIMIT 50
OPTIONAL MATCH (protein)-[r4:ASSOCIATED_WITH]->(tissue:Tissue)
WITH disease, gene, snp, protein, tissue, r1, r2, r3, r4
LIMIT 70
RETURN disease, gene, snp, protein, tissue, r1, r2, r3, r4;

// =============================================================================
// QUERY 4: Full biological context for a gene
// =============================================================================

// 4a. Everything about HLA-DMA gene
MATCH (gene:Gene {id: "HLA-DMA"})
OPTIONAL MATCH (snp:SNP)-[r1:MAPS_TO]->(gene)
OPTIONAL MATCH (gene)-[r2:TRANSLATED_INTO]->(protein:Protein)
OPTIONAL MATCH (gene)-[r3:INCREASES_RISK_OF]->(disease:Disease)
OPTIONAL MATCH (protein)-[r4:ASSOCIATED_WITH]->(tissue:Tissue)
OPTIONAL MATCH (protein)-[r5:ASSOCIATED_WITH]->(bp:Biological_process)
OPTIONAL MATCH (protein)-[r6:ASSOCIATED_WITH]->(mf:Molecular_function)
OPTIONAL MATCH (protein)-[r7:ANNOTATED_IN_PATHWAY]->(pathway:Pathway)
RETURN gene, snp, protein, disease, tissue, bp, mf, pathway,
       r1, r2, r3, r4, r5, r6, r7
LIMIT 100;

// =============================================================================
// QUERY 5: SNP complete context (start from SNP)
// =============================================================================

// 5a. Complete path starting from a specific SNP
MATCH (snp:SNP {id: "rs35705950"})
OPTIONAL MATCH (snp)-[r1:MAPS_TO]->(gene:Gene)
OPTIONAL MATCH (snp)-[r2:PUTATIVE_CAUSAL_EFFECT]->(disease1:Disease)
OPTIONAL MATCH (gene)-[r3:INCREASES_RISK_OF]->(disease2:Disease)
OPTIONAL MATCH (gene)-[r4:TRANSLATED_INTO]->(protein:Protein)
OPTIONAL MATCH (protein)-[r5:ASSOCIATED_WITH]->(tissue:Tissue)
OPTIONAL MATCH (protein)-[r6:ASSOCIATED_WITH]->(bp:Biological_process)
OPTIONAL MATCH (protein)-[r7:ANNOTATED_IN_PATHWAY]->(pathway:Pathway)
RETURN snp, gene, disease1, disease2, protein, tissue, bp, pathway,
       r1, r2, r3, r4, r5, r6, r7
LIMIT 50;

// =============================================================================
// QUERY 6: Protein-centric view with all GO terms
// =============================================================================

// 6a. Protein with all functional annotations
MATCH (protein:Protein {name: "THRB"})
OPTIONAL MATCH (gene:Gene)-[r1:TRANSLATED_INTO]->(protein)
OPTIONAL MATCH (protein)-[r2:ASSOCIATED_WITH]->(tissue:Tissue)
OPTIONAL MATCH (protein)-[r3:ASSOCIATED_WITH]->(bp:Biological_process)
OPTIONAL MATCH (protein)-[r4:ASSOCIATED_WITH]->(mf:Molecular_function)
OPTIONAL MATCH (protein)-[r5:ASSOCIATED_WITH]->(cc:Cellular_component)
OPTIONAL MATCH (protein)-[r6:ANNOTATED_IN_PATHWAY]->(pathway:Pathway)
RETURN protein, gene, tissue, bp, mf, cc, pathway,
       r1, r2, r3, r4, r5, r6
LIMIT 100;

// =============================================================================
// QUERY 7: Summary statistics for complete paths
// =============================================================================

// 7a. Count complete paths with all connections
MATCH (snp:SNP)-[:MAPS_TO]->(gene:Gene {node_type: "risk_gene"})
MATCH (gene)-[:TRANSLATED_INTO]->(protein:Protein)
MATCH (gene)-[:INCREASES_RISK_OF]->(disease:Disease)
WITH count(DISTINCT snp) as snps,
     count(DISTINCT gene) as genes,
     count(DISTINCT protein) as proteins,
     count(DISTINCT disease) as diseases
RETURN snps as SNPs_in_paths,
       genes as Genes_in_paths,
       proteins as Proteins_in_paths,
       diseases as Diseases_in_paths;

// =============================================================================
// QUERY 8: Find genes with complete biological context
// =============================================================================

// 8a. Genes that have SNP, Protein, Disease, Tissue, GO, and Pathway
MATCH (snp:SNP)-[:MAPS_TO]->(gene:Gene {node_type: "risk_gene"})
MATCH (gene)-[:TRANSLATED_INTO]->(protein:Protein)
MATCH (gene)-[:INCREASES_RISK_OF]->(disease:Disease)
MATCH (protein)-[:ASSOCIATED_WITH]->(tissue:Tissue)
MATCH (protein)-[:ASSOCIATED_WITH]->(bp:Biological_process)
MATCH (protein)-[:ANNOTATED_IN_PATHWAY]->(pathway:Pathway)
RETURN DISTINCT gene.id as Gene,
       count(DISTINCT snp) as SNPs,
       count(DISTINCT protein) as Proteins,
       count(DISTINCT disease) as Diseases,
       count(DISTINCT tissue) as Tissues,
       count(DISTINCT bp) as GO_Processes,
       count(DISTINCT pathway) as Pathways
ORDER BY Diseases DESC
LIMIT 20;

// =============================================================================
// QUERY 9: Visualization-friendly query (for Neo4j Browser graph view)
// =============================================================================

// 9a. Small network showing all node types
MATCH (gene:Gene {id: "THRB"})
MATCH (gene)-[:TRANSLATED_INTO]->(protein:Protein)
WITH gene, protein LIMIT 1
MATCH (snp:SNP)-[:MAPS_TO]->(gene)
WITH gene, protein, snp LIMIT 3
OPTIONAL MATCH (gene)-[:INCREASES_RISK_OF]->(disease:Disease)
WITH gene, protein, snp, disease LIMIT 5
OPTIONAL MATCH (protein)-[:ASSOCIATED_WITH]->(tissue:Tissue)
WITH gene, protein, snp, disease, tissue LIMIT 7
OPTIONAL MATCH (protein)-[:ASSOCIATED_WITH]->(bp:Biological_process)
WITH gene, protein, snp, disease, tissue, bp LIMIT 10
OPTIONAL MATCH (protein)-[:ANNOTATED_IN_PATHWAY]->(pathway:Pathway)
RETURN gene, protein, snp, disease, tissue, bp, pathway;

// =============================================================================
// QUERY 10: Path from Disease to all upstream nodes
// =============================================================================

// 10a. What causes diabetes? (reverse path)
MATCH (disease:Disease)
WHERE toLower(disease.name) CONTAINS "diabetes" AND toLower(disease.name) CONTAINS "type 2"
WITH disease LIMIT 1
MATCH (gene:Gene)-[r1:INCREASES_RISK_OF]->(disease)
WITH disease, gene, r1
ORDER BY r1.risk_score DESC LIMIT 5
MATCH (gene)-[r2:TRANSLATED_INTO]->(protein:Protein)
WITH disease, gene, protein, r1, r2 LIMIT 10
OPTIONAL MATCH (snp:SNP)-[r3:MAPS_TO]->(gene)
WITH disease, gene, protein, snp, r1, r2, r3 LIMIT 20
OPTIONAL MATCH (protein)-[r4:ASSOCIATED_WITH]->(bp:Biological_process)
WITH disease, gene, protein, snp, bp, r1, r2, r3, r4 LIMIT 30
RETURN disease, gene, protein, snp, bp, r1, r2, r3, r4;

// =============================================================================
// END OF COMPLETE PATH QUERIES
// =============================================================================
