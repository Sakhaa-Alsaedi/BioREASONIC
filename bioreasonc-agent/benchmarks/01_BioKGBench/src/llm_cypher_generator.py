#!/usr/bin/env python3
"""
LLM-based Cypher Query Generator for KGQA.
Uses LLMs to understand questions and generate appropriate Cypher queries.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from llm_client import LLMClient

# Knowledge Graph Schema for the LLM
KG_SCHEMA = """
## Knowledge Graph Schema

### Node Types:
- Gene: {id (gene symbol like BRCA1, TP53), name, ensembl_id}
- Protein: {id (UniProt ID like P04637), name, accession}
- Disease: {id (DOID like DOID:2841), name}
- Pathway: {id, name, source (Reactome/SMPDB)}
- Biological_process: {id (GO term like GO:0006915), name, description}
- Molecular_function: {id (GO term), name, description}
- Cellular_component: {id (GO term), name, description}
- Tissue: {id, name}
- SNP: {id (rsID like rs12345), chromosome, position}

### Relationships:
- (Gene)-[:TRANSLATED_INTO]->(Protein)
- (Protein)-[:ASSOCIATED_WITH]->(Disease)
- (Protein)-[:ASSOCIATED_WITH]->(Biological_process)
- (Protein)-[:ASSOCIATED_WITH]->(Molecular_function)
- (Protein)-[:ASSOCIATED_WITH]->(Cellular_component)
- (Protein)-[:ASSOCIATED_WITH]->(Tissue)
- (Protein)-[:ANNOTATED_IN_PATHWAY]->(Pathway)
- (Protein)-[r]-(Protein) WHERE type(r) IN ['ACTS_ON', 'CURATED_INTERACTS_WITH', 'INTERACTS_WITH']  # PPI
- (Gene)-[:INCREASES_RISK_OF]->(Disease)
- (SNP)-[:MAPS_TO]->(Gene)
- (SNP)-[:PUTATIVE_CAUSAL_EFFECT]->(Disease)

### Important Notes:
- Gene.id contains the gene SYMBOL (e.g., "TP53", "BRCA1")
- Protein.id contains UniProt ID (e.g., "P04637", "P38398")
- For GO terms, always return the NAME not the ID as the answer
- Use case-insensitive matching when possible
- CRITICAL for protein-protein interactions: Use BIDIRECTIONAL matching with pattern:
  MATCH (p:Protein {id: $id})-[r]-(p2:Protein) WHERE type(r) IN ['ACTS_ON', 'CURATED_INTERACTS_WITH', 'INTERACTS_WITH']
  This ensures you capture all interactions regardless of direction.
"""

CYPHER_GENERATION_PROMPT = """You are an expert at converting natural language questions into Neo4j Cypher queries.

{schema}

## Task
Convert the following question into a Cypher query. The query should:
1. Return answers as names/descriptions, not just IDs
2. Use DISTINCT to avoid duplicates
3. Limit results to 50
4. ONLY use ORDER BY when relationship has a score property. DO NOT order alphabetically.

## Question
{question}

## Response Format
Return a JSON object with:
- "cypher": The Cypher query string
- "parameters": Object with query parameters (if any)
- "description": Brief description of what the query does
- "answer_field": Which field contains the main answer (usually "answer" or "name")

Example for protein-protein interaction (NO ordering):
{{
    "cypher": "MATCH (p:Protein {{id: 'P12345'}})-[r]-(p2:Protein) WHERE type(r) IN ['ACTS_ON', 'CURATED_INTERACTS_WITH', 'INTERACTS_WITH'] RETURN DISTINCT p2.name AS answer, p2.id AS id LIMIT 50",
    "parameters": {{}},
    "description": "Find proteins that interact with P12345",
    "answer_field": "answer"
}}

Example for GO terms (order by score if available):
{{
    "cypher": "MATCH (g:Gene {{id: $gene_id}})-[:TRANSLATED_INTO]->(p:Protein)-[r:ASSOCIATED_WITH]->(mf:Molecular_function) WITH mf.name AS answer, mf.id AS id, CASE WHEN r.score IS NOT NULL THEN toFloat(r.score) ELSE 0.5 END AS score RETURN DISTINCT answer, id ORDER BY score DESC LIMIT 50",
    "parameters": {{"gene_id": "TP53"}},
    "description": "Find molecular functions of proteins from gene TP53",
    "answer_field": "answer"
}}
"""

ENTITY_EXTRACTION_PROMPT = """Extract entities from this biomedical question.

## Question
{question}

## Entity Types to Extract:
- gene: Gene symbols (e.g., TP53, BRCA1, RND3)
- protein: UniProt IDs (e.g., P04637, Q9NYB9)
- disease: Disease names or DOID IDs
- pathway: Pathway names
- go_term: GO term IDs or names

## Response Format
Return a JSON object with extracted entities:
{{
    "entities": {{
        "gene": ["gene1", "gene2"],
        "protein": ["P12345"],
        ...
    }},
    "question_type": "one-hop" | "multi-hop" | "conjunction",
    "target_type": "what we're looking for (e.g., molecular_function, disease, pathway)"
}}
"""


@dataclass
class LLMCypherQuery:
    """Result from LLM-based Cypher generation."""
    query: str
    parameters: Dict[str, str]
    description: str
    answer_field: str


class LLMCypherGenerator:
    """Generate Cypher queries using an LLM."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.schema = KG_SCHEMA

    def extract_entities(self, question: str) -> Dict:
        """Extract entities from a question using LLM."""
        prompt = ENTITY_EXTRACTION_PROMPT.format(question=question)

        try:
            result = self.llm.generate_json(prompt)
            return result
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return {"entities": {}, "question_type": "unknown", "target_type": "unknown"}

    def generate_cypher(self, question: str, metadata: Dict = None) -> Optional[LLMCypherQuery]:
        """Generate a Cypher query for a question using LLM."""
        prompt = CYPHER_GENERATION_PROMPT.format(
            schema=self.schema,
            question=question
        )

        # Add metadata context if available
        if metadata:
            if metadata.get('entities'):
                prompt += f"\n\nKnown entities: {metadata['entities']}"
            if metadata.get('type'):
                prompt += f"\nQuestion type: {metadata['type']}"

        try:
            result = self.llm.generate_json(prompt)

            return LLMCypherQuery(
                query=result.get('cypher', ''),
                parameters=result.get('parameters', {}),
                description=result.get('description', ''),
                answer_field=result.get('answer_field', 'answer')
            )
        except Exception as e:
            print(f"Cypher generation error: {e}")
            return None

    def generate_with_retry(self, question: str, metadata: Dict = None,
                           max_retries: int = 2) -> Optional[LLMCypherQuery]:
        """Generate Cypher with retry on failure."""
        for attempt in range(max_retries):
            result = self.generate_cypher(question, metadata)
            if result and result.query:
                return result

        return None


# Specialized prompts for different question types
CONJUNCTION_PROMPT = """Generate a Cypher query to find entities shared by multiple proteins.
Return your answer as a JSON object.

{schema}

## Question
{question}

The query should find {target_type} that are associated with BOTH/ALL mentioned proteins.
Use INTERSECTION pattern: match from each protein separately and find common results.

## Response Format (JSON)
{{
    "cypher": "MATCH (p1:Protein {{id: 'ID1'}})-[:ASSOCIATED_WITH]->(x) MATCH (p2:Protein {{id: 'ID2'}})-[:ASSOCIATED_WITH]->(x) RETURN DISTINCT x.name AS answer, x.id AS id LIMIT 50",
    "parameters": {{}},
    "description": "...",
    "answer_field": "answer"
}}
"""

MULTIHOP_PROMPT = """Generate a Cypher query for a multi-hop question.
Return your answer as a JSON object.

{schema}

## Question
{question}

This is a multi-hop question requiring traversal through multiple relationships
(e.g., Gene -> Protein -> Molecular_function).
For GO terms (Molecular_function, Biological_process, Cellular_component), return the NAME as the answer.
Use scoring if available on relationships.

## Response Format (JSON)
{{
    "cypher": "MATCH (g:Gene {{id: 'GENE'}})-[:TRANSLATED_INTO]->(p:Protein)-[r:ASSOCIATED_WITH]->(mf:Molecular_function) WITH mf.name AS answer, mf.id AS id, CASE WHEN r.score IS NOT NULL THEN toFloat(r.score) ELSE 0.5 END AS score RETURN DISTINCT answer, id ORDER BY score DESC LIMIT 50",
    "parameters": {{}},
    "description": "...",
    "answer_field": "answer"
}}
"""


class AdvancedLLMCypherGenerator(LLMCypherGenerator):
    """Advanced generator with question-type-specific prompts."""

    def generate_cypher(self, question: str, metadata: Dict = None) -> Optional[LLMCypherQuery]:
        """Generate Cypher with question-type-specific prompts."""
        qtype = metadata.get('type', 'one-hop') if metadata else 'one-hop'

        if qtype == 'conjunction':
            # Extract target type from question
            target = self._infer_target(question)
            prompt = CONJUNCTION_PROMPT.format(
                schema=self.schema,
                question=question,
                target_type=target
            )
        elif qtype == 'multi-hop':
            prompt = MULTIHOP_PROMPT.format(
                schema=self.schema,
                question=question
            )
        else:
            # Use standard prompt
            prompt = CYPHER_GENERATION_PROMPT.format(
                schema=self.schema,
                question=question
            )

        if metadata and metadata.get('entities'):
            prompt += f"\n\nKnown entities: {metadata['entities']}"

        try:
            result = self.llm.generate_json(prompt)
            return LLMCypherQuery(
                query=result.get('cypher', ''),
                parameters=result.get('parameters', {}),
                description=result.get('description', ''),
                answer_field=result.get('answer_field', 'answer')
            )
        except Exception as e:
            print(f"Cypher generation error: {e}")
            return None

    def _infer_target(self, question: str) -> str:
        """Infer target type from question text."""
        q_lower = question.lower()
        if 'biological process' in q_lower:
            return 'Biological_process'
        elif 'molecular function' in q_lower:
            return 'Molecular_function'
        elif 'cellular component' in q_lower:
            return 'Cellular_component'
        elif 'pathway' in q_lower:
            return 'Pathway'
        elif 'disease' in q_lower:
            return 'Disease'
        elif 'tissue' in q_lower:
            return 'Tissue'
        return 'unknown'


# Test
if __name__ == "__main__":
    import yaml
    import os
    from llm_client import create_llm_client

    # Load config
    config_path = "config.yaml"
    if os.path.exists("config.local.yaml"):
        config_path = "config.local.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    llm_config = config.get('llm', {})

    if llm_config.get('provider') == 'none':
        print("LLM provider set to 'none'. Update config.yaml to test.")
    else:
        try:
            client = create_llm_client(llm_config)
            generator = AdvancedLLMCypherGenerator(client)

            # Test questions
            test_questions = [
                "What is the molecular function of the protein translated from the gene TP53?",
                "What proteins does P68133 interact with?",
                "Which pathway are proteins P02778 and P25106 both annotated in?",
            ]

            for q in test_questions:
                print(f"\nQ: {q}")
                result = generator.generate_cypher(q)
                if result:
                    print(f"Cypher: {result.query[:100]}...")
                    print(f"Params: {result.parameters}")
                else:
                    print("Failed to generate query")

        except Exception as e:
            print(f"Error: {e}")
