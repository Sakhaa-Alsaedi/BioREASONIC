#!/usr/bin/env python3
"""
Cypher Query Generator - Convert parsed questions into Cypher queries.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from question_parser import ParsedQuestion, QuestionType, RelationType

@dataclass
class CypherQuery:
    """Represents a generated Cypher query."""
    query: str
    parameters: Dict[str, str]
    description: str
    return_fields: List[str]

class CypherGenerator:
    """Generate Cypher queries from parsed questions."""

    # Query templates for different relationship types
    QUERY_TEMPLATES = {
        # One-hop queries
        RelationType.PROTEIN_INTERACTION: {
            'query': """
                MATCH (p:Protein {id: $protein_id})-[r]-(p2:Protein)
                WHERE type(r) IN ['INTERACTS_WITH', 'ACTS_ON', 'CURATED_INTERACTS_WITH']
                RETURN DISTINCT p2.id AS answer, p2.name AS name
                LIMIT 50
            """,
            'description': 'Find proteins that interact with the given protein',
            'return_fields': ['answer', 'name']
        },

        RelationType.PROTEIN_DISEASE: {
            'query': """
                MATCH (p:Protein {id: $protein_id})-[:ASSOCIATED_WITH]->(d:Disease)
                RETURN DISTINCT d.id AS answer, d.name AS name
                LIMIT 50
            """,
            'description': 'Find diseases associated with the given protein',
            'return_fields': ['answer', 'name']
        },

        RelationType.PROTEIN_PATHWAY: {
            'query': """
                MATCH (p:Protein {id: $protein_id})-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
                RETURN DISTINCT pw.id AS answer, pw.name AS name, pw.source AS source
                LIMIT 50
            """,
            'description': 'Find pathways the protein is annotated in',
            'return_fields': ['answer', 'name', 'source']
        },

        RelationType.PROTEIN_BIOLOGICAL_PROCESS: {
            'query': """
                MATCH (p:Protein {id: $protein_id})-[:ASSOCIATED_WITH]->(bp:Biological_process)
                RETURN DISTINCT bp.id AS answer, bp.name AS name
                LIMIT 50
            """,
            'description': 'Find biological processes associated with the protein',
            'return_fields': ['answer', 'name']
        },

        RelationType.PROTEIN_MOLECULAR_FUNCTION: {
            'query': """
                MATCH (p:Protein {id: $protein_id})-[:ASSOCIATED_WITH]->(mf:Molecular_function)
                RETURN DISTINCT mf.id AS answer, mf.name AS name
                LIMIT 50
            """,
            'description': 'Find molecular functions of the protein',
            'return_fields': ['answer', 'name']
        },

        RelationType.PROTEIN_CELLULAR_COMPONENT: {
            'query': """
                MATCH (p:Protein {id: $protein_id})-[:ASSOCIATED_WITH]->(cc:Cellular_component)
                RETURN DISTINCT cc.id AS answer, cc.name AS name
                LIMIT 50
            """,
            'description': 'Find cellular components where the protein is located',
            'return_fields': ['answer', 'name']
        },

        RelationType.PROTEIN_TISSUE: {
            'query': """
                MATCH (p:Protein {id: $protein_id})-[:ASSOCIATED_WITH]->(t:Tissue)
                RETURN DISTINCT t.id AS answer, t.name AS name
                LIMIT 50
            """,
            'description': 'Find tissues where the protein is expressed',
            'return_fields': ['answer', 'name']
        },

        RelationType.GENE_PROTEIN: {
            'query': """
                MATCH (g:Gene {id: $gene_id})-[:TRANSLATED_INTO]->(p:Protein)
                RETURN DISTINCT p.id AS answer, p.name AS name
                LIMIT 10
            """,
            'description': 'Find proteins translated from the gene',
            'return_fields': ['answer', 'name']
        },
    }

    # Multi-hop query templates with scoring for ranking
    # NOTE: 'answer' field MUST contain the name (not ID) to match benchmark format
    MULTIHOP_TEMPLATES = {
        RelationType.GENE_MOLECULAR_FUNCTION: {
            'query': """
                MATCH (g:Gene {id: $gene_id})-[:TRANSLATED_INTO]->(p:Protein)
                MATCH (p)-[r:ASSOCIATED_WITH]->(mf:Molecular_function)
                WITH mf.name AS answer, mf.id AS id, mf.description AS description,
                     CASE WHEN r.score IS NOT NULL THEN toFloat(r.score) ELSE 0.5 END AS score
                RETURN DISTINCT answer, id, description, max(score) AS relevance_score
                ORDER BY relevance_score DESC, answer
                LIMIT 50
            """,
            'description': 'Find molecular functions of proteins translated from the gene',
            'return_fields': ['answer', 'id', 'description', 'relevance_score']
        },

        RelationType.GENE_BIOLOGICAL_PROCESS: {
            'query': """
                MATCH (g:Gene {id: $gene_id})-[:TRANSLATED_INTO]->(p:Protein)
                MATCH (p)-[r:ASSOCIATED_WITH]->(bp:Biological_process)
                WITH bp.name AS answer, bp.id AS id, bp.description AS description,
                     CASE WHEN r.score IS NOT NULL THEN toFloat(r.score) ELSE 0.5 END AS score
                RETURN DISTINCT answer, id, description, max(score) AS relevance_score
                ORDER BY relevance_score DESC, answer
                LIMIT 50
            """,
            'description': 'Find biological processes of proteins translated from the gene',
            'return_fields': ['answer', 'id', 'description', 'relevance_score']
        },

        RelationType.GENE_CELLULAR_COMPONENT: {
            'query': """
                MATCH (g:Gene {id: $gene_id})-[:TRANSLATED_INTO]->(p:Protein)
                MATCH (p)-[r:ASSOCIATED_WITH]->(cc:Cellular_component)
                WITH cc.name AS answer, cc.id AS id, cc.description AS description,
                     CASE WHEN r.score IS NOT NULL THEN toFloat(r.score) ELSE 0.5 END AS score
                RETURN DISTINCT answer, id, description, max(score) AS relevance_score
                ORDER BY relevance_score DESC, answer
                LIMIT 50
            """,
            'description': 'Find cellular components of proteins translated from the gene',
            'return_fields': ['answer', 'id', 'description', 'relevance_score']
        },
    }

    # Conjunction query templates
    CONJUNCTION_TEMPLATES = {
        'biological_process': {
            'query': """
                MATCH (p1:Protein {id: $protein1})-[:ASSOCIATED_WITH]->(bp:Biological_process)
                MATCH (p2:Protein {id: $protein2})-[:ASSOCIATED_WITH]->(bp)
                RETURN DISTINCT bp.id AS answer, bp.name AS name
                LIMIT 50
            """,
            'description': 'Find biological processes shared by both proteins',
            'return_fields': ['answer', 'name']
        },

        'molecular_function': {
            'query': """
                MATCH (p1:Protein {id: $protein1})-[:ASSOCIATED_WITH]->(mf:Molecular_function)
                MATCH (p2:Protein {id: $protein2})-[:ASSOCIATED_WITH]->(mf)
                RETURN DISTINCT mf.id AS answer, mf.name AS name
                LIMIT 50
            """,
            'description': 'Find molecular functions shared by both proteins',
            'return_fields': ['answer', 'name']
        },

        'cellular_component': {
            'query': """
                MATCH (p1:Protein {id: $protein1})-[:ASSOCIATED_WITH]->(cc:Cellular_component)
                MATCH (p2:Protein {id: $protein2})-[:ASSOCIATED_WITH]->(cc)
                RETURN DISTINCT cc.id AS answer, cc.name AS name
                LIMIT 50
            """,
            'description': 'Find cellular components shared by both proteins',
            'return_fields': ['answer', 'name']
        },

        'pathway': {
            'query': """
                MATCH (p1:Protein {id: $protein1})-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
                MATCH (p2:Protein {id: $protein2})-[:ANNOTATED_IN_PATHWAY]->(pw)
                RETURN DISTINCT pw.id AS answer, pw.name AS name, pw.source AS source
                LIMIT 50
            """,
            'description': 'Find pathways shared by both proteins',
            'return_fields': ['answer', 'name', 'source']
        },

        'disease': {
            'query': """
                MATCH (p1:Protein {id: $protein1})-[:ASSOCIATED_WITH]->(d:Disease)
                MATCH (p2:Protein {id: $protein2})-[:ASSOCIATED_WITH]->(d)
                RETURN DISTINCT d.id AS answer, d.name AS name
                LIMIT 50
            """,
            'description': 'Find diseases shared by both proteins',
            'return_fields': ['answer', 'name']
        },

        'tissue': {
            'query': """
                MATCH (p1:Protein {id: $protein1})-[:ASSOCIATED_WITH]->(t:Tissue)
                MATCH (p2:Protein {id: $protein2})-[:ASSOCIATED_WITH]->(t)
                RETURN DISTINCT t.id AS answer, t.name AS name
                LIMIT 50
            """,
            'description': 'Find tissues shared by both proteins',
            'return_fields': ['answer', 'name']
        },
    }

    def generate(self, parsed: ParsedQuestion) -> Optional[CypherQuery]:
        """Generate a Cypher query from a parsed question."""

        if parsed.question_type == QuestionType.CONJUNCTION:
            return self._generate_conjunction_query(parsed)
        elif parsed.question_type == QuestionType.MULTI_HOP:
            return self._generate_multihop_query(parsed)
        else:
            return self._generate_onehop_query(parsed)

    def _generate_onehop_query(self, parsed: ParsedQuestion) -> Optional[CypherQuery]:
        """Generate a one-hop query."""
        template = self.QUERY_TEMPLATES.get(parsed.relation_type)

        if not template:
            return None

        # Prepare parameters
        params = {}
        if 'protein' in parsed.entities:
            params['protein_id'] = parsed.entities['protein']
        if 'gene' in parsed.entities:
            params['gene_id'] = parsed.entities['gene']

        if not params:
            return None

        return CypherQuery(
            query=template['query'],
            parameters=params,
            description=template['description'],
            return_fields=template['return_fields']
        )

    def _generate_multihop_query(self, parsed: ParsedQuestion) -> Optional[CypherQuery]:
        """Generate a multi-hop query with proper scoring and ranking."""

        # If we have a gene, try gene-based multi-hop templates
        if 'gene' in parsed.entities:
            gene_id = parsed.entities['gene']
            question_lower = parsed.original_question.lower()

            # Determine the right template based on what we're looking for
            if 'molecular function' in question_lower or ('function' in question_lower and 'biological' not in question_lower):
                template = self.MULTIHOP_TEMPLATES.get(RelationType.GENE_MOLECULAR_FUNCTION)
            elif 'biological process' in question_lower or 'process' in question_lower:
                template = self.MULTIHOP_TEMPLATES.get(RelationType.GENE_BIOLOGICAL_PROCESS)
            elif 'cellular component' in question_lower or 'locali' in question_lower:
                template = self.MULTIHOP_TEMPLATES.get(RelationType.GENE_CELLULAR_COMPONENT)
            elif 'disease' in question_lower:
                # Gene -> Protein -> Disease
                template = {
                    'query': """
                        MATCH (g:Gene {id: $gene_id})-[:TRANSLATED_INTO]->(p:Protein)
                        MATCH (p)-[r:ASSOCIATED_WITH]->(d:Disease)
                        WITH d, r,
                             CASE WHEN r.score IS NOT NULL THEN toFloat(r.score) ELSE 0.5 END AS score
                        RETURN DISTINCT d.name AS answer, d.id AS id, max(score) AS relevance_score
                        ORDER BY relevance_score DESC
                        LIMIT 50
                    """,
                    'description': 'Find diseases associated with proteins from the gene',
                    'return_fields': ['answer', 'id', 'relevance_score']
                }
            elif 'pathway' in question_lower:
                # Gene -> Protein -> Pathway
                template = {
                    'query': """
                        MATCH (g:Gene {id: $gene_id})-[:TRANSLATED_INTO]->(p:Protein)
                        MATCH (p)-[:ANNOTATED_IN_PATHWAY]->(pw:Pathway)
                        RETURN DISTINCT pw.name AS answer, pw.id AS id, pw.source AS source
                        ORDER BY answer
                        LIMIT 50
                    """,
                    'description': 'Find pathways for proteins from the gene',
                    'return_fields': ['answer', 'id', 'source']
                }
            elif 'tissue' in question_lower:
                # Gene -> Protein -> Tissue
                template = {
                    'query': """
                        MATCH (g:Gene {id: $gene_id})-[:TRANSLATED_INTO]->(p:Protein)
                        MATCH (p)-[r:ASSOCIATED_WITH]->(t:Tissue)
                        WITH t, r,
                             CASE WHEN r.score IS NOT NULL THEN toFloat(r.score) ELSE 0.5 END AS score
                        RETURN DISTINCT t.name AS answer, t.id AS id, max(score) AS relevance_score
                        ORDER BY relevance_score DESC
                        LIMIT 50
                    """,
                    'description': 'Find tissues where proteins from the gene are expressed',
                    'return_fields': ['answer', 'id', 'relevance_score']
                }
            else:
                # Default: molecular function (most common in benchmark)
                template = self.MULTIHOP_TEMPLATES.get(RelationType.GENE_MOLECULAR_FUNCTION)

            if template:
                return CypherQuery(
                    query=template['query'],
                    parameters={'gene_id': gene_id},
                    description=template['description'],
                    return_fields=template['return_fields']
                )

        # Fall back to one-hop if no gene entity
        return self._generate_onehop_query(parsed)

    def _generate_conjunction_query(self, parsed: ParsedQuestion) -> Optional[CypherQuery]:
        """Generate a conjunction query (find common entities)."""
        # Determine what type of conjunction based on target
        target_type = parsed.target_type.lower()

        # Map target to conjunction template
        template_map = {
            'biological_process': 'biological_process',
            'molecular_function': 'molecular_function',
            'cellular_component': 'cellular_component',
            'pathway': 'pathway',
            'disease': 'disease',
            'tissue': 'tissue',
        }

        template_key = template_map.get(target_type)
        if not template_key:
            # Try to infer from question
            question_lower = parsed.original_question.lower()
            if 'biological process' in question_lower:
                template_key = 'biological_process'
            elif 'molecular function' in question_lower:
                template_key = 'molecular_function'
            elif 'cellular component' in question_lower:
                template_key = 'cellular_component'
            elif 'pathway' in question_lower:
                template_key = 'pathway'
            elif 'disease' in question_lower:
                template_key = 'disease'
            elif 'tissue' in question_lower:
                template_key = 'tissue'
            else:
                return None

        template = self.CONJUNCTION_TEMPLATES.get(template_key)
        if not template:
            return None

        # Get the two proteins
        proteins = parsed.conjunction_entities or []

        # Also check entities dict
        if len(proteins) < 2:
            if 'protein' in parsed.entities:
                proteins.append(parsed.entities['protein'])
            if 'protein2' in parsed.entities:
                proteins.append(parsed.entities['protein2'])

        if len(proteins) < 2:
            return None

        params = {
            'protein1': proteins[0],
            'protein2': proteins[1]
        }

        return CypherQuery(
            query=template['query'],
            parameters=params,
            description=template['description'],
            return_fields=template['return_fields']
        )

    def generate_fallback_query(self, parsed: ParsedQuestion) -> Optional[CypherQuery]:
        """Generate a generic fallback query using pattern matching."""
        protein_id = parsed.entities.get('protein')
        gene_id = parsed.entities.get('gene')

        if protein_id:
            # Generic protein query - find all related entities
            query = """
                MATCH (p:Protein {id: $protein_id})-[r]->(target)
                RETURN DISTINCT labels(target)[0] AS target_type,
                       target.id AS answer,
                       target.name AS name,
                       type(r) AS relationship
                LIMIT 100
            """
            return CypherQuery(
                query=query,
                parameters={'protein_id': protein_id},
                description='Find all entities related to the protein',
                return_fields=['target_type', 'answer', 'name', 'relationship']
            )
        elif gene_id:
            # Generic gene query
            query = """
                MATCH (g:Gene {id: $gene_id})-[r]->(target)
                RETURN DISTINCT labels(target)[0] AS target_type,
                       target.id AS answer,
                       target.name AS name,
                       type(r) AS relationship
                LIMIT 100
            """
            return CypherQuery(
                query=query,
                parameters={'gene_id': gene_id},
                description='Find all entities related to the gene',
                return_fields=['target_type', 'answer', 'name', 'relationship']
            )

        return None


# Test the generator
if __name__ == "__main__":
    from question_parser import QuestionParser

    parser = QuestionParser()
    generator = CypherGenerator()

    test_questions = [
        ("What proteins does the protein P68133 interact with?", {"type": "one-hop", "entities": {"protein": "P68133"}}),
        ("What diseases are associated with protein Q9NYB9?", {"type": "one-hop", "entities": {"protein": "Q9NYB9"}}),
        ("Which pathway are the proteins P02778 and P25106 both annotated in?", {"type": "conjunction"}),
        ("What is the molecular function of the protein translated from the gene COL21A1?", {"type": "multi-hop"}),
        ("What biological processes are associated with protein P12345?", {"type": "one-hop", "entities": {"protein": "P12345"}}),
    ]

    print("Testing Cypher Generator\n" + "="*60)
    for question, metadata in test_questions:
        parsed = parser.parse(question, metadata)
        cypher = generator.generate(parsed)

        print(f"\nQ: {question}")
        print(f"  Type: {parsed.question_type.value}")
        print(f"  Relation: {parsed.relation_type.value}")

        if cypher:
            print(f"  Query: {cypher.query.strip()[:100]}...")
            print(f"  Params: {cypher.parameters}")
        else:
            print("  No query generated!")
