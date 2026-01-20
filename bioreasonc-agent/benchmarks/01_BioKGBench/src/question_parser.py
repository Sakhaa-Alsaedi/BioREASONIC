#!/usr/bin/env python3
"""
Question Parser for KGQA - Extract entities, relationships, and intent from natural language questions.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

class QuestionType(Enum):
    ONE_HOP = "one-hop"
    MULTI_HOP = "multi-hop"
    CONJUNCTION = "conjunction"
    UNKNOWN = "unknown"

class RelationType(Enum):
    PROTEIN_INTERACTION = "protein_interaction"
    PROTEIN_DISEASE = "protein_disease"
    PROTEIN_PATHWAY = "protein_pathway"
    PROTEIN_BIOLOGICAL_PROCESS = "protein_biological_process"
    PROTEIN_MOLECULAR_FUNCTION = "protein_molecular_function"
    PROTEIN_CELLULAR_COMPONENT = "protein_cellular_component"
    PROTEIN_TISSUE = "protein_tissue"
    GENE_PROTEIN = "gene_protein"
    GENE_MOLECULAR_FUNCTION = "gene_molecular_function"
    GENE_BIOLOGICAL_PROCESS = "gene_biological_process"
    GENE_CELLULAR_COMPONENT = "gene_cellular_component"
    GENE_DISEASE = "gene_disease"
    GENE_PATHWAY = "gene_pathway"
    GENE_TISSUE = "gene_tissue"
    UNKNOWN = "unknown"

@dataclass
class ParsedQuestion:
    """Structured representation of a parsed question."""
    original_question: str
    question_type: QuestionType
    relation_type: RelationType
    entities: Dict[str, str]  # entity_type -> entity_id
    target_type: str  # What we're looking for (protein, disease, pathway, etc.)
    is_conjunction: bool = False
    conjunction_entities: List[str] = None  # For conjunction queries

class QuestionParser:
    """Parse natural language questions into structured queries."""

    # Patterns for entity extraction
    PROTEIN_PATTERN = r'\b([A-Z][A-Z0-9]{2,}[-_]?\d*)\b'  # P68133, Q9NYB9, etc.
    UNIPROT_PATTERN = r'\b([PQO][0-9][A-Z0-9]{3}[0-9](?:-\d+)?)\b'  # UniProt format
    GENE_PATTERN = r'\bgene\s+([A-Za-z][A-Za-z0-9\-]+)\b'  # gene BRCA1, gene RND3
    GENE_PATTERN_ALT = r'from\s+(?:the\s+)?gene\s+([A-Za-z][A-Za-z0-9\-]+)'  # from the gene X

    # Relationship detection patterns
    RELATION_PATTERNS = {
        RelationType.PROTEIN_INTERACTION: [
            r'interact(?:s|ing)?\s+with',
            r'protein[s]?\s+(?:does|do|that)\s+.*\s+interact',
            r'show\s+interaction',
            r'interacting\s+proteins?',
        ],
        RelationType.PROTEIN_DISEASE: [
            r'disease[s]?\s+(?:are|is)\s+associated',
            r'associated\s+(?:with\s+)?disease',
            r'disease[s]?\s+.*\s+linked',
            r'what\s+disease',
            r'which\s+disease',
        ],
        RelationType.PROTEIN_PATHWAY: [
            r'pathway[s]?\s+(?:are|is)',
            r'annotated\s+(?:in|to)\s+pathway',
            r'(?:in|to)\s+(?:which|what)\s+pathway',
            r'which\s+pathway',
            r'what\s+.*\s+pathway',
        ],
        RelationType.PROTEIN_BIOLOGICAL_PROCESS: [
            r'biological\s+process(?:es)?',
            r'(?:what|which)\s+biological',
        ],
        RelationType.PROTEIN_MOLECULAR_FUNCTION: [
            r'molecular\s+function[s]?',
            r'(?:what|which)\s+.*\s+molecular\s+function',
            r'function\s+of\s+(?:the\s+)?protein',
        ],
        RelationType.PROTEIN_CELLULAR_COMPONENT: [
            r'cellular\s+component[s]?',
            r'(?:what|which)\s+cellular',
            r'locali[sz](?:ed|ation)',
        ],
        RelationType.PROTEIN_TISSUE: [
            r'tissue[s]?\s+(?:are|is)',
            r'expressed\s+in.*tissue',
            r'(?:what|which)\s+tissue',
        ],
        RelationType.GENE_PROTEIN: [
            r'protein\s+translated\s+from',
            r'gene\s+.*\s+translat',
        ],
        RelationType.GENE_MOLECULAR_FUNCTION: [
            r'molecular\s+function\s+of\s+.*\s+protein\s+translated\s+from\s+.*\s+gene',
            r'function\s+.*\s+gene',
        ],
        RelationType.GENE_BIOLOGICAL_PROCESS: [
            r'biological\s+process\s+.*\s+gene',
        ],
    }

    # Conjunction patterns
    CONJUNCTION_PATTERNS = [
        r'proteins?\s+([A-Z0-9]+)\s+and\s+([A-Z0-9]+)\s+both',
        r'both\s+.*\s+([A-Z0-9]+)\s+and\s+([A-Z0-9]+)',
        r'([A-Z0-9]+)\s+and\s+([A-Z0-9]+)\s+(?:both|all)',
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for rel_type, patterns in self.RELATION_PATTERNS.items():
            self.compiled_patterns[rel_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def parse(self, question: str, metadata: dict = None) -> ParsedQuestion:
        """Parse a question into structured form."""
        question_lower = question.lower()

        # Determine question type
        question_type = self._detect_question_type(question, metadata)

        # Extract entities
        entities = self._extract_entities(question, metadata)

        # Detect relation type
        relation_type = self._detect_relation_type(question_lower)

        # Determine target type
        target_type = self._determine_target_type(relation_type)

        # Check for conjunction
        is_conjunction = question_type == QuestionType.CONJUNCTION
        conjunction_entities = None

        if is_conjunction:
            conjunction_entities = self._extract_conjunction_entities(question)

        return ParsedQuestion(
            original_question=question,
            question_type=question_type,
            relation_type=relation_type,
            entities=entities,
            target_type=target_type,
            is_conjunction=is_conjunction,
            conjunction_entities=conjunction_entities
        )

    def _detect_question_type(self, question: str, metadata: dict = None) -> QuestionType:
        """Detect if question is one-hop, multi-hop, or conjunction."""
        if metadata and 'type' in metadata:
            type_map = {
                'one-hop': QuestionType.ONE_HOP,
                'multi-hop': QuestionType.MULTI_HOP,
                'conjunction': QuestionType.CONJUNCTION,
            }
            return type_map.get(metadata['type'], QuestionType.UNKNOWN)

        question_lower = question.lower()

        # Conjunction detection
        if ' and ' in question_lower and ('both' in question_lower or 'all' in question_lower):
            return QuestionType.CONJUNCTION

        # Multi-hop detection (gene -> protein -> something)
        if 'gene' in question_lower and ('protein' in question_lower or 'translated' in question_lower):
            return QuestionType.MULTI_HOP

        return QuestionType.ONE_HOP

    def _extract_entities(self, question: str, metadata: dict = None) -> Dict[str, str]:
        """Extract entities from question."""
        entities = {}

        # Use metadata if available
        if metadata:
            if 'entities' in metadata and metadata['entities']:
                entities.update(metadata['entities'])
            if 'source' in metadata and metadata['source']:
                # Determine entity type from source format
                source = metadata['source']
                if re.match(self.UNIPROT_PATTERN, source):
                    entities['protein'] = source
                elif source.startswith('ENSG'):
                    entities['gene'] = source

        # Extract from question text
        # UniProt proteins
        uniprot_matches = re.findall(self.UNIPROT_PATTERN, question)
        for match in uniprot_matches:
            if 'protein' not in entities:
                entities['protein'] = match
            elif 'protein2' not in entities:
                entities['protein2'] = match

        # Gene names - try multiple patterns
        gene_matches = re.findall(self.GENE_PATTERN, question, re.IGNORECASE)
        if not gene_matches:
            gene_matches = re.findall(self.GENE_PATTERN_ALT, question, re.IGNORECASE)

        for match in gene_matches:
            if 'gene' not in entities:
                # Normalize gene name (uppercase, handle common variations)
                gene_name = match.upper().strip()
                entities['gene'] = gene_name

        return entities

    def _detect_relation_type(self, question_lower: str) -> RelationType:
        """Detect the type of relationship being asked about."""
        for rel_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(question_lower):
                    return rel_type

        return RelationType.UNKNOWN

    def _determine_target_type(self, relation_type: RelationType) -> str:
        """Determine what entity type we're looking for."""
        target_map = {
            RelationType.PROTEIN_INTERACTION: 'Protein',
            RelationType.PROTEIN_DISEASE: 'Disease',
            RelationType.PROTEIN_PATHWAY: 'Pathway',
            RelationType.PROTEIN_BIOLOGICAL_PROCESS: 'Biological_process',
            RelationType.PROTEIN_MOLECULAR_FUNCTION: 'Molecular_function',
            RelationType.PROTEIN_CELLULAR_COMPONENT: 'Cellular_component',
            RelationType.PROTEIN_TISSUE: 'Tissue',
            RelationType.GENE_PROTEIN: 'Protein',
            RelationType.GENE_MOLECULAR_FUNCTION: 'Molecular_function',
            RelationType.GENE_BIOLOGICAL_PROCESS: 'Biological_process',
        }
        return target_map.get(relation_type, 'unknown')

    def _extract_conjunction_entities(self, question: str) -> List[str]:
        """Extract multiple entities from conjunction queries."""
        entities = []

        # Try conjunction patterns
        for pattern in self.CONJUNCTION_PATTERNS:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                entities.extend(match.groups())
                break

        # Fallback: extract all UniProt IDs
        if not entities:
            entities = re.findall(self.UNIPROT_PATTERN, question)

        return list(set(entities))


# Test the parser
if __name__ == "__main__":
    parser = QuestionParser()

    test_questions = [
        ("What proteins does the protein P68133 interact with?", {"type": "one-hop", "entities": {"protein": "P68133"}}),
        ("What diseases are associated with protein Q9NYB9?", {"type": "one-hop", "entities": {"protein": "Q9NYB9"}}),
        ("Which pathway are the proteins P02778 and P25106 both annotated in?", {"type": "conjunction"}),
        ("What is the molecular function of the protein translated from the gene COL21A1?", {"type": "multi-hop"}),
        ("What biological processes are associated with protein P12345?", {"type": "one-hop"}),
    ]

    print("Testing Question Parser\n" + "="*50)
    for question, metadata in test_questions:
        parsed = parser.parse(question, metadata)
        print(f"\nQ: {question}")
        print(f"  Type: {parsed.question_type.value}")
        print(f"  Relation: {parsed.relation_type.value}")
        print(f"  Entities: {parsed.entities}")
        print(f"  Target: {parsed.target_type}")
        if parsed.conjunction_entities:
            print(f"  Conjunction entities: {parsed.conjunction_entities}")
