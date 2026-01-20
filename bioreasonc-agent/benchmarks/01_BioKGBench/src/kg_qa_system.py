#!/usr/bin/env python3
"""
Knowledge Graph Question Answering System for BioKGBench.

This system answers natural language questions by:
1. Parsing questions to extract entities and relationships
2. Generating Cypher queries
3. Executing against Neo4j
4. Formatting and returning answers
"""

import os
import sys
import yaml
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Add parent directory for config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from question_parser import QuestionParser, ParsedQuestion, QuestionType, RelationType
from cypher_generator import CypherGenerator, CypherQuery

@dataclass
class Answer:
    """Represents an answer to a question."""
    id: str
    name: str
    extra: Dict[str, Any] = None

@dataclass
class QAResult:
    """Result of a QA query."""
    question: str
    parsed: ParsedQuestion
    cypher: Optional[CypherQuery]
    answers: List[Answer]
    success: bool
    error: Optional[str] = None

class KnowledgeGraphQA:
    """Main QA system for biomedical knowledge graph."""

    def __init__(self, config_path: str = None):
        """Initialize the QA system."""
        self.parser = QuestionParser()
        self.generator = CypherGenerator()
        self.driver = None

        # Load config
        if config_path is None:
            config_path = os.path.join(PARENT_DIR, 'config', 'kg_config.yml')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def connect(self):
        """Connect to Neo4j database."""
        import neo4j
        uri = f"bolt://{self.config['db_url']}:{self.config['db_port']}"
        self.driver = neo4j.GraphDatabase.driver(
            uri,
            auth=(self.config['db_user'], self.config['db_password']),
            encrypted=False
        )
        # Test connection
        with self.driver.session() as session:
            session.run("RETURN 1")

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()

    def answer(self, question: str, metadata: dict = None) -> QAResult:
        """Answer a natural language question."""

        # Parse the question
        parsed = self.parser.parse(question, metadata)

        # Generate Cypher query
        cypher = self.generator.generate(parsed)

        if cypher is None:
            # Try fallback query
            cypher = self.generator.generate_fallback_query(parsed)

        if cypher is None:
            return QAResult(
                question=question,
                parsed=parsed,
                cypher=None,
                answers=[],
                success=False,
                error="Could not generate query for this question"
            )

        # Execute query
        try:
            answers = self._execute_query(cypher)
            return QAResult(
                question=question,
                parsed=parsed,
                cypher=cypher,
                answers=answers,
                success=True
            )
        except Exception as e:
            return QAResult(
                question=question,
                parsed=parsed,
                cypher=cypher,
                answers=[],
                success=False,
                error=str(e)
            )

    def _execute_query(self, cypher: CypherQuery) -> List[Answer]:
        """Execute a Cypher query and return answers."""
        answers = []

        with self.driver.session() as session:
            result = session.run(cypher.query, **cypher.parameters)

            for record in result:
                # Handle different return field names
                # Primary answer could be 'answer' or 'name'
                answer_val = record.get('answer', '')
                name_val = record.get('name', '')
                id_val = record.get('id', '')

                # If 'answer' contains the name (for multi-hop queries), use it
                # Otherwise fall back to name field
                if answer_val and not answer_val.startswith(('GO:', 'DOID:', 'R-HSA', 'SMP', 'P', 'Q')):
                    # answer is a name (e.g., "protein binding")
                    primary_answer = str(answer_val)
                    answer_id = str(id_val) if id_val else ''
                else:
                    # answer is an ID
                    primary_answer = str(name_val) if name_val else str(answer_val)
                    answer_id = str(answer_val) if answer_val else str(id_val)

                # Collect extra fields
                extra = {}
                for field in cypher.return_fields:
                    if field not in ['answer', 'name', 'id']:
                        val = record.get(field)
                        if val:
                            extra[field] = val

                answers.append(Answer(
                    id=answer_id,
                    name=primary_answer,
                    extra=extra if extra else None
                ))

        return answers

    def batch_answer(self, questions: List[Dict]) -> List[QAResult]:
        """Answer multiple questions."""
        results = []
        for q in questions:
            question_text = q.get('question', '')
            result = self.answer(question_text, q)
            results.append(result)
        return results


def demo():
    """Demo the QA system."""
    print("="*70)
    print("Knowledge Graph QA System - Demo")
    print("="*70)

    # Initialize
    qa = KnowledgeGraphQA()
    print("\nConnecting to Neo4j...")
    qa.connect()
    print("Connected!")

    # Demo questions
    demo_questions = [
        {
            "question": "What proteins does the protein P68133 interact with?",
            "type": "one-hop",
            "entities": {"protein": "P68133"}
        },
        {
            "question": "What diseases are associated with protein P04637?",
            "type": "one-hop",
            "entities": {"protein": "P04637"}
        },
        {
            "question": "Which pathways is protein P53 annotated in?",
            "type": "one-hop",
            "entities": {"protein": "P04637"}
        },
        {
            "question": "What biological processes are associated with protein P68133?",
            "type": "one-hop",
            "entities": {"protein": "P68133"}
        },
        {
            "question": "What is the molecular function of the protein translated from the gene TP53?",
            "type": "multi-hop"
        },
        {
            "question": "Which biological process are the proteins Q9UKD2 and P62750 both associated with?",
            "type": "conjunction"
        },
    ]

    for q in demo_questions:
        print("\n" + "-"*60)
        print(f"Q: {q['question']}")
        print("-"*60)

        result = qa.answer(q['question'], q)

        print(f"Type: {result.parsed.question_type.value}")
        print(f"Relation: {result.parsed.relation_type.value}")
        print(f"Entities: {result.parsed.entities}")

        if result.success:
            print(f"Answers ({len(result.answers)}):")
            for i, ans in enumerate(result.answers[:5]):
                print(f"  {i+1}. {ans.name} ({ans.id})")
            if len(result.answers) > 5:
                print(f"  ... and {len(result.answers) - 5} more")
        else:
            print(f"Error: {result.error}")

    qa.close()
    print("\n" + "="*70)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
