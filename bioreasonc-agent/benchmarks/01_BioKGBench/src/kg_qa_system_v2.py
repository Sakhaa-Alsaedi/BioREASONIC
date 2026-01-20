#!/usr/bin/env python3
"""
Knowledge Graph Question Answering System v2.
Supports both rule-based and LLM-based query generation.
"""

import os
import sys
import yaml
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

# Rule-based components
from question_parser import QuestionParser, ParsedQuestion
from cypher_generator import CypherGenerator, CypherQuery

# LLM components
from llm_client import LLMClient, create_llm_client
from llm_cypher_generator import AdvancedLLMCypherGenerator, LLMCypherQuery


class QAMode(Enum):
    RULE_BASED = "rule_based"
    LLM = "llm"
    HYBRID = "hybrid"  # Try LLM first, fall back to rule-based


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
    mode: QAMode
    answers: List[Answer]
    success: bool
    cypher_query: str = None
    error: Optional[str] = None


class KnowledgeGraphQAv2:
    """QA system supporting both rule-based and LLM-based modes."""

    def __init__(self, config_path: str = None, mode: QAMode = QAMode.RULE_BASED):
        """Initialize the QA system."""
        self.mode = mode
        self.driver = None

        # Load config
        if config_path is None:
            config_path = os.path.join(SCRIPT_DIR, 'config.yaml')
            # Check for local config
            local_config = os.path.join(SCRIPT_DIR, 'config.local.yaml')
            if os.path.exists(local_config):
                config_path = local_config

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize rule-based components
        self.rule_parser = QuestionParser()
        self.rule_generator = CypherGenerator()

        # Initialize LLM components if needed
        self.llm_client = None
        self.llm_generator = None

        if mode in [QAMode.LLM, QAMode.HYBRID]:
            llm_config = self.config.get('llm', {})
            if llm_config.get('provider', 'none') != 'none':
                try:
                    self.llm_client = create_llm_client(llm_config)
                    self.llm_generator = AdvancedLLMCypherGenerator(self.llm_client)
                    print(f"LLM initialized: {llm_config.get('provider')}")
                except Exception as e:
                    print(f"Warning: Could not initialize LLM: {e}")
                    if mode == QAMode.LLM:
                        raise
                    # Fall back to rule-based for hybrid mode

        # Load Neo4j config from parent
        neo4j_config_path = os.path.join(PARENT_DIR, 'config', 'kg_config.yml')
        if os.path.exists(neo4j_config_path):
            with open(neo4j_config_path) as f:
                self.neo4j_config = yaml.safe_load(f)
        else:
            self.neo4j_config = self.config.get('neo4j', {})

    def connect(self):
        """Connect to Neo4j database."""
        import neo4j
        uri = f"bolt://{self.neo4j_config['db_url']}:{self.neo4j_config['db_port']}"
        self.driver = neo4j.GraphDatabase.driver(
            uri,
            auth=(self.neo4j_config['db_user'], self.neo4j_config['db_password']),
            encrypted=False
        )
        with self.driver.session() as session:
            session.run("RETURN 1")

    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()

    def answer(self, question: str, metadata: dict = None) -> QAResult:
        """Answer a question using the configured mode."""
        if self.mode == QAMode.RULE_BASED:
            return self._answer_rule_based(question, metadata)
        elif self.mode == QAMode.LLM:
            return self._answer_llm(question, metadata)
        else:  # HYBRID
            return self._answer_hybrid(question, metadata)

    def _answer_rule_based(self, question: str, metadata: dict = None) -> QAResult:
        """Answer using rule-based approach."""
        # Parse question
        parsed = self.rule_parser.parse(question, metadata)

        # Generate Cypher
        cypher = self.rule_generator.generate(parsed)
        if cypher is None:
            cypher = self.rule_generator.generate_fallback_query(parsed)

        if cypher is None:
            return QAResult(
                question=question,
                mode=QAMode.RULE_BASED,
                answers=[],
                success=False,
                error="Could not generate query"
            )

        # Execute
        try:
            answers = self._execute_cypher(cypher.query, cypher.parameters)
            return QAResult(
                question=question,
                mode=QAMode.RULE_BASED,
                answers=answers,
                success=True,
                cypher_query=cypher.query
            )
        except Exception as e:
            return QAResult(
                question=question,
                mode=QAMode.RULE_BASED,
                answers=[],
                success=False,
                cypher_query=cypher.query,
                error=str(e)
            )

    def _answer_llm(self, question: str, metadata: dict = None) -> QAResult:
        """Answer using LLM-based approach."""
        if not self.llm_generator:
            return QAResult(
                question=question,
                mode=QAMode.LLM,
                answers=[],
                success=False,
                error="LLM not initialized"
            )

        # Generate Cypher using LLM
        llm_result = self.llm_generator.generate_with_retry(question, metadata)

        if not llm_result or not llm_result.query:
            return QAResult(
                question=question,
                mode=QAMode.LLM,
                answers=[],
                success=False,
                error="LLM could not generate query"
            )

        # Execute
        try:
            answers = self._execute_cypher(
                llm_result.query,
                llm_result.parameters,
                answer_field=llm_result.answer_field
            )
            return QAResult(
                question=question,
                mode=QAMode.LLM,
                answers=answers,
                success=True,
                cypher_query=llm_result.query
            )
        except Exception as e:
            return QAResult(
                question=question,
                mode=QAMode.LLM,
                answers=[],
                success=False,
                cypher_query=llm_result.query,
                error=str(e)
            )

    def _answer_hybrid(self, question: str, metadata: dict = None) -> QAResult:
        """Answer using hybrid approach: try LLM first, fall back to rule-based."""
        # Try LLM first
        if self.llm_generator:
            result = self._answer_llm(question, metadata)
            if result.success and result.answers:
                return result

        # Fall back to rule-based
        return self._answer_rule_based(question, metadata)

    def _execute_cypher(self, query: str, parameters: Dict,
                        answer_field: str = 'answer') -> List[Answer]:
        """Execute a Cypher query and return answers."""
        answers = []

        with self.driver.session() as session:
            result = session.run(query, **parameters)

            for record in result:
                # Get answer value
                answer_val = record.get(answer_field, '')
                name_val = record.get('name', '')
                id_val = record.get('id', '')

                # Determine primary answer (prefer names over IDs)
                if answer_val and not str(answer_val).startswith(('GO:', 'DOID:', 'R-HSA', 'SMP', 'P', 'Q')):
                    primary_answer = str(answer_val)
                    answer_id = str(id_val) if id_val else ''
                else:
                    primary_answer = str(name_val) if name_val else str(answer_val)
                    answer_id = str(answer_val) if answer_val else str(id_val)

                # Collect extra fields
                extra = {}
                for key in record.keys():
                    if key not in [answer_field, 'name', 'id', 'answer']:
                        val = record.get(key)
                        if val:
                            extra[key] = val

                answers.append(Answer(
                    id=answer_id,
                    name=primary_answer,
                    extra=extra if extra else None
                ))

        return answers


def demo():
    """Demo comparing rule-based and LLM modes."""
    print("="*70)
    print("Knowledge Graph QA System v2 - Demo")
    print("="*70)

    # Test questions
    test_questions = [
        {"question": "What is the molecular function of the protein translated from the gene TP53?", "type": "multi-hop"},
        {"question": "What proteins does P68133 interact with?", "type": "one-hop"},
        {"question": "Which pathway are proteins P02778 and P25106 both annotated in?", "type": "conjunction"},
    ]

    # Test rule-based
    print("\n" + "="*70)
    print("RULE-BASED MODE")
    print("="*70)

    qa_rule = KnowledgeGraphQAv2(mode=QAMode.RULE_BASED)
    qa_rule.connect()

    for q in test_questions:
        result = qa_rule.answer(q['question'], q)
        print(f"\nQ: {q['question'][:60]}...")
        print(f"Success: {result.success}")
        if result.answers:
            print(f"Top answer: {result.answers[0].name}")
        else:
            print(f"Error: {result.error}")

    qa_rule.close()

    # Test LLM if available
    try:
        qa_llm = KnowledgeGraphQAv2(mode=QAMode.LLM)
        if qa_llm.llm_client:
            print("\n" + "="*70)
            print("LLM MODE")
            print("="*70)

            qa_llm.connect()

            for q in test_questions:
                result = qa_llm.answer(q['question'], q)
                print(f"\nQ: {q['question'][:60]}...")
                print(f"Success: {result.success}")
                if result.answers:
                    print(f"Top answer: {result.answers[0].name}")
                else:
                    print(f"Error: {result.error}")

            qa_llm.close()
        else:
            print("\nLLM not configured. Set provider in config.yaml")
    except Exception as e:
        print(f"\nLLM mode not available: {e}")

    print("\n" + "="*70)
    print("Demo complete!")


if __name__ == "__main__":
    demo()
