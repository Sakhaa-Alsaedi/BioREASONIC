"""
Benchmark Generator Module

Combines all reasoning modules to generate the complete benchmark:
- Orchestrates S, C, R, M modules
- Paraphrases questions
- Validates with LLM judges
- Filters low-quality items
- Exports final dataset
"""

import json
import random
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime

from .schema import BenchmarkItem, BenchmarkDataset, ReasoningCategory
from .ingest import DataIngestor, load_data
from .reacTax.structure import StructureReasoning, create_structure_module
from .reacTax.causal import CausalReasoning, create_causal_module
from .reacTax.risk import RiskReasoning, create_risk_module
from .reacTax.semantic import SemanticReasoning, create_semantic_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuestionParaphraser:
    """Paraphrases questions to avoid template memorization"""

    def __init__(self):
        self.paraphrase_templates = {
            "what": ["Identify", "Determine", "Find", "State", "Specify"],
            "which": ["What", "Identify which", "Determine which", "Find which"],
            "is": ["Does", "Can you determine if", "Is it true that"],
            "compare": ["Contrast", "Evaluate the difference between", "Assess"],
            "calculate": ["Compute", "Determine", "Find the value of", "Estimate"],
            "rank": ["Order", "Sort", "Arrange", "List in order"],
        }

        self.sentence_starters = [
            "Based on the genetic data, ",
            "Using the available evidence, ",
            "From the GWAS results, ",
            "According to the analysis, ",
            "Given the genetic associations, ",
        ]

    def paraphrase(self, question: str, preserve_entities: bool = True) -> str:
        """
        Paraphrase a question while preserving key entities

        Args:
            question: Original question
            preserve_entities: Whether to preserve gene/variant names

        Returns:
            Paraphrased question
        """
        paraphrased = question

        # Apply simple transformations
        for trigger, alternatives in self.paraphrase_templates.items():
            if question.lower().startswith(trigger):
                replacement = random.choice(alternatives)
                paraphrased = replacement + question[len(trigger):]
                break

        # Optionally add a starter
        if random.random() > 0.5:
            starter = random.choice(self.sentence_starters)
            # Lowercase first letter of paraphrased
            if paraphrased:
                paraphrased = starter + paraphrased[0].lower() + paraphrased[1:]

        return paraphrased


class ExplanationGenerator:
    """Generates scientific explanations for benchmark items"""

    def __init__(self):
        self.explanation_templates = {
            "S": {
                "default": "Graph algorithms like {algorithm} are used to explore biological networks and identify relationships between genes.",
                "pathway": "Pathway analysis helps identify how genes interact and influence disease phenotypes through biological networks.",
            },
            "C": {
                "default": "Causal inference distinguishes true causal relationships from mere correlations using methods like {algorithm}.",
                "mr": "Mendelian Randomization uses genetic variants as instrumental variables to infer causality.",
                "pc": "The PC algorithm discovers causal structure by testing conditional independence relationships.",
            },
            "R": {
                "default": "Risk assessment interprets GWAS statistics like odds ratios to quantify genetic contributions to disease.",
                "aggregate": "Aggregate risk scores combine multiple genetic factors to estimate cumulative disease risk.",
            },
            "M": {
                "default": "Text mining extracts biomedical knowledge from literature using NLP techniques like {algorithm}.",
                "ner": "Named Entity Recognition identifies genes, diseases, and variants in biomedical text.",
            }
        }

    def generate(self, taxonomy: str, algorithm: str = None,
                 context: Dict = None) -> str:
        """Generate explanation for a benchmark item"""
        templates = self.explanation_templates.get(taxonomy, {})

        if algorithm and algorithm.lower() in templates:
            template = templates[algorithm.lower()]
        else:
            template = templates.get("default", "This question tests reasoning in the {taxonomy} category.")

        explanation = template.format(
            algorithm=algorithm or "standard methods",
            taxonomy=taxonomy,
            **(context or {})
        )

        return explanation


class BenchmarkValidator:
    """Validates benchmark items for quality"""

    def __init__(self, min_score: float = 3.5):
        self.min_score = min_score
        self.validation_criteria = {
            "factuality": "Is the question based on accurate data?",
            "answerability": "Can the question be definitively answered?",
            "clarity": "Is the question clear and unambiguous?",
            "difficulty": "Is the difficulty level appropriate?",
            "reasoning": "Does the question require reasoning (not just recall)?",
        }

    def validate_item(self, item: BenchmarkItem) -> Tuple[bool, Dict]:
        """
        Validate a single benchmark item

        Returns:
            Tuple of (is_valid, validation_details)
        """
        scores = {}
        feedback = []

        # Check factuality (answer exists and is not empty)
        if item.answer and len(item.answer) > 0:
            scores["factuality"] = 5.0
        else:
            scores["factuality"] = 1.0
            feedback.append("Answer is missing or empty")

        # Check answerability (question is well-formed)
        if item.question and item.question.endswith("?"):
            scores["answerability"] = 5.0
        else:
            scores["answerability"] = 3.0
            feedback.append("Question may not be properly formed")

        # Check clarity (reasonable length, no excessive jargon)
        q_len = len(item.question.split())
        if 5 <= q_len <= 50:
            scores["clarity"] = 5.0
        elif q_len < 5:
            scores["clarity"] = 2.0
            feedback.append("Question too short")
        else:
            scores["clarity"] = 3.0
            feedback.append("Question may be too long")

        # Check reasoning requirement
        reasoning_keywords = ["compare", "rank", "calculate", "determine", "infer",
                            "analyze", "path", "causal", "risk", "relationship"]
        has_reasoning = any(kw in item.question.lower() for kw in reasoning_keywords)
        scores["reasoning"] = 5.0 if has_reasoning else 3.0

        # Calculate average score
        avg_score = sum(scores.values()) / len(scores)
        is_valid = avg_score >= self.min_score

        item.validation_scores = list(scores.values())
        item.avg_score = avg_score
        item.is_valid = is_valid

        return is_valid, {
            "scores": scores,
            "avg_score": avg_score,
            "feedback": feedback,
            "is_valid": is_valid
        }

    def validate_dataset(self, items: List[BenchmarkItem]) -> Dict:
        """Validate all items in a dataset"""
        results = {
            "total": len(items),
            "valid": 0,
            "invalid": 0,
            "by_taxonomy": {},
            "avg_score": 0.0
        }

        total_score = 0.0

        for item in items:
            is_valid, details = self.validate_item(item)

            if is_valid:
                results["valid"] += 1
            else:
                results["invalid"] += 1

            total_score += details["avg_score"]

            # Track by taxonomy
            if item.taxonomy not in results["by_taxonomy"]:
                results["by_taxonomy"][item.taxonomy] = {"valid": 0, "invalid": 0}

            if is_valid:
                results["by_taxonomy"][item.taxonomy]["valid"] += 1
            else:
                results["by_taxonomy"][item.taxonomy]["invalid"] += 1

        results["avg_score"] = total_score / len(items) if items else 0.0

        return results


class BenchmarkFilter:
    """Filters benchmark items based on quality and diversity"""

    def __init__(self, min_score: float = 3.5):
        self.min_score = min_score
        self.seen_hashes: set = set()

    def _compute_hash(self, item: BenchmarkItem) -> str:
        """Compute hash for deduplication"""
        content = f"{item.question}|{item.answer}"
        return hashlib.md5(content.encode()).hexdigest()

    def filter_items(self, items: List[BenchmarkItem]) -> List[BenchmarkItem]:
        """Filter items based on quality and uniqueness"""
        filtered = []

        for item in items:
            # Check validation score
            if item.avg_score is not None and item.avg_score < self.min_score:
                continue

            # Check for duplicates
            item_hash = self._compute_hash(item)
            if item_hash in self.seen_hashes:
                continue

            self.seen_hashes.add(item_hash)
            filtered.append(item)

        return filtered

    def balance_by_taxonomy(self, items: List[BenchmarkItem],
                            target_per_category: int = 50) -> List[BenchmarkItem]:
        """Balance items across taxonomy categories"""
        by_taxonomy: Dict[str, List[BenchmarkItem]] = {}

        for item in items:
            if item.taxonomy not in by_taxonomy:
                by_taxonomy[item.taxonomy] = []
            by_taxonomy[item.taxonomy].append(item)

        balanced = []
        for taxonomy, taxonomy_items in by_taxonomy.items():
            # Sort by score and take top items
            sorted_items = sorted(
                taxonomy_items,
                key=lambda x: x.avg_score or 0,
                reverse=True
            )
            balanced.extend(sorted_items[:target_per_category])

        return balanced


class BenchmarkGenerator:
    """Main benchmark generation orchestrator"""

    def __init__(self, data_dir: str = "../Data",
                 use_transformers: bool = False):
        self.data_dir = data_dir

        # Initialize modules
        logger.info("Initializing BioREASONC-Bench modules...")
        self.ingestor = DataIngestor(data_dir)
        self.structure_module = create_structure_module()
        self.causal_module = create_causal_module()
        self.risk_module = create_risk_module()
        self.semantic_module = create_semantic_module(use_transformers)

        # Initialize utilities
        self.paraphraser = QuestionParaphraser()
        self.explanation_gen = ExplanationGenerator()
        self.validator = BenchmarkValidator()
        self.filter = BenchmarkFilter()

        # Load data
        self.data_loaded = False

    def load_data(self):
        """Load all genetic data"""
        if not self.data_loaded:
            logger.info("Loading genetic risk data...")
            self.ingestor.load_all()
            self.data_loaded = True
            logger.info(f"Loaded {len(self.ingestor.covid_genes)} COVID genes, "
                       f"{len(self.ingestor.ra_genes)} RA genes")

    def generate_structure_questions(self, n: int = 50) -> List[BenchmarkItem]:
        """Generate Structure-Aware questions"""
        self.load_data()
        genes = self.ingestor.get_all_genes()
        questions = self.structure_module.generate_structure_questions(genes)
        return questions[:n]

    def generate_causal_questions(self, n: int = 50,
                                   causal_data: Dict = None) -> List[BenchmarkItem]:
        """Generate Causal-Aware questions"""
        self.load_data()
        genes = self.ingestor.covid_genes

        # If causal data provided, load it
        if causal_data:
            # Process user-provided PC algorithm results
            pass

        questions = self.causal_module.generate_causal_questions(genes)
        return questions[:n]

    def generate_risk_questions(self, n: int = 50) -> List[BenchmarkItem]:
        """Generate Risk-Aware questions"""
        self.load_data()
        genes = self.ingestor.covid_genes
        questions = self.risk_module.generate_risk_questions(genes)
        return questions[:n]

    def generate_semantic_questions(self, n: int = 50) -> List[BenchmarkItem]:
        """Generate Semantic-Aware questions"""
        self.load_data()
        genes = self.ingestor.get_all_genes()
        questions = self.semantic_module.generate_semantic_questions(genes)
        return questions[:n]

    def generate_all(self, items_per_category: int = 50,
                     paraphrase: bool = True,
                     validate: bool = True) -> BenchmarkDataset:
        """
        Generate complete benchmark dataset

        Args:
            items_per_category: Target items per taxonomy category
            paraphrase: Whether to paraphrase questions
            validate: Whether to validate items

        Returns:
            BenchmarkDataset with all items
        """
        logger.info("Starting benchmark generation...")
        all_items = []

        # Generate questions from each module
        logger.info("Generating Structure-Aware (S) questions...")
        s_questions = self.generate_structure_questions(items_per_category)
        all_items.extend(s_questions)
        logger.info(f"  Generated {len(s_questions)} S questions")

        logger.info("Generating Causal-Aware (C) questions...")
        c_questions = self.generate_causal_questions(items_per_category)
        all_items.extend(c_questions)
        logger.info(f"  Generated {len(c_questions)} C questions")

        logger.info("Generating Risk-Aware (R) questions...")
        r_questions = self.generate_risk_questions(items_per_category)
        all_items.extend(r_questions)
        logger.info(f"  Generated {len(r_questions)} R questions")

        logger.info("Generating Semantic-Aware (M) questions...")
        m_questions = self.generate_semantic_questions(items_per_category)
        all_items.extend(m_questions)
        logger.info(f"  Generated {len(m_questions)} M questions")

        # Paraphrase questions
        if paraphrase:
            logger.info("Paraphrasing questions...")
            for item in all_items:
                if random.random() > 0.3:  # Paraphrase 70% of questions
                    item.original_question = item.question
                    item.question = self.paraphraser.paraphrase(item.question)
                    item.paraphrased = True

        # Validate items
        if validate:
            logger.info("Validating benchmark items...")
            validation_results = self.validator.validate_dataset(all_items)
            logger.info(f"  Valid: {validation_results['valid']}, "
                       f"Invalid: {validation_results['invalid']}, "
                       f"Avg Score: {validation_results['avg_score']:.2f}")

        # Filter items
        logger.info("Filtering items...")
        filtered_items = self.filter.filter_items(all_items)
        balanced_items = self.filter.balance_by_taxonomy(filtered_items, items_per_category)
        logger.info(f"  Final items: {len(balanced_items)}")

        # Reassign IDs
        for i, item in enumerate(balanced_items):
            item.id = f"{item.taxonomy}-{i:04d}"

        # Create dataset
        dataset = BenchmarkDataset(
            version="1.0.0",
            items=balanced_items,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "items_per_category": items_per_category,
                "total_items": len(balanced_items),
                "paraphrased": paraphrase,
                "validated": validate
            }
        )

        return dataset

    def export_dataset(self, dataset: BenchmarkDataset,
                       output_dir: str = "./outputs"):
        """Export dataset to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export JSONL
        jsonl_path = output_path / "bioreasonc_bench_v1.jsonl"
        dataset.to_jsonl(str(jsonl_path))
        logger.info(f"Exported JSONL: {jsonl_path}")

        # Export summary
        summary_path = output_path / "summary.json"
        summary = dataset.to_dict()
        summary.pop("items", None)  # Don't include items in summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Exported summary: {summary_path}")

        return str(jsonl_path), str(summary_path)


# Factory function
def create_generator(data_dir: str = "../Data",
                     use_transformers: bool = False) -> BenchmarkGenerator:
    """Create benchmark generator"""
    return BenchmarkGenerator(data_dir, use_transformers)


if __name__ == "__main__":
    # Test generation
    generator = create_generator()

    # Generate small test dataset
    dataset = generator.generate_all(items_per_category=10)

    print(f"\nGenerated {len(dataset)} items")
    print(f"Statistics: {dataset.get_statistics()}")

    # Show sample items
    print("\n--- Sample Items ---")
    for item in dataset.items[:3]:
        print(f"\n{item.id} [{item.label}]")
        print(f"Q: {item.question}")
        print(f"A: {item.answer[:100]}...")
