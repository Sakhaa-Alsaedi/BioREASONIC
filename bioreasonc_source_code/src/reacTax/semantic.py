"""
Semantic-Aware Reasoning Module (M)

Implements text mining and NLP for biomedical relation extraction:
- Named Entity Recognition (NER) for genes, diseases, variants
- Relation extraction using transformer models
- Semantic similarity computation
- LLM-based information extraction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import re
import logging
from abc import ABC, abstractmethod

from ..schema import RiskGene, BenchmarkItem, SemanticLabel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity"""
    text: str
    label: str  # 'GENE', 'DISEASE', 'VARIANT', 'DRUG', etc.
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class Relation:
    """Represents an extracted relation"""
    subject: Entity
    predicate: str  # 'associated_with', 'causes', 'treats', etc.
    object: Entity
    confidence: float = 1.0
    source_text: Optional[str] = None


@dataclass
class SemanticSimilarity:
    """Represents semantic similarity between entities"""
    entity1: str
    entity2: str
    similarity: float
    method: str


class EntityRecognizer(ABC):
    """Abstract base class for entity recognition"""

    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        pass


class RuleBasedNER(EntityRecognizer):
    """Rule-based Named Entity Recognition"""

    def __init__(self):
        # Gene patterns (common gene name formats)
        self.gene_pattern = re.compile(
            r'\b([A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]+)?)\b'
        )

        # Variant patterns (rs IDs)
        self.variant_pattern = re.compile(
            r'\b(rs\d{1,12})\b', re.IGNORECASE
        )

        # Disease patterns
        self.disease_keywords = {
            'covid-19', 'coronavirus', 'sars-cov-2', 'rheumatoid arthritis',
            'ra', 'autoimmune', 'inflammation', 'disease', 'syndrome',
            'disorder', 'infection', 'severity', 'mortality'
        }

        # Common gene names for validation
        self.known_genes = {
            'ACE2', 'TMPRSS2', 'ABO', 'TYK2', 'IFNAR2', 'IL6', 'TNF',
            'HLA', 'PTPN22', 'STAT4', 'IRF5', 'BLK', 'TNFAIP3', 'CD40',
            'CTLA4', 'TRAF1', 'CCR6', 'IL2RA', 'PADI4', 'SLC22A4',
            'FURIN', 'OAS1', 'DPP9', 'TLR7', 'MX1', 'CCR9', 'CXCR6'
        }

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        entities = []

        # Extract variants
        for match in self.variant_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(1),
                label='VARIANT',
                start=match.start(),
                end=match.end(),
                confidence=0.95
            ))

        # Extract genes
        for match in self.gene_pattern.finditer(text):
            gene_name = match.group(1)
            # Validate against known genes or pattern
            if gene_name in self.known_genes or len(gene_name) <= 8:
                confidence = 0.9 if gene_name in self.known_genes else 0.6
                entities.append(Entity(
                    text=gene_name,
                    label='GENE',
                    start=match.start(),
                    end=match.end(),
                    confidence=confidence
                ))

        # Extract diseases
        text_lower = text.lower()
        for disease in self.disease_keywords:
            idx = text_lower.find(disease)
            if idx != -1:
                entities.append(Entity(
                    text=text[idx:idx+len(disease)],
                    label='DISEASE',
                    start=idx,
                    end=idx + len(disease),
                    confidence=0.85
                ))

        return entities


class TransformerNER(EntityRecognizer):
    """Transformer-based NER using BioBERT/PubMedBERT"""

    def __init__(self, model_name: str = "dmis-lab/biobert-v1.1"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load transformer model"""
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
            logger.info(f"Loaded NER model: {self.model_name}")

        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            self.model = None

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using transformer model"""
        if self.model is None:
            return []

        try:
            results = self.ner_pipeline(text)
            entities = []

            for item in results:
                entities.append(Entity(
                    text=item['word'],
                    label=item['entity'].replace('B-', '').replace('I-', ''),
                    start=item['start'],
                    end=item['end'],
                    confidence=item['score']
                ))

            return entities

        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
            return []


class RelationExtractor:
    """Extracts relations between entities"""

    def __init__(self):
        self.relation_patterns = {
            'associated_with': [
                r'(\w+)\s+(?:is\s+)?associated\s+with\s+(\w+)',
                r'(\w+)\s+(?:is\s+)?linked\s+to\s+(\w+)',
                r'association\s+between\s+(\w+)\s+and\s+(\w+)',
            ],
            'causes': [
                r'(\w+)\s+causes?\s+(\w+)',
                r'(\w+)\s+leads?\s+to\s+(\w+)',
                r'(\w+)\s+results?\s+in\s+(\w+)',
            ],
            'increases_risk': [
                r'(\w+)\s+increases?\s+(?:the\s+)?risk\s+(?:of\s+)?(\w+)',
                r'(\w+)\s+(?:is\s+a\s+)?risk\s+factor\s+for\s+(\w+)',
            ],
            'protects_against': [
                r'(\w+)\s+protects?\s+against\s+(\w+)',
                r'(\w+)\s+reduces?\s+(?:the\s+)?risk\s+(?:of\s+)?(\w+)',
            ],
            'encodes': [
                r'(\w+)\s+gene\s+encodes?\s+(\w+)',
                r'(\w+)\s+encodes?\s+(?:the\s+)?(\w+)\s+protein',
            ],
        }

    def extract_relations(self, text: str,
                          entities: List[Entity] = None) -> List[Relation]:
        """Extract relations from text"""
        relations = []

        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    subject_text = match.group(1)
                    object_text = match.group(2)

                    # Create entities if not provided
                    subject = Entity(
                        text=subject_text,
                        label='ENTITY',
                        start=match.start(1),
                        end=match.end(1)
                    )
                    obj = Entity(
                        text=object_text,
                        label='ENTITY',
                        start=match.start(2),
                        end=match.end(2)
                    )

                    relations.append(Relation(
                        subject=subject,
                        predicate=relation_type,
                        object=obj,
                        confidence=0.7,
                        source_text=match.group(0)
                    ))

        return relations


class SemanticSimilarityCalculator:
    """Calculates semantic similarity between biomedical terms"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
        self.cache: Dict[Tuple[str, str], float] = {}

    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded similarity model: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load similarity model: {e}")
            self.model = None

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts"""
        # Check cache
        cache_key = (text1, text2)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.model is None:
            # Fallback: simple Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            similarity = len(words1 & words2) / len(words1 | words2)
        else:
            try:
                embeddings = self.model.encode([text1, text2])
                similarity = float(np.dot(embeddings[0], embeddings[1]) /
                                  (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
            except Exception:
                similarity = 0.0

        self.cache[cache_key] = similarity
        return similarity

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text"""
        if self.model is None:
            return None
        try:
            return self.model.encode(text)
        except Exception:
            return None


class BiomedicalTextMiner:
    """Comprehensive biomedical text mining"""

    def __init__(self, use_transformers: bool = False):
        self.use_transformers = use_transformers

        if use_transformers:
            self.ner = TransformerNER()
        else:
            self.ner = RuleBasedNER()

        self.relation_extractor = RelationExtractor()
        self.similarity_calc = SemanticSimilarityCalculator()

    def analyze_text(self, text: str) -> Dict:
        """Full text analysis"""
        entities = self.ner.extract_entities(text)
        relations = self.relation_extractor.extract_relations(text, entities)

        return {
            "entities": entities,
            "relations": relations,
            "genes": [e for e in entities if e.label == 'GENE'],
            "diseases": [e for e in entities if e.label == 'DISEASE'],
            "variants": [e for e in entities if e.label == 'VARIANT']
        }

    def extract_gene_disease_pairs(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract gene-disease pairs with relation type"""
        analysis = self.analyze_text(text)
        pairs = []

        for relation in analysis["relations"]:
            pairs.append((
                relation.subject.text,
                relation.predicate,
                relation.object.text
            ))

        return pairs


class SemanticReasoning:
    """Main class for Semantic-Aware reasoning"""

    def __init__(self, use_transformers: bool = False):
        self.text_miner = BiomedicalTextMiner(use_transformers)
        self.similarity_calc = SemanticSimilarityCalculator()

    def generate_semantic_questions(self, genes: List[RiskGene],
                                     sample_texts: List[str] = None) -> List[BenchmarkItem]:
        """Generate semantic-aware benchmark questions"""
        questions = []

        # Default sample texts if not provided
        if sample_texts is None:
            sample_texts = [
                "ACE2 is associated with COVID-19 severity. The ACE2 gene encodes the receptor used by SARS-CoV-2 for cell entry.",
                "TYK2 increases the risk of autoimmune diseases including rheumatoid arthritis. Variants in TYK2 are linked to COVID-19 severity.",
                "The ABO gene determines blood type and is associated with COVID-19 susceptibility. Blood type O may be protective.",
                "IL6 causes inflammation and is a key cytokine in COVID-19 cytokine storm. IL6 inhibitors are used as treatment.",
                "IFNAR2 encodes the interferon receptor and protects against viral infections including COVID-19.",
            ]

        # M-REL-EXTRACT questions
        for i, text in enumerate(sample_texts[:10]):
            analysis = self.text_miner.analyze_text(text)
            relations = analysis.get("relations", [])

            if relations:
                rel = relations[0]
                q = BenchmarkItem(
                    id=f"M-{len(questions):04d}",
                    taxonomy="M",
                    label=SemanticLabel.REL_EXTRACT.value,
                    template_id="M-REL-EXTRACT-01",
                    question=f"Extract the biomedical relationship from: '{text}'",
                    answer=f"{rel.subject.text} {rel.predicate.replace('_', ' ')} {rel.object.text}",
                    explanation=f"Relation extraction identifies subject-predicate-object triples from biomedical text.",
                    source_data={"text": text, "relation": rel.predicate},
                    algorithm_used="relation_extraction"
                )
                questions.append(q)

        # M-ENTITY-RECOGNIZE questions
        for i, text in enumerate(sample_texts[:10]):
            analysis = self.text_miner.analyze_text(text)
            genes_found = [e.text for e in analysis.get("genes", [])]
            diseases_found = [e.text for e in analysis.get("diseases", [])]
            variants_found = [e.text for e in analysis.get("variants", [])]

            if genes_found:
                q = BenchmarkItem(
                    id=f"M-{len(questions):04d}",
                    taxonomy="M",
                    label=SemanticLabel.ENTITY_RECOGNIZE.value,
                    template_id="M-ENTITY-RECOGNIZE-01",
                    question=f"Identify all gene names mentioned in: '{text}'",
                    answer=f"Genes: {', '.join(genes_found)}",
                    explanation="Named Entity Recognition (NER) identifies biomedical entities like genes, diseases, and variants.",
                    source_data={"text": text, "genes": genes_found},
                    algorithm_used="ner"
                )
                questions.append(q)

        # M-SEMANTIC-SIM questions
        gene_pairs = []
        for i in range(min(len(genes) - 1, 10)):
            gene1, gene2 = genes[i], genes[i + 1]
            gene_pairs.append((gene1, gene2))

        for gene1, gene2 in gene_pairs[:10]:
            # Create descriptions
            desc1 = f"{gene1.symbol} gene associated with {', '.join(gene1.associated_diseases)}"
            desc2 = f"{gene2.symbol} gene associated with {', '.join(gene2.associated_diseases)}"

            similarity = self.similarity_calc.compute_similarity(desc1, desc2)

            q = BenchmarkItem(
                id=f"M-{len(questions):04d}",
                taxonomy="M",
                label=SemanticLabel.SEMANTIC_SIM.value,
                template_id="M-SEMANTIC-SIM-01",
                question=f"How semantically similar are {gene1.symbol} and {gene2.symbol} based on their disease associations?",
                answer=f"Semantic similarity: {similarity:.3f} (scale 0-1)",
                explanation=f"Computed using sentence embeddings. {gene1.symbol}: {gene1.associated_diseases}. {gene2.symbol}: {gene2.associated_diseases}.",
                source_genes=[gene1.symbol, gene2.symbol],
                source_data={"similarity": similarity},
                algorithm_used="semantic_similarity"
            )
            questions.append(q)

        # M-TEXT-INFERENCE questions
        for gene in genes[:10]:
            # Skip genes without OR
            if not gene.odds_ratio:
                continue

            inference_text = f"Given that {gene.symbol} has OR={gene.odds_ratio:.2f} for COVID-19, what can be inferred about its role?"

            if gene.odds_ratio > 1.5:
                answer = f"{gene.symbol} is a significant risk factor. Higher OR suggests stronger association with disease severity."
            elif gene.odds_ratio and gene.odds_ratio < 0.8:
                answer = f"{gene.symbol} appears protective. OR<1 suggests reduced disease risk."
            else:
                answer = f"{gene.symbol} has modest effect. OR near 1 suggests weak or no association."

            q = BenchmarkItem(
                id=f"M-{len(questions):04d}",
                taxonomy="M",
                label=SemanticLabel.TEXT_INFERENCE.value,
                template_id="M-TEXT-INFERENCE-01",
                question=inference_text,
                answer=answer,
                explanation="Text inference combines numerical data with domain knowledge to draw conclusions.",
                source_genes=[gene.symbol],
                source_data={"or": gene.odds_ratio},
                algorithm_used="text_inference"
            )
            questions.append(q)

        return questions


# Factory function
def create_semantic_module(use_transformers: bool = False) -> SemanticReasoning:
    """Create Semantic-Aware reasoning module"""
    return SemanticReasoning(use_transformers)


if __name__ == "__main__":
    # Test
    module = create_semantic_module(use_transformers=False)

    # Test text mining
    test_text = "ACE2 is associated with COVID-19 severity. The rs2285666 variant in ACE2 increases infection risk."
    analysis = module.text_miner.analyze_text(test_text)

    print("Entities found:")
    for entity in analysis["entities"]:
        print(f"  {entity.text} ({entity.label}): confidence={entity.confidence:.2f}")

    print("\nRelations found:")
    for relation in analysis["relations"]:
        print(f"  {relation.subject.text} --{relation.predicate}--> {relation.object.text}")

    # Test similarity
    sim = module.similarity_calc.compute_similarity(
        "ACE2 gene COVID-19 severity",
        "TMPRSS2 gene coronavirus infection"
    )
    print(f"\nSemantic similarity: {sim:.3f}")

    # Generate questions
    test_genes = [
        RiskGene(symbol="ACE2", odds_ratio=2.1, associated_diseases=["COVID-19"]),
        RiskGene(symbol="TYK2", odds_ratio=1.6, associated_diseases=["COVID-19", "RA"]),
    ]
    questions = module.generate_semantic_questions(test_genes)
    print(f"\nGenerated {len(questions)} questions")
