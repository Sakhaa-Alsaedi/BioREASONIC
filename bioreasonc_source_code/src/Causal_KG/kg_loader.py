"""
Knowledge Graph Ingestion Module for BioREASONC

================================================================================
MODULE OVERVIEW
================================================================================

Loads CAUSALdb2 Gene-Disease Knowledge Graph and generates benchmark questions.
This module is the DATA FOUNDATION for the entire BioREASONC-Bench pipeline.

This module provides data to STEP 1 (Generator) in the BioREASONC pipeline:
    KG_INGEST → Generator → Validator → Explainer → Paraphraser → Exporter

================================================================================
DATA SOURCE
================================================================================

CAUSALdb2 v2.1 + GENCODE v44:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Statistics:                                                                     │
│ • 66,057 gene-disease relationships                                            │
│ • 15,039 unique genes (HGNC symbols)                                           │
│ • 544 unique diseases (EFO/DOID ontology)                                       │
│                                                                                 │
│ Key Feature: MR (Mendelian Randomization) scores for CAUSAL INFERENCE          │
│ • MR uses genetic variants as natural experiments                              │
│ • High MR score → stronger causal evidence                                     │
│ • This is what distinguishes CAUSALdb2 from pure GWAS databases                │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
EVIDENCE SCORES
================================================================================

Each gene-disease pair has multiple scores indicating evidence strength:

┌─────────────────────────────┬─────────┬──────────────────────────────────────────┐
│ Score                       │ Range   │ Description                              │
├─────────────────────────────┼─────────┼──────────────────────────────────────────┤
│ mr_score                    │ 0-1     │ ⚠️ MOST IMPORTANT for causal claims      │
│                             │         │ Mendelian Randomization evidence         │
│                             │         │ >0.5: Strong causal support              │
│                             │         │ >0.3: Moderate causal support            │
│                             │         │ <0.3: Insufficient for causal claims     │
├─────────────────────────────┼─────────┼──────────────────────────────────────────┤
│ causal_confidence_score     │ 0-1     │ Fine-mapping posterior probability       │
│                             │         │ How likely this gene is the causal gene  │
├─────────────────────────────┼─────────┼──────────────────────────────────────────┤
│ evidence_score              │ 0-1     │ GWAS support strength                    │
│                             │         │ Based on number of supporting SNPs       │
├─────────────────────────────┼─────────┼──────────────────────────────────────────┤
│ go_functional_score         │ 0-1     │ PPI network / GO term enrichment         │
│                             │         │ Biological pathway relevance             │
├─────────────────────────────┼─────────┼──────────────────────────────────────────┤
│ risk_weight_score           │ 0-1     │ Combined weighted score                  │
│                             │         │ Aggregates all evidence types            │
└─────────────────────────────┴─────────┴──────────────────────────────────────────┘

================================================================================
EVIDENCE LEVELS
================================================================================

Gene-disease pairs are classified into evidence levels:

┌─────────────────┬──────────────────────────────────────────────────────────────────┐
│ Level           │ Criteria                                                         │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ VERY_STRONG     │ mr_score > 0.5 AND risk_weight_score > 0.7                       │
│                 │ → Can claim "causal evidence" with confidence                    │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ STRONG          │ risk_weight_score > 0.7                                          │
│                 │ → Strong association, moderate causal support                    │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ MODERATE        │ risk_weight_score > 0.4                                          │
│                 │ → Notable association, limited causal evidence                   │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ SUGGESTIVE      │ risk_weight_score > 0.2                                          │
│                 │ → Weak association, hypothesis-generating only                   │
├─────────────────┼──────────────────────────────────────────────────────────────────┤
│ WEAK            │ risk_weight_score ≤ 0.2                                          │
│                 │ → Used for NEGATIVE EXAMPLES in benchmark                        │
│                 │ → Questions with "No" answers                                    │
└─────────────────┴──────────────────────────────────────────────────────────────────┘

================================================================================
KEY CONCEPTS
================================================================================

RELATIONSHIP DIRECTION:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Gene → Disease                                                                  │
│ "Gene is a RISK FACTOR for Disease"                                            │
│ "Gene variants INCREASE RISK of Disease"                                        │
│                                                                                 │
│ NOT: Disease causes gene (reverse causation is a confound)                     │
│ NOT: Gene protects against disease (unless OR < 1)                             │
└─────────────────────────────────────────────────────────────────────────────────┘

MENDELIAN RANDOMIZATION (MR):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ WHY MR IS SPECIAL:                                                              │
│ • Genetic variants are randomly assigned at conception                         │
│ • This mimics a randomized controlled trial                                    │
│ • Reduces confounding from environmental factors                               │
│ • Enables causal inference from observational data                             │
│                                                                                 │
│ INTERPRETATION:                                                                 │
│ • High MR score: "Nature ran an experiment, gene affects disease"              │
│ • Low MR score: "Association may be confounded or reverse-causal"             │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
OUTPUT SPECIFICATION
================================================================================

GeneDiseasePair dataclass fields:
┌─────────────────────────────┬──────────────┬────────────────────────────────────┐
│ Field                       │ Type         │ Description                        │
├─────────────────────────────┼──────────────┼────────────────────────────────────┤
│ gene_id                     │ str          │ Ensembl gene ID                    │
│ gene_name                   │ str          │ HGNC symbol (e.g., "BRCA1")       │
│ gene_type                   │ str          │ protein_coding, lncRNA, etc.       │
│ disease_id                  │ str          │ EFO/DOID ID                        │
│ disease_name                │ str          │ Disease name                       │
│ snp_count                   │ int          │ Number of associated SNPs          │
│ unique_snps                 │ int          │ Unique SNP count                   │
│ causal_confidence_score     │ float        │ Fine-mapping probability           │
│ evidence_score              │ float        │ GWAS support                       │
│ go_functional_score         │ float        │ PPI/GO pathway score              │
│ mr_score                    │ float        │ Mendelian Randomization score      │
│ risk_weight_score           │ float        │ Combined weighted score            │
│ evidence_level              │ EvidenceLevel│ VERY_STRONG to WEAK               │
│ has_mr_support              │ bool         │ mr_score > 0.3                     │
│ has_pathway_support         │ bool         │ go_functional_score > 0.3          │
└─────────────────────────────┴──────────────┴────────────────────────────────────┘

================================================================================
QUESTION TEMPLATES
================================================================================

POSITIVE EXAMPLES (strong evidence):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Template                  │ Answer                                              │
├───────────────────────────┼─────────────────────────────────────────────────────┤
│ S-GENE-DISEASE           │ Yes, {gene} is associated with {disease}            │
│ C-MR-EVIDENCE            │ Yes, MR score {mr_score} supports causal role       │
│ R-RISK-FACTOR            │ Yes, {gene} is a significant risk factor            │
│ M-PATHWAY                │ Yes, PPI/GO evidence shows functional connection    │
└─────────────────────────────────────────────────────────────────────────────────┘

NEGATIVE EXAMPLES (weak evidence):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Template                  │ Answer                                              │
├───────────────────────────┼─────────────────────────────────────────────────────┤
│ R-RISK-FACTOR-NEG        │ No, {gene} shows weak evidence as risk factor       │
│ C-MR-EVIDENCE-NEG        │ No, insufficient MR evidence for causal claims      │
│ M-PATHWAY-NEG            │ No, limited PPI/GO functional connection            │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
USAGE EXAMPLES
================================================================================

Example 1: Load KG Data
```python
from src.Causal_KG import CAUSALdb2Loader

loader = CAUSALdb2Loader(data_path="data/causaldb2_gene_disease.csv")
pairs = loader.load_gene_disease_pairs()
print(f"Loaded {len(pairs)} gene-disease pairs")
```

Example 2: Filter by Evidence Level
```python
strong_pairs = loader.get_pairs_by_evidence_level(EvidenceLevel.STRONG)
weak_pairs = loader.get_pairs_by_evidence_level(EvidenceLevel.WEAK)
print(f"Strong: {len(strong_pairs)}, Weak: {len(weak_pairs)}")
```

Example 3: Generate Questions
```python
items = loader.generate_benchmark_items(
    max_per_template=100,
    include_negative=True  # Include weak-evidence negative examples
)
```

================================================================================
Author: Sakhaa Alsaedi, sakhaa.alsaedi@kaust.edu.sa
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidenceLevel(str, Enum):
    """Evidence strength levels based on scores."""
    VERY_STRONG = "very_strong"  # MR validated + high scores
    STRONG = "strong"            # High combined score
    MODERATE = "moderate"        # Medium scores
    SUGGESTIVE = "suggestive"    # Low but present evidence
    WEAK = "weak"                # Minimal evidence


class AnswerFormat(str, Enum):
    """
    Answer format types for benchmark questions.

    Each question can be generated in 5 different formats to test
    different reasoning and response capabilities:

    YES_NO:     Binary classification with brief justification
    MCQ:        Multiple choice (A, B, C, D) with one correct answer
    SHORT:      1-2 sentence factual response
    LONG:       Detailed paragraph with explanation
    REASONING:  Step-by-step logical reasoning with conclusion
    """
    YES_NO = "yes_no"
    MCQ = "mcq"
    SHORT = "short"
    LONG = "long"
    REASONING = "reasoning"


@dataclass
class GeneDiseasePair:
    """A gene-disease relationship with evidence scores."""
    gene_id: str
    gene_name: str
    gene_type: str
    disease_id: str
    disease_name: str

    # Evidence scores
    snp_count: int
    unique_snps: int
    causal_confidence_score: float
    evidence_score: float
    go_functional_score: float
    mr_score: float
    risk_weight_score: float

    # Computed fields
    evidence_level: EvidenceLevel = None
    has_mr_support: bool = False
    has_pathway_support: bool = False

    def __post_init__(self):
        """Compute derived fields."""
        self.has_mr_support = self.mr_score > 0.3
        self.has_pathway_support = self.go_functional_score > 0.3
        self.evidence_level = self._compute_evidence_level()

    def _compute_evidence_level(self) -> EvidenceLevel:
        """Compute evidence strength level."""
        # MR validated with high scores = very strong
        if self.mr_score > 0.5 and self.risk_weight_score > 0.7:
            return EvidenceLevel.VERY_STRONG
        # High combined score
        elif self.risk_weight_score > 0.7:
            return EvidenceLevel.STRONG
        # Medium scores
        elif self.risk_weight_score > 0.4:
            return EvidenceLevel.MODERATE
        # Low but present
        elif self.risk_weight_score > 0.2:
            return EvidenceLevel.SUGGESTIVE
        else:
            return EvidenceLevel.WEAK

    def get_evidence_description(self) -> str:
        """Get human-readable evidence description."""
        parts = []

        # SNP support
        if self.snp_count > 100:
            parts.append(f"strong GWAS support ({self.snp_count} SNPs)")
        elif self.snp_count > 10:
            parts.append(f"moderate GWAS support ({self.snp_count} SNPs)")
        else:
            parts.append(f"limited GWAS support ({self.snp_count} SNPs)")

        # MR support
        if self.mr_score > 0.7:
            parts.append("strong Mendelian Randomization evidence")
        elif self.mr_score > 0.3:
            parts.append("moderate MR evidence")

        # Pathway support
        if self.go_functional_score > 0.7:
            parts.append("strong pathway/mechanism relevance")
        elif self.go_functional_score > 0.3:
            parts.append("moderate pathway relevance")

        return "; ".join(parts) if parts else "limited evidence"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'gene_id': self.gene_id,
            'gene_name': self.gene_name,
            'gene_type': self.gene_type,
            'disease_id': self.disease_id,
            'disease_name': self.disease_name,
            'snp_count': self.snp_count,
            'unique_snps': self.unique_snps,
            'causal_confidence_score': self.causal_confidence_score,
            'evidence_score': self.evidence_score,
            'go_functional_score': self.go_functional_score,
            'mr_score': self.mr_score,
            'risk_weight_score': self.risk_weight_score,
            'evidence_level': self.evidence_level.value,
            'has_mr_support': self.has_mr_support,
            'has_pathway_support': self.has_pathway_support,
            'evidence_description': self.get_evidence_description()
        }


class CAUSALdbKnowledgeGraph:
    """
    Loader and query interface for CAUSALdb2 Gene-Disease Knowledge Graph.

    Usage:
        kg = CAUSALdbKnowledgeGraph("/path/to/gene_disease_kg_corrected.csv")

        # Get all pairs for a disease
        diabetes_genes = kg.get_genes_for_disease("Diabetes Mellitus, Type 2")

        # Get top risk factors
        top_genes = kg.get_top_risk_factors("Asthma", n=10)

        # Get MR-validated pairs
        mr_pairs = kg.get_mr_validated_pairs(min_score=0.5)
    """

    def __init__(self, kg_path: str):
        """
        Load the knowledge graph.

        Args:
            kg_path: Path to gene_disease_kg_corrected.csv
        """
        self.kg_path = Path(kg_path)
        self.df = None
        self.pairs: List[GeneDiseasePair] = []

        # Indexes for fast lookup
        self._by_gene: Dict[str, List[GeneDiseasePair]] = {}
        self._by_disease: Dict[str, List[GeneDiseasePair]] = {}
        self._by_evidence_level: Dict[EvidenceLevel, List[GeneDiseasePair]] = {}

        self._load()

    def _load(self):
        """Load and index the knowledge graph."""
        logger.info(f"Loading KG from {self.kg_path}...")

        self.df = pd.read_csv(self.kg_path)
        logger.info(f"Loaded {len(self.df):,} rows")

        # Convert to GeneDiseasePair objects
        for _, row in self.df.iterrows():
            pair = GeneDiseasePair(
                gene_id=row['x_id'],
                gene_name=row['x_name'],
                gene_type=row['gene_type'],
                disease_id=row['y_id'],
                disease_name=row['y_name'],
                snp_count=int(row['snp_count']),
                unique_snps=int(row['unique_snps']),
                causal_confidence_score=float(row['causal_confidence_score']),
                evidence_score=float(row['evidence_score']),
                go_functional_score=float(row['go_functional_score']),
                mr_score=float(row['mr_score']),
                risk_weight_score=float(row['risk_weight_score'])
            )
            self.pairs.append(pair)

            # Index by gene
            if pair.gene_name not in self._by_gene:
                self._by_gene[pair.gene_name] = []
            self._by_gene[pair.gene_name].append(pair)

            # Index by disease
            if pair.disease_name not in self._by_disease:
                self._by_disease[pair.disease_name] = []
            self._by_disease[pair.disease_name].append(pair)

            # Index by evidence level
            if pair.evidence_level not in self._by_evidence_level:
                self._by_evidence_level[pair.evidence_level] = []
            self._by_evidence_level[pair.evidence_level].append(pair)

        logger.info(f"Indexed {len(self.pairs):,} gene-disease pairs")
        logger.info(f"Unique genes: {len(self._by_gene):,}")
        logger.info(f"Unique diseases: {len(self._by_disease):,}")

        # Log evidence level distribution
        for level in EvidenceLevel:
            count = len(self._by_evidence_level.get(level, []))
            logger.info(f"  {level.value}: {count:,} pairs")

    @property
    def genes(self) -> List[str]:
        """Get all unique gene names."""
        return list(self._by_gene.keys())

    @property
    def diseases(self) -> List[str]:
        """Get all unique disease names."""
        return list(self._by_disease.keys())

    def get_genes_for_disease(
        self,
        disease: str,
        min_score: float = 0.0
    ) -> List[GeneDiseasePair]:
        """
        Get all genes associated with a disease.

        Args:
            disease: Disease name
            min_score: Minimum risk_weight_score

        Returns:
            List of GeneDiseasePair sorted by score
        """
        pairs = self._by_disease.get(disease, [])
        filtered = [p for p in pairs if p.risk_weight_score >= min_score]
        return sorted(filtered, key=lambda p: p.risk_weight_score, reverse=True)

    def get_diseases_for_gene(
        self,
        gene: str,
        min_score: float = 0.0
    ) -> List[GeneDiseasePair]:
        """
        Get all diseases associated with a gene.

        Args:
            gene: Gene name
            min_score: Minimum risk_weight_score

        Returns:
            List of GeneDiseasePair sorted by score
        """
        pairs = self._by_gene.get(gene, [])
        filtered = [p for p in pairs if p.risk_weight_score >= min_score]
        return sorted(filtered, key=lambda p: p.risk_weight_score, reverse=True)

    def get_top_risk_factors(
        self,
        disease: str,
        n: int = 10,
        require_mr: bool = False
    ) -> List[GeneDiseasePair]:
        """
        Get top N risk factor genes for a disease.

        Args:
            disease: Disease name
            n: Number of top genes
            require_mr: Only return MR-validated genes

        Returns:
            List of top GeneDiseasePair
        """
        pairs = self.get_genes_for_disease(disease)
        if require_mr:
            pairs = [p for p in pairs if p.has_mr_support]
        return pairs[:n]

    def get_mr_validated_pairs(
        self,
        min_score: float = 0.3
    ) -> List[GeneDiseasePair]:
        """
        Get gene-disease pairs with Mendelian Randomization support.

        MR provides strong causal evidence (natural randomization).

        Args:
            min_score: Minimum MR score

        Returns:
            List of MR-validated pairs
        """
        return [p for p in self.pairs if p.mr_score >= min_score]

    def get_by_evidence_level(
        self,
        level: EvidenceLevel
    ) -> List[GeneDiseasePair]:
        """Get pairs by evidence strength level."""
        return self._by_evidence_level.get(level, [])

    def compare_genes(
        self,
        gene1: str,
        gene2: str,
        disease: str
    ) -> Tuple[Optional[GeneDiseasePair], Optional[GeneDiseasePair]]:
        """
        Compare two genes for the same disease.

        Args:
            gene1: First gene name
            gene2: Second gene name
            disease: Disease name

        Returns:
            Tuple of (pair1, pair2), None if not found
        """
        pair1 = None
        pair2 = None

        for p in self._by_disease.get(disease, []):
            if p.gene_name == gene1:
                pair1 = p
            elif p.gene_name == gene2:
                pair2 = p

        return pair1, pair2

    def get_statistics(self) -> Dict[str, Any]:
        """Get KG statistics."""
        mr_validated = len([p for p in self.pairs if p.has_mr_support])
        pathway_supported = len([p for p in self.pairs if p.has_pathway_support])

        return {
            'total_pairs': len(self.pairs),
            'unique_genes': len(self._by_gene),
            'unique_diseases': len(self._by_disease),
            'mr_validated_pairs': mr_validated,
            'pathway_supported_pairs': pathway_supported,
            'evidence_distribution': {
                level.value: len(self._by_evidence_level.get(level, []))
                for level in EvidenceLevel
            },
            'top_diseases': sorted(
                [(d, len(pairs)) for d, pairs in self._by_disease.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'score_ranges': {
                'risk_weight': (
                    min(p.risk_weight_score for p in self.pairs),
                    max(p.risk_weight_score for p in self.pairs)
                ),
                'mr_score': (
                    min(p.mr_score for p in self.pairs),
                    max(p.mr_score for p in self.pairs)
                ),
                'evidence_score': (
                    min(p.evidence_score for p in self.pairs),
                    max(p.evidence_score for p in self.pairs)
                )
            }
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([p.to_dict() for p in self.pairs])

    def sample_pairs(
        self,
        n: int,
        stratify_by: str = "evidence_level",
        seed: int = 42
    ) -> List[GeneDiseasePair]:
        """
        Sample pairs with stratification.

        Args:
            n: Number of pairs to sample
            stratify_by: Field to stratify by
            seed: Random seed

        Returns:
            Sampled pairs
        """
        np.random.seed(seed)

        if stratify_by == "evidence_level":
            # Sample proportionally from each level
            sampled = []
            per_level = n // len(EvidenceLevel)

            for level in EvidenceLevel:
                level_pairs = self._by_evidence_level.get(level, [])
                if level_pairs:
                    k = min(per_level, len(level_pairs))
                    indices = np.random.choice(len(level_pairs), k, replace=False)
                    sampled.extend([level_pairs[i] for i in indices])

            return sampled
        else:
            # Random sample
            indices = np.random.choice(len(self.pairs), min(n, len(self.pairs)), replace=False)
            return [self.pairs[i] for i in indices]


# =============================================================================
# QUESTION TEMPLATES FOR KG DATA - 5 ANSWER FORMATS
# =============================================================================
#
# Each template now supports 5 answer formats:
# 1. YES_NO:    Binary classification with brief justification
# 2. MCQ:       Multiple choice (A, B, C, D) with one correct answer
# 3. SHORT:     1-2 sentence factual response
# 4. LONG:      Detailed paragraph with explanation
# 5. REASONING: Step-by-step logical reasoning with conclusion
#
# =============================================================================

KG_QUESTION_TEMPLATES = {
    # =========================================================================
    # RISK FACTOR QUESTIONS (R Taxonomy)
    # =========================================================================

    "R-RISK-FACTOR": {
        "taxonomy": "R",
        "category": "Risk Factor Identification",
        "requires": ["gene_name", "disease_name", "evidence_level", "risk_weight_score"],
        "formats": {
            "yes_no": {
                "question": "Is {gene} a genetic risk factor for {disease}?",
                "answer_positive": "Yes. {gene} is a risk factor for {disease} with {evidence_level} evidence (risk score: {risk_score:.2f}).",
                "answer_negative": "No. {gene} shows weak evidence as a risk factor for {disease} (risk score: {risk_score:.2f})."
            },
            "mcq": {
                "question": "What is the evidence level for {gene} as a genetic risk factor for {disease}?",
                "options_generator": "risk_level_options",  # Function to generate options
                "correct_key": "evidence_level"
            },
            "short": {
                "question": "Describe the relationship between {gene} and {disease} risk.",
                "answer": "{gene} is associated with {disease} with {evidence_level} evidence (score: {risk_score:.2f})."
            },
            "long": {
                "question": "Explain the genetic evidence linking {gene} to {disease} risk.",
                "answer": "{gene} has been identified as a genetic risk factor for {disease} through genome-wide association studies. The evidence level is {evidence_level} with a risk weight score of {risk_score:.2f}. {evidence_description}. This association is based on analysis of {snp_count} SNPs across multiple cohorts, providing {evidence_strength} support for the gene's role in disease susceptibility."
            },
            "reasoning": {
                "question": "Based on the available genetic evidence, should {gene} be considered a risk factor for {disease}? Explain your reasoning.",
                "answer": "Step 1: Examine the GWAS evidence - {gene} has {snp_count} associated SNPs with {disease}.\nStep 2: Assess evidence strength - The risk weight score is {risk_score:.2f}, classified as {evidence_level}.\nStep 3: Evaluate supporting data - {evidence_description}.\nStep 4: Consider biological plausibility - {gene} is a {gene_type} gene with known functions.\nConclusion: {conclusion_statement}"
            }
        }
    },

    "R-RISK-LEVEL": {
        "taxonomy": "R",
        "category": "Evidence Strength Assessment",
        "requires": ["gene_name", "disease_name", "evidence_level", "risk_weight_score"],
        "formats": {
            "yes_no": {
                "question": "Does {gene} have strong evidence as a risk factor for {disease}?",
                "answer_positive": "Yes. {gene} has strong evidence with a risk score of {risk_score:.2f}.",
                "answer_negative": "No. {gene} has only {evidence_level} evidence (score: {risk_score:.2f})."
            },
            "mcq": {
                "question": "How would you classify the evidence strength for {gene} in {disease}?",
                "options_generator": "evidence_level_options",
                "correct_key": "evidence_level"
            },
            "short": {
                "question": "What is the evidence strength for {gene} in {disease}?",
                "answer": "The evidence is {evidence_level} (risk weight score: {risk_score:.2f})."
            },
            "long": {
                "question": "Provide a detailed assessment of the evidence strength for {gene} as a risk factor in {disease}.",
                "answer": "The genetic evidence linking {gene} to {disease} is classified as {evidence_level}. The risk weight score of {risk_score:.2f} reflects the combined strength of GWAS signals, fine-mapping probability, and functional annotations. {evidence_description}. This level of evidence {evidence_interpretation}."
            },
            "reasoning": {
                "question": "Evaluate the strength of evidence for {gene} as a risk factor for {disease}. What does the evidence tell us?",
                "answer": "Step 1: Check the risk weight score - {risk_score:.2f} indicates {score_interpretation}.\nStep 2: Examine GWAS support - {snp_count} SNPs support this association.\nStep 3: Consider causal confidence - The causal confidence score is {causal_score:.2f}.\nStep 4: Assess MR evidence - MR score is {mr_score:.2f}, indicating {mr_interpretation}.\nConclusion: The overall evidence is {evidence_level}, {conclusion_statement}."
            }
        }
    },

    "R-COMPARE": {
        "taxonomy": "R",
        "category": "Comparative Risk Assessment",
        "requires": ["gene1", "gene2", "disease_name"],
        "formats": {
            "yes_no": {
                "question": "Is {gene1} a stronger risk factor for {disease} than {gene2}?",
                "answer_positive": "Yes. {gene1} has a higher risk score ({score1:.2f}) compared to {gene2} ({score2:.2f}).",
                "answer_negative": "No. {gene2} has a higher risk score ({score2:.2f}) compared to {gene1} ({score1:.2f})."
            },
            "mcq": {
                "question": "Which gene has stronger evidence as a risk factor for {disease}?",
                "options": {
                    "A": "{gene1} (stronger evidence)",
                    "B": "{gene2} (stronger evidence)",
                    "C": "Both have similar evidence",
                    "D": "Neither has significant evidence"
                },
                "correct_key": "comparison_result"
            },
            "short": {
                "question": "Compare {gene1} and {gene2} as risk factors for {disease}.",
                "answer": "{stronger_gene} has stronger evidence (score: {stronger_score:.2f}) than {weaker_gene} (score: {weaker_score:.2f})."
            },
            "long": {
                "question": "Provide a detailed comparison of {gene1} and {gene2} as genetic risk factors for {disease}.",
                "answer": "Comparing the genetic evidence: {gene1} has a risk weight score of {score1:.2f} with {evidence1}, while {gene2} has a score of {score2:.2f} with {evidence2}. {stronger_gene} shows stronger association with {disease}. The difference in evidence strength can be attributed to {comparison_factors}. Both genes {common_characteristics}."
            },
            "reasoning": {
                "question": "Which gene, {gene1} or {gene2}, should be prioritized as a risk factor for {disease}? Justify your answer.",
                "answer": "Step 1: Compare risk scores - {gene1}: {score1:.2f} vs {gene2}: {score2:.2f}.\nStep 2: Assess MR evidence - {gene1} MR: {mr1:.2f}, {gene2} MR: {mr2:.2f}.\nStep 3: Consider GWAS support - {gene1}: {snp1} SNPs, {gene2}: {snp2} SNPs.\nStep 4: Evaluate functional relevance - {functional_comparison}.\nConclusion: {stronger_gene} should be prioritized because {prioritization_reason}."
            }
        }
    },

    "R-TOP-GENES": {
        "taxonomy": "R",
        "category": "Top Risk Factors",
        "requires": ["disease_name"],
        "formats": {
            "yes_no": {
                "question": "Is {gene1} among the top 3 risk factor genes for {disease}?",
                "answer_positive": "Yes. {gene1} is ranked #{rank} among risk factors for {disease}.",
                "answer_negative": "No. {gene1} is not among the top 3 risk factors for {disease}."
            },
            "mcq": {
                "question": "Which gene is the strongest genetic risk factor for {disease}?",
                "options_generator": "top_genes_options",
                "correct_key": "top_gene"
            },
            "short": {
                "question": "What are the top 3 risk factor genes for {disease}?",
                "answer": "The top 3 are: 1) {gene1}, 2) {gene2}, 3) {gene3}."
            },
            "long": {
                "question": "List and describe the top genetic risk factors for {disease}.",
                "answer": "The top genetic risk factors for {disease} are: 1) {gene1} with a risk score of {score1:.2f} - {desc1}; 2) {gene2} with a score of {score2:.2f} - {desc2}; 3) {gene3} with a score of {score3:.2f} - {desc3}. These genes were identified through comprehensive GWAS meta-analyses and represent the strongest genetic associations with disease risk."
            },
            "reasoning": {
                "question": "Identify and justify the ranking of the top 3 genetic risk factors for {disease}.",
                "answer": "Step 1: Analyze all gene-disease associations for {disease}.\nStep 2: Rank by risk weight score:\n  - #1: {gene1} (score: {score1:.2f}) - {evidence1}\n  - #2: {gene2} (score: {score2:.2f}) - {evidence2}\n  - #3: {gene3} (score: {score3:.2f}) - {evidence3}\nStep 3: Validate rankings with MR evidence.\nConclusion: These rankings reflect the combined GWAS, fine-mapping, and causal inference evidence."
            }
        }
    },

    # =========================================================================
    # CAUSAL EVIDENCE QUESTIONS (C Taxonomy)
    # =========================================================================

    "C-MR-EVIDENCE": {
        "taxonomy": "C",
        "category": "Mendelian Randomization Evidence",
        "requires": ["gene_name", "disease_name", "mr_score"],
        "formats": {
            "yes_no": {
                "question": "Does {gene} have Mendelian Randomization evidence supporting a causal role in {disease}?",
                "answer_positive": "Yes. MR score of {mr_score:.2f} supports a causal relationship.",
                "answer_negative": "No. MR score of {mr_score:.2f} is insufficient for causal claims."
            },
            "mcq": {
                "question": "What does the Mendelian Randomization evidence suggest about {gene}'s relationship with {disease}?",
                "options": {
                    "A": "Strong causal evidence (MR supports causation)",
                    "B": "Moderate causal evidence (MR suggestive)",
                    "C": "Association only (no MR support for causation)",
                    "D": "No evidence of relationship"
                },
                "correct_key": "mr_evidence_category"
            },
            "short": {
                "question": "What is the MR evidence for {gene} in {disease}?",
                "answer": "MR score: {mr_score:.2f}. {mr_interpretation}"
            },
            "long": {
                "question": "Explain the Mendelian Randomization evidence for {gene}'s causal role in {disease}.",
                "answer": "Mendelian Randomization uses genetic variants as instrumental variables to infer causality. For {gene} and {disease}, the MR score is {mr_score:.2f}. {mr_interpretation}. This approach exploits the random allocation of alleles at conception to mimic a randomized controlled trial, reducing confounding from environmental factors. The evidence {mr_conclusion}."
            },
            "reasoning": {
                "question": "Based on Mendelian Randomization analysis, can we infer that {gene} causally affects {disease}? Explain your reasoning.",
                "answer": "Step 1: Understand MR methodology - MR uses genetic variants as natural experiments to test causation.\nStep 2: Examine MR score - {gene} has an MR score of {mr_score:.2f}.\nStep 3: Interpret the score - {mr_interpretation}\nStep 4: Consider MR assumptions - Relevance (genetic variants affect exposure), independence (no confounding of genetic variants), exclusion restriction (effect only through exposure).\nStep 5: Assess potential violations - {mr_violations}.\nConclusion: {mr_conclusion}."
            }
        }
    },

    "C-CAUSAL-STRENGTH": {
        "taxonomy": "C",
        "category": "Causal Evidence Strength",
        "requires": ["gene_name", "disease_name", "causal_confidence_score", "mr_score"],
        "formats": {
            "yes_no": {
                "question": "Is there strong causal evidence linking {gene} to {disease}?",
                "answer_positive": "Yes. Causal confidence score of {causal_score:.2f} with MR support ({mr_score:.2f}) indicates strong causal evidence.",
                "answer_negative": "No. The causal evidence is {evidence_level} (causal score: {causal_score:.2f}, MR: {mr_score:.2f})."
            },
            "mcq": {
                "question": "How would you characterize the causal evidence between {gene} and {disease}?",
                "options": {
                    "A": "Very strong - MR validated with high confidence",
                    "B": "Strong - Good causal support",
                    "C": "Moderate - Some causal evidence",
                    "D": "Weak - Primarily associative"
                },
                "correct_key": "causal_category"
            },
            "short": {
                "question": "Summarize the causal evidence for {gene} in {disease}.",
                "answer": "Causal evidence is {evidence_level}. Causal confidence: {causal_score:.2f}, MR: {mr_score:.2f}."
            },
            "long": {
                "question": "Provide a comprehensive assessment of the causal evidence linking {gene} to {disease}.",
                "answer": "The causal evidence for {gene} in {disease} integrates multiple lines of evidence. The causal confidence score of {causal_score:.2f} reflects fine-mapping posterior probability, while the MR score of {mr_score:.2f} provides {mr_interpretation}. Combined with the GWAS support of {snp_count} variants and GO functional score of {go_score:.2f}, the overall causal evidence is classified as {evidence_level}. {interpretation}"
            },
            "reasoning": {
                "question": "Evaluate whether the evidence supports a causal role for {gene} in {disease}. Distinguish between association and causation.",
                "answer": "Step 1: Define association vs causation - Association shows correlation; causation requires evidence the gene directly affects disease risk.\nStep 2: Review associative evidence - {snp_count} GWAS SNPs, risk score {risk_score:.2f}.\nStep 3: Examine causal evidence - Causal confidence: {causal_score:.2f}, MR score: {mr_score:.2f}.\nStep 4: Apply causal criteria - {causal_criteria_assessment}.\nStep 5: Consider confounding - {confounding_assessment}.\nConclusion: {causal_conclusion}. The evidence {causation_vs_association_verdict}."
            }
        }
    },

    "C-ASSOCIATION-VS-CAUSATION": {
        "taxonomy": "C",
        "category": "Association vs Causation Distinction",
        "requires": ["gene_name", "disease_name", "mr_score", "risk_weight_score"],
        "formats": {
            "yes_no": {
                "question": "Can we claim that {gene} CAUSES {disease} based on available evidence?",
                "answer_positive": "Yes. MR score of {mr_score:.2f} provides causal evidence beyond mere association.",
                "answer_negative": "No. While associated (risk score: {risk_score:.2f}), MR evidence ({mr_score:.2f}) is insufficient for causal claims."
            },
            "mcq": {
                "question": "Which statement best describes the {gene}-{disease} relationship?",
                "options": {
                    "A": "{gene} is causally linked to {disease} (MR validated)",
                    "B": "{gene} is associated with but not proven to cause {disease}",
                    "C": "{gene} shows weak association with {disease}",
                    "D": "No significant relationship between {gene} and {disease}"
                },
                "correct_key": "relationship_type"
            },
            "short": {
                "question": "Is the {gene}-{disease} relationship causal or merely associative?",
                "answer": "{relationship_type}. Association score: {risk_score:.2f}, Causal (MR) score: {mr_score:.2f}."
            },
            "long": {
                "question": "Distinguish between association and causation for the {gene}-{disease} relationship.",
                "answer": "The {gene}-{disease} relationship shows {relationship_type}. GWAS provides evidence of association with a risk weight score of {risk_score:.2f}. However, association does not imply causation due to potential confounding and reverse causation. Mendelian Randomization, which uses genetic variants as natural experiments, shows an MR score of {mr_score:.2f}, {mr_interpretation}. {causal_conclusion}."
            },
            "reasoning": {
                "question": "A colleague claims that {gene} causes {disease}. Critically evaluate this claim using the available evidence.",
                "answer": "Step 1: Acknowledge the association - GWAS shows association with risk score {risk_score:.2f}.\nStep 2: Challenge causation - Association ≠ causation due to confounding, reverse causation, and linkage disequilibrium.\nStep 3: Examine MR evidence - MR score {mr_score:.2f} {mr_interpretation}.\nStep 4: Apply Bradford Hill criteria - {hill_criteria_assessment}.\nStep 5: Consider biological plausibility - {biological_plausibility}.\nConclusion: {claim_verdict}. The evidence {supports_or_refutes} a causal claim."
            }
        }
    },

    # =========================================================================
    # MECHANISM QUESTIONS (M Taxonomy) - M1: Biological Mechanism
    # =========================================================================

    "M-PATHWAY": {
        "taxonomy": "M",
        "category": "Pathway Functional Connection",
        "requires": ["gene_name", "disease_name", "go_functional_score"],
        "formats": {
            "yes_no": {
                "question": "Is {gene} functionally connected to {disease} based on PPI networks and GO term enrichment?",
                "answer_positive": "Yes. PPI/GO functional score of {go_score:.2f} indicates functional connection.",
                "answer_negative": "No. PPI/GO score of {go_score:.2f} shows weak functional connection."
            },
            "mcq": {
                "question": "How strong is the functional/pathway connection between {gene} and {disease}?",
                "options": {
                    "A": "Strong - highly relevant PPI/GO enrichment",
                    "B": "Moderate - some pathway relevance",
                    "C": "Weak - limited functional connection",
                    "D": "None - no significant pathway connection"
                },
                "correct_key": "pathway_strength"
            },
            "short": {
                "question": "Describe the PPI/GO functional connection between {gene} and {disease}.",
                "answer": "{pathway_answer}. PPI/GO score: {go_score:.2f}."
            },
            "long": {
                "question": "Explain the biological pathway evidence linking {gene} to {disease} mechanisms.",
                "answer": "The functional connection between {gene} and {disease} was assessed through protein-protein interaction (PPI) network analysis and Gene Ontology (GO) term enrichment. The GO functional score of {go_score:.2f} indicates {pathway_interpretation}. {gene} interacts with disease-relevant proteins and is enriched for GO terms related to {disease} pathophysiology. {pathway_details}."
            },
            "reasoning": {
                "question": "Based on protein-protein interaction networks and GO annotations, what is the mechanistic basis for {gene}'s role in {disease}?",
                "answer": "Step 1: Examine PPI network - {gene} participates in {ppi_description}.\nStep 2: Analyze GO enrichment - Key GO terms include {go_terms}.\nStep 3: Assess functional relevance - GO functional score: {go_score:.2f}.\nStep 4: Connect to disease mechanisms - {mechanism_connection}.\nStep 5: Consider pathway crosstalk - {pathway_crosstalk}.\nConclusion: {pathway_conclusion}."
            }
        }
    },

    "M-MECHANISM": {
        "taxonomy": "M",
        "category": "Biological Mechanism",
        "requires": ["gene_name", "disease_name", "go_functional_score"],
        "formats": {
            "yes_no": {
                "question": "Does the PPI network and GO annotation of {gene} support its role in {disease} pathogenesis?",
                "answer_positive": "Yes. GO functional score of {go_score:.2f} supports mechanistic involvement.",
                "answer_negative": "No. Limited PPI/GO evidence (score: {go_score:.2f}) for mechanistic role."
            },
            "mcq": {
                "question": "What biological mechanism links {gene} to {disease}?",
                "options_generator": "mechanism_options",
                "correct_key": "primary_mechanism"
            },
            "short": {
                "question": "What is the mechanism by which {gene} contributes to {disease}?",
                "answer": "{gene} shows {go_level} PPI/GO relevance (score: {go_score:.2f}) to {disease} mechanisms."
            },
            "long": {
                "question": "Describe the molecular mechanism by which {gene} may contribute to {disease} development.",
                "answer": "{gene} is a {gene_type} gene that may contribute to {disease} through {primary_mechanism}. PPI network analysis reveals interactions with {ppi_partners}. GO term enrichment shows association with {go_categories}. The functional score of {go_score:.2f} indicates {go_interpretation}. These findings suggest that {gene} {mechanism_hypothesis}."
            },
            "reasoning": {
                "question": "Propose a mechanistic hypothesis for how {gene} might influence {disease} risk. What evidence supports this hypothesis?",
                "answer": "Step 1: Identify gene function - {gene} is a {gene_type} gene involved in {gene_function}.\nStep 2: Examine PPI network - Interacts with {ppi_partners}.\nStep 3: Review GO annotations - Associated with {go_terms}.\nStep 4: Connect to disease biology - {disease} involves {disease_biology}.\nStep 5: Formulate hypothesis - {gene} may affect {disease} by {mechanism_hypothesis}.\nStep 6: Assess supporting evidence - GO score: {go_score:.2f}, {supporting_evidence}.\nConclusion: {mechanism_conclusion}."
            }
        }
    },

    # =========================================================================
    # STRUCTURE QUESTIONS (S Taxonomy)
    # =========================================================================

    "S-GENE-DISEASE": {
        "taxonomy": "S",
        "category": "Gene-Disease Mapping",
        "requires": ["gene_name"],
        "formats": {
            "yes_no": {
                "question": "Does {gene} show genetic associations with multiple diseases?",
                "answer_positive": "Yes. {gene} is associated with {n_diseases} diseases.",
                "answer_negative": "No. {gene} shows association with only one disease."
            },
            "mcq": {
                "question": "How many diseases are associated with {gene}?",
                "options_generator": "disease_count_options",
                "correct_key": "n_diseases"
            },
            "short": {
                "question": "For which diseases does {gene} increase risk?",
                "answer": "{gene} increases risk for {n_diseases} diseases: {disease_list}."
            },
            "long": {
                "question": "Provide a comprehensive overview of diseases associated with {gene}.",
                "answer": "{gene} has been identified as a genetic risk factor for {n_diseases} diseases through GWAS and genetic association studies. The strongest associations are: {detailed_disease_list}. This pattern of associations suggests {gene} may influence {biological_pathway} that is relevant to multiple disease processes. {pleiotropy_interpretation}."
            },
            "reasoning": {
                "question": "Analyze the disease associations of {gene}. What does this tell us about the gene's biological function?",
                "answer": "Step 1: Identify all disease associations - {gene} is linked to {n_diseases} diseases.\nStep 2: List top associations - {disease_list}.\nStep 3: Look for patterns - {disease_categories}.\nStep 4: Consider pleiotropy - {pleiotropy_analysis}.\nStep 5: Infer biological function - {function_inference}.\nConclusion: The multi-disease association pattern suggests {gene} {function_conclusion}."
            }
        }
    },

    "S-SNP-COUNT": {
        "taxonomy": "S",
        "category": "Variant-Gene Mapping",
        "requires": ["gene_name", "disease_name", "snp_count", "unique_snps"],
        "formats": {
            "yes_no": {
                "question": "Is the {gene}-{disease} association supported by multiple independent genetic variants?",
                "answer_positive": "Yes. The association is supported by {snp_count} SNPs ({unique_snps} unique).",
                "answer_negative": "No. Limited SNP support ({snp_count} SNPs) for this association."
            },
            "mcq": {
                "question": "How many genetic variants support the {gene}-{disease} association?",
                "options_generator": "snp_count_options",
                "correct_key": "snp_count_category"
            },
            "short": {
                "question": "How many genetic variants support the {gene}-{disease} association?",
                "answer": "The association is supported by {snp_count} SNPs ({unique_snps} unique variants)."
            },
            "long": {
                "question": "Describe the genetic variant evidence for the {gene}-{disease} association.",
                "answer": "The {gene}-{disease} association is supported by {snp_count} SNPs, of which {unique_snps} are unique variants. This level of SNP support indicates {snp_interpretation}. The variants are distributed across {variant_distribution} and include {variant_types}. Multiple independent signals suggest {replication_interpretation}."
            },
            "reasoning": {
                "question": "Evaluate the robustness of the {gene}-{disease} association based on supporting genetic variants.",
                "answer": "Step 1: Count supporting variants - {snp_count} total SNPs, {unique_snps} unique.\nStep 2: Assess independence - {independence_assessment}.\nStep 3: Consider LD structure - {ld_interpretation}.\nStep 4: Evaluate effect sizes - {effect_size_interpretation}.\nStep 5: Check replication - {replication_status}.\nConclusion: The {gene}-{disease} association is {robustness_verdict} based on {robustness_evidence}."
            }
        }
    },

    # =========================================================================
    # NEGATIVE EXAMPLES (for testing LLM ability to say "No")
    # =========================================================================

    "R-RISK-FACTOR-NEG": {
        "taxonomy": "R",
        "category": "Risk Factor - Negative",
        "requires": ["gene_name", "disease_name", "risk_weight_score"],
        "negative": True,
        "formats": {
            "yes_no": {
                "question": "Is {gene} a significant genetic risk factor for {disease}?",
                "answer": "No. {gene} shows weak evidence as a risk factor for {disease} (risk score: {risk_score:.2f}). {evidence_description}."
            },
            "mcq": {
                "question": "What is the evidence level for {gene} as a risk factor for {disease}?",
                "options": {
                    "A": "Very strong evidence",
                    "B": "Strong evidence",
                    "C": "Moderate evidence",
                    "D": "Weak or no evidence"
                },
                "correct": "D"
            },
            "short": {
                "question": "Is {gene} a risk factor for {disease}?",
                "answer": "No, {gene} has weak evidence (score: {risk_score:.2f}) for {disease}."
            },
            "long": {
                "question": "Evaluate whether {gene} should be considered a risk factor for {disease}.",
                "answer": "Based on current evidence, {gene} should not be considered a significant risk factor for {disease}. The risk weight score is only {risk_score:.2f}, which falls in the weak category. {evidence_description}. This level of evidence is insufficient to establish {gene} as a meaningful genetic risk factor for this disease."
            },
            "reasoning": {
                "question": "Should researchers prioritize {gene} as a risk factor for {disease}? Explain your reasoning.",
                "answer": "Step 1: Check evidence strength - Risk score is {risk_score:.2f} (weak).\nStep 2: Examine GWAS support - {evidence_description}.\nStep 3: Assess MR evidence - MR score: {mr_score:.2f} (insufficient for causal claims).\nStep 4: Consider alternative genes - Other genes show stronger associations.\nConclusion: No, {gene} should not be prioritized as a risk factor for {disease}. The evidence is too weak to warrant further investigation compared to better-supported candidates."
            }
        }
    },

    "C-MR-EVIDENCE-NEG": {
        "taxonomy": "C",
        "category": "MR Evidence - Negative",
        "requires": ["gene_name", "disease_name", "mr_score"],
        "negative": True,
        "formats": {
            "yes_no": {
                "question": "Does {gene} have Mendelian Randomization evidence supporting a causal role in {disease}?",
                "answer": "No. MR score of {mr_score:.2f} is insufficient to support causal claims. {mr_interpretation}"
            },
            "mcq": {
                "question": "Can we claim that {gene} has a causal effect on {disease} based on MR analysis?",
                "options": {
                    "A": "Yes, strong MR evidence supports causation",
                    "B": "Possibly, moderate MR evidence",
                    "C": "Unlikely, weak MR evidence",
                    "D": "No, insufficient MR evidence for causation"
                },
                "correct": "D"
            },
            "short": {
                "question": "What does MR tell us about {gene} causing {disease}?",
                "answer": "MR score ({mr_score:.2f}) is insufficient for causal claims."
            },
            "long": {
                "question": "Evaluate the Mendelian Randomization evidence for {gene}'s causal role in {disease}.",
                "answer": "The Mendelian Randomization analysis for {gene} in {disease} yields a score of {mr_score:.2f}, which is insufficient to support a causal relationship. {mr_interpretation}. While {gene} may show GWAS association with {disease}, this does not imply causation. The lack of MR evidence suggests the observed association may be due to confounding, reverse causation, or linkage disequilibrium rather than a true causal effect."
            },
            "reasoning": {
                "question": "A paper claims {gene} causes {disease}. Based on MR evidence, is this claim justified?",
                "answer": "Step 1: Understand the claim - The paper asserts causal relationship.\nStep 2: Check MR evidence - MR score is {mr_score:.2f}.\nStep 3: Apply MR thresholds - Scores <0.3 are insufficient for causal inference.\nStep 4: Consider alternative explanations - The association may be due to confounding.\nStep 5: Evaluate the claim critically - {claim_evaluation}.\nConclusion: No, the claim is not justified. MR score of {mr_score:.2f} is below the threshold needed to support causation. The authors should be more cautious in their causal language."
            }
        }
    },

    "M-PATHWAY-NEG": {
        "taxonomy": "M",
        "category": "Pathway Connection - Negative",
        "requires": ["gene_name", "disease_name", "go_functional_score"],
        "negative": True,
        "formats": {
            "yes_no": {
                "question": "Is {gene} functionally connected to {disease} based on PPI networks and GO terms?",
                "answer": "No. PPI/GO score of {go_score:.2f} shows weak functional connection. {pathway_interpretation}"
            },
            "mcq": {
                "question": "How would you characterize the functional/pathway connection between {gene} and {disease}?",
                "options": {
                    "A": "Strong connection",
                    "B": "Moderate connection",
                    "C": "Weak connection",
                    "D": "No significant connection"
                },
                "correct": "D"
            },
            "short": {
                "question": "Is there PPI/GO evidence linking {gene} to {disease}?",
                "answer": "No, weak PPI/GO connection (score: {go_score:.2f})."
            },
            "long": {
                "question": "Evaluate the functional pathway evidence connecting {gene} to {disease} mechanisms.",
                "answer": "The protein-protein interaction and Gene Ontology analysis for {gene} in {disease} shows a functional score of only {go_score:.2f}. {pathway_interpretation}. This indicates that {gene} does not share significant functional pathways with known {disease} genes. The lack of PPI network connectivity and GO term enrichment suggests the GWAS association may not reflect a direct mechanistic role for this gene in disease pathogenesis."
            },
            "reasoning": {
                "question": "Does the pathway/mechanism evidence support a functional role for {gene} in {disease}?",
                "answer": "Step 1: Check GO functional score - Score is {go_score:.2f} (weak).\nStep 2: Examine PPI connections - {ppi_analysis}.\nStep 3: Review GO term overlap - {go_term_analysis}.\nStep 4: Consider alternative interpretations - {alternative_interpretation}.\nConclusion: No, the pathway evidence does not support a functional role. The weak PPI/GO score suggests that while {gene} may be genetically associated with {disease}, it likely does not play a direct mechanistic role in disease biology."
            }
        }
    }
}


def load_causaldb_kg(path: str = None) -> CAUSALdbKnowledgeGraph:
    """
    Convenience function to load CAUSALdb2 KG.

    Args:
        path: Path to KG file (default: standard location)

    Returns:
        CAUSALdbKnowledgeGraph instance
    """
    if path is None:
        # Default path
        path = "/ibex/user/alsaedsb/ROCKET/Data/CAUSALdb2/v2.1/kg/gene_disease_kg_corrected.csv"

    return CAUSALdbKnowledgeGraph(path)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Load KG
    kg = load_causaldb_kg()

    # Print statistics
    stats = kg.get_statistics()
    print("\n" + "="*70)
    print("CAUSALdb2 KNOWLEDGE GRAPH STATISTICS")
    print("="*70)
    print(f"Total pairs: {stats['total_pairs']:,}")
    print(f"Unique genes: {stats['unique_genes']:,}")
    print(f"Unique diseases: {stats['unique_diseases']:,}")
    print(f"MR-validated pairs: {stats['mr_validated_pairs']:,}")
    print(f"Pathway-supported pairs: {stats['pathway_supported_pairs']:,}")

    print("\nEvidence distribution:")
    for level, count in stats['evidence_distribution'].items():
        print(f"  {level}: {count:,}")

    print("\nTop diseases:")
    for disease, count in stats['top_diseases']:
        print(f"  {disease}: {count:,} genes")

    # Example queries
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70)

    # Top risk factors for Diabetes
    print("\nTop 5 risk factors for Diabetes Type 2:")
    for pair in kg.get_top_risk_factors("Diabetes Mellitus, Type 2", n=5):
        print(f"  {pair.gene_name}: score={pair.risk_weight_score:.3f}, "
              f"MR={pair.mr_score:.3f}, level={pair.evidence_level.value}")

    # MR-validated pairs
    print("\nMR-validated pairs (top 5 by MR score):")
    mr_pairs = sorted(kg.get_mr_validated_pairs(0.5),
                      key=lambda p: p.mr_score, reverse=True)[:5]
    for pair in mr_pairs:
        print(f"  {pair.gene_name} -> {pair.disease_name}: MR={pair.mr_score:.3f}")


# =============================================================================
# KG QUESTION GENERATOR
# =============================================================================

@dataclass
class KGGeneratedItem:
    """
    A benchmark item generated from KG data.

    Supports 5 answer formats:
    - yes_no: Binary classification with brief justification
    - mcq: Multiple choice (A, B, C, D) with one correct answer
    - short: 1-2 sentence factual response
    - long: Detailed paragraph with explanation
    - reasoning: Step-by-step logical reasoning with conclusion
    """
    id: str
    taxonomy: str
    label: str
    template_id: str
    question: str
    answer: str
    answer_type: str
    answer_format: str  # One of: yes_no, mcq, short, long, reasoning
    entities: Dict[str, Any]
    ground_truth: Dict[str, Any]
    difficulty: str
    evidence_level: str
    source_pair: Dict[str, Any]
    # MCQ-specific fields
    mcq_options: Optional[Dict[str, str]] = None  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    mcq_correct: Optional[str] = None  # "A", "B", "C", or "D"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'id': self.id,
            'taxonomy': self.taxonomy,
            'label': self.label,
            'template_id': self.template_id,
            'question': self.question,
            'answer': self.answer,
            'answer_type': self.answer_type,
            'answer_format': self.answer_format,
            'entities': self.entities,
            'ground_truth': self.ground_truth,
            'difficulty': self.difficulty,
            'evidence_level': self.evidence_level,
            'source_data': self.source_pair
        }
        # Add MCQ fields if present
        if self.mcq_options:
            result['mcq_options'] = self.mcq_options
            result['mcq_correct'] = self.mcq_correct
        return result


class KGQuestionGenerator:
    """
    Generate benchmark questions from CAUSALdb2 Knowledge Graph.

    Supports 5 ANSWER FORMATS per question:
    1. YES_NO:    Binary classification with brief justification
    2. MCQ:       Multiple choice (A, B, C, D) with one correct answer
    3. SHORT:     1-2 sentence factual response
    4. LONG:      Detailed paragraph with explanation
    5. REASONING: Step-by-step logical reasoning with conclusion

    Focuses on testing LLM understanding of:
    - Evidence strength levels (very_strong to weak)
    - Risk factor relationships (R taxonomy)
    - Mendelian Randomization as causal evidence (C taxonomy)
    - Pathway/mechanism relevance (M taxonomy)
    - Gene-disease structural relationships (S taxonomy)

    Usage:
        kg = load_causaldb_kg()
        gen = KGQuestionGenerator(kg)

        # Generate balanced benchmark with all 5 formats
        items = gen.generate_benchmark(n_per_taxonomy=100, formats=["yes_no", "mcq", "short", "long", "reasoning"])

        # Generate from specific evidence levels
        items = gen.generate_by_evidence_level(EvidenceLevel.VERY_STRONG, n=50, format="mcq")

        # Generate single format for a pair
        item = gen.generate_risk_factor_question(pair, format=AnswerFormat.YES_NO)
    """

    # Answer format constants
    ALL_FORMATS = [AnswerFormat.YES_NO, AnswerFormat.MCQ, AnswerFormat.SHORT, AnswerFormat.LONG, AnswerFormat.REASONING]

    def __init__(self, kg: CAUSALdbKnowledgeGraph):
        """
        Initialize with loaded KG.

        Args:
            kg: Loaded CAUSALdbKnowledgeGraph instance
        """
        self.kg = kg
        self.item_counter = {'S': 0, 'C': 0, 'R': 0, 'M': 0}
        self.format_counter = {fmt.value: 0 for fmt in AnswerFormat}

    def _get_evidence_label(self, level: EvidenceLevel) -> str:
        """Get human-readable evidence label."""
        labels = {
            EvidenceLevel.VERY_STRONG: "very strong (MR-validated)",
            EvidenceLevel.STRONG: "strong",
            EvidenceLevel.MODERATE: "moderate",
            EvidenceLevel.SUGGESTIVE: "suggestive",
            EvidenceLevel.WEAK: "weak"
        }
        return labels.get(level, str(level.value))

    def _get_mr_interpretation(self, mr_score: float) -> Tuple[str, str]:
        """Get MR evidence interpretation."""
        if mr_score > 0.7:
            return "Yes", "Strong MR support provides causal evidence through natural genetic randomization."
        elif mr_score > 0.3:
            return "Yes, with moderate confidence", "Moderate MR evidence suggests potential causal relationship."
        elif mr_score > 0:
            return "Weak evidence", "Limited MR support; causal relationship uncertain."
        else:
            return "No MR evidence", "No Mendelian Randomization data available for this relationship."

    def _get_pathway_interpretation(self, go_score: float) -> Tuple[str, str]:
        """Get PPI network and GO term relevance interpretation."""
        if go_score > 0.7:
            return "Yes, strongly connected via PPI networks and enriched GO terms", "strong"
        elif go_score > 0.3:
            return "Moderately connected via PPI/GO analysis", "moderate"
        elif go_score > 0:
            return "Weak PPI/GO connections", "weak"
        else:
            return "No significant PPI/GO enrichment", "unknown"

    def _map_difficulty(self, evidence_level: EvidenceLevel) -> str:
        """Map evidence level to question difficulty."""
        if evidence_level in [EvidenceLevel.VERY_STRONG, EvidenceLevel.WEAK]:
            return "easy"  # Clear-cut cases
        elif evidence_level == EvidenceLevel.MODERATE:
            return "medium"
        else:
            return "hard"  # Ambiguous cases

    def _generate_id(self, taxonomy: str, answer_format: AnswerFormat = None) -> str:
        """Generate unique item ID including format suffix."""
        self.item_counter[taxonomy] += 1
        if answer_format:
            self.format_counter[answer_format.value] += 1
            return f"KG-{taxonomy}-{self.item_counter[taxonomy]:05d}-{answer_format.value}"
        return f"KG-{taxonomy}-{self.item_counter[taxonomy]:05d}"

    # =========================================================================
    # MCQ OPTION GENERATORS
    # =========================================================================

    def _generate_evidence_level_options(self, correct_level: EvidenceLevel) -> Tuple[Dict[str, str], str]:
        """Generate MCQ options for evidence level questions."""
        options = {
            "A": "Very strong (MR-validated)",
            "B": "Strong",
            "C": "Moderate",
            "D": "Weak or suggestive"
        }
        # Map correct answer to option letter
        level_to_option = {
            EvidenceLevel.VERY_STRONG: "A",
            EvidenceLevel.STRONG: "B",
            EvidenceLevel.MODERATE: "C",
            EvidenceLevel.SUGGESTIVE: "D",
            EvidenceLevel.WEAK: "D"
        }
        return options, level_to_option.get(correct_level, "D")

    def _generate_mr_evidence_options(self, mr_score: float) -> Tuple[Dict[str, str], str]:
        """Generate MCQ options for MR evidence questions."""
        options = {
            "A": "Strong causal evidence (MR supports causation)",
            "B": "Moderate causal evidence (MR suggestive)",
            "C": "Association only (no MR support for causation)",
            "D": "No evidence of relationship"
        }
        if mr_score > 0.7:
            correct = "A"
        elif mr_score > 0.3:
            correct = "B"
        elif mr_score > 0:
            correct = "C"
        else:
            correct = "D"
        return options, correct

    def _generate_pathway_strength_options(self, go_score: float) -> Tuple[Dict[str, str], str]:
        """Generate MCQ options for pathway connection questions."""
        options = {
            "A": "Strong - highly relevant PPI/GO enrichment",
            "B": "Moderate - some pathway relevance",
            "C": "Weak - limited functional connection",
            "D": "None - no significant pathway connection"
        }
        if go_score > 0.7:
            correct = "A"
        elif go_score > 0.3:
            correct = "B"
        elif go_score > 0.1:
            correct = "C"
        else:
            correct = "D"
        return options, correct

    def _generate_snp_count_options(self, snp_count: int) -> Tuple[Dict[str, str], str]:
        """Generate MCQ options for SNP count questions."""
        # Generate realistic options based on actual count
        if snp_count > 100:
            options = {
                "A": f"More than 100 SNPs",
                "B": f"50-100 SNPs",
                "C": f"10-50 SNPs",
                "D": f"Fewer than 10 SNPs"
            }
            correct = "A"
        elif snp_count > 50:
            options = {
                "A": f"More than 100 SNPs",
                "B": f"50-100 SNPs",
                "C": f"10-50 SNPs",
                "D": f"Fewer than 10 SNPs"
            }
            correct = "B"
        elif snp_count > 10:
            options = {
                "A": f"More than 50 SNPs",
                "B": f"25-50 SNPs",
                "C": f"10-25 SNPs",
                "D": f"Fewer than 10 SNPs"
            }
            correct = "C"
        else:
            options = {
                "A": f"More than 50 SNPs",
                "B": f"10-50 SNPs",
                "C": f"5-10 SNPs",
                "D": f"Fewer than 5 SNPs"
            }
            correct = "D" if snp_count < 5 else "C"
        return options, correct

    def _generate_comparison_options(
        self,
        gene1: str,
        gene2: str,
        score1: float,
        score2: float
    ) -> Tuple[Dict[str, str], str]:
        """Generate MCQ options for gene comparison questions."""
        options = {
            "A": f"{gene1} (stronger evidence)",
            "B": f"{gene2} (stronger evidence)",
            "C": "Both have similar evidence",
            "D": "Neither has significant evidence"
        }
        diff = abs(score1 - score2)
        if diff < 0.1:
            correct = "C"
        elif score1 > score2:
            correct = "A"
        else:
            correct = "B"
        # Check if neither is significant
        if score1 < 0.2 and score2 < 0.2:
            correct = "D"
        return options, correct

    # =========================================================================
    # MULTI-FORMAT QUESTION GENERATOR
    # =========================================================================

    def generate_question_all_formats(
        self,
        pair: GeneDiseasePair,
        template_id: str,
        formats: List[AnswerFormat] = None
    ) -> List[KGGeneratedItem]:
        """
        Generate a question in multiple answer formats.

        Args:
            pair: GeneDiseasePair to generate questions for
            template_id: Template ID (e.g., "R-RISK-FACTOR")
            formats: List of formats to generate (default: all 5)

        Returns:
            List of KGGeneratedItem, one per format
        """
        if formats is None:
            formats = self.ALL_FORMATS

        items = []
        template = KG_QUESTION_TEMPLATES.get(template_id)
        if not template:
            logger.warning(f"Unknown template: {template_id}")
            return items

        for fmt in formats:
            item = self._generate_single_format(pair, template_id, template, fmt)
            if item:
                items.append(item)

        return items

    def _generate_single_format(
        self,
        pair: GeneDiseasePair,
        template_id: str,
        template: Dict,
        answer_format: AnswerFormat
    ) -> Optional[KGGeneratedItem]:
        """
        Generate a single question in a specific format.

        Args:
            pair: GeneDiseasePair data
            template_id: Template ID
            template: Template dictionary
            answer_format: The answer format to use

        Returns:
            KGGeneratedItem or None if generation fails
        """
        fmt_key = answer_format.value
        formats = template.get("formats", {})

        if fmt_key not in formats:
            return None

        fmt_template = formats[fmt_key]
        taxonomy = template["taxonomy"]
        is_negative = template.get("negative", False)

        # Prepare substitution values
        values = self._prepare_template_values(pair, is_negative)

        try:
            # Generate question
            question = fmt_template.get("question", "").format(**values)

            # Generate answer based on format type
            if fmt_key == "yes_no":
                if is_negative:
                    answer = fmt_template.get("answer", "").format(**values)
                else:
                    # Use positive or negative answer based on evidence
                    if pair.risk_weight_score > 0.3:
                        answer = fmt_template.get("answer_positive", "").format(**values)
                    else:
                        answer = fmt_template.get("answer_negative", "").format(**values)

            elif fmt_key == "mcq":
                # Generate MCQ options
                options, correct = self._get_mcq_options(pair, template_id, fmt_template)
                answer = f"Correct answer: {correct}"
                # Store MCQ-specific data
                return KGGeneratedItem(
                    id=self._generate_id(taxonomy, answer_format),
                    taxonomy=taxonomy,
                    label=template_id,
                    template_id=template_id,
                    question=question,
                    answer=answer,
                    answer_type="mcq",
                    answer_format=fmt_key,
                    entities={
                        'gene': pair.gene_name,
                        'disease': pair.disease_name
                    },
                    ground_truth={
                        'correct_option': correct,
                        'options': options,
                        'evidence': pair.to_dict()
                    },
                    difficulty=self._map_difficulty(pair.evidence_level),
                    evidence_level=pair.evidence_level.value,
                    source_pair=pair.to_dict(),
                    mcq_options=options,
                    mcq_correct=correct
                )

            else:
                # SHORT, LONG, REASONING formats
                answer = fmt_template.get("answer", "").format(**values)

            return KGGeneratedItem(
                id=self._generate_id(taxonomy, answer_format),
                taxonomy=taxonomy,
                label=template_id,
                template_id=template_id,
                question=question,
                answer=answer,
                answer_type=fmt_key,
                answer_format=fmt_key,
                entities={
                    'gene': pair.gene_name,
                    'disease': pair.disease_name
                },
                ground_truth={
                    'answer': answer,
                    'evidence': pair.to_dict(),
                    'confidence': 1.0
                },
                difficulty=self._map_difficulty(pair.evidence_level),
                evidence_level=pair.evidence_level.value,
                source_pair=pair.to_dict()
            )

        except KeyError as e:
            logger.warning(f"Missing key in template {template_id}: {e}")
            return None

    def _prepare_template_values(self, pair: GeneDiseasePair, is_negative: bool = False) -> Dict[str, Any]:
        """Prepare all substitution values for template formatting."""
        mr_answer, mr_interp = self._get_mr_interpretation(pair.mr_score)
        pathway_answer, pathway_level = self._get_pathway_interpretation(pair.go_functional_score)

        # Determine evidence strength description
        if pair.risk_weight_score > 0.7:
            evidence_strength = "strong"
        elif pair.risk_weight_score > 0.4:
            evidence_strength = "moderate"
        else:
            evidence_strength = "limited"

        # Conclusion statement based on evidence
        if pair.evidence_level == EvidenceLevel.VERY_STRONG:
            conclusion = f"Yes, {pair.gene_name} should be considered a significant risk factor for {pair.disease_name}"
        elif pair.evidence_level == EvidenceLevel.STRONG:
            conclusion = f"Yes, {pair.gene_name} is likely a risk factor for {pair.disease_name}"
        elif pair.evidence_level == EvidenceLevel.MODERATE:
            conclusion = f"{pair.gene_name} may be a risk factor for {pair.disease_name}, but more evidence is needed"
        else:
            conclusion = f"{pair.gene_name} should not be prioritized as a risk factor for {pair.disease_name}"

        # MR conclusion
        if pair.mr_score > 0.5:
            mr_conclusion = "supports a causal relationship"
        elif pair.mr_score > 0.3:
            mr_conclusion = "provides suggestive causal evidence"
        else:
            mr_conclusion = "is insufficient to establish causation"

        return {
            # Basic gene/disease info
            'gene': pair.gene_name,
            'gene_name': pair.gene_name,
            'gene_type': pair.gene_type,
            'disease': pair.disease_name,
            'disease_name': pair.disease_name,

            # Scores
            'risk_score': pair.risk_weight_score,
            'mr_score': pair.mr_score,
            'go_score': pair.go_functional_score,
            'causal_score': pair.causal_confidence_score,
            'evidence_score': pair.evidence_score,

            # Counts
            'snp_count': pair.snp_count,
            'unique_snps': pair.unique_snps,

            # Evidence level
            'evidence_level': self._get_evidence_label(pair.evidence_level),
            'evidence_description': pair.get_evidence_description(),
            'evidence_strength': evidence_strength,

            # Interpretations
            'mr_answer': mr_answer,
            'mr_interpretation': mr_interp,
            'mr_conclusion': mr_conclusion,
            'pathway_answer': pathway_answer,
            'go_level': pathway_level,
            'pathway_interpretation': f"The gene shows {pathway_level} pathway relevance to disease mechanisms.",

            # Conclusions
            'conclusion_statement': conclusion,
            'score_interpretation': "strong evidence" if pair.risk_weight_score > 0.7 else "moderate evidence" if pair.risk_weight_score > 0.4 else "weak evidence",

            # Placeholders for complex reasoning (will be populated by LLM or refined templates)
            'mr_violations': "No major violations detected" if pair.mr_score > 0.3 else "Potential weak instrument or pleiotropic effects",
            'causal_criteria_assessment': "Meets several Bradford Hill criteria" if pair.evidence_level in [EvidenceLevel.VERY_STRONG, EvidenceLevel.STRONG] else "Limited support for causal criteria",
            'confounding_assessment': "MR design reduces confounding" if pair.mr_score > 0.3 else "Confounding cannot be ruled out",
            'causal_conclusion': "Evidence supports a causal role" if pair.mr_score > 0.5 else "Causal relationship uncertain",
            'causation_vs_association_verdict': "supports causation" if pair.mr_score > 0.5 else "is primarily associative",
            'pathway_details': f"GO functional score: {pair.go_functional_score:.2f}",
            'mechanism_conclusion': f"The pathway evidence {'supports' if pair.go_functional_score > 0.3 else 'does not strongly support'} a functional role",

            # M-PATHWAY reasoning placeholders
            'ppi_description': f"disease-relevant protein interaction networks (GO score: {pair.go_functional_score:.2f})",
            'go_terms': "biological process, cellular component, and molecular function terms relevant to disease",
            'mechanism_connection': f"The gene's PPI network shows {'strong' if pair.go_functional_score > 0.5 else 'moderate' if pair.go_functional_score > 0.3 else 'weak'} overlap with disease pathways",
            'pathway_crosstalk': "Cross-pathway interactions may contribute to disease phenotype",
            'pathway_conclusion': f"The pathway analysis {'supports' if pair.go_functional_score > 0.3 else 'does not strongly support'} a mechanistic role for {pair.gene_name} in {pair.disease_name}",
            'ppi_partners': "disease-associated proteins",
            'go_categories': "relevant biological processes",
            'go_interpretation': f"{'strong' if pair.go_functional_score > 0.5 else 'moderate' if pair.go_functional_score > 0.3 else 'weak'} functional relevance",
            'gene_function': f"functions related to {pair.disease_name} biology",
            'disease_biology': "disease-specific biological processes",
            'mechanism_hypothesis': f"affecting {pair.disease_name}-related pathways",
            'primary_mechanism': "pathway-mediated effects",
            'supporting_evidence': f"GO score of {pair.go_functional_score:.2f}",

            # S-SNP-COUNT placeholders
            'snp_interpretation': f"{'strong' if pair.snp_count > 50 else 'moderate' if pair.snp_count > 10 else 'limited'} GWAS support for the association",
            'variant_distribution': "intronic and intergenic regions with some coding variants",
            'variant_types': "primarily common variants (MAF > 0.01)",
            'replication_interpretation': f"{'robust replication across cohorts' if pair.snp_count > 50 else 'some independent replication' if pair.snp_count > 10 else 'limited replication data'}",
            'independence_assessment': f"{'Multiple independent signals suggest true association' if pair.unique_snps > 10 else 'Few independent signals available'}",
            'ld_interpretation': f"Considering LD structure, {pair.unique_snps} unique variants from {pair.snp_count} total SNPs suggest {'multiple causal variants' if pair.unique_snps > 10 else 'potential single causal locus'}",
            'effect_size_interpretation': "Effect sizes are typical for complex trait associations",
            'replication_status': f"{'Well-replicated' if pair.snp_count > 50 else 'Partially replicated' if pair.snp_count > 10 else 'Limited replication'}",
            'robustness_verdict': f"{'robust' if pair.snp_count > 50 and pair.unique_snps > 10 else 'moderately robust' if pair.snp_count > 10 else 'preliminary'}",
            'robustness_evidence': f"{pair.snp_count} supporting SNPs with {pair.unique_snps} unique variants",

            # Negative example placeholders
            'ppi_analysis': f"Limited PPI connections (GO score: {pair.go_functional_score:.2f})",
            'go_term_analysis': "Minimal GO term overlap with disease-associated genes",
            'alternative_interpretation': "The association may be due to LD with nearby genes rather than direct functional involvement",
            'claim_evaluation': f"The MR score of {pair.mr_score:.2f} does not support the causal claim",
            'biological_plausibility': f"{pair.gene_name} has {'known' if pair.go_functional_score > 0.3 else 'limited'} biological connections to {pair.disease_name}",
            'hill_criteria_assessment': f"{'Several' if pair.mr_score > 0.5 else 'Few'} Bradford Hill criteria are met",
            'claim_verdict': f"The causal claim is {'supported' if pair.mr_score > 0.5 else 'not well-supported' if pair.mr_score > 0.3 else 'not justified'}",
            'supports_or_refutes': "supports" if pair.mr_score > 0.5 else "does not conclusively support",
            'relationship_type': "Causal (MR-validated)" if pair.mr_score > 0.5 else "Associated but not causally proven" if pair.risk_weight_score > 0.3 else "Weak association",
            'evidence_interpretation': f"{'warrants further investigation' if pair.evidence_level in [EvidenceLevel.VERY_STRONG, EvidenceLevel.STRONG] else 'is hypothesis-generating' if pair.evidence_level == EvidenceLevel.MODERATE else 'requires additional validation'}",
            'interpretation': f"Based on combined evidence, {pair.gene_name} {'is' if pair.evidence_level in [EvidenceLevel.VERY_STRONG, EvidenceLevel.STRONG] else 'may be'} involved in {pair.disease_name} pathogenesis"
        }

    def _get_mcq_options(
        self,
        pair: GeneDiseasePair,
        template_id: str,
        fmt_template: Dict
    ) -> Tuple[Dict[str, str], str]:
        """Get MCQ options and correct answer for a template."""
        # Check for static options in template
        if "options" in fmt_template:
            options = fmt_template["options"]
            # Format option text with pair values
            values = self._prepare_template_values(pair)
            formatted_options = {}
            for key, text in options.items():
                try:
                    formatted_options[key] = text.format(**values)
                except KeyError:
                    formatted_options[key] = text
            # Get correct answer
            if "correct" in fmt_template:
                return formatted_options, fmt_template["correct"]
            else:
                # Determine correct based on template_id
                return formatted_options, self._determine_mcq_correct(pair, template_id)

        # Use generator function
        options_gen = fmt_template.get("options_generator", "")
        if options_gen == "risk_level_options" or options_gen == "evidence_level_options":
            return self._generate_evidence_level_options(pair.evidence_level)
        elif options_gen == "mr_evidence_options":
            return self._generate_mr_evidence_options(pair.mr_score)
        elif options_gen == "pathway_strength_options":
            return self._generate_pathway_strength_options(pair.go_functional_score)
        elif options_gen == "snp_count_options":
            return self._generate_snp_count_options(pair.snp_count)
        else:
            # Default options
            return self._generate_evidence_level_options(pair.evidence_level)

    def _determine_mcq_correct(self, pair: GeneDiseasePair, template_id: str) -> str:
        """Determine correct MCQ option based on evidence."""
        if "C-MR" in template_id or "C-CAUSAL" in template_id:
            if pair.mr_score > 0.7:
                return "A"
            elif pair.mr_score > 0.3:
                return "B"
            else:
                return "C"
        elif "M-PATHWAY" in template_id or "M-MECHANISM" in template_id:
            if pair.go_functional_score > 0.7:
                return "A"
            elif pair.go_functional_score > 0.3:
                return "B"
            else:
                return "C"
        else:
            # Risk/general evidence
            if pair.evidence_level == EvidenceLevel.VERY_STRONG:
                return "A"
            elif pair.evidence_level == EvidenceLevel.STRONG:
                return "B"
            elif pair.evidence_level == EvidenceLevel.MODERATE:
                return "C"
            else:
                return "D"

    # =========================================================================
    # BACKWARD COMPATIBLE GENERATOR METHODS
    # =========================================================================

    def generate_risk_factor_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.LONG
    ) -> Optional[KGGeneratedItem]:
        """
        Generate R-RISK-FACTOR question in specified format.

        Args:
            pair: GeneDiseasePair to generate question for
            answer_format: Answer format (default: LONG)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        template = KG_QUESTION_TEMPLATES["R-RISK-FACTOR"]
        return self._generate_single_format(pair, "R-RISK-FACTOR", template, answer_format)

    def generate_risk_level_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.LONG
    ) -> Optional[KGGeneratedItem]:
        """
        Generate R-RISK-LEVEL question about evidence strength.

        Args:
            pair: GeneDiseasePair to generate question for
            answer_format: Answer format (default: LONG)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        template = KG_QUESTION_TEMPLATES["R-RISK-LEVEL"]
        return self._generate_single_format(pair, "R-RISK-LEVEL", template, answer_format)

    def generate_compare_question(
        self,
        pair1: GeneDiseasePair,
        pair2: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.SHORT
    ) -> Optional[KGGeneratedItem]:
        """
        Generate R-COMPARE question comparing two genes for same disease.

        Args:
            pair1: First gene-disease pair
            pair2: Second gene-disease pair (must have same disease)
            answer_format: Answer format (default: SHORT)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        if pair1.disease_name != pair2.disease_name:
            return None

        # Determine stronger gene
        if pair1.risk_weight_score >= pair2.risk_weight_score:
            stronger, weaker = pair1, pair2
        else:
            stronger, weaker = pair2, pair1

        # Create a custom values dict for comparison questions
        fmt_key = answer_format.value
        template = KG_QUESTION_TEMPLATES["R-COMPARE"]
        formats = template.get("formats", {})

        if fmt_key not in formats:
            return None

        fmt_template = formats[fmt_key]

        # Build comparison-specific values
        values = {
            'gene1': pair1.gene_name,
            'gene2': pair2.gene_name,
            'disease': pair1.disease_name,
            'disease_name': pair1.disease_name,
            'score1': pair1.risk_weight_score,
            'score2': pair2.risk_weight_score,
            'stronger_gene': stronger.gene_name,
            'weaker_gene': weaker.gene_name,
            'stronger_score': stronger.risk_weight_score,
            'weaker_score': weaker.risk_weight_score,
            'mr1': pair1.mr_score,
            'mr2': pair2.mr_score,
            'snp1': pair1.snp_count,
            'snp2': pair2.snp_count,
            'evidence1': pair1.get_evidence_description(),
            'evidence2': pair2.get_evidence_description(),
            'functional_comparison': f"{pair1.gene_name} GO: {pair1.go_functional_score:.2f}, {pair2.gene_name} GO: {pair2.go_functional_score:.2f}",
            'comparison_factors': "differences in GWAS support and MR evidence",
            'common_characteristics': "are protein-coding genes with GWAS associations",
            'prioritization_reason': f"higher risk score ({stronger.risk_weight_score:.2f} vs {weaker.risk_weight_score:.2f})"
        }

        try:
            question = fmt_template.get("question", "").format(**values)

            if fmt_key == "yes_no":
                if pair1.risk_weight_score > pair2.risk_weight_score:
                    answer = fmt_template.get("answer_positive", "").format(**values)
                else:
                    answer = fmt_template.get("answer_negative", "").format(**values)
            elif fmt_key == "mcq":
                options, correct = self._generate_comparison_options(
                    pair1.gene_name, pair2.gene_name,
                    pair1.risk_weight_score, pair2.risk_weight_score
                )
                answer = f"Correct answer: {correct}"
                return KGGeneratedItem(
                    id=self._generate_id("R", answer_format),
                    taxonomy="R",
                    label="R-COMPARE",
                    template_id="R-COMPARE",
                    question=question,
                    answer=answer,
                    answer_type="mcq",
                    answer_format=fmt_key,
                    entities={
                        'gene1': pair1.gene_name,
                        'gene2': pair2.gene_name,
                        'disease': pair1.disease_name
                    },
                    ground_truth={
                        'correct_option': correct,
                        'options': options,
                        'stronger_gene': stronger.gene_name
                    },
                    difficulty="medium",
                    evidence_level=stronger.evidence_level.value,
                    source_pair={'pair1': pair1.to_dict(), 'pair2': pair2.to_dict()},
                    mcq_options=options,
                    mcq_correct=correct
                )
            else:
                answer = fmt_template.get("answer", "").format(**values)

            return KGGeneratedItem(
                id=self._generate_id("R", answer_format),
                taxonomy="R",
                label="R-COMPARE",
                template_id="R-COMPARE",
                question=question,
                answer=answer,
                answer_type=fmt_key,
                answer_format=fmt_key,
                entities={
                    'gene1': pair1.gene_name,
                    'gene2': pair2.gene_name,
                    'disease': pair1.disease_name
                },
                ground_truth={
                    'answer': answer,
                    'stronger_gene': stronger.gene_name,
                    'confidence': 1.0
                },
                difficulty="medium",
                evidence_level=stronger.evidence_level.value,
                source_pair={'pair1': pair1.to_dict(), 'pair2': pair2.to_dict()}
            )
        except KeyError as e:
            logger.warning(f"Missing key in R-COMPARE template: {e}")
            return None

    def generate_top_genes_question(
        self,
        disease: str,
        answer_format: AnswerFormat = AnswerFormat.SHORT
    ) -> Optional[KGGeneratedItem]:
        """
        Generate R-TOP-GENES question for a disease.

        Args:
            disease: Disease name
            answer_format: Answer format (default: SHORT)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        top_genes = self.kg.get_top_risk_factors(disease, n=5)
        if len(top_genes) < 3:
            return None

        fmt_key = answer_format.value
        template = KG_QUESTION_TEMPLATES["R-TOP-GENES"]
        formats = template.get("formats", {})

        if fmt_key not in formats:
            return None

        fmt_template = formats[fmt_key]

        values = {
            'disease': disease,
            'disease_name': disease,
            'gene1': top_genes[0].gene_name,
            'gene2': top_genes[1].gene_name,
            'gene3': top_genes[2].gene_name,
            'score1': top_genes[0].risk_weight_score,
            'score2': top_genes[1].risk_weight_score,
            'score3': top_genes[2].risk_weight_score,
            'desc1': top_genes[0].get_evidence_description(),
            'desc2': top_genes[1].get_evidence_description(),
            'desc3': top_genes[2].get_evidence_description(),
            'evidence1': top_genes[0].get_evidence_description(),
            'evidence2': top_genes[1].get_evidence_description(),
            'evidence3': top_genes[2].get_evidence_description(),
            'rank': 1
        }

        try:
            question = fmt_template.get("question", "").format(**values)

            if fmt_key == "yes_no":
                answer = fmt_template.get("answer_positive", "").format(**values)
            elif fmt_key == "mcq":
                # Generate MCQ with top gene options
                options = {
                    "A": top_genes[0].gene_name,
                    "B": top_genes[1].gene_name if len(top_genes) > 1 else "None",
                    "C": top_genes[2].gene_name if len(top_genes) > 2 else "None",
                    "D": top_genes[3].gene_name if len(top_genes) > 3 else "None"
                }
                correct = "A"  # Top gene is always first
                answer = f"Correct answer: {correct}"
                return KGGeneratedItem(
                    id=self._generate_id("R", answer_format),
                    taxonomy="R",
                    label="R-TOP-GENES",
                    template_id="R-TOP-GENES",
                    question=question,
                    answer=answer,
                    answer_type="mcq",
                    answer_format=fmt_key,
                    entities={'disease': disease, 'genes': [p.gene_name for p in top_genes[:3]]},
                    ground_truth={
                        'correct_option': correct,
                        'options': options,
                        'top_genes': [p.gene_name for p in top_genes[:3]]
                    },
                    difficulty="hard",
                    evidence_level=top_genes[0].evidence_level.value,
                    source_pair={'disease': disease, 'top_pairs': [p.to_dict() for p in top_genes[:3]]},
                    mcq_options=options,
                    mcq_correct=correct
                )
            else:
                answer = fmt_template.get("answer", "").format(**values)

            return KGGeneratedItem(
                id=self._generate_id("R", answer_format),
                taxonomy="R",
                label="R-TOP-GENES",
                template_id="R-TOP-GENES",
                question=question,
                answer=answer,
                answer_type=fmt_key,
                answer_format=fmt_key,
                entities={'disease': disease, 'genes': [p.gene_name for p in top_genes[:3]]},
                ground_truth={
                    'answer': answer,
                    'top_genes': [p.gene_name for p in top_genes[:3]],
                    'confidence': 1.0
                },
                difficulty="hard",
                evidence_level=top_genes[0].evidence_level.value,
                source_pair={'disease': disease, 'top_pairs': [p.to_dict() for p in top_genes[:3]]}
            )
        except KeyError as e:
            logger.warning(f"Missing key in R-TOP-GENES template: {e}")
            return None

    def generate_mr_evidence_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.LONG
    ) -> Optional[KGGeneratedItem]:
        """
        Generate C-MR-EVIDENCE question about Mendelian Randomization.

        Args:
            pair: GeneDiseasePair to generate question for
            answer_format: Answer format (default: LONG)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        template = KG_QUESTION_TEMPLATES["C-MR-EVIDENCE"]
        return self._generate_single_format(pair, "C-MR-EVIDENCE", template, answer_format)

    def generate_causal_strength_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.LONG
    ) -> Optional[KGGeneratedItem]:
        """
        Generate C-CAUSAL-STRENGTH question about causal evidence.

        Args:
            pair: GeneDiseasePair to generate question for
            answer_format: Answer format (default: LONG)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        template = KG_QUESTION_TEMPLATES["C-CAUSAL-STRENGTH"]
        return self._generate_single_format(pair, "C-CAUSAL-STRENGTH", template, answer_format)

    def generate_pathway_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.LONG
    ) -> Optional[KGGeneratedItem]:
        """
        Generate M-PATHWAY question about functional relevance.

        Args:
            pair: GeneDiseasePair to generate question for
            answer_format: Answer format (default: LONG)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        template = KG_QUESTION_TEMPLATES["M-PATHWAY"]
        return self._generate_single_format(pair, "M-PATHWAY", template, answer_format)

    def generate_snp_count_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.SHORT
    ) -> Optional[KGGeneratedItem]:
        """
        Generate S-SNP-COUNT question about genetic variants.

        Args:
            pair: GeneDiseasePair to generate question for
            answer_format: Answer format (default: SHORT)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        template = KG_QUESTION_TEMPLATES["S-SNP-COUNT"]
        return self._generate_single_format(pair, "S-SNP-COUNT", template, answer_format)

    def generate_gene_diseases_question(
        self,
        gene: str,
        answer_format: AnswerFormat = AnswerFormat.SHORT
    ) -> Optional[KGGeneratedItem]:
        """
        Generate S-GENE-DISEASE question about diseases for a gene.

        Args:
            gene: Gene name
            answer_format: Answer format (default: SHORT)

        Returns:
            KGGeneratedItem or None if generation fails
        """
        diseases = self.kg.get_diseases_for_gene(gene)
        if len(diseases) < 1:
            return None

        # Use the first disease pair as the base for template formatting
        pair = diseases[0]
        fmt_key = answer_format.value
        template = KG_QUESTION_TEMPLATES["S-GENE-DISEASE"]
        formats = template.get("formats", {})

        if fmt_key not in formats:
            return None

        fmt_template = formats[fmt_key]

        # Build values for S-GENE-DISEASE
        top_diseases = diseases[:5]
        disease_list = ", ".join([p.disease_name for p in top_diseases])
        detailed_disease_list = "; ".join([
            f"{p.disease_name} (score: {p.risk_weight_score:.2f})"
            for p in top_diseases
        ])

        values = {
            'gene': gene,
            'gene_name': gene,
            'n_diseases': len(diseases),
            'disease_list': disease_list,
            'detailed_disease_list': detailed_disease_list,
            'biological_pathway': "multiple biological pathways",
            'pleiotropy_interpretation': f"This gene shows pleiotropy, affecting {len(diseases)} different diseases.",
            'disease_categories': "various disease categories",
            'pleiotropy_analysis': f"The gene affects {len(diseases)} diseases, suggesting broad biological impact.",
            'function_inference': "The diverse disease associations suggest involvement in fundamental biological processes.",
            'function_conclusion': "plays a role in multiple biological pathways."
        }

        try:
            question = fmt_template.get("question", "").format(**values)

            if fmt_key == "yes_no":
                if len(diseases) > 1:
                    answer = fmt_template.get("answer_positive", "").format(**values)
                else:
                    answer = fmt_template.get("answer_negative", "").format(**values)
            elif fmt_key == "mcq":
                options = {
                    "A": "1 disease",
                    "B": "2-5 diseases",
                    "C": "6-10 diseases",
                    "D": "More than 10 diseases"
                }
                if len(diseases) == 1:
                    correct = "A"
                elif len(diseases) <= 5:
                    correct = "B"
                elif len(diseases) <= 10:
                    correct = "C"
                else:
                    correct = "D"
                answer = f"Correct answer: {correct}"
                return KGGeneratedItem(
                    id=self._generate_id("S", answer_format),
                    taxonomy="S",
                    label="S-GENE-DISEASE",
                    template_id="S-GENE-DISEASE",
                    question=question,
                    answer=answer,
                    answer_type="mcq",
                    answer_format=fmt_key,
                    entities={'gene': gene, 'diseases': [p.disease_name for p in top_diseases]},
                    ground_truth={
                        'correct_option': correct,
                        'options': options,
                        'n_diseases': len(diseases)
                    },
                    difficulty="medium",
                    evidence_level=top_diseases[0].evidence_level.value if top_diseases else "weak",
                    source_pair={'gene': gene, 'disease_pairs': [p.to_dict() for p in top_diseases]},
                    mcq_options=options,
                    mcq_correct=correct
                )
            else:
                answer = fmt_template.get("answer", "").format(**values)

            return KGGeneratedItem(
                id=self._generate_id("S", answer_format),
                taxonomy="S",
                label="S-GENE-DISEASE",
                template_id="S-GENE-DISEASE",
                question=question,
                answer=answer,
                answer_type=fmt_key,
                answer_format=fmt_key,
                entities={'gene': gene, 'diseases': [p.disease_name for p in top_diseases]},
                ground_truth={
                    'answer': answer,
                    'n_diseases': len(diseases),
                    'top_diseases': [p.disease_name for p in top_diseases],
                    'confidence': 1.0
                },
                difficulty="medium",
                evidence_level=top_diseases[0].evidence_level.value if top_diseases else "weak",
                source_pair={'gene': gene, 'disease_pairs': [p.to_dict() for p in top_diseases]}
            )
        except KeyError as e:
            logger.warning(f"Missing key in S-GENE-DISEASE template: {e}")
            return None

    # =========================================================================
    # NEGATIVE EXAMPLE GENERATORS (Low-score pairs)
    # =========================================================================

    def generate_negative_risk_factor_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.YES_NO
    ) -> Optional[KGGeneratedItem]:
        """
        Generate R-RISK-FACTOR-NEG question for weak gene-disease associations.

        Args:
            pair: GeneDiseasePair with weak evidence
            answer_format: Answer format (default: YES_NO)

        Returns:
            KGGeneratedItem or None if pair has strong evidence
        """
        # Only generate for weak/suggestive evidence
        if pair.risk_weight_score > 0.3:
            return None

        template = KG_QUESTION_TEMPLATES["R-RISK-FACTOR-NEG"]
        return self._generate_single_format(pair, "R-RISK-FACTOR-NEG", template, answer_format)

    def generate_negative_mr_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.YES_NO
    ) -> Optional[KGGeneratedItem]:
        """
        Generate C-MR-EVIDENCE-NEG question for pairs lacking MR support.

        Args:
            pair: GeneDiseasePair with weak MR evidence
            answer_format: Answer format (default: YES_NO)

        Returns:
            KGGeneratedItem or None if pair has MR support
        """
        # Only generate for pairs with no/weak MR evidence
        if pair.mr_score > 0.2:
            return None

        template = KG_QUESTION_TEMPLATES["C-MR-EVIDENCE-NEG"]
        return self._generate_single_format(pair, "C-MR-EVIDENCE-NEG", template, answer_format)

    def generate_negative_pathway_question(
        self,
        pair: GeneDiseasePair,
        answer_format: AnswerFormat = AnswerFormat.YES_NO
    ) -> Optional[KGGeneratedItem]:
        """
        Generate M-PATHWAY-NEG question for pairs with weak PPI/GO connections.

        Args:
            pair: GeneDiseasePair with weak pathway evidence
            answer_format: Answer format (default: YES_NO)

        Returns:
            KGGeneratedItem or None if pair has pathway support
        """
        # Only generate for pairs with weak pathway evidence
        if pair.go_functional_score > 0.2:
            return None

        template = KG_QUESTION_TEMPLATES["M-PATHWAY-NEG"]
        return self._generate_single_format(pair, "M-PATHWAY-NEG", template, answer_format)

    def get_weak_evidence_pairs(self, n: int = 100) -> List[GeneDiseasePair]:
        """Get pairs with weak evidence for negative examples."""
        weak_pairs = []
        for pair in self.kg.pairs:
            # Weak overall evidence
            if pair.risk_weight_score < 0.3:
                weak_pairs.append(pair)
        # Sort by weakest first
        weak_pairs.sort(key=lambda p: p.risk_weight_score)
        return weak_pairs[:n]

    def generate_negative_examples(self, n_per_type: int = 50) -> List[KGGeneratedItem]:
        """
        Generate negative examples from low-score gene-disease pairs.

        These are critical for testing LLM ability to say "No" when evidence is weak.

        Args:
            n_per_type: Number of negative examples per type (R, C, M)

        Returns:
            List of negative example items
        """
        items = []
        weak_pairs = self.get_weak_evidence_pairs(n=n_per_type * 3)

        r_count, c_count, m_count = 0, 0, 0

        for pair in weak_pairs:
            # Generate negative risk factor question
            if r_count < n_per_type:
                item = self.generate_negative_risk_factor_question(pair)
                if item:
                    items.append(item)
                    r_count += 1

            # Generate negative MR question
            if c_count < n_per_type and pair.mr_score <= 0.2:
                item = self.generate_negative_mr_question(pair)
                if item:
                    items.append(item)
                    c_count += 1

            # Generate negative pathway question
            if m_count < n_per_type and pair.go_functional_score <= 0.2:
                item = self.generate_negative_pathway_question(pair)
                if item:
                    items.append(item)
                    m_count += 1

            # Stop if we have enough
            if r_count >= n_per_type and c_count >= n_per_type and m_count >= n_per_type:
                break

        logger.info(f"Generated {len(items)} negative examples: R={r_count}, C={c_count}, M={m_count}")
        return items

    def generate_from_pair(
        self,
        pair: GeneDiseasePair,
        question_types: Optional[List[str]] = None
    ) -> List[KGGeneratedItem]:
        """
        Generate all question types for a single gene-disease pair.

        Args:
            pair: GeneDiseasePair to generate questions for
            question_types: List of template IDs to use (None = all)

        Returns:
            List of generated items
        """
        items = []

        generators = {
            "R-RISK-FACTOR": lambda p: self.generate_risk_factor_question(p),
            "R-RISK-LEVEL": lambda p: self.generate_risk_level_question(p),
            "C-MR-EVIDENCE": lambda p: self.generate_mr_evidence_question(p),
            "C-CAUSAL-STRENGTH": lambda p: self.generate_causal_strength_question(p),
            "M-PATHWAY": lambda p: self.generate_pathway_question(p),
            "S-SNP-COUNT": lambda p: self.generate_snp_count_question(p),
        }

        if question_types is None:
            question_types = list(generators.keys())

        for qtype in question_types:
            if qtype in generators:
                item = generators[qtype](pair)
                if item:
                    items.append(item)

        return items

    def generate_benchmark(
        self,
        n_per_taxonomy: int = 100,
        stratify_by_evidence: bool = True,
        include_comparisons: bool = True,
        include_negative_examples: bool = True,
        n_negative_per_type: int = 30,
        seed: int = 42
    ) -> List[KGGeneratedItem]:
        """
        Generate a balanced benchmark dataset with positive AND negative examples.

        Args:
            n_per_taxonomy: Target number of items per taxonomy (S, C, R, M)
            stratify_by_evidence: Stratify by evidence level
            include_comparisons: Include comparison questions
            include_negative_examples: Include negative examples (weak evidence pairs)
            n_negative_per_type: Number of negative examples per type (R, C, M)
            seed: Random seed

        Returns:
            List of generated items
        """
        np.random.seed(seed)
        items = []

        # Sample pairs stratified by evidence level
        if stratify_by_evidence:
            sampled_pairs = self.kg.sample_pairs(
                n=n_per_taxonomy * 2,  # Sample more for variety
                stratify_by="evidence_level",
                seed=seed
            )
        else:
            sampled_pairs = self.kg.sample_pairs(
                n=n_per_taxonomy * 2,
                stratify_by="random",
                seed=seed
            )

        # Track counts per taxonomy
        counts = {'S': 0, 'C': 0, 'R': 0, 'M': 0}

        # Generate questions from sampled pairs (POSITIVE examples)
        for pair in sampled_pairs:
            pair_items = self.generate_from_pair(pair)

            for item in pair_items:
                if counts[item.taxonomy] < n_per_taxonomy:
                    items.append(item)
                    counts[item.taxonomy] += 1

        # Add comparison questions (R-COMPARE)
        if include_comparisons:
            n_compare = min(50, n_per_taxonomy // 4)
            compare_count = 0

            for disease in self.kg.diseases[:50]:  # Top 50 diseases
                pairs = self.kg.get_genes_for_disease(disease)
                if len(pairs) >= 2:
                    # Compare top 2 genes
                    comp_item = self.generate_compare_question(pairs[0], pairs[1])
                    if comp_item:
                        items.append(comp_item)
                        compare_count += 1
                        if compare_count >= n_compare:
                            break

        # Add top-genes questions (R-TOP-GENES)
        for disease in self.kg.diseases[:30]:
            top_item = self.generate_top_genes_question(disease)
            if top_item:
                items.append(top_item)

        # Add gene-diseases questions (S-GENE-DISEASE)
        top_genes = list(self.kg._by_gene.keys())[:50]
        for gene in top_genes:
            gene_item = self.generate_gene_diseases_question(gene)
            if gene_item:
                items.append(gene_item)

        # Add NEGATIVE EXAMPLES (critical for proper LLM evaluation)
        if include_negative_examples:
            negative_items = self.generate_negative_examples(n_per_type=n_negative_per_type)
            items.extend(negative_items)
            logger.info(f"Added {len(negative_items)} negative examples to benchmark")

        logger.info(f"Generated {len(items)} total benchmark items")
        logger.info(f"Positive distribution: {counts}")

        return items

    def generate_benchmark_all_formats(
        self,
        n_per_taxonomy: int = 50,
        formats: List[AnswerFormat] = None,
        stratify_by_evidence: bool = True,
        include_negative_examples: bool = True,
        seed: int = 42
    ) -> List[KGGeneratedItem]:
        """
        Generate benchmark with ALL 5 answer formats for comprehensive evaluation.

        This method generates the same question in all 5 formats, allowing
        comparison of LLM performance across different answer types.

        Args:
            n_per_taxonomy: Target number of unique questions per taxonomy
            formats: List of formats to generate (default: all 5)
            stratify_by_evidence: Stratify by evidence level
            include_negative_examples: Include negative examples
            seed: Random seed

        Returns:
            List of generated items (5x n_per_taxonomy per taxonomy)

        Example output structure:
            - For each gene-disease pair, generates 5 items:
              * R-RISK-FACTOR-yes_no
              * R-RISK-FACTOR-mcq
              * R-RISK-FACTOR-short
              * R-RISK-FACTOR-long
              * R-RISK-FACTOR-reasoning
        """
        if formats is None:
            formats = self.ALL_FORMATS

        np.random.seed(seed)
        items = []

        # Sample pairs stratified by evidence level
        if stratify_by_evidence:
            sampled_pairs = self.kg.sample_pairs(
                n=n_per_taxonomy * 2,
                stratify_by="evidence_level",
                seed=seed
            )
        else:
            sampled_pairs = self.kg.sample_pairs(
                n=n_per_taxonomy * 2,
                stratify_by="random",
                seed=seed
            )

        # Template types per taxonomy (excluding S-GENE-DISEASE which needs special handling)
        template_types = {
            'S': ['S-SNP-COUNT'],  # S-GENE-DISEASE handled separately below
            'R': ['R-RISK-FACTOR', 'R-RISK-LEVEL'],
            'C': ['C-MR-EVIDENCE', 'C-CAUSAL-STRENGTH'],
            'M': ['M-PATHWAY', 'M-MECHANISM']
        }

        # Track counts per taxonomy
        counts = {'S': 0, 'C': 0, 'R': 0, 'M': 0}
        format_counts = {fmt.value: 0 for fmt in formats}

        # Generate questions from sampled pairs in ALL formats
        for pair in sampled_pairs:
            for taxonomy, templates in template_types.items():
                if counts[taxonomy] >= n_per_taxonomy:
                    continue

                for template_id in templates:
                    # Generate question in all requested formats
                    format_items = self.generate_question_all_formats(
                        pair=pair,
                        template_id=template_id,
                        formats=formats
                    )
                    for item in format_items:
                        if item:
                            items.append(item)
                            format_counts[item.answer_format] = format_counts.get(item.answer_format, 0) + 1

                counts[taxonomy] += 1

        # Generate S-GENE-DISEASE questions separately (requires gene-level lookup)
        unique_genes = list(set(p.gene_name for p in sampled_pairs))[:n_per_taxonomy]
        for gene in unique_genes:
            for fmt in formats:
                item = self.generate_gene_diseases_question(gene, answer_format=fmt)
                if item:
                    items.append(item)
                    format_counts[fmt.value] = format_counts.get(fmt.value, 0) + 1

        # Add negative examples in all formats
        if include_negative_examples:
            weak_pairs = self.get_weak_evidence_pairs(n=30)
            for pair in weak_pairs:
                # Negative risk factor
                for fmt in formats:
                    item = self.generate_negative_risk_factor_question(pair, fmt)
                    if item:
                        items.append(item)
                        format_counts[fmt.value] = format_counts.get(fmt.value, 0) + 1

                # Negative MR
                if pair.mr_score <= 0.2:
                    for fmt in formats:
                        item = self.generate_negative_mr_question(pair, fmt)
                        if item:
                            items.append(item)
                            format_counts[fmt.value] = format_counts.get(fmt.value, 0) + 1

                # Negative pathway
                if pair.go_functional_score <= 0.2:
                    for fmt in formats:
                        item = self.generate_negative_pathway_question(pair, fmt)
                        if item:
                            items.append(item)
                            format_counts[fmt.value] = format_counts.get(fmt.value, 0) + 1

        logger.info(f"Generated {len(items)} benchmark items with {len(formats)} formats")
        logger.info(f"Taxonomy distribution: {counts}")
        logger.info(f"Format distribution: {format_counts}")

        return items

    def generate_by_evidence_level(
        self,
        level: EvidenceLevel,
        n: int = 100
    ) -> List[KGGeneratedItem]:
        """
        Generate questions for specific evidence level.

        Args:
            level: Evidence level to filter by
            n: Number of items to generate

        Returns:
            List of generated items
        """
        pairs = self.kg.get_by_evidence_level(level)

        if not pairs:
            logger.warning(f"No pairs found for evidence level: {level}")
            return []

        # Sample pairs
        sample_n = min(n, len(pairs))
        indices = np.random.choice(len(pairs), sample_n, replace=False)
        sampled_pairs = [pairs[i] for i in indices]

        items = []
        for pair in sampled_pairs:
            pair_items = self.generate_from_pair(pair)
            items.extend(pair_items)

        return items[:n]

    def generate_mr_focused_benchmark(self, n: int = 200) -> List[KGGeneratedItem]:
        """
        Generate benchmark focused on MR-validated relationships.

        These have stronger causal evidence and are good for testing
        causal reasoning capabilities.

        Args:
            n: Number of items to generate

        Returns:
            List of generated items
        """
        mr_pairs = self.kg.get_mr_validated_pairs(min_score=0.3)

        if len(mr_pairs) < 10:
            logger.warning(f"Only {len(mr_pairs)} MR-validated pairs found")

        items = []
        for pair in mr_pairs[:n]:
            # Focus on causal questions for MR pairs
            pair_items = self.generate_from_pair(
                pair,
                question_types=["C-MR-EVIDENCE", "C-CAUSAL-STRENGTH", "R-RISK-FACTOR"]
            )
            items.extend(pair_items)

        return items[:n]

    def get_statistics(self, items: List[KGGeneratedItem]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about generated items.

        Returns statistics organized by:
        - taxonomy (S, R, C, M)
        - evidence_level (very_strong, strong, moderate, suggestive, weak)
        - difficulty (easy, medium, hard)
        - template (R-RISK-FACTOR, C-MR-EVIDENCE, etc.)
        - answer_format (yes_no, mcq, short, long, reasoning)
        """
        stats = {
            'total': len(items),
            'by_taxonomy': {},
            'by_evidence_level': {},
            'by_difficulty': {},
            'by_template': {},
            'by_answer_format': {},
            'by_taxonomy_and_format': {}  # Cross-tabulation
        }

        for item in items:
            # By taxonomy
            tax = item.taxonomy
            stats['by_taxonomy'][tax] = stats['by_taxonomy'].get(tax, 0) + 1

            # By evidence level
            level = item.evidence_level
            stats['by_evidence_level'][level] = stats['by_evidence_level'].get(level, 0) + 1

            # By difficulty
            diff = item.difficulty
            stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1

            # By template
            tmpl = item.template_id
            stats['by_template'][tmpl] = stats['by_template'].get(tmpl, 0) + 1

            # By answer format
            fmt = item.answer_format
            stats['by_answer_format'][fmt] = stats['by_answer_format'].get(fmt, 0) + 1

            # Cross-tabulation: taxonomy x format
            tax_fmt_key = f"{tax}-{fmt}"
            stats['by_taxonomy_and_format'][tax_fmt_key] = stats['by_taxonomy_and_format'].get(tax_fmt_key, 0) + 1

        return stats

    def to_dataframe(self, items: List[KGGeneratedItem]) -> pd.DataFrame:
        """Convert items to DataFrame."""
        return pd.DataFrame([item.to_dict() for item in items])

    def export_to_json(
        self,
        items: List[KGGeneratedItem],
        output_path: str
    ) -> None:
        """Export items to JSON file."""
        import json

        data = [item.to_dict() for item in items]

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(items)} items to {output_path}")
