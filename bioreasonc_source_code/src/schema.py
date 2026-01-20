"""
Schema definitions for BioREASONC-Bench

Defines data structures for:
- Benchmark items
- Reasoning taxonomy
- Validation results
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Union, Literal
from enum import Enum
import json


class ReasoningCategory(str, Enum):
    """High-level reasoning categories"""
    STRUCTURE = "S"      # Structure-Aware
    CAUSAL = "C"         # Causal-Aware
    RISK = "R"           # Risk-Aware
    SEMANTIC = "M"       # Semantic-Aware


class StructureLabel(str, Enum):
    """Structure-Aware reasoning labels"""
    GENE_MAP = "S-GENE-MAP"
    PATHWAY_TRACE = "S-PATHWAY-TRACE"
    NETWORK_TRAVERSE = "S-NETWORK-TRAVERSE"
    SHORTEST_PATH = "S-SHORTEST-PATH"
    COMPONENT_DETECT = "S-COMPONENT-DETECT"


class CausalLabel(str, Enum):
    """Causal-Aware reasoning labels"""
    # Original labels
    CAUSAL_VS_ASSOC = "C-CAUSAL-VS-ASSOC"
    DIRECT_CAUSAL = "C-DIRECT-CAUSAL"
    MR_INFERENCE = "C-MR-INFERENCE"
    PC_DISCOVERY = "C-PC-DISCOVERY"
    CONFOUNDER_DETECT = "C-CONFOUNDER-DETECT"
    # New labels for DoWhy integration
    REFUTATION = "C-REFUTATION"
    SENSITIVITY = "C-SENSITIVITY"
    ESTIMATOR_COMPARE = "C-ESTIMATOR-COMPARE"
    IDENTIFIABILITY = "C-IDENTIFIABILITY"
    DISCOVERY_COMPARE = "C-DISCOVERY-COMPARE"


class RiskLabel(str, Enum):
    """Risk-Aware reasoning labels"""
    RISK_LEVEL = "R-RISK-LEVEL"
    RISK_COMPARE = "R-RISK-COMPARE"
    RISK_RANK = "R-RISK-RANK"
    RISK_AGGREGATE = "R-RISK-AGGREGATE"
    BETA_INTERPRET = "R-BETA-INTERPRET"


class SemanticLabel(str, Enum):
    """Semantic-Aware reasoning labels"""
    REL_EXTRACT = "M-REL-EXTRACT"
    ENTITY_RECOGNIZE = "M-ENTITY-RECOGNIZE"
    TEXT_INFERENCE = "M-TEXT-INFERENCE"
    SEMANTIC_SIM = "M-SEMANTIC-SIM"


@dataclass
class GeneticVariant:
    """Represents a genetic variant (SNP)"""
    rsid: str
    chromosome: Optional[str] = None
    position: Optional[int] = None
    ref_allele: Optional[str] = None
    alt_allele: Optional[str] = None
    maf: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RiskGene:
    """Represents a risk gene with associated statistics"""
    symbol: str
    ensembl_id: Optional[str] = None
    gene_type: Optional[str] = None
    chromosome: Optional[str] = None
    position: Optional[float] = None
    odds_ratio: Optional[float] = None
    beta: Optional[float] = None
    p_value: Optional[float] = None
    maf: Optional[float] = None
    associated_variants: List[str] = field(default_factory=list)
    associated_diseases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CausalRelation:
    """Represents a causal relationship"""
    source: str
    target: str
    relation_type: str  # 'causal', 'association', 'confounded'
    method: str  # 'pc', 'mr', 'observational'
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class NetworkEdge:
    """Represents an edge in a biological network"""
    source: str
    target: str
    weight: float = 1.0
    edge_type: str = "interaction"
    source_db: str = "unknown"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkItem:
    """A single benchmark item"""
    id: str
    taxonomy: str  # S, C, R, M
    label: str     # Fine-grained label
    template_id: str
    question: str
    answer: str
    explanation: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    # Source data
    source_genes: List[str] = field(default_factory=list)
    source_variants: List[str] = field(default_factory=list)
    source_data: Dict = field(default_factory=dict)

    # Validation scores
    validation_scores: List[float] = field(default_factory=list)
    avg_score: Optional[float] = None
    is_valid: bool = False

    # Metadata
    paraphrased: bool = False
    original_question: Optional[str] = None
    algorithm_used: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkItem':
        return cls(**data)


@dataclass
class ValidationResult:
    """Result from LLM validation"""
    item_id: str
    judge_model: str
    score: float
    feedback: Optional[str] = None
    is_factual: bool = True
    is_answerable: bool = True
    reasoning_valid: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkDataset:
    """Complete benchmark dataset"""
    version: str
    items: List[BenchmarkItem]
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def get_by_taxonomy(self, taxonomy: str) -> List[BenchmarkItem]:
        return [item for item in self.items if item.taxonomy == taxonomy]

    def get_statistics(self) -> Dict:
        stats = {
            "total_items": len(self.items),
            "by_taxonomy": {},
            "by_label": {},
            "avg_validation_score": 0.0,
            "valid_items": 0
        }

        for item in self.items:
            # By taxonomy
            if item.taxonomy not in stats["by_taxonomy"]:
                stats["by_taxonomy"][item.taxonomy] = 0
            stats["by_taxonomy"][item.taxonomy] += 1

            # By label
            if item.label not in stats["by_label"]:
                stats["by_label"][item.label] = 0
            stats["by_label"][item.label] += 1

            # Validation
            if item.is_valid:
                stats["valid_items"] += 1
            if item.avg_score:
                stats["avg_validation_score"] += item.avg_score

        if self.items:
            stats["avg_validation_score"] /= len(self.items)

        return stats

    def to_jsonl(self, filepath: str):
        with open(filepath, 'w') as f:
            for item in self.items:
                f.write(item.to_jsonl() + '\n')

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "metadata": self.metadata,
            "statistics": self.get_statistics(),
            "items": [item.to_dict() for item in self.items]
        }


# Question templates
QUESTION_TEMPLATES = {
    # Structure-Aware templates
    "S-GENE-MAP": [
        "Which gene is associated with the variant {rsid}?",
        "What gene does the SNP {rsid} map to?",
        "Identify the gene containing variant {rsid}."
    ],
    "S-PATHWAY-TRACE": [
        "What is the shortest path from {gene1} to {gene2} in the protein interaction network?",
        "Trace the pathway connecting {gene1} to {disease} through intermediate genes.",
        "How many steps separate {gene1} from {gene2} in the gene network?"
    ],
    "S-NETWORK-TRAVERSE": [
        "List all genes within 2 hops of {gene} in the interaction network.",
        "Using BFS, find all genes connected to {gene} within distance {n}.",
        "What genes are reachable from {gene} using DFS?"
    ],

    # Causal-Aware templates
    "C-CAUSAL-VS-ASSOC": [
        "Is the relationship between {gene} and {disease} causal or merely associative?",
        "Does {variant} cause {disease}, or is it only correlated?",
        "Distinguish between causal and associative evidence for {gene} in {disease}."
    ],
    "C-MR-INFERENCE": [
        "Based on Mendelian Randomization, what is the causal effect of {exposure} on {outcome}?",
        "Using {variant} as an instrument, estimate the causal effect of {gene} on {disease}.",
        "What does MR analysis suggest about the {gene}-{disease} relationship?"
    ],
    "C-PC-DISCOVERY": [
        "Using PC algorithm, what is the causal structure between {genes}?",
        "Identify the causal direction between {gene1} and {gene2}.",
        "What confounders exist between {gene} and {disease}?"
    ],

    # Risk-Aware templates
    "R-RISK-LEVEL": [
        "What is the risk level of {gene} for {disease} based on OR={or_value}?",
        "Classify the risk associated with {variant} (OR={or_value}, p={p_value}).",
        "Is {gene} a high, moderate, or low risk factor for {disease}?"
    ],
    "R-RISK-COMPARE": [
        "Compare the risk contribution of {gene1} (OR={or1}) vs {gene2} (OR={or2}) for {disease}.",
        "Which variant confers higher risk: {var1} or {var2}?",
        "Rank {gene1}, {gene2}, and {gene3} by their COVID-19 risk contribution."
    ],
    "R-RISK-AGGREGATE": [
        "Calculate the cumulative risk score for genes {genes} in {disease}.",
        "What is the aggregate genetic risk from variants {variants}?",
        "Compute the weighted risk score using OR values for {genes}."
    ],

    # Semantic-Aware templates
    "M-REL-EXTRACT": [
        "Extract the relationship between {entity1} and {entity2} from: '{text}'",
        "What biomedical relation connects {gene} and {disease} in the literature?",
        "Identify gene-disease associations mentioned in: '{text}'"
    ],
    "M-ENTITY-RECOGNIZE": [
        "Identify all gene names in: '{text}'",
        "Extract disease mentions from: '{text}'",
        "Find all genetic variants referenced in: '{text}'"
    ],
    "M-SEMANTIC-SIM": [
        "How semantically similar are {gene1} and {gene2} based on their descriptions?",
        "Compute the similarity between {disease1} and {disease2} phenotypes.",
        "Are {term1} and {term2} referring to the same biological concept?"
    ]
}
