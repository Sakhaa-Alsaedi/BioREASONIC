"""
BioREASONC-Creator: Question Generator Module

================================================================================
MODULE OVERVIEW
================================================================================

Generates expert-level biomedical questions from structured SNP→Gene→Disease data.
Each question has deterministic ground truth answers for LLM evaluation.

This is STEP 1 in the BioREASONC pipeline:
    GENERATOR → Validator → Explainer → Paraphraser → Exporter

Focus: "Does the model tell the truth about causality when explaining biomedical research?"

================================================================================
INPUT SPECIFICATION
================================================================================

The Generator accepts data in multiple formats:

1. PANDAS DATAFRAME (generate_from_dataframe):
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ Required Columns (flexible naming - see column_mapping):                    │
   │                                                                             │
   │ rsid/SNP/variant    : SNP identifier (e.g., "rs1234567")                   │
   │ gene/Gene/symbol    : Gene symbol (e.g., "BRCA1", "GCK")                   │
   │ chromosome/chr      : Chromosome number (e.g., "7", "17")                  │
   │ OR/or_value         : Odds ratio (e.g., 1.45, 2.3)                         │
   │ P-Value/p_value     : P-value for significance (e.g., 5e-10)               │
   │                                                                             │
   │ Additional context parameter:                                               │
   │ disease             : Disease name (passed separately)                      │
   └─────────────────────────────────────────────────────────────────────────────┘

2. CSV FILE (generate_from_csv):
   Same column requirements as DataFrame. Disease name passed as parameter.

3. MULTIPLE FILES (generate_from_multiple_files):
   List of {"path": "file.csv", "disease": "Disease Name"} configurations.

Example Input Data:
```python
df = pd.DataFrame({
    'rsid': ['rs1799884', 'rs4607517', 'rs7903146'],
    'gene': ['GCK', 'GCK', 'TCF7L2'],
    'chromosome': ['7', '7', '10'],
    'OR': [1.15, 1.12, 1.35],
    'P-Value': [5e-12, 3e-8, 2e-50]
})
disease = "Type 2 Diabetes"
```

================================================================================
OUTPUT SPECIFICATION
================================================================================

The Generator produces a list of GeneratedItem objects:

GeneratedItem Fields:
┌────────────────────┬──────────────────┬────────────────────────────────────────┐
│ Field              │ Type             │ Description                            │
├────────────────────┼──────────────────┼────────────────────────────────────────┤
│ id                 │ str              │ Unique ID (e.g., "C-0001", "R-0042")  │
│ taxonomy           │ str              │ S, C, R, or M                          │
│ label              │ str              │ Template label (e.g., "C-CAUSAL-VS-ASSOC") │
│ template_id        │ str              │ Specific template (e.g., "C-CAUSAL-VS-ASSOC-01") │
│ question           │ str              │ Generated question text                │
│ answer             │ str              │ Generated answer with evidence         │
│ answer_type        │ str              │ single_entity/yes_no/numeric/explanation │
│ entities           │ Dict             │ Extracted entities {gene, snp, disease, ...} │
│ ground_truth       │ Dict             │ Validation data (answer, confidence)  │
│ difficulty         │ str              │ easy/medium/hard                       │
│ source_data        │ Dict             │ Original row data for traceability    │
└────────────────────┴──────────────────┴────────────────────────────────────────┘

Example Output:
```python
GeneratedItem(
    id="C-0001",
    taxonomy="C",
    label="C-CAUSAL-VS-ASSOC",
    template_id="C-CAUSAL-VS-ASSOC-01",
    question="Is the relationship between GCK and Type 2 Diabetes causal or associative based on GWAS evidence?",
    answer="The relationship is ASSOCIATIVE, not causal. [CoT reasoning...]",
    answer_type="explanation",
    entities={"gene": "GCK", "disease": "Type 2 Diabetes", "snp": "rs1799884"},
    ground_truth={"answer": "...", "confidence": 1.0},
    difficulty="hard",
    source_data={...}
)
```

================================================================================
TAXONOMY LABELS
================================================================================

┌──────────┬─────────────────────────────────────────────────────────────────────┐
│ Taxonomy │ Description                                                         │
├──────────┼─────────────────────────────────────────────────────────────────────┤
│ S        │ Structure-Aware: SNP-Gene mapping, chromosomal location, genomics   │
│          │ Templates: S-GENE-MAP, S-SNP-GENE, S-CHROM-LOC                      │
│          │ Answer types: single_entity, yes_no                                 │
│          │ Difficulty: easy                                                    │
├──────────┼─────────────────────────────────────────────────────────────────────┤
│ C        │ Causal-Aware: Causal vs associative distinction (CRITICAL)          │
│          │ Templates: C-CAUSAL-VS-ASSOC, C-MR-EVIDENCE                         │
│          │ Answer types: explanation (with CoT reasoning)                      │
│          │ Difficulty: hard                                                    │
│          │ ⚠️  These questions test the CORE METRIC: causal faithfulness       │
├──────────┼─────────────────────────────────────────────────────────────────────┤
│ R        │ Risk-Aware: Odds ratio interpretation, risk classification          │
│          │ Templates: R-RISK-LEVEL, R-OR-INTERPRET, R-PVALUE-SIG              │
│          │ Answer types: explanation, numeric, yes_no                          │
│          │ Difficulty: medium/easy                                             │
├──────────┼─────────────────────────────────────────────────────────────────────┤
│ M        │ Semantic-Aware: Entity recognition, relation extraction             │
│          │ Templates: M-ENTITY-RECOGNIZE, M-REL-EXTRACT                        │
│          │ Answer types: single_entity, multiple_entity                        │
│          │ Difficulty: easy/medium                                             │
└──────────┴─────────────────────────────────────────────────────────────────────┘

================================================================================
PROCESSING FLOW
================================================================================

1. INPUT PREPARATION:
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ • Standardize column names (flexible mapping for different data sources)    │
   │ • Parse OR values and p-values                                              │
   │ • Compute derived fields (risk_level, risk_interpretation)                  │
   │ • Classify statistical significance (genome-wide threshold: p < 5e-8)       │
   └─────────────────────────────────────────────────────────────────────────────┘

2. TEMPLATE MATCHING:
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ For each data row:                                                          │
   │   • Check which templates can be generated (required fields present)        │
   │   • Generate question by filling placeholders                               │
   │   • Generate answer (CoT for Causal, direct for others)                     │
   │   • Create ground truth with evidence                                       │
   │   • Assign unique ID by taxonomy                                            │
   └─────────────────────────────────────────────────────────────────────────────┘

3. OUTPUT GENERATION:
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │ • Create GeneratedItem with all fields                                      │
   │ • Track counts per template (max_per_template limit)                        │
   │ • Return list of generated items                                            │
   └─────────────────────────────────────────────────────────────────────────────┘

================================================================================
CONFIGURATION OPTIONS
================================================================================

USE_COT_ANSWERS (Line 58):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ True  (default): Use Chain-of-Thought answers for Causal taxonomy              │
│                  Answers include step-by-step reasoning                        │
│                  Better for evaluating model reasoning capabilities            │
│                                                                                 │
│ False          : Use short answers without reasoning                           │
│                  Faster evaluation but less insight into model thinking        │
└─────────────────────────────────────────────────────────────────────────────────┘

max_per_template:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Controls maximum questions per template (default: 50)                          │
│ Prevents single template from dominating the benchmark                         │
│ Adjust based on desired benchmark size                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
BEST PRACTICES
================================================================================

★ DATA QUALITY:
  • Ensure OR values are numeric (not strings like "1.2 (1.1-1.3)")
  • Use standard gene symbols (HGNC)
  • Include p-values for significance questions
  • Provide accurate disease names

★ TEMPLATE BALANCE:
  • Use multiple templates to cover all taxonomies
  • Balance easy/medium/hard questions
  • Ensure sufficient Causal (C) questions for core metric

★ EVIDENCE TRACEABILITY:
  • All answers are deterministic from input data
  • Ground truth includes source evidence
  • Confidence = 1.0 means answer is certain from data

★ CAUSAL QUESTIONS (C taxonomy):
  • These are the MOST IMPORTANT for benchmark validity
  • CoT answers teach proper causal reasoning
  • Answers distinguish GWAS association from causation
  • Mention MR, functional studies as needed for causal claims

================================================================================
USAGE EXAMPLES
================================================================================

Example 1: Generate from DataFrame
```python
from bioreasonc_creator.generator import QuestionGenerator
import pandas as pd

generator = QuestionGenerator()
df = pd.read_csv("gwas_data.csv")
items = generator.generate_from_dataframe(df, disease="Type 2 Diabetes")
print(f"Generated {len(items)} items")
```

Example 2: Generate from multiple files
```python
configs = [
    {"path": "t2d_gwas.csv", "disease": "Type 2 Diabetes"},
    {"path": "breast_cancer_gwas.csv", "disease": "Breast Cancer"},
]
items = generator.generate_from_multiple_files(configs)
```

Example 3: Get statistics
```python
stats = generator.get_statistics(items)
print(f"By taxonomy: {stats['by_taxonomy']}")
print(f"By difficulty: {stats['by_difficulty']}")
```

================================================================================
Author: Sakhaa Alsaedi, sakhaa.alsaedi@kaust.edu.sa
================================================================================
"""

import json
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import pandas as pd
import logging

# Import centralized prompts
from .prompts import (
    GeneratorPrompts,
    CoTPrompts,
    Taxonomy as PromptTaxonomy,
    TAXONOMY_DESCRIPTIONS,
    CORE_QUESTION,
    ExpertGeneratorPrompts,
    ExpertAnswerHelper
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Taxonomy(str, Enum):
    """Question taxonomy categories."""
    S = "S"  # Structure-Aware
    C = "C"  # Causal-Aware
    R = "R"  # Risk-Aware
    M = "M"  # seMantic-Aware


class AnswerType(str, Enum):
    """Types of expected answers."""
    SINGLE_ENTITY = "single_entity"
    MULTIPLE_ENTITY = "multiple_entity"
    YES_NO = "yes_no"
    NUMERIC = "numeric"
    EXPLANATION = "explanation"
    RANKING = "ranking"


# Configuration: Use CoT (Chain-of-Thought) answers for Causal taxonomy
USE_COT_ANSWERS = True  # Set to True for detailed reasoning, False for short answers

# Configuration: Use Expert-style prompts with biological explanations
USE_EXPERT_PROMPTS = True  # Set to True for natural expert explanations, False for mechanical templates


@dataclass
class GroundTruth:
    """Ground truth data for validation."""
    answer: str
    answer_normalized: str  # Lowercase, stripped
    answer_type: AnswerType
    evidence: Dict[str, Any]  # Source data proving the answer
    confidence: float = 1.0  # 1.0 = deterministic from data


@dataclass
class GeneratedItem:
    """A generated benchmark item with ground truth."""
    id: str
    taxonomy: str
    label: str
    template_id: str
    question: str
    answer: str
    answer_type: str
    entities: Dict[str, Any]
    ground_truth: Dict[str, Any]
    difficulty: str = "medium"
    source_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class QuestionTemplate:
    """A question template with placeholders."""

    def __init__(
        self,
        template_id: str,
        taxonomy: Taxonomy,
        label: str,
        question_template: str,
        answer_template: str,
        answer_type: AnswerType,
        required_fields: List[str],
        difficulty: str = "medium"
    ):
        self.template_id = template_id
        self.taxonomy = taxonomy
        self.label = label
        self.question_template = question_template
        self.answer_template = answer_template
        self.answer_type = answer_type
        self.required_fields = required_fields
        self.difficulty = difficulty

    def can_generate(self, data: Dict[str, Any]) -> bool:
        """Check if all required fields are present."""
        return all(
            field in data and data[field] is not None and str(data[field]).strip()
            for field in self.required_fields
        )

    def generate(self, data: Dict[str, Any], item_id: str) -> Optional[GeneratedItem]:
        """Generate a question from data."""
        if not self.can_generate(data):
            return None

        # Fill in templates
        question = self.question_template
        answer = self.answer_template

        for field, value in data.items():
            placeholder = f"{{{field}}}"
            if placeholder in question:
                question = question.replace(placeholder, str(value))
            if placeholder in answer:
                answer = answer.replace(placeholder, str(value))

        # Create ground truth
        ground_truth = GroundTruth(
            answer=answer,
            answer_normalized=answer.lower().strip(),
            answer_type=self.answer_type,
            evidence={k: v for k, v in data.items() if k in self.required_fields},
            confidence=1.0
        )

        # Extract entities
        entities = {}
        if 'rsid' in data:
            entities['snp'] = data['rsid']
        if 'gene' in data:
            entities['gene'] = data['gene']
        if 'disease' in data:
            entities['disease'] = data['disease']
        if 'or_value' in data:
            entities['odds_ratio'] = data['or_value']
        if 'p_value' in data:
            entities['p_value'] = data['p_value']

        return GeneratedItem(
            id=item_id,
            taxonomy=self.taxonomy.value,
            label=self.label,
            template_id=self.template_id,
            question=question,
            answer=answer,
            answer_type=self.answer_type.value,
            entities=entities,
            ground_truth=asdict(ground_truth),
            difficulty=self.difficulty,
            source_data=data
        )


class QuestionGenerator:
    """
    Generates biomedical questions from structured data.

    Each question has:
    - Deterministic ground truth answer
    - Taxonomy label (S/C/R/M)
    - Entity annotations
    - Source data evidence
    """

    def __init__(self):
        self.templates = self._create_templates()
        self.item_counter = {t.value: 0 for t in Taxonomy}

    def _create_templates(self) -> List[QuestionTemplate]:
        """Create all question templates."""
        templates = []

        # ============== S (Structure) Templates ==============
        templates.extend([
            QuestionTemplate(
                template_id="S-GENE-MAP-01",
                taxonomy=Taxonomy.S,
                label="S-GENE-MAP",
                question_template="Which gene is the variant {rsid} located in or associated with?",
                answer_template="{gene}",
                answer_type=AnswerType.SINGLE_ENTITY,
                required_fields=["rsid", "gene"],
                difficulty="easy"
            ),
            QuestionTemplate(
                template_id="S-GENE-MAP-02",
                taxonomy=Taxonomy.S,
                label="S-GENE-MAP",
                question_template="What is the gene symbol for the genomic region containing SNP {rsid}?",
                answer_template="{gene}",
                answer_type=AnswerType.SINGLE_ENTITY,
                required_fields=["rsid", "gene"],
                difficulty="easy"
            ),
            QuestionTemplate(
                template_id="S-SNP-GENE-01",
                taxonomy=Taxonomy.S,
                label="S-SNP-GENE",
                question_template="Is the variant {rsid} located within or near the {gene} gene?",
                answer_template="Yes, {rsid} is associated with {gene}.",
                answer_type=AnswerType.YES_NO,
                required_fields=["rsid", "gene"],
                difficulty="easy"
            ),
            QuestionTemplate(
                template_id="S-CHROM-LOC-01",
                taxonomy=Taxonomy.S,
                label="S-CHROM-LOC",
                question_template="On which chromosome is the gene {gene} located?",
                answer_template="Chromosome {chromosome}",
                answer_type=AnswerType.SINGLE_ENTITY,
                required_fields=["gene", "chromosome"],
                difficulty="easy"
            ),
        ])

        # ============== C (Causal) Templates ==============
        # CRITICAL: These templates test causal faithfulness
        # Use CoT (Chain-of-Thought) answers for detailed reasoning

        # Select answer template based on USE_COT_ANSWERS config
        c_causal_assoc_answer = (
            GeneratorPrompts.C_CAUSAL_VS_ASSOC_ANSWER_COT if USE_COT_ANSWERS
            else GeneratorPrompts.C_CAUSAL_VS_ASSOC_ANSWER_SHORT
        )
        c_cannot_conclude_answer = (
            GeneratorPrompts.C_CANNOT_CONCLUDE_CAUSATION_COT if USE_COT_ANSWERS
            else GeneratorPrompts.C_CANNOT_CONCLUDE_CAUSATION_SHORT
        )
        c_mr_evidence_answer = (
            GeneratorPrompts.C_MR_EVIDENCE_ANSWER_COT if USE_COT_ANSWERS
            else GeneratorPrompts.C_MR_EVIDENCE_ANSWER_SHORT
        )

        templates.extend([
            QuestionTemplate(
                template_id="C-CAUSAL-VS-ASSOC-01",
                taxonomy=Taxonomy.C,
                label="C-CAUSAL-VS-ASSOC",
                question_template="Is the relationship between {gene} and {disease} causal or associative based on GWAS evidence?",
                answer_template=c_causal_assoc_answer,
                answer_type=AnswerType.EXPLANATION,
                required_fields=["gene", "disease", "or_value", "p_value"],
                difficulty="hard"
            ),
            QuestionTemplate(
                template_id="C-CAUSAL-VS-ASSOC-02",
                taxonomy=Taxonomy.C,
                label="C-CAUSAL-VS-ASSOC",
                question_template="Based on the genetic association data, can we conclude that {gene} causes {disease}?",
                answer_template=c_cannot_conclude_answer,
                answer_type=AnswerType.EXPLANATION,
                required_fields=["gene", "disease", "or_value"],
                difficulty="hard"
            ),
            QuestionTemplate(
                template_id="C-MR-EVIDENCE-01",
                taxonomy=Taxonomy.C,
                label="C-MR-EVIDENCE",
                question_template="What type of evidence would strengthen the causal claim between {gene} variation and {disease}?",
                answer_template=c_mr_evidence_answer,
                answer_type=AnswerType.EXPLANATION,
                required_fields=["gene", "disease"],
                difficulty="hard"
            ),
        ])

        # ============== R (Risk) Templates ==============
        templates.extend([
            QuestionTemplate(
                template_id="R-RISK-LEVEL-01",
                taxonomy=Taxonomy.R,
                label="R-RISK-LEVEL",
                question_template="What is the risk level conferred by {rsid} for {disease} given OR={or_value}?",
                answer_template="{risk_level} risk. OR={or_value} indicates {risk_interpretation}.",
                answer_type=AnswerType.EXPLANATION,
                required_fields=["rsid", "disease", "or_value", "risk_level", "risk_interpretation"],
                difficulty="medium"
            ),
            QuestionTemplate(
                template_id="R-RISK-LEVEL-02",
                taxonomy=Taxonomy.R,
                label="R-RISK-LEVEL",
                question_template="How would you classify the genetic risk from variant {rsid} with odds ratio {or_value}?",
                answer_template="{risk_level}. An OR of {or_value} represents {risk_interpretation}.",
                answer_type=AnswerType.EXPLANATION,
                required_fields=["rsid", "or_value", "risk_level", "risk_interpretation"],
                difficulty="medium"
            ),
            QuestionTemplate(
                template_id="R-OR-INTERPRET-01",
                taxonomy=Taxonomy.R,
                label="R-OR-INTERPRET",
                question_template="If a patient carries the risk allele at {rsid} (OR={or_value}), how does their disease risk compare to non-carriers?",
                answer_template="Carriers have {or_value}x the odds of developing the disease compared to non-carriers.",
                answer_type=AnswerType.NUMERIC,
                required_fields=["rsid", "or_value"],
                difficulty="medium"
            ),
            QuestionTemplate(
                template_id="R-PVALUE-SIG-01",
                taxonomy=Taxonomy.R,
                label="R-PVALUE-SIG",
                question_template="Is the association between {rsid} and {disease} statistically significant at genome-wide level (p < 5e-8)?",
                answer_template="{significance_answer}",
                answer_type=AnswerType.YES_NO,
                required_fields=["rsid", "disease", "p_value", "significance_answer"],
                difficulty="easy"
            ),
        ])

        # ============== M (Semantic) Templates ==============
        templates.extend([
            QuestionTemplate(
                template_id="M-ENTITY-RECOGNIZE-01",
                taxonomy=Taxonomy.M,
                label="M-ENTITY-RECOGNIZE",
                question_template="Identify the gene symbol mentioned in: '{gene} variant {rsid} is associated with {disease} susceptibility.'",
                answer_template="{gene}",
                answer_type=AnswerType.SINGLE_ENTITY,
                required_fields=["gene", "rsid", "disease"],
                difficulty="easy"
            ),
            QuestionTemplate(
                template_id="M-ENTITY-RECOGNIZE-02",
                taxonomy=Taxonomy.M,
                label="M-ENTITY-RECOGNIZE",
                question_template="Extract the SNP identifier from: 'The variant {rsid} in {gene} shows significant association with {disease}.'",
                answer_template="{rsid}",
                answer_type=AnswerType.SINGLE_ENTITY,
                required_fields=["rsid", "gene", "disease"],
                difficulty="easy"
            ),
            QuestionTemplate(
                template_id="M-REL-EXTRACT-01",
                taxonomy=Taxonomy.M,
                label="M-REL-EXTRACT",
                question_template="What is the relationship between {gene} and {disease} described in: '{gene} is genetically associated with {disease} risk.'?",
                answer_template="genetic_association({gene}, {disease})",
                answer_type=AnswerType.SINGLE_ENTITY,
                required_fields=["gene", "disease"],
                difficulty="medium"
            ),
            QuestionTemplate(
                template_id="M-REL-EXTRACT-02",
                taxonomy=Taxonomy.M,
                label="M-REL-EXTRACT",
                question_template="Extract the biomedical relation from: 'Variant {rsid} maps to {gene} and confers risk for {disease}.'",
                answer_template="SNP_gene_mapping({rsid}, {gene}); gene_disease_association({gene}, {disease})",
                answer_type=AnswerType.MULTIPLE_ENTITY,
                required_fields=["rsid", "gene", "disease"],
                difficulty="medium"
            ),
        ])

        return templates

    def _classify_risk(self, or_value: float) -> Tuple[str, str]:
        """Classify risk level based on odds ratio."""
        if or_value >= 2.0:
            return "HIGH", "a strong increase in disease risk"
        elif or_value >= 1.5:
            return "MODERATE-HIGH", "a moderately elevated disease risk"
        elif or_value >= 1.2:
            return "MODERATE", "a modest increase in disease risk"
        elif or_value >= 1.0:
            return "LOW", "a small increase in disease risk"
        elif or_value >= 0.8:
            return "PROTECTIVE-LOW", "a slight protective effect"
        else:
            return "PROTECTIVE", "a protective effect against the disease"

    def _check_significance(self, p_value: float) -> str:
        """Check genome-wide significance."""
        try:
            if float(p_value) < 5e-8:
                return "Yes, the association is genome-wide significant (p < 5e-8)."
            else:
                return "No, the association does not reach genome-wide significance (p >= 5e-8)."
        except:
            return "Unable to determine significance."

    def _generate_expert_answer(
        self,
        taxonomy: str,
        template_label: str,
        data: Dict[str, Any]
    ) -> str:
        """Generate expert-style answer with biological context."""
        gene = data.get('gene', 'Unknown')
        disease = data.get('disease', 'Unknown')
        or_value = self._parse_number(data.get('or_value'))
        snp_count = data.get('snp_count', 1)
        unique_snps = data.get('unique_snps', 1)
        mr_score = data.get('mr_score', 0.0)
        go_score = data.get('go_functional_score', 0.0)
        evidence_level = data.get('evidence_level', 'moderate')

        # Get biological context from helper
        gene_function = ExpertAnswerHelper.get_gene_function(gene)
        disease_mechanism = ExpertAnswerHelper.get_disease_mechanism(disease)
        or_context = ExpertAnswerHelper.get_or_context(or_value) if or_value else ""

        # Generate answer based on taxonomy and template
        if taxonomy == "S":
            return self._generate_s_expert_answer(
                gene, disease, snp_count, unique_snps, gene_function, evidence_level
            )
        elif taxonomy == "C":
            return self._generate_c_expert_answer(
                template_label, gene, disease, or_value, mr_score,
                gene_function, disease_mechanism
            )
        elif taxonomy == "R":
            result = self._generate_r_expert_answer(
                gene, disease, or_value, snp_count, evidence_level,
                gene_function, disease_mechanism, or_context,
                template_label
            )
            # None signals to use template answer (for P-value questions)
            return result
        elif taxonomy == "M":
            return self._generate_m_expert_answer(
                gene, disease, go_score, gene_function, disease_mechanism
            )
        else:
            return None  # Signal to use template answer

    def _generate_s_expert_answer(
        self, gene: str, disease: str, snp_count: int, unique_snps: int,
        gene_function: str, evidence_level: str
    ) -> str:
        """Generate Structure-Aware expert answer.

        Note: S taxonomy questions are typically simple structure queries
        (gene mapping, chromosome location). The template generates the
        basic answer; we add minimal biological context.
        """
        # For simple structure questions, keep answers concise
        # The biological context is optional enhancement
        if gene_function and gene_function != "a protein with various cellular functions":
            return f"""{gene}. This gene encodes {gene_function}, making its association with {disease} biologically plausible."""
        else:
            return f"""{gene}"""

    def _generate_c_expert_answer(
        self, template_label: str, gene: str, disease: str,
        or_value: Optional[float], mr_score: float,
        gene_function: str, disease_mechanism: str
    ) -> str:
        """Generate Causal-Aware expert answer."""
        or_str = f"{or_value:.2f}" if or_value else "N/A"

        if "CAUSAL-VS-ASSOC" in template_label or "CANNOT-CONCLUDE" in template_label:
            mr_interpretation = ""
            if mr_score and mr_score > 0.7:
                mr_interpretation = f"\n\n{gene} has MR support (score: {mr_score:.2f}), which strengthens the causal argument. However, MR evidence should be combined with functional studies for definitive causal claims."
            elif mr_score and mr_score > 0:
                mr_interpretation = f"\n\nMR evidence for {gene} is limited (score: {mr_score:.2f}), so caution is warranted in making causal claims."

            return f"""No—and this distinction is crucial for interpreting genetic studies correctly.

GWAS tells us that people carrying {gene} variants have higher {disease} rates. The association is robust: OR of {or_str}, replicated across studies. But association is not causation.

Consider the alternatives:

**Confounding:** Perhaps {gene} variants are more common in populations with lifestyle factors that independently increase {disease} risk. The association would be real but not causal.

**Linkage disequilibrium:** The associated variant might simply tag the true causal variant nearby. {gene} could span a large genomic region; the causal variant could be in a regulatory element affecting a different gene entirely.

**Reverse causation:** Less likely for germline variants, but disease-related metabolic changes could theoretically affect {gene} regulation.

{gene} encodes {gene_function}. The biological pathway to {disease} {disease_mechanism}.{mr_interpretation}

The correct statement is: "{gene} variants are associated with increased {disease} risk." Claiming causation requires MR, functional studies, or ideally both."""

        elif "MR-EVIDENCE" in template_label:
            if mr_score and mr_score > 0.7:
                mr_interpretation = "strong causal support"
                conclusion = f"{gene}'s causal contribution to {disease} is well-supported by MR evidence."
            elif mr_score and mr_score > 0.3:
                mr_interpretation = "moderate causal support, though additional evidence would strengthen the case"
                conclusion = f"The MR evidence suggests {gene} may causally contribute to {disease}, but functional validation is recommended."
            else:
                mr_interpretation = "limited causal support—additional evidence types are needed"
                conclusion = f"MR alone does not strongly support causation; functional studies and therapeutic validation would be valuable."

            return f"""The Mendelian Randomization evidence provides insight into whether {gene} causally affects {disease}.

The logic of MR is elegant: genetic variants are assigned at conception, before disease develops, so they can't be affected by reverse causation. If {gene} variants that affect {gene_function} also alter {disease} risk proportionally, it suggests the {gene} pathway itself (not some confounder) influences disease.

The MR score of {mr_score:.2f} indicates {mr_interpretation}.

The biology explains why this relationship might be causal: {gene} encodes {gene_function}. The mechanistic pathway to {disease} {disease_mechanism}.

{conclusion}"""

        return f"Causal analysis for {gene}-{disease} association."

    def _generate_r_expert_answer(
        self, gene: str, disease: str, or_value: Optional[float],
        snp_count: int, evidence_level: str, gene_function: str,
        disease_mechanism: str, or_context: str,
        template_label: str = "R-RISK-LEVEL"
    ) -> str:
        """Generate Risk-Aware expert answer based on template type."""
        or_str = f"{or_value:.2f}" if or_value else "N/A"

        # Handle different R template types
        if "OR-INTERPRET" in template_label:
            # Simple OR interpretation question
            return f"""Carriers have {or_str}x the odds of developing the disease compared to non-carriers.

An odds ratio of {or_str} means that for every person without the risk allele who develops {disease}, approximately {or_str} people with the risk allele will develop it. {or_context}

This is a relative measure. The absolute risk increase depends on the baseline population prevalence of {disease}. For common diseases, even modest ORs can translate to meaningful public health impact; for rare diseases, the individual effect may be small despite the OR."""

        elif "PVALUE-SIG" in template_label:
            # P-value significance question - this should use the template answer
            # since it requires the actual p-value for the yes/no determination
            return None  # Signal to use template answer instead

        # Default: R-RISK-LEVEL template - full expert explanation
        # Determine evidence strength description
        if evidence_level == "very_strong":
            evidence_strength = "compelling"
            risk_conclusion = f"should be considered a significant risk factor"
        elif evidence_level == "strong":
            evidence_strength = "substantial"
            risk_conclusion = f"is a well-supported risk factor"
        elif evidence_level == "moderate":
            evidence_strength = "moderate"
            risk_conclusion = f"may contribute to disease risk, though more evidence would strengthen this conclusion"
        else:
            evidence_strength = "suggestive but limited"
            risk_conclusion = f"shows some evidence of risk contribution, but caution is warranted"

        return f"""Yes, {gene} {risk_conclusion} for {disease}, with {evidence_level} evidence.

The statistical evidence is {evidence_strength}. {gene} variants show OR of {or_str}, supported by {snp_count} associated SNPs. {or_context}

Why does {gene} affect {disease} risk? {gene} encodes {gene_function}. The biological mechanism {disease_mechanism}.

For individual risk interpretation: an OR of {or_str} means carriers have {or_str}x the odds of developing {disease} compared to non-carriers. However, it's important to note that OR describes relative risk—the absolute risk increase depends on baseline population risk. For most people, lifestyle factors (diet, exercise, environmental exposures) may have larger absolute effects on their {disease} risk than any single genetic variant."""

    def _generate_m_expert_answer(
        self, gene: str, disease: str, go_score: float,
        gene_function: str, disease_mechanism: str
    ) -> str:
        """Generate Mechanism-Aware expert answer."""
        # Determine mechanism strength
        if go_score >= 0.7:
            mechanism_strength = "strong"
            conclusion = f"The pathway analysis strongly supports a mechanistic role for {gene} in {disease}."
        elif go_score >= 0.4:
            mechanism_strength = "moderate"
            conclusion = f"The functional evidence suggests {gene} has a plausible mechanistic connection to {disease}, though the pathway may be indirect."
        else:
            mechanism_strength = "limited"
            conclusion = f"The functional connection between {gene} and {disease} requires further investigation."

        return f"""{gene} has a {mechanism_strength} mechanistic connection to {disease}, with a GO functional score of {go_score:.2f}.

{gene} encodes {gene_function}. This molecular function is relevant to understanding how genetic variation might influence disease risk.

The mechanistic pathway to {disease} {disease_mechanism}. The protein encoded by {gene} participates in biological processes that intersect with disease-relevant pathways.

{conclusion}

Understanding the mechanistic basis strengthens causal interpretation and can identify potential therapeutic targets. While statistical association identifies candidate genes, mechanistic understanding explains why the association exists at a molecular level."""

    def _parse_number(self, value: Any) -> Optional[float]:
        """Safely parse a number from various formats."""
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value).strip()
            # Handle scientific notation
            return float(s)
        except:
            return None

    def _prepare_data_row(self, row: Dict[str, Any], disease: str) -> Dict[str, Any]:
        """Prepare a data row with computed fields."""
        data = dict(row)
        data['disease'] = disease

        # Parse OR value
        or_value = self._parse_number(data.get('or_value') or data.get('OR') or data.get('odds_ratio'))
        if or_value:
            data['or_value'] = f"{or_value:.2f}"
            risk_level, risk_interp = self._classify_risk(or_value)
            data['risk_level'] = risk_level
            data['risk_interpretation'] = risk_interp

        # Parse p-value
        p_value = self._parse_number(data.get('p_value') or data.get('P-Value') or data.get('pvalue'))
        if p_value:
            data['p_value'] = f"{p_value:.2e}"
            data['significance_answer'] = self._check_significance(p_value)

        return data

    def generate_from_dataframe(
        self,
        df: pd.DataFrame,
        disease: str,
        max_per_template: int = 50
    ) -> List[GeneratedItem]:
        """Generate questions from a pandas DataFrame."""
        items = []
        template_counts = {t.template_id: 0 for t in self.templates}

        # Standardize column names
        column_mapping = {
            'rsid': ['rsid', 'rsID', 'SNP', 'snp', 'dbsnp.rsid', 'variant'],
            'gene': ['gene', 'Gene', 'gene_symbol', 'symbol', 'Symbol', 'dbsnp.gene.symbol', 'Risk gene'],
            'chromosome': ['chromosome', 'chr', 'Chr', 'chrom', 'dbsnp.chrom'],
            'or_value': ['OR', 'or', 'odds_ratio', 'OddsRatio', ' OR'],
            'p_value': ['P-Value', 'p_value', 'pvalue', 'p-value', 'P_value'],
        }

        # Create standardized dataframe
        std_df = pd.DataFrame()
        for std_col, variants in column_mapping.items():
            for var in variants:
                if var in df.columns:
                    std_df[std_col] = df[var]
                    break

        # Copy any remaining columns
        for col in df.columns:
            if col not in std_df.columns:
                found = False
                for std_col, variants in column_mapping.items():
                    if col in variants:
                        found = True
                        break
                if not found:
                    std_df[col] = df[col]

        logger.info(f"Processing {len(std_df)} rows for disease: {disease}")

        # Generate questions
        for idx, row in std_df.iterrows():
            row_data = self._prepare_data_row(row.to_dict(), disease)

            for template in self.templates:
                if template_counts[template.template_id] >= max_per_template:
                    continue

                if template.can_generate(row_data):
                    # Generate unique ID
                    self.item_counter[template.taxonomy.value] += 1
                    item_id = f"{template.taxonomy.value}-{self.item_counter[template.taxonomy.value]:04d}"

                    item = template.generate(row_data, item_id)
                    if item:
                        # Use expert-style answers for C (Causal) and R (Risk) taxonomies
                        # S and M are simple factual/extraction questions - keep template answers
                        if USE_EXPERT_PROMPTS and item.taxonomy in ['C', 'R']:
                            # Preserve original template answer for comparison
                            original_answer = item.answer

                            expert_answer = self._generate_expert_answer(
                                taxonomy=item.taxonomy,
                                template_label=item.label,
                                data=dict(row_data)  # Copy to avoid mutation
                            )

                            # None means use template answer (e.g., for P-value questions)
                            if expert_answer is not None:
                                # Update item with expert answer
                                item.answer = expert_answer
                                # Also update ground truth
                                item.ground_truth['answer'] = expert_answer
                                item.ground_truth['answer_normalized'] = expert_answer.lower().strip()
                                # Mark as expert-generated in a copy of source_data
                                item.source_data = dict(item.source_data)
                                item.source_data['original_template_answer'] = original_answer
                                item.source_data['expert_prompts_used'] = True
                            else:
                                # Keep template answer
                                item.source_data = dict(item.source_data)
                                item.source_data['expert_prompts_used'] = False

                        items.append(item)
                        template_counts[template.template_id] += 1

        logger.info(f"Generated {len(items)} items")
        return items

    def generate_from_csv(
        self,
        csv_path: str,
        disease: str,
        max_per_template: int = 50
    ) -> List[GeneratedItem]:
        """Generate questions from a CSV file."""
        df = pd.read_csv(csv_path)
        return self.generate_from_dataframe(df, disease, max_per_template)

    def generate_from_multiple_files(
        self,
        file_configs: List[Dict[str, str]],
        max_per_template: int = 50
    ) -> List[GeneratedItem]:
        """
        Generate from multiple files.

        Args:
            file_configs: List of {"path": "...", "disease": "..."}
        """
        all_items = []
        for config in file_configs:
            items = self.generate_from_csv(
                config['path'],
                config['disease'],
                max_per_template
            )
            all_items.extend(items)

        return all_items

    def get_statistics(self, items: List[GeneratedItem]) -> Dict[str, Any]:
        """Get statistics about generated items."""
        stats = {
            'total': len(items),
            'by_taxonomy': {},
            'by_label': {},
            'by_difficulty': {},
            'by_answer_type': {},
        }

        for item in items:
            # By taxonomy
            tax = item.taxonomy
            stats['by_taxonomy'][tax] = stats['by_taxonomy'].get(tax, 0) + 1

            # By label
            label = item.label
            stats['by_label'][label] = stats['by_label'].get(label, 0) + 1

            # By difficulty
            diff = item.difficulty
            stats['by_difficulty'][diff] = stats['by_difficulty'].get(diff, 0) + 1

            # By answer type
            atype = item.answer_type
            stats['by_answer_type'][atype] = stats['by_answer_type'].get(atype, 0) + 1

        return stats

    def regenerate_with_feedback(
        self,
        feedback_items: List[Dict[str, Any]],
        original_items: List[Dict[str, Any]]
    ) -> List[GeneratedItem]:
        """
        Regenerate items based on human feedback.

        This method takes feedback from human experts about items that were
        incorrectly generated (overclaims, wrong causal judgment, etc.) and
        regenerates them with improved prompts.

        Args:
            feedback_items: List of feedback records with improvement hints
                - item_id: ID of item to regenerate
                - issue_type: OVERCLAIM_MISSED, FALSE_POSITIVE_OVERCLAIM, etc.
                - improvement_hint: Specific guidance for improvement
                - corrected_judgment: The correct causal judgment
            original_items: Original items that need regeneration

        Returns:
            List of regenerated items with improved answers
        """
        regenerated = []

        # Build lookup of original items
        original_lookup = {item.get('id', item.get('item_id')): item for item in original_items}

        for feedback in feedback_items:
            item_id = feedback.get('item_id')
            original = original_lookup.get(item_id)

            if not original:
                logger.warning(f"Original item not found for feedback: {item_id}")
                continue

            issue_type = feedback.get('issue_type', 'OTHER')
            improvement_hint = feedback.get('improvement_hint', '')
            corrected_judgment = feedback.get('corrected_judgment', 'ASSOCIATIVE')

            # Apply correction based on issue type
            corrected_answer = self._apply_feedback_correction(
                original_answer=original.get('answer', ''),
                issue_type=issue_type,
                corrected_judgment=corrected_judgment,
                improvement_hint=improvement_hint
            )

            # Create regenerated item
            regen_item = GeneratedItem(
                id=f"{item_id}-REGEN",
                taxonomy=original.get('taxonomy', 'S'),
                label=original.get('label', 'UNKNOWN'),
                template_id=original.get('template_id', 'feedback_correction'),
                question=original.get('question', ''),
                answer=corrected_answer,
                answer_type=original.get('answer_type', 'explanation'),
                entities=original.get('entities', {}),
                ground_truth=original.get('ground_truth', {}),
                difficulty=original.get('difficulty', 'medium'),
                source_data={
                    'original_id': item_id,
                    'feedback_applied': True,
                    'issue_type': issue_type,
                    'corrected_judgment': corrected_judgment
                }
            )
            regenerated.append(regen_item)
            logger.info(f"Regenerated item {item_id} with feedback correction")

        return regenerated

    def _apply_feedback_correction(
        self,
        original_answer: str,
        issue_type: str,
        corrected_judgment: str,
        improvement_hint: str
    ) -> str:
        """
        Apply correction to an answer based on feedback.

        Args:
            original_answer: The original answer that needs correction
            issue_type: Type of issue identified
            corrected_judgment: The correct causal judgment
            improvement_hint: Specific guidance for improvement

        Returns:
            Corrected answer string
        """
        # Define causal language corrections
        causal_to_associative = {
            "causes": "is associated with",
            "leads to": "is associated with",
            "results in": "is associated with",
            "produces": "is associated with",
            "directly causes": "is associated with",
            "is responsible for": "is associated with",
            "triggers": "is associated with",
        }

        associative_to_causal = {
            "is associated with": "causes",
            "is linked to": "leads to",
            "correlates with": "results in",
        }

        corrected = original_answer

        if issue_type == "OVERCLAIM_MISSED":
            # LLM missed an overclaim - convert causal language to associative
            for causal, associative in causal_to_associative.items():
                corrected = corrected.replace(causal, associative)
                corrected = corrected.replace(causal.capitalize(), associative.capitalize())

            # Add disclaimer if not present
            if "does not imply causation" not in corrected.lower():
                corrected += " Note: This association does not imply causation."

        elif issue_type == "FALSE_POSITIVE_OVERCLAIM":
            # LLM wrongly flagged as overclaim - keep original (may need strengthening)
            pass

        elif issue_type == "WRONG_CAUSAL_JUDGMENT":
            # Adjust based on corrected judgment
            if corrected_judgment == "ASSOCIATIVE":
                for causal, associative in causal_to_associative.items():
                    corrected = corrected.replace(causal, associative)
                    corrected = corrected.replace(causal.capitalize(), associative.capitalize())
            elif corrected_judgment == "CAUSAL":
                for associative, causal in associative_to_causal.items():
                    corrected = corrected.replace(associative, causal)
                    corrected = corrected.replace(associative.capitalize(), causal.capitalize())

        return corrected

    def get_feedback_summary(
        self,
        feedback_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get a summary of feedback for generator improvement.

        Args:
            feedback_items: List of feedback records

        Returns:
            Summary statistics and improvement recommendations
        """
        summary = {
            'total_feedback': len(feedback_items),
            'by_issue_type': {},
            'by_taxonomy': {},
            'templates_to_review': set(),
            'recommendations': []
        }

        for feedback in feedback_items:
            issue = feedback.get('issue_type', 'OTHER')
            taxonomy = feedback.get('taxonomy', 'U')

            summary['by_issue_type'][issue] = summary['by_issue_type'].get(issue, 0) + 1
            summary['by_taxonomy'][taxonomy] = summary['by_taxonomy'].get(taxonomy, 0) + 1

        # Generate recommendations
        if summary['by_issue_type'].get('OVERCLAIM_MISSED', 0) > 0:
            summary['recommendations'].append(
                "Review Causal (C) taxonomy templates for overclaim language. "
                "Ensure answers use 'associated with' for GWAS-only evidence."
            )

        if summary['by_issue_type'].get('WRONG_CAUSAL_JUDGMENT', 0) > 0:
            summary['recommendations'].append(
                "Review causal reasoning prompts. Consider adding more examples "
                "of correct causal vs associative language."
            )

        summary['templates_to_review'] = list(summary['templates_to_review'])
        return summary
