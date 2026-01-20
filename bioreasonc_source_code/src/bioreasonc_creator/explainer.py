"""
Explainer Module for BioREASONC-Creator

================================================================================
MODULE OVERVIEW
================================================================================

Generates concise biomedical explanations (~35 words) for each Q&A pair.
Explanations provide context about why the answer is correct based on
biomedical principles.

This is STEP 3 in the BioREASONC pipeline:
    Generator → Validator → EXPLAINER → Paraphraser → Exporter

Focus: "Does the model tell the truth about causality when explaining biomedical research?"
- For causal (C) questions, emphasizes association vs causation distinction
- Uses centralized prompts from prompts.py

================================================================================
PURPOSE
================================================================================

Explanations serve three key purposes:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. EDUCATIONAL                                                                  │
│    Help users understand WHY an answer is correct                               │
│    Teach proper biomedical reasoning patterns                                   │
│                                                                                 │
│ 2. EVALUATION CONTEXT                                                           │
│    Provide models with reasoning context during inference                       │
│    Enable few-shot learning with explanatory examples                           │
│                                                                                 │
│ 3. QUALITY ASSURANCE                                                            │
│    Explicit reasoning can be validated for correctness                          │
│    Catches logical errors in Q&A pairs                                          │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
INPUT SPECIFICATION
================================================================================

generate_explanation() Parameters:
┌────────────────────┬──────────────┬────────────────────────────────────────────┐
│ Parameter          │ Type         │ Description                                │
├────────────────────┼──────────────┼────────────────────────────────────────────┤
│ item_id            │ str          │ Unique identifier (e.g., "C-0001")        │
│ question           │ str          │ Question text                              │
│ answer             │ str          │ Answer text                                │
│ taxonomy           │ str          │ S, C, R, or M                              │
│ label              │ str          │ Template label (e.g., "C-CAUSAL-VS-ASSOC") │
└────────────────────┴──────────────┴────────────────────────────────────────────┘

add_explanations() Input:
```python
items = [
    {
        "id": "C-0001",
        "question": "Is the relationship between GCK and T2D causal?",
        "answer": "The relationship is ASSOCIATIVE, not causal...",
        "taxonomy": "C",
        "label": "C-CAUSAL-VS-ASSOC"
    },
    ...
]
```

================================================================================
OUTPUT SPECIFICATION
================================================================================

ExplanationResult Fields:
┌────────────────────┬──────────────┬────────────────────────────────────────────┐
│ Field              │ Type         │ Description                                │
├────────────────────┼──────────────┼────────────────────────────────────────────┤
│ item_id            │ str          │ Original item ID                           │
│ explanation        │ str          │ Generated explanation text (~35 words)     │
│ word_count         │ int          │ Actual word count                          │
│ method             │ str          │ 'llm-openai', 'llm-anthropic', or 'template' │
└────────────────────┴──────────────┴────────────────────────────────────────────┘

Example Output:
```python
ExplanationResult(
    item_id="C-0001",
    explanation="GWAS identifies statistical associations, NOT causal relationships.
                 Causation requires: (1) consistent association, (2) temporal
                 precedence, (3) biological mechanism, (4) MR evidence.",
    word_count=28,
    method="llm-openai"
)
```

================================================================================
GENERATION METHODS
================================================================================

The Explainer uses a fallback approach:

┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Try LLM Generation (if use_llm=True)                                   │
│         Priority: OpenAI → Anthropic                                            │
│         Benefits: Context-specific, dynamic explanations                        │
│                                                                                 │
│ STEP 2: Fall back to Templates (if LLM unavailable/fails)                       │
│         Uses pre-defined explanations from ExplainerPrompts                     │
│         Benefits: Fast, consistent, no API costs                                │
└─────────────────────────────────────────────────────────────────────────────────┘

Template Selection:
┌──────────┬──────────────────────────────────────────────────────────────────────┐
│ Taxonomy │ Template Focus                                                       │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│ S        │ Network structure, biological pathways, gene mapping                 │
│ C        │ Association vs causation, MR evidence, confounding                   │
│ R        │ Odds ratio interpretation, risk classification, p-values             │
│ M        │ Entity recognition, relation extraction, text mining                 │
└──────────┴──────────────────────────────────────────────────────────────────────┘

================================================================================
CONFIGURATION
================================================================================

ExplanationGenerator Parameters:
┌─────────────────────────┬─────────────┬──────────────────────────────────────────┐
│ Parameter               │ Default     │ Description                              │
├─────────────────────────┼─────────────┼──────────────────────────────────────────┤
│ openai_api_key          │ None        │ OpenAI API key for LLM generation        │
│ anthropic_api_key       │ None        │ Anthropic API key for LLM generation     │
│ target_words            │ 35          │ Target word count for explanations       │
│ use_llm                 │ True        │ Whether to attempt LLM generation        │
└─────────────────────────┴─────────────┴──────────────────────────────────────────┘

Word Count Guidelines:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Target: 35-50 words                                                             │
│ Maximum: 50 words (truncated with "..." if exceeded)                            │
│ Minimum: No strict minimum, but explanations should be meaningful               │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
BEST PRACTICES
================================================================================

★ EXPLANATION QUALITY:
  • Match explanation to question taxonomy
  • For C taxonomy: ALWAYS mention association vs causation
  • Include specific evidence types (MR, OR, p-value)
  • Avoid jargon overload

★ CONSISTENCY:
  • Use template fallback for consistent baseline
  • LLM explanations should align with template themes
  • Explanation should not contradict the answer

★ EDUCATIONAL VALUE:
  • Include actionable information
  • Explain what evidence would strengthen claims
  • Use consistent terminology with Q&A

================================================================================
USAGE EXAMPLES
================================================================================

Example 1: Single Explanation
```python
from bioreasonc_creator.explainer import ExplanationGenerator

explainer = ExplanationGenerator(
    openai_api_key="sk-...",
    target_words=35
)

result = explainer.generate_explanation(
    item_id="C-0001",
    question="Is the relationship between GCK and T2D causal?",
    answer="The relationship is ASSOCIATIVE, not causal...",
    taxonomy="C",
    label="C-CAUSAL-VS-ASSOC"
)

print(f"Explanation ({result.word_count} words): {result.explanation}")
print(f"Method: {result.method}")
```

Example 2: Batch Processing
```python
items = [...]  # List from Validator
explained_items = explainer.add_explanations(
    items,
    progress_callback=lambda done, total: print(f"{done}/{total}")
)
stats = explainer.get_stats(explained_items)
print(f"Avg words: {stats['avg_word_count']:.1f}")
```

Example 3: Template-Only Mode
```python
explainer = ExplanationGenerator(use_llm=False)
result = explainer.generate_explanation(...)  # Uses templates only
```

================================================================================
Author: Sakhaa Alsaedi, sakhaa.alsaedi@kaust.edu.sa
================================================================================
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# Import centralized prompts
from .prompts import ExplainerPrompts, TAXONOMY_DESCRIPTIONS

# LLM clients
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class ExplanationResult:
    """Result of explanation generation."""
    item_id: str
    explanation: str
    word_count: int
    method: str  # 'llm' or 'template'


class ExplanationGenerator:
    """
    Generates concise biomedical explanations for Q&A pairs.
    Target: ~35 words explaining why the answer is correct.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        target_words: int = 35,
        use_llm: bool = True
    ):
        """
        Initialize the explanation generator.

        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            target_words: Target word count for explanations (~35)
            use_llm: Whether to use LLM for explanation generation
        """
        self.target_words = target_words
        self.use_llm = use_llm

        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None

        if use_llm:
            if openai_api_key and HAS_OPENAI:
                self.openai_client = OpenAI(api_key=openai_api_key)
            if anthropic_api_key and HAS_ANTHROPIC:
                self.anthropic_client = Anthropic(api_key=anthropic_api_key)

        # Template explanations by taxonomy and label
        self.explanation_templates = self._load_explanation_templates()

    def _load_explanation_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load template explanations for each taxonomy/label combination.
        Uses centralized prompts from ExplainerPrompts.TEMPLATE_EXPLANATIONS.
        """
        return ExplainerPrompts.TEMPLATE_EXPLANATIONS

    def _load_explanation_templates_legacy(self) -> Dict[str, Dict[str, str]]:
        """Legacy template explanations (kept for reference)."""
        return {
            # Structure (S) - Graph-based reasoning
            "S": {
                "S-PATH-FIND": "Biological pathway analysis identifies sequential molecular interactions. The shortest path represents the most direct functional connection between entities in the biological network.",
                "S-CLUSTER": "Network clustering reveals functionally related gene modules. Genes within clusters often share biological functions or participate in common cellular processes.",
                "S-CENTRALITY": "Centrality metrics identify key regulatory nodes. High-centrality genes are often essential regulators with significant impact on network function and disease phenotypes.",
                "S-CONNECTIVITY": "Network connectivity analysis reveals interaction density. Highly connected genes (hubs) often play critical roles in maintaining cellular function and disease mechanisms.",
                "DEFAULT": "Biological network analysis reveals structural relationships between molecular entities, informing functional interpretations and mechanistic understanding."
            },
            # Causal (C) - Causal inference reasoning
            "C": {
                "C-CAUSAL-VS-ASSOC": "Distinguishing causation from association requires: (1) consistent association, (2) temporal precedence, (3) dose-response, and (4) experimental/MR evidence.",
                "C-MR-ANALYSIS": "Mendelian randomization uses genetic variants as instrumental variables to infer causality, reducing confounding inherent in observational studies.",
                "C-CONFOUND": "Confounding occurs when a third variable influences both exposure and outcome. Proper causal inference requires identifying and controlling for confounders.",
                "DEFAULT": "Causal inference in genetics distinguishes true causal effects from spurious associations using methods like Mendelian randomization and instrumental variables."
            },
            # Risk (R) - Risk assessment reasoning
            "R": {
                "R-RISK-LEVEL": "Risk classification: HIGH (OR\u22651.5), MODERATE (1.2\u2264OR<1.5), LOW (0.8<OR<1.2), PROTECTIVE (OR\u22640.8).",
                "R-PVAL-SIGNIF": "Statistical significance (p<5e-8 for GWAS) indicates reliable association. However, significance alone doesn't determine clinical relevance or effect size.",
                "R-EFFECT-SIZE": "Effect size (OR/beta) quantifies association strength. Larger effects suggest stronger genetic influence on disease risk or trait variation.",
                "R-PRS": "Polygenic risk scores aggregate effects across many variants to estimate overall genetic liability, enabling personalized risk stratification.",
                "DEFAULT": "Genetic risk assessment combines statistical significance, effect size, and biological plausibility to evaluate variant contributions to disease susceptibility."
            },
            # Semantic (M) - Text mining reasoning
            "M": {
                "M-NER": "Named entity recognition identifies biomedical terms (genes, diseases, drugs) in text, enabling structured information extraction from literature.",
                "M-REL-EXTRACT": "Relation extraction identifies subject-predicate-object triples from biomedical text.",
                "M-SENT-CLASS": "Sentiment classification determines the nature of biomedical assertions (positive, negative, speculative), aiding evidence interpretation.",
                "M-SUMM": "Text summarization condenses biomedical literature while preserving key findings, enabling efficient knowledge synthesis across publications.",
                "DEFAULT": "Semantic analysis of biomedical text extracts structured knowledge from unstructured literature, enabling systematic evidence synthesis."
            }
        }

    def _get_template_explanation(self, taxonomy: str, label: str) -> str:
        """Get template explanation for taxonomy/label combination."""
        taxonomy_templates = self.explanation_templates.get(taxonomy, {})
        return taxonomy_templates.get(label, taxonomy_templates.get("DEFAULT",
            "This question tests biomedical reasoning ability by requiring integration of domain knowledge with logical inference."))

    def _generate_llm_prompt(
        self,
        question: str,
        answer: str,
        taxonomy: str,
        label: str
    ) -> str:
        """
        Generate prompt for LLM explanation generation.
        Uses centralized taxonomy descriptions from prompts.py.
        """
        taxonomy_desc = TAXONOMY_DESCRIPTIONS.get(taxonomy, 'Biomedical reasoning')

        return ExplainerPrompts.EXPLANATION_PROMPT_TEMPLATE.format(
            taxonomy=taxonomy,
            taxonomy_description=taxonomy_desc,
            question=question,
            answer=answer
        )

    def _generate_with_openai(
        self,
        question: str,
        answer: str,
        taxonomy: str,
        label: str
    ) -> Optional[str]:
        """Generate explanation using OpenAI with centralized prompts."""
        if not self.openai_client:
            return None

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": ExplainerPrompts.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": self._generate_llm_prompt(question, answer, taxonomy, label)
                    }
                ],
                temperature=0.3,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI explanation error: {e}")
            return None

    def _generate_with_anthropic(
        self,
        question: str,
        answer: str,
        taxonomy: str,
        label: str
    ) -> Optional[str]:
        """Generate explanation using Anthropic with centralized prompts."""
        if not self.anthropic_client:
            return None

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                temperature=0.3,
                system=ExplainerPrompts.SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": self._generate_llm_prompt(question, answer, taxonomy, label)
                    }
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic explanation error: {e}")
            return None

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def _truncate_explanation(self, explanation: str, max_words: int = 50) -> str:
        """Truncate explanation to max words if needed."""
        words = explanation.split()
        if len(words) > max_words:
            return ' '.join(words[:max_words]) + '...'
        return explanation

    def generate_explanation(
        self,
        item_id: str,
        question: str,
        answer: str,
        taxonomy: str,
        label: str
    ) -> ExplanationResult:
        """
        Generate explanation for a single Q&A pair.

        Args:
            item_id: Item identifier
            question: Question text
            answer: Answer text
            taxonomy: Taxonomy category (S/C/R/M)
            label: Question label

        Returns:
            ExplanationResult with generated explanation
        """
        explanation = None
        method = 'template'

        # Try LLM generation first
        if self.use_llm:
            if self.openai_client:
                explanation = self._generate_with_openai(question, answer, taxonomy, label)
                if explanation:
                    method = 'llm-openai'

            if not explanation and self.anthropic_client:
                explanation = self._generate_with_anthropic(question, answer, taxonomy, label)
                if explanation:
                    method = 'llm-anthropic'

        # Fall back to template
        if not explanation:
            explanation = self._get_template_explanation(taxonomy, label)
            method = 'template'

        # Ensure appropriate length
        explanation = self._truncate_explanation(explanation, max_words=50)

        return ExplanationResult(
            item_id=item_id,
            explanation=explanation,
            word_count=self._count_words(explanation),
            method=method
        )

    def add_explanations(
        self,
        items: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Add explanations to a batch of items.

        Args:
            items: List of items (dicts with question, answer, taxonomy, label)
            progress_callback: Optional callback for progress updates

        Returns:
            List of items with explanations added
        """
        results = []
        total = len(items)

        for idx, item in enumerate(items):
            result_item = dict(item)

            # Generate explanation
            explanation_result = self.generate_explanation(
                item_id=item.get('id', f'item_{idx}'),
                question=item.get('question', ''),
                answer=item.get('answer', ''),
                taxonomy=item.get('taxonomy', 'U'),
                label=item.get('label', 'UNKNOWN')
            )

            result_item['explanation'] = explanation_result.explanation
            result_item['explanation_method'] = explanation_result.method
            result_item['explanation_words'] = explanation_result.word_count

            results.append(result_item)

            if progress_callback:
                progress_callback(idx + 1, total)

        return results

    def get_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about explanation generation."""
        word_counts = [r.get('explanation_words', 0) for r in results]
        methods = [r.get('explanation_method', 'unknown') for r in results]

        method_counts = {}
        for m in methods:
            method_counts[m] = method_counts.get(m, 0) + 1

        return {
            'total_items': len(results),
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min_word_count': min(word_counts) if word_counts else 0,
            'max_word_count': max(word_counts) if word_counts else 0,
            'method_distribution': method_counts,
            'by_taxonomy': {
                taxonomy: sum(1 for r in results if r.get('taxonomy') == taxonomy)
                for taxonomy in ['S', 'C', 'R', 'M']
            }
        }
