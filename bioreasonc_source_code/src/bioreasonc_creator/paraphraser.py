"""
Paraphraser Module for BioREASONC-Creator

================================================================================
MODULE OVERVIEW
================================================================================

Generates 2-3 diverse paraphrases per question while preserving:
- Biomedical entity names exactly (genes, variants, diseases)
- Core semantic meaning
- Question intent and structure

This is STEP 4 in the BioREASONC pipeline:
    Generator → Validator → Explainer → PARAPHRASER → Exporter

Focus: "Does the model tell the truth about causality when explaining biomedical research?"
- For causal questions: preserve the causal/associative framing
- Uses centralized prompts from prompts.py

================================================================================
PURPOSE
================================================================================

Paraphrasing tests model ROBUSTNESS:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. PREVENTS PATTERN MATCHING                                                   │
│    Models should understand meaning, not memorize phrasings                    │
│                                                                                 │
│ 2. TESTS GENERALIZATION                                                        │
│    Same answer should work for different question phrasings                    │
│                                                                                 │
│ 3. INCREASES BENCHMARK DIVERSITY                                               │
│    Multiple phrasings expand evaluation coverage                               │
│                                                                                 │
│ 4. DETECTS OVERFITTING                                                         │
│    If model only answers one phrasing correctly, it's overfitting             │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
INPUT SPECIFICATION
================================================================================

paraphrase_question() Parameters:
┌────────────────────┬──────────────┬────────────────────────────────────────────┐
│ Parameter          │ Type         │ Description                                │
├────────────────────┼──────────────┼────────────────────────────────────────────┤
│ question           │ str          │ Original question text                     │
│ item_id            │ str          │ Unique identifier (e.g., "C-0001")        │
│ taxonomy           │ str          │ S, C, R, or M                              │
│ label              │ str          │ Template label (e.g., "C-CAUSAL-VS-ASSOC") │
└────────────────────┴──────────────┴────────────────────────────────────────────┘

paraphrase_batch() Input:
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

ParaphrasedItem Fields:
┌─────────────────────────┬──────────────┬────────────────────────────────────────┐
│ Field                   │ Type         │ Description                            │
├─────────────────────────┼──────────────┼────────────────────────────────────────┤
│ original_id             │ str          │ Original item ID                       │
│ original_question       │ str          │ Original question text                 │
│ paraphrased_question    │ str          │ Paraphrased question text              │
│ paraphrase_index        │ int          │ Index of this paraphrase (0, 1, 2)     │
│ preserved_entities      │ List[str]    │ Entities preserved in paraphrase       │
│ taxonomy                │ str          │ S, C, R, or M                          │
│ label                   │ str          │ Template label                         │
└─────────────────────────┴──────────────┴────────────────────────────────────────┘

Example Output (from paraphrase_batch):
```python
[
    # Original item (preserved)
    {
        "id": "C-0001",
        "question": "Is the relationship between GCK and T2D causal?",
        "paraphrased": False,
        ...
    },
    # Paraphrase 1
    {
        "id": "C-0001-P0",
        "question": "Does GCK have a causal relationship with T2D?",
        "paraphrased": True,
        "original_question": "Is the relationship between GCK and T2D causal?",
        "preserved_entities": ["GCK", "T2D"],
        ...
    },
    # Paraphrase 2
    {
        "id": "C-0001-P1",
        "question": "Can we say GCK causes T2D, or is it just associated?",
        "paraphrased": True,
        ...
    }
]
```

================================================================================
ENTITY PRESERVATION
================================================================================

CRITICAL: Entities must be preserved EXACTLY (case-sensitive):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Entity Pattern               │ Example                                         │
├──────────────────────────────┼─────────────────────────────────────────────────┤
│ Gene names (CAPS)            │ BRCA1, GCK, ACE2, TP53                         │
│ SNP IDs (rs prefix)          │ rs1799884, rs4607517                           │
│ Disease names                │ COVID-19, SARS-CoV-2, Type 2 Diabetes          │
│ Odds ratios (OR=value)       │ OR=1.45, OR=2.3                                 │
│ P-values (p=value)           │ p=5e-8, p=1.2e-10                              │
│ Risk levels                  │ HIGH, MODERATE, LOW, PROTECTIVE                │
└─────────────────────────────────────────────────────────────────────────────────┘

Verification Process:
```
Original:  "Is BRCA1 a risk factor for Breast Cancer?"
Paraphrase: "Does BRCA1 increase risk for Breast Cancer?"

Entities to preserve: ["BRCA1", "Breast Cancer"]
Verification: Both present → VALID

Paraphrase: "Does Brca1 increase risk for breast cancer?"
Verification: Case mismatch → INVALID (fall back to rule-based)
```

================================================================================
PARAPHRASING METHODS
================================================================================

The Paraphraser uses a fallback approach:

┌─────────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Try LLM Generation (if use_llm=True)                                   │
│         Alternates: OpenAI → Anthropic → OpenAI → ...                          │
│         Temperature increases with index for diversity                          │
│         Benefits: Natural, diverse paraphrases                                 │
│                                                                                 │
│ STEP 2: Verify Entity Preservation                                             │
│         All entities must appear in paraphrase (case-insensitive check)        │
│         If verification fails → fall back to rule-based                        │
│                                                                                 │
│ STEP 3: Fall back to Rule-Based (if LLM unavailable/fails verification)        │
│         Uses transformation templates for consistent quality                   │
│         Benefits: Fast, guaranteed entity preservation                         │
└─────────────────────────────────────────────────────────────────────────────────┘

Rule-Based Transformations:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ "Is the relationship" → "Does the relationship", "Can the relationship"        │
│ "What is the" → "Determine the", "Identify the"                                │
│ "causal or merely associative" → "causative or just correlative"               │
│ "risk level" → "level of risk", "degree of risk"                               │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
CONFIGURATION
================================================================================

QuestionParaphraser Parameters:
┌─────────────────────────┬─────────────┬──────────────────────────────────────────┐
│ Parameter               │ Default     │ Description                              │
├─────────────────────────┼─────────────┼──────────────────────────────────────────┤
│ openai_api_key          │ None        │ OpenAI API key for LLM generation        │
│ anthropic_api_key       │ None        │ Anthropic API key for LLM generation     │
│ num_paraphrases         │ 2           │ Number of paraphrases per question (1-3) │
│ use_llm                 │ True        │ Whether to attempt LLM generation        │
└─────────────────────────┴─────────────┴──────────────────────────────────────────┘

================================================================================
CRITICAL: CAUSAL FRAMING PRESERVATION
================================================================================

For Taxonomy C (Causal-Aware), the paraphrase MUST preserve causal framing:

┌─────────────────────────────────────────────────────────────────────────────────┐
│ ✓ CORRECT:                                                                      │
│   Original:   "Is the relationship between GCK and T2D causal or associative?" │
│   Paraphrase: "Does GCK have a causal or merely correlative relationship..."   │
│   → Preserves causal/associative distinction                                   │
│                                                                                 │
│ ✗ INCORRECT:                                                                    │
│   Original:   "Is the relationship between GCK and T2D causal or associative?" │
│   Paraphrase: "Does GCK cause T2D?"                                            │
│   → Drops "associative" option, changes meaning                                │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
BEST PRACTICES
================================================================================

★ ENTITY PRESERVATION:
  • Always verify entities after paraphrasing
  • Fall back to rule-based if LLM drops entities
  • Case-insensitive check, but preserve original case in output

★ DIVERSITY:
  • Use different temperatures for each paraphrase
  • Alternate between OpenAI and Anthropic
  • Avoid duplicate paraphrases

★ MEANING PRESERVATION:
  • Same answer must be valid for all paraphrases
  • For C taxonomy: preserve causal/associative framing
  • For R taxonomy: preserve risk direction (increase vs decrease)

================================================================================
USAGE EXAMPLES
================================================================================

Example 1: Single Question
```python
from bioreasonc_creator.paraphraser import QuestionParaphraser

paraphraser = QuestionParaphraser(
    openai_api_key="sk-...",
    num_paraphrases=2
)

paraphrases = paraphraser.paraphrase_question(
    question="Is BRCA1 a genetic risk factor for Breast Cancer?",
    item_id="R-0001",
    taxonomy="R",
    label="R-RISK-FACTOR"
)

for p in paraphrases:
    print(f"[P{p.paraphrase_index}] {p.paraphrased_question}")
    print(f"    Entities: {p.preserved_entities}")
```

Example 2: Batch Processing
```python
items = [...]  # List from Explainer
paraphrased_items = paraphraser.paraphrase_batch(
    items,
    progress_callback=lambda done, total: print(f"{done}/{total}")
)
stats = paraphraser.get_stats(paraphrased_items)
print(f"Total items: {stats['total_items']} ({stats['paraphrased_items']} paraphrased)")
```

Example 3: Rule-Based Only
```python
paraphraser = QuestionParaphraser(use_llm=False, num_paraphrases=2)
paraphrases = paraphraser.paraphrase_question(...)  # Uses templates only
```

================================================================================
Author: Sakhaa Alsaedi, sakhaa.alsaedi@kaust.edu.sa
================================================================================
"""

import re
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import asyncio

# Import centralized prompts
from .prompts import ParaphraserPrompts

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
class ParaphrasedItem:
    """A paraphrased question item."""
    original_id: str
    original_question: str
    paraphrased_question: str
    paraphrase_index: int
    preserved_entities: List[str]
    taxonomy: str
    label: str


class QuestionParaphraser:
    """
    Generates diverse paraphrases of biomedical questions while
    preserving entity names and semantic meaning.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        num_paraphrases: int = 2,
        use_llm: bool = True
    ):
        """
        Initialize the paraphraser.

        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            num_paraphrases: Number of paraphrases to generate (2-3)
            use_llm: Whether to use LLM for paraphrasing (vs rule-based)
        """
        self.num_paraphrases = min(max(num_paraphrases, 1), 3)
        self.use_llm = use_llm

        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None

        if use_llm:
            if openai_api_key and HAS_OPENAI:
                self.openai_client = OpenAI(api_key=openai_api_key)
            if anthropic_api_key and HAS_ANTHROPIC:
                self.anthropic_client = Anthropic(api_key=anthropic_api_key)

        # Entity patterns for preservation
        self.entity_patterns = [
            r'\b[A-Z][A-Z0-9]{1,10}\b',  # Gene names (e.g., BRCA1, ACE2)
            r'\brs\d+\b',  # SNP IDs
            r'\b(?:COVID-19|SARS-CoV-2)\b',  # Disease names
            r'\bOR\s*=\s*[\d.]+\b',  # Odds ratios
            r'\bp\s*=\s*[\d.e-]+\b',  # P-values
            r'\b(?:HIGH|MODERATE|LOW|PROTECTIVE)\b',  # Risk levels
        ]

        # Paraphrase templates for rule-based fallback
        self.paraphrase_templates = self._load_paraphrase_templates()

    def _load_paraphrase_templates(self) -> Dict[str, List[str]]:
        """Load paraphrase transformation templates."""
        return {
            # Question starters
            "Is the relationship": [
                "Does the relationship",
                "Can the relationship",
                "Would you characterize the relationship"
            ],
            "What is the": [
                "Determine the",
                "Identify the",
                "What would be the"
            ],
            "Extract the": [
                "Identify and extract the",
                "Parse and extract the",
                "What is the extracted"
            ],
            "Which genes": [
                "What genes",
                "Identify which genes",
                "List the genes that"
            ],
            "Describe the": [
                "Explain the",
                "Characterize the",
                "What is the nature of the"
            ],
            # Relationship terms
            "causal or merely associative": [
                "causative or just correlative",
                "a cause or simply associated",
                "causal in nature or merely a correlation"
            ],
            "based on the genetic evidence": [
                "considering the genetic data",
                "according to genetic findings",
                "given the genetic evidence available"
            ],
            "risk level": [
                "level of risk",
                "degree of risk",
                "risk classification"
            ],
            "biomedical relationship": [
                "biological relationship",
                "medical relationship",
                "biomedical connection"
            ],
        }

    def _extract_entities(self, text: str) -> List[str]:
        """Extract biomedical entities that must be preserved."""
        entities = set()
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(matches)
        return list(entities)

    def _rule_based_paraphrase(self, question: str, index: int) -> str:
        """Generate a paraphrase using rule-based transformations."""
        paraphrased = question

        # Apply transformations based on index for variety
        applied = 0
        for original, alternatives in self.paraphrase_templates.items():
            if original.lower() in paraphrased.lower() and applied < 2:
                # Use different alternative based on index
                alt_idx = (index + applied) % len(alternatives)
                # Case-insensitive replacement
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                paraphrased = pattern.sub(alternatives[alt_idx], paraphrased, count=1)
                applied += 1

        # If no transformations applied, add prefix variation
        if applied == 0:
            prefixes = [
                "Based on the data, ",
                "Given the evidence, ",
                "Considering the information, "
            ]
            paraphrased = prefixes[index % len(prefixes)] + paraphrased[0].lower() + paraphrased[1:]

        return paraphrased

    def _llm_paraphrase_prompt(self, question: str, entities: List[str]) -> str:
        """
        Generate prompt for LLM paraphrasing.
        Uses centralized prompt from ParaphraserPrompts.
        """
        entities_str = ", ".join(entities) if entities else "none identified"
        return ParaphraserPrompts.PARAPHRASE_PROMPT.format(
            entities=entities_str,
            question=question
        )

    def _paraphrase_with_openai(
        self,
        question: str,
        entities: List[str],
        index: int
    ) -> Optional[str]:
        """Generate paraphrase using OpenAI with centralized prompts."""
        if not self.openai_client:
            return None

        try:
            # Vary temperature based on index for diversity
            temperature = 0.5 + (index * 0.15)

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": ParaphraserPrompts.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": self._llm_paraphrase_prompt(question, entities)
                    }
                ],
                temperature=temperature,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI paraphrase error: {e}")
            return None

    def _paraphrase_with_anthropic(
        self,
        question: str,
        entities: List[str],
        index: int
    ) -> Optional[str]:
        """Generate paraphrase using Anthropic with centralized prompts."""
        if not self.anthropic_client:
            return None

        try:
            temperature = 0.5 + (index * 0.15)

            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=temperature,
                system=ParaphraserPrompts.SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": self._llm_paraphrase_prompt(question, entities)
                    }
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic paraphrase error: {e}")
            return None

    def _verify_entity_preservation(
        self,
        original: str,
        paraphrased: str,
        entities: List[str]
    ) -> bool:
        """Verify that all entities are preserved in paraphrase."""
        paraphrased_lower = paraphrased.lower()
        for entity in entities:
            if entity.lower() not in paraphrased_lower:
                return False
        return True

    def paraphrase_question(
        self,
        question: str,
        item_id: str,
        taxonomy: str,
        label: str
    ) -> List[ParaphrasedItem]:
        """
        Generate paraphrases for a single question.

        Args:
            question: Original question text
            item_id: Item identifier
            taxonomy: Taxonomy category (S/C/R/M)
            label: Question label

        Returns:
            List of ParaphrasedItem objects
        """
        entities = self._extract_entities(question)
        paraphrases = []

        for i in range(self.num_paraphrases):
            paraphrased = None

            # Try LLM paraphrasing first
            if self.use_llm:
                # Alternate between providers for variety
                if i % 2 == 0 and self.openai_client:
                    paraphrased = self._paraphrase_with_openai(question, entities, i)
                elif self.anthropic_client:
                    paraphrased = self._paraphrase_with_anthropic(question, entities, i)
                elif self.openai_client:
                    paraphrased = self._paraphrase_with_openai(question, entities, i)

            # Verify entity preservation or fall back to rule-based
            if paraphrased:
                if not self._verify_entity_preservation(question, paraphrased, entities):
                    paraphrased = None

            # Fall back to rule-based if LLM failed
            if not paraphrased:
                paraphrased = self._rule_based_paraphrase(question, i)

            # Ensure paraphrase is different from original
            if paraphrased.strip() != question.strip():
                paraphrases.append(ParaphrasedItem(
                    original_id=item_id,
                    original_question=question,
                    paraphrased_question=paraphrased,
                    paraphrase_index=i,
                    preserved_entities=entities,
                    taxonomy=taxonomy,
                    label=label
                ))

        return paraphrases

    def paraphrase_batch(
        self,
        items: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate paraphrases for a batch of items.

        Args:
            items: List of generated items (dicts with question, id, taxonomy, label)
            progress_callback: Optional callback for progress updates

        Returns:
            List of items with paraphrased versions added
        """
        results = []
        total = len(items)

        for idx, item in enumerate(items):
            # Keep original
            original_item = dict(item)
            original_item['paraphrased'] = False
            original_item['original_question'] = None
            results.append(original_item)

            # Generate paraphrases
            paraphrases = self.paraphrase_question(
                question=item.get('question', ''),
                item_id=item.get('id', f'item_{idx}'),
                taxonomy=item.get('taxonomy', 'U'),
                label=item.get('label', 'UNKNOWN')
            )

            # Add paraphrased versions
            for p_idx, paraphrase in enumerate(paraphrases):
                paraphrased_item = dict(item)
                paraphrased_item['id'] = f"{item.get('id', f'item_{idx}')}-P{p_idx}"
                paraphrased_item['question'] = paraphrase.paraphrased_question
                paraphrased_item['paraphrased'] = True
                paraphrased_item['original_question'] = paraphrase.original_question
                paraphrased_item['preserved_entities'] = paraphrase.preserved_entities
                results.append(paraphrased_item)

            if progress_callback:
                progress_callback(idx + 1, total)

        return results

    def get_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about paraphrasing results."""
        original_count = sum(1 for r in results if not r.get('paraphrased', False))
        paraphrased_count = sum(1 for r in results if r.get('paraphrased', False))

        return {
            'total_items': len(results),
            'original_items': original_count,
            'paraphrased_items': paraphrased_count,
            'paraphrases_per_original': paraphrased_count / original_count if original_count > 0 else 0,
            'by_taxonomy': {
                taxonomy: sum(1 for r in results if r.get('taxonomy') == taxonomy)
                for taxonomy in ['S', 'C', 'R', 'M']
            }
        }
