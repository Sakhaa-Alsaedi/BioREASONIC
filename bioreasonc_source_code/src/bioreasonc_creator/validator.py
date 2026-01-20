"""
Validator Module for BioREASONC-Creator

================================================================================
MODULE OVERVIEW
================================================================================

Multi-LLM validation using 3 different LLM providers (OpenAI, Anthropic, Gemini)
to score generated Q&A pairs on quality, accuracy, and reasoning.

This is STEP 2 in the BioREASONC pipeline:
    Generator → VALIDATOR → Explainer → Paraphraser → Exporter

Focus: "Does the model tell the truth about causality when explaining biomedical research?"
- Extra validation criteria for Causal (C) taxonomy
- Penalizes overclaiming causation from GWAS evidence
- Uses centralized prompts from prompts.py

================================================================================
WHY MULTI-LLM VALIDATION?
================================================================================

Single-model validation has blind spots:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Problem                          │ Solution                                    │
├──────────────────────────────────┼─────────────────────────────────────────────┤
│ Model-specific biases            │ 3 different models average out biases       │
│ Hallucination detection          │ Consensus catches inconsistencies           │
│ Overclaim detection              │ Majority vote on causal language            │
│ Quality variance                 │ Agreement score flags controversial items   │
└──────────────────────────────────┴─────────────────────────────────────────────┘

Validators Used:
- OpenAI GPT-4o-mini: Fast, good at structured output
- Anthropic Claude-3-Haiku: Strong reasoning, catches subtle overclaims
- Google Gemini-1.5-Flash: Fast, good biomedical knowledge

================================================================================
INPUT SPECIFICATION
================================================================================

The Validator accepts items from the Generator:

validate_item() Parameters:
┌────────────────────┬──────────────┬────────────────────────────────────────────┐
│ Parameter          │ Type         │ Description                                │
├────────────────────┼──────────────┼────────────────────────────────────────────┤
│ item_id            │ str          │ Unique identifier (e.g., "C-0001")        │
│ question           │ str          │ Question text to validate                  │
│ answer             │ str          │ Answer text to validate                    │
│ explanation        │ str          │ Explanation (may be empty)                 │
│ taxonomy           │ str          │ S, C, R, or M                              │
│ label              │ str          │ Template label (e.g., "C-CAUSAL-VS-ASSOC") │
│ ground_truth       │ Optional[str]│ Expected answer for comparison             │
└────────────────────┴──────────────┴────────────────────────────────────────────┘

validate_batch() Input:
```python
items = [
    {
        "id": "C-0001",
        "question": "Is the relationship between GCK and T2D causal?",
        "answer": "The relationship is ASSOCIATIVE, not causal...",
        "explanation": "",
        "taxonomy": "C",
        "label": "C-CAUSAL-VS-ASSOC",
        "ground_truth": {"answer": "ASSOCIATIVE", "confidence": 1.0}
    },
    ...
]
```

================================================================================
OUTPUT SPECIFICATION
================================================================================

ValidationResult Fields:
┌────────────────────────┬──────────────────┬────────────────────────────────────┐
│ Field                  │ Type             │ Description                        │
├────────────────────────┼──────────────────┼────────────────────────────────────┤
│ item_id                │ str              │ Original item ID                   │
│ scores                 │ Dict[str, float] │ Provider → score (1-5)            │
│ avg_score              │ float            │ Average across providers           │
│ is_valid               │ bool             │ Passes validation threshold?       │
│ feedback               │ Dict[str, str]   │ Provider → feedback text          │
│ ground_truth_match     │ Optional[bool]   │ Answer matches ground truth?       │
│ agreement_score        │ int              │ 1 if all agree, 0 if disagree     │
│ consensus_judgment     │ str              │ ASSOCIATIVE | CAUSAL | NOT_APPLICABLE │
│ overclaim_consensus    │ bool             │ Majority says overclaim?          │
│ causal_judgments       │ Dict[str, str]   │ Provider → causal judgment        │
│ evidence               │ Dict[str, str]   │ Provider → quoted evidence        │
└────────────────────────┴──────────────────┴────────────────────────────────────┘

Example Output:
```python
ValidationResult(
    item_id="C-0001",
    scores={"openai": 4.5, "anthropic": 5.0, "gemini": 4.0},
    avg_score=4.5,
    is_valid=True,
    feedback={
        "openai": "Correctly distinguishes association from causation",
        "anthropic": "Excellent causal reasoning",
        "gemini": "Good explanation"
    },
    ground_truth_match=True,
    agreement_score=1,
    consensus_judgment="ASSOCIATIVE",
    overclaim_consensus=False,
    causal_judgments={"openai": "ASSOCIATIVE", "anthropic": "ASSOCIATIVE", "gemini": "ASSOCIATIVE"},
    evidence={"openai": "'associated with' language used", ...}
)
```

================================================================================
VALIDATION CRITERIA
================================================================================

Quality Dimensions (scored 1-5):
┌─────────────────────────┬──────────────────────────────────────────────────────┐
│ Dimension               │ What it Measures                                     │
├─────────────────────────┼──────────────────────────────────────────────────────┤
│ ACCURACY                │ Is the answer scientifically correct?                │
│ CLARITY                 │ Is the question clear and unambiguous?               │
│ COMPLETENESS            │ Does the answer fully address the question?          │
│ REASONING               │ Does the explanation provide sound reasoning?        │
│ SCIENTIFIC_VALIDITY     │ Is it consistent with biomedical knowledge?          │
└─────────────────────────┴──────────────────────────────────────────────────────┘

TAXONOMY-SPECIFIC CRITERIA:

┌──────────┬──────────────────────────────────────────────────────────────────────┐
│ Taxonomy │ Extra Criteria                                                       │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│ C        │ ⚠️ CAUSAL FAITHFULNESS (CRITICAL)                                    │
│          │ • Correctly distinguishes ASSOCIATION from CAUSATION                │
│          │ • Avoids overclaiming causality from GWAS evidence                  │
│          │ • Mentions limitations (confounding, reverse causation)              │
│          │ • Preserves uncertainty language ("may", "suggests")                │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│ R        │ RISK INTERPRETATION                                                  │
│          │ • Odds ratio interpretation correct                                  │
│          │ • Risk level classification appropriate                              │
│          │ • Avoids deterministic language ("will cause" vs "increased risk")  │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│ M        │ SEMANTIC FAITHFULNESS                                                │
│          │ • Correct biomedical entities identified                             │
│          │ • Relationship type correctly extracted                              │
│          │ • Preserves original relationship strength from source               │
├──────────┼──────────────────────────────────────────────────────────────────────┤
│ S        │ STRUCTURE ACCURACY                                                   │
│          │ • Gene/SNP mapping correct                                           │
│          │ • Chromosomal locations accurate                                     │
│          │ • Standard nomenclature used                                         │
└──────────┴──────────────────────────────────────────────────────────────────────┘

================================================================================
VALIDATION THRESHOLDS
================================================================================

ValidatorConfig Parameters:
┌─────────────────────────┬─────────────┬──────────────────────────────────────────┐
│ Parameter               │ Default     │ Description                              │
├─────────────────────────┼─────────────┼──────────────────────────────────────────┤
│ min_score               │ 4.0         │ Minimum acceptable score                 │
│ max_score               │ 5.0         │ Maximum possible score                   │
│ passing_threshold       │ 4.0         │ Score required to pass                   │
│ require_majority        │ True        │ Majority of validators must pass?        │
│ min_validators          │ 2           │ Minimum number of validators required    │
└─────────────────────────┴─────────────┴──────────────────────────────────────────┘

Item Fails Validation If:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. Average score < 4.0                                                          │
│ 2. Majority of validators give score < 4.0 (if require_majority=True)           │
│ 3. Overclaim consensus = True (majority detect overclaim)                       │
│ 4. Ground truth mismatch (if ground_truth provided)                             │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
CONSENSUS CALCULATION
================================================================================

Causal Judgment Consensus:
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Collect judgments from all validators                                   │
│         {"openai": "ASSOCIATIVE", "anthropic": "ASSOCIATIVE", "gemini": "CAUSAL"}│
│                                                                                 │
│ Step 2: Calculate agreement                                                     │
│         - All same → agreement_score = 1                                        │
│         - Any different → agreement_score = 0                                   │
│                                                                                 │
│ Step 3: Determine consensus (majority vote)                                     │
│         - 2 ASSOCIATIVE, 1 CAUSAL → consensus = "ASSOCIATIVE"                  │
│                                                                                 │
│ Step 4: Overclaim consensus                                                     │
│         - Count is_overclaim=True votes                                         │
│         - Majority True → overclaim_consensus = True → FAIL                    │
└─────────────────────────────────────────────────────────────────────────────────┘

================================================================================
BEST PRACTICES
================================================================================

★ API KEY SETUP:
  Provide at least 2 API keys for reliable consensus:
  ```python
  validator = MultiLLMValidator(
      openai_api_key=os.environ.get("OPENAI_API_KEY"),
      anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
      gemini_api_key=os.environ.get("GEMINI_API_KEY")
  )
  ```

★ BATCH PROCESSING:
  Use validate_batch() for efficiency with progress tracking:
  ```python
  valid, invalid = validator.validate_batch(
      items,
      progress_callback=lambda done, total: print(f"{done}/{total}")
  )
  ```

★ ANALYZING FAILURES:
  Check feedback for failed items to understand issues:
  ```python
  for item in invalid_items:
      if item.get('overclaim_consensus'):
          print(f"Overclaim detected: {item['id']}")
      if item.get('agreement_score') == 0:
          print(f"Validators disagreed: {item['causal_judgments']}")
  ```

★ TAXONOMY C SPECIAL HANDLING:
  Pay extra attention to Causal taxonomy validation:
  - Check consensus_judgment for ASSOCIATIVE vs CAUSAL
  - Review items where overclaim_consensus = True
  - Examine evidence quotes for causal language detection

================================================================================
USAGE EXAMPLES
================================================================================

Example 1: Single Item Validation
```python
from bioreasonc_creator.validator import MultiLLMValidator

validator = MultiLLMValidator(
    openai_api_key="sk-...",
    anthropic_api_key="sk-ant-..."
)

result = validator.validate_item(
    item_id="C-0001",
    question="Is the relationship between GCK and T2D causal?",
    answer="The relationship is associative, not causal...",
    explanation="",
    taxonomy="C",
    label="C-CAUSAL-VS-ASSOC"
)

print(f"Valid: {result.is_valid}, Score: {result.avg_score:.1f}")
print(f"Consensus: {result.consensus_judgment}")
```

Example 2: Batch Validation
```python
items = [...]  # List from Generator
valid_items, invalid_items = validator.validate_batch(items)
stats = validator.get_stats(valid_items, invalid_items)
print(f"Validation rate: {stats['validation_rate']:.1%}")
```

Example 3: Custom Threshold
```python
from bioreasonc_creator.validator import ValidatorConfig

config = ValidatorConfig(
    passing_threshold=3.5,  # Lower threshold
    require_majority=False  # Any score >= threshold passes
)
validator = MultiLLMValidator(config=config, ...)
```

================================================================================
Author: Sakhaa Alsaedi, sakhaa.alsaedi@kaust.edu.sa
================================================================================
"""

import json
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import centralized prompts
from .prompts import ValidatorPrompts, TAXONOMY_DESCRIPTIONS, CORE_QUESTION

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

HAS_GEMINI = False
genai = None
try:
    import google.generativeai as genai_module
    # Check if the module has GenerativeModel
    if hasattr(genai_module, 'GenerativeModel'):
        genai = genai_module
        HAS_GEMINI = True
except ImportError:
    pass
except Exception:
    pass


class ValidationCriteria(Enum):
    """Validation criteria for Q&A quality assessment."""
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    REASONING = "reasoning"
    SCIENTIFIC_VALIDITY = "scientific_validity"


@dataclass
class ValidationResult:
    """Result of multi-LLM validation with consensus scoring."""
    item_id: str
    scores: Dict[str, float]  # Provider -> score
    avg_score: float
    is_valid: bool
    feedback: Dict[str, str]  # Provider -> feedback
    ground_truth_match: Optional[bool] = None
    # Consensus-based causal judgment fields
    agreement_score: int = 0  # 1 if all validators agree, 0 if disagree
    consensus_judgment: str = "NOT_APPLICABLE"  # ASSOCIATIVE | CAUSAL | NOT_APPLICABLE
    overclaim_consensus: bool = False  # True if majority say overclaim
    causal_judgments: Dict[str, str] = None  # Provider -> causal judgment
    evidence: Dict[str, str] = None  # Provider -> evidence quoted


@dataclass
class ValidatorConfig:
    """Configuration for the validator."""
    min_score: float = 4.0
    max_score: float = 5.0
    passing_threshold: float = 4.0
    require_majority: bool = True
    min_validators: int = 2


class MultiLLMValidator:
    """
    Multi-LLM validator using OpenAI, Anthropic, and Gemini
    to assess Q&A quality with majority voting.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        config: Optional[ValidatorConfig] = None
    ):
        """
        Initialize the multi-LLM validator.

        Args:
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            gemini_api_key: Google Gemini API key
            config: Validator configuration
        """
        self.config = config or ValidatorConfig()

        # Initialize clients
        self.validators = {}

        if openai_api_key and HAS_OPENAI:
            self.validators['openai'] = OpenAI(api_key=openai_api_key)

        if anthropic_api_key and HAS_ANTHROPIC:
            self.validators['anthropic'] = Anthropic(api_key=anthropic_api_key)

        if gemini_api_key and HAS_GEMINI and genai is not None:
            try:
                genai.configure(api_key=gemini_api_key)
                self.validators['gemini'] = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                print(f"Warning: Could not initialize Gemini: {e}")

        if len(self.validators) < self.config.min_validators:
            print(f"Warning: Only {len(self.validators)} validators available "
                  f"(minimum recommended: {self.config.min_validators})")

    def _get_validation_prompt(
        self,
        question: str,
        answer: str,
        explanation: str,
        taxonomy: str,
        label: str,
        ground_truth: Optional[str] = None
    ) -> str:
        """
        Generate validation prompt for LLMs using centralized prompts.

        Uses ValidatorPrompts.get_validation_prompt() which includes:
        - Extra causal faithfulness criteria for C taxonomy
        - Risk interpretation criteria for R taxonomy
        - Standard 1-5 scoring rubric
        """
        return ValidatorPrompts.get_validation_prompt(
            question=question,
            answer=answer,
            explanation=explanation,
            taxonomy=taxonomy,
            label=label,
            ground_truth=ground_truth
        )

    def _parse_score_response(self, response: str) -> Dict[str, Any]:
        """
        Parse score, causal judgment, and evidence from LLM response.

        Returns dict with:
        - score: 1-5 quality score
        - causal_judgment: ASSOCIATIVE | CAUSAL | NOT_APPLICABLE
        - is_overclaim: true/false
        - evidence: quoted evidence from answer
        - feedback: explanation
        """
        result = {
            'score': 3.0,
            'causal_judgment': 'NOT_APPLICABLE',
            'is_overclaim': False,
            'evidence': '',
            'feedback': 'Could not parse response'
        }

        try:
            # Try to extract JSON (handle multi-line JSON)
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result['score'] = min(max(float(data.get('score', 3)), 1), 5)
                result['causal_judgment'] = data.get('causal_judgment', 'NOT_APPLICABLE')
                result['is_overclaim'] = data.get('is_overclaim', False)
                result['evidence'] = data.get('evidence', '')
                result['feedback'] = data.get('feedback', 'No feedback provided')
                return result
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract score directly
        score_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/\s*5)?', response)
        if score_match:
            result['score'] = min(max(float(score_match.group(1)), 1), 5)
            result['feedback'] = response[:200]

        return result

    def _calculate_consensus(self, judgments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate consensus score from multiple LLM judgments.

        Returns:
        - agreement_score: 1 if all agree, 0 if disagree
        - consensus_judgment: the agreed-upon judgment (or majority)
        - overclaim_consensus: True if majority say overclaim
        """
        if not judgments:
            return {'agreement_score': 0, 'consensus_judgment': 'UNKNOWN', 'overclaim_consensus': False}

        # Get causal judgments (excluding NOT_APPLICABLE)
        causal_judgments = [j['causal_judgment'] for j in judgments
                           if j['causal_judgment'] != 'NOT_APPLICABLE']

        # Get overclaim judgments
        overclaim_votes = [j['is_overclaim'] for j in judgments]

        # Calculate agreement
        if len(causal_judgments) == 0:
            # All NOT_APPLICABLE - full agreement
            agreement_score = 1
            consensus_judgment = 'NOT_APPLICABLE'
        elif len(set(causal_judgments)) == 1:
            # All validators agree
            agreement_score = 1
            consensus_judgment = causal_judgments[0]
        else:
            # Disagreement - use majority vote
            agreement_score = 0
            from collections import Counter
            consensus_judgment = Counter(causal_judgments).most_common(1)[0][0]

        # Overclaim consensus (majority vote)
        overclaim_consensus = sum(overclaim_votes) > len(overclaim_votes) / 2

        return {
            'agreement_score': agreement_score,
            'consensus_judgment': consensus_judgment,
            'overclaim_consensus': overclaim_consensus,
            'num_validators': len(judgments),
            'judgments': [j['causal_judgment'] for j in judgments]
        }

    def _validate_with_openai(
        self,
        question: str,
        answer: str,
        explanation: str,
        taxonomy: str,
        label: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate using OpenAI with centralized prompts."""
        client = self.validators.get('openai')
        if not client:
            return {'score': 0.0, 'feedback': "OpenAI not available",
                    'causal_judgment': 'NOT_APPLICABLE', 'is_overclaim': False, 'evidence': ''}

        try:
            prompt = self._get_validation_prompt(
                question, answer, explanation, taxonomy, label, ground_truth
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": ValidatorPrompts.SYSTEM_PROMPT
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            return self._parse_score_response(response.choices[0].message.content)
        except Exception as e:
            return {'score': 0.0, 'feedback': f"OpenAI error: {str(e)}",
                    'causal_judgment': 'NOT_APPLICABLE', 'is_overclaim': False, 'evidence': ''}

    def _validate_with_anthropic(
        self,
        question: str,
        answer: str,
        explanation: str,
        taxonomy: str,
        label: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate using Anthropic with centralized prompts."""
        client = self.validators.get('anthropic')
        if not client:
            return {'score': 0.0, 'feedback': "Anthropic not available",
                    'causal_judgment': 'NOT_APPLICABLE', 'is_overclaim': False, 'evidence': ''}

        try:
            prompt = self._get_validation_prompt(
                question, answer, explanation, taxonomy, label, ground_truth
            )

            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=300,
                temperature=0.1,
                system=ValidatorPrompts.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )

            return self._parse_score_response(response.content[0].text)
        except Exception as e:
            return {'score': 0.0, 'feedback': f"Anthropic error: {str(e)}",
                    'causal_judgment': 'NOT_APPLICABLE', 'is_overclaim': False, 'evidence': ''}

    def _validate_with_gemini(
        self,
        question: str,
        answer: str,
        explanation: str,
        taxonomy: str,
        label: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate using Google Gemini with centralized prompts."""
        model = self.validators.get('gemini')
        if not model:
            return {'score': 0.0, 'feedback': "Gemini not available",
                    'causal_judgment': 'NOT_APPLICABLE', 'is_overclaim': False, 'evidence': ''}

        try:
            prompt = self._get_validation_prompt(
                question, answer, explanation, taxonomy, label, ground_truth
            )
            # Prepend system context for Gemini
            full_prompt = f"{ValidatorPrompts.SYSTEM_PROMPT}\n\n{prompt}"

            response = model.generate_content(
                full_prompt,
                generation_config={
                    'temperature': 0.1,
                    'max_output_tokens': 300
                }
            )

            return self._parse_score_response(response.text)
        except Exception as e:
            return {'score': 0.0, 'feedback': f"Gemini error: {str(e)}",
                    'causal_judgment': 'NOT_APPLICABLE', 'is_overclaim': False, 'evidence': ''}

    def _check_ground_truth_match(
        self,
        answer: str,
        ground_truth
    ) -> bool:
        """Check if answer matches ground truth."""
        if not ground_truth:
            return True

        # Handle dict ground_truth (from GeneratedItem)
        if isinstance(ground_truth, dict):
            gt_answer = ground_truth.get('answer') or ground_truth.get('answer_normalized', '')
        else:
            gt_answer = str(ground_truth)

        if not gt_answer:
            return True

        # Normalize both strings
        answer_normalized = answer.lower().strip()
        gt_normalized = gt_answer.lower().strip()

        # Direct match
        if answer_normalized == gt_normalized:
            return True

        # Check if ground truth key terms are in answer
        gt_terms = set(gt_normalized.split())
        answer_terms = set(answer_normalized.split())

        # At least 50% of ground truth terms should be in answer
        overlap = gt_terms.intersection(answer_terms)
        if len(overlap) >= len(gt_terms) * 0.5:
            return True

        return False

    def validate_item(
        self,
        item_id: str,
        question: str,
        answer: str,
        explanation: str,
        taxonomy: str,
        label: str,
        ground_truth: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a single Q&A item using multiple LLMs with consensus scoring.

        Args:
            item_id: Item identifier
            question: Question text
            answer: Answer text
            explanation: Explanation text
            taxonomy: Taxonomy category
            label: Question label
            ground_truth: Optional ground truth answer

        Returns:
            ValidationResult with scores, feedback, and consensus from all validators
        """
        scores = {}
        feedback = {}
        causal_judgments = {}
        evidence = {}
        all_judgments = []

        # Validate with each available provider
        if 'openai' in self.validators:
            result = self._validate_with_openai(
                question, answer, explanation, taxonomy, label, ground_truth
            )
            if result['score'] > 0:
                scores['openai'] = result['score']
                feedback['openai'] = result['feedback']
                causal_judgments['openai'] = result['causal_judgment']
                evidence['openai'] = result['evidence']
                all_judgments.append(result)

        if 'anthropic' in self.validators:
            result = self._validate_with_anthropic(
                question, answer, explanation, taxonomy, label, ground_truth
            )
            if result['score'] > 0:
                scores['anthropic'] = result['score']
                feedback['anthropic'] = result['feedback']
                causal_judgments['anthropic'] = result['causal_judgment']
                evidence['anthropic'] = result['evidence']
                all_judgments.append(result)

        if 'gemini' in self.validators:
            result = self._validate_with_gemini(
                question, answer, explanation, taxonomy, label, ground_truth
            )
            if result['score'] > 0:
                scores['gemini'] = result['score']
                feedback['gemini'] = result['feedback']
                causal_judgments['gemini'] = result['causal_judgment']
                evidence['gemini'] = result['evidence']
                all_judgments.append(result)

        # Calculate average score
        valid_scores = [s for s in scores.values() if s > 0]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # Calculate consensus for causal judgments
        consensus = self._calculate_consensus(all_judgments)

        # Determine if valid based on:
        # 1. Average score threshold
        # 2. Majority passing (if required)
        # 3. No overclaim consensus
        is_valid = avg_score >= self.config.passing_threshold

        if self.config.require_majority and len(valid_scores) >= 2:
            passing_count = sum(1 for s in valid_scores if s >= self.config.passing_threshold)
            is_valid = passing_count > len(valid_scores) / 2

        # If consensus says overclaim, mark as invalid
        if consensus['overclaim_consensus']:
            is_valid = False

        # Check ground truth match if provided
        gt_match = None
        if ground_truth:
            gt_match = self._check_ground_truth_match(answer, ground_truth)
            # Penalize if ground truth doesn't match
            if not gt_match:
                is_valid = False

        return ValidationResult(
            item_id=item_id,
            scores=scores,
            avg_score=avg_score,
            is_valid=is_valid,
            feedback=feedback,
            ground_truth_match=gt_match,
            # Consensus-based fields
            agreement_score=consensus['agreement_score'],
            consensus_judgment=consensus['consensus_judgment'],
            overclaim_consensus=consensus['overclaim_consensus'],
            causal_judgments=causal_judgments,
            evidence=evidence
        )

    def validate_batch(
        self,
        items: List[Dict[str, Any]],
        progress_callback: Optional[callable] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate a batch of items.

        Args:
            items: List of items to validate
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (valid_items, invalid_items) with validation results added
        """
        valid_items = []
        invalid_items = []
        total = len(items)

        for idx, item in enumerate(items):
            # Validate item
            result = self.validate_item(
                item_id=item.get('id', f'item_{idx}'),
                question=item.get('question', ''),
                answer=item.get('answer', ''),
                explanation=item.get('explanation', ''),
                taxonomy=item.get('taxonomy', 'U'),
                label=item.get('label', 'UNKNOWN'),
                ground_truth=item.get('ground_truth')
            )

            # Add validation results to item (including consensus fields)
            validated_item = dict(item)
            validated_item['scores'] = result.scores
            validated_item['validation_scores'] = list(result.scores.values())
            validated_item['validation_providers'] = list(result.scores.keys())
            validated_item['avg_score'] = result.avg_score
            validated_item['is_valid'] = result.is_valid
            validated_item['feedback'] = result.feedback
            validated_item['validation_feedback'] = result.feedback
            # Add consensus fields
            validated_item['agreement_score'] = result.agreement_score
            validated_item['consensus_judgment'] = result.consensus_judgment
            validated_item['overclaim_consensus'] = result.overclaim_consensus
            validated_item['causal_judgments'] = result.causal_judgments
            validated_item['evidence'] = result.evidence
            if result.ground_truth_match is not None:
                validated_item['ground_truth_match'] = result.ground_truth_match

            if result.is_valid:
                valid_items.append(validated_item)
            else:
                invalid_items.append(validated_item)

            if progress_callback:
                progress_callback(idx + 1, total)

        return valid_items, invalid_items

    def get_stats(
        self,
        valid_items: List[Dict[str, Any]],
        invalid_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get validation statistics."""
        all_items = valid_items + invalid_items
        total = len(all_items)

        if total == 0:
            return {'error': 'No items to analyze'}

        all_scores = [item.get('avg_score', 0) for item in all_items]

        # Calculate provider-specific stats
        provider_stats = {}
        for provider in ['openai', 'anthropic', 'gemini']:
            provider_scores = []
            for item in all_items:
                scores = item.get('validation_scores', [])
                providers = item.get('validation_providers', [])
                if provider in providers:
                    idx = providers.index(provider)
                    if idx < len(scores):
                        provider_scores.append(scores[idx])

            if provider_scores:
                provider_stats[provider] = {
                    'avg_score': sum(provider_scores) / len(provider_scores),
                    'num_validations': len(provider_scores)
                }

        # Ground truth stats
        gt_matches = [item.get('ground_truth_match') for item in all_items
                      if item.get('ground_truth_match') is not None]

        return {
            'total_items': total,
            'valid_items': len(valid_items),
            'invalid_items': len(invalid_items),
            'validation_rate': len(valid_items) / total if total > 0 else 0,
            'avg_score': sum(all_scores) / len(all_scores) if all_scores else 0,
            'min_score': min(all_scores) if all_scores else 0,
            'max_score': max(all_scores) if all_scores else 0,
            'provider_stats': provider_stats,
            'ground_truth_match_rate': sum(gt_matches) / len(gt_matches) if gt_matches else None,
            'by_taxonomy': {
                taxonomy: {
                    'valid': sum(1 for i in valid_items if i.get('taxonomy') == taxonomy),
                    'invalid': sum(1 for i in invalid_items if i.get('taxonomy') == taxonomy)
                }
                for taxonomy in ['S', 'C', 'R', 'M']
            }
        }
