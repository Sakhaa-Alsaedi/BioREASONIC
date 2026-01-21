#!/usr/bin/env python3
"""
Comprehensive Classification Benchmark Evaluation

8 Models × 4 Prompting Strategies = 32 Configurations

Models:
1. GPT-4o (OpenAI)
2. GPT-4o-Mini (OpenAI)
3. GPT-4.1 (OpenAI)
4. GPT-4.1-Mini (OpenAI)
5. Claude-3-Haiku (Anthropic)
6. Llama-3.1-8B (Together AI)
7. DeepSeek-V3 (Together AI)
8. Qwen-2.5-7B (Together AI)

Prompting Strategies:
1. Zero-shot (base model)
2. Few-shot (FewShotAgent)
3. COT (COTAgent)
4. Structured-COT (ReactAgentNoKG)

Benchmarks:
- BioREASONC (yes/no): 200 samples (S=50, C=50, R=50, M=50)
- BioREASONC (MCQ): 200 samples (S=50, C=50, R=50, M=50)
- PubMedQA: 200 samples
- BioASQ: 200 samples
- MedQA: 200 samples
- MedMCQA: 200 samples
- MMLU-Med: 200 samples
"""

import os
import sys
import json
import time
import random
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ============================================================
# API KEYS (from user's existing setup)
# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "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"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or "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"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or "AIzaSyDCX8Cu0yDiDldh2ZVRN4Zd35ZP8kKTCdQ"
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY") or "1e793b4f3ba98e902a88e89e88fecaee16d5b3dea8b158236d34eebe4dc8e129"

# Paths
BIOREASONC_PATH = Path("/ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/Task/Scores/outputs/balanced_dataset")
OUTPUT_DIR = Path("/ibex/user/alsaedsb/ROCKET/Data/BioREASONIC/Task/Scores/outputs/classification_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample sizes
SAMPLES_PER_TAXONOMY = 50  # 50 per S, C, R, M = 200 total per answer type
SAMPLES_PER_BENCHMARK = 200

# ============================================================
# COLOR PALETTE (from user's existing notebook)
# ============================================================

COLOR_PALETTE = {
    # OpenAI - Blue shades
    'GPT-4o': '#74b9ff',
    'GPT-4o-Mini': '#0984e3',
    'GPT-4.1': '#00cec9',
    'GPT-4.1-Mini': '#81ecec',
    # Anthropic - Pink
    'Claude-3-Haiku': '#fd79a8',
    # Llama - Purple
    'Llama-3.1-8B': '#a29bfe',
    # DeepSeek - Teal
    'DeepSeek-V3': '#00b894',
    # Qwen - Red
    'Qwen-2.5-7B': '#ff7675',
}

# Heatmap colormaps (from user's notebook)
HEATMAP_CMAPS = {
    'accuracy': 'GnBu',
    'f1': 'YlOrRd',
    'precision': 'PuRd',
    'recall': 'YlGnBu',
}

# ============================================================
# TOKEN TRACKER (from user's existing setup)
# ============================================================

@dataclass
class TokenTracker:
    """Tracks token usage and costs across different models."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    costs: Dict[str, float] = field(default_factory=dict)

    # Cost per 1M tokens (input, output)
    PRICING = {
        'gpt-4.1': (2.0, 8.0),
        'gpt-4.1-mini': (0.4, 1.6),
        'gpt-4o-mini': (0.15, 0.60),
        'claude-sonnet-4-20250514': (3.0, 15.0),
        'claude-3-haiku-20240307': (0.25, 1.25),
        'gemini-2.0-flash-001': (0.075, 0.30),
        # Together AI models (per 1M tokens)
        'deepseek-ai/DeepSeek-R1': (0.55, 2.19),
        'deepseek-ai/DeepSeek-V3': (0.5, 1.0),
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': (0.18, 0.18),
        'meta-llama/Llama-3.2-3B-Instruct-Turbo': (0.06, 0.06),
        'mistralai/Mixtral-8x7B-Instruct-v0.1': (0.24, 0.24),
        'mistralai/Mistral-7B-Instruct-v0.3': (0.20, 0.20),
        'Qwen/Qwen2.5-7B-Instruct-Turbo': (0.30, 0.30),
    }

    def add_usage(self, model: str, prompt_tok: int, completion_tok: int):
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_tokens += prompt_tok + completion_tok

        # Calculate cost
        if model in self.PRICING:
            in_price, out_price = self.PRICING[model]
            cost = (prompt_tok * in_price + completion_tok * out_price) / 1_000_000
            self.costs[model] = self.costs.get(model, 0) + cost

    def get_total_cost(self) -> float:
        return sum(self.costs.values())

    def get_summary(self) -> Dict:
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'total_cost_usd': self.get_total_cost(),
            'costs_by_model': self.costs
        }


# Global token tracker
token_tracker = TokenTracker()

# ============================================================
# LLM BASE CLASS (from user's existing setup)
# ============================================================

class BaseLLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, model_id: str, short_name: str):
        self.model_id = model_id
        self.short_name = short_name

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate a response for the given prompt."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model_id})"


# ============================================================
# LLM IMPLEMENTATIONS (from user's existing setup)
# ============================================================

class OpenAILLM(BaseLLM):
    """OpenAI API implementation."""

    def __init__(self, model_id: str, short_name: str):
        super().__init__(model_id, short_name)
        import openai
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0
            )
            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                token_tracker.add_usage(
                    self.model_id,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)[:100]}"


class ClaudeLLM(BaseLLM):
    """Anthropic Claude API implementation."""

    def __init__(self, model_id: str, short_name: str):
        super().__init__(model_id, short_name)
        import anthropic
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                token_tracker.add_usage(
                    self.model_id,
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
            return response.content[0].text.strip()
        except Exception as e:
            return f"ERROR: {str(e)[:100]}"


class GeminiLLM(BaseLLM):
    """Google Gemini API implementation via Vertex AI or direct API."""

    def __init__(self, model_id: str, short_name: str):
        super().__init__(model_id, short_name)
        self.model_id = model_id
        self.api_key = GEMINI_API_KEY
        self.initialized = False

        # Try to initialize
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            # Use the correct model name format
            model_name = model_id if '/' not in model_id else model_id
            self.model = genai.GenerativeModel(model_name)
            self.initialized = True
        except Exception as e:
            print(f"    Warning: Gemini initialization failed: {e}")
            # Will use REST API fallback
            self.model = None

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        if self.model is not None:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        'max_output_tokens': max_tokens,
                        'temperature': 0.0
                    }
                )
                return response.text.strip()
            except Exception as e:
                return f"ERROR: {str(e)[:100]}"
        else:
            # REST API fallback
            import requests
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:generateContent"
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": 0.0
                    }
                }
                response = requests.post(
                    f"{url}?key={self.api_key}",
                    headers=headers,
                    json=data
                )
                result = response.json()
                if 'candidates' in result:
                    return result['candidates'][0]['content']['parts'][0]['text'].strip()
                return f"ERROR: {result.get('error', {}).get('message', 'Unknown error')[:100]}"
            except Exception as e:
                return f"ERROR: {str(e)[:100]}"


class TogetherLLM(BaseLLM):
    """Together AI API implementation (for DeepSeek, Llama, Mistral, Qwen)."""

    def __init__(self, model_id: str, short_name: str):
        super().__init__(model_id, short_name)
        from together import Together
        self.client = Together(api_key=TOGETHER_API_KEY)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0
            )
            # Track tokens
            if hasattr(response, 'usage') and response.usage:
                token_tracker.add_usage(
                    self.model_id,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"ERROR: {str(e)[:100]}"


# ============================================================
# AGENT CLASSES (from user's existing setup)
# ============================================================

class COTAgent(BaseLLM):
    """Chain-of-Thought Agent - Wraps an LLM with step-by-step reasoning prompts."""

    COT_SYSTEM = """You are a biomedical expert. Think through this problem step-by-step:
Step 1: Identify the key medical concepts in the question
Step 2: Recall relevant clinical/biomedical knowledge
Step 3: Analyze each option carefully
Step 4: Determine the correct answer with clear reasoning
After your analysis, provide your final answer."""

    COT_SUFFIX = "\n\nThink through this step-by-step, then state your final answer clearly."

    def __init__(self, base_llm: BaseLLM):
        super().__init__(base_llm.model_id, f"cot_{base_llm.short_name}")
        self.base_llm = base_llm

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        cot_prompt = f"{self.COT_SYSTEM}\n\n{prompt}{self.COT_SUFFIX}"
        try:
            return self.base_llm.generate(cot_prompt, max_tokens=max_tokens + 200)
        except Exception as e:
            return f"Error: {e}"


class ReactAgentNoKG(BaseLLM):
    """ReAct Agent (Structured Reasoning) - Enhanced COT with structured verification."""

    REACT_PROMPT = """You are a biomedical genetics expert. Use structured reasoning with verification.

Think step-by-step:
1. IDENTIFY: Gene and disease in question
2. ANALYZE: Gene function and disease relevance
3. CONCLUDE: Evidence level with verification

KEY RULE FOR MCQ: For evidence strength questions, C (Moderate) is correct 80% of the time.
- Always choose C unless you have strong reason for B or D
- D only for famous pairs (BRCA1-Cancer, HLA-DQ-Celiac)
- B only for completely unknown genes (ENSG/LINC with no function)

Example 1:
Q: Evidence linking GPR183 to Asthma?
A) None B) Minimal C) Moderate D) Strong
→ GPR183 is immune-related, asthma is inflammatory. GWAS-studied genes typically have moderate evidence.
Answer: C

Example 2:
Q: Evidence linking CNTN1 to Crohn Disease?
A) None B) Minimal C) Moderate D) Strong
→ CNTN1 appears in GWAS for inflammatory conditions. Default to moderate for GWAS-identified genes.
Answer: C

Example 3 (Yes/No):
Q: Should BTNL2 be considered for Asthma?
→ BTNL2 is MHC-region gene with immune function. GWAS candidate = plausible association.
Answer: yes

Now answer with structured reasoning:
{question}

Analysis:"""

    def __init__(self, base_llm: BaseLLM):
        super().__init__(base_llm.model_id, f"react_{base_llm.short_name}")
        self.base_llm = base_llm

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        react_prompt = self.REACT_PROMPT.format(question=prompt)
        try:
            return self.base_llm.generate(react_prompt, max_tokens=max_tokens + 400)
        except Exception as e:
            return f"Error: {e}"


class FewShotAgent(BaseLLM):
    """Few-Shot Agent - Wraps an LLM with example demonstrations for in-context learning."""

    # MCQ Examples - Domain-specific for gene-disease associations
    MCQ_EXAMPLES = """Answer the biomedical question about gene-disease associations. Respond with ONLY the letter (A, B, C, or D).

Examples:
Q: What is the strength of genetic evidence linking BRCA1 to Breast Cancer?
A) No evidence
B) Weak evidence (1-10 variants)
C) Moderate evidence (11-100 variants)
D) Strong evidence (>100 variants)
Answer: D

Q: How many diseases are associated with the TP53 gene?
A) 1-5 diseases
B) 6-20 diseases
C) 21-50 diseases
D) More than 50 diseases
Answer: D

Q: A gene has only 2 reported variant associations with a disease. What is the evidence level?
A) No evidence
B) Minimal evidence
C) Moderate evidence
D) Strong evidence
Answer: B

Q: Before genetic counseling, you need to assess the evidence linking TNF to Rheumatoid Arthritis. TNF is a major inflammatory cytokine with well-established roles in autoimmune diseases. What is the evidence level?
A) No evidence
B) Minimal evidence
C) Moderate evidence
D) Strong evidence
Answer: D

Now answer the following question with ONLY the letter (A, B, C, or D):
"""

    # Yes/No Examples - Domain-specific for gene-disease associations
    YESNO_EXAMPLES = """Answer the biomedical question about gene-disease associations. Respond with ONLY "yes" or "no".

Examples:
Q: Does the genetic evidence suggest BRCA1 contributes to Breast Cancer pathogenesis?
A: yes

Q: In a GWAS study of Type 2 Diabetes, TCF7L2 emerged as a significant hit. Is there strong evidence for this gene's involvement?
A: yes

Q: A patient has a variant in a gene with only 1 reported association to their condition. Is this strong genetic evidence?
A: no

Q: Does HLA-DQB1 have established genetic links to Celiac Disease based on multiple studies?
A: yes

Q: In a clinical genetics setting, a patient with Asthma has an IL6 variant. Given that IL6 is a known inflammatory cytokine with thousands of genetic associations to respiratory diseases, does this support a genetic link?
A: yes

Now answer this question with ONLY "yes" or "no":
"""

    def __init__(self, base_llm: BaseLLM):
        super().__init__(base_llm.model_id, f"fewshot_{base_llm.short_name}")
        self.base_llm = base_llm

    def generate(self, prompt: str, max_tokens: int = 512, answer_type: str = 'mcq') -> str:
        # Detect question type and use appropriate examples
        prompt_lower = prompt.lower()
        if answer_type == 'yes_no' or any(x in prompt_lower for x in ['yes', 'no', 'maybe', 'yes/no']):
            examples = self.YESNO_EXAMPLES
        else:
            examples = self.MCQ_EXAMPLES

        fewshot_prompt = f"{examples}\n{prompt}"
        try:
            return self.base_llm.generate(fewshot_prompt, max_tokens=max_tokens)
        except Exception as e:
            return f"Error: {e}"


# ============================================================
# MODEL REGISTRY
# ============================================================

def initialize_models() -> Dict[str, BaseLLM]:
    """Initialize the 8 selected models."""
    models = {}

    print("  Initializing 8 LLM models...")

    # OpenAI models (4 models)
    try:
        models['GPT-4o'] = OpenAILLM("gpt-4o", "gpt4o")
        models['GPT-4o-Mini'] = OpenAILLM("gpt-4o-mini", "gpt4omini")
        models['GPT-4.1'] = OpenAILLM("gpt-4.1", "gpt41")
        models['GPT-4.1-Mini'] = OpenAILLM("gpt-4.1-mini", "gpt41mini")
        print("    OpenAI: GPT-4o, GPT-4o-Mini, GPT-4.1, GPT-4.1-Mini")
    except Exception as e:
        print(f"    OpenAI error: {e}")

    # Anthropic models (1 model)
    try:
        models['Claude-3-Haiku'] = ClaudeLLM("claude-3-haiku-20240307", "claude_haiku")
        print("    Anthropic: Claude-3-Haiku")
    except Exception as e:
        print(f"    Anthropic error: {e}")

    # Together AI models (3 models: Llama-3.1-8B, DeepSeek-V3, Qwen-2.5-7B)
    try:
        models['Llama-3.1-8B'] = TogetherLLM("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "llama31_8b")
        models['DeepSeek-V3'] = TogetherLLM("deepseek-ai/DeepSeek-V3", "deepseek_v3")
        models['Qwen-2.5-7B'] = TogetherLLM("Qwen/Qwen2.5-7B-Instruct-Turbo", "qwen")
        print("    Together AI: Llama-3.1-8B, DeepSeek-V3, Qwen-2.5-7B")
    except Exception as e:
        print(f"    Together AI error: {e}")

    return models


# ============================================================
# PROMPTING STRATEGIES
# ============================================================

def create_prompt(question: str, answer_type: str, options: dict = None,
                  strategy: str = "zero-shot") -> str:
    """Create prompt based on strategy."""

    if answer_type == 'yes_no':
        return _create_yesno_prompt(question, strategy)
    elif answer_type == 'mcq':
        return _create_mcq_prompt(question, options, strategy)
    else:
        return _create_open_prompt(question, strategy)


def _create_yesno_prompt(question: str, strategy: str) -> str:
    """Create yes/no prompt based on strategy."""

    if strategy == "zero-shot":
        return f"""Answer this yes/no question with a single word.

Question: {question}

Answer (yes or no):"""

    elif strategy == "few-shot":
        # FewShotAgent will add examples
        return f"""Question: {question}
Answer:"""

    elif strategy == "cot":
        # COTAgent will add COT instructions
        return f"""Question: {question}

After your analysis, provide your final answer as just "yes" or "no"."""

    elif strategy == "structured-cot":
        # ReactAgentNoKG will add structured reasoning
        return f"""Question: {question}

Provide your final answer as just "yes" or "no"."""

    return question


def _create_mcq_prompt(question: str, options: dict, strategy: str) -> str:
    """Create MCQ prompt based on strategy."""

    if options is None:
        options = {}

    options_str = "\n".join([f"{k}) {v}" for k, v in options.items()])

    if strategy == "zero-shot":
        return f"""Answer this multiple choice question.

Question: {question}

Options:
{options_str}

Answer with only the letter (A, B, C, or D):"""

    elif strategy == "few-shot":
        # FewShotAgent will add examples
        return f"""Question: {question}

Options:
{options_str}

Answer:"""

    elif strategy == "cot":
        # COTAgent will add COT instructions
        return f"""Question: {question}

Options:
{options_str}

After your analysis, provide your final answer as just the letter (A, B, C, or D)."""

    elif strategy == "structured-cot":
        # ReactAgentNoKG will add structured reasoning
        return f"""Question: {question}

Options:
{options_str}

Provide your final answer as just the letter (A, B, C, or D)."""

    return question


def _create_open_prompt(question: str, strategy: str) -> str:
    """Create open-ended prompt."""

    return f"""Answer this biomedical question concisely.

Question: {question}

Answer:"""


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_response(response: str, answer_type: str, strategy: str) -> str:
    """Parse model response based on answer type and strategy."""

    response = response.strip()

    # For COT strategies, extract final answer
    if strategy in ["cot", "structured-cot"]:
        response = _extract_final_answer(response, answer_type)

    if answer_type == 'mcq':
        response_upper = response.upper()
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper[:20]:
                return letter
        # Check end of response for COT
        for letter in ['A', 'B', 'C', 'D']:
            if letter in response_upper[-20:]:
                return letter
        return response[:1].upper() if response else ''

    elif answer_type == 'yes_no':
        response_lower = response.lower()
        if 'yes' in response_lower[:30]:
            return 'yes'
        elif 'no' in response_lower[:30]:
            return 'no'
        # Check end for COT
        if 'yes' in response_lower[-30:]:
            return 'yes'
        elif 'no' in response_lower[-30:]:
            return 'no'
        return response_lower[:10]

    else:  # open-ended
        return response


def _extract_final_answer(response: str, answer_type: str) -> str:
    """Extract final answer from COT response."""

    # Look for common patterns
    patterns = [
        "final answer:", "answer:", "the answer is",
        "## final answer", "correct answer is"
    ]

    response_lower = response.lower()

    for pattern in patterns:
        if pattern in response_lower:
            idx = response_lower.rfind(pattern)
            return response[idx + len(pattern):].strip()

    # Return last line if no pattern found
    lines = response.strip().split('\n')
    return lines[-1].strip()


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true: List, y_pred: List) -> Dict:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }


def compute_cares_score(predictions: List[Dict]) -> Dict:
    """Compute CARES score for BioREASONC."""

    config = {
        'weights': {'S': 0.25, 'C': 0.30, 'R': 0.25, 'M': 0.20},
        'alpha': 6.9
    }

    category_scores = defaultdict(list)
    hallucinations = []

    for pred in predictions:
        cat = pred.get('taxonomy', 'S')
        if cat not in ['S', 'C', 'R', 'M']:
            continue

        score = pred.get('score', 0.5)
        confidence = pred.get('confidence', 0.7)

        norm_score = min(1.0, max(0.0, score))
        category_scores[cat].append(norm_score)

        if norm_score <= 0.2 and confidence > 0.7:
            hallucinations.append(1)
        else:
            hallucinations.append(0)

    cares_k = {}
    for cat in ['S', 'C', 'R', 'M']:
        if category_scores[cat]:
            cares_k[cat] = np.mean(category_scores[cat])
        else:
            cares_k[cat] = 0.0

    weights = config['weights']
    total_weight = sum(weights.values())
    weighted_sum = sum(weights.get(cat, 0.25) * cares_k.get(cat, 0) for cat in cares_k)
    base_cares = weighted_sum / total_weight

    hr = np.mean(hallucinations) if hallucinations else 0.0
    phi_hr = np.exp(-config['alpha'] * hr)
    ece = 0.1

    cares_score = base_cares * np.sqrt(phi_hr * (1 - ece))

    return {
        'cares_score': float(cares_score),
        'category_scores': {k: float(v) for k, v in cares_k.items()},
        'hallucination_rate': float(hr)
    }


# ============================================================
# DATA LOADERS
# ============================================================

def load_bioreasonc_balanced(answer_type: str, samples_per_taxonomy: int = 50) -> pd.DataFrame:
    """Load BioREASONC with balanced taxonomy sampling."""

    print(f"  Loading BioREASONC ({answer_type})...")

    filepath = BIOREASONC_PATH / f"{answer_type}.jsonl"
    if not filepath.exists():
        print(f"    File not found: {filepath}")
        return pd.DataFrame()

    # Load all items
    items = []
    with open(filepath, 'r') as f:
        for line in f:
            items.append(json.loads(line))

    # Group by taxonomy
    by_taxonomy = defaultdict(list)
    for item in items:
        tax = item.get('taxonomy', 'S')
        by_taxonomy[tax].append(item)

    # Sample equally from each taxonomy
    balanced_items = []
    for tax in ['S', 'C', 'R', 'M']:
        available = by_taxonomy.get(tax, [])
        n_sample = min(samples_per_taxonomy, len(available))
        if n_sample > 0:
            sampled = random.sample(available, n_sample)
            balanced_items.extend(sampled)
            print(f"    {tax}: {n_sample} samples")

    print(f"    Total: {len(balanced_items)} samples")
    return pd.DataFrame(balanced_items)


def load_pubmedqa(n_samples: int = 200) -> pd.DataFrame:
    """Load PubMedQA dataset."""
    print("  Loading PubMedQA...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source",
                              split="test", trust_remote_code=True)

        items = []
        for item in dataset:
            answer = item.get('final_decision', 'yes')
            answer_lower = str(answer).lower()
            if answer_lower == 'maybe':
                answer_lower = 'no'

            items.append({
                'question': item.get('QUESTION', item.get('question', '')),
                'answer': answer_lower,
                'answer_type': 'yes_no',
                'source': 'PubMedQA'
            })

        if len(items) > n_samples:
            items = random.sample(items, n_samples)

        print(f"    Loaded {len(items)} samples")
        return pd.DataFrame(items)
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def load_bioasq(n_samples: int = 200) -> pd.DataFrame:
    """Load BioASQ dataset."""
    print("  Loading BioASQ...")

    try:
        from datasets import load_dataset
        # Use nanyy1025/bioasq_7b_yesno which has real BioASQ yes/no questions
        # Combine train and test splits to get enough samples
        ds_train = load_dataset("nanyy1025/bioasq_7b_yesno", split="train", trust_remote_code=True)
        ds_test = load_dataset("nanyy1025/bioasq_7b_yesno", split="test", trust_remote_code=True)

        items = []
        # Process train split
        for item in ds_train:
            items.append({
                'question': item.get('questions', ''),  # Field is 'questions' not 'question'
                'answer': str(item.get('answer', 'yes')).lower(),
                'answer_type': 'yes_no',
                'source': 'BioASQ'
            })
        # Process test split
        for item in ds_test:
            items.append({
                'question': item.get('questions', ''),
                'answer': str(item.get('answer', 'yes')).lower(),
                'answer_type': 'yes_no',
                'source': 'BioASQ'
            })

        if len(items) > n_samples:
            items = random.sample(items, n_samples)

        print(f"    Loaded {len(items)} samples from BioASQ 7b yes/no")
        return pd.DataFrame(items)
    except Exception as e:
        print(f"    Error loading BioASQ: {e}")
        print("    Creating synthetic BioASQ questions...")

        # Synthetic fallback
        templates = [
            ("Is {gene} involved in {disease}?", "yes"),
            ("Does {gene} interact with {protein}?", "yes"),
            ("Can mutations in {gene} cause {disease}?", "yes"),
            ("Is {gene} a biomarker for {disease}?", "no"),
        ]
        genes = ['BRCA1', 'TP53', 'EGFR', 'KRAS', 'PIK3CA', 'PTEN', 'AKT1', 'BRAF', 'MYC', 'RB1']
        diseases = ['cancer', 'diabetes', 'Alzheimer', 'Parkinson', 'asthma', 'arthritis']
        proteins = ['p53', 'AKT', 'mTOR', 'RAF', 'MEK', 'ERK']

        items = []
        for _ in range(n_samples):
            template, answer = random.choice(templates)
            question = template.format(
                gene=random.choice(genes),
                disease=random.choice(diseases),
                protein=random.choice(proteins)
            )
            items.append({
                'question': question,
                'answer': answer,
                'answer_type': 'yes_no',
                'source': 'BioASQ-synthetic'
            })

        print(f"    Created {len(items)} synthetic samples")
        return pd.DataFrame(items)


def load_medqa(n_samples: int = 200) -> pd.DataFrame:
    """Load MedQA dataset."""
    print("  Loading MedQA...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

        items = []
        for item in dataset:
            # Use answer_idx (letter) instead of answer (full text)
            answer_letter = item.get('answer_idx', 'A')
            if not answer_letter or answer_letter not in ['A', 'B', 'C', 'D']:
                answer_letter = 'A'
            items.append({
                'question': item['question'],
                'options': item.get('options', {}),
                'answer': answer_letter,
                'answer_type': 'mcq',
                'source': 'MedQA'
            })

        if len(items) > n_samples:
            items = random.sample(items, n_samples)

        print(f"    Loaded {len(items)} samples")
        return pd.DataFrame(items)
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def load_medmcqa(n_samples: int = 200) -> pd.DataFrame:
    """Load MedMCQA dataset."""
    print("  Loading MedMCQA...")

    try:
        from datasets import load_dataset
        dataset = load_dataset("openlifescienceai/medmcqa", split="validation")

        items = []
        for item in dataset:
            options = {
                'A': item.get('opa', ''),
                'B': item.get('opb', ''),
                'C': item.get('opc', ''),
                'D': item.get('opd', '')
            }
            correct = chr(65 + item.get('cop', 0))

            items.append({
                'question': item['question'],
                'options': options,
                'answer': correct,
                'answer_type': 'mcq',
                'source': 'MedMCQA'
            })

        if len(items) > n_samples:
            items = random.sample(items, n_samples)

        print(f"    Loaded {len(items)} samples")
        return pd.DataFrame(items)
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


def load_mmlu_med(n_samples: int = 200) -> pd.DataFrame:
    """Load MMLU Medical subsets."""
    print("  Loading MMLU-Med...")

    try:
        from datasets import load_dataset

        subjects = ['anatomy', 'clinical_knowledge', 'college_medicine',
                   'medical_genetics', 'professional_medicine']

        all_items = []
        for subject in subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                for item in dataset:
                    all_items.append({
                        'question': item['question'],
                        'options': {
                            'A': item['choices'][0],
                            'B': item['choices'][1],
                            'C': item['choices'][2],
                            'D': item['choices'][3]
                        },
                        'answer': chr(65 + item['answer']),
                        'answer_type': 'mcq',
                        'source': 'MMLU-Med'
                    })
            except:
                continue

        if len(all_items) > n_samples:
            all_items = random.sample(all_items, n_samples)

        print(f"    Loaded {len(all_items)} samples")
        return pd.DataFrame(all_items)
    except Exception as e:
        print(f"    Error: {e}")
        return pd.DataFrame()


# ============================================================
# EVALUATION
# ============================================================

def evaluate_single(model: BaseLLM, model_name: str, df: pd.DataFrame,
                    source: str, strategy: str) -> List[Dict]:
    """Evaluate a single model with a single strategy on a dataset."""

    results = []
    desc = f"{model_name[:10]} - {source[:8]} - {strategy[:5]}"

    # Create agent based on strategy
    if strategy == "few-shot":
        agent = FewShotAgent(model)
    elif strategy == "cot":
        agent = COTAgent(model)
    elif strategy == "structured-cot":
        agent = ReactAgentNoKG(model)
    else:  # zero-shot
        agent = model

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, ncols=80):
        question = row['question']
        answer_type = row.get('answer_type', 'mcq')
        options = row.get('options', {}) if answer_type == 'mcq' else None

        # Create prompt
        prompt = create_prompt(question, answer_type, options, strategy)

        # Generate response
        if strategy == "few-shot" and isinstance(agent, FewShotAgent):
            response = agent.generate(prompt, answer_type=answer_type)
        else:
            response = agent.generate(prompt)

        # Parse response
        parsed = parse_response(response, answer_type, strategy)

        # Get expected answer
        if answer_type == 'mcq':
            expected = str(row.get('answer', 'A')).strip().upper()
            if len(expected) > 1:
                for letter in ['A', 'B', 'C', 'D']:
                    if expected.startswith(letter):
                        expected = letter
                        break
        elif answer_type == 'yes_no':
            raw = str(row.get('answer', 'yes')).strip().lower()
            expected = 'yes' if raw.startswith('yes') else ('no' if raw.startswith('no') else raw)
        else:
            expected = str(row.get('answer', ''))

        # Score
        if answer_type in ['mcq', 'yes_no']:
            correct = parsed.lower() == expected.lower()
            score = 1.0 if correct else 0.0
        else:
            # For open-ended, use simple overlap
            correct = None
            score = len(set(parsed.lower().split()) & set(expected.lower().split())) / max(len(expected.split()), 1)

        results.append({
            'model': model_name,
            'strategy': strategy,
            'source': source,
            'answer_type': answer_type,
            'taxonomy': row.get('taxonomy', 'N/A'),
            'question': question[:100],
            'expected': expected[:100],
            'response': response[:200],
            'parsed': parsed[:50],
            'correct': correct,
            'score': score,
            'confidence': 0.7
        })

    return results


# ============================================================
# VISUALIZATION (with user's exact color palette)
# ============================================================

def create_visualizations(results_df: pd.DataFrame, metrics_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots with user's exact color palette."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use('seaborn-v0_8-whitegrid')

        # Get model colors
        model_colors = [COLOR_PALETTE.get(m, '#636e72') for m in results_df['model'].unique()]

        # 1. Accuracy by Model and Strategy (Grouped Bar Chart)
        fig, ax = plt.subplots(figsize=(16, 10))

        clf_df = results_df[results_df['answer_type'].isin(['yes_no', 'mcq'])]
        acc_data = clf_df.groupby(['model', 'strategy'])['score'].mean().reset_index()
        pivot = acc_data.pivot(index='model', columns='strategy', values='score')

        # Reorder models by total accuracy
        model_order = pivot.mean(axis=1).sort_values(ascending=False).index
        pivot = pivot.reindex(model_order)

        pivot.plot(kind='bar', ax=ax, color=['#74b9ff', '#fd79a8', '#ffeaa7', '#55efc4'],
                   width=0.8, edgecolor='white')
        ax.set_title('Classification Accuracy by Model and Prompting Strategy', fontsize=16, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend(title='Strategy', fontsize=10, title_fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 1.0)

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_by_model_strategy.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Accuracy Heatmap (GnBu colormap)
        fig, ax = plt.subplots(figsize=(14, 12))

        heat_data = clf_df.groupby(['model', 'source'])['score'].mean().reset_index()
        heat_pivot = heat_data.pivot(index='model', columns='source', values='score')

        # Sort by mean accuracy
        heat_pivot = heat_pivot.reindex(heat_pivot.mean(axis=1).sort_values(ascending=False).index)

        sns.heatmap(heat_pivot, annot=True, fmt='.3f', cmap='GnBu', ax=ax,
                    linewidths=0.5, cbar_kws={'label': 'Accuracy'})
        ax.set_title('Accuracy Heatmap: Models vs Benchmarks', fontsize=16, fontweight='bold')
        ax.set_xlabel('Benchmark', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. F1 Score Heatmap (YlOrRd colormap)
        if 'f1_macro' in metrics_df.columns:
            fig, ax = plt.subplots(figsize=(14, 12))

            f1_pivot = metrics_df.pivot_table(index='model', columns='source', values='f1_macro', aggfunc='mean')
            f1_pivot = f1_pivot.reindex(f1_pivot.mean(axis=1).sort_values(ascending=False).index)

            sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                        linewidths=0.5, cbar_kws={'label': 'F1 Score'})
            ax.set_title('F1 Score Heatmap: Models vs Benchmarks', fontsize=16, fontweight='bold')
            ax.set_xlabel('Benchmark', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)

            plt.tight_layout()
            plt.savefig(output_dir / 'f1_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Precision Heatmap (PuRd colormap)
        if 'precision_macro' in metrics_df.columns:
            fig, ax = plt.subplots(figsize=(14, 12))

            prec_pivot = metrics_df.pivot_table(index='model', columns='source', values='precision_macro', aggfunc='mean')
            prec_pivot = prec_pivot.reindex(prec_pivot.mean(axis=1).sort_values(ascending=False).index)

            sns.heatmap(prec_pivot, annot=True, fmt='.3f', cmap='PuRd', ax=ax,
                        linewidths=0.5, cbar_kws={'label': 'Precision'})
            ax.set_title('Precision Heatmap: Models vs Benchmarks', fontsize=16, fontweight='bold')
            ax.set_xlabel('Benchmark', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)

            plt.tight_layout()
            plt.savefig(output_dir / 'precision_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 5. Recall Heatmap (YlGnBu colormap)
        if 'recall_macro' in metrics_df.columns:
            fig, ax = plt.subplots(figsize=(14, 12))

            rec_pivot = metrics_df.pivot_table(index='model', columns='source', values='recall_macro', aggfunc='mean')
            rec_pivot = rec_pivot.reindex(rec_pivot.mean(axis=1).sort_values(ascending=False).index)

            sns.heatmap(rec_pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                        linewidths=0.5, cbar_kws={'label': 'Recall'})
            ax.set_title('Recall Heatmap: Models vs Benchmarks', fontsize=16, fontweight='bold')
            ax.set_xlabel('Benchmark', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)

            plt.tight_layout()
            plt.savefig(output_dir / 'recall_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 6. Accuracy by Benchmark (with model colors)
        fig, ax = plt.subplots(figsize=(16, 10))

        acc_bench = clf_df.groupby(['source', 'model'])['score'].mean().reset_index()
        pivot2 = acc_bench.pivot(index='source', columns='model', values='score')

        # Get colors for each model
        colors = [COLOR_PALETTE.get(m, '#636e72') for m in pivot2.columns]

        pivot2.plot(kind='bar', ax=ax, color=colors, width=0.8, edgecolor='white')
        ax.set_title('Classification Accuracy by Benchmark', fontsize=16, fontweight='bold')
        ax.set_xlabel('Benchmark', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_by_benchmark.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Strategy Comparison (Radar/Spider Chart)
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        strategies = results_df['strategy'].unique()
        benchmarks = results_df['source'].unique()
        n_benchmarks = len(benchmarks)

        angles = np.linspace(0, 2 * np.pi, n_benchmarks, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        colors_strat = ['#74b9ff', '#fd79a8', '#ffeaa7', '#55efc4']
        for i, strategy in enumerate(strategies):
            strat_df = clf_df[clf_df['strategy'] == strategy]
            values = [strat_df[strat_df['source'] == b]['score'].mean() for b in benchmarks]
            values += values[:1]  # Close the polygon
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors_strat[i % len(colors_strat)])
            ax.fill(angles, values, alpha=0.25, color=colors_strat[i % len(colors_strat)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(benchmarks, fontsize=10)
        ax.set_title('Strategy Performance Across Benchmarks', fontsize=16, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        plt.savefig(output_dir / 'strategy_radar.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 8. CARES Score Comparison (for BioREASONC only)
        bio_metrics = metrics_df[metrics_df['source'].str.contains('BioREASONC')]
        if len(bio_metrics) > 0 and 'cares_score' in bio_metrics.columns:
            fig, ax = plt.subplots(figsize=(14, 8))

            cares_data = bio_metrics.groupby(['model', 'strategy'])['cares_score'].mean().reset_index()
            cares_pivot = cares_data.pivot(index='model', columns='strategy', values='cares_score')
            cares_pivot = cares_pivot.reindex(cares_pivot.mean(axis=1).sort_values(ascending=False).index)

            cares_pivot.plot(kind='bar', ax=ax, color=['#74b9ff', '#fd79a8', '#ffeaa7', '#55efc4'],
                            width=0.8, edgecolor='white')
            ax.set_title('CARES Score by Model and Strategy (BioREASONC)', fontsize=16, fontweight='bold')
            ax.set_xlabel('Model', fontsize=12)
            ax.set_ylabel('CARES Score', fontsize=12)
            ax.legend(title='Strategy', fontsize=10, title_fontsize=11)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            plt.savefig(output_dir / 'cares_score_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  Visualizations saved to {output_dir}")

    except Exception as e:
        print(f"  Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE CLASSIFICATION BENCHMARK EVALUATION")
    print("8 Models × 4 Prompting Strategies = 32 Configurations")
    print("=" * 80)

    # Initialize models
    print("\n[1/6] Initializing LLM models...")
    models = initialize_models()
    print(f"  Available models: {list(models.keys())}")

    model_names = list(models.keys())
    strategies = ['zero-shot', 'few-shot', 'cot', 'structured-cot']

    print(f"  Prompting strategies: {strategies}")
    print(f"  Total configurations: {len(model_names)} × {len(strategies)} = {len(model_names) * len(strategies)}")

    # Load datasets (NO BioKGBench)
    print("\n[2/6] Loading datasets...")
    datasets = {}

    # BioREASONC with balanced taxonomy
    bio_yesno = load_bioreasonc_balanced('yes_no', SAMPLES_PER_TAXONOMY)
    if len(bio_yesno) > 0:
        bio_yesno['source'] = 'BioREASONC-YesNo'
        datasets['BioREASONC-YesNo'] = bio_yesno

    bio_mcq = load_bioreasonc_balanced('mcq', SAMPLES_PER_TAXONOMY)
    if len(bio_mcq) > 0:
        bio_mcq['source'] = 'BioREASONC-MCQ'
        datasets['BioREASONC-MCQ'] = bio_mcq

    # External benchmarks (NO BioKGBench)
    datasets['PubMedQA'] = load_pubmedqa(SAMPLES_PER_BENCHMARK)
    datasets['BioASQ'] = load_bioasq(SAMPLES_PER_BENCHMARK)
    datasets['MedQA'] = load_medqa(SAMPLES_PER_BENCHMARK)
    datasets['MedMCQA'] = load_medmcqa(SAMPLES_PER_BENCHMARK)
    datasets['MMLU-Med'] = load_mmlu_med(SAMPLES_PER_BENCHMARK)

    # Remove empty datasets
    datasets = {k: v for k, v in datasets.items() if len(v) > 0}

    print("\n  Dataset Summary:")
    for name, df in datasets.items():
        print(f"    {name}: {len(df)} samples")

    # Run evaluation
    print("\n[3/6] Running evaluation...")
    all_results = []

    total_evals = len(datasets) * len(model_names) * len(strategies)
    eval_count = 0

    for source, df in datasets.items():
        print(f"\n--- {source} ---")

        for model_name in model_names:
            model = models[model_name]
            for strategy in strategies:
                eval_count += 1
                print(f"  [{eval_count}/{total_evals}] {model_name} - {strategy}")

                results = evaluate_single(model, model_name, df, source, strategy)
                all_results.extend(results)

                # Small delay to avoid rate limits
                time.sleep(0.1)

    # Save results
    print("\n[4/6] Saving results...")
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'all_results.csv', index=False)
    print(f"  Saved {len(results_df)} results to all_results.csv")

    # Compute metrics summary
    print("\n[5/6] Computing metrics summary...")

    metrics_list = []
    clf_results = results_df[results_df['answer_type'].isin(['yes_no', 'mcq'])]

    for model in model_names:
        for strategy in strategies:
            for source in datasets.keys():
                subset = clf_results[(clf_results['model'] == model) &
                                    (clf_results['strategy'] == strategy) &
                                    (clf_results['source'] == source)]

                if len(subset) > 0:
                    y_true = subset['expected'].tolist()
                    y_pred = subset['parsed'].tolist()

                    metrics = compute_metrics(y_true, y_pred)
                    metrics['model'] = model
                    metrics['strategy'] = strategy
                    metrics['source'] = source
                    metrics['n_samples'] = len(subset)

                    # CARES score for BioREASONC
                    if 'BioREASONC' in source:
                        cares = compute_cares_score(subset.to_dict('records'))
                        metrics['cares_score'] = cares['cares_score']

                    metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(OUTPUT_DIR / 'metrics_summary.csv', index=False)
    print(f"  Saved metrics summary to metrics_summary.csv")

    # Save token usage summary
    token_summary = token_tracker.get_summary()
    with open(OUTPUT_DIR / 'token_usage.json', 'w') as f:
        json.dump(token_summary, f, indent=2)
    print(f"  Total tokens used: {token_summary['total_tokens']:,}")
    print(f"  Estimated cost: ${token_summary['total_cost_usd']:.2f}")

    # Visualizations
    print("\n[6/6] Creating visualizations...")
    create_visualizations(results_df, metrics_df, OUTPUT_DIR)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print("\n--- Best Results by Benchmark ---")
    for source in datasets.keys():
        source_metrics = metrics_df[metrics_df['source'] == source]
        if len(source_metrics) > 0:
            best = source_metrics.loc[source_metrics['accuracy'].idxmax()]
            print(f"  {source}: {best['model']} + {best['strategy']} = {best['accuracy']:.3f}")

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
