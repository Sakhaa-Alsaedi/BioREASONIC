"""
LLM-as-Judge API for CARES Scoring

Supports multiple LLM providers for evaluating biomedical reasoning responses:
- OpenAI (GPT-4, GPT-4-turbo)
- Anthropic (Claude-3)
- Google Gemini
- Together AI
- Local models (via Ollama, vLLM, or HuggingFace)

Configuration is loaded from environment variables or .env file.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _load_env_file():
    """Load .env file if it exists"""
    env_paths = ['.env', '../.env', '../../.env']
    for path in env_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key.strip() not in os.environ:
                            os.environ[key.strip()] = value.strip()
            break

# Load .env on import
_load_env_file()


@dataclass
class JudgeResult:
    """Result from LLM-as-Judge evaluation"""
    score: int          # 0-5 scale
    confidence: float   # 0-1
    reasoning: str      # Explanation
    raw_response: str   # Raw LLM output

    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'confidence': self.confidence,
            'reasoning': self.reasoning
        }


JUDGE_PROMPT_TEMPLATE = """You are an expert biomedical scientist evaluating AI responses on genetic risk and causal reasoning tasks.

## Task Category: {category_name}
{category_description}

## Question
{question}

## Gold Standard Answer
{gold_answer}

## Model Response to Evaluate
{model_response}

## Scoring Rubric (0-5 scale)
- **5**: Fully correct. Semantically equivalent to gold answer, all key facts present.
- **4**: Mostly correct. Minor imprecisions or missing non-critical details.
- **3**: Partially correct. Contains correct information but missing >20% of key details.
- **2**: Safe abstention. Model appropriately expresses uncertainty rather than guessing.
- **1**: Partial hallucination. Mix of correct and fabricated information.
- **0**: Complete hallucination. Confidently incorrect, fabricated facts.

## Evaluation Criteria for {category_name}:
{evaluation_criteria}

## Instructions
1. Compare the model response to the gold standard
2. Check for factual accuracy in biomedical content
3. Verify correct interpretation of genetic/statistical data
4. Assess reasoning quality and logical consistency
5. Identify any hallucinated information

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{"score": <0-5>, "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}
"""

CATEGORY_INFO = {
    'S': {
        'name': 'Structure-Aware Reasoning',
        'description': 'Evaluates understanding of biological network structure, graph traversal, and pathway relationships.',
        'criteria': '''- Correct identification of network relationships
- Accurate path finding between biological entities
- Proper interpretation of protein-protein interactions
- Understanding of pathway hierarchies'''
    },
    'C': {
        'name': 'Causal-Aware Reasoning',
        'description': 'Evaluates causal inference, distinguishing correlation from causation, and Mendelian randomization interpretation.',
        'criteria': '''- Correct distinction between association and causation
- Accurate interpretation of MR results (IVW, Egger, etc.)
- Understanding of confounding and instrumental variables
- Proper causal direction inference'''
    },
    'R': {
        'name': 'Risk-Aware Reasoning',
        'description': 'Evaluates genetic risk assessment, odds ratio interpretation, and risk score calculation.',
        'criteria': '''- Correct interpretation of odds ratios and beta coefficients
- Accurate risk level classification (HIGH/MODERATE/LOW/PROTECTIVE)
- Proper understanding of MAF and effect sizes
- Correct aggregate risk calculations'''
    },
    'M': {
        'name': 'Semantic-Aware Reasoning',
        'description': 'Evaluates entity recognition, relation extraction, and semantic understanding from biomedical text.',
        'criteria': '''- Accurate gene/disease/variant entity recognition
- Correct relation extraction between entities
- Proper semantic similarity assessment
- Understanding of biomedical terminology'''
    }
}


class LLMJudge(ABC):
    """Abstract base class for LLM-as-Judge"""

    @abstractmethod
    def score_response(
        self,
        question: str,
        model_response: str,
        gold_answer: str,
        category: str
    ) -> JudgeResult:
        """Score a model response against gold answer"""
        pass

    def _build_prompt(
        self,
        question: str,
        model_response: str,
        gold_answer: str,
        category: str
    ) -> str:
        """Build the evaluation prompt"""
        cat_info = CATEGORY_INFO.get(category.upper(), CATEGORY_INFO['R'])

        return JUDGE_PROMPT_TEMPLATE.format(
            category_name=cat_info['name'],
            category_description=cat_info['description'],
            question=question,
            gold_answer=gold_answer,
            model_response=model_response,
            evaluation_criteria=cat_info['criteria']
        )

    def _parse_response(self, response_text: str) -> JudgeResult:
        """Parse LLM response into JudgeResult"""
        try:
            # Try to extract JSON from response
            # Handle cases where LLM adds markdown or extra text
            text = response_text.strip()

            # Remove markdown code blocks if present
            if text.startswith('```'):
                text = text.split('```')[1]
                if text.startswith('json'):
                    text = text[4:]

            # Find JSON object
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                data = json.loads(json_str)

                return JudgeResult(
                    score=int(data.get('score', 0)),
                    confidence=float(data.get('confidence', 0.5)),
                    reasoning=str(data.get('reasoning', '')),
                    raw_response=response_text
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        # Default fallback
        return JudgeResult(
            score=0,
            confidence=0.0,
            reasoning="Failed to parse LLM response",
            raw_response=response_text
        )

    def batch_score(
        self,
        items: List[Dict]
    ) -> List[JudgeResult]:
        """Score multiple items"""
        results = []
        for item in items:
            result = self.score_response(
                question=item['question'],
                model_response=item['model_response'],
                gold_answer=item['gold_answer'],
                category=item['category']
            )
            results.append(result)
        return results


class OpenAIJudge(LLMJudge):
    """OpenAI GPT-based judge"""

    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Initialize OpenAI judge

        Args:
            model: OpenAI model name (gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

    def score_response(
        self,
        question: str,
        model_response: str,
        gold_answer: str,
        category: str
    ) -> JudgeResult:
        """Score using OpenAI API"""
        prompt = self._build_prompt(question, model_response, gold_answer, category)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert biomedical evaluator. Respond only with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # Force JSON response
            )

            response_text = response.choices[0].message.content
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return JudgeResult(
                score=0,
                confidence=0.0,
                reasoning=f"API error: {str(e)}",
                raw_response=""
            )


class AnthropicJudge(LLMJudge):
    """Anthropic Claude-based judge"""

    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Initialize Anthropic judge

        Args:
            model: Claude model (claude-3-opus-20240229, claude-3-sonnet-20240229)
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

    def score_response(
        self,
        question: str,
        model_response: str,
        gold_answer: str,
        category: str
    ) -> JudgeResult:
        """Score using Anthropic API"""
        prompt = self._build_prompt(question, model_response, gold_answer, category)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = response.content[0].text
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return JudgeResult(
                score=0,
                confidence=0.0,
                reasoning=f"API error: {str(e)}",
                raw_response=""
            )


class LocalLLMJudge(LLMJudge):
    """Local LLM judge via Ollama or HuggingFace"""

    def __init__(
        self,
        model: str = "llama3:70b",
        backend: str = "ollama",  # "ollama", "huggingface", "vllm"
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Initialize local LLM judge

        Args:
            model: Model name
            backend: Backend to use (ollama, huggingface, vllm)
            base_url: API base URL for Ollama/vLLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        """
        self.model = model
        self.backend = backend
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        if backend == "huggingface":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                self.tokenizer = AutoTokenizer.from_pretrained(model)
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            except ImportError:
                raise ImportError("transformers package required for HuggingFace backend")

    def score_response(
        self,
        question: str,
        model_response: str,
        gold_answer: str,
        category: str
    ) -> JudgeResult:
        """Score using local LLM"""
        prompt = self._build_prompt(question, model_response, gold_answer, category)

        if self.backend == "ollama":
            return self._score_ollama(prompt)
        elif self.backend == "huggingface":
            return self._score_huggingface(prompt)
        elif self.backend == "vllm":
            return self._score_vllm(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _score_ollama(self, prompt: str) -> JudgeResult:
        """Score using Ollama API"""
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                },
                timeout=120
            )
            response.raise_for_status()

            response_text = response.json().get("response", "")
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return JudgeResult(
                score=0,
                confidence=0.0,
                reasoning=f"API error: {str(e)}",
                raw_response=""
            )

    def _score_huggingface(self, prompt: str) -> JudgeResult:
        """Score using HuggingFace transformers"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.hf_model.device)

            outputs = self.hf_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 0.01),
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

            response_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f"HuggingFace error: {e}")
            return JudgeResult(
                score=0,
                confidence=0.0,
                reasoning=f"Error: {str(e)}",
                raw_response=""
            )

    def _score_vllm(self, prompt: str) -> JudgeResult:
        """Score using vLLM server"""
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                },
                timeout=120
            )
            response.raise_for_status()

            response_text = response.json()["choices"][0]["text"]
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            return JudgeResult(
                score=0,
                confidence=0.0,
                reasoning=f"API error: {str(e)}",
                raw_response=""
            )


class GeminiJudge(LLMJudge):
    """Google Gemini-based judge"""

    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Initialize Gemini judge

        Args:
            model: Gemini model (gemini-pro, gemini-1.5-pro)
            api_key: Google API key (uses GEMINI_API_KEY env var if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")

    def score_response(
        self,
        question: str,
        model_response: str,
        gold_answer: str,
        category: str
    ) -> JudgeResult:
        """Score using Gemini API"""
        prompt = self._build_prompt(question, model_response, gold_answer, category)

        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )

            response_text = response.text
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return JudgeResult(
                score=0,
                confidence=0.0,
                reasoning=f"API error: {str(e)}",
                raw_response=""
            )


class TogetherJudge(LLMJudge):
    """Together AI-based judge"""

    def __init__(
        self,
        model: str = "meta-llama/Llama-3-70b-chat-hf",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Initialize Together AI judge

        Args:
            model: Together AI model
            api_key: Together API key (uses TOGETHER_API_KEY env var if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.base_url = "https://api.together.xyz/v1"

    def score_response(
        self,
        question: str,
        model_response: str,
        gold_answer: str,
        category: str
    ) -> JudgeResult:
        """Score using Together AI API"""
        import requests

        prompt = self._build_prompt(question, model_response, gold_answer, category)

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert biomedical evaluator. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                },
                timeout=120
            )
            response.raise_for_status()

            response_text = response.json()["choices"][0]["message"]["content"]
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f"Together AI API error: {e}")
            return JudgeResult(
                score=0,
                confidence=0.0,
                reasoning=f"API error: {str(e)}",
                raw_response=""
            )


def create_judge(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs
) -> LLMJudge:
    """
    Factory function to create an LLM judge

    Args:
        provider: "openai", "anthropic", "gemini", "together", "ollama", "huggingface", "vllm"
        model: Model name (uses default if not specified)
        **kwargs: Additional arguments for the judge

    Returns:
        LLMJudge instance
    """
    if provider == "openai":
        return OpenAIJudge(model=model or "gpt-4-turbo-preview", **kwargs)
    elif provider == "anthropic":
        return AnthropicJudge(model=model or "claude-3-opus-20240229", **kwargs)
    elif provider == "gemini":
        return GeminiJudge(model=model or "gemini-1.5-pro", **kwargs)
    elif provider == "together":
        return TogetherJudge(model=model or "meta-llama/Llama-3-70b-chat-hf", **kwargs)
    elif provider == "ollama":
        return LocalLLMJudge(model=model or "llama3:70b", backend="ollama", **kwargs)
    elif provider == "huggingface":
        return LocalLLMJudge(model=model or "meta-llama/Llama-3-70b-chat-hf", backend="huggingface", **kwargs)
    elif provider == "vllm":
        return LocalLLMJudge(model=model or "meta-llama/Llama-3-70b-chat-hf", backend="vllm", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# Example usage
if __name__ == "__main__":
    # Test with mock data
    print("LLM Judge Module")
    print("=" * 50)
    print("\nSupported providers:")
    print("  - openai (GPT-4)")
    print("  - anthropic (Claude-3)")
    print("  - ollama (local)")
    print("  - huggingface (local)")
    print("  - vllm (local server)")

    print("\nExample usage:")
    print("""
    from config.apis.llm_judge import create_judge

    # Create judge
    judge = create_judge(provider="openai", model="gpt-4-turbo-preview")

    # Score a response
    result = judge.score_response(
        question="What is the risk level of TYK2 for rheumatoid arthritis?",
        model_response="TYK2 has a protective effect (OR=0.63)",
        gold_answer="PROTECTIVE. TYK2 confers protective effect (OR=0.63).",
        category="R"
    )

    print(f"Score: {result.score}/5")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning: {result.reasoning}")
    """)
