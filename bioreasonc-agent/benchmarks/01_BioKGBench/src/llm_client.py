#!/usr/bin/env python3
"""
LLM Client - Unified interface for OpenAI and Claude APIs.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: str = None) -> Dict:
        """Generate a JSON response from the LLM."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini",
                 temperature: float = 0.0, max_tokens: int = 1000):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def generate_json(self, prompt: str, system_prompt: str = None) -> Dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)


class TogetherClient(LLMClient):
    """Together AI API client (OpenAI-compatible)."""

    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3-70b-chat-hf",
                 temperature: float = 0.0, max_tokens: int = 1000):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1"
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def generate_json(self, prompt: str, system_prompt: str = None) -> Dict:
        # Add JSON instruction to prompt (Together doesn't always support response_format)
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no other text or explanation."

        response = self.generate(json_prompt, system_prompt)

        # Extract JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            # Try to extract JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                return json.loads(json_match.group(1))
            # Try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")


class ClaudeClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022",
                 temperature: float = 0.0, max_tokens: int = 1000):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)
        return response.content[0].text

    def generate_json(self, prompt: str, system_prompt: str = None) -> Dict:
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no other text."

        response = self.generate(json_prompt, system_prompt)

        # Extract JSON from response
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                return json.loads(json_match.group(1))
            # Try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group(0))
            raise ValueError(f"Could not parse JSON from response: {response[:200]}")


# Together AI model mappings (short names to full model IDs)
TOGETHER_MODELS = {
    'llama3-70b': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    'llama3-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'llama-3.1-8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
    'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
    'qwen-2.5-7b': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
    'deepseek': 'deepseek-ai/DeepSeek-V3',
    'deepseek-v3': 'deepseek-ai/DeepSeek-V3',
}

# OpenAI model mappings
OPENAI_MODELS = {
    'gpt-4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4.1': 'gpt-4-turbo',
    'gpt-4-turbo': 'gpt-4-turbo',  # Added: direct mapping
    'gpt-4.1-mini': 'gpt-4-turbo-preview',
    'gpt-4-turbo-preview': 'gpt-4-turbo-preview',  # Added: direct mapping
    'gpt-3.5-turbo': 'gpt-3.5-turbo',
}

# Claude model mappings
CLAUDE_MODELS = {
    'claude-3-haiku': 'claude-3-haiku-20240307',
    'claude-3-sonnet': 'claude-3-sonnet-20240229',
    'claude-3-opus': 'claude-3-opus-20240229',
    'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
}


def create_llm_client(config: Dict) -> Optional[LLMClient]:
    """Create an LLM client based on configuration."""
    provider = config.get('provider', 'none')

    if provider == 'none':
        return None

    # OpenAI models (including gpt-4o, gpt-4.1, etc.)
    elif provider == 'openai' or provider in OPENAI_MODELS:
        openai_config = config.get('openai', {})
        api_key = openai_config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not provided in config or OPENAI_API_KEY env var")

        # Get model from provider name or config
        if provider in OPENAI_MODELS:
            model = OPENAI_MODELS[provider]
        else:
            model = openai_config.get('model', 'gpt-4o-mini')

        return OpenAIClient(
            api_key=api_key,
            model=model,
            temperature=openai_config.get('temperature', 0.0),
            max_tokens=openai_config.get('max_tokens', 1000)
        )

    # Claude models (including claude-3-haiku, etc.)
    elif provider == 'claude' or provider in CLAUDE_MODELS:
        claude_config = config.get('claude', {})
        api_key = claude_config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Claude API key not provided in config or ANTHROPIC_API_KEY env var")

        # Get model from provider name or config
        if provider in CLAUDE_MODELS:
            model = CLAUDE_MODELS[provider]
        else:
            model = claude_config.get('model', 'claude-3-5-sonnet-20241022')

        return ClaudeClient(
            api_key=api_key,
            model=model,
            temperature=claude_config.get('temperature', 0.0),
            max_tokens=claude_config.get('max_tokens', 1000)
        )

    # Together AI models (Llama, Qwen, DeepSeek, Mixtral)
    elif provider in ['together'] or provider in TOGETHER_MODELS:
        together_config = config.get('together', {})
        api_key = together_config.get('api_key') or os.environ.get('TOGETHER_API_KEY')
        if not api_key:
            raise ValueError("Together API key not provided in config or TOGETHER_API_KEY env var")

        # Get model from provider name or config
        if provider in TOGETHER_MODELS:
            model = TOGETHER_MODELS[provider]
        else:
            model = together_config.get('model', TOGETHER_MODELS['llama3-70b'])

        return TogetherClient(
            api_key=api_key,
            model=model,
            temperature=together_config.get('temperature', 0.0),
            max_tokens=together_config.get('max_tokens', 1000)
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# Test
if __name__ == "__main__":
    import yaml

    # Load config
    config_path = "config.yaml"
    if os.path.exists("config.local.yaml"):
        config_path = "config.local.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    llm_config = config.get('llm', {})
    print(f"Provider: {llm_config.get('provider', 'none')}")

    if llm_config.get('provider') != 'none':
        try:
            client = create_llm_client(llm_config)
            response = client.generate("What is 2+2?")
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
