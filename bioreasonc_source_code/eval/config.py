"""
BioREASONC-Bench Configuration

API keys should be set via environment variables for security.
NEVER hardcode API keys in source code.

Set environment variables:
    export OPENAI_API_KEY="your-key"
    export ANTHROPIC_API_KEY="your-key"
    export GEMINI_API_KEY="your-key"
    export TOGETHER_API_KEY="your-key"
    export NCBI_API_KEY="your-key"  # Optional, for faster ClinVar access
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration with environment variable support"""

    # LLM API Keys (from environment variables)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    together_api_key: Optional[str] = None

    # Biomedical API Keys (optional)
    ncbi_api_key: Optional[str] = None  # For ClinVar

    # Default LLM settings
    default_judge_provider: str = "openai"
    default_judge_model: str = "gpt-4"

    # API timeouts
    llm_timeout: int = 60
    biomedical_api_timeout: int = 30

    def __post_init__(self):
        """Load API keys from environment variables"""
        self.openai_api_key = os.environ.get("OPENAI_API_KEY") or self.openai_api_key
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY") or self.anthropic_api_key
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY") or self.gemini_api_key
        self.together_api_key = os.environ.get("TOGETHER_API_KEY") or self.together_api_key
        self.ncbi_api_key = os.environ.get("NCBI_API_KEY") or self.ncbi_api_key

    def get_llm_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a specific LLM provider"""
        key_map = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "gemini": self.gemini_api_key,
            "together": self.together_api_key,
        }
        return key_map.get(provider.lower())

    def has_key(self, provider: str) -> bool:
        """Check if API key is available for provider"""
        return self.get_llm_api_key(provider) is not None

    def validate(self) -> dict:
        """Validate configuration and return status"""
        status = {
            "openai": bool(self.openai_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "gemini": bool(self.gemini_api_key),
            "together": bool(self.together_api_key),
            "ncbi": bool(self.ncbi_api_key),
        }
        return status


# Global configuration instance
_config: Optional[APIConfig] = None


def get_config() -> APIConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = APIConfig()
    return _config


def configure(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    together_api_key: Optional[str] = None,
    ncbi_api_key: Optional[str] = None,
    **kwargs
) -> APIConfig:
    """
    Configure API keys programmatically

    Args:
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        gemini_api_key: Google Gemini API key
        together_api_key: Together AI API key
        ncbi_api_key: NCBI API key for ClinVar

    Returns:
        APIConfig instance
    """
    global _config
    _config = APIConfig(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        gemini_api_key=gemini_api_key,
        together_api_key=together_api_key,
        ncbi_api_key=ncbi_api_key,
        **kwargs
    )
    return _config


def load_from_env_file(filepath: str = ".env"):
    """
    Load configuration from .env file

    Args:
        filepath: Path to .env file
    """
    if not os.path.exists(filepath):
        logger.warning(f".env file not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

    # Reload config
    global _config
    _config = APIConfig()
    logger.info(f"Loaded configuration from {filepath}")


# Available LLM providers and their models
LLM_PROVIDERS = {
    "openai": {
        "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"],
        "default": "gpt-4"
    },
    "anthropic": {
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "default": "claude-3-sonnet-20240229"
    },
    "gemini": {
        "models": ["gemini-pro", "gemini-1.5-pro"],
        "default": "gemini-pro"
    },
    "together": {
        "models": ["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
        "default": "meta-llama/Llama-3-70b-chat-hf"
    },
    "local": {
        "models": ["llama2", "mistral", "codellama"],
        "default": "llama2"
    }
}


def get_available_providers() -> list:
    """Get list of providers with configured API keys"""
    config = get_config()
    available = []

    for provider in LLM_PROVIDERS:
        if provider == "local" or config.has_key(provider):
            available.append(provider)

    return available
