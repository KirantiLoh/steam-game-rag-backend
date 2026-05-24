"""LLM configuration for Gemini Flash 1.5."""
from typing import Optional
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """Configuration for free Gemini Flash 1.5 integration."""

    # Provider
    llm_provider: str = "gemini"
    gemini_api_key: Optional[str] = None

    # Model configuration
    model_name: str = "gemini-1.5-flash"
    max_tokens: int = 8192
    temperature: float = 0.7

    # RAG configuration
    rag_top_k: int = 25
    rag_rerank: bool = True

    # Rate limiting (Gemini free tier: 15 RPM)
    max_requests_per_minute: int = 15

    # Session management
    session_ttl_seconds: int = 1800  # 30 minutes
    max_conversation_turns: int = 5

    class Config:
        env_file = ".env"
        env_prefix = "LLM_"


settings = LLMSettings()

# Validate API key on import
if not settings.gemini_api_key:
    import warnings
    warnings.warn(
        "LLM_GEMINI_API_KEY not set. "
        "Get free key from https://aistudio.google.com/app/apikey"
    )
