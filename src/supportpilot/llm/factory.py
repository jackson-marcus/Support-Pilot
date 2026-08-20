"""The swap point: build the configured LLM provider from settings."""

from __future__ import annotations

from supportpilot.llm.base import FakeProvider, LLMProvider
from supportpilot.settings import get_settings


def get_provider(name: str | None = None) -> LLMProvider:
    """Return the provider named by `name` or the LLM_PROVIDER env var."""
    provider = (name or get_settings().llm_provider).lower()
    if provider == "claude":
        from supportpilot.llm.claude import ClaudeProvider

        return ClaudeProvider()
    if provider == "ollama":
        from supportpilot.llm.ollama import OllamaProvider

        return OllamaProvider()
    if provider == "fake":
        return FakeProvider()
    raise ValueError(f"Unknown LLM provider: {provider!r} (expected claude|ollama|fake)")
