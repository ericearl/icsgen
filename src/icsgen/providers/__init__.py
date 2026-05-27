"""LLM provider clients.

`get_provider_client(provider_config)` returns the correct client instance for a
given provider name. Adding a new provider means: add a module, subclass
`BaseProvider`, and register it in the `_REGISTRY` below.
"""

from __future__ import annotations

from icsgen.config import ProviderConfig
from icsgen.providers.base import BaseProvider
from icsgen.providers.claude import ClaudeProvider
from icsgen.providers.gemini import GeminiProvider
from icsgen.providers.openai import OpenAIProvider
from icsgen.providers.openrouter import OpenRouterProvider


_REGISTRY: dict[str, type[BaseProvider]] = {
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider_client(cfg: ProviderConfig) -> BaseProvider:
    """Return an instantiated provider client for the given config."""
    cls = _REGISTRY.get(cfg.name)
    if cls is None:
        raise ValueError(f"Unknown provider: {cfg.name}")
    return cls(cfg)


__all__ = [
    "BaseProvider",
    "ClaudeProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "get_provider_client",
]
