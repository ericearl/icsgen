"""Abstract provider interface.

Each LLM provider implements `complete(system_prompt, user_message) -> str`
returning the raw text content of the model's reply. Higher layers handle JSON
parsing and validation — providers stay dumb on purpose.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from icsgen.config import ProviderConfig


class ProviderError(Exception):
    """Raised when an LLM call fails (network, auth, parse, etc.)."""


class BaseProvider(ABC):
    """Base class for LLM provider clients."""

    def __init__(self, cfg: ProviderConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        """Send the prompts to the LLM and return the response text."""
        ...

    # --- shared utilities -------------------------------------------------

    @staticmethod
    def _check_status(resp, provider_name: str) -> None:
        """Raise ProviderError with a helpful message if the response is not 2xx."""
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise ProviderError(
                f"{provider_name} API returned HTTP {resp.status_code}: {body}"
            )
