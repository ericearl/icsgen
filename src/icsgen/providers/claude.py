"""Anthropic Claude provider — calls the Messages API."""

from __future__ import annotations

import httpx

from icsgen.providers.base import BaseProvider, ProviderError


class ClaudeProvider(BaseProvider):
    """Client for the Anthropic Messages API.

    Endpoint shape (default):
        POST https://api.anthropic.com/v1/messages
        headers: x-api-key, anthropic-version, content-type
        body:    {model, max_tokens, system, messages: [{role, content}]}
        reply:   {content: [{type, text}, ...]}
    """

    NAME = "claude"
    ANTHROPIC_VERSION = "2023-06-01"

    def complete(self, system_prompt: str, user_message: str) -> str:
        headers = {
            "x-api-key": self.cfg.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        body = {
            "model": self.cfg.model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(self.cfg.endpoint, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"Claude request failed: {e}") from e

        self._check_status(resp, "Claude")

        try:
            data = resp.json()
        except Exception as e:
            raise ProviderError(f"Claude returned non-JSON response: {e}") from e

        # Concatenate all text blocks in the response (typically just one).
        chunks = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                chunks.append(block.get("text", ""))

        text = "".join(chunks).strip()
        if not text:
            raise ProviderError(f"Claude returned an empty response: {data!r}")
        return text
