"""OpenRouter provider — calls the OpenAI-compatible Chat Completions API.

OpenRouter routes a single API to many models (including a tier of `:free`
models). The request/response shape matches OpenAI's Chat Completions, so this
client is structurally identical to OpenAIProvider, with two differences:

  * The endpoint defaults to https://openrouter.ai/api/v1/chat/completions.
  * `response_format=json_object` is omitted by default: many of the free
    models on OpenRouter do not support strict JSON mode, and the system
    prompt already constrains the output. `parse_llm_response` handles
    fenced or prose-wrapped JSON.
"""

from __future__ import annotations

import httpx

from icsgen.providers.base import BaseProvider, ProviderError


class OpenRouterProvider(BaseProvider):
    """Client for OpenRouter's Chat Completions API."""

    NAME = "openrouter"

    def complete(self, system_prompt: str, user_message: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(self.cfg.endpoint, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenRouter request failed: {e}") from e

        self._check_status(resp, "OpenRouter")

        try:
            data = resp.json()
        except Exception as e:
            raise ProviderError(f"OpenRouter returned non-JSON response: {e}") from e

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"Unexpected OpenRouter response shape: {data!r}") from e

        if not text or not text.strip():
            raise ProviderError(f"OpenRouter returned an empty response: {data!r}")
        return text.strip()
