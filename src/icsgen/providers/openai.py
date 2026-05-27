"""OpenAI (ChatGPT) provider — calls the Chat Completions API."""

from __future__ import annotations

import httpx

from icsgen.providers.base import BaseProvider, ProviderError


class OpenAIProvider(BaseProvider):
    """Client for the OpenAI Chat Completions API.

    Endpoint shape (default):
        POST https://api.openai.com/v1/chat/completions
        headers: Authorization: Bearer <key>
        body:    {model, messages: [{role: system, ...}, {role: user, ...}],
                  response_format: {type: json_object}}
        reply:   {choices: [{message: {content: "..."}}]}
    """

    NAME = "openai"

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
            # response_format=json_object hard-constrains the output to JSON;
            # the API requires the word "json" to appear in the messages, which
            # our system prompt satisfies.
            "response_format": {"type": "json_object"},
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(self.cfg.endpoint, headers=headers, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI request failed: {e}") from e

        self._check_status(resp, "OpenAI")

        try:
            data = resp.json()
        except Exception as e:
            raise ProviderError(f"OpenAI returned non-JSON response: {e}") from e

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"Unexpected OpenAI response shape: {data!r}") from e

        if not text or not text.strip():
            raise ProviderError(f"OpenAI returned an empty response: {data!r}")
        return text.strip()
