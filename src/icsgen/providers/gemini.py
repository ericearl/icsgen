"""Google Gemini provider — calls the generateContent endpoint.

Note: Gemini's REST URL embeds the model name and takes the API key as a query
parameter. We accept the base URL up to `/models` from config and build the
full URL on each request, which lets the user point at a different region or
version without code changes.
"""

from __future__ import annotations

import httpx

from icsgen.providers.base import BaseProvider, ProviderError


class GeminiProvider(BaseProvider):
    """Client for the Gemini generateContent API.

    Default endpoint base: https://generativelanguage.googleapis.com/v1beta/models
    Built URL: {base}/{model}:generateContent?key={api_key}

    Request body:
        {system_instruction: {parts: [{text: ...}]},
         contents: [{role: user, parts: [{text: ...}]}],
         generationConfig: {responseMimeType: application/json}}

    Reply: {candidates: [{content: {parts: [{text: ...}]}}]}
    """

    NAME = "gemini"

    def _build_url(self) -> str:
        base = self.cfg.endpoint.rstrip("/")
        return f"{base}/{self.cfg.model}:generateContent"

    def complete(self, system_prompt: str, user_message: str) -> str:
        url = self._build_url()
        params = {"key": self.cfg.api_key}
        headers = {"Content-Type": "application/json"}
        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {"role": "user", "parts": [{"text": user_message}]},
            ],
            "generationConfig": {
                "responseMimeType": "application/json",
            },
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, params=params, json=body)
        except httpx.HTTPError as e:
            raise ProviderError(f"Gemini request failed: {e}") from e

        self._check_status(resp, "Gemini")

        try:
            data = resp.json()
        except Exception as e:
            raise ProviderError(f"Gemini returned non-JSON response: {e}") from e

        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"Unexpected Gemini response shape: {data!r}") from e

        if not text:
            raise ProviderError(f"Gemini returned an empty response: {data!r}")
        return text
