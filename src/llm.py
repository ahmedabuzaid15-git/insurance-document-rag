"""LLM clients for the answer-generation step.

Two implementations share one informal interface (`generate(prompt) -> str`):
`ScriptedLLM`, a test double that returns pre-queued responses so the
citation-bearing answer step can be tested with no network and no API key,
and `OpenAILLM`, an optional live client used only when OPENAI_API_KEY is set
and network access is available. Retrieval and the hallucination guard do not
depend on either -- both work from embedding similarity alone.
"""

from __future__ import annotations

import os
import time


class ScriptedLLM:
    """Test double returning queued canned responses instead of calling a real model."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("ScriptedLLM called with no queued responses left")
        return self._responses.pop(0)


class OpenAILLM:
    """Live OpenAI-backed client. Requires OPENAI_API_KEY and network access."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; live generation is unavailable")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content or ""


class GeminiLLM:
    """Live Gemini-backed client, alternative to OpenAILLM. Requires GEMINI_API_KEY."""

    def __init__(self, model: str = "gemini-2.5-flash-lite") -> None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; live generation is unavailable")
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = _generate_with_retry(self._client, self._model, prompt)
        return response.text or ""


def _generate_with_retry(client, model: str, contents: str, attempts: int = 6):
    """Gemini free-tier keys are capped at ~5 requests/minute per model and
    occasionally return transient 503s under load; the SDK's own retry budget
    is too short to ride either out, so retry here with real backoff -- long
    enough to clear a per-minute quota window -- before giving up.
    """
    from google.genai.errors import ClientError, ServerError

    delay = 15.0
    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except ServerError:
            if attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 60.0)
        except ClientError as exc:
            if getattr(exc, "code", None) != 429 or attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 60.0)
