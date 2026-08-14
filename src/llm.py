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
