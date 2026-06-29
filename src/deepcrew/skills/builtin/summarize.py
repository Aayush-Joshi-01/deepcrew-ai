from __future__ import annotations

from typing import Any

from ..base import Skill


class SummarizeSkill(Skill):
    """Summarize a block of text using an LLM."""

    name = "summarize"
    description = "Summarize a block of text into a shorter form."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to summarize"},
            "max_words": {
                "type": "integer",
                "description": "Target maximum word count for the summary",
                "default": 200,
            },
        },
        "required": ["text"],
    }

    def __init__(self, model: str = "openai/gpt-4o-mini") -> None:
        self._model = model

    async def execute(self, text: str, max_words: int = 200, **_: Any) -> str:
        import litellm

        prompt = (
            f"Summarize the following text in at most {max_words} words. "
            "Be concise and preserve the key facts.\n\n"
            f"{text}"
        )
        resp = await litellm.acompletion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return resp.choices[0].message.content or ""
