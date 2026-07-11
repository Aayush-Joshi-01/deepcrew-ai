from __future__ import annotations

from typing import Any

from ..base import Skill


class WebSearchSkill(Skill):
    """Search the web using DuckDuckGo Instant Answer API."""

    name = "web_search"
    description = "Search the web for information about a topic."
    # Class-level default satisfying the Skill instance annotation (shared, never mutated)
    parameters: dict[str, Any] = {  # noqa: RUF012
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, query: str, max_results: int = 5, **_: Any) -> str:  # type: ignore[override]
        try:
            import httpx
        except ImportError:
            return f"[web_search] httpx not installed; cannot search for: {query}"

        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            return f"[web_search] Error searching for {query!r}: {exc}"

        results: list[str] = []

        abstract = data.get("AbstractText", "")
        if abstract:
            results.append(f"Summary: {abstract}")

        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(topic["Text"])

        if not results:
            return f"[web_search] No results found for: {query!r}"

        return "\n\n".join(results[:max_results])
