"""
Episodic memory: a durable log of specific past runs — what was asked, what the
agent answered, which tools it used, and how well it went — with similarity +
recency recall of relevant prior experience before a new run.

Distinct from the two existing memory concepts:

* :class:`~deepcrew.MemoryProvider` (``Agent.memory``) — raw substring key/value.
* :class:`~deepcrew.ProceduralMemory` — a distilled *how-to* playbook (helpful /
  harmful bullets), specifics discarded.

``EpisodicMemory`` keeps the specifics. It layers on any ``MemoryProvider`` as a
ring buffer of JSON episodes (``episodic:{namespace}:{i}`` + a ``__meta__`` key),
mirroring :class:`ProceduralMemory`'s storage pattern.

The record/recall cycle is automatic: set ``Agent.episodic_memory`` and every
:func:`~deepcrew.run_agent` call recalls relevant episodes into context before the
run and records the outcome after (so a refinement loop records one episode per
iteration).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from .logging_config import get_logger

if TYPE_CHECKING:
    from .memory.base import MemoryProvider

log = get_logger("episodic_memory")

Outcome = Literal["success", "failure", "partial", "unknown"]

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "it",
        "as",
        "at",
        "by",
        "from",
        "into",
        "about",
        "how",
        "what",
        "why",
        "when",
        "which",
        "who",
        "whom",
        "your",
        "you",
        "i",
        "we",
        "they",
        "he",
        "she",
        "them",
        "our",
    ]
)


@dataclass
class Episode:
    """A single recorded run."""

    query: str
    response: str
    outcome: Outcome = "unknown"
    score: float | None = None
    tools_used: list[str] = field(default_factory=list)
    reflection: str | None = None
    tags: list[str] = field(default_factory=list)
    timestamp: str = ""
    agent_id: str = ""


class EpisodicMemory:
    """A ring buffer of :class:`Episode` records with similarity + recency recall.

    Parameters
    ----------
    backend:
        Any :class:`MemoryProvider` used as the flat key/value store.
    namespace:
        Isolates one episodic log from another in a shared backend.
    max_episodes:
        Ring-buffer capacity. The oldest episode is overwritten once full.
    recall_k:
        Default number of episodes :meth:`recall` returns.
    reflect_model:
        Optional LiteLLM model string. When set, :meth:`record` asks it for a
        one-sentence lesson and stores it on ``episode.reflection``. Off by
        default — no extra LLM call unless you opt in.
    """

    def __init__(
        self,
        backend: MemoryProvider,
        *,
        namespace: str = "default",
        max_episodes: int = 100,
        recall_k: int = 3,
        reflect_model: str | None = None,
    ) -> None:
        self._backend = backend
        self._namespace = namespace
        self._max = max(1, max_episodes)
        self._recall_k = max(1, recall_k)
        self._reflect_model = reflect_model

    # --- persistence -------------------------------------------------

    async def _read_meta(self) -> tuple[int, int]:
        raw = await self._backend.retrieve(self._meta_key())
        if not raw:
            return 0, 0
        try:
            data = json.loads(raw)
            return int(data["count"]), int(data["next"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return 0, 0

    async def load(self) -> list[Episode]:
        """Return every stored episode, oldest first."""
        count, nxt = await self._read_meta()
        if count == 0:
            return []
        # physical slots 0..count-1; logical order starts at the oldest slot
        start = nxt % count if count >= self._max else 0
        order = [(start + i) % count for i in range(count)]
        episodes: list[Episode] = []
        for slot in order:
            raw = await self._backend.retrieve(self._entry_key(slot))
            if not raw:
                continue
            try:
                episodes.append(Episode(**json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                continue
        return episodes

    async def record(self, episode: Episode) -> Episode:
        """Persist *episode* (filling ``timestamp`` and, if configured, ``reflection``)."""
        if not episode.timestamp:
            episode.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if self._reflect_model and episode.reflection is None:
            episode.reflection = await self.reflect(episode)

        count, nxt = await self._read_meta()
        slot = nxt if count >= self._max else count
        new_count = min(count + 1, self._max)
        new_next = (slot + 1) % self._max

        await self._backend.store(self._entry_key(slot), json.dumps(asdict(episode)))
        await self._backend.store(
            self._meta_key(), json.dumps({"count": new_count, "next": new_next})
        )
        log.info(
            "episode recorded",
            event="episode_recorded",
            agent_id=episode.agent_id,
            namespace=self._namespace,
            outcome=episode.outcome,
            score=episode.score,
            reflected=episode.reflection is not None,
        )
        return episode

    # --- recall ----------------------------------------------------

    async def recall(self, query: str, *, top_k: int | None = None) -> list[Episode]:
        """Return up to *top_k* past episodes most relevant to *query*.

        Scoring is dependency-free: keyword (Jaccard) overlap between *query* and
        each episode's query/tags, plus a small recency bonus. The underlying
        provider's substring ``search`` is unusable for ranked recall.
        """
        k = top_k if top_k is not None else self._recall_k
        episodes = await self.load()
        if not episodes:
            return []
        q_terms = _terms(query)
        n = len(episodes)
        scored: list[tuple[float, float, int, Episode]] = []
        for i, ep in enumerate(episodes):
            ep_terms = _terms(ep.query) | {t.lower() for t in ep.tags}
            overlap = _jaccard(q_terms, ep_terms)
            if overlap <= 0.0:
                continue  # no topical relevance -> not "relevant past experience"
            recency = (i + 1) / n * 0.15  # newest gets +0.15 at most as a tie-breaker
            scored.append((overlap + recency, overlap, i, ep))
        scored.sort(key=lambda t: (t[0], t[2]), reverse=True)
        return [ep for _, _, _, ep in scored[:k]]

    def render(self, episodes: list[Episode]) -> str:
        """Format *episodes* as a compact system-prompt block."""
        if not episodes:
            return ""
        lines = ["## Relevant past experience"]
        for ep in episodes:
            tag = f"{ep.outcome}" + (f", {ep.score:.2f}" if ep.score is not None else "")
            summary = (ep.reflection or ep.response).strip().replace("\n", " ")[:220]
            lines.append(f"- [{tag}] Q: {ep.query.strip()[:120]} -> {summary}")
        return "\n".join(lines)

    async def reflect(self, episode: Episode) -> str | None:
        """One LLM call distilling a one-sentence lesson. ``None`` without ``reflect_model``."""
        if not self._reflect_model:
            return None
        try:
            import litellm

            resp = await litellm.acompletion(
                model=self._reflect_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You distill a single, reusable one-sentence lesson from one "
                            "agent run. Respond with ONLY that sentence, no preamble."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Query: {episode.query}\n"
                            f"Outcome: {episode.outcome}"
                            + (f" (score {episode.score:.2f})" if episode.score is not None else "")
                            + f"\nAnswer: {episode.response[:1500]}"
                        ),
                    },
                ],
                stream=False,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or None
        except Exception:
            log.warning("episode reflection failed", namespace=self._namespace, exc_info=True)
            return None

    async def clear(self) -> None:
        """Drop every episode in this namespace (stale slots are simply ignored by ``load``)."""
        await self._backend.store(self._meta_key(), json.dumps({"count": 0, "next": 0}))

    # --- keys ----------------------------------------------------

    def _meta_key(self) -> str:
        return f"episodic:{self._namespace}:__meta__"

    def _entry_key(self, index: int) -> str:
        return f"episodic:{self._namespace}:{index}"


def _terms(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
