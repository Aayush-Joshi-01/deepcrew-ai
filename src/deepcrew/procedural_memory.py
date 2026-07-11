from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .memory.base import MemoryProvider
    from .types import AgentResult
    from .verifier import VerifierFeedback


@dataclass
class PlaybookEntry:
    """A single distilled strategy in a :class:`ProceduralMemory` playbook."""

    content: str
    kind: Literal["helpful", "harmful"]
    uses: int = 0
    last_score: float | None = None


class ProceduralMemory:
    """
    An ACE-inspired, opt-in evolving playbook of durable strategies, built on
    top of any :class:`MemoryProvider` as its backing store.

    Unlike the raw key/value ``MemoryProvider`` it wraps, ``ProceduralMemory``
    imposes structure: each entry is a "helpful" or "harmful" bullet with a
    usage count and last-seen score. Curation is always incremental — new
    insights are merged into existing entries (bumping their usage/score) or
    appended, and the playbook is pruned to ``max_entries`` by usefulness —
    never rewritten wholesale. That incremental discipline is what avoids the
    "context collapse" failure mode of naive full-context-rewrite approaches.
    """

    def __init__(self, backend: "MemoryProvider", max_entries: int = 30) -> None:
        self._backend = backend
        self._max_entries = max_entries

    async def load(self, task_tag: str) -> list[PlaybookEntry]:
        """Return all persisted entries for *task_tag*, most valuable first."""
        meta_raw = await self._backend.retrieve(self._meta_key(task_tag))
        if not meta_raw:
            return []
        try:
            count = int(json.loads(meta_raw)["count"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return []

        entries: list[PlaybookEntry] = []
        for i in range(count):
            raw = await self._backend.retrieve(self._entry_key(task_tag, i))
            if not raw:
                continue
            try:
                data = json.loads(raw)
                entries.append(PlaybookEntry(**data))
            except (json.JSONDecodeError, TypeError):
                continue
        return entries

    def render(self, entries: list[PlaybookEntry]) -> str:
        """Format *entries* as a compact bullet block for system-prompt injection."""
        if not entries:
            return ""
        lines = ["## Known strategies for this task"]
        for e in entries:
            label = "helpful" if e.kind == "helpful" else "avoid"
            lines.append(f"- ({label}) {e.content}")
        return "\n".join(lines)

    async def curate(
        self,
        task_tag: str,
        feedback: "VerifierFeedback",
        trajectory: list["AgentResult"],
    ) -> int:
        """
        Reflector + Curator step: distill new strategy candidates from
        *feedback*/*trajectory*, merge them into the existing playbook
        (bumping usage/score on near-duplicates instead of adding new
        entries), prune to ``max_entries`` by usefulness, and persist.

        Returns the number of entries in the playbook after curation.
        """
        existing = await self.load(task_tag)

        candidates: list[PlaybookEntry] = []
        for issue in feedback.issues[:3]:
            content = issue.strip()
            if content:
                candidates.append(
                    PlaybookEntry(content=f"Avoid: {content}", kind="harmful", last_score=feedback.score)
                )

        if feedback.score >= 0.8 and trajectory:
            summary = trajectory[-1].text.strip().replace("\n", " ")[:200]
            if summary:
                candidates.append(
                    PlaybookEntry(content=f"Worked well: {summary}", kind="helpful", last_score=feedback.score)
                )

        merged = list(existing)
        for cand in candidates:
            match = next((e for e in merged if self._is_similar(e.content, cand.content)), None)
            if match is not None:
                match.uses += 1
                match.last_score = cand.last_score
            else:
                cand.uses = 1
                merged.append(cand)

        merged.sort(key=lambda e: (e.uses, e.last_score or 0.0), reverse=True)
        pruned = merged[: self._max_entries]

        for i, entry in enumerate(pruned):
            await self._backend.store(self._entry_key(task_tag, i), json.dumps(asdict(entry)))
        await self._backend.store(self._meta_key(task_tag), json.dumps({"count": len(pruned)}))

        return len(pruned)

    @staticmethod
    def _is_similar(a: str, b: str) -> bool:
        a_l, b_l = a.lower().strip(), b.lower().strip()
        if not a_l or not b_l:
            return False
        return a_l == b_l or a_l in b_l or b_l in a_l

    @staticmethod
    def _meta_key(task_tag: str) -> str:
        return f"procedural:{task_tag}:__meta__"

    @staticmethod
    def _entry_key(task_tag: str, index: int) -> str:
        return f"procedural:{task_tag}:{index}"
