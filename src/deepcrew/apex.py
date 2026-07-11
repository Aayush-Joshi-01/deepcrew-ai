from __future__ import annotations

import asyncio
from dataclasses import dataclass

import litellm

from .types import AgentResult, EventType, StreamEvent, ToolDef

_APEX_SYSTEM = """\
You are APEX, an expert synthesis specialist.
You receive results from multiple specialized agents who worked in parallel on a query.

Your task:
1. Synthesize all findings into a single, cohesive, well-structured response.
2. Preserve accuracy and nuance from each source.
{citation_instruction}
3. Assess your own confidence in the final answer on a scale from 0.0 (uncertain) to 1.0 (certain).

At the very end of your response, on a new line, output ONLY:
CONFIDENCE: <float between 0.0 and 1.0>
"""

_APEX_CITATION_INSTRUCTION = (
    "When a fact comes primarily from one agent, mark it inline as [source: agent_name]."
)


@dataclass
class ApexConfig:
    """Configuration for the APEX synthesizer."""

    confidence_threshold: float = 0.7
    cite_sources: bool = True
    allow_tools: bool = False
    system_prompt: str | None = None


@dataclass
class ApexCitation:
    """A fact attributed to a specific agent."""

    agent_id: str
    claim: str
    confidence: float


class APEXSynthesizer:
    """
    APEX — the intelligent synthesis engine for deepcrew-ai.

    Merges parallel agent outputs into a single cohesive response with
    confidence scoring and optional source citation.
    """

    def __init__(
        self,
        model: str,
        config: ApexConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config or ApexConfig()

    async def synthesize(
        self,
        original_query: str,
        results: list[AgentResult],
        queue: asyncio.Queue[StreamEvent | None] | None = None,
        tool_defs: list[ToolDef] | None = None,
    ) -> AgentResult:
        """Synthesize multiple agent results into one authoritative response."""
        if queue:
            await queue.put(StreamEvent(EventType.APEX_START, {"agents": len(results)}, "apex"))

        cfg = self._config
        citation_instr = _APEX_CITATION_INSTRUCTION if cfg.cite_sources else ""
        system = cfg.system_prompt or _APEX_SYSTEM.format(citation_instruction=citation_instr)

        parts = [f"Original query: {original_query}\n"]
        for r in results:
            parts.append(f"--- Agent: {r.agent_id} ---\n{r.text}\n")
        synthesis_prompt = "\n".join(parts)

        if cfg.allow_tools and tool_defs:
            from .agent import Agent
            from .runner import run_agent

            apex_agent = Agent(
                name="apex",
                model=self._model,
                system_prompt=system,
            )
            result = await run_agent(
                apex_agent,
                [{"role": "user", "content": synthesis_prompt}],
                tool_defs=tool_defs,
                queue=queue,
                agent_id="apex",
            )
            text = result.text
            in_tok = result.input_tokens
            out_tok = result.output_tokens
        else:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": synthesis_prompt},
                ],
                stream=False,
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) or 0
            out_tok = getattr(usage, "completion_tokens", 0) or 0

        confidence = self._extract_confidence(text)
        clean_text = self._strip_confidence_line(text)

        apex_result = AgentResult(
            agent_id="apex",
            text=clean_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=self._model,
            confidence=confidence,
        )

        if queue:
            await queue.put(
                StreamEvent(
                    EventType.APEX_DONE,
                    {"confidence": confidence},
                    "apex",
                )
            )

        return apex_result

    def _extract_confidence(self, text: str) -> float:
        import re

        m = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
        return 0.8  # sensible default when not parseable

    def _strip_confidence_line(self, text: str) -> str:
        import re

        return re.sub(r"\nCONFIDENCE:\s*[\d.]+\s*$", "", text).rstrip()

    def build_citations(self, results: list[AgentResult], synthesis: str) -> list[ApexCitation]:
        """Extract inline [source: agent_name] citations from the synthesis text."""
        import re

        citations: list[ApexCitation] = []
        for match in re.finditer(r"\[source:\s*(\w+)\]", synthesis):
            agent_id = match.group(1)
            start = max(0, match.start() - 120)
            claim = synthesis[start : match.start()].strip().split(".")[-1].strip()
            citations.append(ApexCitation(agent_id=agent_id, claim=claim, confidence=0.9))
        return citations
