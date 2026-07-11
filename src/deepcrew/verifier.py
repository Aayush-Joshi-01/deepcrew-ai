from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import litellm

from .types import AgentResult

_VERIFIER_SYSTEM = """\
You are a strict quality verifier for an AI agent's answer.
Grade how well the answer addresses the original query.
{rubric_section}
Respond with ONLY valid JSON, exactly in this form:
{{"score": <float 0.0-1.0>, "issues": ["<specific problem>", ...], "suggestion": "<concrete next step to improve the answer>"}}

If the answer is already excellent, return an empty "issues" list and a score close to 1.0.
"""

_RUBRIC_SECTION = "Grading criteria:\n{rubric}\n"


@dataclass
class VerifierFeedback:
    """Structured critique produced by a :class:`Verifier`."""

    score: float
    issues: list[str] = field(default_factory=list)
    suggestion: str = ""
    converged: bool = False


@dataclass
class VerifierConfig:
    """Configuration for the :class:`Verifier`."""

    model: str | None = None
    threshold: float = 0.8
    rubric: str | None = None
    evaluate_fn: Callable[[str, AgentResult], Awaitable[VerifierFeedback]] | None = None


class Verifier:
    """
    Grades an agent's result against the original query, producing structured
    feedback (score + specific issues + a concrete suggestion) instead of a
    single boolean convergence check.
    """

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self._config = config or VerifierConfig()

    async def evaluate(
        self,
        query: str,
        result: AgentResult,
        *,
        default_model: str,
    ) -> VerifierFeedback:
        """Grade *result* against *query*, returning structured feedback."""
        cfg = self._config

        if cfg.evaluate_fn is not None:
            return await cfg.evaluate_fn(query, result)

        model = cfg.model or default_model
        rubric_section = _RUBRIC_SECTION.format(rubric=cfg.rubric) if cfg.rubric else ""
        system = _VERIFIER_SYSTEM.format(rubric_section=rubric_section)
        prompt = f"Original query: {query}\n\nAnswer to grade:\n{result.text}"

        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = self._parse_json(raw)
        except Exception:
            data = {}

        score = self._coerce_score(data.get("score"))
        issues = data.get("issues") if isinstance(data.get("issues"), list) else []
        issues = [str(i) for i in issues]
        suggestion = str(data.get("suggestion") or "")

        return VerifierFeedback(
            score=score,
            issues=issues,
            suggestion=suggestion,
            converged=score >= cfg.threshold,
        )

    async def assess_complexity(self, task: str, *, default_model: str) -> bool:
        """
        Best-effort check for whether *task* looks complex enough to warrant
        further sub-delegation. Defaults to permissive (True) on any failure
        to parse a judgment, since this is only ever used as an optional gate.
        """
        cfg = self._config
        if cfg.evaluate_fn is not None:
            # evaluate_fn-only verifiers have no notion of pre-execution
            # complexity assessment; stay permissive.
            return True

        model = cfg.model or default_model
        prompt = (
            f"Task: {task}\n\n"
            "Does this task genuinely require decomposing into smaller sub-tasks "
            "handled by separate sub-agents, or can one agent handle it directly? "
            'Respond with ONLY valid JSON: {"needs_decomposition": true|false}'
        )
        try:
            resp = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or "{}"
            data = self._parse_json(raw)
            value = data.get("needs_decomposition")
            if isinstance(value, bool):
                return value
        except Exception:
            pass
        return True

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _coerce_score(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
