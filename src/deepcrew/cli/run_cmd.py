from __future__ import annotations

import asyncio
from typing import Any


def _load_builtin_tools() -> dict[str, Any]:
    """Return a registry of built-in tool-name → Skill instance."""
    try:
        from ..skills.builtin import CodeExecutionSkill, SummarizeSkill, WebSearchSkill
        return {
            "web_search": WebSearchSkill(),
            "summarize": SummarizeSkill(),
            "code_exec": CodeExecutionSkill(),
        }
    except ImportError:
        return {}


async def run_workflow_file(
    path: str,
    input_text: str | None = None,
    stream: bool = True,
) -> None:
    """Load a YAML workflow file, validate it, and execute it."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "pyyaml is required to run workflow files. "
            "Install it with: pip install deepcrew-ai"
        )

    from ..agent import Agent
    from ..workflow import WorkflowBuilder
    from .yaml_schema import WorkflowYAML

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    spec = WorkflowYAML.model_validate(raw)
    tool_registry = _load_builtin_tools()

    agent_map: dict[str, Agent] = {}
    for a in spec.agents:
        skills = [tool_registry[t] for t in a.tools if t in tool_registry]
        agent_map[a.name] = Agent(
            name=a.name,
            model=a.model,
            system_prompt=a.system_prompt,
            temperature=a.temperature,
            max_tokens=a.max_tokens,
            max_turns=a.max_turns,
            skills=skills,
        )

    builder = WorkflowBuilder()
    for step in spec.workflow:
        if step.agent not in agent_map:
            raise ValueError(
                f"Workflow step {step.step!r} references unknown agent {step.agent!r}."
            )
        builder.add_agent(step.step, agent_map[step.agent], task=step.task)
    for step in spec.workflow:
        for dep in step.depends_on:
            builder.then(dep, step.step)

    run_input = input_text or spec.input or ""

    if stream:
        async for event in builder.stream(run_input):
            print(event.to_sse(), end="", flush=True)
    else:
        result = await builder.run(run_input)
        if result.final_output:
            print(result.final_output.text)
