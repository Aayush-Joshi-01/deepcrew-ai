from __future__ import annotations

from pydantic import BaseModel, Field


class AgentYAML(BaseModel):
    name: str
    model: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    max_turns: int = 10


class WorkflowStepYAML(BaseModel):
    step: str
    agent: str
    task: str = "{input}"
    depends_on: list[str] = Field(default_factory=list)


class WorkflowYAML(BaseModel):
    agents: list[AgentYAML]
    workflow: list[WorkflowStepYAML]
    input: str | None = None
    router_model: str = "openai/gpt-4o-mini"
