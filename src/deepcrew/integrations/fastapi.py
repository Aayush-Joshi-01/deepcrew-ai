"""
Optional FastAPI integration: turn an Agent, Orchestrator, or WorkflowBuilder
into a streaming SSE chat endpoint in one call.

Requires the ``fastapi`` extra::

    pip install deepcrew-ai[fastapi]

Example
-------
::

    from fastapi import FastAPI
    from deepcrew import Agent, StreamPolicy
    from deepcrew.integrations.fastapi import create_stream_router

    agent = Agent(name="assistant", model="openai/gpt-4o")
    app = FastAPI()
    app.include_router(create_stream_router(agent, policy=StreamPolicy.standard()))
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncGenerator
from typing import Any

try:
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised via import-time test
    raise ImportError(
        "fastapi is required for deepcrew.integrations.fastapi. "
        "Install it with: pip install deepcrew-ai[fastapi]"
    ) from exc

from ..agent import Agent
from ..content import ContentPart, TextPart, image, pdf
from ..exceptions import ContentError
from ..orchestrator import Orchestrator
from ..runner import run_agent
from ..stream import StreamPolicy, filter_stream, make_error_event, queue_to_stream
from ..types import EventType, StreamEvent
from ..workflow import WorkflowBuilder

_POLICY_PRESETS: dict[str, StreamPolicy] = {
    "chat": StreamPolicy.chat(),
    "standard": StreamPolicy.standard(),
    "verbose": StreamPolicy.verbose(),
}


class ChatRequest(BaseModel):
    """Request body for both the streaming and ``/complete`` endpoints."""

    query: str
    images: list[str] = []
    pdfs: list[str] = []
    policy: str | None = None


def _build_attachments(req: ChatRequest) -> list[ContentPart]:
    try:
        attachments: list[ContentPart] = [image(url) for url in req.images]
        attachments.extend(pdf(url) for url in req.pdfs)
    except ContentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return attachments


def _resolve_policy(
    req: ChatRequest, default_policy: StreamPolicy, allow_override: bool
) -> StreamPolicy:
    if not allow_override or req.policy is None:
        return default_policy
    preset = _POLICY_PRESETS.get(req.policy)
    if preset is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown policy {req.policy!r}; expected one of {sorted(_POLICY_PRESETS)}",
        )
    return preset


def _agent_user_message(query: str, attachments: list[ContentPart]) -> dict[str, Any]:
    if not attachments:
        return {"role": "user", "content": query}
    return {
        "role": "user",
        "content": [TextPart(query).to_block(), *(p.to_block() for p in attachments)],
    }


async def _run_agent_to_queue(
    agent: Agent,
    messages: list[dict[str, Any]],
    queue: asyncio.Queue[StreamEvent | None],
) -> None:
    """Run a bare Agent and terminate its queue the same way Orchestrator does."""
    try:
        result = await run_agent(agent, messages, queue=queue)
        await queue.put(StreamEvent(EventType.DONE, {"final_text": result.text}, agent.name))
    except Exception as exc:
        await queue.put(make_error_event(agent.name, str(exc)))
    finally:
        await queue.put(None)


def create_stream_router(
    target: Agent | Orchestrator | WorkflowBuilder,
    *,
    path: str = "/chat",
    policy: StreamPolicy | None = None,
    allow_policy_override: bool = False,
) -> APIRouter:
    """
    Build a FastAPI router exposing ``target`` as an SSE streaming chat
    endpoint (``POST {path}``) plus a non-streaming ``POST {path}/complete``.

    ``target`` may be a bare :class:`~deepcrew.Agent`,
    :class:`~deepcrew.Orchestrator`, or :class:`~deepcrew.WorkflowBuilder`.
    ``WorkflowBuilder`` does not accept multimodal attachments; sending
    ``images``/``pdfs`` to a workflow endpoint returns ``422``.

    Parameters
    ----------
    path:
        Route path for the streaming endpoint. The non-streaming endpoint is
        ``f"{path}/complete"``.
    policy:
        Default :class:`~deepcrew.StreamPolicy` applied to the stream.
        Defaults to :meth:`StreamPolicy.chat`.
    allow_policy_override:
        If True, the request body's ``policy`` field (one of ``"chat"``,
        ``"standard"``, ``"verbose"``) overrides the default per-request.
    """
    default_policy = policy or StreamPolicy.chat()
    router = APIRouter()

    def _build_raw_stream(req: ChatRequest, attachments: list[ContentPart]) -> Any:
        if isinstance(target, Orchestrator):
            return target.stream(req.query, attachments=attachments or None)
        if isinstance(target, WorkflowBuilder):
            if attachments:
                raise HTTPException(
                    status_code=422, detail="WorkflowBuilder does not accept attachments"
                )
            return target.stream(req.query)

        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        message = _agent_user_message(req.query, attachments)
        task = asyncio.create_task(_run_agent_to_queue(target, [message], queue))
        return queue_to_stream(queue, task)

    async def _sse_body(
        raw_stream: Any, effective_policy: StreamPolicy
    ) -> AsyncGenerator[str, None]:
        async for event in filter_stream(raw_stream, effective_policy):
            yield event.to_sse()
        yield "event: done\ndata: {}\n\n"

    @router.post(path)
    async def chat(req: ChatRequest) -> StreamingResponse:
        # Validate before the response starts: once StreamingResponse begins
        # sending headers, an exception raised from inside the body iterator
        # can no longer become a clean 4xx.
        attachments = _build_attachments(req)
        effective_policy = _resolve_policy(req, default_policy, allow_policy_override)
        raw_stream = _build_raw_stream(req, attachments)
        return StreamingResponse(
            _sse_body(raw_stream, effective_policy), media_type="text/event-stream"
        )

    @router.post(f"{path}/complete")
    async def complete(req: ChatRequest) -> dict[str, Any]:
        attachments = _build_attachments(req)

        if isinstance(target, Orchestrator):
            orch_result = await target.run(req.query, attachments=attachments or None)
            return dataclasses.asdict(orch_result)
        if isinstance(target, WorkflowBuilder):
            if attachments:
                raise HTTPException(
                    status_code=422, detail="WorkflowBuilder does not accept attachments"
                )
            workflow_result = await target.run(req.query)
            return dataclasses.asdict(workflow_result)

        message = _agent_user_message(req.query, attachments)
        agent_result = await run_agent(target, [message])
        return dataclasses.asdict(agent_result)

    return router
