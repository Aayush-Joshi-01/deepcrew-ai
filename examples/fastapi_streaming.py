"""
FastAPI SSE streaming endpoint for a deepcrew Agent.

Requires: OPENAI_API_KEY environment variable, and the fastapi extra:
    pip install deepcrew-ai[fastapi] uvicorn

Run: uvicorn examples.fastapi_streaming:app --reload
Then:
    curl -N -X POST http://localhost:8000/chat \\
        -H "Content-Type: application/json" \\
        -d '{"query": "What is the capital of France?"}'

    curl -X POST http://localhost:8000/chat/complete \\
        -H "Content-Type: application/json" \\
        -d '{"query": "What is the capital of France?"}'
"""

from fastapi import FastAPI

from deepcrew import Agent, StreamPolicy
from deepcrew.integrations.fastapi import create_stream_router

agent = Agent(
    name="assistant",
    model="openai/gpt-4o",
    system_prompt="You are a helpful, concise assistant.",
)

app = FastAPI(title="deepcrew-ai FastAPI streaming example")

# Chat-only visibility by default; allow a client to opt into "standard" or
# "verbose" per-request via the request body's `policy` field.
app.include_router(
    create_stream_router(
        agent,
        path="/chat",
        policy=StreamPolicy.chat(),
        allow_policy_override=True,
    )
)
