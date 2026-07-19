# deepcrew-ai examples

Each script is self-contained and runnable with `python examples/<name>.py`. Check the module
docstring at the top of each file for required environment variables (API keys) before running.

| File | Demonstrates |
|---|---|
| `simple_agent.py` | A single agent with an `@tool`-decorated function. |
| `workflow_example.py` | `WorkflowBuilder` DAG, streaming and non-streaming. |
| `automated_example.py` | `Orchestrator` router + APEX synthesis, streaming events. |
| `mcp_example.py` | All three MCP transports (Stdio/HTTP/SSE) + `MCPManager`. |
| `self_improving_research.py` | `LoopConfig` with a `Verifier`, adaptive early-stop. |
| `consensus_code_review.py` | Self-consistency branching + skill distillation. |
| `autonomous_task_planning.py` | Bounded nested agent spawning (`enable_spawn=True`). |
| `multimodal_agent.py` | Attaching an image and a PDF with `image()`/`pdf()`. |
| `structured_output.py` | `response_model` validated against a pydantic schema. |
| `human_in_the_loop.py` | `AgentHooks.approve_tool` denying a specific tool call. |
| `fastapi_streaming.py` | `create_stream_router` wired into a FastAPI app. |

Full write-ups with expected output: [Examples guide](https://deepcrew-ai.aayushjoshi.dev/examples.html).
