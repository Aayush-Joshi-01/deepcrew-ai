# deepcrew-ai

Multi-agent spawning library for Python. Build parallel AI workflows using any LLM provider, with MCP tool integration and SSE streaming — all with minimal boilerplate.

```bash
pip install deepcrew-ai
```

---

## Features

- **100+ LLM providers** via [LiteLLM](https://github.com/BerriAI/litellm) — OpenAI, Anthropic, Gemini, Bedrock, Azure, and more
- **Two orchestration modes** — explicit DAG workflows or automated AI-driven routing
- **Parallel execution** — independent agents always run concurrently via `asyncio.gather`
- **MCP tool integration** — stdio (subprocess), SSE (legacy), and streamable-HTTP transports
- **Python function tools** — decorate any function with `@tool`, schema auto-generated from type hints
- **SSE streaming** — stream events as they happen, compatible with FastAPI `StreamingResponse`
- **PyPI-ready** — clean package structure, no extra config needed

---

## Quick Start

```python
import asyncio
from deepcrew import Agent, run_agent, tool

@tool
def get_weather(city: str) -> dict:
    "Get current weather."
    return {"city": city, "temp": 22, "condition": "sunny"}

async def main():
    agent = Agent(
        name="assistant",
        model="openai/gpt-4o",
        system_prompt="You are a helpful assistant.",
        tools=[get_weather],
    )
    result = await run_agent(agent, [{"role": "user", "content": "Weather in Paris?"}])
    print(result.text)

asyncio.run(main())
```

---

## Orchestration Modes

### 1. Workflow Builder (explicit DAG)

Define agents and their dependencies. Independent nodes run in parallel automatically.

```python
from deepcrew import Agent, WorkflowBuilder

researcher = Agent("researcher", model="openai/gpt-4o-mini",
                   system_prompt="Research the topic thoroughly.")
critic     = Agent("critic",     model="openai/gpt-4o-mini",
                   system_prompt="Find gaps and weaknesses in the research.")
writer     = Agent("writer",     model="openai/gpt-4o",
                   system_prompt="Write a polished report.")

workflow = (
    WorkflowBuilder()
    .add_agent("research", researcher, task="{input}")
    .add_agent("critique", critic,     task="Critique this research:\n{research}")
    .add_agent("report",   writer,     task="Write a report using:\n{research}\n\nCritique:\n{critique}")
    .then("research", "critique")  # critique waits for research
    .then("research", "report")    # report waits for research
    .then("critique", "report")    # report waits for critique
)

# research runs first, then critique + expand in PARALLEL, then report
result = await workflow.run("The future of renewable energy")
print(result.final_output.text)
print(f"Total tokens: {result.total_tokens}")
```

### 2. Automated Orchestrator (AI-driven routing)

The router LLM decides at runtime: use one agent, or fan out to multiple in parallel.

```python
from deepcrew import Agent, Orchestrator

agents = [
    Agent("researcher", model="openai/gpt-4o-mini", system_prompt="Research specialist."),
    Agent("analyst",    model="openai/gpt-4o-mini", system_prompt="Data analyst."),
    Agent("writer",     model="openai/gpt-4o-mini", system_prompt="Content writer."),
]

orch = Orchestrator(
    agents=agents,
    router_model="openai/gpt-4o-mini",
    synthesizer_model="openai/gpt-4o",
)

# Non-streaming
result = await orch.run("Summarize the state of quantum computing")
print(result.final_text)

# Streaming (yields StreamEvent objects)
async for event in orch.stream("Summarize the state of quantum computing"):
    print(event.to_sse(), end="")  # ready for FastAPI StreamingResponse
```

---

## MCP Tools

### Stdio MCP (subprocess)

```python
from deepcrew import Agent, run_agent
from deepcrew.mcp import StdioMCP

async with StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]) as mcp:
    agent = Agent("file_agent", model="openai/gpt-4o", mcps=[mcp])
    result = await run_agent(agent, [{"role": "user", "content": "List files here"}])
```

### HTTP MCP (streamable-HTTP, modern)

```python
from deepcrew.mcp import HTTPMCP

async with HTTPMCP("https://my-mcp.example.com/mcp",
                   headers={"Authorization": "Bearer sk-..."}) as mcp:
    agent = Agent("agent", model="openai/gpt-4o", mcps=[mcp])
```

### SSE MCP (legacy)

```python
from deepcrew.mcp import SSEMCP

async with SSEMCP("http://localhost:3000") as mcp:
    agent = Agent("agent", model="openai/gpt-4o", mcps=[mcp])
```

### MCPManager (multiple servers)

```python
from deepcrew.mcp import MCPManager, StdioMCP, HTTPMCP

async with MCPManager([
    StdioMCP("npx", ["-y", "@modelcontextprotocol/server-filesystem", "."]),
    HTTPMCP("https://search-mcp.example.com/mcp"),
]) as manager:
    tools = await manager.discover_tools()
    agent = Agent("agent", model="openai/gpt-4o", mcps=[manager])
```

---

## Python Function Tools

```python
from deepcrew import tool, fn_to_tool_def

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.

    Args:
        expression (str): A Python math expression like '2 ** 10'.
    """
    return eval(expression)  # noqa: S307 — replace with safe parser in production

# Or without decorator (schema still auto-generated):
def search_db(query: str, limit: int = 10) -> list:
    ...

agent = Agent("agent", model="openai/gpt-4o", tools=[calculate, search_db])
```

---

## Streaming with FastAPI

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from deepcrew import Orchestrator, Agent

app = FastAPI()
orch = Orchestrator([Agent("assistant", model="openai/gpt-4o")])

@app.post("/chat")
async def chat(query: str):
    async def event_stream():
        async for event in orch.stream(query):
            yield event.to_sse()
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## Stream Events

All events have `event`, `agent_id`, and a `data` dict:

| Event | Data keys | Meaning |
|---|---|---|
| `agent_start` | `model` | Agent loop begins |
| `text_delta` | `chunk` | Incremental text from LLM |
| `thinking_delta` | `chunk`, `tokens` | Thinking/reasoning text |
| `tool_call` | `tool`, `args` | Tool about to be called |
| `tool_result` | `tool`, `result` | Tool call completed |
| `agent_done` | `input_tokens`, `output_tokens` | Agent finished |
| `step_start` | `node` | Workflow node beginning |
| `step_done` | `node` | Workflow node completed |
| `error` | `message` | An error occurred |
| `done` | `final_text` | Entire run complete |

---

## Agent Configuration

```python
Agent(
    name="my_agent",
    model="anthropic/claude-opus-4-8",   # any litellm model string
    system_prompt="You are...",
    mcps=[mcp1, mcp2],                   # MCP tool servers
    tools=[my_fn, another_fn],           # Python function tools
    max_turns=10,                         # LLM→tool→LLM cycles
    temperature=0.7,
    max_tokens=4096,
    extra_params={"thinking": {"type": "enabled", "budget_tokens": 5000}},
)
```

---

## Provider Examples

```python
# OpenAI
Agent("a", model="openai/gpt-4o")
Agent("a", model="openai/gpt-4o-mini")

# Anthropic
Agent("a", model="anthropic/claude-opus-4-8")
Agent("a", model="anthropic/claude-haiku-4-5-20251001")

# Google
Agent("a", model="gemini/gemini-2.0-flash")
Agent("a", model="gemini/gemini-2.5-pro")

# AWS Bedrock
Agent("a", model="bedrock/anthropic.claude-opus-4-8-20250514-v1:0")

# Azure OpenAI
Agent("a", model="azure/gpt-4o", extra_params={"api_base": "https://..."})

# Local (Ollama)
Agent("a", model="ollama/llama3.2")
```

---

## Installation

```bash
pip install deepcrew-ai
```

With development dependencies:

```bash
pip install deepcrew-ai[dev]
```

Requires Python 3.11+.

---

## License

MIT
