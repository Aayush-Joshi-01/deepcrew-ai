# Observability
deepcrew-ai emits OpenTelemetry spans for every LLM call, tool execution, and workflow step. When `observability=None` (the default), all span context managers are `nullcontext()` — absolutely zero overhead.

### Installation

```python
pip install "deepcrew-ai[otel]"
# Installs: opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-otlp
```

### Quick start with Jaeger

```python
from deepcrew import Agent, run_agent, ObservabilityConfig

obs = ObservabilityConfig(
    otel_endpoint="http://localhost:4317",  # gRPC endpoint
    service_name="my-ai-app",
    enabled=True,
    export_format="grpc",   # or "http" for HTTP/protobuf
)

agent = Agent("researcher", model="openai/gpt-4o-mini", system_prompt="Research thoroughly.")
result = await run_agent(
    agent,
    [{"role": "user", "content": "Explain blockchain technology"}],
    observability=obs,
)
```

> Note: `Orchestrator` does not currently accept an `ObservabilityConfig` at all — its `run()`/`stream()` methods have no `observability` parameter, and it never passes one through to the individual `run_agent()` calls it makes internally. If you need OTel spans around an orchestrated run today, wrap the whole `orch.run(...)` call in your own manually-created span, or use `WorkflowBuilder` instead (see below), which does support it directly.

### With run_agent()

```python
from deepcrew import Agent, run_agent, ObservabilityConfig

obs = ObservabilityConfig(otel_endpoint="http://localhost:4317")

result = await run_agent(
    agent,
    [{"role": "user", "content": "Hello!"}],
    observability=obs,   # that's it
)

# Spans emitted (see "Span attributes" below for exact names —
# there is no "deepcrew." prefix on the span name itself):
#   agent.run     → covers entire agent lifecycle
#     llm.call    → each LLM request, one per turn
#     tool.call   → each turn's tool executions, grouped into one span
```

### With WorkflowBuilder

```python
from deepcrew import WorkflowBuilder, ObservabilityConfig

obs = ObservabilityConfig(otel_endpoint="http://localhost:4317", service_name="workflow-app")

workflow = (
    WorkflowBuilder(observability=obs)  # pass at construction time
    .add_agent("step1", agent1, task="{input}")
    .add_agent("step2", agent2, task="Refine:\n{step1}")
    .then("step1", "step2")
)

result = await workflow.run("My query")

# Spans emitted per step:
#   workflow.step   → covers each DAG node
#     agent.run     → the agent within the step
#       llm.call
#       tool.call
```

### Start Jaeger locally (Docker)

```python
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 6831:6831/udp \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Open http://localhost:16686 to view traces
```

### ObservabilityConfig reference

- **otel_endpoint** (str | None): OTLP collector endpoint. For gRPC: `http://localhost:4317`. For HTTP: `http://localhost:4318/v1/traces`.

- **service_name** (str = "deepcrew"): Service name in traces. Use your app name for easy filtering in Jaeger/Grafana.

- **enabled** (bool = True): Master switch. Set to False for no-op without removing the ObservabilityConfig object.

- **export_format** ("grpc" | "http"): OTLP export protocol. `"grpc"` for port 4317, `"http"` for port 4318.

### Span attributes

Span names are exactly as shown below — there is no `"deepcrew."` or other namespace prefix added to them. Token counts and tool names are not set as span attributes; if you need per-call token accounting, read it off the returned `AgentResult.input_tokens`/`output_tokens` instead, or add your own OTel instrumentation around that.

      | Span name | Attributes actually set | Wraps |

        | `agent.run` | `agent.id`, `agent.model` | The entire agentic loop for one agent, all turns. |

        | `llm.call` | `llm.model`, `agent.id` | One `litellm.acompletion` call — one per turn. |

        | `tool.call` | `tool.name` (comma-joined if multiple tools ran that turn), `agent.id` | All tool calls executed in parallel within one turn, as a single span — not one span per individual tool call. |

        | `workflow.step` | `step.name` | One DAG node in a `WorkflowBuilder` run. Does not carry an `agent.id` attribute even though it wraps an `agent.run` span. |

### Common pitfalls

    - `Orchestrator` has no observability hook at all yet. Neither the router call, the fan-out agents, nor APEX synthesis get wrapped in spans — see the callout above.
    - `tool.call` is one span per turn, not per tool. If an agent calls three tools in parallel in one turn, that's a single span with a comma-joined `tool.name` attribute, not three separate spans.
    - Token counts aren't in the spans. Pull them from `AgentResult` after the call, or instrument separately.
    - `enabled=False` is the cheap way to disable, not deleting the config. Every span helper checks `config is None or not config.enabled` and falls through to a plain `nullcontext()` — flip `enabled` off for a quick toggle without restructuring your code.
