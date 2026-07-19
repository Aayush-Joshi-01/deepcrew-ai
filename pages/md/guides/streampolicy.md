# StreamPolicy
Every agent start/done, tool call/result, memory op, retry, fallback, and verifier score is already a `StreamEvent`. `StreamPolicy` controls which of those a given consumer actually sees, without touching execution, logging, or OpenTelemetry spans.

- **StreamPolicy.chat()** (preset): Response text deltas plus the terminal `done`/`error` events only. For simple chatbot UIs.

- **StreamPolicy.standard()** (preset): Chat events plus tool calls/results/denials and agent/step lifecycle. A good default for most apps.

- **StreamPolicy.verbose()** (preset): Every event type — no filtering. For technical/debug UIs.

- **StreamPolicy(include=..., exclude=...)** (custom): Build your own set. Presets always keep `done`/`error` visible; a fully custom `include` set can exclude them too, at your own risk.

```python
from deepcrew import Orchestrator, StreamPolicy

# A simple chatbot only wants the reply text
async for event in orch.stream("Explain quantum entanglement", policy=StreamPolicy.chat()):
    if event.event == "text_delta":
        print(event.data["chunk"], end="", flush=True)

# A technical/debug UI wants everything
async for event in orch.stream("...", policy=StreamPolicy.verbose()):
    print(event.to_dict())
```

`WorkflowBuilder.stream()` takes the same `policy=` keyword.

### Which events are in which preset

The presets are fixed sets defined in `stream.py`, not computed dynamically — here is every `EventType` member and exactly which preset(s) include it.

      | Event | chat() | standard() | verbose() |

        | `text_delta` | &#10003; | &#10003; | &#10003; |

        | `done` | &#10003; | &#10003; | &#10003; |

        | `error` | &#10003; | &#10003; | &#10003; |

        | `agent_start` | &#8212; | &#10003; | &#10003; |

        | `agent_done` | &#8212; | &#10003; | &#10003; |

        | `tool_call` | &#8212; | &#10003; | &#10003; |

        | `tool_result` | &#8212; | &#10003; | &#10003; |

        | `step_start` | &#8212; | &#10003; | &#10003; |

        | `step_done` | &#8212; | &#10003; | &#10003; |

        | `spawn_agent` | &#8212; | &#10003; | &#10003; |

        | `thinking_delta` | &#8212; | &#8212; | &#10003; |

        | `tool_denied` | &#8212; | &#8212; | &#10003; |

        | `retry_attempt` | &#8212; | &#8212; | &#10003; |

        | `fallback_triggered` | &#8212; | &#8212; | &#10003; |

        | `memory_store` | &#8212; | &#8212; | &#10003; |

        | `memory_retrieve` | &#8212; | &#8212; | &#10003; |

        | `loop_iteration` | &#8212; | &#8212; | &#10003; |

        | `apex_start` | &#8212; | &#8212; | &#10003; |

        | `apex_done` | &#8212; | &#8212; | &#10003; |

        | `verifier_scored` | &#8212; | &#8212; | &#10003; |

        | `playbook_updated` | &#8212; | &#8212; | &#10003; |

        | `branch_selected` | &#8212; | &#8212; | &#10003; |

        | `skill_extracted` | &#8212; | &#8212; | &#10003; |

Notice `retry_attempt`/`fallback_triggered`, every Self-Improving Loop event (`loop_iteration`, `verifier_scored`, `playbook_updated`, `branch_selected`, `skill_extracted`), and both memory events are `verbose()`-only — a `standard()` consumer sees an agent producing output and calling tools, but nothing about retries, memory, or self-improvement happening underneath.

### Custom policies and filter_stream()

`allows()` is a plain include-then-exclude check: if `include` is set, the event type must be in it; then it must not be in `exclude`. You can combine both, or use `filter_stream()` standalone against any async generator of `StreamEvent`s — it isn't tied to `Orchestrator`/`WorkflowBuilder`.

```python
from deepcrew import StreamPolicy, filter_stream
from deepcrew.types import EventType

# Only text and tool activity — no lifecycle noise, no terminal events either
# (a fully custom include set is NOT protected the way presets are — see pitfalls)
custom = StreamPolicy(include=frozenset({EventType.TEXT_DELTA, EventType.TOOL_CALL}))

async for event in filter_stream(orch.stream("..."), custom):
    ...

# Or: take a preset and additionally silence one specific event type
quieter = StreamPolicy(include=None, exclude=frozenset({EventType.SPAWN_AGENT}))
```

### Common pitfalls

    - A bare `Agent` run via `run_agent()` has no `policy=` parameter. `StreamPolicy` only wires into `Orchestrator.stream()` and `WorkflowBuilder.stream()`. For a single agent, wrap your own queue with `filter_stream()` manually, or use [FastAPI Integration](fastapi.html), which applies a policy uniformly across all three target types.
    - Presets protect `done`/`error`; fully custom policies don't. If you hand-build a `StreamPolicy(include={...})` that omits `EventType.DONE` and `EventType.ERROR`, your consumer genuinely never learns the stream ended — that's on you, not a bug.
    - Filtering is view-only. Every event still fires, gets logged, and reaches OpenTelemetry spans regardless of policy — `StreamPolicy` only changes what a specific consumer of the event queue sees, never what actually executes.
    - The event-to-preset mapping is fixed, not configurable per event. There's no way to make `standard()` include `verifier_scored` without building your own custom policy from scratch.

### See also

    - [FastAPI Integration](fastapi.html) — applies a `StreamPolicy` uniformly to an Agent, Orchestrator, or WorkflowBuilder behind one SSE endpoint.
    - [Retry & Fallback](retry.html) and [Looping](looping.html) — the features whose events only surface under `verbose()`.
