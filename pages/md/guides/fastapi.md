# FastAPI Integration
Ship a streaming endpoint in one call. Requires the `fastapi` extra: `pip install deepcrew-ai[fastapi]` — it is never imported unless you use it.

```python
from fastapi import FastAPI
from deepcrew import Agent, StreamPolicy
from deepcrew.integrations.fastapi import create_stream_router

agent = Agent(name="assistant", model="openai/gpt-4o")
app = FastAPI()
app.include_router(create_stream_router(agent, policy=StreamPolicy.chat()))
# POST /chat streams Server-Sent Events; POST /chat/complete returns final JSON.
```

`create_stream_router` also accepts an `Orchestrator` or `WorkflowBuilder` as the target. The request body accepts `query`, optional `images`/`pdfs` (URLs or data URIs), and — if `allow_policy_override=True` — a per-request `policy` name.

### create_stream_router() reference

- **target*** (Agent | Orchestrator | WorkflowBuilder): Dispatch is by `isinstance` check. A bare `Agent` is wrapped with the same queue-plus-background-task pattern `Orchestrator` uses internally, so streaming behaves consistently across all three target types.

- **path** (str = "/chat"): Route for the streaming endpoint. The non-streaming endpoint is always `f"{path}/complete"` — with the default, that's `/chat/complete`.

- **policy** (StreamPolicy | None = None): Default applied to every request. Falls back to `StreamPolicy.chat()` if omitted — the router is opinionated toward "just the reply text" out of the box, not verbose by default.

- **allow_policy_override** (bool = False): When True, the request body's `policy` field (one of the literal strings `"chat"`, `"standard"`, or `"verbose"`) overrides the router's default for that one request. An unrecognized policy name returns `422`. There is no way to send a fully custom `include`/`exclude` set over the wire — only the three preset names.

### Request/response shapes

      | Field | Type | Notes |

        | `query` | str | Required. The text prompt. |

        | `images` | list[str] = [] | Each a URL or `data:` URI, built into an `ImagePart` via `image()`. Local file paths are not accepted over HTTP — send bytes as a data URI instead. |

        | `pdfs` | list[str] = [] | Same URL/data-URI-only rule, via `pdf()`. |

        | `policy` | str | None | Only read when the router was created with `allow_policy_override=True`. |

`POST {path}` streams `text/event-stream`: each `StreamEvent` serialized via its existing `to_sse()` method, followed by one final `event: done\ndata: {}\n\n` sentinel line appended by the router itself — this is separate from and always sent in addition to any `EventType.DONE` event your policy already lets through, so a client can rely on it unconditionally as the true end-of-stream marker regardless of policy. `POST {path}/complete` runs the same target non-streaming and returns the final result as JSON (a plain dict from `dataclasses.asdict()` on the `AgentResult`/`OrchestratorResult`/`WorkflowResult`).

### Trying it with curl

```python
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain quantum entanglement"}'

curl -X POST http://localhost:8000/chat/complete \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain quantum entanglement"}'
```

### WorkflowBuilder and attachments

`WorkflowBuilder` has no `attachments` parameter anywhere in deepcrew — sending `images`/`pdfs` in a request to a router built over a `WorkflowBuilder` target returns `422` before anything runs, rather than silently dropping the attachments.

### Common pitfalls

    - Validation happens before the stream starts, deliberately. Invalid attachments or an unrecognized policy name are checked and rejected with a clean `422` before the `StreamingResponse` begins sending headers — once a streaming response starts, an exception raised from inside the body generator can no longer become a clean HTTP error code, only a broken connection.
    - The default policy is `chat()`, not `verbose()`. If your endpoint seems to be missing tool-call events, check whether you passed a more permissive `policy=` or set `allow_policy_override=True`.
    - Only preset names are settable per-request. A custom `StreamPolicy(include=..., exclude=...)` can only be set as the router's fixed default at construction time — there's no JSON representation for it in the request body.
    - Images/PDFs must be URLs or data URIs over HTTP. Local filesystem paths that work fine when calling `image()`/`pdf()` directly in Python are meaningless to a remote client — encode as base64 data URIs instead.

### See also

    - [StreamPolicy](streampolicy.html) — the presets and event-visibility model this router applies uniformly.
    - [Multimodal Input](multimodal.html) — how `image()`/`pdf()` validate and encode the `images`/`pdfs` request fields.
