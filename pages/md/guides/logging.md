# Logging

deepcrew-ai ships a general-purpose structured logging layer that is **off by default** —
a bare `import deepcrew` attaches only a `NullHandler` and emits nothing, following the
Python logging guidance for libraries. Turn it on with one call.

This is separate from [Observability](observability.md): `ObservabilityConfig` is
OpenTelemetry **spans** (tracing/metrics); this is plain human/JSON **logs**. They are
independent and can be used together. Logging also never touches the `StreamEvent` queue —
the streaming view is unaffected.

### Turn it on

```python
from deepcrew import configure_logging

configure_logging()  # the "dev" profile: colour, DEBUG, timestamps
configure_logging("prod")  # JSON lines, INFO, no colour
configure_logging("verbose")  # everything, including TRACE token-level detail
configure_logging(level="DEBUG", format="json")  # ad-hoc knobs
```

Or entirely from the environment, with no code change:

```bash
DEEPCREW_LOG_PROFILE=dev python my_app.py
DEEPCREW_LOG_LEVEL=DEBUG DEEPCREW_LOG_FORMAT=json python my_app.py
```

When any `DEEPCREW_LOG*` variable is set, logging auto-configures on first use.

### Profiles

A profile just presets four knobs; each is individually overridable, and `DEEPCREW_LOG*`
env vars always win over both the profile and explicit arguments.

| Profile | Level | Format | Colour | Timestamps |
|---|---|---|---|---|
| `dev` | DEBUG | text | on | on |
| `prod` | INFO | JSON | off | on (ISO-8601 UTC) |
| `verbose` | TRACE | text | on | on |
| `quiet` | WARNING | text | auto | off |

### Levels

deepcrew-ai uses the standard library level names — no custom levels — plus `TRACE` (5),
registered the same way `structlog` and `loguru` add it.

| Level | Value | Used for |
|---|---|---|
| `CRITICAL` | 50 | process cannot continue |
| `ERROR` | 40 | an operation failed (`max_turns` exhausted, orchestrator crash) |
| `WARNING` | 30 | recoverable / degraded (retry, model fallback, spawn-depth ceiling) |
| `INFO` | 20 | lifecycle milestones (agent start/done, routing decision, loop iteration, APEX) |
| `DEBUG` | 10 | per-turn / per-tool detail |
| `TRACE` | 5 | token-level deltas |

### Environment variables

| Variable | Effect |
|---|---|
| `DEEPCREW_LOG_PROFILE` | `dev` / `prod` / `verbose` / `quiet` |
| `DEEPCREW_LOG_LEVEL` | `TRACE` / `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `DEEPCREW_LOG_FORMAT` | `text` or `json` |
| `DEEPCREW_LOG_TIME` | `1` / `0` — show timestamps |
| `DEEPCREW_LOG_COLOR` | `1` / `0` / `auto` |

Colour also honours the community `NO_COLOR` and `FORCE_COLOR` conventions and falls back
to plain text when stdout is not a TTY.

### Colour rendering

`configure_logging` uses [`rich`](https://rich.readthedocs.io) for coloured output when it
is installed and colour is enabled; otherwise it falls back to a plain ANSI formatter with
no extra dependency. Install the extra for the nicer rendering:

```bash
pip install "deepcrew-ai[log]"
```

### Sample output

Text (`dev`):

```
15:30:47  INFO      deepcrew.runner   agent start  agent_id=researcher model=openai/gpt-4o max_turns=10
15:30:48  DEBUG     deepcrew.runner   turn start  step=turn:1 turn=1
15:30:51  INFO      deepcrew.orchestrator  routing decision  mode=parallel agents=["research", "analysis"]
```

JSON (`prod`), one object per line:

```json
{"ts":"2026-08-30T15:30:47.731Z","level":"INFO","logger":"deepcrew.runner","msg":"agent start","event":"agent_start","agent_id":"researcher","model":"openai/gpt-4o","max_turns":10}
```

Every lifecycle log carries an `event` field whose value matches the corresponding
`StreamEvent` type, so logs and the event stream share a vocabulary.

### `get_logger` — use it in your own components

`deepcrew.get_logger(name)` returns a structured logger namespaced under `deepcrew.<name>`,
so a single `configure_logging()` call styles your logs and the library's identically.
Extra keyword arguments become structured fields (rendered as a `key=value` suffix in text,
top-level keys in JSON).

```python
from deepcrew import get_logger

log = get_logger("myapp.tools.pricing")


@tool
def lookup_price(sku: str, region: str) -> str:
    """Look up the current price for a SKU."""
    log.info("price lookup", sku=sku, region=region)
    ...


# attach context carried on every subsequent call
bound = log.bind(request_id=req_id)
bound.warning("slow path", latency_ms=812)
bound.trace("raw payload", body=payload)  # TRACE level helper
```

**Design standard for custom components:** prefer `get_logger` over `print` /
`logging.getLogger`; pass fields as keyword arguments rather than formatting them into the
message string; add an `event="..."` slug for lifecycle logs; keep `INFO` for milestones and
push detail to `DEBUG`/`TRACE`.

### `LoggingConfig` reference

- **level** (int | str = "INFO"): threshold, by name or number.
- **profile** ("dev" | "prod" | "verbose" | "quiet" | None): preset bundle.
- **format** ("text" | "json" = "text"): renderer.
- **color** (bool | None = None): `None` auto-detects the TTY (honouring `NO_COLOR` /
  `FORCE_COLOR`).
- **timestamps** (bool = True): include a time column / `ts` key.
- **stream** (TextIO | None = None): defaults to `sys.stderr`.
- **propagate** (bool = False): whether the `deepcrew` logger propagates to the root logger.
  Off by default to avoid double emission; set True to route through your own root config.

### CLI

The `deepcrew` CLI is quiet by default (warnings and errors only). Flags:

```bash
deepcrew run workflow.yaml -v            # verbose profile
deepcrew run workflow.yaml --log-profile dev
deepcrew run workflow.yaml --log-level DEBUG --log-json
deepcrew run workflow.yaml -q            # errors only
```

### Common pitfalls

- **Nothing logged?** You have not called `configure_logging()` and no `DEEPCREW_LOG*` env
  var is set — that is the intended silent default for a library.
- **Double log lines?** Your app configured the root logger and you set `propagate=True`.
  Leave it at the default `False`, or remove your own `deepcrew` handler.
- **No colour in CI / pipes?** Colour auto-disables off a TTY. Force it with
  `DEEPCREW_LOG_COLOR=1` or `FORCE_COLOR=1`.
- **This is not OpenTelemetry.** For spans/traces use `ObservabilityConfig` — see the
  Observability guide. The two are complementary.
