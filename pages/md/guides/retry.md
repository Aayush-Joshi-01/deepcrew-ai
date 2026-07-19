# Retry & Fallback
Configure per-agent retry behavior with exponential backoff, and model fallback chains that activate when all retries fail.

### Basic retry

```python
from deepcrew import Agent, RetryPolicy

agent = Agent(
    name="resilient",
    model="openai/gpt-4o",
    system_prompt="Be helpful.",
    retry_policy=RetryPolicy(
        max_retries=3,          # try up to 3 more times after the first failure
        backoff_seconds=1.0,    # base wait between retries
        exponential=True,       # 1s, 2s, 4s, 8s ... (doubles each time)
        retry_on=(Exception,),  # retry on any exception (default)
    ),
)
```

### Retry specific exceptions only

```python
import litellm
from deepcrew import RetryPolicy

# Only retry on rate limit and connection errors
agent = Agent(
    "selective_retry",
    model="openai/gpt-4o",
    retry_policy=RetryPolicy(
        max_retries=5,
        backoff_seconds=2.0,
        retry_on=(
            litellm.RateLimitError,
            litellm.APIConnectionError,
            TimeoutError,
        ),
    ),
)
```

### Fallback chain

```python
from deepcrew import Agent, RetryPolicy, FallbackChain

agent = Agent(
    name="fault_tolerant",
    model="openai/gpt-4o",           # primary model
    retry_policy=RetryPolicy(
        max_retries=2,
        backoff_seconds=1.0,
    ),
    fallback_chain=FallbackChain(models=[
        "anthropic/claude-haiku-4-5-20251001",  # try first if gpt-4o fails all retries
        "gemini/gemini-2.0-flash",               # try second
        "ollama/llama3.2",                       # local fallback
    ]),
)

# Flow: gpt-4o → retry 1 → retry 2 → claude-haiku → retry 1 → retry 2 → gemini → ...
```

### How retry and fallback actually interact

This is easy to get an intuition for that's slightly wrong, so it's worth spelling out exactly: `retry_on` doesn't just decide whether to wait-and-retry — it decides whether fallback runs at all for that failure. When an LLM call raises, the wrapper checks the raised exception against `retry_on`. If it matches, the call retries (with backoff) up to `max_retries` times on the same model, and only after those retries are exhausted does it advance to the next model in `fallback_chain`. But if the exception does not match `retry_on`, it is re-raised immediately — the fallback chain is never consulted, even if you configured one. A `ValueError` from a bug in your own tool code, for example, will propagate straight out rather than triggering a fallback to your next model, unless you deliberately included it in `retry_on`.

Each model in the chain gets its own full retry budget: with `max_retries=2` and a 3-model chain (primary + 2 fallbacks), a persistently-failing sequence of retryable errors results in up to `3 attempts × 3 models = 9` total LLM calls before the final exception propagates to your code. If every model in the chain is exhausted, the last exception raised (from the last model tried) is what you ultimately catch — earlier models' specific failures aren't preserved or chained.

Retry/fallback wraps a single `litellm.acompletion` call, scoped to one turn — it has no interaction with `Agent.max_turns` (the tool-call cycle limit) or `LoopConfig` (the outer refinement loop). A retried-and-recovered turn still counts as just one turn toward `max_turns`.

### Retry events

```python
from deepcrew.types import EventType

while True:
    event = await queue.get()
    if event is None: break
    if event.event == EventType.RETRY_ATTEMPT:
        data = event.data
        print(f"Retry {data['attempt']} on {data['model']} — waiting {data['delay']:.1f}s")
    elif event.event == EventType.FALLBACK_TRIGGERED:
        print(f"Falling back from {event.data['from_model']} → {event.data['to_model']}")
```

### RetryPolicy reference

- **max_retries** (int = 3): Number of additional attempts after the first failure. `max_retries=3` means up to 4 total calls per model.

- **backoff_seconds** (float = 1.0): Base wait time in seconds between retries. With `exponential=True`, this doubles each retry.

- **retry_on** (tuple[type[Exception], ...] = (Exception,)): Exception types that trigger a retry. Use specific types like `litellm.RateLimitError` to avoid retrying logic errors.

- **exponential** (bool = True): Whether to double the backoff time on each retry. False gives fixed backoff.

### FallbackChain reference

- **models** (list[str]): LiteLLM model strings tried in order, after the primary `Agent.model` exhausts its own retry budget. Each entry gets the same `RetryPolicy` as the primary model — there's no per-fallback-model override.

### Common pitfalls

    - A non-retryable exception skips fallback entirely. See "How retry and fallback actually interact" above — `retry_on` gates whether fallback is even attempted, not just whether backoff happens.
    - Setting `fallback_chain` without `retry_policy` still works, but with zero retries per model. Without a `RetryPolicy`, `max_attempts` defaults to 1 — each model in the chain gets exactly one try before moving to the next, no backoff at all.
    - Every fallback model shares the same retry budget and exception filter. You can't give your local Ollama fallback a longer backoff or a different `retry_on` than your primary OpenAI model.
    - Total call count multiplies quickly. `max_retries` × number of models in the chain — a generous retry policy on a long fallback chain can mean a slow failure path that's expensive in both time and token spend before it finally gives up.

### See also

    - [Observability](observability.html) — pair retry/fallback with OTel spans to see exactly which model handled a given call.
    - [StreamPolicy](streampolicy.html) — `RETRY_ATTEMPT`/`FALLBACK_TRIGGERED` are only visible under `.verbose()`, not `.chat()` or `.standard()`.
