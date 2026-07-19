# Structured Output
Set `response_model` to a pydantic model and the agent's final text is validated against it, landing on `AgentResult.parsed`. One automatic repair attempt is made if the first response isn't valid JSON matching the schema; if that also fails, `OutputParseError` is raised with the raw text attached.

```python
from pydantic import BaseModel
from deepcrew import Agent, run_agent

class Verdict(BaseModel):
    approved: bool
    reason: str

agent = Agent(name="reviewer", model="openai/gpt-4o", response_model=Verdict)
result = await run_agent(agent, [{"role": "user", "content": "Review this PR diff: ..."}])

result.parsed.approved  # bool
result.parsed.reason    # str
```

### How it actually works

This is prompt-based, not provider-JSON-mode-enforced — worth knowing since it's a weaker guarantee than passing `response_format={"type": "json_object"}` directly to the provider. When `response_model` is set, deepcrew appends one extra system message to the conversation before the first turn:

```python
Respond ONLY with JSON matching this schema: {"properties": {...}, "required": [...], ...}
```

...where the schema is `response_model.model_json_schema()`, pydantic's own schema dump. The model is trusted to follow that instruction; nothing on the LiteLLM call itself forces JSON output. Parsing happens once the agent's final turn produces text with no more tool calls:

    - Any leading/trailing ````json ... ```` (or plain `````) fence around the response is stripped first.
    - The result is validated with `response_model.model_validate_json(...)`.
    - On failure (invalid JSON, or valid JSON that doesn't match the schema), deepcrew makes exactly one repair attempt: it re-prompts the same model with the full conversation history plus the failed response and the validation error text, asking for corrected JSON.
    - If the repair attempt also fails to validate, `OutputParseError` is raised — not returned as a degraded result — carrying the repaired (still-invalid) text on `.raw_text`.

```python
from deepcrew import OutputParseError

try:
    result = await run_agent(agent, [{"role": "user", "content": "..."}])
except OutputParseError as exc:
    print("Model never produced valid JSON, even after one repair attempt:")
    print(exc.raw_text)  # the raw (still invalid) text from the repair attempt
```

### Combining with tools and with LoopConfig

An agent with both `response_model` and `tools`/`skills` works fine: the schema instruction is injected once as a system message, and the agent can still make tool calls across multiple turns as usual — parsing only happens against the text of the final turn (the one with no more tool calls), never against intermediate turns.

`response_model` also composes with `loop_config`: every outer-loop iteration runs the schema-instruction + parse-with-repair logic independently, and whichever `AgentResult` the loop ultimately returns (converged, adaptive-stopped, or the last iteration) carries that iteration's own `.parsed` value. If you want convergence to depend on the parsed structure rather than just the raw text, write a `convergence_fn` that inspects `result.parsed` directly — nothing does that for you automatically.

### Common pitfalls

    - This is a prompt instruction, not enforced JSON mode. Weaker models or highly complex schemas can still produce non-JSON text on both the first attempt and the repair attempt, ending in `OutputParseError`.
    - Only one repair attempt, always. There's no configurable retry count for structured-output parsing specifically (separate from `RetryPolicy`, which governs LLM-call-level failures, not schema-validation failures).
    - `result.parsed` is `None` whenever `response_model` isn't set. Always None-check it, or make it a habit to only read `.parsed` on agents you know were constructed with a schema.
    - The repair prompt reuses the same model. There's no way to route the repair attempt to a different (e.g. stronger) model than the one that produced the invalid output.

### See also

    - [Looping](looping.html) — how `response_model` behaves across outer-loop iterations.
    - [Human-in-the-Loop Hooks](hooks.html) — the other "Agent Behavior" feature, for intercepting tool calls rather than validating final output.
