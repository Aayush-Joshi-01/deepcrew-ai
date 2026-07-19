# APEX Synthesizer
APEX is deepcrew's synthesis engine for merging the outputs of several agents that worked on the same query in parallel. It is not just string concatenation or a second summarization pass: it produces a self-reported confidence score (0.0–1.0) on every result, can optionally cite which agent contributed each fact with inline `[source: agent_name]` markers, and — if you give it tools — can call them mid-synthesis to fact-check a claim before committing to it.

APEX lives in `src/deepcrew/apex.py` as the `APEXSynthesizer` class. You will rarely construct it directly in normal use: `Orchestrator` builds and owns one internally whenever it routes a query to multiple agents. You only reach for the standalone API in [the section below](#standalone) when you already have a list of `AgentResult` objects from somewhere else (a cache, a previous run, a custom fan-out you wrote yourself) and want APEX's merging behavior without going through the full router pipeline.

### When APEX actually runs

This is the single most common point of confusion, so it is worth being precise: APEX only runs on the parallel branch of orchestration. When the router decides a query needs exactly one agent (`{"route": "single", ...}`), that agent's raw `AgentResult` is returned as-is — `result.final_text` is that agent's text verbatim, and `result.agent_results[-1].confidence` is `None`, because no synthesis step ever ran. APEX is invoked exactly once per orchestration, from inside `Orchestrator._orchestrate`, immediately after all agents in a parallel fan-out finish (or fail — a failed agent's exception is swallowed into an `ERROR` event and it is simply excluded from the list passed to APEX; APEX never sees agents that raised).

There is a second, less obvious place APEX runs: inside the self-improving loop's [branching](looping.html#branching) feature. When `LoopConfig(branches=N)` runs N parallel candidate continuations for one iteration and no `verifier` is configured, there is no score to pick a single winner by — so `loop.py` falls back to merging all N branches through a fresh, default-configured `APEXSynthesizer` instead of just discarding N-1 of them. If a verifier is configured, branching skips APEX entirely and picks the highest-scoring branch directly. See [Looping → Branching](looping.html#branching) for the full mechanics.

### How it works

    - All agent results from the parallel fan-out are collected into a list (agents that raised are already filtered out by this point).
    - APEX builds one prompt containing the original query followed by a `--- Agent: {agent_id} ---` block for every surviving result, in the order the router listed them.
    - It synthesizes a unified answer using its own model (`apex_model`, defaulting to the same model as the router) and ends its response with a literal trailing line: `CONFIDENCE: 0.85`.
    - The confidence line is parsed out with a regex (`CONFIDENCE:\s*([\d.]+)`), clamped to `[0.0, 1.0]`, and stored on `AgentResult.confidence` of the returned synthesis result. If the model doesn't emit a parseable confidence line at all — most likely because you overrode `system_prompt` and forgot to keep that instruction — APEX silently falls back to a default confidence of 0.8. This fallback is a deliberate "don't crash the pipeline over a formatting slip" choice, not a signal that anything went well; don't rely on 0.8 meaning anything about answer quality.
    - The confidence line itself is stripped from the final text via the same regex before `result.final_text`/`synthesis.text` is returned to you — you will never see the literal `CONFIDENCE: 0.85` string in output shown to an end user.
    - When `cite_sources=True` (the default), the system prompt additionally instructs APEX to mark facts inline as `[source: agent_name]` as it writes them, so `build_citations()` has something to extract afterward.

Two details worth internalizing about the prompt: APEX sees the full text of every agent's result, not a summary or excerpt — with many agents or very long individual outputs this can push the synthesis call's input tokens surprisingly high, since it scales linearly with the number of parallel agents. And critically, none of the individual agents' `tool_calls`, `input_tokens`, or `output_tokens` are forwarded into the synthesis prompt — APEX only ever sees the final `.text` of each result, never how that text was produced.

### Basic usage

The common case: give `Orchestrator` an `ApexConfig` and let it decide when synthesis is needed.

```python
from deepcrew import Agent, Orchestrator, ApexConfig

orch = Orchestrator(
    agents=[
        Agent("researcher", model="openai/gpt-4o-mini", system_prompt="Research facts."),
        Agent("analyst",    model="anthropic/claude-haiku-4-5-20251001", system_prompt="Analyze data."),
    ],
    router_model="openai/gpt-4o-mini",
    apex_model="openai/gpt-4o",       # defaults to router_model if omitted
    apex_config=ApexConfig(
        cite_sources=True,           # [source: researcher] / [source: analyst] inline
        confidence_threshold=0.75,   # advisory only — see "Common pitfalls" below
        allow_tools=False,           # APEX can call tools mid-synthesis (experimental)
    ),
)

result = await orch.run("What causes inflation?")
print(result.final_text)

# Only meaningful if the router actually fanned out to >1 agent for this query —
# on a single-agent route, confidence is None because APEX never ran.
last = result.agent_results[-1]
if last.confidence is not None:
    print(f"Confidence: {last.confidence:.2%}")
```

### Standalone APEX

Use `APEXSynthesizer` directly when you have `AgentResult` objects from somewhere other than a live `Orchestrator` run — for example, results you cached from an earlier run, or a custom fan-out you built by hand with `asyncio.gather` over several `run_agent()` calls.

```python
from deepcrew import APEXSynthesizer, ApexConfig, AgentResult

# Use APEX outside of Orchestrator — e.g., synthesize cached results
apex = APEXSynthesizer(
    model="openai/gpt-4o",
    config=ApexConfig(cite_sources=True),
)

results: list[AgentResult] = [...]  # your pre-computed results

synthesis = await apex.synthesize(
    original_query="Explain quantum entanglement",
    results=results,
    # queue=my_queue,        # optional: emits APEX_START/APEX_DONE if you pass one
    # tool_defs=my_tools,    # required if config.allow_tools=True
)

print(synthesis.text)
print(f"Confidence: {synthesis.confidence:.2f}")

# Citations
for citation in apex.build_citations(results, synthesis.text):
    print(f"[{citation.agent_id}] {citation.claim[:80]}")
```

`synthesize()` returns a full `AgentResult` with `agent_id="apex"`, so it slots into anywhere else in deepcrew that expects one — `WorkflowBuilder` outputs, further loop iterations, and so on.

### ApexConfig reference

- **confidence_threshold** (float = 0.7): Purely advisory metadata for your code to act on — deepcrew does not read this value internally, retry, request more agents, or otherwise change behavior when the actual confidence falls below it. Check `result.agent_results[-1].confidence `" instruction — omit it and every synthesis silently falls back to the default confidence of 0.8 (see "How it works" above).

### ApexCitation reference

`build_citations(results, synthesis_text)` scans the synthesis text for every `[source: agent_name]` marker and returns one `ApexCitation` per match — it does not call the LLM again, this is pure string parsing against the text you already have.

- **agent_id** (str): The agent name exactly as it appeared inside the `[source: ...]` marker — matching is case-sensitive and does not validate against the actual agent names you passed in, so a hallucinated agent name in the marker still produces a citation.

- **claim** (str): The sentence immediately preceding the marker (up to 120 characters back, split on the last period). This is a heuristic, not a real sentence-boundary parse — it can occasionally grab a partial or unrelated clause on unusually punctuated text.

- **confidence** (float): Always hardcoded to `0.9` for every citation — this is a per-citation placeholder, not derived from the synthesis-level confidence score or from anything the source agent reported.

### APEX events

Both events are emitted only when you pass a `queue` to `synthesize()` — the standalone API is silent by default. Through `Orchestrator`, the queue is always wired up automatically.

- **APEX_START** ({"agents": int}): Fired once, right before the synthesis prompt is built. `agents` is the count of results being merged.

- **APEX_DONE** ({"confidence": float}): Fired once synthesis text and confidence have both been parsed out. There is no separate "failed" event — if the underlying LLM call raises, that exception propagates up through `Orchestrator._orchestrate`'s own try/except into a generic `ERROR` event instead.

```python
from deepcrew.types import EventType

async for event in orch.stream("..."):
    if event.event == EventType.APEX_START:
        agents = event.data["agents"]
        print(f"APEX synthesizing from {len(agents)} agents: {agents}")
    elif event.event == EventType.APEX_DONE:
        conf = event.data["confidence"]
        print(f"APEX done — confidence {conf:.2f}")
        if conf < 0.7:
            print("Warning: Low confidence — consider adding more specialist agents")
```

### Common pitfalls

    - Confidence is `None` on single-agent routes. Always guard `result.agent_results[-1].confidence is not None` before formatting it — a query the router sent to exactly one agent never touches APEX.
    - `confidence_threshold` does nothing by itself. It is metadata for your own conditional logic, not a gate deepcrew enforces (see the ApexConfig table above).
    - Overriding `system_prompt` drops the confidence instruction. If your custom prompt doesn't end with a request for a `CONFIDENCE: ` line, every result silently gets the 0.8 fallback instead of a real signal.
    - Citations depend entirely on model compliance. `cite_sources=True` is an instruction, not a constraint — some models under-cite, especially on short syntheses with only one dominant source.
    - Token cost scales with agent count. APEX receives every parallel agent's full output text verbatim; a 6-agent fan-out with long individual answers means a proportionally large synthesis prompt.

### See also

    - [Agent Spawning](spawning.html) — the other half of `Orchestrator`'s execution stage.
    - [Looping → Branching](looping.html#branching) — the other place APEX runs, merging self-consistency candidates when no verifier is configured.
    - [StreamPolicy](streampolicy.html) — `StreamPolicy.standard()` and `.verbose()` both surface `APEX_START`/`APEX_DONE`; `.chat()` hides them.
