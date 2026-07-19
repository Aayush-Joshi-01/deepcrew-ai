# Looping
The outer iteration loop is distinct from the inner per-turn `max_turns` cycle. The loop runs the entire agent (including all its tool calls) and re-runs it if the result doesn't meet a convergence criterion — ideal for search-refine, draft-critique, and iterative research patterns.

### How it works

    - Agent runs (all inner turns until no more tool calls)
    - `convergence_fn(result)` is called — if it returns `True`, the loop exits
    - If not converged, `refine_prompt` is appended and the agent runs again
    - Loop exits when `max_iterations` is reached or convergence is achieved
    - `result.loop_iterations` records how many outer iterations ran

### Basic loop with convergence

```python
from deepcrew import Agent, run_agent, LoopConfig, tool

@tool
def search_web(query: str) -> str:
    "Search the web for information."
    ...

agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    system_prompt="You are a thorough researcher. Use search to gather comprehensive information.",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=4,
        # Convergence: result is long enough to be a proper answer
        convergence_fn=lambda r: len(r.text) > 800 and r.text.count("\n") > 5,
        # What to say to the agent if not converged
        refine_prompt="Your answer is incomplete. Search for more details and expand your response significantly.",
    ),
)

result = await run_agent(
    agent,
    [{"role": "user", "content": "Explain the mechanism of CRISPR-Cas9 gene editing"}],
)

print(result.text)
print(f"Iterations: {result.loop_iterations}")
```

### Verifier-driven refinement

A `Verifier` grades each iteration's answer with structured feedback — a score, specific issues, and a suggestion — instead of a boolean, and drives a targeted refinement prompt. See the full [Verifier feature guide](#verifier) for a showcase of usage patterns, from a basic quality gate to a fully custom grading function.

### run_agent_loop() directly

`run_agent()` already delegates to `run_agent_loop()` automatically whenever `agent.loop_config` is set — you never need to call this yourself in normal use. It's exposed for cases where you want to call the outer loop directly without going through `run_agent()`'s signature (for example, from inside other deepcrew internals, or your own orchestration code that already has a fully-resolved `tool_defs` list and wants to skip the discovery step).

```python
from deepcrew import run_agent_loop, LoopConfig

result = await run_agent_loop(
    agent=my_agent,
    messages=[{"role": "user", "content": "Draft an executive summary"}],
    tool_defs=None,
    queue=my_queue,
    agent_id="drafter",
)
# result.loop_iterations is set on every exit path — converged, adaptive
# early-stop, or plain max_iterations exhaustion — so you can always tell
# how many outer iterations actually ran.
```

### How-to: adaptive early-stop

By default the loop always runs the full `max_iterations` unless `convergence_fn`/the verifier's `converged` flag fires first. With `adaptive=True` and a `verifier` configured, the loop additionally tracks the verifier score across iterations and stops as soon as improvement plateaus — saving compute once refinement stops paying off. It requires at least two scored iterations before it can detect a plateau, so it never fires on iteration 0 or 1; `max_iterations` remains a hard ceiling either way, adaptive can only shorten the loop, never lengthen it.

```python
from deepcrew import Agent, run_agent, LoopConfig, Verifier

agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=8,
        verifier=Verifier(),
        adaptive=True,
        min_improvement=0.02,   # smaller deltas than this count as "not improving"
        plateau_patience=2,     # stop after 2 consecutive non-improving iterations
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
print(f"Stopped after {result.loop_iterations} iterations")
```

When the plateau fires, the loop returns the best-scoring result seen so far — not necessarily the most recent one. If iteration 4 scored higher than iterations 5 and 6 before patience ran out, iteration 4's result is what you get back, even though two more (worse) iterations ran after it.

### How-to: self-evolving skill distillation

Set `auto_extract_skill=True` to turn a genuinely converged, high-quality loop run into a reusable `Skill` automatically, registered in the process-wide `SkillRegistry`. "Genuinely converged" is strict: this only fires when the loop actually converged via `convergence_fn` or the verifier's `converged` flag — plain `max_iterations` exhaustion, or an adaptive plateau stop, never triggers distillation, no matter how good the final score was.

```python
from deepcrew import Agent, run_agent, LoopConfig, Verifier, SkillRegistry

agent = Agent(
    name="sql_analyst",
    model="openai/gpt-4o-mini",
    tools=[run_sql],
    loop_config=LoopConfig(
        max_iterations=4,
        verifier=Verifier(),
        auto_extract_skill=True,
        skill_confidence_threshold=0.85,  # quality bar to distill
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Find top 10 customers by revenue"}])

# If it converged with score/confidence >= 0.85, a new Skill now exists
# in SkillRegistry, named deterministically from task_tag + a query hash.
print([s.name for s in SkillRegistry.list_all()])
```

The distilled skill is replayable, not a frozen memoized answer: calling it re-runs a fresh copy of the original agent (same `system_prompt`, `tools`, `mcps`, `skills`, model, and generation params) against whatever new `task` string you pass the skill — it generalizes to similar future tasks rather than always returning the one answer that happened to converge. See [Skills → Self-evolving skills](skills.html#self-evolving-skills) for how to actually invoke a distilled skill afterward.

### search_loop() — confidence-based iteration

`search_loop()` is a convenience wrapper that builds a fresh `Agent` with a single search tool and a `convergence_fn` based on `AgentResult.confidence`:

```python
from deepcrew import search_loop, Agent, tool

@tool
def search_web(query: str) -> str:
    "Search the web."
    ...

agent = Agent("searcher", model="openai/gpt-4o-mini",
              system_prompt="You are a research agent.", tools=[search_web])

# Runs at most 3 iterations, stopping early if result.confidence >= 0.8
result = await search_loop(
    query="What is the current state of nuclear fusion research?",
    search_tool=search_web,
    agent=agent,
    max_iterations=3,
    confidence_threshold=0.8,
)
```

> Caveat: `AgentResult.confidence` defaults to `None` and is only ever populated by `APEXSynthesizer` (multi-agent synthesis) or by a custom `evaluate_fn` you write and attach to a `Verifier` yourself — a bare agent run through `run_agent()` never sets it. `search_loop()`'s convergence check is `(r.confidence or 0.0) >= confidence_threshold`, so with the plain single-agent setup shown above, confidence stays `None` → treated as `0.0` → the check never passes, and the loop always runs the full `max_iterations` regardless of the threshold you set. If you want a real early-stop signal for a plain search agent, use [a `Verifier`](#looping) and set `convergence_fn` yourself instead of relying on `search_loop()`'s confidence check.

### Stop condition (early exit)

`stop_condition` (and `convergence_fn`, for that matter) should be a plain predicate — return `True`/`False`, don't raise anything yourself. When `stop_condition` returns `True`, the loop itself raises `LoopConvergedError` carrying that `AgentResult` on its `.result` attribute; the loop never catches this exception internally, so it propagates all the way out of `run_agent()`/`run_agent_loop()` to whoever called them. That means using `stop_condition` requires wrapping the call in a `try/except`:

```python
from deepcrew import Agent, run_agent, LoopConfig, LoopConvergedError

agent = Agent(
    "reasoner", model="openai/gpt-4o",
    system_prompt="When you have a final answer, prefix it with 'FINAL ANSWER:'.",
    loop_config=LoopConfig(
        max_iterations=6,
        stop_condition=lambda r: "FINAL ANSWER:" in r.text,  # plain bool, no raise
    ),
)

try:
    result = await run_agent(agent, [{"role": "user", "content": "..."}])
except LoopConvergedError as exc:
    result = exc.result   # the AgentResult that triggered the stop
```

`convergence_fn` is different: returning `True` from it makes the loop exit and return the result normally, no exception involved. Use `convergence_fn` for the common case (stop and return); reach for `stop_condition` only when you specifically want the exception-based early-exit path — e.g. to unwind through several stack frames of your own code, or to distinguish "converged normally" from "stopped early" at the call site via `except` vs. normal return.

### LoopConfig reference

- **max_iterations** (int = 5): Hard limit on outer loop iterations. Loop always exits after this many runs, converged or not.

- **convergence_fn** (Callable | None): Called with the current `AgentResult`. Return `True` to stop. Raise `LoopConvergedError(result)` for immediate exit with that result.

- **stop_condition** (Callable | None): Alternative to convergence_fn. Raises `LoopConvergedError` on its own — useful for externalizing early-exit logic.

- **refine_prompt** (str): Appended to conversation on each non-converged iteration. Default: `"Your answer is incomplete. Please search for more information and expand your response."`

- **verifier v0.2.1** (Verifier | None): Structured critic. When set, its `VerifierFeedback` (score + issues + suggestion) drives both convergence and the next refinement prompt, replacing the static `refine_prompt` text.

- **procedural_memory v0.2.2** (ProceduralMemory | None): Evolving playbook, read before iteration 1 and curated on loop exit. Requires `verifier` to be set too — see the [Procedural Memory guide](#procedural-memory).

- **task_tag v0.2.2** (str | None): Playbook namespace for `procedural_memory`. Defaults to `agent.name`.

- **adaptive v0.2.3** (bool = False): Plateau-detection early exit based on verifier score deltas. No-op without `verifier`. Never exceeds `max_iterations`.

- **min_improvement v0.2.3** (float = 0.02): Minimum verifier-score delta between iterations to count as still improving.

- **plateau_patience v0.2.3** (int = 2): Consecutive non-improving iterations tolerated before an adaptive early stop.

- **branches v0.2.4** (int = 1): When > 1, run this many parallel candidates per iteration (self-consistency). Best picked by `verifier` score, or merged via `APEXSynthesizer` when no verifier is set. Multiplies LLM calls per iteration.

- **auto_extract_skill v0.2.5** (bool = False): Distills a genuinely converged run into a reusable `Skill` registered in `SkillRegistry` — see the [Skills guide](#skills). Never triggers on plain `max_iterations` exhaustion.

- **skill_confidence_threshold v0.2.5** (float = 0.85): Minimum quality signal (verifier score, or `AgentResult.confidence`) required to distill a skill.

### How-to: self-consistency branching v0.2.4

Instead of one linear refinement path, run several candidate continuations per iteration in parallel and keep the best — a lightweight tree-search/self-consistency pattern. Each parallel call already samples independently from the model, so branches naturally diverge without any extra seeding logic. This costs `branches`× the LLM calls per iteration, so pair it with a low `max_iterations`.

```python
from deepcrew import Agent, run_agent, LoopConfig, Verifier, VerifierConfig

agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=3,
        verifier=Verifier(VerifierConfig(threshold=0.85)),
        branches=3,  # 3x the LLM calls per iteration, in exchange for picking the best
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
```

Without a `verifier`, branching still works — the `branches` candidates are merged into one cohesive answer via the same `APEXSynthesizer` used for multi-agent orchestration, rather than picking a single winner.

### Loop events

```python
from deepcrew.types import EventType

while True:
    event = await queue.get()
    if event is None: break
    if event.event == EventType.LOOP_ITERATION:
        i = event.data["iteration"]
        converged = event.data["converged"]
        print(f"Loop iteration {i} — {'converged' if converged else 'refining...'}")
    elif event.event == EventType.VERIFIER_SCORED:  # v0.2.1
        print(f"Verifier score: {event.data['score']} — issues: {event.data['issues']}")
    elif event.event == EventType.BRANCH_SELECTED:  # v0.2.4
        print(f"Branch {event.data['winning_index']} won with score {event.data['winning_score']}")
    elif event.event == EventType.SKILL_EXTRACTED:  # v0.2.5
        print(f"Distilled new skill: {event.data['skill_name']} (score {event.data['score']})")
```

### Common pitfalls

    - `search_loop()`'s confidence check can silently never fire. See the caveat above — with no verifier or APEX in the path, `AgentResult.confidence` stays `None` and the loop always runs to `max_iterations`.
    - `stop_condition` requires a `try/except LoopConvergedError` at the call site. The predicate itself must just return a bool — the loop raises the exception, not your callable.
    - `adaptive`, `procedural_memory`, and `auto_extract_skill` are all no-ops without `verifier`. Each of them either reads a verifier score or a converged-via-verifier flag; set a `Verifier` or these fields do nothing silently.
    - `branches` multiplies LLM call volume linearly. `branches=3` means 3× the calls per iteration — pair a high branch count with a low `max_iterations` to keep total cost bounded.
    - Adaptive early-stop returns the best iteration, not the last one. Don't assume `result.loop_iterations` tells you which iteration's text you're holding — it only tells you how many ran in total.

### See also

    - [Verifier](verifier.html) — the structured critic that drives convergence, adaptive stopping, and branch selection.
    - [Procedural Memory](procedural-memory.html) — the evolving playbook this loop reads from and curates into on exit.
    - [Skills → Self-evolving skills](skills.html#self-evolving-skills) — what happens to a distilled skill after `auto_extract_skill` registers it.
    - [APEX Synthesizer](apex.html) — merges branches when no verifier is set.
