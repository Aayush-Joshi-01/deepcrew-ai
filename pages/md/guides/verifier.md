# Verifier
A `Verifier` grades an agent's result against the original query and returns structured feedback — a score, specific issues, and a concrete suggestion — instead of a plain boolean. Attached to a `LoopConfig`, it drives both convergence and a targeted refinement prompt built from its critique, replacing the static default refine message. This page is a showcase of the ways to use it, end to end.

### How-to: basic quality gate

The simplest use — stop refining once the built-in LLM grader is confident enough.

```python
from deepcrew import Agent, run_agent, LoopConfig, Verifier, VerifierConfig

agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=4,
        verifier=Verifier(VerifierConfig(threshold=0.85)),
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
print(result.text)
```

### How-to: a task-specific rubric

Pass grading criteria specific to your domain so the verifier checks for what actually matters — e.g. code review, not just "is this a complete sentence."

```python
code_reviewer = Agent(
    name="code_reviewer",
    model="openai/gpt-4o",
    system_prompt="Review the given diff for bugs, security issues, and style.",
    loop_config=LoopConfig(
        max_iterations=3,
        verifier=Verifier(VerifierConfig(
            threshold=0.9,
            rubric=(
                "1. Every changed function must be covered by the review.\n"
                "2. Security issues (injection, auth, secrets) must be called out explicitly.\n"
                "3. Style nits are optional but bugs are not."
            ),
        )),
    ),
)

result = await run_agent(code_reviewer, [{"role": "user", "content": diff_text}])
```

### How-to: fully custom grading (no LLM call)

`evaluate_fn` replaces the built-in LLM grader entirely — useful when you have a deterministic check (schema validation, a unit test, a regex) that's cheaper and more reliable than asking another model.

```python
import json
from deepcrew import VerifierFeedback

async def json_schema_grader(query: str, result) -> VerifierFeedback:
    try:
        data = json.loads(result.text)
    except json.JSONDecodeError:
        return VerifierFeedback(score=0.0, issues=["Output is not valid JSON"], suggestion="Return valid JSON only.")
    missing = [k for k in ("summary", "action_items") if k not in data]
    if missing:
        return VerifierFeedback(score=0.4, issues=[f"Missing key: {k}" for k in missing], suggestion="Include all required keys.")
    return VerifierFeedback(score=1.0, converged=True)

agent = Agent(
    name="extractor",
    model="openai/gpt-4o-mini",
    loop_config=LoopConfig(
        max_iterations=3,
        verifier=Verifier(VerifierConfig(evaluate_fn=json_schema_grader)),
    ),
)
```

### How-to: adaptive compute budget v0.2.3

By default the loop always runs `max_iterations` times unless it converges early. With `adaptive=True`, it also tracks the verifier score across iterations and stops as soon as improvement plateaus — saving compute once refinement stops paying off. `max_iterations` is still a hard ceiling; adaptive can only shorten the loop, never lengthen it.

```python
agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    loop_config=LoopConfig(
        max_iterations=8,
        verifier=Verifier(VerifierConfig(threshold=0.9)),
        adaptive=True,
        min_improvement=0.02,   # minimum score delta to still count as "improving"
        plateau_patience=2,     # stop after this many non-improving iterations in a row
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
print(f"Stopped after {result.loop_iterations} iterations (cap was 8)")
```

When the loop stops early on a plateau, it returns the highest-scoring result seen so far — not necessarily the very last one — and emits a `LOOP_ITERATION` event with `{"early_stop": "plateau"}` in its data. Adaptive is a no-op without a `verifier` configured (there's no score to track).

### How-to: grading something outside a loop

`Verifier` doesn't require a `LoopConfig` at all — call `evaluate()` directly whenever you have a query and an `AgentResult` you want graded, e.g. to build your own custom retry/escalation logic instead of using the built-in loop.

```python
from deepcrew import Verifier, VerifierConfig, run_agent, Agent

verifier = Verifier(VerifierConfig(threshold=0.85))
agent = Agent(name="drafter", model="openai/gpt-4o-mini")

result = await run_agent(agent, [{"role": "user", "content": "Draft a release note"}])

feedback = await verifier.evaluate(
    "Draft a release note",
    result,
    default_model=agent.model,   # used only if VerifierConfig.model wasn't set
)

if not feedback.converged:
    print(f"Score {feedback.score:.2f} — issues: {feedback.issues}")
    print(f"Suggestion: {feedback.suggestion}")
```

### assess_complexity() — the other Verifier method

Separate from `evaluate()` (which grades a finished answer), `Verifier` also exposes `assess_complexity(task, default_model=...)` — a lightweight, pre-execution judgment call: "does this task genuinely need decomposing into sub-tasks, or can one agent handle it directly?" This is what powers `Orchestrator`'s `spawn_complexity_check` parameter, gating whether a newly-spawned sub-agent gets its own nested spawn tool (see [Agent Spawning → Bounded nested spawning](spawning.html#nested-spawning)). You can call it standalone too:

```python
verifier = Verifier()
needs_decomposition = await verifier.assess_complexity(
    "Build a full REST API with auth, rate limiting, and tests",
    default_model="openai/gpt-4o-mini",
)
print(needs_decomposition)  # bool
```

If the underlying LLM call fails or returns something unparseable, `assess_complexity()` defaults to `True` (permissive) rather than `False` — the reasoning being that a failed complexity check should not silently prevent legitimate decomposition. If you constructed the `Verifier` with `evaluate_fn` (a fully custom grader), `assess_complexity()` has no way to reuse that custom function for this different purpose and always returns `True` unconditionally.

### Verifier reference

- **VerifierConfig.model** (str | None): LiteLLM model string used to grade results. Defaults to the looped agent's own model.

- **VerifierConfig.threshold** (float = 0.8): Minimum score for `VerifierFeedback.converged` to be `True`.

- **VerifierConfig.rubric** (str | None): Optional task-specific grading criteria appended to the built-in verifier prompt.

- **VerifierConfig.evaluate_fn** (Callable | None): Full override: an async `(query, result) -> VerifierFeedback` function that replaces the built-in LLM grader entirely.

- **VerifierFeedback.score** (float): 0.0-1.0 quality estimate.

- **VerifierFeedback.issues** (list[str]): Specific problems found in the result.

- **VerifierFeedback.suggestion** (str): Actionable next-step guidance, used to build the refinement prompt.

- **VerifierFeedback.converged** (bool): `score >= threshold`.

### Common pitfalls

    - A failed grading call returns score 0.0, not an error. If the LLM call raises, or the response can't be parsed as JSON even with the regex fallback, `evaluate()` swallows the exception and returns `VerifierFeedback(score=0.0, ...)` — indistinguishable from a genuinely terrible answer. If your loop seems to never converge, check whether the grading model itself is actually reachable before assuming your agent's output is at fault.
    - `rubric` is appended text, not a replacement. It's inserted into the default grading prompt as extra criteria — it doesn't change the required JSON response shape (`score`/`issues`/`suggestion`), so a rubric describing a different output format won't be honored.
    - `evaluate_fn` disables `assess_complexity()`'s real logic. Once you supply a fully custom grader, complexity assessment (used by spawn nesting) always returns `True` — it has no way to reuse your custom function.
    - `adaptive` and `procedural_memory` both require `verifier`. Neither has any effect if you set them on a `LoopConfig` without also setting `verifier`.

### See also

    - [Looping](looping.html) — how `LoopConfig.verifier` integrates with convergence, adaptive early-stop, and refinement.
    - [Procedural Memory](procedural-memory.html) — verifier feedback also feeds the evolving playbook.
    - [Agent Spawning](spawning.html#nested-spawning) — `assess_complexity()` gates nested spawn-tool attachment.
