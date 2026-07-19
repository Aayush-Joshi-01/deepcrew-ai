# Procedural Memory
`ProceduralMemory` is an opt-in, durable "the system learns from its own past runs" store, inspired by ACE (Agentic Context Engineering, ICLR 2026). It's built on top of any `MemoryProvider` as its backing store and adds structure: each entry is a "helpful" or "harmful" bullet with a usage count and last-seen score. It's read on every run of an agent it's attached to (looped or single-shot), and curated — incrementally merged, never wholesale rewritten — whenever a loop with a `Verifier` converges. This page is a showcase of the ways to use it, end to end.

### How-to: a research agent that gets smarter over time

Same agent, same task type, run repeatedly — each run's high-confidence result and each failure's specific issue become durable strategy bullets injected into the next run's context.

```python
from deepcrew import (
    Agent, run_agent, LoopConfig, Verifier, VerifierConfig,
    FileMemoryProvider, ProceduralMemory,
)

playbook = ProceduralMemory(FileMemoryProvider("playbook.json"), max_entries=30)

agent = Agent(
    name="researcher",
    model="openai/gpt-4o-mini",
    tools=[search_web],
    procedural_memory=playbook,   # read on every run, even single-shot
    loop_config=LoopConfig(
        max_iterations=4,
        verifier=Verifier(VerifierConfig(threshold=0.85)),
        procedural_memory=playbook,  # curated after a converged loop
    ),
)

result = await run_agent(agent, [{"role": "user", "content": "Explain CRISPR"}])
# Run the same agent again later (even a new process, with FileMemoryProvider) and
# it will already know what worked and what to avoid for this task.
```

Curation requires a `verifier` on the same `LoopConfig` — without one, `procedural_memory` there is a no-op (there's no `VerifierFeedback` to grade the run against). Reading the playbook (`Agent.procedural_memory`) works independently of looping.

### How-to: a shared playbook across an agent pool

Multiple agents that handle the same kind of task (e.g. every "support_triage" agent spawned by an `Orchestrator`) can share one `ProceduralMemory` instance and namespace it with an explicit `task_tag` instead of the default (which is keyed by `agent.name`), so lessons pool together regardless of which specific agent instance ran.

```python
shared_playbook = ProceduralMemory(FileMemoryProvider("support_playbook.json"))

def make_support_agent(name: str) -> Agent:
    return Agent(
        name=name,
        model="openai/gpt-4o-mini",
        procedural_memory=shared_playbook,
        loop_config=LoopConfig(
            verifier=Verifier(VerifierConfig(threshold=0.8)),
            procedural_memory=shared_playbook,
            task_tag="support_triage",  # shared namespace, not tied to agent.name
        ),
    )

agent_a = make_support_agent("triage_shift_1")
agent_b = make_support_agent("triage_shift_2")
# Both read from and write to the same "support_triage" playbook.
```

### How-to: inspect the playbook directly

You don't need to run an agent to read or seed a playbook — `ProceduralMemory` is usable standalone for debugging, exporting, or manual curation.

```python
entries = await playbook.load("researcher")
for e in entries:
    print(f"[{e.kind}] {e.content} (used {e.uses}x, last score {e.last_score})")

print(playbook.render(entries))  # the exact text block injected into context
```

### How curation actually works

`curate()` is deliberately conservative — it never rewrites the playbook wholesale, only ever merges or appends, which is precisely what avoids the "context collapse" failure mode of naive full-rewrite approaches. Each call does exactly this, in order:

    - Loads the existing entries for `task_tag`.
    - Builds candidate entries from this run: up to the first 3 issues in `feedback.issues` each become a `kind="harmful"` candidate prefixed `"Avoid: "`. If `feedback.score >= 0.8`, the final result's text (truncated to 200 characters, newlines flattened) becomes one `kind="helpful"` candidate prefixed `"Worked well: "`. A low-scoring run with fewer than 3 issues contributes fewer candidates; a run scoring below 0.8 contributes no helpful candidate at all.
    - Each candidate is checked against existing entries with a cheap similarity heuristic (case-insensitive exact match, or one string being a substring of the other) — not an embedding or LLM-based similarity check. A match bumps that entry's `uses` counter and updates `last_score`; no match appends a new entry.
    - All entries are sorted by `(uses, last_score)` descending and truncated to `max_entries` — the least-used, lowest-scoring entries are the ones dropped when the playbook is full.
    - The pruned list is persisted back to the backend, one key per entry index plus a small metadata key recording the count.

Because similarity is substring-based, near-duplicate phrasing ("avoid rate limits" vs. "avoid hitting the rate limit") won't merge — they'll accumulate as separate low-usage entries competing for the same `max_entries` slots. Keep your agent's own issue/summary text reasonably consistent in phrasing if you want related lessons to actually consolidate over time.

### PlaybookEntry reference

- **content** (str): The strategy text itself, already prefixed `"Avoid: "` or `"Worked well: "` by `curate()`.

- **kind** ("helpful" | "harmful"): Rendered as `(helpful)` or `(avoid)` in the injected system-prompt block.

- **uses** (int = 0): Incremented every time a new candidate matches this entry as a near-duplicate. Higher `uses` means this lesson has recurred across more runs, and is sorted first / pruned last.

- **last_score** (float | None): The verifier score of the most recent run that reinforced this entry.

### ProceduralMemory reference

- **ProceduralMemory(backend, max_entries=30)** (MemoryProvider, int): Wraps any `MemoryProvider` as the backing store; caps the playbook at `max_entries`, pruned by usage/score.

- **load(task_tag)** (async -> list[PlaybookEntry]): Returns all persisted entries for the given namespace. Returns an empty list (not an error) if nothing has been curated yet, or if the stored metadata is corrupt/unparseable.

- **render(entries)** (list[PlaybookEntry] -> str): Formats entries as a compact `## Known strategies for this task` bullet block for system-prompt injection; empty string for an empty list (so nothing extra is injected when there's no playbook yet).

- **curate(task_tag, feedback, trajectory)** (async -> int): Reflector+Curator step described above. `trajectory` is the loop's full list of `AgentResult`s so far; only the last one's text is actually used, as the "worked well" summary source.

### Playbook events

```python
from deepcrew.types import EventType

async for event in run_agent_loop(agent, messages, queue=queue):
    if event.event == EventType.PLAYBOOK_UPDATED:
        print(f"Playbook now has {event.data['entry_count']} entries")
```

### Common pitfalls

    - Curation is a no-op without a converged loop. `LoopConfig.procedural_memory` only curates when the loop exits via convergence (verifier or `convergence_fn`) — an adaptive plateau stop or plain `max_iterations` exhaustion still calls `curate()` internally in the current implementation, but always requires `verifier` to be set for there to be any `VerifierFeedback` to curate from at all.
    - Similarity matching is substring-based, not semantic. Rephrased near-duplicates won't merge (see "How curation actually works" above) — they'll pile up as separate entries instead.
    - Only the last 3 issues and 1 summary are ever considered per run. A run with many issues doesn't get them all recorded — extra issues beyond the first 3 are silently dropped from that run's curation pass.
    - Reading (`Agent.procedural_memory`) and curating (`LoopConfig.procedural_memory`) are two separate assignments. Setting only one of them means either the agent never sees the playbook, or nothing ever gets written to it — you usually want both pointed at the same instance, as in the examples above.

### See also

    - [Verifier](verifier.html) — produces the `VerifierFeedback` that `curate()` consumes.
    - [Looping](looping.html) — the outer-loop lifecycle procedural memory plugs into.
    - [Memory Providers](memory.html) — the raw key/value backend procedural memory is built on top of.
