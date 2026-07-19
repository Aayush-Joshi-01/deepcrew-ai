# Human-in-the-Loop Hooks
`AgentHooks` intercepts and can alter execution — unlike the observe-only event stream. `approve_tool` returning `False` denies a tool call outright: the callable never runs, the tool result becomes `"Tool call denied by user."`, and a `TOOL_DENIED` event is emitted.

```python
from deepcrew import Agent, AgentHooks, run_agent

async def approve(tool_name: str, args: dict) -> bool:
    return tool_name != "delete_file"  # block one specific tool

agent = Agent(
    name="assistant",
    model="openai/gpt-4o",
    tools=[delete_file, read_file],
    hooks=AgentHooks(approve_tool=approve),
)
```

- **on_agent_start** (() -> Awaitable[None]): Called once before the first LLM call.

- **on_tool_start** ((name, args) -> Awaitable[None]): Called after approval, before the tool runs.

- **on_tool_end** ((name, result) -> Awaitable[None]): Called after the tool returns successfully.

- **approve_tool** ((name, args) -> Awaitable[bool]): Returning `False` denies the call. A raising hook is treated as denial.

> A hook that raises is caught and logged (WARNING) — it never crashes the run, except `approve_tool`, where a raise is treated the same as returning `False`.

### Execution order, per tool call

For each tool call the model requests, hooks fire in this exact order:

    - `approve_tool(name, args)` — if it returns `False` (or raises), the callable never runs. The tool's result becomes the literal string `"Tool call denied by user."`, appended to conversation history as a normal tool message, and a `TOOL_DENIED` event fires. Execution moves straight to step 4 — `on_tool_start`/`on_tool_end` are both skipped entirely for a denied call.
    - `on_tool_start(name, args)` — fires only if the call was approved (or if `approve_tool` is unset).
    - The tool actually executes.
    - `on_tool_end(name, result)` — fires only if execution succeeded without raising. If the tool itself raises an exception, `on_tool_end` is skipped, and the exception is folded into the tool's result content as an error string instead (unrelated to hooks — this is the same error-folding behavior that happens whether or not any hooks are configured).

`on_agent_start` is separate from all of this — it fires exactly once per `run_agent()` call, before the first LLM request, regardless of how many turns or tool calls follow.

### Hooks vs. events — pick the right one

These solve different problems and are easy to reach for interchangeably by mistake:

    - Hooks (`AgentHooks`) run inside the agent's execution, synchronously, and can change what happens — `approve_tool` is the only hook with the power to prevent a tool call outright.
    - Events (`StreamEvent` via a `queue`, filtered by [StreamPolicy](streampolicy.html)) only ever describe what already happened, for a UI or logging pipeline — nothing consuming the event queue can stop or alter execution, no matter how fast it reacts.

If you need a human to approve a specific tool call before it runs, that's `approve_tool` — there's no way to build that with events alone.

### A realistic human-approval pattern

`approve_tool` is awaited, so it can genuinely pause execution on something slower than a synchronous check — a database lookup, a Slack approval button, a queued request to a human reviewer.

```python
import asyncio
from deepcrew import Agent, AgentHooks, run_agent

pending_approvals: dict[str, asyncio.Future] = {}

async def ask_a_human(tool_name: str, args: dict) -> bool:
    request_id = f"{tool_name}:{id(args)}"
    future = asyncio.get_event_loop().create_future()
    pending_approvals[request_id] = future
    # e.g. post a Slack message with Approve/Deny buttons here, then wait:
    send_approval_request(request_id, tool_name, args)  # your integration
    return await future  # resolved elsewhere when the human responds

agent = Agent(
    name="ops_assistant",
    model="openai/gpt-4o",
    tools=[restart_service, delete_records],
    hooks=AgentHooks(approve_tool=ask_a_human),
)

# Elsewhere, e.g. in a webhook handler for the Slack button click:
# pending_approvals[request_id].set_result(True)  # or False to deny
```

### Common pitfalls

    - A denied call still appears in `AgentResult.tool_calls`. Tool calls are recorded from the model's request, before `approve_tool` is ever consulted — you can't distinguish a denied call from an executed one by looking at `tool_calls` alone; check for a `TOOL_DENIED` event or the literal `"Tool call denied by user."` content instead.
    - `on_tool_end` doesn't fire on a tool exception. Only a successful, non-denied execution reaches it — don't rely on it for cleanup that must run unconditionally.
    - Hooks intercept; events only observe. Don't reach for a `StreamPolicy`-filtered event consumer when you actually need to block a tool call — only `approve_tool` can do that.
    - Hooks apply to one `Agent`, not globally. There's no process-wide hook registry — set `hooks=` on every agent instance that needs interception, including spawned sub-agents, which do not inherit the parent's hooks automatically.

### See also

    - [StreamPolicy](streampolicy.html) — the observe-only counterpart to hooks; both can be used together.
    - [Agent Spawning](spawning.html) — spawned sub-agents don't inherit the parent's `AgentHooks`.
