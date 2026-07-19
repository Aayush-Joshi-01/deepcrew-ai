# CLI
Run declarative YAML workflow files from the terminal. No Python code required for simple workflows.

### Installation check

```python
pip install deepcrew-ai
deepcrew --version
# deepcrew-ai 0.4.0
```

### Write a workflow YAML

```python
agents:
  - name: researcher
    model: openai/gpt-4o-mini
    system_prompt: Research the topic thoroughly using all available information.
    tools:
      - web_search   # built-in skill by name

  - name: analyst
    model: anthropic/claude-haiku-4-5-20251001
    system_prompt: Critically analyze the research findings. Identify gaps and strengths.

  - name: writer
    model: openai/gpt-4o
    system_prompt: Write a clear, well-structured executive summary report.
    tools:
      - summarize   # built-in summarize skill

workflow:
  - step: research
    agent: researcher
    task: "{input}"

  - step: analysis
    agent: analyst
    task: |
      Analyze this research:
      {research}
    depends_on:
      - research

  - step: report
    agent: writer
    task: |
      Write an executive summary based on:

      Research: {research}
      Analysis: {analysis}
    depends_on:
      - research
      - analysis
```

### Run it

```python
# Stream output to terminal
deepcrew run workflow.yaml --input "The future of autonomous vehicles"

# Non-streaming (prints only final result)
deepcrew run workflow.yaml --input "Quantum computing in 2026" --no-stream

# List all agents in a config
deepcrew agents list --config workflow.yaml
```

### YAML schema

- **agents[].name*** (str): Agent identifier, referenced in workflow steps.

- **agents[].model*** (str): LiteLLM model string.

- **agents[].system_prompt** (str): Agent's system prompt.

- **agents[].tools** (list[str]): Built-in skill names: `web_search`, `summarize`, `code_exec`. Any name not in that set is silently ignored — it is dropped, not an error, so a typo in a tool name fails silently rather than raising at load time.

- **agents[].max_turns** (int = 10): Max inner loop turns for this agent.

- **agents[].temperature** (float | null): Sampling temperature.

- **agents[].max_tokens** (int | null): Maximum output tokens per LLM call.

- **workflow[].step*** (str): Step name — used as a variable `{step_name}` in subsequent task templates.

- **workflow[].agent*** (str): References an agent by `name`. Referencing a name not defined in `agents:` raises a `ValueError` at load time, before anything runs.

- **workflow[].task** (str = "{input}"): Task template. `{input}` is the CLI `--input` value (or the top-level `input:` YAML field if `--input` is omitted). `{step_name}` is the text output of that step.

- **workflow[].depends_on** (list[str] = []): Step names this step depends on. Steps without overlapping dependencies run in parallel.

- **input** (str | null): Top-level fallback for `{input}` when `--input` isn't passed on the command line.

- **router_model** (str = "openai/gpt-4o-mini"): Parsed from the YAML but currently unused by `deepcrew run` — the CLI always builds an explicit `WorkflowBuilder` DAG from your `workflow:` steps, never an `Orchestrator`, so there's no router LLM call for this to configure. Setting it has no effect today.

### Supported built-in tool names

      | YAML name | Skill class | Description |

        | `web_search` | `WebSearchSkill` | DuckDuckGo search |

        | `summarize` | `SummarizeSkill` | LLM-backed summarization |

        | `code_exec` | `CodeExecutionSkill` | Python subprocess execution |

> For advanced workflows with custom Python tools, memory providers, or observability, use the Python API directly. The CLI is designed for simple, shareable workflows that don't need custom code — there is no YAML way to attach a custom `@tool` function, an MCP server, a `MemoryProvider`, or an `ObservabilityConfig`; only the three built-in skills above are reachable from YAML.

### Common pitfalls

    - Unknown tool names fail silently. A typo like `web_serach` in `agents[].tools` is simply dropped — the agent runs with one fewer tool than you intended, no warning.
    - `router_model` does nothing yet. The CLI never routes — every workflow you write in YAML is an explicit DAG.
    - Only three skills are reachable from YAML. There's no config surface for custom tools, MCP servers, memory, or observability — reach for the Python API once you need any of those.
    - A bad agent reference in `workflow:` fails before any LLM call. This is a fast, cheap validation error — check your step's `agent:` field spelling against your `agents[].name` list first if you see it.

### See also

    - [Skills](skills.html) — the three built-in skills reachable from YAML, and how to write your own (Python API only).
    - [Memory Providers](memory.html) and [Observability](observability.html) — both require the Python API; there's no YAML equivalent.
