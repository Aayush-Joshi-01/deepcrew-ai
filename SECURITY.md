# Security Policy

## Supported versions

Security fixes are applied to the latest released minor version.

| Version | Supported |
| --- | --- |
| 0.4.x | Yes |
| < 0.4 | No |

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Use GitHub's private vulnerability reporting instead:

👉 https://github.com/Aayush-Joshi-01/deepcrew-ai/security/advisories/new

Please include:

- A description of the issue and its impact
- Steps to reproduce, or a proof-of-concept
- The version of `deepcrew-ai` affected
- Any suggested mitigation

You can expect an initial response within a few days. If the report is accepted,
we'll work on a fix and coordinate a release; you'll be credited in the advisory
unless you prefer otherwise.

## Scope

This project orchestrates calls to third-party LLM providers via LiteLLM, executes
user-supplied tools, and can spawn sub-agents. Reports that are especially relevant:

- Ways for untrusted model output to cause unintended local code execution
- Sandbox escapes in `CodeExecutionSkill`
- Leakage of API keys or memory contents into logs, stream events, or telemetry
- Bypasses of `AgentHooks.approve_tool` (the human-in-the-loop denial path)
- Bypasses of the `max_spawn_depth` bound on recursive agent spawning

Out of scope:

- Vulnerabilities in upstream LLM providers or in LiteLLM itself — please report
  those to the relevant project
- Attacks that require the operator to deliberately configure an unsafe tool

## Handling secrets

`deepcrew-ai` never persists provider credentials. API keys are read from the
environment by LiteLLM at call time. If you find a path where a key is written to
a log line, a `StreamEvent` payload, or an OpenTelemetry span attribute, treat it
as a security issue and report it privately.
