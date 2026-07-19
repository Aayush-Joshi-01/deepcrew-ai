# Contributing to deepcrew-ai

Thanks for your interest in contributing. This document covers how to get set up,
what the project's conventions are, and what a good pull request looks like.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** — open an issue with a minimal reproduction.
- **Request a feature** — open an issue describing the problem you're solving, not just the API you want.
- **Improve documentation** — the site under `pages/` and the guides are always improvable.
- **Send a pull request** — bug fixes, new providers, new skills, tests.

If you're planning a large change, please open an issue first so we can agree on the
approach before you invest the time.

## Getting set up

You need Python 3.11+.

```bash
git clone https://github.com/<your-username>/deepcrew-ai.git
cd deepcrew-ai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[dev,otel,fastapi]
```

Run the full check suite exactly as CI does:

```bash
ruff check .
ruff format --check .
mypy src/
pytest
```

`ruff format .` (without `--check`) will fix formatting for you.

## Pull request workflow

External contributors do not have push access — fork the repo and open a PR from
your fork against `main`.

1. Fork, then branch off `main`: `git checkout -b fix/short-description`
2. Make your change, with tests.
3. Ensure `ruff check .`, `ruff format --check .`, `mypy src/`, and `pytest` all pass.
4. Push to your fork and open a PR against `main`.
5. Fill in the PR template — especially *what* changed and *why*.

Keep PRs focused. One logical change per PR is much easier to review than a
sweeping refactor.

### Commit messages

Use a conventional prefix and an imperative subject:

```
feat: add Anthropic prompt caching support
fix: prevent recursion when loop_config is set
docs: expand the StreamPolicy guide
test: cover the retry/fallback interaction
chore: bump ruff to 0.6
ci: scope workflows by path
```

## Project conventions

These are enforced by review, and mostly by ruff/mypy:

- **Async-first.** Public APIs are `async`. Parallel fan-out uses
  `asyncio.gather(..., return_exceptions=True)`.
- **`from __future__ import annotations`** at the top of every module.
- **Dataclasses** for configuration objects.
- **All exceptions inherit from `DeepCrewError`** (`src/deepcrew/exceptions.py`).
- **Optional dependencies are lazily imported.** `fastapi`, `redis`, and
  `opentelemetry` must be imported *inside* the function or class that needs them and
  exposed via `__getattr__` in `deepcrew/__init__.py`. A bare `pip install deepcrew-ai`
  must always import cleanly.
- **Every consequential backend action emits a `StreamEvent`.** If you add a new event
  type, also decide whether it belongs in `StreamPolicy`'s `standard` preset
  (`src/deepcrew/stream.py`).
- **The version lives in two places** — `pyproject.toml` and
  `src/deepcrew/__init__.py` (`__version__`). They must match; CI fails otherwise.

See [CLAUDE.md](CLAUDE.md) for a fuller architecture map.

## Testing

**Tests must never call a live LLM API.** Mock `litellm.acompletion` with
`unittest.mock.patch` / `AsyncMock`, using the hand-built `MagicMock` chunk helpers you'll
find in the existing test modules.

One sharp edge worth knowing: if you add a field that the runner reads off a streamed
chunk's `delta` via `getattr(...)` (as it does for `reasoning_content`), you **must** also
set it explicitly in every test's chunk-builder helper. `MagicMock` returns a truthy mock
for any unset attribute, which will silently break assertions across the whole suite.

Add tests for the behaviour you change. Bug fixes should come with a regression test.

## Documentation

The site under `pages/` is hand-written static HTML — there is no site generator.

- Each feature has a guide in `pages/guides/`.
- Every HTML page has a Markdown twin in `pages/md/`. **Update both together.**
- The "Copy page" button reads Markdown inlined into the page itself
  (`<script type="text/plain" class="page-md">`) rather than fetching a file, which is why
  it works offline and from `file://`. If you edit a `pages/md/*.md` twin, re-inline it so
  the button serves current content.
- **No emoji** anywhere in the documentation.
- Keep pages mobile-responsive; wide content must scroll inside its own container.

## CI

CI is scoped by path, so it only runs when it can actually tell you something:

| Change | What runs |
| --- | --- |
| `src/**`, `tests/**`, `pyproject.toml` | Lint, mypy, pytest on 3.11 / 3.12 / 3.13 |
| `pages/**` | Docs deploy (on `main` only) |
| Markdown, `examples/**` | Nothing |

If your PR touches only docs and you see no checks, that is expected.

## Releases

Releases are cut by the maintainer. See [RELEASE.md](RELEASE.md) for the procedure.

## Licence

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers this project.
