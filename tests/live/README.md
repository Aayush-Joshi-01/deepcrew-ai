# Live tests

Opt-in integration tests that call a **real** LLM (Gemini). They are **skipped by
default** so `pytest` stays fully mocked and offline — see `CONTRIBUTING.md`.

## Running

```bash
export GEMINI_API_KEY=...            # required; tests skip without it
pytest --live tests/live -v          # run the ones whose code changed since last pass
pytest --live --relive tests/live    # ignore the ledger, run every live test
DEEPCREW_LIVE_MODEL=gemini/gemini-2.5-flash-lite pytest --live tests/live   # override the model
```

Default model: `gemini/gemini-3.5-flash-lite`.

## The ledger

`LEDGER.json` (machine) and `LEDGER.md` (human) record which **features** have
passed a live test. Each live test declares the source files it exercises:

```python
pytestmark = pytest.mark.live(
    feature="episodic_memory",
    watch=["src/deepcrew/episodic_memory.py", "src/deepcrew/runner.py"],
)
```

On `pytest --live`:

1. A feature whose `watch` files hash identically to the last recorded pass is
   **skipped** ("unchanged since live pass on …") — no tokens spent.
2. Any other feature runs. On success the ledger is rewritten with the new hash,
   today's date, and the model used, and `LEDGER.md` is regenerated.
3. `--relive` bypasses step 1 and runs everything.

So a feature is live-verified once, then only re-verified when its code changes.
Commit `LEDGER.json` / `LEDGER.md` so that "verified" state travels with the repo.

## CI

`.github/workflows/ci.yml` has a `live` job that runs only on **manual**
`workflow_dispatch` and needs a `GEMINI_API_KEY` repository secret. It never runs
on pull requests, so the secret is never exposed to forks.
