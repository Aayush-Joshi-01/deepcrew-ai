## What does this change?

<!-- A short description of the change and the reasoning behind it. -->

## Why?

<!-- The problem being solved. Link any related issue: Fixes #123 -->

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (existing behaviour changes)
- [ ] Documentation only
- [ ] Internal / tooling

## Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `mypy src/` passes
- [ ] `pytest` passes
- [ ] Added or updated tests covering this change
- [ ] Tests do not call any live LLM API (`litellm.acompletion` is mocked)
- [ ] Updated documentation in `pages/` **and** its Markdown twin in `pages/md/`, if user-facing
- [ ] Any new optional dependency is lazily imported

## Notes for the reviewer

<!-- Anything worth calling out: trade-offs, follow-ups, areas you're unsure about. -->
