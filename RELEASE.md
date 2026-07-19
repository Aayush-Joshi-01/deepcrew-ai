# Release & CI Runbook

How the automation works and exactly what to do to cut a release.

## Workflows at a glance

| Workflow | File | Fires when | What it does |
| --- | --- | --- | --- |
| **CI** | `.github/workflows/ci.yml` | push/PR to `main` touching `src/**`, `tests/**`, `pyproject.toml` | ruff, ruff format, mypy, version-sync check, pytest on 3.11/3.12/3.13 |
| **Deploy Docs** | `.github/workflows/docs.yml` | push to `main` touching `pages/**` | Uploads `pages/` and deploys to GitHub Pages |
| **Release** | `.github/workflows/publish.yml` | push of a `v*` tag | Builds, verifies, **waits for approval**, publishes to PyPI, creates the GitHub Release |

Anything not matching those path filters runs **nothing**. A commit that only edits
`*.md`, `examples/`, or docs prose will not start CI. To force a run anyway, use
**Actions → (workflow) → Run workflow** (`workflow_dispatch` is enabled on CI and Deploy Docs).

## One-time setup (already done — listed so it can be rebuilt)

1. **Settings → Pages → Source = "GitHub Actions".**
   Branch-based Pages only supports `/` or `/docs`; it can never serve `pages/`.
2. **Custom domain** `deepcrew-ai.aayushjoshi.dev` set in Settings → Pages, with
   `pages/CNAME` committed so the artifact carries it. Enforce HTTPS on.
3. **`pypi` environment** with **required reviewers** (Settings → Environments → pypi).
   This is what makes the release pause for manual approval.
4. **`PYPI_API_TOKEN`** added as an *environment secret on the `pypi` environment*
   (not a repo secret, not on any other environment).

> Environment secrets are only readable by jobs that declare that environment.
> A token stored on a different environment resolves to empty and twine fails with 403.

## Cutting a new release

### 1. Bump the version in **both** places
CI fails the build if these disagree.

- `pyproject.toml` → `[project] version`
- `src/deepcrew/__init__.py` → `__version__`

### 2. Update `CHANGELOG.md`
Add the new section and update the compare links at the bottom.

### 3. Merge to `main`
Land everything through a PR (CI runs on the PR). Confirm CI is green.

### 4. Tag and push
The tag must be `v` + the exact version, or the Release workflow fails its
tag/version check before building anything.

```bash
git checkout main
git pull
git tag -a v0.5.0 -m "deepcrew-ai v0.5.0 — <summary>"
git push origin v0.5.0
```

### 5. Approve the release
The run builds, verifies `py.typed` is in the wheel, runs `twine check`, then **stops**.

- Go to **Actions → Release → the run → Review deployments**
- Tick `pypi` → **Approve and deploy**

On approval: PyPI upload runs, then the GitHub Release is created automatically
with the `.whl` and `.tar.gz` attached and generated notes.

### 6. Verify
```bash
curl -s https://pypi.org/pypi/deepcrew-ai/json | python -c "import sys,json;print(json.load(sys.stdin)['info']['version'])"
gh release view v0.5.0
```

## Docs-only change

Edit under `pages/` (and the matching `pages/md/*.md` twin), push to `main`.
**Deploy Docs** runs automatically. Verify:

```bash
curl -s https://deepcrew-ai.aayushjoshi.dev/ | grep -o '<title>[^<]*</title>'
```

Expect the site title, not a Jekyll-rendered README.

> The "Copy page" button reads markdown **inlined** into each HTML page
> (`<script type="text/plain" class="page-md">`), not a fetched file — that is what makes it
> work on `file://` too. After editing a `pages/md/*.md` twin, re-inline it so the button
> serves current content.

## Undo / recovery

| Situation | Fix |
| --- | --- |
| Tagged the wrong commit, **not yet approved** | `git push --delete origin vX.Y.Z && git tag -d vX.Y.Z`, then re-tag |
| Release job failed before PyPI upload | Fix, delete + re-push the same tag |
| **Already published to PyPI** | **Irreversible.** A version can never be re-uploaded — bump to the next patch |
| Pages showing the README again | Settings → Pages → Source must be "GitHub Actions" |

## Gotchas

- **PyPI versions are permanent.** Deleting a release on PyPI does not free the number.
- **Path filters and required status checks:** if you ever make CI a *required* check in
  branch protection, a docs-only PR will sit forever waiting for a check that never runs.
  Either don't require it, or add a skip-job that reports success.
- **`pages-build-deployment`** is GitHub's built-in legacy Pages builder. It has no file in
  this repo and stops firing once Source = GitHub Actions. It only lingers in the Actions
  sidebar because of past runs.
