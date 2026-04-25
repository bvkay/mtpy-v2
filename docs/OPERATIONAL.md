# Operational rules — mtpy-v2 GB decomposition contribution

How to work in this fork: env, git safety, cross-repo workflow,
pre-commit, test discipline.

mtpy-v2's own `CONTRIBUTING.md` and `pyproject.toml` are the parent
conventions. Where this file is silent, follow theirs.

---

## Python environment

Use `conda run -n strike <command>` for everything (this is the
existing env from the strike_py work; mtpy-v2 has been installed into
it editable).

Verify env at session start:

```bash
conda run -n strike python -c "import sys; print(sys.executable)"
# path includes envs/strike

conda run -n strike python -c "import mtpy; print(mtpy.__file__)"
# path is inside ~/mtpy-v2
```

If either fails, the env or install is broken. Do not proceed.

When running pytest, always use `python -m pytest ...` (not bare
`pytest`). On this machine the bare `pytest` resolves to
`/usr/bin/pytest` which uses the system Python and cannot import
matplotlib or other mtpy deps. `python -m pytest` ensures the
strike-env interpreter is used.

---

## Git workflow

Two remotes:

- `origin` → `bvkay/mtpy-v2` (the fork)
- `upstream` → `MTgeophysics/mtpy-v2` (canonical)

Feature branch: `feature/groom-bailey-decomposition`. All work for the
deterministic implementation goes here. Future feature branches
(plotting, multi-site, Bayesian) are siblings off `main`, not
children of this one.

### Staying current with upstream

Periodically:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
git checkout feature/groom-bailey-decomposition
git merge main
```

Resolve conflicts in `feature/groom-bailey-decomposition`. Do not
rebase the feature branch once it has any commits — we'll be
opening PRs from it and rewriting history breaks the PR.

### Commit messages

Format: `<area>: <imperative summary>`. Examples:
- `decomposition: scaffold module and DecompositionResult dataclass`
- `decomposition: pure-Python implementation of estim_imp`
- `tests/decomposition: add Z type-roundtrip contract test`

No "co-authored-by" lines for Claude. Write messages as instructions
to the codebase.

### Pushing

The fork's branches push to `origin`. The user pushes manually after
reviewing the diff. Do not run `git push` from a Claude Code session
without explicit approval per-push.

---

## Cross-repo workflow

`~/MT_Decomp/` is the strike_py development repo. From the perspective
of this fork:

- It is read-only. We do not commit to it from sessions in this
  fork.
- We do not import from it at runtime. The contribution must work
  for users who don't have it.
- Cross-validation tests in `tests/cross_validation/` (added later)
  detect `~/MT_Decomp`'s presence at test-collection time and skip
  when absent.

The detection pattern (added in a later session, documented here for
reference):

```python
import os
import pytest

MT_DECOMP_PATH = os.environ.get("MT_DECOMP_PATH",
                                  os.path.expanduser("~/MT_Decomp"))
HAS_FORTRAN_REF = os.path.isfile(
    os.path.join(MT_DECOMP_PATH, "fortran", "kernels.pyf")
)

pytestmark = pytest.mark.skipif(
    not HAS_FORTRAN_REF,
    reason=f"Fortran reference not at {MT_DECOMP_PATH}; set "
           f"MT_DECOMP_PATH or skip cross-validation",
)
```

This means: cross-validation tests are local-developer-only. CI does
not run them. The contribution stands on its synthetic and
specification-level tests for correctness; the cross-validation is
extra confirmation for us.

---

## Pre-commit and code style

mtpy-v2 uses pre-commit hooks. Install once:

```bash
cd ~/mtpy-v2
pre-commit install
```

After installation, `git commit` automatically runs Black (88-col),
isort, autoflake, and any other hooks declared in
`.pre-commit-config.yaml`. Failures abort the commit; fix and retry.

To run hooks manually before commit (recommended):

```bash
pre-commit run --files <files>     # only the files we changed
# or
pre-commit run --all-files          # whole repo (slow)
```

Do not bypass hooks (`--no-verify`). PRs with style failures bounce.

---

## Testing

Run our tests:

```bash
conda run -n strike python -m pytest tests/core/transfer_function/z_analysis/ -v
```

Run the full z_analysis suite (existing + ours):

```bash
conda run -n strike python -m pytest tests/core/transfer_function/z_analysis/
```

Avoid running the full mtpy-v2 suite — there are known collection
errors from optional modules (mtpy_data, aurora/mth5 spectre
mismatch) that are not our concern. They'll dominate the output and
distract from real signal.

If we want to confirm we haven't broken anything broader:

```bash
conda run -n strike python -m pytest tests/core/ tests/imaging/ -q --ignore=tests/processing/aurora --ignore=tests/modeling/simpeg
```

Cross-validation tests (later sessions):

```bash
conda run -n strike python -m pytest tests/cross_validation/ -v
```

These skip cleanly without `~/MT_Decomp`. With it, they run.

---

## Logging

mtpy-v2 uses loguru everywhere. From the existing `distortion.py`:

```python
from loguru import logger
```

When our module needs to log (which it shouldn't in stubs, but
eventually it will), use loguru. Never stdlib `logging`.

---

## No emoji or decorative Unicode

Same rule as the strike_py port. Emoji break Windows cp1252 consoles
and Git Bash. No emoji in source, output, commit messages, or
docstrings. Plain ASCII for status: `[OK]`, `[FAIL]`, `[WARN]`.

mtpy-v2 itself has loguru emoji icons in level indicators. Those are
loguru's; we don't add our own.

---

## Scratch directory

Any throwaway diagnostic scripts go in `scratch/` (gitignored). Same
convention as strike_py.

Before any commit:

```bash
ls *.py 2>/dev/null  # should show no loose Python at the repo root
git status            # verify clean working tree apart from intended changes
```

---

## Working tree safety

Same rules as strike_py: no `git reset --hard`, no `git clean -fd`,
no `rm -rf` under the repo root without explicit approval. Use
`git stash push -u -m <name>` for reversible "clean state" needs.

`git reset --soft` for restructuring unpushed commits requires:
1. Backup branch first: `git branch backup-$(date +%Y%m%d-%H%M)`
2. Verify backup exists.
3. User approval with the exact reset command.
4. Execute.
