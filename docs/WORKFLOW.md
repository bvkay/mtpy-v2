# Workflow — planned sessions and checklists

The contribution arrives in mtpy-v2 over a planned sequence of
sessions, each producing a single commit (or rarely two) on
`feature/groom-bailey-decomposition`. The branch pushes to
`origin` and the PR opens against `MTgeophysics/mtpy-v2:main` after
session 8.

---

## Session sequence

**Session 1 — Scaffolding and governance** (this session).
Create `decomposition.py` with `DecompositionResult` and stub
functions raising `NotImplementedError`. Place CLAUDE.md and the
three docs/. Add `.gitignore` rules for Claude artefacts. Add a
single test file exercising the type round-trip contract. No
numerics.

**Session 2 — Pure-Python kernels: simple ones**.
Reimplement the seven trivially-portable Fortran kernels in pure
NumPy: `mat_multiply`, `extreme`, `convz2r`, `convz2p`,
`calc_error`, `jkvar`, `estim_imp`. Each gets a synthetic-input
unit test. Cross-validation against the f2py-wrapped Fortran in
`~/MT_Decomp` is established as a separate test module that skips
when absent.

**Session 3 — Pure-Python `objfun`**.
The intricate kernel: residuals + analytic Jacobian for the GB
problem. Port the formulas verbatim from the Fortran. Validate
against the f2py-wrapped reference and against the integration
test (estim_imp → alpha → objfun → zero residuals).

**Session 4 — `decompose(z: Z)` single-site driver**.
Wire the pure-Python kernels into the banded-decomposition driver
with `Z` as input and `DecompositionResult` as output. The numerics
are the strike_py `_solve_band` and banded loop, ported and
adapted. Continuity test against the Fortran reference using Task 7
tolerances.

**Session 5 — Bootstrap with Chave 2014 caveats**.
Bootstrap CIs for direct GB parameters (g, t, e, s, θ, regional Z).
Documentation flags reliability for direct parameters and
unreliability for invariant-derived quantities per Chave 2014.

**Session 6 — Multi-site joint decomposition**.
`decompose_joint(collection: MTCollection, ...)`. McNeice-Jones-
style joint fit with shared strike and per-site distortion.
Synthetic multi-site validation.

**Session 7 — Plotting**.
`mtpy/imaging/plot_decomposition.py` with strike rose, twist/shear
vs period, χ² fit, parameter bounds. Without a plot the feature
won't get used; per the recon, this is not skippable.

**Session 8 — Integration polish**.
`Z.decompose()` and `MT.decompose()` thin methods.
`MTCollection.decompose()` for joint fits. Final regression. PR
preparation: clean up commits if needed, write the PR description.

After Session 8: PR opens against `MTgeophysics/mtpy-v2:main`. The
PR review and discussion may produce follow-up sessions before
merge.

---

## Per-session checklist

Every session begins with:

1. State of branch is clean and on
   `feature/groom-bailey-decomposition`.
2. Verify env (`conda run -n strike python -c ...`).
3. Verify imports (`mtpy`, `Z`, `MT`, etc.).
4. Confirm baseline test pass count (`python -m pytest
   tests/core/transfer_function/z_analysis/`).

Every session ends with:

1. Pre-commit hooks pass on changed files.
2. New tests pass.
3. Existing z_analysis tests still pass.
4. Staged diff is shown and approved.
5. Single commit with clear message.
6. Branch is not pushed; user pushes when satisfied.

---

## Things to watch across sessions

- Cross-validation drift. As sessions add features, the
  `tests/cross_validation/` module grows. After every numerical
  change, run cross-validation against `~/MT_Decomp` and confirm the
  Fortran-vs-Python agreement holds (within Task 7 tolerances).

- `Z` API drift. mtpy-v2 is actively developed. If `Z`'s public API
  changes upstream during our work, our internal type-translation
  helpers may need updating. Pull `upstream/main` before each
  substantive session.

- Test fixture changes. mtpy-v2's `tests/conftest.py` may evolve;
  fixtures we depend on may change shape. Same treatment as the `Z`
  API — pull and re-test before substantive work.

- mtpy-v2 known failures. The test suite has 4 collection errors
  from missing optional modules. We ignore those, but we want to
  notice if more appear that *do* touch our area.
