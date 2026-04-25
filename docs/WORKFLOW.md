# Workflow — planned sessions and checklists

The contribution arrives in mtpy-v2 over a planned sequence of
sessions, each producing a single commit (or rarely two) on
`feature/groom-bailey-decomposition`. The branch pushes to
`origin` and the PR opens against `MTgeophysics/mtpy-v2:main` after
session 9.

---

## Session sequence

**Session 1 — Scaffolding and governance** (complete, b9b1fdd).
Module scaffold, DecompositionResult dataclass, governance docs,
type contract tests.

**Session 2 — Pure-Python kernels: simple ones** (complete, 52c0bbb).
Seven simple Fortran kernels reimplemented in NumPy with unit and
cross-validation tests.

**Session 3 — Pure-Python `objfun`** (complete, 573ef2e).
Residuals + analytic Jacobian, validated by finite differences and
Fortran cross-validation (kernel-level cross-validation deferred for
objfun specifically due to convention divergence; documented in
project memory).

**Session 4 — End-to-end single-site decomposition** (complete, c5e06f2).
Wired Session 2-3 kernels into _solve_band, decompose(z: Z), with
synthetic recovery, Z type-handling, and Fortran continuity tests.
Single-start TRF; multi-modal-surface limitation acknowledged.

**Session 5 — Multi-start optimisation** (planned).
Add multi-start to address the multi-modal-surface limitation
discovered in Session 4. Each band runs n_starts optimisations from
diverse initial guesses (hybrid: canonical + 90-rotated + random
perturbations). Modes are discovered via canonical-form clustering;
mode probabilities computed via the Laplace approximation. Result
metadata exposes per-band per-mode information.

This session is necessary before bootstrap because bootstrap CIs
around a wrong local minimum are not meaningful.

**Session 6 — Bootstrap with Chave 2014 caveats** (planned).
Bootstrap CIs for direct GB parameters (gain, twist, shear, strike,
regional Z) on top of multi-start. Documentation flags reliability
for direct parameters and unreliability for invariant-derived
quantities per Chave (2014).

**Session 7 — Multi-site joint decomposition** (planned).
decompose_joint(collection: MTCollection, ...). McNeice-Jones-style
joint fit with shared strike and per-site distortion. Synthetic
multi-site validation.

**Session 8 — Plotting** (planned).
mtpy/imaging/plot_decomposition.py with strike rose, twist/shear vs
period, chi-squared fit, parameter bounds. Per the recon, without a
plot the feature won't get used.

**Session 9 — Integration polish + PR prep** (planned).
Z.decompose(), MT.decompose(), MTCollection.decompose() thin
methods. Documentation, final regression, PR description.

After Session 9: PR opens against MTgeophysics/mtpy-v2:main. The PR
review may produce follow-up sessions before merge.

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
