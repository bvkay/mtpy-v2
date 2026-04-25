# mtpy-v2 GB decomposition contribution — working constitution

Loaded at the start of every Claude Code session in this fork.
Overrides ad-hoc prompts. If instructions here conflict with mtpy-v2's
own conventions in `CONTRIBUTING.md`, `pyproject.toml`, or
`tests/conftest.py`, **mtpy-v2's conventions win**. We are guests in
this repo; our job is to add a feature cleanly, not to remake the
project.

---

## Mission

Add Groom-Bailey / McNeice-Jones magnetotelluric decomposition to
mtpy-v2 as a sibling to the existing Bibby 2005 `find_distortion`
analysis. The deliverable is a series of pull requests against
`MTgeophysics/mtpy-v2:main` from `bvkay/mtpy-v2`.

**Audience**: the MT community. Code is read by external reviewers,
maintained by people other than us, and used in published research
indefinitely. Comments, docstrings, error messages assume an external
audience. No internal-shorthand, no acronyms without expansion, no
"obvious" steps left implicit.

**Explicit non-goals**:

- Replacing or modifying `find_distortion` (Bibby 2005). It stays
  exactly where it is.
- Reimplementing mtpy-v2 infrastructure we don't need to (readers,
  the `Z` class, the phase tensor, etc.). We use what's there.
- Inventing new public types where mtpy-v2 already has equivalents.
  Inputs are `Z`, `MT`, `MTCollection`. Output is a result dataclass
  carrying an `xarray.Dataset` and a regenerated `Z`.
- Pursuing the Bayesian extension in this contribution. That is
  separate work, scoped after the deterministic implementation lands.

---

## Canonical sources

| What | Where | Status |
|---|---|---|
| Fortran reference (validation oracle) | `~/MT_Decomp/` (separate repo) | Read-only; consulted via cross-validation tests that skip when absent |
| Existing distortion code (do not modify) | `mtpy/core/transfer_function/z_analysis/distortion.py` | Bibby 2005 implementation; sibling to ours |
| `Z` class (canonical impedance type) | `mtpy/core/transfer_function/z.py` | Public input type for single-site decompose |
| `MT` class (station with TF + metadata) | `mtpy/core/mt.py` | Public input type (delegates to `MT.Z`) |
| `MTCollection` (multi-site, MTH5-backed) | `mtpy/core/mt_collection.py` | Public input type for joint multi-site decompose |
| Existing tests fixtures | `tests/conftest.py` | 26K of Z and MT fixtures; we use them |

The strike_py development repo at `~/MT_Decomp` contains the
f2py-wrapped Fortran kernels that established Task 7 tolerances and
the optimiser-gap finding. Cross-validation against those kernels
happens via a separate test module that imports conditionally. We do
not depend on `~/MT_Decomp` being present; CI doesn't have it.

---

## Stack — tiered commitments

**Tier 1 — Architectural commitments. Hard.**

- Pure Python implementation. No Fortran, no f2py, no compiled
  extensions in the contribution. NumPy + SciPy + xarray only.
- `scipy.optimize.least_squares(method='trf', jac=user, bounds=...)`
  is the optimiser. Same as strike_py used.
- The GB89 mathematical specification is the ground truth for
  numerical correctness. The Fortran reference is consulted for
  continuity validation, not as authoritative truth (per the
  optimiser-gap finding documented in
  `~/MT_Decomp/forensic_report/18_optimiser_gap_inversion.md`).
- mtpy-v2's existing `Z`, `MT`, `MTCollection` types are the public
  API surface. No internal dataclasses leak out.
- No new core dependencies. Bayesian work goes in
  `[project.optional-dependencies] bayesian` later.

**Tier 2 — Default implementation choices. Strong preference.**

- `xarray.Dataset` for per-period results. Matches `TFBase`'s pattern;
  gives free serialisation, alignment, and a natural slot for the
  `sample` dimension when Bayesian samples appear.
- `dataclasses.dataclass` for the `DecompositionResult` shell. Plain
  Python; `__repr__` is auto-generated.
- `numpy.random.Generator` (PCG64) for any RNG. Seeded explicitly.
- Loguru for any logging (matching mtpy-v2's house style); never
  stdlib `logging`.
- Tests use mtpy-v2's existing fixtures from `tests/conftest.py`.
  We do not roll our own Z/MT factories.

**Tier 3 — Replaceable backend components.**

- Alternate optimisers (NLopt SLSQP) if TRF proves inadequate.
- Bootstrap implementation (the Chave 2014 caveat is documented
  there; resampling is for direct GB parameters only, not invariant-
  derived quantities).
- Plotting backends.

---

## Critical invariants

Violations are review-blocking defects.

1. **GB89 specification is the ground truth.** The decomposition
   solves the Groom & Bailey 1989 problem. Cross-validation against
   the Fortran reference is for continuity (catches accidental
   divergence in places where Python and Fortran *should* agree),
   not for correctness (the Fortran terminates early; scipy finds a
   lower-cost minimum on the canonical example).

2. **The existing `find_distortion` is not modified.** Our
   contribution is a sibling, not a replacement. The signatures of
   `Z.remove_distortion`, `Z.estimate_distortion`,
   `find_distortion`, and `remove_distortion_from_z_object` are
   untouched.

3. **Public types are mtpy-v2 native.** `decompose(z: Z, ...)` takes
   a `Z`. `decompose_joint(collection: MTCollection, ...)` takes a
   `MTCollection`. The result holds an `xarray.Dataset` and a
   regenerated `Z` for `regional_z`. Internal helpers may use NumPy
   arrays, but they do not appear in the public API.

4. **Unit and coordinate conventions are documented and asserted.**
   `Z` carries an internal scale factor between user-facing `.z` and
   `_dataset.transfer_function`. We read via `.z` and write by
   constructing fresh `Z` objects (the same pattern
   `remove_distortion_from_z_object` uses). Coordinate frame
   (NED vs ENU, time-harmonic sign, clockwise-from-x) is documented
   in every result and asserted in tests.

5. **Tolerances follow Task 7 conventions.** Continuity validation
   against the Fortran reference uses ±1° on regional azimuth, ±0.5°
   on twist/shear, ±0.01 on log10(rho_a) and log10(rho_b), ±0.5° on
   phases, ±5% relative on RMS misfit. Bands where the Fortran
   reference reports `ifail != 0` are marginal and not required to
   match.

6. **GB 90° / shear-sign symmetry is handled explicitly.** Two
   decompositions differing by `(strike + 90° mod 180°, -shear,
   twist)` represent the same physical solution. The comparison
   utility recognises both branches as equivalent.

7. **No new core dependencies.** Bayesian work uses
   `[project.optional-dependencies] bayesian = [...]`, with the
   relevant module raising an informative `ImportError` if not
   installed. Standard scientific Python pattern (sklearn,
   statsmodels, arviz).

8. **Static-shift sign convention agrees with `MT.remove_static_shift`.**
   Whatever convention `MT.remove_static_shift` uses for `ss_x`,
   `ss_y` (factors that multiply apparent resistivity), our `g`
   (gain) is consistent with it. Users must not be able to
   double-correct.

9. **Error propagation on regional Z is not optional.** The Jacobian
   of the GB transformation at the optimum gives propagated errors on
   `regional_z`. Downstream inversion needs them. They ship from day
   one.

10. **Tests use mtpy-v2's existing fixture system.** No reinventing of
    `make_random_z` or similar. Use what's in `tests/conftest.py`.

---

## How to behave in a session

### Before writing code

1. Read the relevant existing module (`distortion.py`, `z.py`,
   `pt.py`) to match conventions.
2. State the plan in plain text, 1-2 paragraphs.
3. Name the invariants at risk (by number).
4. If the request would violate an invariant, say so and ask before
   proceeding.

### While writing code

- Black 88-col. isort. autoflake. Pre-commit will check; run it.
- Type hints on public functions. Match the typing style of nearby
  modules (mostly informal in mtpy-v2; don't impose stricter).
- Loguru for logging if any.
- xarray.Dataset where the parent code uses it (it does, throughout
  the transfer_function package).
- Diffs stay scoped. If a session naturally produces a 600-line diff,
  it's two sessions.

### After writing code

- `pre-commit run --all-files` (or just on changed files) before
  commit. Failures are fix-and-recommit, not bypass.
- `pytest tests/core/transfer_function/z_analysis/ -q` to confirm
  z_analysis tests still pass.
- Report what changed and which invariants were touched.

### Refuse

- New core dependencies.
- Modifications to `distortion.py` or `Z.remove_distortion`.
- Public APIs that don't accept mtpy-v2 native types.
- Tests that don't use the existing fixture system.
- Silent unit-convention assumptions (always document NED vs ENU,
  clockwise-from-x, etc.).

---

## Cross-repo workflow

`~/MT_Decomp` is a separate clone of the strike_py development repo.
It holds:

- The f2py-wrapped Fortran kernels (validation oracle for kernels)
- The container reference (validation oracle for end-to-end
  decompositions)
- The forensic deliverables (audit trail of what we know about the
  Fortran reference, including the optimiser-gap finding)

The relationship between `~/MT_Decomp` and `~/mtpy-v2`:

- `~/MT_Decomp` is read-only from this fork's perspective.
- The fork does not import from `~/MT_Decomp` at runtime.
- Cross-validation tests in `tests/cross_validation/` (added in a
  later session) detect `~/MT_Decomp`'s presence and run
  Fortran-vs-Python comparisons when it's available; they skip
  cleanly when it's not.
- CI (when set up) does not have `~/MT_Decomp`. Cross-validation tests
  are local-developer-only.

This separation matters because the contribution must work for users
who don't have the Fortran reference. The Fortran is *our* validation
infrastructure, not theirs.

---

## Deeper docs (lazy-loaded on demand)

- `docs/ARCHITECTURE.md` — module layout in the fork, public/private
  boundary, data flow Z → DecompositionResult
- `docs/OPERATIONAL.md` — environment, git, cross-repo workflow,
  pre-commit, pytest invocations
- `docs/WORKFLOW.md` — checklists for the planned sessions

The strike_py development repo (`~/MT_Decomp/`) has its own
governance docs covering the Fortran reference and the f2py
infrastructure. Those are not duplicated here.
