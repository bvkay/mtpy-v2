# Decomposition validation strategy

This document records what each decomposition method in
[`mtpy/core/transfer_function/z_analysis/decomposition/`](../mtpy/core/transfer_function/z_analysis/decomposition/)
is validated against, what kind of confidence each validation
provides, and which validations remain open.

## Single-site Groom-Bailey (`decompose`)

**What is checked.** The unit and integration suite in
[`tests/core/transfer_function/z_analysis/test_decomposition.py`](../tests/core/transfer_function/z_analysis/test_decomposition.py),
plus the cross-validation tests in
[`tests/cross_validation/`](../tests/cross_validation/), exercise:

- Recovery of GB parameters on noiseless single-band synthetics
  built with `_estim_imp` (the GB forward model). Every synthetic
  recovery test asserts the optimiser returns the input parameters
  to machine precision.
- The 90-degree symmetry: synthetic data fitted in a strike-bounded
  upper branch round-trips through `alternate_branch()` and gives
  back the lower-branch fit byte-for-byte where possible.
- Multi-start mode clustering on bimodal cost surfaces.
- Parametric bootstrap CI generation under a controllable noise
  model.
- A continuity test against the older `groom_bailey-decomposition`
  branch's output on synthetic data
  ([`test_decomposition_continuity.py`](../tests/cross_validation/test_decomposition_continuity.py)).
- A Fortran-kernel parity test
  ([`test_kernels_against_fortran.py`](../tests/cross_validation/test_kernels_against_fortran.py))
  comparing `_estim_imp`, `_objfun`, and the residual normalisation
  against the historical GB Fortran implementation at machine
  precision.

**Confidence.** High. The synthetic recovery is exact, the Fortran
parity is verified, and the parametric-bootstrap CIs are tested
under a known noise model.

## Disambiguation framework (`symmetries`)

**What is checked.**
[`tests/test_disambiguation.py`](../tests/test_disambiguation.py)
exercises each fold strategy individually and verifies the four
named strategies all recover the same C tensor (up to the GB
symmetry's gauge equivalence) on a synthetic site. The
`alternate_branch` complement is checked in
[`tests/test_alternate_branch.py`](../tests/test_alternate_branch.py).

**Confidence.** High. Direct unit tests on each fold callable plus
end-to-end tests on the `decompose` integration path.

## Bibby-Caldwell-Brown (`decompose_bibby`)

**What is checked.**
[`tests/test_bibby.py`](../tests/test_bibby.py) covers exact recovery
on a 2-D synthetic in BCB's diagonal-unity gauge, the no-distortion
identity case, and Frobenius-normalised consistency with the
Groom-Bailey output on a small-distortion synthetic.

**Confidence.** Medium. The diagonal-unity gauge differs from the
GB gauge in an established way; cross-method agreement is verified
on the Frobenius-normalised C tensors (gauge-invariant). No
published-data validation yet.

## McNeice-Jones joint (`decompose_mcneice_jones`)

This is the most computationally expensive method in the package
and the one most exposed to local-minima problems at high site
counts. Validation is layered:

### 1. Algorithm correctness (Phase-1 unit tests)

[`tests/test_mcneice_jones.py`](../tests/test_mcneice_jones.py)
contains four tests:

- `test_mj_synthetic_2D_recovery`: 5 sites with a known shared
  strike of 30°, varied per-site twist/shear, no noise. Recovers
  shared strike to <1° and per-site Frobenius-normalised C tensors
  to <0.05.
- `test_mj_collapses_to_gb_at_single_site`: at `n_sites=1`, MJ
  agrees with single-site GB on strike, twist, shear, and (with
  loose tolerance) gain. The gain comparison is loose because the
  GB gain is gauge-equivalent to a regional-impedance rescaling and
  the optimiser is free to pick any value; this is a known
  property documented on `decompose`'s docstring.
- `test_mj_disambiguation_passes_through`: all four named
  disambiguation strategies are accepted; the resulting
  Frobenius-normalised C tensors agree across strategies within
  0.05.
- `test_mj_input_validation`: guards on `share_strike_within_band`,
  `per_site_distortion`, list-length and uniqueness checks.

**Confidence.** High for algorithm correctness on small synthetics.

### 2. Scaling validation (synthetic 9-site / 27-site)

Two layered scaling tests in
[`tests/test_mj_bc87_validation.py`](../tests/test_mj_bc87_validation.py):

- `test_mj_synthetic_9_sites` (default-on, ~40 s): 9 synthetic
  sites with a shared 30° strike and varied per-site distortion
  (~470 parameters, n_starts=3). Bridges from the 5-site unit
  test to the BC87-shaped 27-site case. Asserts shared strike
  recovery to <2° and per-site Frobenius-normalised C tensors to
  <0.05.
- `test_mj_synthetic_27_sites` (opt-in via
  `MTPY_RUN_SLOW_VALIDATION=1`, ~13 min): the full 27-site BC87-
  shaped synthetic (~1400 parameters, n_starts=5). Same
  tolerances, larger optimiser problem. Validated to pass on
  2026-05-08; gated for routine runs because of the cost.

**Why this matters.** Real surveys typically have tens of sites,
not five; the 5-site unit test cannot expose the local-minima or
conditioning problems that show up at higher site counts. The
27-site synthetic is the smallest reproducible test that exercises
the joint cost function at realistic dimensionality (~1400
parameters per band for 27 sites × 12 frequencies).

**Confidence.** Medium-high. Synthetic data is built with
`_estim_imp` (the same forward model the optimiser fits), so this
is a closed-loop check; it does not validate that the algorithm
recovers ground truth on data the model under-specifies (anisotropy,
3-D effects, processing artefacts in real EDIs).

### 3. Real-data smoke test (`test_mj_bc87_strike_recovery`)

[`tests/test_mj_bc87_validation.py::test_mj_bc87_strike_recovery`](../tests/test_mj_bc87_validation.py),
**opt-in** via `MTPY_RUN_BC87=1`, loads the 27 EDIs of the BC87
LITHOPROBE LIT line (the dataset used by McNeice & Jones, 2001),
restricts every site to the common frequency grid, and runs MJ on
a 1–100 s band. Asserts only that:

- the run completes,
- the joint result is a populated `JointDecompositionResult`,
- the recovered shared strike is finite and inside `[0, 180°)`,
- per-site distortion records are present for every input site,
- chi-squared is positive.

**What this test does NOT assert.** It does not currently compare
the recovered shared strike to McNeice & Jones 2001 Figure 12
strike-vs-period values. A preliminary probe at 27 sites in a
single 1–100 s band converged to a shared strike of ~5° in the
geometric-fold convention; the published Fig 12 reads roughly
25–40° in the same period range. Possible explanations are open:

1. The single-band aggregation (one shared strike for all of
   1–100 s) differs from MJ's per-period strike reporting in
   Fig 12; running narrower bands per period decade may align
   better.
2. With ~1400 parameters per band and only 3 multi-starts, the
   optimiser may have settled in a local minimum that is not the
   physical global one. The test
   `test_mj_synthetic_27_sites` shows the algorithm reaches the
   correct strike on synthetic data with the same site count, so
   the most likely culprits are real-data effects (anisotropy
   in the lower crust, noise, processing artefacts) interacting
   with the local-minima problem.
3. Convention offsets — clockwise from x-axis vs from N, or the
   PT-strike's mtpy 90° offset — could account for an
   ambiguity (90° vs 0°) but not the difference between 5° and
   30°.

**Why the test is conservative anyway.** Until the cause of the
strike discrepancy is investigated, asserting an exact Fig 12
match would lock in a wrong answer or a flaky test. The current
formulation lets the validation infrastructure exist (the test
runs, the data path resolution works, the result type is
populated) without false confidence.

**Confidence.** Low for the strike magnitude. Medium for the
infrastructure (the algorithm runs to completion on real data
with realistic site counts). The exact Fig 12 comparison is open
work.

### Why the layered approach is sufficient for the current claims

The package's claim is that `decompose_mcneice_jones` implements
the McNeice & Jones (2001) algorithm and exposes it cleanly. The
combination of:

1. exact synthetic-recovery on small site counts (algorithm
   correctness),
2. synthetic-recovery on a 27-site BC87-shaped input (scaling), and
3. a smoke test that the algorithm runs on real BC87 EDIs

establishes correctness of the implementation against the
algorithm specification. The claim that the implementation
*reproduces published Fig 12 strikes for the BC87 LIT line* is
**not yet established** and is called out as Phase 2 follow-up
work in the change log. That investigation will need:

- per-period (narrow-band) strike output instead of one big band;
- higher `n_starts` (likely 10+) for the joint problem;
- careful bookkeeping of the published convention vs. mtpy's;
- possibly a side-by-side run against the original McNeice-Jones
  Fortran code on the same EDIs.

## Known limitations

- Pickle-vs-NetCDF round-trip on `DecompositionResult`: 6 NetCDF
  tests fail because the netCDF4 library rejects boolean dataset
  attributes. Pre-dates this branch; tracked in `CHANGELOG.md`.
- BC87 strike magnitude: see above. Open Phase-2 work.
- Bibby ↔ GB scale comparison: only valid up to gauge
  normalisation; the absolute scale of either result is not
  determinable from MT data alone (Bibby et al., 2005). Documented
  on the BCB module docstring and on `test_bibby_consistency_with_gb`.

## Adding a validation

When adding a new decomposition method or changing an existing one:

1. Add a synthetic recovery unit test that hits the GB / BCB / new
   forward model exactly.
2. Add a method-vs-method consistency test on the
   Frobenius-normalised C tensor where applicable.
3. If the method has a published reference dataset, add an opt-in
   real-data test that at minimum smoke-tests the run on that
   data; record any open discrepancy in this document.

References
----------
- McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
  tensor decomposition of magnetotelluric data. *Geophysics*, 66(1),
  158–173.
- Groom, R. W., & Bailey, R. C. (1989). Decomposition of
  magnetotelluric impedance tensors in the presence of local
  three-dimensional galvanic distortion. *Journal of Geophysical
  Research*, 94(B2), 1913–1925.
- Bibby, H. M., Caldwell, T. G., & Brown, C. (2005). Determinable
  and non-determinable parameters of galvanic distortion in
  magnetotellurics. *Geophysical Journal International*, 163(3),
  915–930.
- Caldwell, T. G., Bibby, H. M., & Brown, C. (2004). The
  magnetotelluric phase tensor. *Geophysical Journal
  International*, 158, 457–469.
- Jones, A. G. (1993). The BC87 dataset: tectonic setting, previous
  EM results, and recorded MT data. *Journal of Geomagnetism and
  Geoelectricity*, 45(9), 1089–1105.
- Jones, A. G., Groom, R. W., & Kurtz, R. D. (1993). Decomposition
  and modelling of the BC87 dataset. *Journal of Geomagnetism and
  Geoelectricity*, 45(9), 1127–1150.
