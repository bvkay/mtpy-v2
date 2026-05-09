# Changelog

All notable changes to the mtpy-v2 fork are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning will follow [Semantic Versioning](https://semver.org/) once a
release is cut from this branch.

## [Unreleased] — distortion-as-signal refactor

This branch refactors the `decomposition` module into a package that
hosts distortion-analysis methods from multiple intellectual
traditions in MT, and lays the groundwork for adding
non-Groom-Bailey methods (Bahr, WAL, Lilley, García & Jones) under
the same API.

### Bootstrap CIs in compute_site_observables
- **Parametric bootstrap promoted from Phase-2 deferred to
  Phase-1 enabled**
  (:func:`compute_site_observables(bootstrap_n_replicates=N)`,
  :func:`compute_collection_observables(parallel=True)`,
  :data:`BOOTSTRAP_OBSERVABLES`). Each replicate adds independent
  Gaussian noise to ``mt_object.Z`` (σ per component =
  ``z.z_error / √2``) and re-runs the full per-site pipeline (GB
  + BCB + Lilley + Marti + Gomez-Treviño + PT + cross-method
  disagreement); per-replicate values are aggregated to per-band
  5 / 50 / 95 percentiles. Default ``bootstrap_n_replicates=0``
  preserves backward-compatible behaviour.

  Schema gains 30 CI columns —
  ``<obs>_p05``, ``<obs>_p50``, ``<obs>_p95`` for each of the 10
  primary observables in :data:`BOOTSTRAP_OBSERVABLES` (strike,
  twist, shear, |C-I|_F, gamma_magnitude,
  gamma_magnitude_periodwise, PT_alpha, PT_beta, PT_abs_beta,
  PT_ellipticity). Columns are ``NaN`` when the bootstrap was
  not run (``bootstrap_n_replicates=0``).

  ``discordance_significance`` is now a real signal-to-noise
  ratio (``mean(disc) / std(disc, ddof=1)`` across replicates),
  replacing the prior ``NaN`` placeholder.

  ``compute_collection_observables`` gains a ``parallel=False``
  kwarg that distributes the per-site pipeline across worker
  processes via :class:`multiprocessing.Pool`. Sites are reduced
  to raw-array payloads before being sent to workers (``Z``
  instances hold ``loguru`` references that don't pickle under
  some test runners). Per-site bootstrap RNG is seeded from a
  stable hash of the station id + the global seed, so sequential
  and parallel runs produce *bit-identical* tables.

  ``spatial_coherence.compute_coherence`` now consumes the
  ``<obs>_p05`` / ``<obs>_p95`` columns directly: the
  bootstrap-variance reference is the median half-width across
  rows. The inter-band-variance fallback warning fires only when
  the CI columns are absent (``bootstrap_n_replicates=0``); the
  documented Phase-1 transition is to run with bootstrap > 0 by
  default on production AusLAMP runs.

  Approximate AusLAMP-scale runtimes (1353 sites, 6 default
  bands, ``n_starts=5``, on a 16-core workstation):

  * ``bootstrap_n_replicates=0`` — ~ 5 minutes sequential.
  * ``bootstrap_n_replicates=50, parallel=True`` —
    ~ 1.5 hours.
  * ``bootstrap_n_replicates=50, parallel=False`` —
    ~ 19 hours (don't).

  Five new tests in
  ``tests/.../distortion/test_continental_observables.py``:
  the regression check at N=0; the CI-population check at N=20
  on a low-noise synthetic; the noise-widening check (aggregate
  spread grows with noise); the spatial-coherence
  warning-suppression check; and the parallel /
  sequential bit-identical equivalence check.

### Dependency removed
- **``pyarrow`` is no longer referenced** by ``ObservableTable``.
  ``to_parquet`` / ``from_parquet`` have been replaced with
  ``to_netcdf`` / ``from_netcdf`` (canonical xarray-backed I/O,
  matching :meth:`DecompositionResult.to_netcdf` elsewhere in
  the package); a one-way ``to_csv`` plus sidecar
  ``<path>.metadata.json`` is provided for human-readable
  export. ``netcdf4`` is already in the project closure
  (``mt_decomp`` requires it transitively via ``xarray``); no
  new dependency is added. The two tests that previously
  ``importorskip``-ed ``pyarrow`` are now first-class and run
  unconditionally.

### Changed
- **Restructured `decomposition.py` into a package** at
  `mtpy/core/transfer_function/z_analysis/decomposition/`. The
  monolithic 5400-line module is split into:
  - `__init__.py` — public API and re-exports for backward
    compatibility.
  - `common.py` — tensor algebra primitives, conversions, residual
    helpers, optimiser plumbing, period-band partitioning, and
    joint-input validation. Method-agnostic; reusable by future
    decomposition methods.
  - `results.py` — `DecompositionResult` and pickle / NetCDF
    serialisation.
  - `symmetries.py` — GB 90-degree symmetry handling and multi-start
    mode clustering / disambiguation.
  - `groom_bailey.py` — single-site GB and joint MJ algorithms
    plus parametric bootstrap.
- **Renamed `_canonicalise_solution` → `_geometric_fold`** to make
  room for sibling fold strategies. The `canonicalise=True/False`
  flag on `decompose()` is preserved as a backward-compatibility
  shim that maps to the new `disambiguation` argument.
- **Module docstrings** now describe each module's role within the
  multi-tradition package and cite the foundational papers.

### Added
- **Disambiguation strategy framework** (`symmetries.py`,
  `groom_bailey.py`). New `disambiguation` argument on
  `decompose()` accepts `"geometric"` (default, historical
  behaviour), `"identity"` (no fold; both branches visible),
  `"pt_aligned"` (choose the branch closest to the phase tensor's
  principal axis at the band's median period), `"min_shear"`
  (low-shear-preferring tiebreaker), or any
  `fold(strike, twist, shear) -> (strike, twist, shear)` callable.
  Mode clustering and the bootstrap path always use the geometric
  fold internally.
- **`DecompositionResult.alternate_branch()`** returns a fresh
  result on the GB-symmetry-equivalent branch
  (`strike + 90 mod 180`, `-shear`, with `Z_xy`/`Z_yx` swapped in
  `regional_z`). Idempotent; preserves chi-squared, RMS, errors,
  metadata, and the reconstructed C tensor.
- **Bibby-Caldwell-Brown decomposition**
  (`decompose_bibby`, `BibbyResult`). Fits a single real 2x2
  distortion matrix per period and band-averages it. Uses the
  diagonal-unity gauge to fix the BCB non-determinable scale.
  Methodologically the simplest distortion-analysis method in
  the package and a comparison baseline for the parameterised
  Groom-Bailey result.
- **McNeice-Jones (Phase 1)**
  (`decompose_mcneice_jones`, `JointDecompositionResult`). New
  API for multi-site joint GB decomposition. Takes parallel
  `z_objs` / `site_ids` lists, supports the disambiguation
  framework, and returns a purpose-built result with per-site
  distortion dicts and per-band shared strikes. Thin façade
  over the existing joint optimisation machinery in
  `groom_bailey.py` (no algorithm reimplementation). Phase 2
  will add validation against published results (BC87) and
  edge-case coverage.
- **Lilley Mohr-circle decomposition**
  (`decompose_lilley`, `LilleyResult`). Per-period parametric-
  free Mohr-circle decomposition: in-phase and quadrature
  centres / radii / rotation angles, the WAL-equivalent
  invariants (central impedance, anisotropy angle ``lambda``,
  per-part 3-D measure ``mu``), the noise-stability strike
  (Lilley 2018) computed via per-period bootstrap, and a
  per-period dimensionality classifier (``"1D"`` / ``"2D"`` /
  ``"3D-distorted"`` / ``"3D"``).
- **Marti WALDIM dimensionality**
  (`decompose_marti`, `MartiResult`, `wal_invariants`,
  `waldim_dimensionality`). The Weaver-Agarwal-Lilley (2000)
  seven rotational invariants ``I1`` ... ``I7`` plus the auxiliary
  ``Q``, the Marti et al. (2009) WALDIM classifier (cases 1, 2,
  3a, 4, 5, 7), Mohr-fold strikes ``St_3`` / ``St_4`` for 2-D
  periods, and the Bahr equal-phase strike ``St_5`` solved
  numerically. Cases 3c (diagonal-regional) and the full Smith
  (1995) twist / shear angles are flagged Phase-2.
- **Garcia-Jones extended decomposition**
  (`decompose_garcia_jones`, `GarciaJonesResult`). Phase 1
  implementation of the Garcia & Jones (2002) 3-D regional
  extension: per-site real distortion (twist, shear) plus a free
  3-D regional ``Z`` per period, fitted jointly across two-or-
  more sites via TRF nonlinear least-squares with multi-start.
  Gain and anisotropy fixed (the 2002 paper demonstrates these
  are non-identifiable). Reduces gracefully to a 2-D regional
  when the data warrant it; out-performs MJ on a genuinely 3-D
  regional synthetic (verified in tests).
- **Irreducible-decomposition utilities**
  (`distortion_geometry.py`: `irreducible_decomposition`,
  `gamma_field`, `gamma_magnitude`, `principal_axis`,
  `gamma_to_complex`, `complex_to_gamma`). Splits a real
  distortion ``D = C - I`` into its irreducible ``SO(2)``
  representations: spin-0 trace, spin-2 deviatoric shear
  (``gamma = gamma_1 + i gamma_2``, transforming as ``exp(2 i
  theta) gamma`` under rotation), and spin-0 antisymmetric
  pseudo-scalar. The spin-2 ``gamma`` field is the input to
  array-level E / B-mode analysis of distortion fields.
- **Synthetic test harness**
  (`tests/synthetics.py`). Categorical knobs for regional type
  (1-D / 2-D / 3-D), distortion strength, distortion shear, and
  noise level produce a ground-truth synthetic ``Z`` and the
  associated true ``C``. ``run_all_methods_on_synthetic`` runs
  every implemented method end-to-end;
  ``compute_method_accuracy`` returns the per-method Frobenius
  recovery accuracy. Used for cross-method validation.
- **CHANGELOG.md** (this file).

### Refactoring
Architectural cleanup of the decomposition package after the
multi-method landings, focused on shrinking ``groom_bailey.py``
back to GB-specific concerns and extracting the genuinely
method-agnostic plumbing into ``common.py``. ``groom_bailey.py``
shrunk from 3681 lines to 3337 lines (−344, ~9%) across these
moves; no behavioural changes — full decomposition test suite
remains 280 passing / 6 pre-existing netCDF failures / 2 slow-
opt-in skipped.

- **Method-agnostic plumbing moved to ``common.py``** (commit
  `fce7576`): ``_perturbed_initial_guess`` (bound-clipped Gaussian
  perturbation of an initial guess), ``_resample_residuals``
  (parametric bootstrap noise generator), and
  ``_compute_ci_percentile`` (NaN- and complex-aware percentile
  CI). New ``tests/.../distortion/test_common.py`` with 14 unit
  tests covering each function's contract.
- **Joint multi-site orchestration moved to ``common.py`` with
  factory injection** (commit `592ccaf`): ``_solve_band_joint``,
  ``_solve_band_joint_multistart``,
  ``_generate_starting_points_joint``, and
  ``_decompose_bands_with_modes_joint``. The orchestrators accept
  GB-specific helpers (``_objfun_joint``, ``_build_bounds_joint``,
  ``_canonical_initial_guess_joint``, ``_rotated_initial_guess_joint``)
  as required keyword-only factory callables; both
  ``groom_bailey.py`` and ``mcneice_jones.py`` consume the same
  generic orchestrator and pass GB's joint helpers as the
  factories. The smell that prompted the cleanup —
  ``mcneice_jones.py`` reaching into ``groom_bailey.py``'s
  ``_solve_band_joint_multistart`` (peer module's private
  orchestration) — is gone.
- **Test layout** (commit `0681f3a`): every per-method test file
  (and ``tests/synthetics.py``) relocated under
  ``tests/core/transfer_function/z_analysis/distortion/``,
  matching the source layout of the decomposition package.
  ``__init__.py`` files added at intermediate levels so the test
  modules are importable.

- **GB joint cost function and initial guesses moved to
  ``common.py``** (this commit): ``_canonical_initial_guess``,
  ``_rotated_initial_guess``, ``_canonical_initial_guess_joint``,
  ``_rotated_initial_guess_joint``, ``_build_bounds_joint``, and
  ``_objfun_joint`` (the GB joint cost function with analytic
  Jacobian) all relocated. Function bodies and docstrings
  unchanged. ``mcneice_jones.py`` no longer imports anything from
  ``groom_bailey.py``; both modules consume the GB joint primitives
  uniformly from ``common.py``. ``groom_bailey.py`` final size is
  2775 lines (down from 3681 at the start of the refactor — −906
  / −25 %).

### New pipelines
- **Dimensionality-trust stratification**
  (:mod:`...dimensionality_stratification`,
  :func:`stratify_table`, :func:`trust_score`,
  :func:`apply_trust_filter`, :func:`stratification_summary`,
  :func:`threshold_sensitivity`,
  :class:`StratificationSummary`,
  :data:`DEFAULT_TRUST_RULES`). Filters and stratifies the
  long-format observable table into four trust tiers based on
  whether the GB / MJ model assumption (2-D regional with
  galvanic distortion) is plausible at each (site, period_band):

  * ``"high_trust"`` (Tier A) — model ideal: WALDIM ∈ {2, 3},
    Lilley category ∈ {2D, 3D-2D}, ``|β_PT| < 3°``,
    ``magnetic_distortion_flag == "low_risk"``,
    ``GB_mode_warning is False``,
    ``cross_method_strike_disagreement_deg < 5°``,
    ``GB_rms_misfit < 2.0``.
  * ``"moderate_trust"`` (Tier B) — borderline: WALDIM ∈
    {2, 3, 4}, ``|β_PT| < 6°``,
    ``magnetic_distortion_flag ∈ {"low_risk", "moderate_risk"}``,
    ``cross_method_strike_disagreement_deg < 15°``.
  * ``"low_trust"`` (Tier C) — neither high nor moderate, but
    not excluded; results to be interpreted with care.
  * ``"excluded"`` (Tier D) — WALDIM == 1 (1-D, no strike) **or**
    WALDIM ∈ {6, 7} (true 3-D, model misfit) **or**
    ``magnetic_distortion_flag == "high_risk"`` **or**
    ``|β_PT| ≥ 6°`` **or**
    ``Lilley_category == "indeterminate"``. Excluded sites are
    *flagged*, not deleted from the source table.

  ``DEFAULT_TRUST_RULES`` is a parameterised dict; every
  threshold is overridable for sensitivity analysis.
  ``threshold_sensitivity`` sweeps a single threshold across a
  user-supplied range and reports the high-trust fraction and
  the mean of a chosen observable, producing the "how does the
  result move when I move the threshold?" curves Paper 1 needs.

  Companion :func:`trust_score` is the smooth-edged version of
  the high-trust criteria: each numeric criterion contributes a
  sigmoid score around its threshold (default width
  ``max(0.05 · |threshold|, 0.3)`` — sharp enough that a fully
  excluded row scores ``≤ 0.1``, smooth enough that
  :func:`trust_score` has no step discontinuities and so is
  suitable as a continental-map colour scale).

  The module docstring states explicitly that stratification is
  a *research choice* (not a measurement), that the default
  thresholds will be reported in Paper 1, that sensitivity
  analysis is mandatory, and that excluded sites are flagged
  rather than discarded. These caveats are load-bearing for any
  downstream claim about model trustworthiness.

  Eleven tests in
  ``tests/.../distortion/test_dimensionality_stratification.py``
  cover the six required scenarios — partition correctness,
  ``trust_score`` continuity, default rules vs categorical
  agreement at the high-trust ≥ 0.9 / excluded ≤ 0.1 endpoints,
  monotonic threshold-sensitivity sweep, no input-mutation
  guarantee, and the round-trip through
  :func:`compute_collection_observables` — plus three smoke
  tests for rule shape, missing-value handling, and the ≤ 5 s
  pipeline-budget on a 1000-row synthetic table. Stratification
  on the 1000-row table runs in < 1 s; the AusLAMP-scale
  (~6500-row) call is comfortably under the 5 s target.

- **Spatial-coherence pipeline**
  (:mod:`...spatial_coherence`,
  :func:`compute_coherence`, :func:`compute_coherence_all`,
  :func:`empirical_variogram`, :func:`randomisation_null`,
  :func:`pairwise_distances`, :class:`CoherenceResult`,
  :data:`PRIMARY_OBSERVABLES`). Quantifies whether each
  observable produced by the continental pipeline is
  *geographically structured* versus per-site noise — the
  empirical foundation for the Paper 1 "distortion is signal,
  not nuisance" claim. Per observable, per period band:

  1. Empirical haversine variogram
     ``γ(h) = (1/2) · mean[(z_i - z_j)²]`` over all site pairs at
     separation ``h`` (great-circle, kilometres). Default 20
     log-spaced bins from 50 to 2000 km plus a near-neighbour
     bin for separations below 50 km.
  2. Permutation null: shuffle observable values across sites
     (preserving coordinates and the pair-binning), recompute
     the variogram, repeat 100×, take the 5/50/95 percentiles
     per bin as the "no spatial structure" envelope.
  3. Geostat summary: nugget (smallest-h variance), sill
     (large-h plateau), range (half-sill crossing in km), and
     the nugget-to-sill ratio.
  4. Coherence label: ``"structured"`` if ≥ 50 % of bins exceed
     the null 95th percentile and ``nugget / sill < 0.4``;
     ``"weakly_structured"`` for partial agreement;
     ``"noise_dominated"`` when the variogram cannot be
     distinguished from the shuffled null or
     ``nugget / sill ≥ 0.9``; ``"insufficient_data"`` for too
     few finite values.

  Per-observable handling: magnitudes (``gamma_magnitude``,
  ``C_minus_I_F``, ``GB_rms_misfit``, …) use ``log10`` before
  variogramming; line-direction angles (``C_strike_deg``,
  ``gamma_principal_axis_deg``, ``PT_alpha_deg``) use the
  circular metric ``1 - cos(2 Δθ)`` so a 5° / 175° pair is
  ``≈ 10°`` apart, not ``≈ 170°``; ordinal flags
  (``magnetic_distortion_flag``, ``Lilley_category``) are mapped
  to integer codes per :data:`ORDINAL_OBSERVABLES`.

  Bootstrap-variance reference: pulled from a
  ``f"{observable_name}_bootstrap_var"`` column when the input
  table contains it (a Phase-2 follow-up on
  :func:`compute_site_observables`); otherwise the median
  inter-band variance per site, with a clear warning about the
  upper-bound nature of the fallback.

  Visualisation lives in a separate
  :mod:`...spatial_coherence_plots` module so the core API
  has no matplotlib dependency. :func:`plot_variogram` produces
  a publication-style γ(h) plot with the null 5-95 % envelope,
  bootstrap-variance reference line, and an annotated nugget /
  sill / range box; :func:`plot_coherence_summary` tiles
  variograms across all primary observables for a one-figure
  array-level summary.

  Eleven tests in
  ``tests/.../distortion/test_spatial_coherence.py`` cover the
  six required scenarios (pure noise → ``noise_dominated``;
  smooth Gaussian random field → ``structured``; mixed
  signal + noise → ``structured`` / ``weakly_structured`` with
  positive nugget; circular variogram for line directions
  including the 0/180° wrap; randomisation-null bracket
  asymmetry; AusLAMP-scale 1353-site recovery of a 500-km
  correlation length within 100 km in <60 s) plus five smoke
  tests for the distance / classification helpers. The
  AusLAMP-scale test auto-skips when no
  ``site_summary.csv`` is found; on this machine it ran in 2 s.

- **Continental-scale observables pipeline**
  (:mod:`...continental_observables`,
  :func:`compute_site_observables`,
  :func:`compute_collection_observables`,
  :class:`SiteObservables`, :class:`ObservableTable`,
  :data:`OBSERVABLE_COLUMNS`). The canonical "compute everything
  for an MT collection" entry point for the Paper 1 empirical
  foundation. Composes the existing per-method modules (no
  decomposition logic in the pipeline itself) into a tidy
  long-format pandas DataFrame with one row per (site,
  period_band) and a fixed schema covering distortion-tensor
  primary observables, irreducible (spin-2) decomposition,
  phase-tensor invariants, cross-method consistency,
  dimensionality (WALDIM + Lilley 2020), magnetic-distortion
  flag, Tipper amplitude, and the Paper 1 hero observable
  ``discordance_deg`` (the angle between the recovered C
  principal axis and the phase-tensor alpha, range
  ``[0°, 90°]``).

  Default :data:`DEFAULT_PERIOD_BANDS` are six 1-decade-wide
  bands tiled across ``[0.01, 10000]`` s (the AusLAMP design
  range); pass ``period_bands=`` to override. Aggregation rules
  are documented per observable type: circular weighted mean
  (mod 180°) for line-direction quantities, geometric mean for
  positive magnitudes, arithmetic mean for misfits, "worst-case"
  promotion for dimensionality labels, and integer-mode for
  WALDIM cases.

  :class:`ObservableTable` round-trips through parquet
  (``to_parquet`` / ``from_parquet``) with a sidecar JSON for
  the provenance metadata (mtpy version, decomposition module
  git sha, timestamp, method versions, period-band spec,
  ``canonical_gauge`` choice, RNG seed, input-collection hash).
  Parquet I/O lazy-loads ``pyarrow``; if neither ``pyarrow`` nor
  ``fastparquet`` is installed, the I/O methods raise
  :class:`ImportError` with installation guidance. The package
  does **not** declare ``pyarrow`` as a hard dependency — it is
  optional and only required by the parquet path.

  Five integration tests in
  ``tests/.../distortion/test_continental_observables.py`` cover
  the schema-completeness, multi-site assembly, period-band
  aggregation, provenance / reproducibility, and discordance
  scenarios; two further tests (parquet round-trip, ≤ 10 s
  pipeline budget) auto-skip when no parquet engine is
  installed. With three sites at ``n_starts=3`` the full
  pipeline runs in ~1 s end-to-end.

### Unified dimensionality classifier
- **Lilley 2020 unified phase-tensor / Bahr-eigenvector / Mohr-
  circle dimensionality classifier**
  (:mod:`...lilley_dimensionality`,
  :func:`classify_dimensionality`,
  :func:`compare_lilley_marti`,
  :class:`LilleyDimensionalityResult`). Lilley 2020
  (*Exploration Geophysics* 51:4, 401-421) establishes the
  formal equivalence of three apparently-distinct dimensionality
  tools — the Caldwell-Bibby-Brown phase tensor invariants
  (``alpha``, ``beta``, eigenvalues), Bahr 1988 strike directions
  (eigenvectors of ``Phi``), and the Mohr-circle representation
  of the phase tensor — and this module makes that equivalence
  operational. Per-period classification rules:

  * ``"1D"``: ``|beta| < 1°`` and ``ellipticity < 0.05``.
  * ``"2D"``: small skew with eigenvectors close to perpendicular.
  * ``"3D-2D"``: small skew but eigenvectors deviate from
    perpendicular (Lilley's "approximately 2-D" sub-case).
  * ``"3D"``: ``|beta| >= 3°``.

  Thresholds configurable via keyword arguments and exposed as
  module constants (``BETA_1D_THRESHOLD_DEG``,
  ``BETA_2D_THRESHOLD_DEG``, etc.).

  Companion :func:`compare_lilley_marti` cross-checks against the
  Marti / WALDIM classifier with a documented mapping (WALDIM
  3-D / 2-D sub-cases 3, 4, 6, 7 → Lilley ``"2D"``, since galvanic
  distortion is gauge-invisible to the phase tensor by
  construction). 10 tests in
  ``tests/.../distortion/test_lilley_dimensionality.py`` cover the
  seven required scenarios — 1-D / 2-D-strike / 2-D-rotated /
  2-D-galvanic / 3-D classifications, the Lilley 2020 formal
  identities (``alpha`` = major-eigenvector strike,
  ``Mohr radius = (lambda_max − lambda_min)/2``,
  ``Mohr mu = 2 * beta``), and 5/5 Lilley-vs-Marti agreement on
  canonical synthetics.

### Cross-method consolidation
- **Cross-method comparison entry point**
  (:mod:`...cross_method`,
  :func:`compute_cross_method`,
  :func:`compute_cross_method_collection`,
  :func:`agreement_summary`,
  :class:`CrossMethodResult`). Runs every implemented
  decomposition method on the same site and projects each
  method's native output onto a common comparison space
  (per-period strike, twist / shear, regional ``Z``,
  dimensionality). Methods that fail or that don't produce a
  particular output field record the failure in
  ``method_status`` / ``method_messages`` rather than raising;
  per-field dicts simply omit absent methods.

  Adapter design is local: each method has a private
  ``_adapter_<name>`` function that takes ``(z_object,
  periods, **kwargs)`` and returns a comparison-space dict.
  Adding a new method means adding an adapter and registering
  the name in ``ALL_METHODS``. Single-site GJ correctly
  reports ``no_solution`` (it requires >= 2 sites); other six
  methods complete unaffected. The companion
  :func:`agreement_summary` returns per-method RMS deltas vs a
  reference (default ``"groom_bailey"``) for strike (circular
  mod 90°), twist / shear (linear), and regional ``Z`` (log10
  fractional in amplitude, radians in phase). 16 unit tests in
  ``tests/.../distortion/test_cross_method.py`` cover the four
  required scenarios plus subset / unknown-method handling.

  This is the canonical entry point for the "three traditions
  converge" empirical claim of Paper 1: GB / MJ / BCB / GJ all
  agreeing within a tolerance is direct cross-tradition
  evidence the recovery is correct, and disagreement is a
  diagnostic flag (cross-referenced with the F4
  ``canonical_gauge`` and the ``magnetic_distortion_diagnostic``
  module).

### New diagnostics
- **Magnetic-galvanic-distortion heuristic flag**
  (:mod:`...magnetic_distortion_diagnostic`,
  :func:`compute_magnetic_distortion_flag`,
  :class:`MagneticDistortionFlag`). Flags sites where the standard
  MT decomposition's magnetic-distortion-negligible assumption may
  break down (Garcia, Boerner & Pedersen 2003; Chave & Smith 1994).
  Three sub-diagnostics combine into a single per-site flag
  (``low_risk`` / ``moderate_risk`` / ``high_risk`` /
  ``indeterminate``):

  1. ``tipper_diagnostic`` — anomalous Tipper magnitude / strong
     band-to-band frequency dependence.
  2. ``frequency_dependence_diagnostic`` — coefficient of
     variation of the recovered ``C`` tensor across bands above
     threshold.
  3. ``method_inconsistency_diagnostic`` — strike or |Z_TE|
     disagreement between methods (e.g. GB vs GJ).

  Combination rule: 2 or more diagnostics flag, *or* peak Tipper
  amplitude exceeds the strong-override threshold (default 0.5),
  yields ``high_risk``; exactly one diagnostic flagging yields
  ``moderate_risk``; zero yields ``low_risk``; insufficient input
  data yields ``indeterminate``. Thresholds are configurable via
  keyword arguments and exposed as module-level constants for
  researcher tuning.

  **Heuristic, not corrective.** A flag does not prove magnetic
  distortion is present; it indicates the assumption may be
  violated. Intended use: exclude or annotate flagged sites in
  continental aggregation. Quantitative magnetic-distortion
  handling (Bayesian inversion with explicit ``Q_h``, ``Q_z``
  priors) is Paper 6 territory and a future PR. The
  heuristic-not-corrective scope is documented in three places:
  the module docstring, the ``MagneticDistortionFlag`` dataclass
  docstring, and this entry.

### New (exploratory) modules
Modules added for *exploration* — algebra and rotational-invariance
properties verified by unit tests, but not benchmarked against
forward-modelled 3-D synthetic data or real-data ground truth.
Treat outputs as diagnostic, not as production-quality apparent-
resistivity / decomposition estimates, until a follow-up
benchmarking PR.

- **Gomez-Treviño 2018 rotational-invariant TE / TM resistivities**
  (:mod:`...gomez_trevino`,
  :func:`decompose_gomez_trevino`,
  :class:`GomezTrevinoResult`). Constructs the series and parallel
  resistivities ``rho_s = trace(Z^T Z) / (2 omega mu_0)``,
  ``rho_p = 2 / (omega mu_0 trace(Y^T Y))`` and pulls the invariant
  TE / TM analogues ``rho_+`` / ``rho_-`` out of the symmetric
  quadratic ``lambda² − 2 rho_s lambda + rho_s rho_p = 0``. Also
  exposes the iterative ``(arithmetic, harmonic) mean`` chain whose
  geometric-mean limit is the determinant resistivity (Gomez-
  Treviño 2018, fig. 1). 19 unit tests in
  ``tests/.../distortion/test_gomez_trevino.py`` lock in the 1-D
  reduction (``rho_s = rho_p``, ``rho_+ = rho_- = rho_d``), the
  explicit 2-D reduction to TE / TM, rotational invariance under
  measurement-axis rotation (with and without galvanic distortion),
  the chain's convergence to ``rho_d``, and the geometric-mean
  invariant ``rho_+ · rho_-`` preserved at every chain step. The
  framework's general 3-D interpretation — and the assignment of
  ``+`` / ``−`` to TE vs TM — is *not* validated and is left to a
  benchmarking follow-up. Marked exploratory in three places:
  module docstring, dataclass docstring, this entry.

### Tests
- All decomposition test suites pass: GB single-site, MJ joint,
  BCB, Lilley, Marti, Garcia-Jones, disambiguation, alternate-
  branch, distortion-geometry, synthetic harness, and the new
  end-to-end integration tests. Two opt-in slow validations
  (BC87 and the integration tests) skip cleanly under
  ``pytest -m "not slow"``. The 6 `TestDecompositionResultNetcdf`
  failures pre-date this branch (a netCDF4 library limitation
  around boolean attributes) and are unrelated.
- New test files: `tests/test_disambiguation.py`,
  `tests/test_alternate_branch.py`, `tests/test_bibby.py`,
  `tests/test_mcneice_jones.py`,
  `tests/test_mj_bc87_validation.py`,
  `tests/test_lilley.py`, `tests/test_marti.py`,
  `tests/test_garcia_jones.py`,
  `tests/test_distortion_geometry.py`,
  `tests/test_synthetics_harness.py`,
  `tests/test_integration.py`.
- New test utility module: `tests/synthetics.py`
  (ground-truth synthetic generation, per-method accuracy).

### Documentation
- `docs/decomposition_validation.md` records the validation
  strategy for each method and what is and is not validated
  against published results.
- `docs/distortion_methods.md` (new) provides a tradition-by-
  tradition overview, when-to-use-which guidance, and known
  limitations for every implemented method.
- The package `__init__.py` docstring carries an at-a-glance
  method-selection guide for callers.

### Planned
- Implement Bahr (1991) decomposition and class scheme.
- Adopt a shared `DecompositionResult` flavour across every method
  so cross-tradition comparison is a one-liner.
- Resolve the pre-existing `TestDecompositionResultNetcdf` failures
  by switching boolean metadata attributes to `int8` at the NetCDF
  boundary.
- Garcia-Jones Phase 2: gain / anisotropy recovery experiments
  (the 2002 paper notes both are unstable; verify on synthetic),
  magnetic-distortion extension (Garcia, Boerner & Pedersen 2003).
- Marti Phase 2: case 3c (diagonal regional) discriminator via
  the Bahr ``xi_4`` / ``eta_4`` test, full Smith (1995) twist /
  shear angle decomposition.
- Investigate the BC87 LIT-line strike discrepancy
  (`test_mj_bc87_strike_recovery` finds ~5° in the geometric-fold
  convention vs Fig 12's ~25–40°). Likely follow-ups: per-period
  narrow-band fits, higher `n_starts`, side-by-side run against
  the original McNeice-Jones Fortran on the same EDIs.
