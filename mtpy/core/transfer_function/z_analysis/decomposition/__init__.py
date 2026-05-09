"""Magnetotelluric distortion analysis: a multi-tradition package.

The MT community has produced several independent intellectual
traditions for separating the regional 2-D inductive response from
local galvanic distortion. They differ in parameterisation, in the
rotational invariants they emphasise, and in how they classify
dimensionality. This package collects implementations of these
methods in a single, unified API so that a user can run any of them
on the same impedance tensor and compare results without leaving
mtpy-v2.

Traditions (foundational citations below):

- **Groom-Bailey / McNeice-Jones (GB / MJ)** — factorise the 2x2
  distortion matrix into four named geometric parameters
  (``gain``, ``twist``, ``shear``, ``anisotropy``) plus a regional
  strike, fit jointly across multiple periods at one site (GB) or
  across multiple sites with a shared strike (MJ).
- **Bahr / WAL / Marti** — rotational-invariant decomposition and
  dimensionality classifiers (Bahr's "skew" and class scheme;
  Weaver-Agarwal-Lilley invariants ``I1`` ... ``I7``; Marti's
  WALDIM dimensionality codes).
- **Lilley** — Mohr-circle representation of the impedance tensor
  with a graphical decomposition into amplitude, phase, and strike.
- **García & Jones** — explicitly 3-D extension of GB-style
  decomposition.

Currently implemented
---------------------
- **GB single-site** (:func:`decompose`): per-band TRF least-squares
  with multi-start optimisation, parametric bootstrap, and a
  user-selectable strategy for resolving the GB 90-degree symmetry.
- **MJ joint** (:func:`decompose_joint` and the newer
  :func:`decompose_mcneice_jones`): the multi-site extension with
  a shared regional strike across stations.
  ``decompose_mcneice_jones`` is the cleaner Phase-1 API that
  returns a :class:`JointDecompositionResult`; both functions
  share the same underlying optimisation machinery.
- **BCB single-site** (:func:`decompose_bibby`): Bibby-Caldwell-
  Brown decomposition. Fits a single real 2x2 distortion matrix
  ``C`` per period and band-averages it; the methodologically
  simplest comparison point for the parameterised GB result.
- **Disambiguation framework** (:mod:`.symmetries`): four named
  fold strategies (``geometric``, ``identity``, ``pt_aligned``,
  ``min_shear``) and a callable hook for user-defined strategies,
  exposed on :func:`decompose`.
- **Lilley Mohr-circle** (:func:`decompose_lilley`): per-period
  Mohr-circle parametric-free decomposition with WAL invariants and
  noise-stability strike.
- **Marti WALDIM** (:func:`decompose_marti`,
  :func:`wal_invariants`, :func:`waldim_dimensionality`): per-period
  dimensionality classifier on the WAL invariants.
- **Garcia-Jones extended** (:func:`decompose_garcia_jones`): the
  3-D regional extension of MJ, Phase 1. Per-site real distortion
  (twist, shear) plus a free 3-D regional Z per period, fitted
  jointly across two-or-more sites.
- **Irreducible decomposition utilities**
  (:mod:`.distortion_geometry`): split a 2x2 real distortion
  ``C - I`` into its irreducible ``SO(2)`` parts (trace, spin-2
  deviatoric shear, antisymmetric pseudo-scalar). The spin-2
  ``gamma`` field is the input to array-level E / B-mode analysis.

Method-selection guide
----------------------
At-a-glance pointers; see ``docs/distortion_methods.md`` for the
full discussion and tradeoffs.

* **Single site, 2-D regional, want C and a regional Z** ->
  :func:`decompose`. Pick a disambiguation strategy that matches
  your prior knowledge (``"geometric"`` is the historical GB
  default; ``"pt_aligned"`` aligns to the phase tensor).
* **Multiple sites, shared 2-D regional** ->
  :func:`decompose_mcneice_jones`. Per-site distortion plus a
  shared per-band strike. The Phase 1 API
  (:class:`JointDecompositionResult`) is the cleaner public
  surface; :func:`decompose_joint` is retained for backward
  compatibility.
* **Sanity check on GB without any GB parametrisation** ->
  :func:`decompose_bibby`. Fits a single real ``C`` per period
  with the diagonal-unity gauge.
* **Dimensionality classification** -> :func:`decompose_marti`
  for the Marti 2009 WALDIM codes; :func:`waldim_dimensionality`
  is the per-period classifier and :func:`wal_invariants` exposes
  the underlying WAL invariants.
* **Mohr-circle invariants and noise-stability strike** ->
  :func:`decompose_lilley`. Gives the per-period Mohr-circle
  centres / radii / angles plus the bootstrap-based strike
  standard deviation.
* **Multiple sites, genuinely 3-D regional** ->
  :func:`decompose_garcia_jones`. Per-site distortion plus a free
  3-D regional Z per period; out-performs MJ on a 3-D regional
  synthetic.
* **Spin-2 / E-mode analysis of a distortion field** ->
  :func:`gamma_field`, :func:`principal_axis`, or
  :func:`irreducible_decomposition` on each site's recovered
  ``C``. The spin-2 shear ``gamma = gamma_1 + i gamma_2`` is the
  station-level input to array-level E / B-mode decompositions.

Planned
-------
- Bahr 1991 decomposition and class scheme.
- A common :class:`DecompositionResult` flavour shared across all
  methods so cross-tradition comparison is a one-liner.

Package layout
--------------
- :mod:`.common` : math primitives, residual normalisations, band
  partitioning, and joint-input validation that any decomposition
  method in this package can reuse.
- :mod:`.results` : :class:`DecompositionResult` and the pickle /
  NetCDF serialisation helpers.
- :mod:`.symmetries` : GB-symmetry fold strategies and mode
  clustering / disambiguation helpers used by the multi-start
  optimiser.
- :mod:`.groom_bailey` : the GB single-site and MJ joint
  algorithms, public ``decompose*`` entry points, and the
  parametric-bootstrap machinery.
- :mod:`.bibby` : the Bibby-Caldwell-Brown single-site
  decomposition (:func:`decompose_bibby`).
- :mod:`.lilley` : the Lilley Mohr-circle decomposition.
- :mod:`.marti` : the Marti WALDIM dimensionality classifier.
- :mod:`.garcia_jones` : the Garcia-Jones (2002) 3-D-regional
  extended decomposition.
- :mod:`.distortion_geometry` : irreducible-decomposition utilities
  for the spin-0 / spin-2 split of a real distortion tensor.

See also
--------
:mod:`mtpy.core.transfer_function.z_analysis.distortion` :
    Bibby (2005) frequency-averaged single-distortion-matrix
    decomposition. A simpler, complementary method.
:mod:`mtpy.core.transfer_function.pt` :
    Phase tensor (Caldwell-Bibby-Brown 2004); distortion-invariant
    representation that is the natural reference for
    cross-tradition comparison.

References
----------
Bahr, K. (1991). Geological noise in magnetotelluric data: a
classification of distortion types. Physics of the Earth and
Planetary Interiors, 66, 24-38.

Bibby, H. M., Caldwell, T. G., & Brown, C. (2005). Determinable and
non-determinable parameters of galvanic distortion in magnetotellurics.
Geophysical Journal International, 163(3), 915-930.

Caldwell, T. G., Bibby, H. M., & Brown, C. (2004). The magnetotelluric
phase tensor. Geophysical Journal International, 158, 457-469.

García, X., & Jones, A. G. (2002). Decomposition of three-dimensional
magnetotelluric data. In Three-Dimensional Electromagnetics
(M. S. Zhdanov & P. E. Wannamaker, eds.), Methods in Geochemistry and
Geophysics, 35, 235-250.

Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.

Lilley, F. E. M. (1998). Magnetotelluric tensor decomposition: Parts I
and II. Geophysics, 63(6), 1885-1907.

Marti, A., Queralt, P., & Ledo, J. (2009). WALDIM: A code for the
dimensionality analysis of magnetotelluric data using the rotational
invariants of the magnetotelluric tensor. Computers & Geosciences,
35, 2295-2303.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
tensor decomposition of magnetotelluric data. Geophysics, 66(1),
158-173.

Weaver, J. T., Agarwal, A. K., & Lilley, F. E. M. (2000).
Characterization of the magnetotelluric tensor in terms of its
invariants. Geophysical Journal International, 141, 321-336.

Notes
-----
This package ships pure-Python implementations of the GB and MJ
algorithms. Validation has been performed against historical Fortran
implementations (Strike, McNeice-Jones); see the project's
documentation for tolerance specifications and the optimiser-gap
analysis.

Interpretation guide
====================
Cross-cutting items that affect how observables produced by this
package should be read. Each paragraph links to the relevant
module's "Caveats" section for the rigorous detail; this is the
package-level summary that ``help(decomposition)`` surfaces.

* **gamma_magnitude vs gamma_magnitude_periodwise.** Two ``|γ|``
  columns are emitted by :func:`compute_collection_observables`
  for each (site, band): ``gamma_magnitude`` is the spin-2
  magnitude of the band-aggregate ``C`` tensor (internally
  consistent with the strike / twist / shear in the same row);
  ``gamma_magnitude_periodwise`` is the geometric mean of
  per-period ``|γ|`` values within the band (robust to outlier
  periods, decoupled from the angular-aggregation choice). Both
  are valid; they answer different questions. Use both for
  sensitivity analysis. See
  :mod:`...continental_observables` for the rigorous
  side-by-side definitions.

* **cross_method_*_disagreement_deg semantics (post-F4).** The
  three columns
  ``cross_method_strike_disagreement_deg``,
  ``cross_method_twist_disagreement_deg``,
  ``cross_method_shear_disagreement_deg`` are per-period-pair
  RMS aggregated to per-band (post-F4 — pre-F4 they were
  per-band-median-of-medians). A value of 5° means **method
  outputs typically differ by 5° at this site**; treat
  continental aggregations stratified on this column with care
  above ~10° (model invalidity dominates). The columns are
  **per-band uncertainty estimates**, not central tendencies —
  variograms on them describe the spatial scale of *model
  trustworthiness variation*, not any underlying physical
  signal (see :mod:`...spatial_coherence`).

* **bootstrap_variance interpretation.** Two paths feed the
  ``bootstrap_variance`` reference reported by
  :func:`compute_coherence`:
  the bootstrap-driven path (post-F6) reads ``<obs>_p05`` /
  ``<obs>_p95`` columns directly; the inter-band fallback path
  uses cross-band variance per site as an upper bound. The
  fallback over-estimates noise (it absorbs genuine
  frequency-dependence) — treat it as an upper bound, not a
  tight estimate. The Phase-1 transition is to default to
  ``compute_site_observables(bootstrap_n_replicates=50)`` on
  production AusLAMP runs; the warning then does not fire.

* **Period bands tile by default.** ``band_overlap_fraction=0.0``
  is the default; the six AusLAMP-design-range bands span
  ``[0.01, 10000]`` s with no overlap. With ``> 0`` overlap each
  per-period observable contributes to multiple bands and the
  long-format table has correlated rows. **Continental papers
  must report which** ``n_bands`` and ``band_overlap_fraction``
  were used — both travel in the table's ``metadata`` dict.

* **dimensionality_concordant=False on clean 2-D-distorted
  sites is expected**, not a data quality flag. Galvanic
  distortion is gauge-invisible to the phase tensor (Caldwell
  et al. 2004), so a 2-D + galvanic site is ``"2D"`` to Lilley
  but a 3-D-flavoured case (3, 4, 6, 7) in Marti's WALDIM. The
  resulting concordance flag reads ``False`` even on physically
  clean sites; this is by physics, not data error.

* **trust_score post-F7 is min-of-criteria, floor 0.05.** A row
  is only as trustworthy as its weakest criterion (replacing
  the pre-F7 geometric mean). Default sigmoid scale
  ``max(0.1·|threshold|, 0.5)``. A score of 0.5 corresponds to
  the weakest criterion at exactly its rule threshold;
  ``apply_trust_filter(min_trust=0.5)`` is the canonical Paper-1
  filter. Numerical values are not directly comparable to
  pre-F7 outputs.

* **Joint MJ failure modes surfaced in metadata (F9).** The
  joint :func:`...mcneice_jones.decompose_mcneice_jones` requires
  identical frequency grids across sites; mixed-grid AusLAMP
  collections (EDL log-base-10 + LEMI power-of-2) fail the
  joint validator. ``MJ_rms_misfit`` is then ``NaN`` for every
  row, *and*
  :func:`compute_collection_observables` emits a
  :class:`UserWarning` and records the failure mode in
  ``metadata["joint_mj_status"]``
  (``"success"`` / ``"single_site"`` /
  ``"frequency_grid_mismatch"`` / ``"convergence_failed"`` /
  ``"other_error"``) plus
  ``metadata["joint_mj_failure_message"]`` for the failure
  cases. Always check
  ``metadata["joint_mj_status"] == "success"`` before reading
  the MJ column — a ``NaN`` column with status
  ``"frequency_grid_mismatch"`` is expected on mixed-grid data,
  not a bug in the pipeline.

* **AusLAMP-scale runtime budgets** (1353 sites, 6 default
  bands, ``n_starts=5``, 16-core workstation; order-of-magnitude
  only):

  * Default pipeline (``bootstrap_n_replicates=0``): ~5 min
    sequential.
  * Bootstrap ``n=50, parallel=True``: ~1.5 hours.
  * Bootstrap ``n=50, parallel=False``: ~19 hours (don't).
  * :func:`compute_coherence_all` on default observables:
    ~60 s (variograms + 100-shuffle nulls).
  * Stratification (:func:`stratify_table` +
    :func:`stratification_summary`): ~5 s.

* **Reproducibility.** Every :func:`compute_collection_observables`
  call records ``mtpy_version``, ``decomposition_git_sha``,
  ``timestamp_utc``, ``rng_seed``, and every kwarg in the
  table's ``metadata``. Two same-seed runs with the same input
  hash are bit-identical *except for the timestamp*. Parallel
  and sequential runs at the same seed are also bit-identical
  (per-site bootstrap RNG is seeded from a stable hash of
  ``station_id`` + ``base_seed``).

References
==========
Foundational citations for the decomposition methods, the phase
tensor, and the dimensionality classifiers used by this package.
The per-method module docstrings carry the full bibliographic
detail; this is the consolidated package-level reference list.

* **Bibby, Caldwell & Brown 2005 (GJI, 163, 915–930)** —
  determinable and non-determinable parameters of galvanic
  distortion in MT (BCB).
* **Caldwell, Bibby & Brown 2004 (GJI, 158, 457–469)** —
  the magnetotelluric phase tensor.
* **Garcia & Jones 2002 (Methods Geochem. Geophys., 35,
  235–250)** — 3-D regional MT decomposition (GJ).
* **Garcia, Boerner & Pedersen 2003 (GJI, 154, 957–969)** —
  electric and magnetic galvanic distortion in tensor CSAMT
  (heuristic flag in :mod:`...magnetic_distortion_diagnostic`).
* **Gomez-Treviño, Esparza & Romo 2018 (Earth, Planets and
  Space, 70:35)** — invariant TE / TM resistivities. *Mark
  exploratory*; see :mod:`...gomez_trevino` for the
  field-validation caveat.
* **Groom & Bailey 1989 (JGR, 94(B2), 1913–1925)** — original
  parameterised distortion decomposition (GB).
* **Lilley 1976/1993/1998/2018 (Geophysics / Exploration
  Geophysics)** — Mohr-circle MT representation; 2018 paper
  introduces the seven-invariant set.
* **Lilley 2020 (Exploration Geophysics, 51(4), 401–421)** —
  formal equivalence of the CBB phase tensor with Bahr's 1988
  strike analysis (the basis for
  :mod:`...lilley_dimensionality`).
* **Marti, Queralt & Ledo 2009 (Computers & Geosciences, 35,
  2295–2303)** — WALDIM dimensionality classifier; the
  Weaver-Agarwal-Lilley 2000 invariants are the algebraic
  foundation.
* **McNeice & Jones 2001 (Geophysics, 66(1), 158–173)** —
  multi-site joint GB decomposition (MJ).

Notes
-----
The :func:`compute_collection_observables` pipeline is
composition-only — every observable in the long-format table is
computed by one of the per-method modules above; this package
adds the band-level aggregation, the cross-method consolidation,
the bootstrap CIs, and the spatial-coherence / trust-tier
analyses on top.
"""

from .common import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    _BandResult,
    _band_arrays_to_z,
    _build_bounds_joint,
    _calc_error,
    _canonical_initial_guess,
    _compute_ci_percentile,
    _convz2p,
    _convz2r,
    _estim_imp,
    _extract_bands,
    _extreme,
    _jkvar,
    _mat_multiply,
    _normalise_collection_input,
    _objfun_joint,
    _perturbed_initial_guess,
    _resample_residuals,
    _rotated_initial_guess,
    _unpack_x,
    _unpack_x_joint,
    _validate_joint_input,
    _z_to_band_arrays,
)
from .groom_bailey import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    _bootstrap_decompose,
    _build_bounds,
    _generate_starting_points,
    _objfun,
    _predict_z_from_primary_modes,
    _solve_band,
    _solve_band_multistart,
    decompose,
    decompose_each_station,
    decompose_joint,
)
from .bibby import decompose_bibby
from .continental_observables import (
    DEFAULT_PERIOD_BANDS,
    OBSERVABLE_COLUMNS,
    OBSERVABLE_DTYPES,
    compute_collection_observables,
    compute_site_observables,
    default_period_bands,
)
from .cross_method import (
    ALL_METHODS,
    DEFAULT_METHODS,
    METHOD_CAPABILITIES,
    agreement_summary,
    compute_cross_method,
    compute_cross_method_collection,
)
from .dimensionality_stratification import (
    DEFAULT_SUMMARY_OBSERVABLES,
    DEFAULT_TRUST_RULES,
    STRATA_ORDER,
    apply_trust_filter,
    stratification_summary,
    stratify_table,
    threshold_sensitivity,
    trust_score,
)
from .distortion_geometry import (
    complex_to_gamma,
    gamma_field,
    gamma_magnitude,
    gamma_to_complex,
    irreducible_decomposition,
    principal_axis,
)
from .garcia_jones import decompose_garcia_jones
from .gomez_trevino import (
    decompose_gomez_trevino,
    determinant_resistivity,
    invariant_resistivities,
    iterative_chain,
    series_parallel_resistivities,
)
from .lilley import decompose_lilley
from .lilley_dimensionality import (
    classify_dimensionality,
    compare_lilley_marti,
    eigenvector_strike,
    mohr_circle_phase_tensor,
    phase_tensor,
    phase_tensor_invariants,
)
from .magnetic_distortion_diagnostic import (
    compute_magnetic_distortion_flag,
    frequency_dependence_diagnostic,
    method_inconsistency_diagnostic,
    tipper_diagnostic,
)
from .marti import decompose_marti, wal_invariants, waldim_dimensionality
from .mcneice_jones import decompose_mcneice_jones
from .results import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    BibbyResult,
    CoherenceResult,
    CrossMethodResult,
    DecompositionResult,
    GarciaJonesResult,
    GomezTrevinoResult,
    JointDecompositionResult,
    LilleyDimensionalityResult,
    LilleyResult,
    MagneticDistortionFlag,
    MartiResult,
    ObservableTable,
    SiteObservables,
    StratificationSummary,
    _desanitize_station_id,
    _sanitize_station_id,
)
from .spatial_coherence import (
    ANGULAR_LINE_OBSERVABLES,
    DEFAULT_BIN_EDGES_KM,
    MAGNITUDE_OBSERVABLES,
    ORDINAL_OBSERVABLES,
    PRIMARY_OBSERVABLES,
    compute_coherence,
    compute_coherence_all,
    default_bin_edges_km,
    empirical_variogram,
    haversine_distances_km,
    pairwise_distances,
    randomisation_null,
)
from .symmetries import (  # noqa: F401  -- private helpers re-exported for backward compatibility
    _DEFAULT_MODE_TOLERANCE,
    _cluster_modes,
    _compute_mode_probabilities,
    _detect_band_disagreement,
    _detect_primary_mode_warning,
    _geometric_fold,
    _identity_fold,
    _min_shear_fold,
    _pt_aligned_fold,
    _resolve_disambiguation,
)

__all__ = [
    "ALL_METHODS",
    "ANGULAR_LINE_OBSERVABLES",
    "BibbyResult",
    "CoherenceResult",
    "CrossMethodResult",
    "DEFAULT_BIN_EDGES_KM",
    "DEFAULT_METHODS",
    "DEFAULT_PERIOD_BANDS",
    "DEFAULT_SUMMARY_OBSERVABLES",
    "DEFAULT_TRUST_RULES",
    "DecompositionResult",
    "GarciaJonesResult",
    "GomezTrevinoResult",
    "JointDecompositionResult",
    "LilleyDimensionalityResult",
    "LilleyResult",
    "MAGNITUDE_OBSERVABLES",
    "METHOD_CAPABILITIES",
    "MagneticDistortionFlag",
    "MartiResult",
    "OBSERVABLE_COLUMNS",
    "OBSERVABLE_DTYPES",
    "ORDINAL_OBSERVABLES",
    "ObservableTable",
    "PRIMARY_OBSERVABLES",
    "STRATA_ORDER",
    "SiteObservables",
    "StratificationSummary",
    "agreement_summary",
    "apply_trust_filter",
    "classify_dimensionality",
    "compare_lilley_marti",
    "complex_to_gamma",
    "compute_coherence",
    "compute_coherence_all",
    "compute_collection_observables",
    "compute_cross_method",
    "compute_cross_method_collection",
    "compute_magnetic_distortion_flag",
    "compute_site_observables",
    "decompose",
    "decompose_bibby",
    "decompose_each_station",
    "decompose_garcia_jones",
    "decompose_gomez_trevino",
    "decompose_joint",
    "decompose_lilley",
    "decompose_marti",
    "decompose_mcneice_jones",
    "default_bin_edges_km",
    "default_period_bands",
    "determinant_resistivity",
    "eigenvector_strike",
    "empirical_variogram",
    "frequency_dependence_diagnostic",
    "gamma_field",
    "gamma_magnitude",
    "gamma_to_complex",
    "haversine_distances_km",
    "invariant_resistivities",
    "irreducible_decomposition",
    "iterative_chain",
    "method_inconsistency_diagnostic",
    "mohr_circle_phase_tensor",
    "pairwise_distances",
    "phase_tensor",
    "phase_tensor_invariants",
    "principal_axis",
    "randomisation_null",
    "series_parallel_resistivities",
    "stratification_summary",
    "stratify_table",
    "threshold_sensitivity",
    "tipper_diagnostic",
    "trust_score",
    "wal_invariants",
    "waldim_dimensionality",
]
