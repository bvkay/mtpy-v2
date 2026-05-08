"""Groom-Bailey single-site and McNeice-Jones joint decomposition.

Algorithm
---------
The Groom-Bailey model (Groom & Bailey, 1989) factorises the 2x2
galvanic-distortion tensor as

    C = g * R(strike) * T(twist) * S(shear) * A(anisotropy)

with named geometric operators, and writes the observed measurement-
frame impedance as

    Z_obs = R(strike) * C * Z_2D(strike) * R(strike).T

where ``Z_2D`` is the regional 2-D impedance (anti-diagonal in the
strike frame, with TE response ``a`` and TM response ``b``). The
unknowns per period are ``(strike, twist, shear, anisotropy, gain, a,
b)``; ``a`` and ``b`` are complex, the rest are real. ``anisotropy`` is
structurally non-identifiable from MT alone.

This module fits all unknowns jointly within a frequency *band* using
weighted nonlinear least-squares (scipy ``trust-constr``). The
optimiser sees a single residual vector packing real and imaginary
parts of the predicted-minus-observed Z entries, normalised by
``z_error``. Each band is fitted from multiple starting points and the
converged points are clustered by canonical (strike, twist, shear) into
"modes"; the lowest-RMS mode is reported as primary, with the
remaining modes preserved on the result for inspection. A parametric
bootstrap is available (synthetic Gaussian noise on the primary-mode
predicted Z) for empirical confidence intervals on the direct GB
parameters.

The McNeice-Jones extension (McNeice & Jones, 2001) ties the regional
strike across multiple sites within a band while leaving twist, shear,
and gain per-site, on the geological argument that a regional 2-D
strike is a property of the survey and not of any single station.

This module implements the public entry points
:func:`decompose` (single-site), :func:`decompose_joint`
(multi-site shared-strike), and :func:`decompose_each_station`
(convenience wrapper that runs single-site GB on every station of a
collection), plus the per-band optimisation, multi-start
orchestration, and parametric-bootstrap helpers they rely on.

References
----------
Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency tensor
decomposition of magnetotelluric data. Geophysics, 66(1), 158-173.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from loguru import logger
from scipy.optimize import least_squares

from .common import (
    _BandResult,
    _band_arrays_to_z,
    _build_bounds_joint,
    _canonical_initial_guess,
    _canonical_initial_guess_joint,
    _compute_ci_percentile,
    _decompose_bands_with_modes_joint,
    _estim_imp,
    _extract_bands,
    _normalise_collection_input,
    _objfun_joint,
    _perturbed_initial_guess,
    _resample_residuals,
    _rotated_initial_guess,
    _rotated_initial_guess_joint,
    _unpack_x,
    _unpack_x_joint,
    _validate_joint_input,
)
from .results import DecompositionResult
from .symmetries import (
    _DEFAULT_MODE_TOLERANCE,
    _Mode,
    _canonical_form_summary_joint,
    _cluster_modes,
    _compute_mode_probabilities,
    _detect_band_disagreement,
    _detect_primary_mode_warning,
    _geometric_fold,
    _resolve_disambiguation,
)

if TYPE_CHECKING:
    from mtpy.core.mt import MT
    from mtpy.core.mt_collection import MTCollection
    from mtpy.core.transfer_function.z import Z


__all__ = [
    "decompose",
    "decompose_each_station",
    "decompose_joint",
]


_DISAMBIGUATION_DESCRIPTIONS = {
    "geometric": (
        "(strike + 90 mod 180, -shear, twist) symmetry folded so "
        "strike in [0, 90); shear sign flipped if pre-fold strike "
        "was in [90, 180)"
    ),
    "identity": (
        "no symmetry fold; strike wrapped to [0, 180) and shear "
        "sign preserved"
    ),
    "pt_aligned": (
        "GB branch chosen per band to minimise mod-180 angular "
        "distance to z.phase_tensor.alpha at the band's median "
        "period; shear sign flipped iff the rotated branch was "
        "selected"
    ),
    "min_shear": (
        "GB branch chosen per band to minimise |shear|; shear "
        "sign flipped iff the rotated branch was selected"
    ),
}


def _disambiguation_description(disambiguation) -> str:
    if isinstance(disambiguation, str):
        return _DISAMBIGUATION_DESCRIPTIONS.get(
            disambiguation, f"unknown disambiguation: {disambiguation!r}"
        )
    return "user-supplied callable disambiguation strategy"


def decompose(
    z: "Z",
    periods: tuple[float, float] | None = None,
    bandwidth: float = 1.0,
    overlap: float = 0.0,
    norm_type: str = "GAVSD2",
    bounds_override: dict[str, tuple[float, float]] | None = None,
    initial_guess: dict[str, float] | None = None,
    realisations: int = 0,
    seed: int | None = None,
    n_starts: int = 5,
    mode_tolerance: dict[str, float] | None = None,
    return_all_modes: bool = False,
    mode_warning_threshold: float = 1.5,
    perturbation_scale: float = 0.1,
    ci_level: float = 0.95,
    ci_method: str = "percentile",
    canonicalise: bool = True,
    disambiguation: "str | callable" = "geometric",
) -> DecompositionResult:
    """Single-site Groom-Bailey decomposition with multi-start.

    Parameters
    ----------
    z : Z
        Impedance tensor object. Reads use ``z.z`` (user-facing,
        unit-corrected). Errors come from ``z.z_error``.

    periods : tuple of float, optional
        ``(permin, permax)`` window in seconds. Default: full range
        of ``z.frequency``.

    bandwidth : float, default 1.0
        Width of each frequency band in log10(period) decades.

    overlap : float, default 0.0
        Overlap between adjacent bands in log10(period) decades.

    norm_type : str, default "GAVSD2"
        Normalisation type for the residuals. See the GB89 paper for
        the available conventions.

    bounds_override : dict, optional
        Override default parameter bounds. Keys: ``"strike"``,
        ``"twist"``, ``"shear"``, ``"gain"``, ``"anisotropy"``.
        Values: ``(lower, upper)``.

    initial_guess : dict, optional
        Override default initial guess. Same keys as
        ``bounds_override``.

    realisations : int, default 0
        Number of parametric bootstrap replicas. ``0`` disables
        bootstrap and reports only the analytical Jacobian-based
        errors. ``> 0`` enables bootstrap; CIs appear in additional
        Dataset fields (``<name>_ci_lower`` / ``<name>_ci_upper``).
        Recommended values: 100 for default quality, 500 for
        publication-quality. Cost scales as
        ``realisations × n_starts × n_bands`` optimisations. Note:
        bootstrap is reliable for direct GB parameters but
        unreliable for invariant-derived quantities per Chave
        (2014); see ``metadata['bootstrap_caveats']``.

    seed : int, optional
        RNG seed. Used both for bootstrap (when implemented) and
        for the multi-start perturbation generation. ``None``
        defaults to a fixed seed (42), so multi-start results are
        reproducible by default.

    n_starts : int, default 5
        Number of optimisation starting points per band. Hybrid
        strategy: canonical phase-tensor guess + 90-rotated guess
        + ``n_starts - 2`` random perturbations. Default 5 is
        sufficient for most cases; pathological multi-modal fits
        can benefit from 10-20. Setting ``n_starts=1`` falls
        through to single-start behaviour.

    mode_tolerance : dict, optional
        Per-parameter tolerances for clustering converged points
        into modes. Keys: ``strike_deg`` (default 0.5),
        ``twist_deg`` (default 0.5), ``shear_deg`` (default 0.5).
        The previously-supported ``rms_relative`` and ``log10_gain``
        keys are no longer used; passing either raises a
        :class:`UserWarning` and the value is ignored. (RMS-based
        matching was removed because of TRF path-dependent noise;
        log10_gain because of gauge equivalence with the regional
        impedance magnitudes at the band level.)

    return_all_modes : bool, default False
        If True, ``parameters`` includes a ``mode`` dimension with
        per-mode parameter values per period (primary mode at
        ``mode=0``). If False, ``parameters`` contains only the
        primary mode, preserving the simple per-period API.

    mode_warning_threshold : float, default 1.5
        If any band's second-best mode has RMS within this factor
        of the primary mode's RMS,
        ``metadata['primary_mode_warning']`` is set to True with
        explanatory text. Default 1.5 means warn when modes are
        within 1.5x of each other.

    perturbation_scale : float, default 0.1
        Standard deviation of the Gaussian perturbation for random
        starting points, as a fraction of the per-parameter bound
        width.

    ci_level : float, default 0.95
        Nominal coverage level for bootstrap CIs. Only used when
        ``realisations > 0``.

    ci_method : str, default 'percentile'
        Bootstrap CI computation method. Currently only
        ``'percentile'`` is implemented; passing anything else
        raises :class:`NotImplementedError`. BCa (bias-corrected
        and accelerated) is planned for a future session.

    canonicalise : bool, default True
        Backward-compatibility shim for the older two-state flag.
        ``canonicalise=True`` is equivalent to
        ``disambiguation='geometric'`` (default), and
        ``canonicalise=False`` is equivalent to
        ``disambiguation='identity'``. New code should pass
        ``disambiguation`` directly. Passing ``canonicalise=False``
        together with an explicit non-default ``disambiguation``
        raises :class:`ValueError`.

    disambiguation : str or callable, default 'geometric'
        Strategy for resolving the GB 90-degree / shear-sign
        symmetry per band. The forward model, predicted Z,
        residuals, and reconstructed C tensor are independent of
        this choice; only the per-band and per-period reporting of
        strike and shear changes. Mode clustering and the bootstrap
        always use ``'geometric'`` internally regardless of this
        choice.

        Accepted values:

        - ``'geometric'`` (default): fold strike into ``[0, 90)``
          and flip shear sign when the fold triggers; matches the
          historical Fortran-validated convention.
        - ``'identity'``: wrap strike to ``[0, 180)`` and leave
          shear unchanged; both branches remain visible.
        - ``'pt_aligned'``: pick the branch closest (mod 180) to
          the phase tensor's principal axis at the band's median
          period. Computed from ``z`` via
          :attr:`Z.phase_tensor`; if the phase tensor cannot be
          built, raises :class:`ValueError`.
        - ``'min_shear'``: pick the branch with smaller
          ``|shear|`` per band.
        - callable: a function with signature
          ``fold(strike_rad, twist_rad, shear_rad) ->
          (strike_rad, twist_rad, shear_rad)``. Returned strikes
          are wrapped to ``[0, 180)`` for reporting.

    Returns
    -------
    DecompositionResult

    Notes
    -----
    Strike convention: clockwise from the x-axis defined by
    ``mt.coordinate_reference_frame`` of the parent station, in
    degrees. Each band is canonicalised through the
    ``(strike + 90 mod 180, -shear, twist)`` symmetry so that the
    reported ``strike`` lands in ``[0, 90)`` (see
    :func:`_canonicalise_solution`).

    Per-band-independent fitting. Each band of periods is optimised
    separately. Joint cross-band fitting and geological-domain
    grouping are planned future work, not implemented in this
    contribution.

    Regional impedance frame. ``regional_z`` is rotated back to the
    measurement frame so it is directly comparable to the input
    ``z`` (e.g., for plotting). The strike-frame anti-diagonal
    representation is recoverable by rotating each period by its
    own ``-strike``.

    Multi-start optimisation. The Groom-Bailey optimisation surface
    is multi-modal: a single TRF run from one starting point can
    converge to a local minimum that does not represent the global
    minimum (Session 4 of the contribution observed up to 33-degree
    azimuth errors against the Fortran reference on the canonical
    strike_example dataset). Multi-start runs the optimiser from
    ``n_starts`` diverse initial guesses, clusters the converged
    points into modes by canonical-form parameters (with user-
    configurable tolerances), and reports the lowest-RMS mode as
    primary. Mode probabilities via the Laplace approximation are
    stored in ``metadata['per_band'][i]['modes']``.

    Primary-mode reporting. The user-facing ``parameters`` Dataset
    and ``regional_z`` always reflect each band's primary (lowest-
    RMS) mode. When ``metadata['primary_mode_warning']`` is True,
    multiple modes are competitive within a band and the primary-
    mode values may not be the best single-physical-solution
    answer; inspect ``metadata['per_band'][i]['modes']`` for
    alternatives. To access all modes as parallel arrays, set
    ``return_all_modes=True``.

    Parametric bootstrap. When ``realisations > 0``, decompose()
    generates ``realisations`` synthetic-data replicas by adding
    parametric Gaussian noise to the primary-mode-fitted Z (with
    sigma per component matching ``z.z_error``). Each replica runs
    the full multi-start machinery; per-replica primary-mode
    parameters form the bootstrap distribution and the
    ``ci_level``-percentile bounds are stored as additional
    ``parameters`` Dataset fields. The full per-replica arrays are
    available in ``metadata['bootstrap_replicates']`` for custom
    analysis.

    Bootstrap reliability. Parametric bootstrap measures sensitivity
    under the assumption that the noise model and the GB89 forward
    model are correct; model-misspecification uncertainty is not
    captured. Replicas that hit a different primary mode than the
    original-data fit are counted via
    ``metadata['bootstrap_mode_warning_fraction']``; if that
    exceeds 50%, ``metadata['bootstrap_robustness_warning']`` flags
    that single-mode CIs are unreliable for this dataset.

    Gauge equivalences and non-identifiability. The Groom-Bailey
    parameterisation has two structural identifiability issues that
    surface in the result:

    - ``anisotropy``: structurally non-identifiable from MT data
      alone. The forward model has no anisotropy dependence, so
      this column of the Jacobian is identically zero; the
      reported ``anisotropy_error`` is ``inf``.
    - ``gain``: gauge-equivalent at the band level — the forward
      model satisfies ``(a, b, gain) -> (gain*a, gain*b, 1)`` so
      different starts converge to physically-equivalent solutions
      with different gain / |a| / |b| splits. Reported gain values
      are the optimiser's chosen gauge, not a discriminating
      physical signal.

    Both parameters are kept in the result for symmetric structure;
    plot the regional impedance magnitudes (or apparent resistivity)
    rather than gain alone for physical interpretation.

    Examples
    --------
    Single-site decomposition with default settings:

    >>> from mtpy.core.transfer_function.z_analysis.decomposition import (
    ...     decompose,
    ... )
    >>> result = decompose(z, n_starts=5, seed=42)
    >>> float(result.parameters["strike"].median())
    >>> result.metadata["primary_mode_warning"]

    With bootstrap for empirical CIs:

    >>> result = decompose(z, n_starts=5, realisations=100, seed=42)
    >>> result.parameters["strike_ci_lower"].values

    Save and reload:

    >>> result.save("decomp.pkl")
    >>> from mtpy.core.transfer_function.z_analysis.decomposition import (
    ...     DecompositionResult,
    ... )
    >>> result2 = DecompositionResult.load("decomp.pkl")

    For the McNeice-Jones multi-site joint decomposition, see
    :func:`decompose_joint`.
    """
    from mtpy.core.transfer_function.z import Z as _Z

    if ci_method != "percentile":
        raise NotImplementedError(
            f"ci_method={ci_method!r} not implemented; only "
            f"'percentile' is currently supported. BCa is planned "
            f"for a future session."
        )

    # Reconcile the legacy ``canonicalise`` flag with the new
    # ``disambiguation`` kwarg. ``canonicalise=False`` is the
    # historical way to ask for the identity fold; honour it only
    # when the caller has not also passed an explicit non-default
    # ``disambiguation`` (mixing both is a programmer error).
    if not canonicalise:
        if disambiguation == "geometric":
            disambiguation = "identity"
        elif disambiguation != "identity":
            raise ValueError(
                "decompose: canonicalise=False conflicts with "
                f"disambiguation={disambiguation!r}; pass "
                "disambiguation='identity' instead."
            )

    frequencies = np.asarray(z.frequency, dtype=np.float64)
    all_periods = 1.0 / frequencies
    if periods is not None:
        permin, permax = periods
        period_mask = (all_periods >= permin) & (all_periods <= permax)
    else:
        period_mask = np.ones(len(all_periods), dtype=bool)
    if not period_mask.any():
        raise ValueError(f"decompose: no periods in window {periods}")

    selected_periods = all_periods[period_mask]
    z_full = np.asarray(z.z, dtype=np.complex128)[period_mask]
    z_err_attr = z.z_error
    if z_err_attr is None:
        raise ValueError(
            "decompose: z.z_error is None; cannot run weighted "
            "least squares without per-component errors."
        )
    sigma_full = np.asarray(z_err_attr, dtype=np.float64)[period_mask]
    if np.any(sigma_full <= 0):
        raise ValueError(
            "decompose: z.z_error contains non-positive entries in "
            "the selected period range"
        )

    sort_idx = np.argsort(selected_periods)
    selected_periods = selected_periods[sort_idx]
    z_selected = z_full[sort_idx]
    sigma_selected = sigma_full[sort_idx]

    bands = _extract_bands(selected_periods, bandwidth=bandwidth, overlap=overlap)

    rng = np.random.default_rng(42 if seed is None else seed)

    band_modes_list = _decompose_bands_with_modes(
        z_obs_full=z_selected,
        sigma_full=sigma_selected,
        selected_periods=selected_periods,
        bands=bands,
        n_starts=n_starts,
        bounds_override=bounds_override,
        rng=rng,
        mode_tolerance=mode_tolerance,
        perturbation_scale=perturbation_scale,
    )

    # Pick primary mode per band for the existing aggregation logic.
    band_results = [
        (band_idx, modes[0].band_result) for band_idx, modes in band_modes_list
    ]

    n_periods = len(selected_periods)
    strike_pp = np.full(n_periods, np.nan)
    twist_pp = np.full(n_periods, np.nan)
    shear_pp = np.full(n_periods, np.nan)
    gain_pp = np.full(n_periods, np.nan)
    aniso_pp = np.zeros(n_periods)
    strike_err_pp = np.full(n_periods, np.nan)
    twist_err_pp = np.full(n_periods, np.nan)
    shear_err_pp = np.full(n_periods, np.nan)
    gain_err_pp = np.full(n_periods, np.nan)
    aniso_err_pp = np.full(n_periods, np.nan)

    log10_rho_a_pp = np.full(n_periods, np.nan)
    phase_a_pp = np.full(n_periods, np.nan)
    log10_rho_b_pp = np.full(n_periods, np.nan)
    phase_b_pp = np.full(n_periods, np.nan)
    log10_rho_a_err_pp = np.full(n_periods, np.nan)
    phase_a_err_pp = np.full(n_periods, np.nan)
    log10_rho_b_err_pp = np.full(n_periods, np.nan)
    phase_b_err_pp = np.full(n_periods, np.nan)

    chi_squared_pp = np.full(n_periods, np.nan)

    # Inverse-variance weights at the band level. Single weight per
    # band, applied to all four scalar parameters (strike, twist,
    # shear, log10_gain). See decomposition.py module notes for the
    # rationale (band-fit-quality weighting vs per-parameter
    # weighting).
    band_weight = {}
    for i, (band_idx, br) in enumerate(band_results):
        n_band = len(band_idx)
        rms = max(br.rms_misfit, 1e-15)
        band_weight[i] = 1.0 / (rms * rms * n_band)

    period_band_indices: dict[int, list[int]] = {i: [] for i in range(n_periods)}
    for i, (band_idx, _br) in enumerate(band_results):
        for global_i in band_idx:
            period_band_indices[int(global_i)].append(i)

    # Per-band PT strikes are needed only for disambiguation='pt_aligned'.
    pt_strike_per_band_rad: list[float | None] = [None] * len(band_results)
    if disambiguation == "pt_aligned":
        try:
            pt_alpha_full_deg = np.asarray(
                z.phase_tensor.alpha, dtype=np.float64
            )[period_mask][sort_idx]
        except Exception as exc:
            raise ValueError(
                "decompose: disambiguation='pt_aligned' needs a phase "
                "tensor strike from z.phase_tensor.alpha but it could "
                "not be computed; pass a callable disambiguation that "
                "supplies the strike explicitly, or use 'geometric' / "
                f"'identity' / 'min_shear' instead. Underlying error: {exc}"
            ) from exc
        for i, (band_idx, _br) in enumerate(band_results):
            band_alpha = pt_alpha_full_deg[band_idx]
            band_alpha = band_alpha[np.isfinite(band_alpha)]
            if band_alpha.size == 0:
                raise ValueError(
                    f"decompose: disambiguation='pt_aligned' band "
                    f"{i} has no finite phase-tensor strike values."
                )
            # Median PT alpha per band; convert deg -> rad.
            pt_strike_per_band_rad[i] = float(
                np.radians(np.median(band_alpha))
            )

    canon_per_band = []
    for i, (band_idx, br) in enumerate(band_results):
        fold = _resolve_disambiguation(
            disambiguation, pt_strike_rad=pt_strike_per_band_rad[i]
        )
        theta_c, twist_c, shear_c = fold(
            br.x_opt[0], br.x_opt[1], br.x_opt[2]
        )
        canon_per_band.append((theta_c, twist_c, shear_c))

    for global_i in range(n_periods):
        b_idxs = period_band_indices[global_i]
        if not b_idxs:
            continue

        ws = np.array([band_weight[i] for i in b_idxs])
        w_sum = float(np.sum(ws))

        thetas = np.array([canon_per_band[i][0] for i in b_idxs])
        twists = np.array([canon_per_band[i][1] for i in b_idxs])
        shears = np.array([canon_per_band[i][2] for i in b_idxs])
        log10_gains = np.array([band_results[i][1].x_opt[3] for i in b_idxs])

        # Circular weighted mean for strike (mod-180 ambiguity is
        # already collapsed by canonicalisation, so a circular mean
        # of 2*strike is appropriate)
        strike_pp[global_i] = (
            np.arctan2(
                np.sum(ws * np.sin(2.0 * thetas)),
                np.sum(ws * np.cos(2.0 * thetas)),
            )
            / 2.0
        )
        twist_pp[global_i] = float(np.average(twists, weights=ws))
        shear_pp[global_i] = float(np.average(shears, weights=ws))
        log10_gain_avg = float(np.average(log10_gains, weights=ws))
        gain_pp[global_i] = 10.0**log10_gain_avg

        # Errors: combine per-band optimiser errors using the
        # inverse-variance combination formula. Use raw (un-canonical)
        # x_err entries since canonicalisation is a sign flip on shear
        # and a 90-degree shift on strike — neither changes variance.
        strike_errs = np.array([band_results[i][1].x_err[0] for i in b_idxs])
        twist_errs = np.array([band_results[i][1].x_err[1] for i in b_idxs])
        shear_errs = np.array([band_results[i][1].x_err[2] for i in b_idxs])
        log10_gain_errs = np.array([band_results[i][1].x_err[3] for i in b_idxs])

        def _combine(errs: np.ndarray) -> float:
            finite = np.isfinite(errs) & (errs > 0)
            if not finite.any():
                return float("inf")
            inv_var = np.zeros_like(errs)
            inv_var[finite] = 1.0 / (errs[finite] ** 2)
            denom = float(np.sum(inv_var))
            if denom <= 0.0:
                return float("inf")
            return float(np.sqrt(1.0 / denom))

        strike_err_pp[global_i] = _combine(strike_errs)
        twist_err_pp[global_i] = _combine(twist_errs)
        shear_err_pp[global_i] = _combine(shear_errs)
        log10_gain_err_combined = _combine(log10_gain_errs)
        # gain = 10**log10_gain; sigma_gain = gain * ln(10) * sigma_log10
        if np.isfinite(log10_gain_err_combined):
            gain_err_pp[global_i] = (
                gain_pp[global_i] * np.log(10.0) * log10_gain_err_combined
            )
        else:
            gain_err_pp[global_i] = float("inf")

        # Per-frequency regional impedance: pull from each band that
        # contains this period. If multiple bands contain it,
        # inverse-variance combine the (rho, phase) values.
        log10_rho_as = []
        phase_as = []
        log10_rho_bs = []
        phase_bs = []
        log10_rho_a_errs = []
        phase_a_errs = []
        log10_rho_b_errs = []
        phase_b_errs = []
        chi_sq_at_period = []
        for i in b_idxs:
            band_idx_i, br_i = band_results[i]
            local_i = int(np.where(band_idx_i == global_i)[0][0])
            n_band_i = len(band_idx_i)
            log10_rho_as.append(br_i.x_opt[5 + 0 * n_band_i + local_i])
            phase_as.append(br_i.x_opt[5 + 1 * n_band_i + local_i])
            log10_rho_bs.append(br_i.x_opt[5 + 2 * n_band_i + local_i])
            phase_bs.append(br_i.x_opt[5 + 3 * n_band_i + local_i])
            log10_rho_a_errs.append(br_i.x_err[5 + 0 * n_band_i + local_i])
            phase_a_errs.append(br_i.x_err[5 + 1 * n_band_i + local_i])
            log10_rho_b_errs.append(br_i.x_err[5 + 2 * n_band_i + local_i])
            phase_b_errs.append(br_i.x_err[5 + 3 * n_band_i + local_i])
            # 8 residuals per period, packed as
            # [Re/Im of XX, XY, YX, YY] / sigma.
            resid_slice = br_i.residuals[8 * local_i : 8 * (local_i + 1)]
            chi_sq_at_period.append(float(np.sum(resid_slice**2)))

        log10_rho_a_pp[global_i] = float(np.average(log10_rho_as, weights=ws))
        phase_a_pp[global_i] = float(np.average(phase_as, weights=ws))
        log10_rho_b_pp[global_i] = float(np.average(log10_rho_bs, weights=ws))
        phase_b_pp[global_i] = float(np.average(phase_bs, weights=ws))
        log10_rho_a_err_pp[global_i] = _combine(np.array(log10_rho_a_errs))
        phase_a_err_pp[global_i] = _combine(np.array(phase_a_errs))
        log10_rho_b_err_pp[global_i] = _combine(np.array(log10_rho_b_errs))
        phase_b_err_pp[global_i] = _combine(np.array(phase_b_errs))
        # Average chi-squared across bands containing this period
        chi_squared_pp[global_i] = float(np.mean(chi_sq_at_period))

    # For non-geometric folds, per-band strikes live in [0, pi) so the
    # circular weighted mean (via arctan2 of doubled angles divided by
    # two) can land in (-pi/2, pi/2]; wrap into [0, pi) so reported
    # strike covers the full un-folded range. For the geometric fold,
    # per-band strikes live in [0, pi/2) and the wrap is a no-op.
    finite_mask = np.isfinite(strike_pp)
    strike_pp[finite_mask] = strike_pp[finite_mask] % np.pi

    # Build regional Z in strike frame, then rotate to measurement
    # frame.
    z_regional_strike, z_regional_strike_err = _band_arrays_to_z(
        log10_rho_a_pp,
        phase_a_pp,
        log10_rho_b_pp,
        phase_b_pp,
        log10_rho_a_err_pp,
        phase_a_err_pp,
        log10_rho_b_err_pp,
        phase_b_err_pp,
        selected_periods,
    )
    z_regional_meas = np.empty_like(z_regional_strike)
    z_regional_meas_err = np.empty_like(z_regional_strike_err)
    for k in range(n_periods):
        theta_k = strike_pp[k]
        if not np.isfinite(theta_k):
            z_regional_meas[k] = z_regional_strike[k]
            z_regional_meas_err[k] = z_regional_strike_err[k]
            continue
        c, s = np.cos(theta_k), np.sin(theta_k)
        R = np.array([[c, -s], [s, c]])
        z_regional_meas[k] = R @ z_regional_strike[k] @ R.T
        # Variance propagation through orthogonal rotation: each
        # measurement-frame entry is a linear combination of strike-
        # frame entries with coefficients in {c^2, s^2, +/- c*s}.
        # Use the conservative diagonal-only propagation.
        c2, s2 = c * c, s * s
        cs = abs(c * s)
        sigma_strike = z_regional_strike_err[k]
        z_regional_meas_err[k, 0, 0] = np.sqrt(
            (s2 * sigma_strike[0, 1]) ** 2
            + (cs * sigma_strike[0, 0]) ** 2
            + (cs * sigma_strike[1, 1]) ** 2
            + (s2 * sigma_strike[1, 0]) ** 2
        )
        z_regional_meas_err[k, 0, 1] = np.sqrt(
            (c2 * sigma_strike[0, 1]) ** 2
            + (cs * sigma_strike[0, 0]) ** 2
            + (cs * sigma_strike[1, 1]) ** 2
            + (s2 * sigma_strike[1, 0]) ** 2
        )
        z_regional_meas_err[k, 1, 0] = np.sqrt(
            (c2 * sigma_strike[1, 0]) ** 2
            + (cs * sigma_strike[0, 0]) ** 2
            + (cs * sigma_strike[1, 1]) ** 2
            + (s2 * sigma_strike[0, 1]) ** 2
        )
        z_regional_meas_err[k, 1, 1] = np.sqrt(
            (s2 * sigma_strike[1, 0]) ** 2
            + (cs * sigma_strike[0, 0]) ** 2
            + (cs * sigma_strike[1, 1]) ** 2
            + (s2 * sigma_strike[0, 1]) ** 2
        )

    regional_frequencies = 1.0 / selected_periods
    regional_z_obj = _Z(
        z=z_regional_meas,
        z_error=z_regional_meas_err,
        frequency=regional_frequencies,
    )

    params = xr.Dataset(
        {
            "strike": ("period", np.degrees(strike_pp)),
            "twist": ("period", np.degrees(twist_pp)),
            "shear": ("period", np.degrees(shear_pp)),
            "gain": ("period", gain_pp),
            "anisotropy": ("period", aniso_pp),
            "strike_error": ("period", np.degrees(strike_err_pp)),
            "twist_error": ("period", np.degrees(twist_err_pp)),
            "shear_error": ("period", np.degrees(shear_err_pp)),
            "gain_error": ("period", gain_err_pp),
            "anisotropy_error": ("period", aniso_err_pp),
        },
        coords={"period": selected_periods},
    )
    strike_range_str = "[0, 90)" if disambiguation == "geometric" else "[0, 180)"
    params["strike"].attrs.update(
        units="degrees",
        range=strike_range_str,
        convention="clockwise from x-axis",
    )
    params["twist"].attrs.update(units="degrees")
    params["shear"].attrs.update(units="degrees")
    params["gain"].attrs.update(units="dimensionless")
    params["anisotropy"].attrs.update(units="dimensionless", non_identifiable=True)
    params["period"].attrs.update(units="seconds")

    total_chi_sq = sum(br.chi_squared for _, br in band_results)
    total_n_resid = sum(len(br.residuals) for _, br in band_results)
    rms_misfit = float(np.sqrt(total_chi_sq / max(total_n_resid, 1)))

    chi_squared_da = xr.DataArray(
        chi_squared_pp,
        coords={"period": selected_periods},
        dims=["period"],
        attrs={
            "degrees_of_freedom": 8,
            "description": (
                "Per-period chi-squared = sum of 8 sigma-weighted "
                "squared residuals (Re/Im of XX, XY, YX, YY). For "
                "periods in multiple bands, averaged across bands."
            ),
        },
    )

    per_band_metadata = []
    for band_idx, modes in band_modes_list:
        mode_dicts = []
        for mode in modes:
            br_m = mode.band_result
            mode_dicts.append(
                {
                    "rms_misfit": mode.rms_misfit,
                    "chi_squared": mode.chi_squared,
                    "n_starts_landing_here": mode.n_starts_landing_here,
                    "probability": mode.probability,
                    "x_opt": br_m.x_opt.tolist(),
                    "x_err": br_m.x_err.tolist(),
                    "converged": br_m.converged,
                    "n_iter": br_m.n_iter,
                    "canonical_form": mode.canonical_form,
                }
            )
        per_band_metadata.append(
            {
                "band_period_indices": band_idx.tolist(),
                "band_periods": selected_periods[band_idx].tolist(),
                "modes": mode_dicts,
                "n_modes": len(modes),
                "primary_mode_index": 0,
            }
        )

    primary_mode_warning, primary_mode_warning_text = _detect_primary_mode_warning(
        band_modes_list, mode_warning_threshold
    )
    band_disagreement = _detect_band_disagreement(
        band_modes_list, mode_tolerance or _DEFAULT_MODE_TOLERANCE
    )

    metadata = {
        "convention": "clockwise from x-axis",
        "strike_range_degrees": strike_range_str,
        "disambiguation": (
            disambiguation if isinstance(disambiguation, str) else "callable"
        ),
        "canonicalisation": _disambiguation_description(disambiguation),
        "static_shift_convention": ("gain consistent with MT.remove_static_shift"),
        "regional_z_frame": "measurement",
        "per_band": per_band_metadata,
        "n_bands": len(band_results),
        "n_starts_per_band": n_starts,
        "mode_tolerance": (
            {k: v for k, v in (mode_tolerance or {}).items() if k != "rms_relative"}
            or dict(_DEFAULT_MODE_TOLERANCE)
        ),
        "primary_mode_warning": primary_mode_warning,
    }
    if primary_mode_warning:
        metadata["primary_mode_warning_text"] = primary_mode_warning_text
    if band_disagreement is not None:
        metadata["band_disagreement"] = band_disagreement

    if return_all_modes:
        params = _build_per_mode_parameters_dataset(band_modes_list, selected_periods)

    if realisations > 0:
        z_predicted = _predict_z_from_primary_modes(band_modes_list, selected_periods)
        bootstrap_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        bootstrap_replicates = _bootstrap_decompose(
            z_obs_full=z_selected,
            sigma_full=sigma_selected,
            selected_periods=selected_periods,
            bands=bands,
            z_predicted=z_predicted,
            realisations=realisations,
            n_starts=n_starts,
            bounds_override=bounds_override,
            rng=bootstrap_rng,
            mode_tolerance=mode_tolerance,
            perturbation_scale=perturbation_scale,
            mode_warning_threshold=mode_warning_threshold,
        )

        ci_strike_lower, ci_strike_upper = _compute_ci_percentile(
            bootstrap_replicates["strike"], level=ci_level
        )
        ci_twist_lower, ci_twist_upper = _compute_ci_percentile(
            bootstrap_replicates["twist"], level=ci_level
        )
        ci_shear_lower, ci_shear_upper = _compute_ci_percentile(
            bootstrap_replicates["shear"], level=ci_level
        )
        ci_gain_lower, ci_gain_upper = _compute_ci_percentile(
            bootstrap_replicates["gain"], level=ci_level
        )

        params["strike_ci_lower"] = (("period",), ci_strike_lower)
        params["strike_ci_upper"] = (("period",), ci_strike_upper)
        params["twist_ci_lower"] = (("period",), ci_twist_lower)
        params["twist_ci_upper"] = (("period",), ci_twist_upper)
        params["shear_ci_lower"] = (("period",), ci_shear_lower)
        params["shear_ci_upper"] = (("period",), ci_shear_upper)
        params["gain_ci_lower"] = (("period",), ci_gain_lower)
        params["gain_ci_upper"] = (("period",), ci_gain_upper)
        for k in (
            "strike_ci_lower",
            "strike_ci_upper",
            "twist_ci_lower",
            "twist_ci_upper",
            "shear_ci_lower",
            "shear_ci_upper",
        ):
            params[k].attrs["units"] = "degrees"
        for k in ("gain_ci_lower", "gain_ci_upper"):
            params[k].attrs["units"] = "dimensionless"

        metadata["bootstrap_replicates"] = bootstrap_replicates
        metadata["bootstrap_realisations"] = realisations
        metadata["bootstrap_ci_method"] = ci_method
        metadata["bootstrap_ci_level"] = ci_level
        metadata["bootstrap_n_failed"] = int(
            np.sum(np.isnan(bootstrap_replicates["rms_misfit"]))
        )
        metadata["bootstrap_caveats"] = (
            "Parametric bootstrap CIs assume the GB89 forward model "
            "is correct and the noise model (independent Gaussian "
            "real and imaginary parts per tensor component) is "
            "correct. Model-misspecification uncertainty is not "
            "captured. CIs are reliable for direct GB parameters "
            "(strike, twist, shear, gain, regional Z components). "
            "Per Chave (2014), CIs derived from these via phase-"
            "tensor invariants or Mohr-circle quantities are "
            "formally meaningless — those derived quantities have "
            "infinite asymptotic variance and are not amenable to "
            "bootstrap CI methods."
        )
        mode_warning_fraction = float(np.mean(bootstrap_replicates["mode_warning"]))
        metadata["bootstrap_mode_warning_fraction"] = mode_warning_fraction
        if mode_warning_fraction > 0.5:
            metadata["bootstrap_robustness_warning"] = (
                f"More than half of bootstrap replicas "
                f"({100*mode_warning_fraction:.0f}%) triggered the "
                f"primary-mode warning. Single-mode CIs may be "
                f"misleading; the data does not strongly distinguish "
                f"between competing physical solutions."
            )

    options_record = {
        "periods": periods,
        "bandwidth": bandwidth,
        "overlap": overlap,
        "norm_type": norm_type,
        "bounds_override": bounds_override,
        "initial_guess": initial_guess,
        "realisations": realisations,
        "seed": seed,
        "n_starts": n_starts,
        "mode_tolerance": mode_tolerance,
        "return_all_modes": return_all_modes,
        "mode_warning_threshold": mode_warning_threshold,
        "perturbation_scale": perturbation_scale,
        "ci_level": ci_level,
        "ci_method": ci_method,
        "canonicalise": canonicalise,
        "disambiguation": (
            disambiguation if isinstance(disambiguation, str) else "callable"
        ),
    }

    return DecompositionResult(
        parameters=params,
        regional_z=regional_z_obj,
        chi_squared=chi_squared_da,
        rms_misfit=rms_misfit,
        method="groom_bailey",
        options=options_record,
        metadata=metadata,
        frame="measurement",
    )

def decompose_joint(
    collection_or_list: "MTCollection | list[MT]",
    periods: tuple[float, float] | None = None,
    bandwidth: float = 1.0,
    overlap: float = 0.0,
    norm_type: str = "GAVSD2",
    bounds_override: dict[str, tuple[float, float]] | None = None,
    initial_guess: dict[str, float] | None = None,
    realisations: int = 0,
    seed: int | None = None,
    n_starts: int = 5,
    mode_tolerance: dict[str, float] | None = None,
    return_all_modes: bool = False,
    mode_warning_threshold: float = 1.5,
    perturbation_scale: float = 0.1,
    ci_level: float = 0.95,
    ci_method: str = "percentile",
    strike_sharing: str = "per_band_shared",
    mode_clustering: str = "all_sites_agree",
    regional_z_format: str = "dict",
) -> DecompositionResult:
    """Multi-site joint Groom-Bailey decomposition (McNeice & Jones, 2001).

    Fits a single shared regional strike per band across all sites,
    with per-site distortion (twist, shear, gain) and per-site per-
    frequency regional impedance.

    Parameters
    ----------
    collection_or_list : MTCollection or list of MT
        Multi-site input. List-of-MT path is the primary supported
        entry; MTCollection is supported via its ``dataframe`` and
        ``get_tf`` interface. All stations must share a common
        frequency grid.
    periods, bandwidth, overlap, norm_type, bounds_override,
    initial_guess, realisations, seed, n_starts, mode_tolerance,
    return_all_modes, mode_warning_threshold, perturbation_scale,
    ci_level, ci_method
        See :func:`decompose`.

    Future-scope parameters
    -----------------------
    strike_sharing : str, default ``'per_band_shared'``
        How regional strike is shared across stations. Currently
        only ``'per_band_shared'`` is implemented (one strike per
        band, shared across all stations within the band). Planned:
        ``'survey_global'`` (single strike across all bands and
        stations), ``'per_station_smoothed'`` (regional
        decomposition with spatial smoothness prior),
        ``'per_band_per_station_unconstrained'`` (no sharing). Other
        values raise :class:`NotImplementedError`.
    mode_clustering : str, default ``'all_sites_agree'``
        Predicate for grouping multi-start replicas into modes.
        Currently only ``'all_sites_agree'`` is implemented (every
        site's (twist, shear) must agree for two replicas to be the
        same mode). Planned: ``'majority_sites_agree'``,
        ``'strike_only'``. Other values raise
        :class:`NotImplementedError`.
    regional_z_format : str, default ``'dict'``
        Format for ``result.regional_z``. Currently only ``'dict'``
        (``dict[station_id, Z]``). Planned: ``'mtcollection'``
        (returns a fresh ``MTCollection``). Other values raise
        :class:`NotImplementedError`.

    Returns
    -------
    DecompositionResult
        ``parameters`` is an :class:`xarray.Dataset` with mixed
        dimensionality:

        - ``strike``, ``strike_error`` (and optional bootstrap CI
          fields): shape ``(period,)`` — shared across stations
        - ``twist``, ``shear``, ``gain``, ``anisotropy`` (with
          ``_error`` and CI fields): shape ``(period, station)``

        ``regional_z`` is ``dict[station_id, Z]`` (per
        ``regional_z_format``).
        ``method`` is ``'mcneice_jones_joint'``.

    Notes
    -----
    Joint single-band optimisation has roughly ``1 + 4*n_sites*(1 +
    n_freqs)`` parameters. With 5 sites and 5 freqs/band, that is
    ~120 parameters per band. TRF with the analytic Jacobian handles
    this efficiently; per-band fits typically converge in well under
    a second on synthetic data.

    Real-data observations from the Vulcan_2022 BBMT dataset (62
    periods per station, ~5 bands at default ``bandwidth=1.0``):

      - ``decompose_joint`` on 9 stations, ``n_starts=5``,
        ``realisations=0``: ~19 minutes
      - With ``realisations=50``: ~22 minutes
      - With ``realisations=100``: extrapolates to ~30+ minutes

    Cost scales as roughly
    ``realisations × n_starts × n_bands × per_band_cost``, with
    ``per_band_cost`` rising sharply with ``n_sites`` (more sites
    means more parameters and slower TRF iterations). For
    exploratory analysis on real data, start with
    ``realisations=0`` and ``n_starts=2``; promote to the full
    multi-start with bootstrap for final analysis only.

    Real data also routinely has ragged frequency grids across
    stations (only ~1 of 25 stations matched the first station's
    grid in the Burra_2017-18 and Kalkaroo_2022 datasets).
    :func:`decompose_joint` requires a common grid and raises if
    inputs disagree. For surveys with ragged grids, either
    pre-resample to a common grid or use
    :func:`decompose_each_station` for per-station analysis.

    For ``n_sites == 1``, joint decomposition reduces to single-site
    :func:`decompose` modulo the result data structure (verified by
    ``TestDecomposeJointSingleSiteReduction``).

    Bootstrap scope. This session ships strike/twist/shear/gain
    bootstrap CIs as ``parameters`` Dataset fields. Per-station
    regional Z bootstrap CIs are not stored on the Dataset (they
    would require five-dimensional arrays); the per-replica
    primary-mode parameters are available in
    ``metadata['bootstrap_replicates']`` for custom analysis.

    References
    ----------
    McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
    tensor decomposition of magnetotelluric data. Geophysics, 66(1),
    158-173.
    """
    from mtpy.core.transfer_function.z import Z as _Z

    if strike_sharing != "per_band_shared":
        raise NotImplementedError(
            f"strike_sharing={strike_sharing!r} is planned future work; "
            f"only 'per_band_shared' is currently implemented."
        )
    if mode_clustering != "all_sites_agree":
        raise NotImplementedError(
            f"mode_clustering={mode_clustering!r} is planned future "
            f"work; only 'all_sites_agree' is currently implemented."
        )
    if regional_z_format != "dict":
        raise NotImplementedError(
            f"regional_z_format={regional_z_format!r} is planned "
            f"future work; only 'dict' is currently implemented."
        )
    if ci_method != "percentile":
        raise NotImplementedError(
            f"ci_method={ci_method!r} not implemented; only "
            f"'percentile' is supported."
        )

    stations = _normalise_collection_input(collection_or_list)
    _validate_joint_input(stations)

    n_sites = len(stations)
    station_ids = [sid for sid, _ in stations]
    ref_z = stations[0][1]
    frequencies = np.asarray(ref_z.frequency, dtype=np.float64)
    all_periods = 1.0 / frequencies

    if periods is not None:
        permin, permax = periods
        period_mask = (all_periods >= permin) & (all_periods <= permax)
    else:
        period_mask = np.ones(len(all_periods), dtype=bool)
    if not period_mask.any():
        raise ValueError(f"decompose_joint: no periods in window {periods}")

    selected_periods = all_periods[period_mask]
    sort_idx = np.argsort(selected_periods)
    selected_periods = selected_periods[sort_idx]
    n_periods = len(selected_periods)

    z_per_site = np.empty((n_sites, n_periods, 2, 2), dtype=np.complex128)
    sigma_per_site = np.empty((n_sites, n_periods, 2, 2), dtype=np.float64)
    for i, (_, z) in enumerate(stations):
        z_full = np.asarray(z.z, dtype=np.complex128)[period_mask]
        sigma_full = np.asarray(z.z_error, dtype=np.float64)[period_mask]
        z_per_site[i] = z_full[sort_idx]
        sigma_per_site[i] = sigma_full[sort_idx]

    bands = _extract_bands(selected_periods, bandwidth=bandwidth, overlap=overlap)

    rng = np.random.default_rng(42 if seed is None else seed)

    band_modes_list = _decompose_bands_with_modes_joint(
        z_obs_per_site_full=z_per_site,
        sigma_per_site_full=sigma_per_site,
        selected_periods=selected_periods,
        bands=bands,
        n_starts=n_starts,
        bounds_override=bounds_override,
        rng=rng,
        mode_tolerance=mode_tolerance,
        perturbation_scale=perturbation_scale,
        objfun_joint=_objfun_joint,
        build_bounds_joint=_build_bounds_joint,
        canonical_initial_guess_joint=_canonical_initial_guess_joint,
        rotated_initial_guess_joint=_rotated_initial_guess_joint,
    )

    # Aggregation: pick each band's primary mode, expand band-level
    # scalars to per-period values, build per-site arrays.
    strike_pp = np.full(n_periods, np.nan)
    strike_err_pp = np.full(n_periods, np.nan)
    twist_pp = np.full((n_periods, n_sites), np.nan)
    shear_pp = np.full((n_periods, n_sites), np.nan)
    gain_pp = np.full((n_periods, n_sites), np.nan)
    aniso_pp = np.zeros((n_periods, n_sites))
    twist_err_pp = np.full((n_periods, n_sites), np.nan)
    shear_err_pp = np.full((n_periods, n_sites), np.nan)
    gain_err_pp = np.full((n_periods, n_sites), np.nan)
    aniso_err_pp = np.full((n_periods, n_sites), np.nan)
    chi_squared_pp = np.full(n_periods, np.nan)

    # Per-site per-period regional Z (measurement frame)
    z_regional_per_site = np.full(
        (n_sites, n_periods, 2, 2), np.nan, dtype=np.complex128
    )

    # For overlapping bands, lower-RMS band wins per period.
    best_rms_per_period = np.full(n_periods, np.inf)

    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0

    for band_idx, modes in band_modes_list:
        primary = modes[0]
        br = primary.band_result
        n_band_freqs = len(band_idx)
        band_periods = selected_periods[band_idx]

        cf = _canonical_form_summary_joint(br, n_sites, n_band_freqs)
        strike_can_rad = np.radians(cf["strike_deg"])
        per_site_twist_rad = np.radians(cf["per_site_twist_deg"])
        per_site_shear_rad = np.radians(cf["per_site_shear_deg"])
        per_site_log10_gain = cf["per_site_log10_gain"]

        # Per-band Jacobian-based parameter errors
        x_err = br.x_err
        strike_err_rad = float(x_err[0])
        per_site_twist_err_rad = np.array(
            [float(x_err[1 + 4 * i + 0]) for i in range(n_sites)]
        )
        per_site_shear_err_rad = np.array(
            [float(x_err[1 + 4 * i + 1]) for i in range(n_sites)]
        )
        per_site_log10_gain_err = np.array(
            [float(x_err[1 + 4 * i + 2]) for i in range(n_sites)]
        )

        # Regional Z per site at each band period (measurement frame)
        unpacked = _unpack_x_joint(br.x_opt, n_sites, n_band_freqs)
        log10_rho_a_band = unpacked[5]
        phase_a_band = unpacked[6]
        log10_rho_b_band = unpacked[7]
        phase_b_band = unpacked[8]

        c, s = np.cos(strike_can_rad), np.sin(strike_can_rad)
        R = np.array([[c, -s], [s, c]])

        for i in range(n_sites):
            rho_a_i = 10.0 ** log10_rho_a_band[i]
            rho_b_i = 10.0 ** log10_rho_b_band[i]
            abs_a_i = np.sqrt(rho_a_i * factor / band_periods)
            abs_b_i = np.sqrt(rho_b_i * factor / band_periods)
            a_band_i = abs_a_i * np.exp(1j * phase_a_band[i])
            b_band_i = abs_b_i * np.exp(1j * phase_b_band[i])

            for local_k, global_k in enumerate(band_idx):
                if primary.rms_misfit >= best_rms_per_period[global_k]:
                    continue
                z_strike = np.array(
                    [
                        [0.0, a_band_i[local_k]],
                        [-b_band_i[local_k], 0.0],
                    ],
                    dtype=np.complex128,
                )
                z_regional_per_site[i, global_k] = R @ z_strike @ R.T

        # Per-period scalars: take this band's values at each period
        # it covers (lower-RMS wins on overlap).
        for local_k, global_k in enumerate(band_idx):
            if primary.rms_misfit >= best_rms_per_period[global_k]:
                continue
            best_rms_per_period[global_k] = primary.rms_misfit
            strike_pp[global_k] = np.degrees(strike_can_rad)
            strike_err_pp[global_k] = np.degrees(strike_err_rad)
            for i in range(n_sites):
                twist_pp[global_k, i] = np.degrees(per_site_twist_rad[i])
                shear_pp[global_k, i] = np.degrees(per_site_shear_rad[i])
                gain_pp[global_k, i] = 10.0 ** per_site_log10_gain[i]
                twist_err_pp[global_k, i] = np.degrees(per_site_twist_err_rad[i])
                shear_err_pp[global_k, i] = np.degrees(per_site_shear_err_rad[i])
                if np.isfinite(per_site_log10_gain_err[i]):
                    gain_err_pp[global_k, i] = (
                        gain_pp[global_k, i] * np.log(10.0) * per_site_log10_gain_err[i]
                    )
                else:
                    gain_err_pp[global_k, i] = np.inf
            # Per-period chi-squared (sum of 8 weighted resids per
            # site at this freq, summed across sites).
            resid_at_period = 0.0
            n_band_freqs_b = len(band_idx)
            for i in range(n_sites):
                start = 8 * (i * n_band_freqs_b + local_k)
                resid_at_period += float(np.sum(br.residuals[start : start + 8] ** 2))
            chi_squared_pp[global_k] = resid_at_period

    # Build params Dataset
    params = xr.Dataset(
        {
            "strike": ("period", strike_pp),
            "strike_error": ("period", strike_err_pp),
            "twist": (("period", "station"), twist_pp),
            "twist_error": (("period", "station"), twist_err_pp),
            "shear": (("period", "station"), shear_pp),
            "shear_error": (("period", "station"), shear_err_pp),
            "gain": (("period", "station"), gain_pp),
            "gain_error": (("period", "station"), gain_err_pp),
            "anisotropy": (("period", "station"), aniso_pp),
            "anisotropy_error": (("period", "station"), aniso_err_pp),
        },
        coords={
            "period": selected_periods,
            "station": station_ids,
        },
    )
    params["strike"].attrs.update(
        units="degrees",
        range="[0, 90)",
        convention="clockwise from x-axis",
        sharing="per_band_shared (one strike per band across stations)",
    )
    params["twist"].attrs.update(units="degrees")
    params["shear"].attrs.update(units="degrees")
    params["gain"].attrs.update(units="dimensionless")
    params["anisotropy"].attrs.update(units="dimensionless", non_identifiable=True)
    params["period"].attrs.update(units="seconds")

    # Build regional_z dict[station_id, Z]
    regional_z_meas_err = np.full((n_sites, n_periods, 2, 2), np.nan)
    regional_z = {}
    for i, sid in enumerate(station_ids):
        regional_z[sid] = _Z(
            z=z_regional_per_site[i],
            z_error=regional_z_meas_err[i],
            frequency=1.0 / selected_periods,
        )

    # Total RMS across all bands' primary modes
    total_chi_sq = sum(modes[0].chi_squared for _, modes in band_modes_list)
    total_n_resid = sum(
        len(modes[0].band_result.residuals) for _, modes in band_modes_list
    )
    rms_misfit = float(np.sqrt(total_chi_sq / max(total_n_resid, 1)))

    chi_squared_da = xr.DataArray(
        chi_squared_pp,
        coords={"period": selected_periods},
        dims=["period"],
        attrs={
            "degrees_of_freedom": 8 * n_sites,
            "description": (
                f"Per-period chi-squared = sum across {n_sites} "
                f"stations of 8 sigma-weighted squared residuals."
            ),
        },
    )

    # Per-band metadata with per-mode info
    per_band_metadata = []
    for band_idx, modes in band_modes_list:
        mode_dicts = []
        for mode in modes:
            br_m = mode.band_result
            mode_dicts.append(
                {
                    "rms_misfit": mode.rms_misfit,
                    "chi_squared": mode.chi_squared,
                    "n_starts_landing_here": mode.n_starts_landing_here,
                    "probability": mode.probability,
                    "x_opt": br_m.x_opt.tolist(),
                    "x_err": br_m.x_err.tolist(),
                    "converged": br_m.converged,
                    "n_iter": br_m.n_iter,
                    "canonical_form": {
                        "strike_deg": mode.canonical_form["strike_deg"],
                        "per_site_twist_deg": mode.canonical_form[
                            "per_site_twist_deg"
                        ].tolist(),
                        "per_site_shear_deg": mode.canonical_form[
                            "per_site_shear_deg"
                        ].tolist(),
                        "per_site_log10_gain": mode.canonical_form[
                            "per_site_log10_gain"
                        ].tolist(),
                        "rms_misfit": mode.canonical_form["rms_misfit"],
                    },
                }
            )
        per_band_metadata.append(
            {
                "band_period_indices": band_idx.tolist(),
                "band_periods": selected_periods[band_idx].tolist(),
                "modes": mode_dicts,
                "n_modes": len(modes),
                "primary_mode_index": 0,
            }
        )

    primary_mode_warning = False
    primary_mode_warning_text = ""
    for _, modes in band_modes_list:
        if len(modes) >= 2:
            ratio = modes[1].rms_misfit / max(modes[0].rms_misfit, 1e-30)
            if ratio <= mode_warning_threshold:
                primary_mode_warning = True
                primary_mode_warning_text = (
                    f"At least one band discovered multiple joint "
                    f"modes within {mode_warning_threshold}x RMS of "
                    f"each other. Single-mode CIs are misleading."
                )
                break

    metadata = {
        "convention": "clockwise from x-axis",
        "strike_range_degrees": "[0, 90)",
        "regional_z_frame": "measurement",
        "n_sites": n_sites,
        "station_ids": list(station_ids),
        "n_bands": len(band_modes_list),
        "n_starts_per_band": n_starts,
        "strike_sharing": strike_sharing,
        "mode_clustering": mode_clustering,
        "regional_z_format": regional_z_format,
        "mode_tolerance": (
            {
                k: v
                for k, v in (mode_tolerance or {}).items()
                if k not in ("rms_relative", "log10_gain")
            }
            or dict(_DEFAULT_MODE_TOLERANCE)
        ),
        "primary_mode_warning": primary_mode_warning,
        "per_band": per_band_metadata,
    }
    if primary_mode_warning:
        metadata["primary_mode_warning_text"] = primary_mode_warning_text

    # Bootstrap (minimal: strike/twist/shear/gain CIs only;
    # regional Z bootstrap CIs deferred — see docstring).
    if realisations > 0:
        bootstrap_rng = np.random.default_rng(int(rng.integers(0, 2**32)))
        bootstrap_replicates = _bootstrap_decompose_joint(
            z_obs_per_site_full=z_per_site,
            sigma_per_site_full=sigma_per_site,
            selected_periods=selected_periods,
            bands=bands,
            band_modes_list=band_modes_list,
            realisations=realisations,
            n_starts=n_starts,
            bounds_override=bounds_override,
            rng=bootstrap_rng,
            mode_tolerance=mode_tolerance,
            perturbation_scale=perturbation_scale,
            mode_warning_threshold=mode_warning_threshold,
        )

        ci_strike_lower, ci_strike_upper = _compute_ci_percentile(
            bootstrap_replicates["strike"], level=ci_level
        )
        ci_twist_lower, ci_twist_upper = _compute_ci_percentile(
            bootstrap_replicates["twist"], level=ci_level
        )
        ci_shear_lower, ci_shear_upper = _compute_ci_percentile(
            bootstrap_replicates["shear"], level=ci_level
        )
        ci_gain_lower, ci_gain_upper = _compute_ci_percentile(
            bootstrap_replicates["gain"], level=ci_level
        )

        params["strike_ci_lower"] = (("period",), ci_strike_lower)
        params["strike_ci_upper"] = (("period",), ci_strike_upper)
        params["twist_ci_lower"] = (("period", "station"), ci_twist_lower)
        params["twist_ci_upper"] = (("period", "station"), ci_twist_upper)
        params["shear_ci_lower"] = (("period", "station"), ci_shear_lower)
        params["shear_ci_upper"] = (("period", "station"), ci_shear_upper)
        params["gain_ci_lower"] = (("period", "station"), ci_gain_lower)
        params["gain_ci_upper"] = (("period", "station"), ci_gain_upper)
        for k in (
            "strike_ci_lower",
            "strike_ci_upper",
            "twist_ci_lower",
            "twist_ci_upper",
            "shear_ci_lower",
            "shear_ci_upper",
        ):
            params[k].attrs["units"] = "degrees"
        for k in ("gain_ci_lower", "gain_ci_upper"):
            params[k].attrs["units"] = "dimensionless"

        metadata["bootstrap_replicates"] = bootstrap_replicates
        metadata["bootstrap_realisations"] = realisations
        metadata["bootstrap_ci_method"] = ci_method
        metadata["bootstrap_ci_level"] = ci_level
        metadata["bootstrap_n_failed"] = int(
            np.sum(np.isnan(bootstrap_replicates["rms_misfit"]))
        )
        metadata["bootstrap_caveats"] = (
            "Parametric bootstrap CIs assume the GB89 forward model "
            "and the per-component Gaussian noise model are correct. "
            "Per Chave (2014), CIs derived via phase-tensor "
            "invariants or Mohr-circle quantities are formally "
            "meaningless. Per-station regional Z bootstrap CIs are "
            "not stored on the Dataset in this session; the per-"
            "replica primary-mode parameters are available in "
            "metadata['bootstrap_replicates']."
        )
        warning_fraction = float(np.mean(bootstrap_replicates["mode_warning"]))
        metadata["bootstrap_mode_warning_fraction"] = warning_fraction

    options_record = {
        "periods": periods,
        "bandwidth": bandwidth,
        "overlap": overlap,
        "norm_type": norm_type,
        "bounds_override": bounds_override,
        "initial_guess": initial_guess,
        "realisations": realisations,
        "seed": seed,
        "n_starts": n_starts,
        "mode_tolerance": mode_tolerance,
        "return_all_modes": return_all_modes,
        "mode_warning_threshold": mode_warning_threshold,
        "perturbation_scale": perturbation_scale,
        "ci_level": ci_level,
        "ci_method": ci_method,
        "strike_sharing": strike_sharing,
        "mode_clustering": mode_clustering,
        "regional_z_format": regional_z_format,
    }

    return DecompositionResult(
        parameters=params,
        regional_z=regional_z,
        chi_squared=chi_squared_da,
        rms_misfit=rms_misfit,
        method="mcneice_jones_joint",
        options=options_record,
        metadata=metadata,
        frame="measurement",
    )

def decompose_each_station(
    collection_or_list: "MTCollection | list[MT]",
    *,
    skip_failed: bool = True,
    **decompose_kwargs,
) -> dict[str, DecompositionResult]:
    """Run single-site :func:`decompose` per station; return as dict.

    For surveys where per-station period grids differ enough that
    :func:`decompose_joint` cannot run (the common case for real
    data — see ``~/MT_Decomp/findings/real_data_exploration.md``),
    this function provides the per-station equivalent. Each station's
    impedance tensor is decomposed independently using
    :func:`decompose`; results are returned in a dict keyed by
    station identifier.

    The plotting module accepts ``dict[station_id,
    DecompositionResult]`` directly for multi-station overlay plots,
    so this is the natural input path for plot functions when joint
    fitting isn't usable.

    Parameters
    ----------
    collection_or_list : MTCollection or list of MT
        Input stations. Same accepted types as :func:`decompose_joint`.
    skip_failed : bool, default True
        If True, stations whose :func:`decompose` raises are logged
        and skipped (excluded from the result dict). If False, the
        first failure raises.
    **decompose_kwargs
        Forwarded to :func:`decompose` for each station. Common
        choices: ``n_starts``, ``realisations``, ``mode_tolerance``,
        ``seed``.

    Returns
    -------
    dict
        Mapping from station identifier (``mt.station``) to
        :class:`DecompositionResult`.

    Raises
    ------
    ValueError
        If no stations are provided.

    Notes
    -----
    Cost scales as ``n_stations × decompose() cost``. On real BBMT
    data this is typically 5-30 s per station with default settings.
    With bootstrap (``realisations=100``), 1-5 minutes per station.
    Long-running scripts should save the result via
    :meth:`DecompositionResult.save` for later plotting rather than
    re-running.
    """
    stations = _normalise_collection_input(collection_or_list)
    if not stations:
        raise ValueError("decompose_each_station: no stations provided")

    results: dict[str, DecompositionResult] = {}
    failed: list[tuple[str, str]] = []

    for station_id, z in stations:
        try:
            result = decompose(z, **decompose_kwargs)
            results[station_id] = result
        except Exception as exc:
            if skip_failed:
                logger.warning(
                    f"decompose_each_station: station {station_id!r} "
                    f"failed: {type(exc).__name__}: {exc}"
                )
                failed.append((station_id, str(exc)))
            else:
                raise

    if failed:
        logger.info(
            f"decompose_each_station: {len(failed)} of {len(stations)} "
            f"stations failed; {len(results)} succeeded"
        )

    return results


# ---------------------------------------------------------------------------
# Private kernels: numerical building blocks for the GB89 / MJ01
# decomposition. None of these are part of the public API; they are
# called only by the (forthcoming) banded-decomposition driver and by
# the cross-validation tests against the Fortran reference. All are
# pure-NumPy implementations of the GB89 specification, not
# transcriptions of the Fortran reference.

def _objfun(
    x: np.ndarray,
    z_obs: np.ndarray,
    sigma: np.ndarray,
    periods: np.ndarray,
    compute_jacobian: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Residuals and analytic Jacobian for single-band GB optimisation.

    The function :func:`scipy.optimize.least_squares` sees: given a
    flat parameter vector ``x``, return the per-observation residuals
    and (optionally) the analytic Jacobian.

    Parameters
    ----------
    x : (5 + 4*n_freqs,) float ndarray
        Flat parameter vector. See :func:`_unpack_x` for the
        unpacking convention.
    z_obs : (n_freqs, 2, 2) complex ndarray
        Observed impedance tensors per frequency, in SI units.
    sigma : (n_freqs, 2, 2) float ndarray
        Per-component standard errors. Must be strictly positive.
    periods : (n_freqs,) float ndarray
        Periods in seconds, matching ``z_obs`` along axis 0.
    compute_jacobian : bool, default True
        If True, return the analytic Jacobian as the second element
        of the return tuple. If False, return ``None`` for the
        Jacobian and skip the (expensive) computation.

    Returns
    -------
    residuals : (8*n_freqs,) float ndarray
        Per-observation residuals, packed as
        ``[Re(dZ_xx)/sigma, Im(dZ_xx)/sigma,
        Re(dZ_xy)/sigma, Im(dZ_xy)/sigma,
        Re(dZ_yx)/sigma, Im(dZ_yx)/sigma,
        Re(dZ_yy)/sigma, Im(dZ_yy)/sigma]``
        per frequency, in frequency order.
    jacobian : (8*n_freqs, 5+4*n_freqs) float ndarray, or None
        Analytic Jacobian if ``compute_jacobian``, else ``None``.

    Raises
    ------
    ValueError
        If ``z_obs.shape != (len(periods), 2, 2)``,
        ``sigma.shape != z_obs.shape``,
        ``len(x) != 5 + 4*len(periods)``, or ``sigma`` contains
        non-positive entries.

    Notes
    -----
    The gain parameter ``log10_gain`` enters as a multiplicative
    factor ``g = 10**log10_gain`` on the predicted impedance:
    ``Z_pred = g * R(theta) C T(t) S(e) Z_2D R(theta).T``. This
    matches the strike_py convention where gain is identifiable
    only up to the regional impedance amplitudes (the static-shift
    ambiguity).

    The anisotropy parameter ``s`` is structurally non-identifiable
    from MT data alone (see CLAUDE.md invariant 4). It is included
    in the parameter vector so that bounds and external
    constraints can act on it, but the Jacobian column for ``s``
    is exactly zero. This is documented behaviour, not a bug.

    Convention divergence from the Fortran reference. The
    ``strike_py`` Fortran ``objfun`` adapter
    (``objfun_wrapped``) parameterises the problem differently
    from this Python implementation: it stores impedances
    directly as ``(re(a), im(a), re(b), im(b))`` per frequency,
    has no gain or anisotropy parameter, and computes residuals
    in alpha-space (Pauli-spin combinations) with a non-standard
    sigma weighting. Because of this, element-wise comparison of
    residual vectors or Jacobians between the two implementations
    is not meaningful. The forward-model kernel ``_estim_imp`` is
    cross-validated against Fortran in
    ``tests/cross_validation/test_kernels_against_fortran.py`` --
    that is the continuity link.
    """
    n_freqs = len(periods)

    if z_obs.shape != (n_freqs, 2, 2):
        raise ValueError(
            f"_objfun: z_obs shape {z_obs.shape} does not match "
            f"(n_freqs, 2, 2) = ({n_freqs}, 2, 2)"
        )
    if sigma.shape != z_obs.shape:
        raise ValueError(
            f"_objfun: sigma shape {sigma.shape} does not match "
            f"z_obs shape {z_obs.shape}"
        )
    if x.size != 5 + 4 * n_freqs:
        raise ValueError(
            f"_objfun: x size {x.size} does not match "
            f"5 + 4*n_freqs = {5 + 4 * n_freqs}"
        )
    if np.any(sigma <= 0):
        raise ValueError("_objfun: sigma contains non-positive entries")

    (
        theta,
        twist,
        shear,
        log10_gain,
        _anisotropy,
        log10_rho_a,
        phase_a,
        log10_rho_b,
        phase_b,
    ) = _unpack_x(x, n_freqs)

    t = np.tan(twist)
    e = np.tan(shear)
    sec2_t = 1.0 / np.cos(twist) ** 2
    sec2_e = 1.0 / np.cos(shear) ** 2
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    gain = 10.0**log10_gain
    ln10 = np.log(10.0)

    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0

    # Per-frequency complex impedances a, b from (rho, phase).
    # |a|^2 * T = rho * factor, so |a| = sqrt(rho * factor / T).
    rho_a = 10.0**log10_rho_a
    rho_b = 10.0**log10_rho_b
    abs_a = np.sqrt(rho_a * factor / periods)
    abs_b = np.sqrt(rho_b * factor / periods)
    a = abs_a * np.exp(1j * phase_a)
    b = abs_b * np.exp(1j * phase_b)

    residuals = np.empty(8 * n_freqs, dtype=np.float64)

    if compute_jacobian:
        jacobian = np.zeros((8 * n_freqs, 5 + 4 * n_freqs), dtype=np.float64)
    else:
        jacobian = None

    for k in range(n_freqs):
        a_k = a[k]
        b_k = b[k]

        alpha0 = -b_k * (e - t) + a_k * (t + e)
        alpha1 = (
            s2 * (-b_k * (e - t))
            + c2 * (-b_k * (1.0 + t * e))
            + c2 * (a_k * (1.0 - e * t))
            - s2 * (a_k * (t + e))
        )
        alpha2 = -b_k * (1.0 + t * e) - a_k * (1.0 - e * t)
        alpha3 = (
            c2 * (-b_k * (e - t))
            - s2 * (-b_k * (1.0 + t * e))
            - s2 * (a_k * (1.0 - e * t))
            - c2 * (a_k * (t + e))
        )

        z_pred = np.empty((2, 2), dtype=np.complex128)
        z_pred[0, 0] = (alpha0 + alpha3) / 2.0
        z_pred[1, 1] = (alpha0 - alpha3) / 2.0
        z_pred[0, 1] = (alpha1 - alpha2) / 2.0
        z_pred[1, 0] = (alpha1 + alpha2) / 2.0
        z_pred = gain * z_pred

        diff = z_pred - z_obs[k]
        s_k = sigma[k]
        residuals[8 * k + 0] = diff[0, 0].real / s_k[0, 0]
        residuals[8 * k + 1] = diff[0, 0].imag / s_k[0, 0]
        residuals[8 * k + 2] = diff[0, 1].real / s_k[0, 1]
        residuals[8 * k + 3] = diff[0, 1].imag / s_k[0, 1]
        residuals[8 * k + 4] = diff[1, 0].real / s_k[1, 0]
        residuals[8 * k + 5] = diff[1, 0].imag / s_k[1, 0]
        residuals[8 * k + 6] = diff[1, 1].real / s_k[1, 1]
        residuals[8 * k + 7] = diff[1, 1].imag / s_k[1, 1]

        if not compute_jacobian:
            continue

        # ------------------------------------------------------------------
        # Partial derivatives. We compute d(alpha_i)/d(parameter) for
        # each parameter, then build d(z_pred)/d(parameter) via the
        # same Pauli-spin inversion used in the forward pass, scale by
        # gain, and pack into Jacobian columns divided by sigma.
        #
        # d(c2)/d(theta) = -2*s2; d(s2)/d(theta) = +2*c2.
        # alpha0 and alpha2 have no theta dependence.
        # ------------------------------------------------------------------
        d_alpha1_dtheta = (
            (2.0 * c2) * (-b_k * (e - t))
            + (-2.0 * s2) * (-b_k * (1.0 + t * e))
            + (-2.0 * s2) * (a_k * (1.0 - e * t))
            - (2.0 * c2) * (a_k * (t + e))
        )
        d_alpha3_dtheta = (
            (-2.0 * s2) * (-b_k * (e - t))
            - (2.0 * c2) * (-b_k * (1.0 + t * e))
            - (2.0 * c2) * (a_k * (1.0 - e * t))
            - (-2.0 * s2) * (a_k * (t + e))
        )

        # d(alpha)/d(t) (twist's tangent). Multiply by sec2_t for
        # the chain through twist itself.
        d_alpha0_dt = -b_k * (-1.0) + a_k * (1.0)
        d_alpha1_dt = (
            s2 * (-b_k * (-1.0))
            + c2 * (-b_k * e)
            + c2 * (a_k * (-e))
            - s2 * (a_k * 1.0)
        )
        d_alpha2_dt = -b_k * e - a_k * (-e)
        d_alpha3_dt = (
            c2 * (-b_k * (-1.0))
            - s2 * (-b_k * e)
            - s2 * (a_k * (-e))
            - c2 * (a_k * 1.0)
        )

        # d(alpha)/d(e) (shear's tangent). Multiply by sec2_e.
        d_alpha0_de = -b_k * 1.0 + a_k * 1.0
        d_alpha1_de = (
            s2 * (-b_k * 1.0) + c2 * (-b_k * t) + c2 * (a_k * (-t)) - s2 * (a_k * 1.0)
        )
        d_alpha2_de = -b_k * t - a_k * (-t)
        d_alpha3_de = (
            c2 * (-b_k * 1.0) - s2 * (-b_k * t) - s2 * (a_k * (-t)) - c2 * (a_k * 1.0)
        )

        # alpha is linear in a and b. Coefficients used by the
        # rho/phase chain rule below.
        coef_a_alpha0 = t + e
        coef_a_alpha1 = c2 * (1.0 - e * t) - s2 * (t + e)
        coef_a_alpha2 = -(1.0 - e * t)
        coef_a_alpha3 = -s2 * (1.0 - e * t) - c2 * (t + e)
        coef_b_alpha0 = -(e - t)
        coef_b_alpha1 = -s2 * (e - t) - c2 * (1.0 + t * e)
        coef_b_alpha2 = -(1.0 + t * e)
        coef_b_alpha3 = -c2 * (e - t) + s2 * (1.0 + t * e)

        def z_from_dalpha(d0, d1, d2, d3):
            """Build dZ/dx (2x2 complex) from dalpha_i/dx values via
            the same Pauli-spin inversion as in the forward pass.
            Multiplied by gain because gain enters z_pred linearly.
            """
            dz = np.empty((2, 2), dtype=np.complex128)
            dz[0, 0] = (d0 + d3) / 2.0
            dz[1, 1] = (d0 - d3) / 2.0
            dz[0, 1] = (d1 - d2) / 2.0
            dz[1, 0] = (d1 + d2) / 2.0
            return gain * dz

        dz_dtheta = z_from_dalpha(0.0, d_alpha1_dtheta, 0.0, d_alpha3_dtheta)
        dz_dtwist = sec2_t * z_from_dalpha(
            d_alpha0_dt, d_alpha1_dt, d_alpha2_dt, d_alpha3_dt
        )
        dz_dshear = sec2_e * z_from_dalpha(
            d_alpha0_de, d_alpha1_de, d_alpha2_de, d_alpha3_de
        )
        dz_dlog10_gain = z_pred * ln10

        # da/d(log10_rho_a) = a * ln(10) / 2 (since rho ~ |a|^2 and
        # log10_rho enters via |a| = sqrt(10^log10_rho * ...)).
        # da/d(phase_a) = i * a.
        da_dlog10_rho = a_k * ln10 / 2.0
        da_dphase = 1j * a_k
        dz_dlog10_rho_a = z_from_dalpha(
            coef_a_alpha0 * da_dlog10_rho,
            coef_a_alpha1 * da_dlog10_rho,
            coef_a_alpha2 * da_dlog10_rho,
            coef_a_alpha3 * da_dlog10_rho,
        )
        dz_dphase_a = z_from_dalpha(
            coef_a_alpha0 * da_dphase,
            coef_a_alpha1 * da_dphase,
            coef_a_alpha2 * da_dphase,
            coef_a_alpha3 * da_dphase,
        )

        db_dlog10_rho = b_k * ln10 / 2.0
        db_dphase = 1j * b_k
        dz_dlog10_rho_b = z_from_dalpha(
            coef_b_alpha0 * db_dlog10_rho,
            coef_b_alpha1 * db_dlog10_rho,
            coef_b_alpha2 * db_dlog10_rho,
            coef_b_alpha3 * db_dlog10_rho,
        )
        dz_dphase_b = z_from_dalpha(
            coef_b_alpha0 * db_dphase,
            coef_b_alpha1 * db_dphase,
            coef_b_alpha2 * db_dphase,
            coef_b_alpha3 * db_dphase,
        )

        def pack_dz(dz):
            """Pack a (2,2) complex partial into 8 sigma-weighted
            real entries matching the residuals layout."""
            return np.array(
                [
                    dz[0, 0].real / s_k[0, 0],
                    dz[0, 0].imag / s_k[0, 0],
                    dz[0, 1].real / s_k[0, 1],
                    dz[0, 1].imag / s_k[0, 1],
                    dz[1, 0].real / s_k[1, 0],
                    dz[1, 0].imag / s_k[1, 0],
                    dz[1, 1].real / s_k[1, 1],
                    dz[1, 1].imag / s_k[1, 1],
                ]
            )

        row_start = 8 * k
        row_end = 8 * (k + 1)
        jacobian[row_start:row_end, 0] = pack_dz(dz_dtheta)
        jacobian[row_start:row_end, 1] = pack_dz(dz_dtwist)
        jacobian[row_start:row_end, 2] = pack_dz(dz_dshear)
        jacobian[row_start:row_end, 3] = pack_dz(dz_dlog10_gain)
        # column 4 (anisotropy) stays zero; non-identifiable.
        jacobian[row_start:row_end, 5 + 0 * n_freqs + k] = pack_dz(dz_dlog10_rho_a)
        jacobian[row_start:row_end, 5 + 1 * n_freqs + k] = pack_dz(dz_dphase_a)
        jacobian[row_start:row_end, 5 + 2 * n_freqs + k] = pack_dz(dz_dlog10_rho_b)
        jacobian[row_start:row_end, 5 + 3 * n_freqs + k] = pack_dz(dz_dphase_b)

    return residuals, jacobian


# ---------------------------------------------------------------------------
# Banded driver and Z<->band-array translation. These are the pieces that
# turn _objfun into a working public decompose() entrypoint. Private; the
# public API is decompose() above.

def _build_bounds(
    n_freqs: int,
    bounds_override: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Default parameter bounds for the GB state vector.

    Defaults (radians, log10 units):
      - strike:        ``[0, pi)``
      - twist:         ``[-pi/4, pi/4]``
      - shear:         ``[-pi/4, pi/4]``
      - log10_gain:    ``[-2, 2]``
      - anisotropy:    ``[-2, 2]`` (immaterial; J-column is zero)
      - log10_rho_a/b: ``[-3, 6]``
      - phase_a/b:     ``[-pi/4, 3*pi/4]``

    Phase bounds are wider than the textbook causal first quadrant
    so that real data near the high-frequency edge (where phases dip
    below 0 deg) and processing-convention shifts don't pin the
    solution at the boundary.

    Overrides accept the keys listed above as ``(lower, upper)``
    tuples.

    Returns
    -------
    lower, upper : (5 + 4*n_freqs,) float64
    """
    PI_4 = np.pi / 4.0
    defaults = {
        "strike": (0.0, np.pi),
        "twist": (-PI_4, PI_4),
        "shear": (-PI_4, PI_4),
        "log10_gain": (-2.0, 2.0),
        "anisotropy": (-2.0, 2.0),
        "log10_rho_a": (-3.0, 6.0),
        "phase_a": (-PI_4, 3.0 * PI_4),
        "log10_rho_b": (-3.0, 6.0),
        "phase_b": (-PI_4, 3.0 * PI_4),
    }
    if bounds_override:
        for key, val in bounds_override.items():
            if key not in defaults:
                raise ValueError(f"_build_bounds: unknown override key {key!r}")
            defaults[key] = val

    lower = np.empty(5 + 4 * n_freqs)
    upper = np.empty(5 + 4 * n_freqs)
    lower[0], upper[0] = defaults["strike"]
    lower[1], upper[1] = defaults["twist"]
    lower[2], upper[2] = defaults["shear"]
    lower[3], upper[3] = defaults["log10_gain"]
    lower[4], upper[4] = defaults["anisotropy"]

    base = 5
    lower[base : base + n_freqs] = defaults["log10_rho_a"][0]
    upper[base : base + n_freqs] = defaults["log10_rho_a"][1]
    lower[base + n_freqs : base + 2 * n_freqs] = defaults["phase_a"][0]
    upper[base + n_freqs : base + 2 * n_freqs] = defaults["phase_a"][1]
    lower[base + 2 * n_freqs : base + 3 * n_freqs] = defaults["log10_rho_b"][0]
    upper[base + 2 * n_freqs : base + 3 * n_freqs] = defaults["log10_rho_b"][1]
    lower[base + 3 * n_freqs : base + 4 * n_freqs] = defaults["phase_b"][0]
    upper[base + 3 * n_freqs : base + 4 * n_freqs] = defaults["phase_b"][1]
    return lower, upper

def _solve_band(
    z_obs: np.ndarray,
    sigma: np.ndarray,
    periods: np.ndarray,
    x0: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    bounds_override: dict | None = None,
    max_nfev: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
) -> _BandResult:
    """Solve the GB optimisation for one period band.

    Wraps :func:`scipy.optimize.least_squares` with the analytic
    Jacobian computed by :func:`_objfun`. Method is TRF
    (Trust-Region Reflective), which respects bounds.

    Parameters
    ----------
    z_obs, sigma, periods : ndarrays
        Per-band observation arrays. See :func:`_objfun`.
    x0 : (5 + 4*n_freqs,) float64, optional
        Initial parameter vector. Default: :func:`_canonical_initial_guess`,
        clipped into the bounds.
    bounds : tuple of (lower, upper), optional
        Parameter bounds. Default: :func:`_build_bounds`.
    bounds_override : dict, optional
        Per-parameter bounds overrides; passed to
        :func:`_build_bounds` if ``bounds`` is None.
    max_nfev : int
    ftol, xtol : float

    Returns
    -------
    _BandResult
    """
    n_freqs = len(periods)

    if bounds is None:
        bounds = _build_bounds(n_freqs, bounds_override)
    lower, upper = bounds

    if x0 is None:
        x0 = _canonical_initial_guess(z_obs, sigma, periods)
    # Clip x0 into the bounds: scipy's TRF rejects out-of-bounds x0.
    x0 = np.clip(x0, lower, upper)

    def fun(x):
        r, _ = _objfun(x, z_obs, sigma, periods, compute_jacobian=False)
        return r

    def jac(x):
        _, j = _objfun(x, z_obs, sigma, periods, compute_jacobian=True)
        return j

    result = least_squares(
        fun=fun,
        x0=x0,
        jac=jac,
        bounds=(lower, upper),
        method="trf",
        ftol=ftol,
        xtol=xtol,
        max_nfev=max_nfev,
    )

    x_opt = result.x
    residuals = result.fun
    chi_squared = float(np.sum(residuals**2))
    n_resid = len(residuals)
    rms_misfit = float(np.sqrt(chi_squared / n_resid))

    # Parameter errors from the final Jacobian.
    # cov = (J^T J)^{-1}; sigma_i = sqrt(diag(cov)). Detect zero-norm
    # columns explicitly so structurally non-identifiable parameters
    # report inf (truthful) rather than 0 (misleading).
    J = result.jac
    col_norms = np.linalg.norm(J, axis=0)
    zero_cols = col_norms < 1e-15
    try:
        cov = np.linalg.pinv(J.T @ J, rcond=1e-12)
        x_err = np.sqrt(np.maximum(np.diag(cov), 0.0))
        x_err[zero_cols] = np.inf
    except np.linalg.LinAlgError:
        x_err = np.full_like(x_opt, np.nan)

    return _BandResult(
        x_opt=x_opt,
        x_err=x_err,
        residuals=residuals,
        chi_squared=chi_squared,
        rms_misfit=rms_misfit,
        n_iter=int(result.nfev),
        converged=int(result.status) > 0,
        cost_at_opt=float(result.cost),
        jacobian=J,
    )

def _generate_starting_points(
    z_obs: np.ndarray,
    sigma: np.ndarray,
    periods: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
    perturbation_scale: float = 0.1,
) -> list[np.ndarray]:
    """Generate hybrid starting points for multi-start optimisation.

    Strategy:
      - Position 0: canonical phase-tensor guess (zero distortion).
      - Position 1: 90-rotated guess (strike + pi/2 with TE/TM
        swap), if ``n_starts >= 2``.
      - Positions 2..n_starts-1: random Gaussian perturbations of
        the canonical guess, scaled to the per-parameter bound
        width.

    All starts are clipped into the bounds.

    Parameters
    ----------
    z_obs, sigma, periods : ndarrays
        Single-band observation arrays.
    lower, upper : (n_params,) ndarrays
        Parameter bounds.
    n_starts : int
        Number of starting points. Must be >= 1.
    rng : np.random.Generator
        Used only for the random perturbations.
    perturbation_scale : float, default 0.1

    Returns
    -------
    list of (n_params,) ndarrays
        Length ``n_starts``.
    """
    if n_starts < 1:
        raise ValueError(
            f"_generate_starting_points: n_starts must be >= 1, " f"got {n_starts}"
        )

    canonical = np.clip(_canonical_initial_guess(z_obs, sigma, periods), lower, upper)
    starts = [canonical]

    if n_starts >= 2:
        rotated = np.clip(_rotated_initial_guess(z_obs, sigma, periods), lower, upper)
        starts.append(rotated)

    for _ in range(n_starts - 2):
        starts.append(
            _perturbed_initial_guess(canonical, lower, upper, rng, perturbation_scale)
        )

    return starts


def _solve_band_multistart(
    z_obs: np.ndarray,
    sigma: np.ndarray,
    periods: np.ndarray,
    n_starts: int = 5,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    bounds_override: dict | None = None,
    rng: np.random.Generator | None = None,
    mode_tolerance: dict | None = None,
    perturbation_scale: float = 0.1,
    max_nfev: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
) -> list[_Mode]:
    """Multi-start single-band optimisation.

    Generates ``n_starts`` initial guesses via
    :func:`_generate_starting_points`, runs :func:`_solve_band`
    from each, then clusters the converged results into modes via
    :func:`_cluster_modes`. Returns the modes sorted by RMS misfit
    with mode probabilities populated.

    Parameters
    ----------
    z_obs, sigma, periods : ndarrays
        Single-band observation arrays.
    n_starts : int, default 5
    bounds : tuple, optional
        Pre-computed (lower, upper) bounds; computed if None.
    bounds_override : dict, optional
        Passed to :func:`_build_bounds` when computing bounds.
    rng : np.random.Generator, optional
        Reproducibility. Defaults to ``np.random.default_rng(42)``.
    mode_tolerance : dict, optional
        Per-parameter tolerances for clustering.
    perturbation_scale : float, default 0.1
    max_nfev, ftol, xtol : optimiser parameters.

    Returns
    -------
    list of _Mode
        Sorted by RMS misfit. Mode 0 is primary. Mode probabilities
        are populated on ``mode.probability``.

    Raises
    ------
    RuntimeError
        If every starting point's optimisation fails.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_freqs = len(periods)
    if bounds is None:
        bounds = _build_bounds(n_freqs, bounds_override)
    lower, upper = bounds

    starting_points = _generate_starting_points(
        z_obs,
        sigma,
        periods,
        lower,
        upper,
        n_starts,
        rng,
        perturbation_scale=perturbation_scale,
    )

    band_results: list[_BandResult] = []
    for x0 in starting_points:
        try:
            br = _solve_band(
                z_obs=z_obs,
                sigma=sigma,
                periods=periods,
                x0=x0,
                bounds=bounds,
                max_nfev=max_nfev,
                ftol=ftol,
                xtol=xtol,
            )
            band_results.append(br)
        except Exception:
            continue

    if not band_results:
        raise RuntimeError("_solve_band_multistart: all starting points failed")

    modes = _cluster_modes(band_results, mode_tolerance)
    probabilities = _compute_mode_probabilities(modes)
    for mode, prob in zip(modes, probabilities):
        mode.probability = prob

    return modes

def _decompose_bands_with_modes(
    z_obs_full: np.ndarray,
    sigma_full: np.ndarray,
    selected_periods: np.ndarray,
    bands: list[np.ndarray],
    n_starts: int,
    bounds_override: dict | None,
    rng: np.random.Generator,
    mode_tolerance: dict | None,
    perturbation_scale: float,
) -> list[tuple[np.ndarray, list[_Mode]]]:
    """Run multi-start optimisation per band; return per-band mode lists.

    Used both by :func:`decompose` on the original data and by
    :func:`_bootstrap_decompose` per replica. Pure orchestrator: each
    band is independently optimised via :func:`_solve_band_multistart`.

    Parameters
    ----------
    z_obs_full : (n_periods, 2, 2) complex128
    sigma_full : (n_periods, 2, 2) float64
    selected_periods : (n_periods,) float64
    bands : list of int ndarrays
        Output of :func:`_extract_bands`.
    n_starts, bounds_override, rng, mode_tolerance, perturbation_scale
        Passed through to :func:`_solve_band_multistart`.

    Returns
    -------
    list of (band_idx, list[_Mode])
    """
    band_modes_list: list[tuple[np.ndarray, list[_Mode]]] = []
    for band_idx in bands:
        z_obs_b = z_obs_full[band_idx]
        sigma_b = sigma_full[band_idx]
        periods_b = selected_periods[band_idx]
        modes = _solve_band_multistart(
            z_obs=z_obs_b,
            sigma=sigma_b,
            periods=periods_b,
            n_starts=n_starts,
            bounds_override=bounds_override,
            rng=rng,
            mode_tolerance=mode_tolerance,
            perturbation_scale=perturbation_scale,
        )
        band_modes_list.append((band_idx, modes))
    return band_modes_list

def _build_per_mode_parameters_dataset(
    band_modes_list: list[tuple[np.ndarray, list[_Mode]]],
    selected_periods: np.ndarray,
) -> xr.Dataset:
    """Build a ``(period, mode)``-shaped parameters Dataset.

    For each band, each discovered mode contributes its (canonical-
    form) band-level scalar parameters to all periods that band
    covers, indexed by the mode's rank within the band (0 = primary).
    Modes that don't exist for a given band are filled with NaN.

    Overlapping bands: a later band's modes overwrite earlier bands'
    modes at shared periods. This is acceptable for v1; if it
    becomes an issue, add an explicit band index to the Dataset.

    Parameters
    ----------
    band_modes_list : list of (band_idx, list[_Mode])
    selected_periods : (n_periods,) float64

    Returns
    -------
    xr.Dataset
        Same variables as the primary-mode Dataset (strike, twist,
        shear, gain, anisotropy, plus errors), with shape
        ``(n_periods, max_modes)``.
    """
    n_periods = len(selected_periods)
    max_modes = max((len(modes) for _, modes in band_modes_list), default=1)

    shape = (n_periods, max_modes)
    strike = np.full(shape, np.nan)
    twist = np.full(shape, np.nan)
    shear = np.full(shape, np.nan)
    gain = np.full(shape, np.nan)
    aniso = np.full(shape, np.nan)
    strike_err = np.full(shape, np.nan)
    twist_err = np.full(shape, np.nan)
    shear_err = np.full(shape, np.nan)
    gain_err = np.full(shape, np.nan)
    aniso_err = np.full(shape, np.nan)

    for band_idx, modes in band_modes_list:
        for mode_i, mode in enumerate(modes):
            br = mode.band_result
            cf = mode.canonical_form
            log10_gain = float(br.x_opt[3])
            gain_value = 10.0**log10_gain
            log10_gain_err = float(br.x_err[3])
            gain_value_err = (
                gain_value * np.log(10.0) * log10_gain_err
                if np.isfinite(log10_gain_err)
                else np.inf
            )
            for global_i in band_idx:
                strike[global_i, mode_i] = cf["strike_deg"]
                twist[global_i, mode_i] = cf["twist_deg"]
                shear[global_i, mode_i] = cf["shear_deg"]
                gain[global_i, mode_i] = gain_value
                aniso[global_i, mode_i] = 0.0
                strike_err[global_i, mode_i] = float(np.degrees(br.x_err[0]))
                twist_err[global_i, mode_i] = float(np.degrees(br.x_err[1]))
                shear_err[global_i, mode_i] = float(np.degrees(br.x_err[2]))
                gain_err[global_i, mode_i] = gain_value_err

    ds = xr.Dataset(
        {
            "strike": (("period", "mode"), strike),
            "twist": (("period", "mode"), twist),
            "shear": (("period", "mode"), shear),
            "gain": (("period", "mode"), gain),
            "anisotropy": (("period", "mode"), aniso),
            "strike_error": (("period", "mode"), strike_err),
            "twist_error": (("period", "mode"), twist_err),
            "shear_error": (("period", "mode"), shear_err),
            "gain_error": (("period", "mode"), gain_err),
            "anisotropy_error": (("period", "mode"), aniso_err),
        },
        coords={
            "period": selected_periods,
            "mode": np.arange(max_modes),
        },
    )
    ds["strike"].attrs.update(
        units="degrees", range="[0, 90)", convention="clockwise from x-axis"
    )
    ds["twist"].attrs.update(units="degrees")
    ds["shear"].attrs.update(units="degrees")
    ds["gain"].attrs.update(units="dimensionless")
    ds["anisotropy"].attrs.update(units="dimensionless", non_identifiable=True)
    ds["period"].attrs.update(units="seconds")
    ds["mode"].attrs.update(
        description="Discovered mode index per band; 0 = primary (lowest RMS)"
    )
    return ds

def _predict_z_from_primary_modes(
    band_modes_list: list[tuple[np.ndarray, list[_Mode]]],
    selected_periods: np.ndarray,
) -> np.ndarray:
    """Build the model-predicted Z from per-band primary modes.

    For each band, the primary mode's converged parameters specify
    a forward model. We evaluate that model at each of the band's
    periods to produce the per-period predicted tensor. Where periods
    appear in multiple bands (overlap > 0), the prediction comes
    from the band whose primary mode has the lowest RMS at that
    period.

    Parameters
    ----------
    band_modes_list : list of (band_idx, modes)
    selected_periods : (n_periods,) float64

    Returns
    -------
    z_predicted : (n_periods, 2, 2) complex128
        Measurement-frame predicted tensor.

    Raises
    ------
    RuntimeError
        If any selected period is not covered by at least one band.
    """
    n_periods = len(selected_periods)
    z_predicted = np.full((n_periods, 2, 2), np.nan, dtype=np.complex128)
    best_rms = np.full(n_periods, np.inf)

    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0

    for band_idx, modes in band_modes_list:
        primary = modes[0]
        x_opt = primary.band_result.x_opt
        n_band_freqs = len(band_idx)
        band_periods = selected_periods[band_idx]

        unpacked = _unpack_x(x_opt, n_band_freqs)
        theta = unpacked[0]
        twist = unpacked[1]
        shear = unpacked[2]
        log10_gain = unpacked[3]
        log10_rho_a = unpacked[5]
        phase_a = unpacked[6]
        log10_rho_b = unpacked[7]
        phase_b = unpacked[8]
        t = np.tan(twist)
        e = np.tan(shear)
        gain = 10.0**log10_gain
        rho_a = 10.0**log10_rho_a
        rho_b = 10.0**log10_rho_b
        abs_a = np.sqrt(rho_a * factor / band_periods)
        abs_b = np.sqrt(rho_b * factor / band_periods)
        a = abs_a * np.exp(1j * phase_a)
        b = abs_b * np.exp(1j * phase_b)

        for local_i, global_i in enumerate(band_idx):
            if primary.rms_misfit < best_rms[global_i]:
                z_predicted[global_i] = gain * _estim_imp(
                    a[local_i], b[local_i], t, e, theta
                )
                best_rms[global_i] = primary.rms_misfit

    if np.any(np.isnan(z_predicted)):
        raise RuntimeError(
            "_predict_z_from_primary_modes: some periods not covered "
            "by any band's primary mode prediction"
        )
    return z_predicted

def _bootstrap_decompose(
    z_obs_full: np.ndarray,
    sigma_full: np.ndarray,
    selected_periods: np.ndarray,
    bands: list[np.ndarray],
    z_predicted: np.ndarray,
    realisations: int,
    n_starts: int,
    bounds_override: dict | None,
    rng: np.random.Generator,
    mode_tolerance: dict | None,
    perturbation_scale: float,
    mode_warning_threshold: float,
) -> dict[str, np.ndarray]:
    """Run parametric bootstrap; return per-replica primary-mode arrays.

    For each of ``realisations`` replicas:
      1. Generate a synthetic Z via :func:`_resample_residuals`.
      2. Run multi-start optimisation per band on the synthetic Z.
      3. Extract per-period primary-mode canonical-form parameters
         and per-period regional impedance components (rotated to
         measurement frame).

    Failed replicas (uncaught exceptions inside the per-band fit)
    leave the corresponding row as NaN in every output array; the
    caller can count failures via ``np.isnan(rms_misfit)``.

    Parameters
    ----------
    z_obs_full : (n_periods, 2, 2) complex128
    sigma_full : (n_periods, 2, 2) float64
    selected_periods : (n_periods,) float64
    bands : list of int ndarrays
    z_predicted : (n_periods, 2, 2) complex128
        Model prediction at the original-data primary modes.
    realisations : int, >= 1
    n_starts : int
    bounds_override : dict or None
    rng : np.random.Generator
    mode_tolerance : dict or None
    perturbation_scale : float
    mode_warning_threshold : float
        Per-replica primary-mode-warning is recorded when any band's
        second-best mode has RMS within this factor of its primary's
        RMS. Passed through from ``decompose``.

    Returns
    -------
    dict with keys:
        'strike'      : (realisations, n_periods) float, degrees
        'twist'       : (realisations, n_periods) float, degrees
        'shear'       : (realisations, n_periods) float, degrees
        'gain'        : (realisations, n_periods) float, dimensionless
        'regional_z'  : (realisations, n_periods, 2, 2) complex128
        'rms_misfit'  : (realisations,) float, per-replica overall RMS
        'mode_warning': (realisations,) bool, per-replica warning flag
    """
    if realisations < 1:
        raise ValueError(
            f"_bootstrap_decompose: realisations must be >= 1, " f"got {realisations}"
        )

    n_periods = len(selected_periods)
    replicates = {
        "strike": np.full((realisations, n_periods), np.nan),
        "twist": np.full((realisations, n_periods), np.nan),
        "shear": np.full((realisations, n_periods), np.nan),
        "gain": np.full((realisations, n_periods), np.nan),
        "regional_z": np.full(
            (realisations, n_periods, 2, 2), np.nan, dtype=np.complex128
        ),
        "rms_misfit": np.full(realisations, np.nan),
        "mode_warning": np.full(realisations, False),
    }

    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0

    for replica_i in range(realisations):
        z_replica = _resample_residuals(z_obs_full, sigma_full, z_predicted, rng)

        try:
            band_modes_list = _decompose_bands_with_modes(
                z_obs_full=z_replica,
                sigma_full=sigma_full,
                selected_periods=selected_periods,
                bands=bands,
                n_starts=n_starts,
                bounds_override=bounds_override,
                rng=rng,
                mode_tolerance=mode_tolerance,
                perturbation_scale=perturbation_scale,
            )
        except Exception:
            continue

        for band_idx, modes in band_modes_list:
            primary = modes[0]
            br = primary.band_result
            n_band_freqs = len(band_idx)

            strike_can, twist_can, shear_can = _geometric_fold(
                float(br.x_opt[0]), float(br.x_opt[1]), float(br.x_opt[2])
            )
            log10_gain = float(br.x_opt[3])

            unpacked = _unpack_x(br.x_opt, n_band_freqs)
            log10_rho_a = unpacked[5]
            phase_a = unpacked[6]
            log10_rho_b = unpacked[7]
            phase_b = unpacked[8]

            band_periods = selected_periods[band_idx]
            rho_a = 10.0**log10_rho_a
            rho_b = 10.0**log10_rho_b
            abs_a = np.sqrt(rho_a * factor / band_periods)
            abs_b = np.sqrt(rho_b * factor / band_periods)
            a_band = abs_a * np.exp(1j * phase_a)
            b_band = abs_b * np.exp(1j * phase_b)

            c, s = np.cos(strike_can), np.sin(strike_can)
            R = np.array([[c, -s], [s, c]])

            for local_i, global_i in enumerate(band_idx):
                replicates["strike"][replica_i, global_i] = np.degrees(strike_can)
                replicates["twist"][replica_i, global_i] = np.degrees(twist_can)
                replicates["shear"][replica_i, global_i] = np.degrees(shear_can)
                replicates["gain"][replica_i, global_i] = 10.0**log10_gain

                a_k = a_band[local_i]
                b_k = b_band[local_i]
                z_strike_frame = np.array(
                    [[0.0, a_k], [-b_k, 0.0]], dtype=np.complex128
                )
                replicates["regional_z"][replica_i, global_i] = R @ z_strike_frame @ R.T

        all_rms = np.array([modes[0].rms_misfit for _, modes in band_modes_list])
        replicates["rms_misfit"][replica_i] = float(np.sqrt(np.mean(all_rms**2)))

        for _, modes in band_modes_list:
            if len(modes) >= 2:
                ratio = modes[1].rms_misfit / max(modes[0].rms_misfit, 1e-30)
                if ratio <= mode_warning_threshold:
                    replicates["mode_warning"][replica_i] = True
                    break

    return replicates

def _bootstrap_decompose_joint(
    z_obs_per_site_full: np.ndarray,
    sigma_per_site_full: np.ndarray,
    selected_periods: np.ndarray,
    bands: list[np.ndarray],
    band_modes_list: list[tuple[np.ndarray, list[_Mode]]],
    realisations: int,
    n_starts: int,
    bounds_override: dict | None,
    rng: np.random.Generator,
    mode_tolerance: dict | None,
    perturbation_scale: float,
    mode_warning_threshold: float,
) -> dict[str, np.ndarray]:
    """Joint parametric bootstrap (minimal: strike/twist/shear/gain CIs).

    Generates per-site parametric noise around the original-data
    primary-mode prediction, runs joint multi-start per band per
    replica, extracts per-replica primary-mode canonical-form
    parameters. Per-station regional Z bootstrap CIs are not stored;
    the per-replica primary-mode parameters are sufficient for users
    to compute custom CIs offline.

    Parameters
    ----------
    z_obs_per_site_full : (n_sites, n_periods, 2, 2) complex128
    sigma_per_site_full : (n_sites, n_periods, 2, 2) float64
    selected_periods : (n_periods,) float64
    bands : list of int ndarrays
    band_modes_list : list of (band_idx, modes) from the original-
        data joint fit; used to build per-replica predictions.
    realisations : int, >= 1
    n_starts, bounds_override, rng, mode_tolerance,
        perturbation_scale, mode_warning_threshold
        Passed through.

    Returns
    -------
    dict with keys:
        'strike'      : (realisations, n_periods) float, degrees
        'twist'       : (realisations, n_periods, n_sites) float
        'shear'       : (realisations, n_periods, n_sites) float
        'gain'        : (realisations, n_periods, n_sites) float
        'rms_misfit'  : (realisations,) float
        'mode_warning': (realisations,) bool
    """
    if realisations < 1:
        raise ValueError(
            f"_bootstrap_decompose_joint: realisations must be >= 1, "
            f"got {realisations}"
        )

    n_sites = z_obs_per_site_full.shape[0]
    n_periods = z_obs_per_site_full.shape[1]

    # Build per-site predictions from the original-data primary modes.
    # Reuse single-site _predict_z_from_primary_modes by viewing each
    # site's modes as if it were a stand-alone single-site problem;
    # we only need the forward-model predictions, so rebuild here.
    z_predicted_per_site = np.full(
        (n_sites, n_periods, 2, 2), np.nan, dtype=np.complex128
    )
    best_rms = np.full((n_sites, n_periods), np.inf)
    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0
    for band_idx, modes in band_modes_list:
        primary = modes[0]
        n_band_freqs = len(band_idx)
        band_periods = selected_periods[band_idx]
        unpacked = _unpack_x_joint(primary.band_result.x_opt, n_sites, n_band_freqs)
        theta = unpacked[0]
        twist = unpacked[1]
        shear = unpacked[2]
        log10_gain = unpacked[3]
        log10_rho_a = unpacked[5]
        phase_a = unpacked[6]
        log10_rho_b = unpacked[7]
        phase_b = unpacked[8]
        for i in range(n_sites):
            t_i = np.tan(twist[i])
            e_i = np.tan(shear[i])
            gain_i = 10.0 ** log10_gain[i]
            rho_a = 10.0 ** log10_rho_a[i]
            rho_b = 10.0 ** log10_rho_b[i]
            abs_a = np.sqrt(rho_a * factor / band_periods)
            abs_b = np.sqrt(rho_b * factor / band_periods)
            a_band = abs_a * np.exp(1j * phase_a[i])
            b_band = abs_b * np.exp(1j * phase_b[i])
            for local_k, global_k in enumerate(band_idx):
                if primary.rms_misfit < best_rms[i, global_k]:
                    z_predicted_per_site[i, global_k] = gain_i * _estim_imp(
                        a_band[local_k], b_band[local_k], t_i, e_i, theta
                    )
                    best_rms[i, global_k] = primary.rms_misfit

    if np.any(np.isnan(z_predicted_per_site)):
        raise RuntimeError(
            "_bootstrap_decompose_joint: some periods not covered by "
            "any band's primary-mode prediction"
        )

    replicates = {
        "strike": np.full((realisations, n_periods), np.nan),
        "twist": np.full((realisations, n_periods, n_sites), np.nan),
        "shear": np.full((realisations, n_periods, n_sites), np.nan),
        "gain": np.full((realisations, n_periods, n_sites), np.nan),
        "rms_misfit": np.full(realisations, np.nan),
        "mode_warning": np.full(realisations, False),
    }

    for rep_i in range(realisations):
        z_replica = np.empty_like(z_predicted_per_site)
        for i in range(n_sites):
            z_replica[i] = _resample_residuals(
                z_obs_per_site_full[i],
                sigma_per_site_full[i],
                z_predicted_per_site[i],
                rng,
            )

        try:
            bm_list = _decompose_bands_with_modes_joint(
                z_obs_per_site_full=z_replica,
                sigma_per_site_full=sigma_per_site_full,
                selected_periods=selected_periods,
                bands=bands,
                n_starts=n_starts,
                bounds_override=bounds_override,
                rng=rng,
                mode_tolerance=mode_tolerance,
                perturbation_scale=perturbation_scale,
                objfun_joint=_objfun_joint,
                build_bounds_joint=_build_bounds_joint,
                canonical_initial_guess_joint=_canonical_initial_guess_joint,
                rotated_initial_guess_joint=_rotated_initial_guess_joint,
            )
        except Exception:
            continue

        for band_idx, modes in bm_list:
            primary = modes[0]
            n_band_freqs = len(band_idx)
            cf = _canonical_form_summary_joint(
                primary.band_result, n_sites, n_band_freqs
            )
            for global_k in band_idx:
                replicates["strike"][rep_i, global_k] = cf["strike_deg"]
                replicates["twist"][rep_i, global_k, :] = cf["per_site_twist_deg"]
                replicates["shear"][rep_i, global_k, :] = cf["per_site_shear_deg"]
                replicates["gain"][rep_i, global_k, :] = (
                    10.0 ** cf["per_site_log10_gain"]
                )

        all_rms = np.array([m[0].rms_misfit for _, m in bm_list])
        replicates["rms_misfit"][rep_i] = float(np.sqrt(np.mean(all_rms**2)))
        for _, modes in bm_list:
            if len(modes) >= 2:
                ratio = modes[1].rms_misfit / max(modes[0].rms_misfit, 1e-30)
                if ratio <= mode_warning_threshold:
                    replicates["mode_warning"][rep_i] = True
                    break

    return replicates


# ---------------------------------------------------------------------------
# Serialisation helpers (Session 8)
