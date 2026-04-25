"""Groom-Bailey / McNeice-Jones magnetotelluric tensor decomposition.

This module provides single-site (Groom & Bailey, 1989) and multi-site
joint (McNeice & Jones, 2001) decomposition of the magnetotelluric
impedance tensor, recovering the regional 2-D impedance and the
galvanic distortion parameters at each site.

It is a sibling to the existing
:mod:`mtpy.core.transfer_function.z_analysis.distortion` module which
provides Bibby et al. (2005) decomposition. The two methods are
distinct: Bibby fits a single 2x2 distortion matrix per site by
frequency-averaging; Groom-Bailey factorises the distortion into named
geometric parameters (gain, twist, shear, anisotropy) and fits these
jointly with the regional strike and impedances across multiple
frequencies. For most use cases, Groom-Bailey gives a richer and more
interpretable result; Bibby is faster and more robust at sites with
poor multi-frequency coverage.

References
----------
Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
tensor decomposition of magnetotelluric data. Geophysics, 66(1),
158-173.

See also
--------
:mod:`mtpy.core.transfer_function.z_analysis.distortion` :
    Bibby (2005) decomposition.
:mod:`mtpy.core.transfer_function.pt` :
    Phase tensor (Caldwell et al. 2004); distortion-invariant
    representation.

Notes
-----
This module ships pure-Python implementations of the GB and MJ
algorithms. Validation has been performed against historical Fortran
implementations (Strike, McNeice-Jones); see the project's
documentation for tolerance specifications and the optimiser-gap
analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.optimize import least_squares


if TYPE_CHECKING:
    from mtpy.core.mt import MT
    from mtpy.core.mt_collection import MTCollection
    from mtpy.core.transfer_function.z import Z


__all__ = [
    "DecompositionResult",
    "decompose",
    "decompose_joint",
]


@dataclass
class DecompositionResult:
    """Result of a Groom-Bailey or McNeice-Jones decomposition.

    Attributes
    ----------
    parameters : xarray.Dataset
        Per-period decomposition parameters. Coordinates: ``period``
        (in seconds). For multi-site joint decompositions an
        additional ``station`` coordinate.

        Data variables (all real-valued):

        - ``strike`` : regional azimuth in degrees, in
          ``[0, 180)``.
        - ``twist``, ``shear`` : Groom-Bailey distortion angles in
          degrees.
        - ``gain`` : Groom-Bailey site gain (dimensionless).
        - ``anisotropy`` : Groom-Bailey anisotropy parameter
          (dimensionless; structurally non-identifiable from MT
          alone, reported as fitted but flagged in metadata).
        - ``strike_error``, ``twist_error``, ``shear_error``,
          ``gain_error``, ``anisotropy_error`` : 1-sigma
          uncertainties from the analytic Jacobian, in matching
          units.

    regional_z : Z
        The decomposed regional impedance, as a fresh
        :class:`mtpy.core.transfer_function.z.Z` object with
        propagated errors.

    chi_squared : xarray.DataArray
        Per-period (or per-band) chi-squared values; coordinate
        ``period``.

    rms_misfit : float
        Overall RMS misfit, weighted by the input ``z_error`` (or
        ``z_model_error`` if used).

    method : str
        Identifier for the algorithm used:

        - ``"groom_bailey"`` for single-site GB.
        - ``"mcneice_jones_joint"`` for multi-site joint.

    options : dict
        Options passed to :func:`decompose` or
        :func:`decompose_joint` for this result.

    metadata : dict
        Provenance: software versions, input identifier, RNG seed,
        coordinate frame, strike convention, timestamp.

    frame : str, default ``"measurement"``
        Coordinate frame of ``regional_z``. ``"measurement"`` means
        ``regional_z`` is in the same frame as the input ``z``;
        rotating it by ``-strike`` per period recovers the
        anti-diagonal strike-frame regional tensor. ``"strike"``
        means ``regional_z`` is already in the strike frame
        (anti-diagonal). Public callers receive ``"measurement"``
        by default so plotting and downstream tools behave
        consistently with the input ``z``.

    Notes
    -----
    The 90-degree strike branch is folded into a canonical form:
    each band's ``(strike, twist, shear)`` is mapped through the
    ``(strike + 90 mod 180, -shear, twist)`` symmetry so that
    ``strike`` lands in ``[0, 90)`` after the fold. The shear sign
    is flipped together with the strike shift; twist is unchanged.
    See :func:`_canonicalise_solution`.

    The static-shift convention of ``gain`` matches that of
    :meth:`mtpy.core.mt.MT.remove_static_shift`. Applying
    :meth:`~mtpy.core.mt.MT.remove_static_shift` with the fitted
    ``gain`` recovers the un-shifted regional impedance.

    Examples
    --------
    Single-site decomposition of a station's impedance::

        from mtpy.core.transfer_function.z_analysis.decomposition import decompose
        result = decompose(my_mt.Z)
        result.regional_z.plot_resistivity_phase()
        print(result.parameters['strike'].values)
    """

    parameters: xr.Dataset
    regional_z: "Z"
    chi_squared: xr.DataArray
    rms_misfit: float
    method: str
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    frame: str = "measurement"


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
        Number of bootstrap realisations for parameter uncertainty.
        ``0`` disables bootstrap (analytic-Jacobian errors are still
        reported). Note: bootstrap is reliable for direct GB
        parameters but unreliable for invariant-derived quantities
        per Chave (2014); see module documentation.

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

    For the McNeice-Jones multi-site joint decomposition, see
    :func:`decompose_joint`.
    """
    from mtpy.core.transfer_function.z import Z as _Z

    if realisations > 0:
        raise NotImplementedError(
            "Bootstrap realisations are planned for a future "
            "contribution; pass realisations=0."
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

    band_modes_list: list[tuple[np.ndarray, list[_Mode]]] = []
    for band_idx in bands:
        z_obs_b = z_selected[band_idx]
        sigma_b = sigma_selected[band_idx]
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

    canon_per_band = []
    for band_idx, br in band_results:
        theta_c, twist_c, shear_c = _canonicalise_solution(
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
    params["strike"].attrs.update(
        units="degrees",
        range="[0, 90)",
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
        "strike_range_degrees": "[0, 90)",
        "canonicalisation": (
            "(strike + 90 mod 180, -shear, twist) symmetry folded "
            "so strike in [0, 90); shear sign flipped if pre-fold "
            "strike was in [90, 180)"
        ),
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
    collection: "MTCollection | list[MT]",
    periods: tuple[float, float] | None = None,
    bandwidth: float = 1.0,
    overlap: float = 0.0,
    norm_type: str = "GAVSD2",
    bounds_override: dict[str, tuple[float, float]] | None = None,
    initial_guess: dict[str, float] | None = None,
    realisations: int = 0,
    seed: int | None = None,
) -> DecompositionResult:
    """Multi-site joint Groom-Bailey decomposition (McNeice & Jones, 2001).

    Fits a single shared regional strike across all sites, with
    per-site distortion parameters and per-site regional
    impedances.

    Parameters
    ----------
    collection : MTCollection or list of MT
        Multi-site input. ``MTCollection`` is MTH5-backed;
        ``list[MT]`` is in-memory.

    periods, bandwidth, overlap, norm_type, bounds_override,
    initial_guess, realisations, seed
        See :func:`decompose`.

    Returns
    -------
    DecompositionResult
        With an additional ``station`` coordinate on
        ``parameters`` and ``regional_z`` per-station.

    Raises
    ------
    NotImplementedError
        Until the implementation lands.

    Notes
    -----
    See :func:`decompose` for the single-site equivalent and for the
    documentation of conventions.
    """
    raise NotImplementedError(
        "Multi-site joint decomposition implementation lands in a "
        "subsequent contribution session. See the project roadmap."
    )


# ---------------------------------------------------------------------------
# Private kernels: numerical building blocks for the GB89 / MJ01
# decomposition. None of these are part of the public API; they are
# called only by the (forthcoming) banded-decomposition driver and by
# the cross-validation tests against the Fortran reference. All are
# pure-NumPy implementations of the GB89 specification, not
# transcriptions of the Fortran reference.
# ---------------------------------------------------------------------------


def _mat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two complex 2x2 matrices.

    Implementation note: this is a one-liner via ``a @ b``. It exists
    as a named helper because the Groom-Bailey forward model uses
    matrix products in named steps, and giving them a name makes the
    higher-level code (:func:`_estim_imp`) read closer to the
    mathematical spec.

    Parameters
    ----------
    a, b : (2, 2) complex ndarray
        Input matrices.

    Returns
    -------
    (2, 2) complex ndarray
        The matrix product ``a @ b``.

    Notes
    -----
    No shape or type checking. Internal helper; callers ensure
    inputs.
    """
    return a @ b


def _extreme(values: np.ndarray) -> tuple[float, float]:
    """Return the (min, max) of a 1-D array.

    Used in the bootstrap path when the number of realisations is
    below the threshold for jackknife variance: extremal bounds are
    reported instead of standard deviations.

    Parameters
    ----------
    values : (n,) float ndarray
        Real-valued samples.

    Returns
    -------
    (vmin, vmax) : tuple of float
        Minimum and maximum of ``values``.

    Raises
    ------
    ValueError
        If ``values`` is empty.

    Notes
    -----
    The Fortran reference returns ``(vmax, vmin)`` (max first) per
    its argument list; this Python helper returns ``(vmin, vmax)``
    in the more conventional ascending order. Cross-validation
    tests adapt the order at the call site.
    """
    if values.size == 0:
        raise ValueError("_extreme: input array is empty")
    return float(np.min(values)), float(np.max(values))


def _convz2r(z: complex, period: float) -> float:
    """Convert a complex impedance to apparent resistivity.

    Formula: ``rho = |Z|**2 * T / (2 * pi * mu_0)``, where
    ``mu_0 = 4*pi*1e-7`` H/m. Z is in SI units (V/m/T); period in
    seconds; rho in ohm-metres.

    Parameters
    ----------
    z : complex
        Impedance, SI units.
    period : float
        Period in seconds.

    Returns
    -------
    float
        Apparent resistivity in ohm-metres.

    Notes
    -----
    The Fortran reference uses a hardcoded ``factor = 1.0`` and
    expects Z in SI units; so does this. If the input is in field
    units (mV/km/nT), the caller must convert beforehand --
    :class:`mtpy.core.transfer_function.z.Z` exposes ``.z`` in SI.
    """
    mu0 = 4.0 * np.pi * 1.0e-7
    return float(abs(z) ** 2 * period / (2.0 * np.pi * mu0))


def _convz2p(z: complex, period: float = 1.0) -> float:
    """Phase of a complex impedance, in degrees.

    Parameters
    ----------
    z : complex
        Impedance.
    period : float, default 1.0
        Period argument retained for signature compatibility with
        the Fortran reference; the value does not affect the output.
        See Notes.

    Returns
    -------
    float
        Phase in degrees, in ``(-180, 180]`` via ``atan2``.

    Notes
    -----
    The ``period`` parameter is vestigial: the Fortran ``convz2p``
    declares it in its signature but never references it in the
    body. We retain it for API parity to keep cross-validation
    direct, and document this explicitly here so that a future
    reviewer doesn't conclude the parameter is meaningful.
    """
    return float(np.degrees(np.arctan2(z.imag, z.real)))


def _calc_error(z_data: np.ndarray, z_pred: np.ndarray, sigma: np.ndarray) -> float:
    """Chi-squared residual between two impedance tensors.

    Computes ``sum_{ij} |z_data[i,j] - z_pred[i,j]|**2 /
    sigma[i,j]**2``, the standard chi-squared for complex-Gaussian
    residuals.

    Parameters
    ----------
    z_data, z_pred : (2, 2) complex ndarray
        Observed and model-predicted impedance tensors.
    sigma : (2, 2) float ndarray
        Per-component standard deviations. Must be strictly
        positive.

    Returns
    -------
    float
        Chi-squared value.

    Raises
    ------
    ValueError
        If ``sigma`` contains non-positive entries (the Fortran
        reference does not check; we are stricter here).
    """
    if np.any(sigma <= 0):
        raise ValueError(
            "_calc_error: sigma contains non-positive entries; "
            "cannot compute chi-squared"
        )
    diff = z_data - z_pred
    return float(np.sum((diff.real**2 + diff.imag**2) / sigma**2))


def _jkvar(values: np.ndarray) -> float:
    """Delete-1 jackknife variance estimator.

    For a sample ``values`` of size ``n >= 2``, computes the
    jackknife variance of the mean: ``((n - 1)/n) * sum_i (m_i -
    m_dot)^2``, where ``m_i`` is the mean omitting sample ``i`` and
    ``m_dot`` is the overall mean.

    Parameters
    ----------
    values : (n,) float ndarray
        Real-valued samples, n >= 2.

    Returns
    -------
    float
        Jackknife variance estimate, or ``-1.0`` as a sentinel if
        ``n < 2`` (matching the Fortran reference's behaviour).

    Notes
    -----
    The ``-1.0`` sentinel for ``n < 2`` is a Fortran-era convention
    and not how a modern Python API would handle the case; we keep
    it here for cross-validation parity. Callers in the modern
    Python code should check ``n`` themselves before calling this.

    The Fortran reference also writes "jkvar: n too small" to
    stderr when n is small; we silently return the sentinel
    instead. (mtpy-v2 uses loguru, but emitting a log line here is
    overkill for an internal helper.)
    """
    n = values.size
    if n < 2:
        return -1.0
    total = float(values.sum())
    mean = total / n
    means_minus_i = (total - values) / (n - 1)
    return float((n - 1) / n * np.sum((means_minus_i - mean) ** 2))


def _estim_imp(
    a: complex,
    b: complex,
    twist_tan: float,
    shear_tan: float,
    theta: float,
) -> np.ndarray:
    """Groom-Bailey forward model: regional impedance from
    parameters.

    Computes the measured-frame 2x2 impedance tensor predicted by
    the Groom-Bailey model given the regional impedances ``a``,
    ``b`` (the off-diagonal-frame TE and TM responses), the
    distortion parameters in tangent form (``twist_tan``,
    ``shear_tan``), and the regional azimuth.

    Parameters
    ----------
    a : complex
        Regional impedance ``Z_TE`` in the strike frame, SI units.
    b : complex
        Regional impedance ``Z_TM`` in the strike frame, SI units.
    twist_tan : float
        Tangent of the twist angle. The twist itself is
        ``arctan(twist_tan)``.
    shear_tan : float
        Tangent of the shear angle.
    theta : float
        Regional azimuth in radians, clockwise from x-axis.

    Returns
    -------
    z : (2, 2) complex ndarray
        The forward-modelled measured-frame impedance tensor.

    Notes
    -----
    Implementation: this is the cleaner mathematical form derived
    from the GB89 formulas, computing the four Pauli-spin
    combinations (alpha) directly and reconstructing the tensor
    from them. The Fortran reference internally negates and
    restores ``twist_tan`` and ``theta`` during construction, with
    no net effect; the cleaner form skips that. Cross-validation
    against the Fortran reference confirms agreement at machine
    precision (see ``test_kernels_against_fortran.py``).

    The four alpha values (Pauli-spin combinations) are::

        alpha[0] = Z_xx + Z_yy   (trace)
        alpha[1] = Z_xy + Z_yx
        alpha[2] = Z_yx - Z_xy
        alpha[3] = Z_xx - Z_yy

    Inverting::

        Z_xx = (alpha[0] + alpha[3]) / 2
        Z_yy = (alpha[0] - alpha[3]) / 2
        Z_xy = (alpha[1] - alpha[2]) / 2
        Z_yx = (alpha[1] + alpha[2]) / 2

    The Fortran ``estim_imp`` signature is
    ``(gamma1, gamma2, a, b, theta)`` with ``gamma1 = shear_tan``
    and ``gamma2 = twist_tan``; this Python signature reorders to
    ``(a, b, twist_tan, shear_tan, theta)`` to match the natural
    grouping (regional impedances first, then distortion, then
    rotation). Cross-validation tests adapt the call order.
    """
    c2 = np.cos(2.0 * theta)
    s2 = np.sin(2.0 * theta)
    t = twist_tan
    e = shear_tan

    # GB89 alpha combinations (Pauli-spin form). See the
    # docstring above for the inversion to Z and the strike_py
    # forensic_report 17 for the derivation from the GB89
    # scattering matrix.
    alpha0 = -b * (e - t) + a * (t + e)
    alpha1 = (
        s2 * (-b * (e - t))
        + c2 * (-b * (1.0 + t * e))
        + c2 * (a * (1.0 - e * t))
        - s2 * (a * (t + e))
    )
    alpha2 = -b * (1.0 + t * e) - a * (1.0 - e * t)
    alpha3 = (
        c2 * (-b * (e - t))
        - s2 * (-b * (1.0 + t * e))
        - s2 * (a * (1.0 - e * t))
        - c2 * (a * (t + e))
    )

    z = np.empty((2, 2), dtype=np.complex128)
    z[0, 0] = (alpha0 + alpha3) / 2.0
    z[1, 1] = (alpha0 - alpha3) / 2.0
    z[0, 1] = (alpha1 - alpha2) / 2.0
    z[1, 0] = (alpha1 + alpha2) / 2.0
    return z


def _unpack_x(
    x: np.ndarray, n_freqs: int
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Unpack the GB optimisation state vector.

    Inverse of the packing convention used by :func:`_objfun` and
    (in a subsequent session) ``_solve_band``. Returns the named
    parameters as a tuple, suitable for unpacking with multiple-
    assignment.

    Parameters
    ----------
    x : (5 + 4*n_freqs,) float ndarray
        Flat parameter vector.

        - ``x[0]`` : strike theta in radians
        - ``x[1]`` : twist t in radians
        - ``x[2]`` : shear e in radians
        - ``x[3]`` : log10 of site gain g
        - ``x[4]`` : anisotropy s (dimensionless)
        - ``x[5 : 5 + n_freqs]`` : log10(rho_a) per frequency
        - ``x[5 + n_freqs : 5 + 2*n_freqs]`` : phase_a (radians)
          per frequency
        - ``x[5 + 2*n_freqs : 5 + 3*n_freqs]`` : log10(rho_b) per
          frequency
        - ``x[5 + 3*n_freqs : 5 + 4*n_freqs]`` : phase_b (radians)
          per frequency

    n_freqs : int
        Number of frequencies in the band.

    Returns
    -------
    theta, twist, shear, log10_gain, anisotropy : float
        Scalar parameters.
    log10_rho_a, phase_a, log10_rho_b, phase_b : (n_freqs,) ndarray
        Per-frequency parameters.

    Raises
    ------
    ValueError
        If ``len(x) != 5 + 4*n_freqs``.

    Notes
    -----
    The packing places scalar parameters first so that the
    optimiser's preconditioner (TRF's column scaling) can be set
    independently for the structurally distinct parameter blocks.
    """
    expected_len = 5 + 4 * n_freqs
    if x.size != expected_len:
        raise ValueError(
            f"_unpack_x: expected x of size {expected_len} for "
            f"n_freqs={n_freqs}, got {x.size}"
        )
    theta = float(x[0])
    twist = float(x[1])
    shear = float(x[2])
    log10_gain = float(x[3])
    anisotropy = float(x[4])

    base = 5
    log10_rho_a = x[base : base + n_freqs]
    phase_a = x[base + n_freqs : base + 2 * n_freqs]
    log10_rho_b = x[base + 2 * n_freqs : base + 3 * n_freqs]
    phase_b = x[base + 3 * n_freqs : base + 4 * n_freqs]

    return (
        theta,
        twist,
        shear,
        log10_gain,
        anisotropy,
        log10_rho_a,
        phase_a,
        log10_rho_b,
        phase_b,
    )


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
# ---------------------------------------------------------------------------


@dataclass
class _BandResult:
    """Result of a single-band optimisation. Internal type."""

    x_opt: np.ndarray
    x_err: np.ndarray
    residuals: np.ndarray
    chi_squared: float
    rms_misfit: float
    n_iter: int
    converged: bool
    cost_at_opt: float
    jacobian: np.ndarray | None = None


def _canonicalise_solution(
    strike: float, twist: float, shear: float
) -> tuple[float, float, float]:
    """Apply the GB 90-degree / shear-sign symmetry to fold a
    solution to a canonical branch.

    The Groom-Bailey decomposition has a discrete symmetry: the
    triple ``(strike, twist, shear)`` and ``(strike + 90 mod 180,
    twist, -shear)`` represent the same physical solution (twist is
    invariant; the strike shift by 90 degrees is paired with a sign
    flip on shear). This function chooses the branch with strike in
    ``[0, 90)``, which matches the strike_py reference's
    ``symmetry.compare_band_against_dcmp`` convention.

    Parameters
    ----------
    strike, twist, shear : float
        Radians.

    Returns
    -------
    strike_c, twist_c, shear_c : float
        Canonicalised values, with ``strike_c in [0, pi/2)``.
    """
    half_pi = np.pi / 2.0
    # Tolerance for the boundary at strike = pi/2 (90 deg). Without
    # this, a strike that lands exactly at pi/2 can be rounded to one
    # ULP below pi/2 by ``% np.pi`` and miss the fold.
    fold_tol = 1e-9
    strike_mod_pi = strike % np.pi
    if strike_mod_pi >= half_pi - fold_tol:
        strike_c = strike_mod_pi - half_pi
        shear_c = -shear
        # Re-fold strike_c into [0, pi/2): if the input was very close
        # to pi/2 from below, strike_c is a tiny negative; map to 0.
        if strike_c < 0.0:
            strike_c = 0.0
    else:
        strike_c = strike_mod_pi
        shear_c = shear
    return float(strike_c), float(twist), float(shear_c)


def _extract_bands(
    periods: np.ndarray,
    bandwidth: float = 1.0,
    overlap: float = 0.0,
) -> list[np.ndarray]:
    """Partition periods into (possibly overlapping) bands.

    Each band is described by the array of indices into ``periods``
    that fall within the band. Bands span ``bandwidth`` decades in
    ``log10(period)`` and successive bands shift by
    ``bandwidth - overlap`` decades.

    Parameters
    ----------
    periods : (n_periods,) float64
        Strictly positive periods in seconds. Need not be sorted;
        the returned indices index the input array as given.
    bandwidth : float, default 1.0
        Band width in ``log10(period)`` decades.
    overlap : float, default 0.0
        Overlap between adjacent bands. Must satisfy
        ``0 <= overlap < bandwidth``.

    Returns
    -------
    list[ndarray]
        One ``int`` ndarray of indices per band. Bands containing
        fewer than 2 periods are dropped (under-determined).
    """
    if periods.size == 0:
        raise ValueError("_extract_bands: periods is empty")
    if np.any(periods <= 0):
        raise ValueError("_extract_bands: periods must be strictly positive")
    if overlap >= bandwidth:
        raise ValueError(
            f"_extract_bands: overlap ({overlap}) must be less "
            f"than bandwidth ({bandwidth})"
        )

    log_periods = np.log10(periods)
    log_min = float(log_periods.min())
    log_max = float(log_periods.max())
    step = bandwidth - overlap
    eps = 1e-12

    bands: list[np.ndarray] = []
    band_start = log_min
    while band_start < log_max + eps:
        band_end = band_start + bandwidth
        in_band = (log_periods >= band_start - eps) & (log_periods < band_end - eps)
        # Include the maximum period in the band whose band_end is
        # at-or-past it (right-open elsewhere, right-closed at the
        # global maximum). Without this, a period exactly on a band
        # boundary that is also log_max can fall through the cracks.
        if band_end >= log_max - eps:
            in_band = in_band | (np.abs(log_periods - log_max) < eps)
        idx = np.where(in_band)[0]
        if idx.size >= 2:
            bands.append(idx)
        band_start += step

    if not bands:
        raise ValueError(
            "_extract_bands: no band has 2 or more periods. "
            "Reduce bandwidth or use a wider period range."
        )
    return bands


def _z_to_band_arrays(
    z: "Z", band_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-band ``(z_obs, sigma, periods)`` arrays from a Z.

    Reads through the public ``Z.z`` and ``Z.z_error`` attributes
    (the unit-corrected, user-facing values), per CLAUDE.md
    invariant 4. Never via internal storage attributes.

    Parameters
    ----------
    z : Z
        Full impedance object.
    band_idx : (n_band_freqs,) int ndarray
        Indices into ``z.frequency`` of the periods in this band.

    Returns
    -------
    z_obs : (n_band_freqs, 2, 2) complex128
    sigma : (n_band_freqs, 2, 2) float64
    periods : (n_band_freqs,) float64

    Raises
    ------
    ValueError
        If ``z.z_error`` is None or contains non-positive entries.
    """
    z_full = np.asarray(z.z, dtype=np.complex128)
    z_err = z.z_error
    if z_err is None:
        raise ValueError(
            "_z_to_band_arrays: z.z_error is None; cannot run "
            "weighted least squares without per-component errors."
        )
    sigma_full = np.asarray(z_err, dtype=np.float64)
    if np.any(sigma_full <= 0):
        raise ValueError(
            "_z_to_band_arrays: z.z_error contains non-positive " "entries"
        )

    z_obs = z_full[band_idx]
    sigma = sigma_full[band_idx]
    frequencies = np.asarray(z.frequency, dtype=np.float64)[band_idx]
    periods = 1.0 / frequencies
    return z_obs, sigma, periods


def _band_arrays_to_z(
    log10_rho_a: np.ndarray,
    phase_a: np.ndarray,
    log10_rho_b: np.ndarray,
    phase_b: np.ndarray,
    log10_rho_a_err: np.ndarray,
    phase_a_err: np.ndarray,
    log10_rho_b_err: np.ndarray,
    phase_b_err: np.ndarray,
    periods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a strike-frame regional impedance and error arrays from
    the recovered ``(log10_rho, phase)`` parameters.

    The regional impedance is the canonical anti-diagonal 2D tensor
    in the strike frame: ``Z_2D = [[0, a], [-b, 0]]`` where ``a``
    is the TE-mode impedance and ``b`` the TM-mode impedance.

    Parameters
    ----------
    log10_rho_a, phase_a, log10_rho_b, phase_b : (n_periods,) float64
        Recovered regional parameters per period.
    log10_rho_a_err, phase_a_err, log10_rho_b_err, phase_b_err
        : (n_periods,) float64. 1-sigma uncertainties.
    periods : (n_periods,) float64
        Periods in seconds.

    Returns
    -------
    z_regional : (n_periods, 2, 2) complex128
        Strike-frame regional tensors.
    z_regional_error : (n_periods, 2, 2) float64
        Per-component 1-sigma propagated uncertainties.
    """
    n_periods = len(periods)
    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0
    ln10 = np.log(10.0)

    rho_a = 10.0**log10_rho_a
    rho_b = 10.0**log10_rho_b
    abs_a = np.sqrt(rho_a * factor / periods)
    abs_b = np.sqrt(rho_b * factor / periods)
    a = abs_a * np.exp(1j * phase_a)
    b = abs_b * np.exp(1j * phase_b)

    z_regional = np.zeros((n_periods, 2, 2), dtype=np.complex128)
    z_regional[:, 0, 1] = a
    z_regional[:, 1, 0] = -b

    abs_a_err = np.sqrt(
        (abs_a * ln10 / 2.0) ** 2
        * np.where(np.isfinite(log10_rho_a_err), log10_rho_a_err, 0.0) ** 2
        + abs_a**2 * np.where(np.isfinite(phase_a_err), phase_a_err, 0.0) ** 2
    )
    abs_b_err = np.sqrt(
        (abs_b * ln10 / 2.0) ** 2
        * np.where(np.isfinite(log10_rho_b_err), log10_rho_b_err, 0.0) ** 2
        + abs_b**2 * np.where(np.isfinite(phase_b_err), phase_b_err, 0.0) ** 2
    )

    z_regional_error = np.zeros((n_periods, 2, 2), dtype=np.float64)
    z_regional_error[:, 0, 1] = abs_a_err
    z_regional_error[:, 1, 0] = abs_b_err
    return z_regional, z_regional_error


def _canonical_initial_guess(
    z_obs: np.ndarray,
    sigma: np.ndarray,
    periods: np.ndarray,
) -> np.ndarray:
    """Canonical heuristic starting point for the GB optimisation.

    Strategy:
      - Strike: circular median of per-period phase-tensor azimuths
        (folds the 90-degree ambiguity by averaging ``2*az``).
      - Twist, shear, log10_gain, anisotropy: zero (no a priori
        distortion, gain = 1).
      - Per-period ``(rho_a, phase_a, rho_b, phase_b)``: derived
        from the rotated tensor assuming no distortion.

    Parameters
    ----------
    z_obs : (n_freqs, 2, 2) complex128
    sigma : (n_freqs, 2, 2) float64
    periods : (n_freqs,) float64

    Returns
    -------
    x0 : (5 + 4*n_freqs,) float64
    """
    n_freqs = len(periods)
    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0

    azimuths = np.empty(n_freqs)
    for k in range(n_freqs):
        X = z_obs[k].real
        Y = z_obs[k].imag
        try:
            phi = np.linalg.solve(X, Y)
        except np.linalg.LinAlgError:
            azimuths[k] = 0.0
            continue
        phi_sym = (phi + phi.T) / 2.0
        num = phi_sym[0, 1] + phi_sym[1, 0]
        den = phi_sym[0, 0] - phi_sym[1, 1]
        azimuths[k] = 0.5 * np.arctan2(num, den)
    median_2az = np.arctan2(
        np.median(np.sin(2.0 * azimuths)),
        np.median(np.cos(2.0 * azimuths)),
    )
    theta0 = median_2az / 2.0

    c, s = np.cos(theta0), np.sin(theta0)
    R = np.array([[c, -s], [s, c]])
    log10_rho_a = np.empty(n_freqs)
    phase_a = np.empty(n_freqs)
    log10_rho_b = np.empty(n_freqs)
    phase_b = np.empty(n_freqs)
    for k in range(n_freqs):
        z_rot = R.T @ z_obs[k] @ R
        a_k = z_rot[0, 1]
        b_k = -z_rot[1, 0]
        log10_rho_a[k] = np.log10(max(abs(a_k), 1e-15) ** 2 * periods[k] / factor)
        phase_a[k] = np.arctan2(a_k.imag, a_k.real)
        log10_rho_b[k] = np.log10(max(abs(b_k), 1e-15) ** 2 * periods[k] / factor)
        phase_b[k] = np.arctan2(b_k.imag, b_k.real)

    x0 = np.concatenate(
        [
            [theta0, 0.0, 0.0, 0.0, 0.0],
            log10_rho_a,
            phase_a,
            log10_rho_b,
            phase_b,
        ]
    )
    return x0


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


def _rotated_initial_guess(
    z_obs: np.ndarray,
    sigma: np.ndarray,
    periods: np.ndarray,
) -> np.ndarray:
    """Alternative initial guess at the 90-degree symmetry branch.

    Same as :func:`_canonical_initial_guess` but with strike rotated
    by ``pi/2`` (mod pi) and the per-frequency TE/TM regional
    impedance parameters swapped. This deliberately seeds an
    optimisation run at the alternate branch of the GB89 90-degree
    ambiguity. Combined with the canonical guess, the pair covers
    both sides of the dominant symmetry; if the global minimum lives
    on the alternate branch, the optimiser starting here will find
    it.

    Parameters
    ----------
    z_obs : (n_freqs, 2, 2) complex128
    sigma : (n_freqs, 2, 2) float64
    periods : (n_freqs,) float64

    Returns
    -------
    x0 : (5 + 4*n_freqs,) float64
    """
    n_freqs = len(periods)
    canonical = _canonical_initial_guess(z_obs, sigma, periods)

    rotated = canonical.copy()
    rotated[0] = (canonical[0] + np.pi / 2.0) % np.pi

    base = 5
    log10_rho_a = canonical[base : base + n_freqs].copy()
    phase_a = canonical[base + n_freqs : base + 2 * n_freqs].copy()
    log10_rho_b = canonical[base + 2 * n_freqs : base + 3 * n_freqs].copy()
    phase_b = canonical[base + 3 * n_freqs : base + 4 * n_freqs].copy()

    rotated[base : base + n_freqs] = log10_rho_b
    rotated[base + n_freqs : base + 2 * n_freqs] = phase_b
    rotated[base + 2 * n_freqs : base + 3 * n_freqs] = log10_rho_a
    rotated[base + 3 * n_freqs : base + 4 * n_freqs] = phase_a

    return rotated


def _perturbed_initial_guess(
    canonical_x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    perturbation_scale: float = 0.1,
) -> np.ndarray:
    """Random Gaussian perturbation of a canonical initial guess.

    Each parameter is perturbed by a Gaussian with standard
    deviation ``perturbation_scale * (upper - lower)`` for that
    parameter, then clipped to the bounds so scipy's TRF accepts
    the start.

    Parameters
    ----------
    canonical_x0 : (n_params,) float64
    lower, upper : (n_params,) float64
    rng : np.random.Generator
    perturbation_scale : float, default 0.1
        Standard deviation of the Gaussian perturbation as a
        fraction of the bound width per parameter.

    Returns
    -------
    x0 : (n_params,) float64
        Perturbed and clipped.
    """
    bound_widths = upper - lower
    sigma_per_param = perturbation_scale * bound_widths
    perturbation = rng.normal(scale=sigma_per_param)
    x0 = canonical_x0 + perturbation
    return np.clip(x0, lower, upper)


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


_DEFAULT_MODE_TOLERANCE = {
    "strike_deg": 0.5,
    "twist_deg": 0.5,
    "shear_deg": 0.5,
}
# Note: log10_gain was originally intended as a fourth discriminator
# but is gauge-equivalent at the band level (the forward model
# satisfies (a, b, gain) -> (gain*a, gain*b, 1)), so different starts
# converge to physically-equivalent solutions with different
# gain/|a|/|b| splits. Including it as a clustering criterion would
# fragment a single physical mode into multiple gauge-equivalent
# clusters; we exclude it.


@dataclass
class _Mode:
    """Internal type: one discovered mode of a multi-start fit.

    Each mode collects the converged starts that landed in the same
    physical basin (matching strike/twist/shear/gain within the
    user-configurable tolerances). The representative
    :class:`_BandResult` is the lowest-RMS member of the cluster.
    """

    rms_misfit: float
    chi_squared: float
    n_starts_landing_here: int
    band_result: _BandResult
    canonical_form: dict
    probability: float | None = None


def _canonical_form_summary(br: _BandResult) -> dict:
    """Extract canonical-form scalar parameters for clustering.

    Applies :func:`_canonicalise_solution` so that two converged
    points differing only by the GB 90-degree symmetry produce
    identical summaries.

    Returns a dict with keys: ``strike_deg``, ``twist_deg``,
    ``shear_deg``, ``log10_gain``, ``rms_misfit``.
    """
    strike_rad, twist_rad, shear_rad = _canonicalise_solution(
        float(br.x_opt[0]), float(br.x_opt[1]), float(br.x_opt[2])
    )
    return {
        "strike_deg": float(np.degrees(strike_rad)),
        "twist_deg": float(np.degrees(twist_rad)),
        "shear_deg": float(np.degrees(shear_rad)),
        "log10_gain": float(br.x_opt[3]),
        "rms_misfit": float(br.rms_misfit),
    }


def _modes_match_within_band(cf_a: dict, cf_b: dict, tolerance: dict) -> bool:
    """Three physical-parameter conditions, used during clustering
    of starts within a single band.

    Two converged points within a band are the same mode when
    strike, twist, and shear all agree within their respective
    tolerances. We deliberately exclude log10_gain (gauge-equivalent;
    see :data:`_DEFAULT_MODE_TOLERANCE` notes) and RMS (TRF's path-
    dependent numerical noise produces O(1e-4) relative RMS
    differences between starts converging to the same physical
    basin).
    """
    return (
        abs(cf_a["strike_deg"] - cf_b["strike_deg"]) <= tolerance["strike_deg"]
        and abs(cf_a["twist_deg"] - cf_b["twist_deg"]) <= tolerance["twist_deg"]
        and abs(cf_a["shear_deg"] - cf_b["shear_deg"]) <= tolerance["shear_deg"]
    )


def _modes_match_across_bands(cf_a: dict, cf_b: dict, tolerance: dict) -> bool:
    """Same four physical-parameter conditions, used to compare
    bands' primary modes.

    The within-band and across-band matchers check the same fields
    today; the function exists as a separate name because the
    intents at the two call sites differ. Within a band, RMS is
    the natural tie-breaker but we deliberately don't use it (see
    :func:`_modes_match_within_band` notes). Across bands, RMS is
    meaningless to compare because each band fits a different
    period subset, so any RMS-aware matcher would always report
    disagreement.
    """
    return _modes_match_within_band(cf_a, cf_b, tolerance)


def _cluster_modes(
    band_results: list[_BandResult],
    mode_tolerance: dict | None = None,
) -> list[_Mode]:
    """Greedy clustering of converged points into modes.

    Algorithm:
      1. Compute canonical-form summary for each
         :class:`_BandResult`.
      2. Sort summaries ascending by RMS misfit.
      3. The lowest-RMS point seeds mode 0 and becomes its
         representative.
      4. Each subsequent point is compared to all existing modes
         via :func:`_modes_match_within_band`; if it matches any,
         increment that mode's count, else open a new mode.

    By sorting first, each mode's representative is always the
    best converged point in its cluster.

    Parameters
    ----------
    band_results : list of _BandResult
    mode_tolerance : dict, optional
        Overrides for the per-parameter tolerances. Keys:
        ``strike_deg``, ``twist_deg``, ``shear_deg``. Defaults
        applied for missing keys. The deprecated ``rms_relative``
        and ``log10_gain`` keys raise :class:`UserWarning` and are
        ignored.

    Returns
    -------
    list of _Mode
        Sorted ascending by RMS misfit. Mode 0 is primary.
    """
    if not band_results:
        return []

    tol = dict(_DEFAULT_MODE_TOLERANCE)
    if mode_tolerance:
        deprecated = {"rms_relative", "log10_gain"} & set(mode_tolerance)
        if deprecated:
            warnings.warn(
                f"mode_tolerance keys {sorted(deprecated)} are no longer "
                "used and will be ignored. Clustering uses strike_deg, "
                "twist_deg, and shear_deg only: rms_relative was "
                "fragile under TRF's path-dependent noise, and "
                "log10_gain is gauge-equivalent with the regional "
                "impedance magnitudes at the band level.",
                UserWarning,
                stacklevel=2,
            )
            mode_tolerance = {
                k: v for k, v in mode_tolerance.items() if k not in deprecated
            }
        tol.update(mode_tolerance)

    summaries = [(br, _canonical_form_summary(br)) for br in band_results]
    summaries.sort(key=lambda x: x[1]["rms_misfit"])

    modes: list[_Mode] = []
    for br, cf in summaries:
        matched = False
        for mode in modes:
            if _modes_match_within_band(mode.canonical_form, cf, tol):
                mode.n_starts_landing_here += 1
                matched = True
                break
        if not matched:
            modes.append(
                _Mode(
                    rms_misfit=br.rms_misfit,
                    chi_squared=br.chi_squared,
                    n_starts_landing_here=1,
                    band_result=br,
                    canonical_form=cf,
                )
            )
    return modes


def _compute_mode_probabilities(modes: list[_Mode]) -> list[float]:
    """Laplace approximation of mode probabilities.

    For each mode i, the unnormalised log-weight is::

        log_w_i = -0.5 * chi_squared_i + 0.5 * log det(cov_i)

    where ``cov_i = (J_eff^T J_eff)^{-1}`` is the parameter
    covariance restricted to identifiable parameters (the
    anisotropy column of the Jacobian is identically zero, so we
    drop it from the determinant calculation).

    Probabilities are obtained by max-subtract on the log-weights
    for numerical stability::

        log_w_max = max(log_w_i)
        w_i = exp(log_w_i - log_w_max)
        p_i = w_i / sum_j w_j

    Parameters
    ----------
    modes : list of _Mode
        Each mode's ``band_result.jacobian`` must be populated.

    Returns
    -------
    list of float
        Mode probabilities, summing to 1, in the same order as
        the input.

    Notes
    -----
    The Laplace approximation assumes a Gaussian likelihood near
    each mode. For MT decomposition the residuals are complex-
    Gaussian by assumption, so the approximation is reasonable.

    For modes whose effective Gram matrix is rank-deficient (more
    than the anisotropy column unidentifiable), the determinant
    is evaluated via eigenvalues of the pseudo-inverse with a
    small floor.
    """
    if not modes:
        return []
    if len(modes) == 1:
        return [1.0]

    log_weights: list[float] = []
    for mode in modes:
        chi_sq = mode.chi_squared
        J = mode.band_result.jacobian
        if J is None:
            # Should not happen in normal use; fall back to a
            # likelihood-only weight.
            log_weights.append(-0.5 * chi_sq)
            continue
        # Drop anisotropy column (index 4) from the Jacobian for
        # the determinant: it's identically zero so det(J^T J)
        # would otherwise be exactly zero.
        J_eff = np.delete(J, 4, axis=1)
        gram = J_eff.T @ J_eff
        sign, log_det_gram = np.linalg.slogdet(gram)
        if sign > 0:
            log_det_cov = -log_det_gram
        else:
            cov = np.linalg.pinv(gram)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-30]
            if eigenvalues.size == 0:
                log_det_cov = -np.inf
            else:
                log_det_cov = float(np.sum(np.log(eigenvalues)))
        log_weights.append(-0.5 * chi_sq + 0.5 * log_det_cov)

    log_w_max = max(log_weights)
    weights = [float(np.exp(lw - log_w_max)) for lw in log_weights]
    total = sum(weights)
    if total <= 0.0:
        # Degenerate; fall back to uniform.
        return [1.0 / len(modes)] * len(modes)
    return [w / total for w in weights]


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


def _detect_band_disagreement(
    band_modes_list: list[tuple[np.ndarray, list[_Mode]]],
    mode_tolerance: dict,
) -> dict | None:
    """Check whether bands' primary modes agree on physical params.

    Uses :func:`_modes_match_across_bands`, which checks
    strike/twist/shear/gain only (RMS comparison is meaningless
    across bands fitting different period subsets).

    Returns ``None`` if all primaries agree. Otherwise returns a
    dict listing each disagreeing band pair with their canonical-
    form summaries.
    """
    if len(band_modes_list) < 2:
        return None

    tol = dict(_DEFAULT_MODE_TOLERANCE)
    if mode_tolerance:
        # Tolerate deprecated keys silently here; _cluster_modes has
        # already warned upstream.
        tol.update(
            {
                k: v
                for k, v in mode_tolerance.items()
                if k not in ("rms_relative", "log10_gain")
            }
        )

    primaries = [modes[0].canonical_form for _, modes in band_modes_list]

    disagreements: list[dict] = []
    for i in range(len(primaries)):
        for j in range(i + 1, len(primaries)):
            if not _modes_match_across_bands(primaries[i], primaries[j], tol):
                disagreements.append(
                    {
                        "bands": [i, j],
                        "values_i": primaries[i],
                        "values_j": primaries[j],
                    }
                )

    if not disagreements:
        return None

    return {
        "disagreements": disagreements,
        "interpretation": (
            "Bands' primary modes disagree on canonical-form "
            "physical parameters. This may indicate model "
            "misspecification (deeper/shallower structure differs "
            "across the period range) or that one band's primary "
            "mode is not the global minimum. Inspect "
            "metadata['per_band'] to see all modes per band."
        ),
    }


def _detect_primary_mode_warning(
    band_modes_list: list[tuple[np.ndarray, list[_Mode]]],
    threshold: float,
) -> tuple[bool, str]:
    """Warn when any band's second-best mode is within ``threshold``
    of its primary by RMS ratio.

    Returns ``(True, text)`` if any band has a competitive second
    mode, else ``(False, "")``.
    """
    for _, modes in band_modes_list:
        if len(modes) < 2:
            continue
        ratio = modes[1].rms_misfit / max(modes[0].rms_misfit, 1e-30)
        if ratio <= threshold:
            text = (
                f"At least one band discovered multiple modes within "
                f"{threshold}x RMS of each other. Single-mode "
                f"confidence intervals are misleading; the data does "
                f"not strongly distinguish between competing physical "
                f"solutions. See metadata['per_band']."
            )
            return True, text
    return False, ""


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
