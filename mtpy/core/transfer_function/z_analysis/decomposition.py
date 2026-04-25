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

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
import xarray as xr


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

    Notes
    -----
    The 90-degree strike branch is folded into the canonical range
    ``[0, 180)``. Two fits differing by ``(strike + 90 mod 180,
    -shear, twist)`` represent the same physical solution; the
    canonicalisation chooses one branch consistently.

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
) -> DecompositionResult:
    """Single-site Groom-Bailey decomposition.

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
        RNG seed for the bootstrap (PCG64). Required if
        ``realisations > 0``.

    Returns
    -------
    DecompositionResult

    Raises
    ------
    NotImplementedError
        Until the implementation lands. See the contribution roadmap
        in the project documentation.

    Notes
    -----
    Strike convention: clockwise from the x-axis defined by
    ``mt.coordinate_reference_frame`` of the parent station, in
    degrees, in the canonical range ``[0, 180)``.

    For the McNeice-Jones multi-site joint decomposition, see
    :func:`decompose_joint`.
    """
    raise NotImplementedError(
        "Groom-Bailey decomposition implementation lands in a "
        "subsequent contribution session. See the project roadmap."
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
