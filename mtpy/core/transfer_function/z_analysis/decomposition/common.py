"""Shared utilities for decomposition methods.

This module collects the helpers that are not specific to any single
decomposition tradition (Groom-Bailey, Bahr, WAL, Lilley, ...) and
are intended for reuse as additional methods are added to the
package. Each helper is method-agnostic: it deals with the impedance
tensor, its uncertainty model, period banding, or the input
collection structure, and never with a particular GB / Bahr / etc.
parameterisation.

What lives here
---------------
- **Tensor algebra primitives** : :func:`_mat_multiply`,
  :func:`_extreme`. Tiny helpers that wrap idioms repeated across the
  optimisers.
- **Conversions** : :func:`_convz2r` (impedance -> apparent
  resistivity), :func:`_convz2p` (impedance -> phase),
  :func:`_estim_imp` (Groom-Bailey forward model — provided here
  because the regional impedance reconstruction it wraps is a
  natural building block for any future GB-derived method).
- **Residual / error helpers** : :func:`_calc_error`,
  :func:`_jkvar`. Common normalisations used in nonlinear least-
  squares cost functions.
- **Optimiser plumbing** : :func:`_unpack_x`, :func:`_unpack_x_joint`,
  the per-band parameter-vector packing used by both the GB and MJ
  cost functions.
- **Period-band partitioning** : :func:`_extract_bands`,
  :func:`_z_to_band_arrays`, :func:`_band_arrays_to_z`. Splitting an
  impedance tensor into log-period bands is a method-agnostic
  operation — every tradition that fits in bands needs the same
  helpers.
- **Per-band result container** : :class:`_BandResult`. The internal
  per-band optimisation record consumed by all method-specific
  aggregation code.
- **Joint-input validation** : :func:`_normalise_collection_input`,
  :func:`_validate_joint_input`. Multi-site methods accept either an
  :class:`MTCollection` or a list of ``MT`` objects; these helpers
  canonicalise the input and check the cross-site invariants
  (matching frequency grids, distinct station ids, etc.) any joint
  method needs.

What does **not** live here
---------------------------
Method-specific machinery — the GB cost function, mode clustering,
canonicalisation, and bootstrap — lives in the method's own module
(:mod:`.groom_bailey`, :mod:`.symmetries`). When a future Bahr or
Lilley implementation lands, its forward model, parameter-specific
bounds, and result aggregation will live in a peer module rather
than here.

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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


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

def _normalise_collection_input(
    collection_or_list,
) -> list[tuple[str, "Z"]]:
    """Convert MTCollection or list[MT] into (station_id, Z) tuples."""
    if isinstance(collection_or_list, list):
        return [
            (getattr(mt, "station", str(i)), mt.Z)
            for i, mt in enumerate(collection_or_list)
        ]
    if hasattr(collection_or_list, "dataframe") and hasattr(
        collection_or_list, "get_tf"
    ):
        out: list[tuple[str, "Z"]] = []
        df = collection_or_list.dataframe
        if df is None or len(df) == 0:
            return out
        for row in df.itertuples():
            tf_id = getattr(row, "tf_id", getattr(row, "station", None))
            if tf_id is None:
                continue
            mt = collection_or_list.get_tf(tf_id)
            out.append((getattr(mt, "station", tf_id), mt.Z))
        return out
    raise TypeError(
        f"decompose_joint: unexpected input type "
        f"{type(collection_or_list)!r}; expected list of MT or MTCollection"
    )

def _validate_joint_input(stations: list[tuple[str, "Z"]]) -> None:
    """Sanity-check joint input compatibility."""
    if not stations:
        raise ValueError("decompose_joint: no stations provided")

    for station_id, z in stations:
        if z.z_error is None:
            raise ValueError(
                f"decompose_joint: station {station_id!r} has no "
                f"z_error; required for weighted least squares"
            )
        if np.any(np.asarray(z.z_error) <= 0):
            raise ValueError(
                f"decompose_joint: station {station_id!r} has "
                f"non-positive z_error entries"
            )

    ref_freq = np.asarray(stations[0][1].frequency, dtype=np.float64)
    for station_id, z in stations[1:]:
        freq_i = np.asarray(z.frequency, dtype=np.float64)
        if freq_i.shape != ref_freq.shape or not np.allclose(
            freq_i, ref_freq, rtol=1e-6
        ):
            raise ValueError(
                f"decompose_joint: station {station_id!r} has a "
                f"different frequency grid than the first station; "
                f"joint analysis requires a common frequency grid"
            )


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


def _unpack_x_joint(
    x: np.ndarray, n_sites: int, n_freqs: int
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Unpack the joint multi-site optimisation state vector.

    Layout (all units consistent with single-site :func:`_unpack_x`)::

        x[0]                                         theta_shared
        x[1 + 4*i + 0..3]                            site i's
                                                     (twist, shear,
                                                      log10_gain,
                                                      anisotropy)
        x[1 + 4*n_sites + 4*n_freqs*i + 0*n_freqs + k]
                                                     site i's
                                                     log10_rho_a at
                                                     freq k
        x[... + 1*n_freqs + k]                       site i's phase_a
        x[... + 2*n_freqs + k]                       site i's
                                                     log10_rho_b
        x[... + 3*n_freqs + k]                       site i's phase_b

    Total length: ``1 + 4 * n_sites * (1 + n_freqs)``.

    For ``n_sites == 1`` the length coincides with the single-site
    layout (``5 + 4*n_freqs``), and the leading-five layout matches
    bit-for-bit. This is the foundation of the single-site reduction
    sanity tests.

    Parameters
    ----------
    x : (1 + 4*n_sites*(1+n_freqs),) float ndarray
    n_sites : int
    n_freqs : int

    Returns
    -------
    theta_shared : float
    twist, shear, log10_gain, anisotropy : (n_sites,) ndarrays
    log10_rho_a, phase_a, log10_rho_b, phase_b : (n_sites, n_freqs)
        ndarrays.

    Raises
    ------
    ValueError
        If ``len(x)`` does not match the joint layout.
    """
    expected_len = 1 + 4 * n_sites * (1 + n_freqs)
    if x.size != expected_len:
        raise ValueError(
            f"_unpack_x_joint: expected x of size {expected_len} "
            f"for n_sites={n_sites}, n_freqs={n_freqs}, got {x.size}"
        )

    theta_shared = float(x[0])

    site_scalars = x[1 : 1 + 4 * n_sites].reshape(n_sites, 4)
    twist = site_scalars[:, 0].copy()
    shear = site_scalars[:, 1].copy()
    log10_gain = site_scalars[:, 2].copy()
    anisotropy = site_scalars[:, 3].copy()

    regional_base = 1 + 4 * n_sites
    log10_rho_a = np.empty((n_sites, n_freqs))
    phase_a = np.empty((n_sites, n_freqs))
    log10_rho_b = np.empty((n_sites, n_freqs))
    phase_b = np.empty((n_sites, n_freqs))
    for i in range(n_sites):
        site_start = regional_base + 4 * n_freqs * i
        log10_rho_a[i] = x[site_start : site_start + n_freqs]
        phase_a[i] = x[site_start + n_freqs : site_start + 2 * n_freqs]
        log10_rho_b[i] = x[site_start + 2 * n_freqs : site_start + 3 * n_freqs]
        phase_b[i] = x[site_start + 3 * n_freqs : site_start + 4 * n_freqs]

    return (
        theta_shared,
        twist,
        shear,
        log10_gain,
        anisotropy,
        log10_rho_a,
        phase_a,
        log10_rho_b,
        phase_b,
    )
