"""Lilley Mohr-circle distortion analysis.

The Lilley intellectual tradition (Lilley 1976, 1993, 2012, 2016,
2018, 2020; Lilley & Phillips 2018) represents a magnetotelluric
impedance tensor as a pair of Mohr circles — one for the in-phase
part and one for the quadrature part — together with the rotational
invariants visible on those diagrams. Unlike the Groom-Bailey or
Bibby-Caldwell-Brown methods this is a parametrisation-free
analysis: it does not factor ``C`` into named operators, it does
not solve a non-linear inverse, and it does not pick a regional
strike via a single fold convention. Instead it makes the
geometric structure of the tensor's behaviour under axes rotation
directly visible.

Algebra
-------
The forward equations for the Mohr-circle representation are
Lilley 2018 (eqs. 23-32). For an impedance tensor with elements
``[Z_xx, Z_xy; Z_yx, Z_yy]``, the in-phase part has elements
``Z_xxp`` etc. (real). Under a clockwise rotation of the
measuring axes by angle ``θ'`` the in-phase tensor elements vary
as

    Z'_xxp = (Z_xxp + Z_yyp)/2 + C_p sin(2θ' + β_p)
    Z'_xyp = (Z_xyp - Z_yxp)/2 + C_p cos(2θ' + β_p)
    Z'_yxp = -(Z_xyp - Z_yxp)/2 + C_p cos(2θ' + β_p)
    Z'_yyp = (Z_xxp + Z_yyp)/2 - C_p sin(2θ' + β_p)

where the radius is

    C_p = (1/2) * sqrt[(Z_xxp - Z_yyp)^2 + (Z_xyp + Z_yxp)^2]

and the phase

    tan β_p = (Z_xxp - Z_yyp) / (Z_xyp + Z_yxp).

Plotting ``Z'_xxp`` against ``Z'_xyp`` as ``θ'`` varies traces a
Mohr circle of radius ``C_p`` centred at

    (c_x, c_y) = ((Z_xyp - Z_yxp)/2, (Z_xxp + Z_yyp)/2).

The same equations apply to the quadrature part (subscript ``q``)
giving an independent Mohr circle.

A number of quantities are *invariant* under axes rotation and so
characterise the tensor itself:

- ``Z^L`` (Lilley's "central impedance"): the distance from the
  origin to the circle centre,
  ``Z^L_p = (1/2) * sqrt[(Z_xxp + Z_yyp)^2 + (Z_xyp - Z_yxp)^2]``.
  Equivalent to a 1-D scale.
- ``λ`` (the "anisotropy angle"): ``λ_p = arcsin(C_p / Z^L_p)``.
  A 2-D measure (zero for 1-D, π/2 for singularity).
- ``μ``: angle at the origin between the line to the circle
  centre and the horizontal axis. ``tan μ_p = (Z_xxp + Z_yyp) /
  (Z_xyp - Z_yxp)``. A per-part 3-D measure (zero for 2-D).
- ``δβ = β_q - β_p``: the angle between the in-phase and
  quadrature radial arms. The "linking" 7th invariant.

These are direct algebraic relatives of the WAL invariants
(Weaver-Agarwal-Lilley 2000): WAL's ``I_1`` is ``Z^L_p``, ``I_2``
is ``Z^L_q``, ``I_3`` is ``sin λ_p``, ``I_4`` is ``sin λ_q``,
``I_5`` is ``sin(μ_p + μ_q)``, ``I_6`` is ``sin(μ_q - μ_p)``, and
``I_7`` is the Bahr-distortion-aware seventh invariant; here we
return the underlying angles directly, since taking the sine
loses information when angles can exceed π/2.

Strike
------
For a clean 2-D regional structure the in-phase and quadrature
Mohr circles both have their centres on the horizontal axis
(``μ_p = μ_q = 0``) and their radial arms are parallel
(``δβ = 0``). The strike is then the rotation that takes either
radial arm onto the horizontal axis, which is ``-β_p / 2`` (with
the usual 90° ambiguity).

For non-2-D data this rotation is still a candidate strike — a
"closest 2-D strike" in the per-part sense. The
:func:`noise_stability_strike` routine perturbs ``Z`` with random
noise at a user-configurable amplitude and reports the spread of
the strike across realisations: stable sites have a tight
distribution; sites where the in-phase Mohr circle's radial arm
is sensitive to noise have a wide one (Lilley 2018).

Dimensionality classification
-----------------------------
Following the visual classifications in Lilley 2018 §"The
depiction of MT tensors using Mohr diagrams":

- **1-D** when both circles reduce to points (``C / Z^L``
  effectively zero in both parts).
- **2-D** when both circle centres lie on the horizontal axis
  (``|μ_p|, |μ_q| ≈ 0``) and the radial arms are parallel
  (``|δβ| ≈ 0``).
- **3-D-distorted** when the circles have non-zero ``μ`` but
  the ``Δβ = (μ_q - μ_p) - δβ`` invariant introduced by
  Lilley 2018 is small — the "Bahr 2-D regional + 3-D galvanic
  distortion" regime.
- **3-D** otherwise.

A user-tunable ``threshold`` controls how strict each comparison
is.

Notational caveat
-----------------
The user-facing :func:`mohr_circle_parameters` packs the 2-D
Mohr-circle centre into a single complex scalar ``c_x + 1j*c_y``
where ``c_x`` is the ``Z'_xy`` axis of the Mohr diagram and
``c_y`` is the ``Z'_xx`` axis. The suffix ``_real`` / ``_imag`` on
the dict keys distinguishes the in-phase and quadrature circles
(Lilley's ``p`` / ``q``), not the components of a complex number.

References
----------
Lilley, F. E. M. (1976). Diagrams for magnetotelluric data.
Geophysics, 41(4), 766-770.

Lilley, F. E. M. (1993). Magnetotelluric analysis using Mohr
circles. Geophysics, 58(10), 1498-1506.

Lilley, F. E. M. (1998). Magnetotelluric tensor decomposition:
Part I, theory for a basic procedure. Geophysics, 63(6), 1885-1897.

Lilley, F. E. M. (2012). Magnetotelluric tensor decomposition:
insights from linear algebra and Mohr diagrams. In *New
Achievements in Geoscience*.

Lilley, F. E. M. (2016). The distortion tensor of
magnetotellurics: a tutorial on some properties. Exploration
Geophysics, 47(2), 85-99.

Lilley, F. E. M. (2018). The magnetotelluric tensor: improved
invariants for its decomposition, especially the 7th. Exploration
Geophysics, 49(5), 622-636.

Lilley, F. E. M. (2020). Magnetotellurics: the CBB or phase
tensor and Bahr's 1988 analysis. Exploration Geophysics, 51(4),
401-421.

Lilley, F. E. M., & Phillips, C. J. E. (2018). A property of the
determinant of a 2x2 tensor relevant to magnetotellurics.
Geophysics, 83(4), A59-A64.

Weaver, J. T., Agarwal, A. K., & Lilley, F. E. M. (2000).
Characterization of the magnetotelluric tensor in terms of its
invariants. Geophysical Journal International, 141, 321-336.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .results import LilleyResult

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


__all__ = [
    "decompose_lilley",
    "mohr_circle_dimensionality",
    "mohr_circle_invariants",
    "mohr_circle_parameters",
    "noise_stability_strike",
]


# ---------------------------------------------------------------------------
# Mohr-circle parameters
# ---------------------------------------------------------------------------


def mohr_circle_parameters(z: np.ndarray) -> dict:
    """Mohr-circle parameters for a complex 2x2 impedance tensor.

    Parameters
    ----------
    z : ndarray
        Complex impedance tensor. Either a single ``(2, 2)`` array
        (one period) or an ``(n_periods, 2, 2)`` array.

    Returns
    -------
    dict
        Keys ``center_real``, ``center_imag``, ``radius_real``,
        ``radius_imag``, ``rotation_real_rad``,
        ``rotation_imag_rad``. Each value is a numpy array of shape
        ``(n_periods,)`` (or a scalar if a single tensor was
        supplied).

        - ``center_real`` and ``center_imag`` are *complex*: the
          real component is the ``Z'_xy`` coordinate of the Mohr
          circle's centre and the imaginary component is the
          ``Z'_xx`` coordinate. The ``_real`` and ``_imag`` suffix
          distinguishes the in-phase and quadrature circles
          (Lilley's ``p`` and ``q`` subscripts), *not* the parts
          of a complex number — see the module docstring.
        - ``radius_real`` and ``radius_imag`` are real positive
          scalars: Lilley's ``C_p`` and ``C_q``.
        - ``rotation_real_rad`` and ``rotation_imag_rad`` are
          real (radians): the per-circle 2-D-strike candidate
          ``-β / 2`` from each part.
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    single = z_arr.ndim == 2
    if single:
        z_arr = z_arr[np.newaxis, :, :]

    z_p = z_arr.real
    z_q = z_arr.imag

    cr_p, ci_p, rad_p, rot_p = _mohr_one_part(z_p)
    cr_q, ci_q, rad_q, rot_q = _mohr_one_part(z_q)

    out = {
        "center_real": cr_p + 1j * ci_p,
        "center_imag": cr_q + 1j * ci_q,
        "radius_real": rad_p,
        "radius_imag": rad_q,
        "rotation_real_rad": rot_p,
        "rotation_imag_rad": rot_q,
    }
    if single:
        out = {k: v[0] if hasattr(v, "shape") else v for k, v in out.items()}
    return out


def _mohr_one_part(
    zr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One Mohr circle (real-valued tensor): centre, radius, strike.

    Returns ``(cx, cy, radius, rotation)``, all shape ``(n,)``.
    """
    z_xx = zr[..., 0, 0]
    z_xy = zr[..., 0, 1]
    z_yx = zr[..., 1, 0]
    z_yy = zr[..., 1, 1]

    c_x = (z_xy - z_yx) / 2.0
    c_y = (z_xx + z_yy) / 2.0
    radius = 0.5 * np.sqrt((z_xx - z_yy) ** 2 + (z_xy + z_yx) ** 2)
    # tan(beta) = (Z_xx - Z_yy) / (Z_xy + Z_yx); strike = -beta / 2.
    beta = np.arctan2(z_xx - z_yy, z_xy + z_yx)
    rotation = -0.5 * beta
    return c_x, c_y, radius, rotation


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def mohr_circle_invariants(z: np.ndarray) -> dict:
    """Rotational invariants from the Mohr-circle representation.

    Returns the seven invariants of Lilley 2018, Figure 4:
    central impedances ``Z^L_p`` and ``Z^L_q`` (1-D scale),
    anisotropy angles ``lambda_p`` and ``lambda_q`` (2-D measures),
    "twist" angles ``mu_p`` and ``mu_q`` (per-part 3-D measures),
    and ``delta beta`` (the 7th, "linking", invariant).

    Relationship to WAL (Weaver-Agarwal-Lilley 2000):
    ``I_1 = Z^L_p``, ``I_2 = Z^L_q``, ``I_3 = sin(lambda_p)``,
    ``I_4 = sin(lambda_q)``, ``I_5 = sin(mu_p + mu_q)``,
    ``I_6 = sin(mu_q - mu_p)``. The seventh WAL invariant is
    Bahr-distortion-aware; here we return ``delta beta`` directly
    so the caller can take whatever sine they want.

    Parameters
    ----------
    z : ndarray
        Complex impedance, ``(2, 2)`` or ``(n, 2, 2)``.

    Returns
    -------
    dict
        Keys: ``central_impedance_real``,
        ``central_impedance_imag``, ``anisotropy_real_rad``,
        ``anisotropy_imag_rad``, ``threed_real_rad``,
        ``threed_imag_rad``, ``delta_beta_rad``.
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    single = z_arr.ndim == 2
    if single:
        z_arr = z_arr[np.newaxis, :, :]

    z_p = z_arr.real
    z_q = z_arr.imag

    zl_p, lam_p, mu_p, beta_p = _invariants_one_part(z_p)
    zl_q, lam_q, mu_q, beta_q = _invariants_one_part(z_q)
    delta_beta = beta_q - beta_p

    # Wrap delta_beta into (-pi, pi].
    delta_beta = (delta_beta + np.pi) % (2 * np.pi) - np.pi

    out = {
        "central_impedance_real": zl_p,
        "central_impedance_imag": zl_q,
        "anisotropy_real_rad": lam_p,
        "anisotropy_imag_rad": lam_q,
        "threed_real_rad": mu_p,
        "threed_imag_rad": mu_q,
        "delta_beta_rad": delta_beta,
    }
    if single:
        out = {k: v[0] for k, v in out.items()}
    return out


def _invariants_one_part(
    zr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-part invariants ``(Z^L, lambda, mu, beta)``."""
    z_xx = zr[..., 0, 0]
    z_xy = zr[..., 0, 1]
    z_yx = zr[..., 1, 0]
    z_yy = zr[..., 1, 1]

    radius = 0.5 * np.sqrt((z_xx - z_yy) ** 2 + (z_xy + z_yx) ** 2)
    central = 0.5 * np.sqrt((z_xx + z_yy) ** 2 + (z_xy - z_yx) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(central > 0, np.clip(radius / central, 0.0, 1.0), 0.0)
    lam = np.arcsin(ratio)
    mu = np.arctan2(z_xx + z_yy, z_xy - z_yx)
    beta = np.arctan2(z_xx - z_yy, z_xy + z_yx)
    return central, lam, mu, beta


# ---------------------------------------------------------------------------
# Noise-stability strike (Lilley 2018)
# ---------------------------------------------------------------------------


def noise_stability_strike(
    z: np.ndarray,
    n_realisations: int = 200,
    noise_fraction: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Lilley noise-stability test on the Mohr-circle strike.

    Adds Gaussian noise of amplitude ``noise_fraction`` times the
    per-component magnitude of ``Z`` (independently to the real
    and imaginary parts, per realisation), recomputes the strike
    from the in-phase Mohr circle (the rotation that takes the
    radial arm onto the horizontal axis), and reports the
    distribution.

    Sites where the strike is well-determined produce a tight
    histogram; sites where the in-phase Mohr circle's radial arm
    is short or near-parallel to the existing horizontal-axis
    direction produce a wide one (Lilley 2018, "noise-stability"
    discussion).

    The strike is wrapped to ``(-pi/2, pi/2]`` (one of two valid
    branches; the GB 90-degree ambiguity is *not* resolved here).

    Parameters
    ----------
    z : ndarray
        Complex impedance tensor or array of tensors. Shape
        ``(2, 2)`` or ``(n_periods, 2, 2)``.
    n_realisations : int, default 200
        Number of noise replicates per period.
    noise_fraction : float, default 0.05
        Standard deviation of the additive Gaussian noise as a
        fraction of the per-component ``|Z|``.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``mean_rad``, ``std_rad``, ``histograms``. ``mean``
        and ``std`` are arrays of shape ``(n_periods,)``;
        ``histograms`` is a list of length ``n_periods`` of
        ``(n_realisations,)`` arrays giving every replicate's
        strike (radians).
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    single = z_arr.ndim == 2
    if single:
        z_arr = z_arr[np.newaxis, :, :]

    rng = np.random.default_rng(seed)
    n_periods = z_arr.shape[0]
    means = np.full(n_periods, np.nan, dtype=np.float64)
    stds = np.full(n_periods, np.nan, dtype=np.float64)
    histograms: list[np.ndarray] = []

    for k in range(n_periods):
        z_k = z_arr[k]
        scale = np.maximum(np.abs(z_k), 1e-300) * noise_fraction
        strikes = np.empty(n_realisations, dtype=np.float64)
        for r in range(n_realisations):
            noise_re = rng.normal(scale=scale, size=(2, 2))
            noise_im = rng.normal(scale=scale, size=(2, 2))
            z_pert = z_k + noise_re + 1j * noise_im
            cx_p, cy_p, _, rot = _mohr_one_part(z_pert.real[np.newaxis, :, :])
            strikes[r] = float(rot[0])
        # Wrap strikes to (-pi/2, pi/2] before stats — GB symmetry
        # ambiguity is one rotation, but unwrapped jumps would
        # inflate the std artefactually.
        strikes_wrapped = (strikes + np.pi / 2) % np.pi - np.pi / 2
        # Use circular statistics to avoid the wrap-around bias.
        c = np.cos(2 * strikes_wrapped)
        s = np.sin(2 * strikes_wrapped)
        means[k] = 0.5 * np.arctan2(np.mean(s), np.mean(c))
        # Approximate std via circular dispersion converted back to
        # the strike angle.
        r_bar = float(np.sqrt(np.mean(s) ** 2 + np.mean(c) ** 2))
        if r_bar > 0:
            stds[k] = 0.5 * np.sqrt(-2.0 * np.log(min(r_bar, 1.0)))
        else:
            stds[k] = np.nan
        histograms.append(strikes_wrapped)

    if single:
        return {
            "mean_rad": float(means[0]),
            "std_rad": float(stds[0]),
            "histograms": histograms[0],
        }
    return {"mean_rad": means, "std_rad": stds, "histograms": histograms}


# ---------------------------------------------------------------------------
# Dimensionality classification
# ---------------------------------------------------------------------------


def mohr_circle_dimensionality(
    z: np.ndarray, threshold: float = 0.05
) -> "list[str] | str":
    """Classify each period as 1-D, 2-D, 3-D-distorted, or 3-D.

    Uses the Mohr-circle geometry summaries as test statistics
    against a single user-chosen ``threshold``:

    - **1-D**: both circles are effectively points,
      ``C_p / Z^L_p < threshold`` and same for ``q``.
    - **2-D**: both circle centres lie on the horizontal axis
      and the radial arms are parallel:
      ``|sin(mu_p)|, |sin(mu_q)|, |sin(delta beta)| < threshold``.
    - **3-D-distorted**: the Lilley-2018 ``Delta beta = (mu_q -
      mu_p) - delta beta`` is small (``|sin(Delta beta)| <
      threshold``); this is the Bahr 2-D-regional +
      3-D-galvanic-distortion regime.
    - **3-D**: otherwise.

    Parameters
    ----------
    z : ndarray
        Complex impedance tensor, ``(2, 2)`` or ``(n, 2, 2)``.
    threshold : float, default 0.05
        Tolerance applied to dimensionless ratios and to the
        sines of the angle invariants.

    Returns
    -------
    list of str (or str if a single tensor was supplied)
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    single = z_arr.ndim == 2
    if single:
        z_arr = z_arr[np.newaxis, :, :]

    inv = mohr_circle_invariants(z_arr)
    zl_p = inv["central_impedance_real"]
    zl_q = inv["central_impedance_imag"]
    lam_p = inv["anisotropy_real_rad"]
    lam_q = inv["anisotropy_imag_rad"]
    mu_p = inv["threed_real_rad"]
    mu_q = inv["threed_imag_rad"]
    delta_beta = inv["delta_beta_rad"]

    # Recover per-part radii from the anisotropy and central impedance.
    radius_p = np.sin(lam_p) * zl_p
    radius_q = np.sin(lam_q) * zl_q
    big_p = np.maximum(zl_p, np.finfo(float).tiny)
    big_q = np.maximum(zl_q, np.finfo(float).tiny)
    one_d_p = (radius_p / big_p) < threshold
    one_d_q = (radius_q / big_q) < threshold

    two_d_mu = (np.abs(np.sin(mu_p)) < threshold) & (
        np.abs(np.sin(mu_q)) < threshold
    )
    two_d_arms = np.abs(np.sin(delta_beta)) < threshold

    delta_b = (mu_q - mu_p) - delta_beta
    delta_b = (delta_b + np.pi) % (2 * np.pi) - np.pi
    distorted = np.abs(np.sin(delta_b)) < threshold

    classes: list[str] = []
    for i in range(z_arr.shape[0]):
        if bool(one_d_p[i] & one_d_q[i]):
            classes.append("1D")
        elif bool(two_d_mu[i] & two_d_arms[i]):
            classes.append("2D")
        elif bool(distorted[i]):
            classes.append("3D-distorted")
        else:
            classes.append("3D")
    if single:
        return classes[0]
    return classes


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decompose_lilley(
    z_obj: "Z",
    *,
    n_realisations: int = 200,
    noise_fraction: float = 0.05,
    threshold: float = 0.05,
    seed: int | None = 42,
) -> LilleyResult:
    """Lilley Mohr-circle distortion analysis on a Z object.

    Computes the Mohr-circle parameters, rotational invariants,
    noise-stability strike statistics, and dimensionality
    classification for every valid period of ``z_obj``. Returns a
    :class:`LilleyResult` with all of these as parallel arrays
    indexed by period.

    Parameters
    ----------
    z_obj : Z
        mtpy ``Z`` object. Periods with a NaN tensor are skipped.
    n_realisations : int, default 200
        Replicates per period for the noise-stability strike test.
    noise_fraction : float, default 0.05
        Standard deviation of the additive noise as a fraction of
        per-component ``|Z|``.
    threshold : float, default 0.05
        Tolerance for the dimensionality classifier.
    seed : int, default 42
        RNG seed for reproducibility of the noise-stability test.

    Returns
    -------
    LilleyResult
    """
    z_full = np.asarray(z_obj.z, dtype=np.complex128)
    frequencies = np.asarray(z_obj.frequency, dtype=np.float64)
    all_periods = 1.0 / frequencies
    sort_idx = np.argsort(all_periods)
    periods = all_periods[sort_idx]
    z_sorted = z_full[sort_idx]

    valid = np.all(np.isfinite(z_sorted.reshape(z_sorted.shape[0], -1)), axis=1)
    z_use = z_sorted[valid]
    p_use = periods[valid]

    params = mohr_circle_parameters(z_use)
    invariants = mohr_circle_invariants(z_use)
    stability = noise_stability_strike(
        z_use,
        n_realisations=n_realisations,
        noise_fraction=noise_fraction,
        seed=seed,
    )
    dim = mohr_circle_dimensionality(z_use, threshold=threshold)

    metadata = {
        "method": "lilley_mohr",
        "n_realisations": int(n_realisations),
        "noise_fraction": float(noise_fraction),
        "threshold": float(threshold),
        "seed": seed,
        "n_valid_periods": int(valid.sum()),
        "n_total_periods": int(valid.size),
    }

    return LilleyResult(
        periods=p_use,
        center_real=params["center_real"],
        center_imag=params["center_imag"],
        radius_real=params["radius_real"],
        radius_imag=params["radius_imag"],
        rotation_real_rad=params["rotation_real_rad"],
        rotation_imag_rad=params["rotation_imag_rad"],
        central_impedance_real=invariants["central_impedance_real"],
        central_impedance_imag=invariants["central_impedance_imag"],
        anisotropy_real_rad=invariants["anisotropy_real_rad"],
        anisotropy_imag_rad=invariants["anisotropy_imag_rad"],
        threed_real_rad=invariants["threed_real_rad"],
        threed_imag_rad=invariants["threed_imag_rad"],
        delta_beta_rad=invariants["delta_beta_rad"],
        strike_mean_rad=stability["mean_rad"],
        strike_std_rad=stability["std_rad"],
        strike_histograms=stability["histograms"],
        dimensionality=dim,
        metadata=metadata,
    )
