"""Marti WALDIM dimensionality analysis using the WAL invariants.

The Marti intellectual tradition (Marti et al. 2004, 2005, 2009,
2010, 2013) takes the seven rotational invariants of
Weaver-Agarwal-Lilley (WAL, 2000) and uses them as a
dimensionality classifier — the WALDIM code of Marti et al. 2009.
Each MT impedance tensor (per-period) is classified into a
dimensionality regime by direct sign tests on the WAL invariants
against a single user-tunable threshold.

The seven WAL invariants
-------------------------
Following the algebra in Lilley 2018 (which is identical to
Weaver-Agarwal-Lilley 2000 up to notation) and Marti et al. 2009
Section 1, the impedance tensor is decomposed via Mohr-circle
algebra. With ``Z = Z_p + i*Z_q``, define for each part the
"central impedance" ``Z^L`` (Mohr-circle-centre distance from
origin) and the "anisotropy radius" ``C`` (Mohr-circle radius):

    Z^L_p = (1/2) sqrt[(Z_xxp + Z_yyp)^2 + (Z_xyp - Z_yxp)^2]
    Z^L_q = (1/2) sqrt[(Z_xxq + Z_yyq)^2 + (Z_xyq - Z_yxq)^2]
    C_p   = (1/2) sqrt[(Z_xxp - Z_yyp)^2 + (Z_xyp + Z_yxp)^2]
    C_q   = (1/2) sqrt[(Z_xxq - Z_yyq)^2 + (Z_xyq + Z_yxq)^2]

and the angles

    tan mu_p  = (Z_xxp + Z_yyp) / (Z_xyp - Z_yxp)
    tan mu_q  = (Z_xxq + Z_yyq) / (Z_xyq - Z_yxq)
    tan beta_p = (Z_xxp - Z_yyp) / (Z_xyp + Z_yxp)
    tan beta_q = (Z_xxq - Z_yyq) / (Z_xyq + Z_yxq)
    delta_beta = beta_q - beta_p

The seven independent WAL invariants are then

    I_1 = Z^L_p
    I_2 = Z^L_q
    I_3 = C_p / Z^L_p             = sin(lambda_p)
    I_4 = C_q / Z^L_q             = sin(lambda_q)
    I_5 = sin(mu_p + mu_q)
    I_6 = sin(mu_q - mu_p)
    I_7 = [sin(mu_q - mu_p) - sin(lambda_p) sin(lambda_q) sin(delta_beta)] / Q

with auxiliary

    Q^2 = sin^2(lambda_p) + sin^2(lambda_q)
        - 2 sin(lambda_p) sin(lambda_q) cos(mu_q - mu_p - delta_beta)

``Q`` is itself a useful invariant: when ``Q`` is small, ``I_7``
is undefined (``0/0``), and the data is consistent with the
Bahr-2-D regime (Bahr 1988).

WALDIM classification (Marti et al. 2009 Table 1)
-------------------------------------------------
With a threshold ``tau`` (typically 0.1-0.2; the WALDIM default
is 0.15), each period is classified as:

* **0 — undetermined.** Cannot be assigned to any of the
  categories below.
* **1 — 1-D.** ``I_3, I_4, I_5, I_6`` all below threshold.
* **2 — 2-D.** ``I_3`` or ``I_4`` non-zero; ``I_5, I_6`` below
  threshold; ``I_7`` *or* ``Q`` below threshold.
* **3 — 3-D / 2-D twist-only** (Marti 2009 case 3a). ``I_3`` or
  ``I_4`` non-zero; ``I_5`` non-zero; ``I_6`` below threshold;
  ``I_7`` below threshold.
* **4 — 3-D / 2-D general** (case 4). ``I_3`` or ``I_4``
  non-zero; ``I_5, I_6`` non-zero; ``I_7`` below threshold.
* **5 — 3-D.** ``I_7`` non-zero. (Distortion may also be
  present; the WAL invariants alone cannot discriminate.)
* **6 — 3-D / 2-D with diagonal regional tensor** (case 3c).
  ``I_3`` or ``I_4`` non-zero; ``I_5, I_6`` below threshold;
  ``I_7`` *or* ``Q`` below threshold; with the additional Bahr
  ``xi_4`` and ``eta_4`` test that distinguishes a diagonal
  regional tensor from the pure-2-D case 2. **Phase 1 note**:
  this code currently treats case 3c as part of case 2 because
  the ``xi_4`` / ``eta_4`` test requires the Bahr-style
  decomposition to be already computed; the distinction is
  flagged as a follow-up TODO.
* **7 — 3-D / 1-D-2-D indistinguishable** (case 3b). ``I_3`` or
  ``I_4`` non-zero; ``I_5`` non-zero; ``I_6`` below threshold;
  ``Q`` below threshold (so ``I_7`` is undefined).

Strike and distortion
---------------------
For 2-D and 3-D / 2-D classifications, the per-period strike and
Smith (1995) distortion angles are computed and reported on the
:class:`MartiResult`. For pure 2-D the in-phase and quadrature
Mohr folds give independent strike candidates ``St_3 = -beta_p
/ 2`` and ``St_4 = -beta_q / 2`` (radians); for 3-D / 2-D the
Bahr equal-phase condition is solved numerically to give
``St_5``. The Smith distortion angles ``f1 = (twist + shear) /
2`` and ``f2 = (twist - shear) / 2`` plus ``twist`` and ``shear``
themselves are exposed for the regimes where they are
well-defined.

Phase 1 scope
-------------
The current implementation:

* covers the WAL invariants in full,
* covers the Marti 2009 classification cases 1, 2, 3a, 4, 5,
  and 7 (all the ones that depend solely on the WAL invariants
  and a single threshold),
* computes ``St_3``, ``St_4`` from the Mohr folds and ``St_5``
  from the Bahr equal-phase condition,
* leaves the 2-D-vs-case-3c (diagonal regional) discriminator
  and the explicit twist / shear angle decomposition as
  follow-up work — ``twist`` and ``shear`` are returned as NaN
  with a TODO marker. ``f1`` / ``f2`` are computed
  approximately from ``mu_p`` and ``mu_q``.

References
----------
Bahr, K. (1988). Interpretation of the magnetotelluric impedance
tensor: regional induction and local telluric distortion.
Journal of Geophysics, 62(2), 119-127.

Booker, J. R. (2014). The magnetotelluric phase tensor: a
critical review. Surveys in Geophysics, 35, 7-40.

Marti, A., Queralt, P., Jones, A. G., & Ledo, J. (2005).
Improving Bahr's invariant parameters using the WAL approach.
Geophysical Journal International, 163, 38-41.

Marti, A., Queralt, P., & Ledo, J. (2009). WALDIM: A code for the
dimensionality analysis of magnetotelluric data using the
rotational invariants of the magnetotelluric tensor. Computers &
Geosciences, 35, 2295-2303.

Marti, A., Queralt, P., Ledo, J., & Farquharson, C. (2010).
Dimensionality imprint of electrical anisotropy in magnetotelluric
responses. Physics of the Earth and Planetary Interiors, 182,
139-151.

Smith, J. T. (1995). Understanding telluric distortion matrices.
Geophysical Journal International, 122, 219-226.

Weaver, J. T., Agarwal, A. K., & Lilley, F. E. M. (2000).
Characterization of the magnetotelluric tensor in terms of its
invariants. Geophysical Journal International, 141, 321-336.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from .results import MartiResult

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


__all__ = [
    "decompose_marti",
    "wal_invariants",
    "waldim_dimensionality",
]


# ---------------------------------------------------------------------------
# WAL invariants
# ---------------------------------------------------------------------------


def wal_invariants(z: np.ndarray) -> dict:
    """The seven Weaver-Agarwal-Lilley (2000) rotational invariants.

    Parameters
    ----------
    z : ndarray
        Complex impedance tensor: ``(2, 2)`` (one period) or
        ``(n_periods, 2, 2)``.

    Returns
    -------
    dict
        Keys ``I1`` through ``I7`` and ``Q``. Values are scalars
        for a single tensor input or ``(n_periods,)`` arrays
        otherwise.

        * ``I1``, ``I2`` carry units of ``Z`` (impedance):
          they are the "central impedances"
          ``Z^L_p`` and ``Z^L_q``, the distances from the origin
          to the in-phase and quadrature Mohr-circle centres.
        * ``I3``, ``I4`` are dimensionless: the sines of the
          anisotropy angles ``lambda_p`` and ``lambda_q``,
          equivalent to ``C / Z^L`` per part.
        * ``I5``, ``I6`` are dimensionless: ``sin(mu_p + mu_q)``
          and ``sin(mu_q - mu_p)`` with ``mu`` the per-part 3-D
          measure (``Mohr-centre direction from origin``).
        * ``I7`` is dimensionless: the Bahr-aware seventh
          invariant. Undefined when ``Q`` is zero; returned as
          ``nan`` in that case.
        * ``Q`` is dimensionless: the auxiliary that controls
          whether ``I_7`` is well-defined.
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    single = z_arr.ndim == 2
    if single:
        z_arr = z_arr[np.newaxis, :, :]

    z_p = z_arr.real
    z_q = z_arr.imag

    z_l_p, c_p, mu_p, beta_p = _per_part_quantities(z_p)
    z_l_q, c_q, mu_q, beta_q = _per_part_quantities(z_q)

    delta_beta = beta_q - beta_p
    delta_beta = (delta_beta + np.pi) % (2 * np.pi) - np.pi

    with np.errstate(divide="ignore", invalid="ignore"):
        sin_lam_p = np.where(z_l_p > 0, np.clip(c_p / z_l_p, -1.0, 1.0), 0.0)
        sin_lam_q = np.where(z_l_q > 0, np.clip(c_q / z_l_q, -1.0, 1.0), 0.0)
    lam_p = np.arcsin(sin_lam_p)
    lam_q = np.arcsin(sin_lam_q)

    i1 = z_l_p
    i2 = z_l_q
    i3 = sin_lam_p
    i4 = sin_lam_q
    i5 = np.sin(mu_p + mu_q)
    i6 = np.sin(mu_q - mu_p)

    q_sq = (
        np.sin(lam_p) ** 2
        + np.sin(lam_q) ** 2
        - 2.0 * np.sin(lam_p) * np.sin(lam_q) * np.cos(mu_q - mu_p - delta_beta)
    )
    q = np.sqrt(np.maximum(q_sq, 0.0))

    numerator = np.sin(mu_q - mu_p) - np.sin(lam_p) * np.sin(lam_q) * np.sin(delta_beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        i7 = np.where(q > 0, numerator / q, np.nan)

    out = {
        "I1": i1,
        "I2": i2,
        "I3": i3,
        "I4": i4,
        "I5": i5,
        "I6": i6,
        "I7": i7,
        "Q": q,
    }
    if single:
        out = {k: float(v[0]) for k, v in out.items()}
    return out


def _per_part_quantities(
    zr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-part Mohr-circle quantities ``(Z^L, C, mu, beta)``.

    All four are computed from a real-valued 2x2 tensor (either
    the in-phase or quadrature part of an MT tensor).
    """
    z_xx = zr[..., 0, 0]
    z_xy = zr[..., 0, 1]
    z_yx = zr[..., 1, 0]
    z_yy = zr[..., 1, 1]

    z_l = 0.5 * np.sqrt((z_xx + z_yy) ** 2 + (z_xy - z_yx) ** 2)
    c = 0.5 * np.sqrt((z_xx - z_yy) ** 2 + (z_xy + z_yx) ** 2)
    mu = np.arctan2(z_xx + z_yy, z_xy - z_yx)
    beta = np.arctan2(z_xx - z_yy, z_xy + z_yx)
    return z_l, c, mu, beta


# ---------------------------------------------------------------------------
# WALDIM dimensionality classification
# ---------------------------------------------------------------------------


def waldim_dimensionality(
    z: np.ndarray, threshold: float = 0.15
) -> "int | np.ndarray":
    """WALDIM classification per Marti et al. (2009).

    Parameters
    ----------
    z : ndarray
        Complex impedance, ``(2, 2)`` or ``(n, 2, 2)``.
    threshold : float, default 0.15
        Sine-of-angle tolerance for sign tests on the
        dimensionless invariants ``I_3`` ... ``I_7`` and ``Q``.
        Values below ``threshold`` are treated as zero. The
        Marti-2009 default is 0.15; the original WAL paper's
        worked example used 0.1.

    Returns
    -------
    int (single tensor) or ndarray of int (array)
        Per-period classification — see the :mod:`.marti` module
        docstring for the meaning of each code (0 = undetermined,
        1-7 = the WALDIM categories).
    """
    inv = wal_invariants(z)
    is_single = isinstance(inv["I3"], float)
    if is_single:
        inv = {k: np.asarray([v]) for k, v in inv.items()}

    i3, i4, i5, i6, i7, q = (
        np.abs(inv["I3"]),
        np.abs(inv["I4"]),
        np.abs(inv["I5"]),
        np.abs(inv["I6"]),
        np.abs(inv["I7"]),
        np.abs(inv["Q"]),
    )

    n = i3.size
    out = np.zeros(n, dtype=np.int64)

    i3_or_i4 = (i3 >= threshold) | (i4 >= threshold)
    i5_zero = i5 < threshold
    i6_zero = i6 < threshold
    i7_zero = (i7 < threshold) | np.isnan(i7)
    q_zero = q < threshold

    is_1d = (~i3_or_i4) & i5_zero & i6_zero
    out[is_1d] = 1

    is_2d = (~is_1d) & i3_or_i4 & i5_zero & i6_zero & (i7_zero | q_zero)
    out[is_2d] = 2

    is_3d_2d_twist = (
        (~is_1d)
        & (~is_2d)
        & i3_or_i4
        & (~i5_zero)
        & i6_zero
        & i7_zero
        & (~q_zero)
    )
    out[is_3d_2d_twist] = 3

    is_3d_2d_general = (
        (~is_1d) & (~is_2d) & (~is_3d_2d_twist) & i3_or_i4 & (~i5_zero) & (~i6_zero) & i7_zero
    )
    out[is_3d_2d_general] = 4

    is_3d_1d_2d = (
        (~is_1d)
        & (~is_2d)
        & (~is_3d_2d_twist)
        & (~is_3d_2d_general)
        & i3_or_i4
        & (~i5_zero)
        & i6_zero
        & q_zero
    )
    out[is_3d_1d_2d] = 7

    is_3d = (
        (~is_1d)
        & (~is_2d)
        & (~is_3d_2d_twist)
        & (~is_3d_2d_general)
        & (~is_3d_1d_2d)
        & (i7 >= threshold)
        & (~np.isnan(i7))
    )
    out[is_3d] = 5

    if is_single:
        return int(out[0])
    return out


# ---------------------------------------------------------------------------
# Strike helpers
# ---------------------------------------------------------------------------


def _strike_2d_real(z: np.ndarray) -> np.ndarray:
    """In-phase 2-D strike candidate ``-beta_p / 2`` (radians)."""
    z_p = z.real if z.ndim == 3 else z[np.newaxis, :, :].real
    *_, beta_p = _per_part_quantities(z_p)
    return -0.5 * beta_p


def _strike_2d_imag(z: np.ndarray) -> np.ndarray:
    """Quadrature 2-D strike candidate ``-beta_q / 2`` (radians)."""
    z_q = z.imag if z.ndim == 3 else z[np.newaxis, :, :].imag
    *_, beta_q = _per_part_quantities(z_q)
    return -0.5 * beta_q


def _bahr_strike_one_tensor(z_2x2: np.ndarray) -> float:
    """Bahr equal-phase strike for one tensor, by numerical search.

    Solves Bahr (1988) eq 9: rotation angle ``alpha`` such that
    ``Z'_xxp / Z'_yxp = Z'_xxq / Z'_yxq``. Two solutions exist
    that differ by 90 degrees; we return the one closest to zero
    in ``[-pi/2, pi/2]``.

    Returns ``nan`` if no real-axis root is found in the search
    interval (e.g. when the data are far from the Bahr-2-D model).
    """

    def residual(alpha: float) -> float:
        c, s = np.cos(alpha), np.sin(alpha)
        r = np.array([[c, s], [-s, c]])
        z_rot = r @ z_2x2 @ r.T
        zxxp = z_rot[0, 0].real
        zxxq = z_rot[0, 0].imag
        zyxp = z_rot[1, 0].real
        zyxq = z_rot[1, 0].imag
        # Avoid division by zero — if either denominator is zero,
        # treat the residual as a large value at this alpha.
        if abs(zyxp) < 1e-15 or abs(zyxq) < 1e-15:
            return float("inf")
        return zxxp / zyxp - zxxq / zyxq

    # Sweep the unit half-circle for sign changes.
    alphas = np.linspace(-np.pi / 2, np.pi / 2, 91)
    resids = np.array([residual(a) for a in alphas])
    finite = np.isfinite(resids)
    if not finite.any():
        return float("nan")
    sign_change = np.where(
        np.diff(np.sign(resids[finite])) != 0
    )[0]
    if sign_change.size == 0:
        return float("nan")
    idx = sign_change[0]
    valid_alphas = alphas[finite]
    a_lo, a_hi = valid_alphas[idx], valid_alphas[idx + 1]
    try:
        return float(brentq(residual, a_lo, a_hi))
    except (ValueError, RuntimeError):
        return float("nan")


def _strike_3d_2d(z: np.ndarray) -> np.ndarray:
    """Per-period Bahr strike (radians); ``nan`` where no root."""
    z_arr = z if z.ndim == 3 else z[np.newaxis, :, :]
    out = np.full(z_arr.shape[0], np.nan, dtype=np.float64)
    for k in range(z_arr.shape[0]):
        out[k] = _bahr_strike_one_tensor(z_arr[k])
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decompose_marti(
    z_obj: "Z", *, threshold: float = 0.15
) -> MartiResult:
    """WALDIM dimensionality analysis on a Z object.

    Computes the WAL invariants, classifies each period per
    Marti et al. (2009), and reports per-period strike candidates
    (``St_3``, ``St_4``, ``St_5``) and approximate distortion
    angles for the regimes where they are well-defined.

    Parameters
    ----------
    z_obj : Z
        mtpy ``Z`` object. Periods with a NaN tensor are skipped.
    threshold : float, default 0.15
        WALDIM threshold (Marti 2009 default).

    Returns
    -------
    MartiResult
    """
    z_full = np.asarray(z_obj.z, dtype=np.complex128)
    frequencies = np.asarray(z_obj.frequency, dtype=np.float64)
    all_periods = 1.0 / frequencies
    sort_idx = np.argsort(all_periods)
    periods = all_periods[sort_idx]
    z_sorted = z_full[sort_idx]
    valid = np.all(
        np.isfinite(z_sorted.reshape(z_sorted.shape[0], -1)), axis=1
    )
    z_use = z_sorted[valid]
    p_use = periods[valid]

    inv = wal_invariants(z_use)
    dim = waldim_dimensionality(z_use, threshold=threshold)

    strike_real = _strike_2d_real(z_use)
    strike_imag = _strike_2d_imag(z_use)
    strike_bahr = _strike_3d_2d(z_use)

    # f1 / f2: approximate using mu_p and mu_q. Per Marti 2005:
    # f1 = (twist + shear) / 2 ~ (mu_p + mu_q) / 4 in the Bahr-2-D
    # regime; this is a documented Phase-1 approximation. Twist
    # and shear individually require the full Smith decomposition,
    # which is left as a follow-up TODO and reported as nan.
    z_p = z_use.real
    z_q = z_use.imag
    _, _, mu_p, _ = _per_part_quantities(z_p)
    _, _, mu_q, _ = _per_part_quantities(z_q)
    f1 = (mu_p + mu_q) / 4.0
    f2 = (mu_p - mu_q) / 4.0

    is_2d_3d2d = np.isin(dim, (2, 3, 4))
    f1_out = np.where(is_2d_3d2d, f1, np.nan)
    f2_out = np.where(is_2d_3d2d, f2, np.nan)
    twist_out = np.full_like(f1_out, np.nan)
    shear_out = np.full_like(f1_out, np.nan)

    metadata = {
        "method": "marti_waldim",
        "threshold": float(threshold),
        "n_valid_periods": int(valid.sum()),
        "n_total_periods": int(valid.size),
        "twist_shear_status": (
            "twist and shear angles individually require the full "
            "Smith (1995) distortion decomposition; deferred as "
            "Phase-2 work. The f1 / f2 linear combinations are "
            "approximated from mu_p and mu_q."
        ),
    }

    return MartiResult(
        periods=p_use,
        I1=inv["I1"],
        I2=inv["I2"],
        I3=inv["I3"],
        I4=inv["I4"],
        I5=inv["I5"],
        I6=inv["I6"],
        I7=inv["I7"],
        Q=inv["Q"],
        dimensionality=dim,
        strike_2d_real_rad=strike_real,
        strike_2d_imag_rad=strike_imag,
        strike_3d_2d_rad=strike_bahr,
        f1_rad=f1_out,
        f2_rad=f2_out,
        twist_rad=twist_out,
        shear_rad=shear_out,
        metadata=metadata,
    )
