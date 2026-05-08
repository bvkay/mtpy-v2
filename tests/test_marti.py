"""Tests for the Marti WALDIM dimensionality module.

Covers the WAL invariants on clean 2-D and 3-D synthetics, the
WALDIM classifier on the four required regimes (1-D / 2-D /
3-D-distorted-2-D / pure 3-D), and per-period strike candidates.

References
----------
Marti, A., Queralt, P., & Ledo, J. (2009). WALDIM: A code for the
dimensionality analysis of magnetotelluric data using the
rotational invariants of the magnetotelluric tensor. *Computers
& Geosciences* 35, 2295-2303.

Weaver, J. T., Agarwal, A. K., & Lilley, F. E. M. (2000).
Characterization of the magnetotelluric tensor in terms of its
invariants. *Geophysical Journal International* 141, 321-336.
"""

from __future__ import annotations

import numpy as np

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    MartiResult,
    _estim_imp,
    decompose_marti,
    wal_invariants,
    waldim_dimensionality,
)


def _two_d_tensor(
    a: complex = 1.0 + 2.0j,
    b: complex = 5.0 - 1.5j,
    theta_rad: float = 0.0,
) -> np.ndarray:
    """Anti-diagonal 2-D regional tensor at strike ``theta_rad``."""
    z_strike = np.array([[0.0, a], [-b, 0.0]], dtype=np.complex128)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    return R @ z_strike @ R.T


def _three_d_2d_tensor(
    theta_deg: float = 30.0,
    twist_deg: float = 12.0,
    shear_deg: float = 8.0,
) -> np.ndarray:
    """3-D / 2-D distorted tensor: GB forward model with non-trivial
    twist and shear, gain 1. ``I_7`` should be (close to) zero
    because the distortion is real-valued, but ``I_5`` and / or
    ``I_6`` are non-zero from the rotation."""
    a = 100.0 * np.exp(1j * np.radians(60.0))
    b = 10.0 * np.exp(1j * np.radians(45.0))
    twist_tan = np.tan(np.radians(twist_deg))
    shear_tan = np.tan(np.radians(shear_deg))
    return _estim_imp(a, b, twist_tan, shear_tan, np.radians(theta_deg))


def _three_d_tensor() -> np.ndarray:
    """Hand-crafted "clearly 3-D" tensor where the in-phase and
    quadrature radial arms are non-parallel by construction.

    The GB forward model produces tensors with ``I_7 ~ 0`` (the
    distortion is real), so a synthetic 3-D regional structure
    must be constructed differently. Here we take a 2-D anti-
    diagonal regional tensor in the strike frame, and apply a
    *different* real distortion to the in-phase and quadrature
    parts — effectively making the distortion imaginary, which
    is non-physical for galvanic distortion but produces the
    desired ``I_7 != 0`` signature.
    """
    z_strike_p = np.array([[0.0, 1.0], [-3.0, 0.0]])
    z_strike_q = np.array([[0.0, 2.0], [-1.0, 0.0]])
    # Real and quadrature parts get different "twists" of their
    # own, breaking the GB assumption that they share a single
    # real distortion matrix.
    twist_p = np.tan(np.radians(15.0))
    twist_q = np.tan(np.radians(-5.0))
    T_p = np.array([[1.0, -twist_p], [twist_p, 1.0]]) / np.sqrt(1 + twist_p**2)
    T_q = np.array([[1.0, -twist_q], [twist_q, 1.0]]) / np.sqrt(1 + twist_q**2)
    z_p = T_p @ z_strike_p
    z_q = T_q @ z_strike_q
    return z_p + 1j * z_q


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_wal_invariants_clean_2D():
    """A clean 2-D tensor: ``I_3`` and / or ``I_4`` non-zero
    (from the off-diagonal asymmetry), ``I_5`` and ``I_6`` zero
    (centres on the horizontal axis), ``Q`` small (radial arms
    parallel), ``I_7`` either zero or undefined.

    Using a tensor with substantial difference between ``a`` and
    ``b`` so the anisotropy radii ``C_p`` / ``C_q`` are well
    above any reasonable threshold.
    """
    z_2d = _two_d_tensor(a=1.0, b=5.0, theta_rad=np.radians(30.0))
    inv = wal_invariants(z_2d)

    # I3 or I4 non-zero (the 2-D off-diagonal asymmetry).
    assert max(abs(inv["I3"]), abs(inv["I4"])) > 0.1, (
        f"expected non-trivial 2-D anisotropy, got I3={inv['I3']:.4f}, "
        f"I4={inv['I4']:.4f}"
    )

    # I5, I6 effectively zero for pure 2-D.
    assert abs(inv["I5"]) < 1e-9, (
        f"clean 2-D should give I5 = 0; got {inv['I5']:.6e}"
    )
    assert abs(inv["I6"]) < 1e-9, (
        f"clean 2-D should give I6 = 0; got {inv['I6']:.6e}"
    )

    # I7 either undefined (Q ~ 0) or numerically zero.
    assert np.isnan(inv["I7"]) or abs(inv["I7"]) < 1e-9, (
        f"clean 2-D should give I7 ~ 0 or nan; got {inv['I7']!r}"
    )


def test_wal_invariants_3D():
    """A 3-D synthetic: ``I_7`` non-zero (the in-phase and
    quadrature radial arms are not parallel)."""
    z_3d = _three_d_tensor()
    inv = wal_invariants(z_3d)
    assert not np.isnan(inv["I7"]), (
        "3-D synthetic should have well-defined I7"
    )
    assert abs(inv["I7"]) > 0.1, (
        f"3-D synthetic should give |I7| > 0.1; got {inv['I7']:.4f}"
    )


def test_waldim_classification_clean_2D():
    """A clean 2-D tensor classifies as ``2`` (2-D)."""
    z_2d = _two_d_tensor(a=1.0, b=5.0, theta_rad=np.radians(30.0))
    cls = waldim_dimensionality(z_2d, threshold=0.15)
    assert cls == 2, f"expected 2-D classification (2), got {cls}"


def test_waldim_classification_3D():
    """A 3-D synthetic with ``I_7 != 0`` classifies as ``5``
    (3-D)."""
    z_3d = _three_d_tensor()
    cls = waldim_dimensionality(z_3d, threshold=0.15)
    assert cls == 5, f"expected 3-D classification (5), got {cls}"


def test_waldim_classification_3D_2D():
    """A 3-D-distorted-2-D synthetic (GB forward model with
    non-trivial twist and shear) classifies as one of the
    Marti-2009 3-D / 2-D sub-cases:

    - ``3``: 3-D / 2-D twist-only (``I_5 != 0``, ``I_6 = 0``,
      ``I_7 = 0``, ``Q != 0``).
    - ``4``: 3-D / 2-D general (``I_5 != 0``, ``I_6 != 0``,
      ``I_7 = 0``).
    - ``7``: 3-D / 1-D-2-D indistinguishable (``I_5 != 0``,
      ``I_6 = 0``, ``Q = 0``).

    Which sub-case the classifier reports depends on the precise
    values of ``I_5``, ``I_6``, and ``Q`` relative to the
    threshold. For the GB forward model with a single shared
    real distortion, ``I_7`` is always zero (the in-phase and
    quadrature radial arms are parallel by construction), so we
    never expect a 3-D classification (``5``).
    """
    z = _three_d_2d_tensor(theta_deg=30.0, twist_deg=15.0, shear_deg=10.0)
    cls = waldim_dimensionality(z, threshold=0.15)
    assert cls in (3, 4, 7), (
        f"expected 3-D / 2-D classification (3, 4, or 7), got {cls}"
    )


def test_marti_strikes_for_2D():
    """For a 2-D synthetic at known strike, ``St_3`` and ``St_4``
    agree on the input strike (mod 90°).

    The Mohr-fold strikes ``-beta_p / 2`` and ``-beta_q / 2``
    have the GB 90-degree ambiguity; we check that the recovered
    angle matches one of the two equivalent branches.
    """
    true_strike_deg = 30.0
    z_2d = _two_d_tensor(theta_rad=np.radians(true_strike_deg))

    n_freqs = 4
    z = np.tile(z_2d, (n_freqs, 1, 1))
    sigma = np.maximum(0.01 * np.abs(z), 1e-15)
    z_obj = Z(
        z=z,
        z_error=sigma,
        frequency=np.array([1.0, 0.5, 0.1, 0.01]),
    )

    res = decompose_marti(z_obj, threshold=0.15)
    assert isinstance(res, MartiResult)

    def _wrap_to_strike(deg: float) -> float:
        # Bring strike into [0, 90) accounting for the GB
        # 90-degree ambiguity.
        return float(deg) % 90.0

    for k in range(n_freqs):
        s_real = float(np.degrees(res.strike_2d_real_rad[k]))
        s_imag = float(np.degrees(res.strike_2d_imag_rad[k]))
        diff_real = (
            min(
                abs(_wrap_to_strike(s_real) - true_strike_deg),
                90.0 - abs(_wrap_to_strike(s_real) - true_strike_deg),
            )
        )
        diff_imag = (
            min(
                abs(_wrap_to_strike(s_imag) - true_strike_deg),
                90.0 - abs(_wrap_to_strike(s_imag) - true_strike_deg),
            )
        )
        assert diff_real < 1.0, (
            f"period {k}: in-phase strike {s_real:.2f}° not within "
            f"1° of true 30° (mod 90); diff = {diff_real:.4f}"
        )
        assert diff_imag < 1.0, (
            f"period {k}: quadrature strike {s_imag:.2f}° not "
            f"within 1° of true 30° (mod 90); diff = {diff_imag:.4f}"
        )
