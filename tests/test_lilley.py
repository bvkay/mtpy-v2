"""Tests for the Lilley Mohr-circle distortion module.

Covers algebraic identities on synthetic 2-D and 3-D tensors,
WAL-invariant equivalence, the noise-stability strike, and the
dimensionality classifier.

References
----------
Lilley, F. E. M. (2018). The magnetotelluric tensor: improved
invariants for its decomposition, especially the 7th.
*Exploration Geophysics* 49, 622-636.

Weaver, J. T., Agarwal, A. K., & Lilley, F. E. M. (2000).
Characterization of the magnetotelluric tensor in terms of its
invariants. *Geophysical Journal International* 141, 321-336.
"""

from __future__ import annotations

import numpy as np

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    LilleyResult,
    _estim_imp,
    decompose_lilley,
)
from mtpy.core.transfer_function.z_analysis.decomposition.lilley import (
    mohr_circle_dimensionality,
    mohr_circle_invariants,
    mohr_circle_parameters,
    noise_stability_strike,
)


def _two_d_tensor(
    a: complex = 1.0 + 2.0j,
    b: complex = 5.0 - 1.5j,
    theta_rad: float = 0.0,
) -> np.ndarray:
    """A 2-D regional tensor at strike ``theta_rad``.

    In the strike frame the tensor is anti-diagonal with off-
    diagonals ``a`` and ``-b``. After rotation by ``theta_rad``
    we land in the measurement frame.
    """
    z_strike = np.array([[0.0, a], [-b, 0.0]], dtype=np.complex128)
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]])
    return R @ z_strike @ R.T


def _three_d_tensor() -> np.ndarray:
    """A 3-D distorted tensor built via the GB forward model.

    Non-trivial twist and shear yield a Mohr circle whose centre
    is off the horizontal axis (mu non-zero) and whose in-phase /
    quadrature radial arms are not parallel.
    """
    a = 100.0 * np.exp(1j * np.radians(60.0))
    b = 10.0 * np.exp(1j * np.radians(45.0))
    twist_tan = np.tan(np.radians(15.0))
    shear_tan = np.tan(np.radians(10.0))
    return _estim_imp(a, b, twist_tan, shear_tan, np.radians(30.0))


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_mohr_circle_2D_tensor():
    """A clean 2-D tensor: centres on the horizontal axis, radius
    matches Lilley 2018 eq. 27.

    For a tensor anti-diagonal in the strike frame with
    off-diagonals ``a`` and ``-b``, the measurement-frame trace
    ``Zxxp + Zyyp`` is zero (and same for the quadrature part),
    so the in-phase and quadrature Mohr-circle centres lie on
    the horizontal axis: their packed-complex form has zero
    imaginary component. The simpler "Cp = |Zxy + Zyx| / 2"
    reading only holds in the strike frame; here we check the
    full Lilley formula
    ``Cp = (1/2) * sqrt[(Zxxp - Zyyp)^2 + (Zxyp + Zyxp)^2]``,
    which is rotationally invariant.
    """
    z_2d = _two_d_tensor(theta_rad=np.radians(30.0))
    params = mohr_circle_parameters(z_2d)

    # Centres on the horizontal axis (imag component zero for 2-D).
    assert abs(params["center_real"].imag) < 1e-12
    assert abs(params["center_imag"].imag) < 1e-12

    # Radius matches the full Lilley 2018 eq. 27.
    z_p = z_2d.real
    expected_p_full = 0.5 * np.sqrt(
        (z_p[0, 0] - z_p[1, 1]) ** 2 + (z_p[0, 1] + z_p[1, 0]) ** 2
    )
    z_q = z_2d.imag
    expected_q_full = 0.5 * np.sqrt(
        (z_q[0, 0] - z_q[1, 1]) ** 2 + (z_q[0, 1] + z_q[1, 0]) ** 2
    )
    assert np.isclose(params["radius_real"], expected_p_full)
    assert np.isclose(params["radius_imag"], expected_q_full)

    # In the strike frame (theta = 0) the simpler form
    # Cp = |Zxy + Zyx| / 2 also holds. Verify on a strike-frame
    # tensor for completeness.
    z_strike = _two_d_tensor(theta_rad=0.0)
    p2 = mohr_circle_parameters(z_strike)
    cp_strike_simple = 0.5 * abs(z_strike[0, 1].real + z_strike[1, 0].real)
    cq_strike_simple = 0.5 * abs(z_strike[0, 1].imag + z_strike[1, 0].imag)
    assert np.isclose(p2["radius_real"], cp_strike_simple)
    assert np.isclose(p2["radius_imag"], cq_strike_simple)


def test_mohr_circle_invariants_match_wal():
    """The Mohr-circle invariants reproduce the WAL definitions.

    WAL (Weaver-Agarwal-Lilley 2000) define their I1..I6 in terms
    of tensor elements; we recompute them by hand here and check
    they match the Mohr-circle quantities exactly. The seventh
    Bahr-aware WAL invariant is *not* checked — see the module
    docstring for why we return ``delta_beta`` directly instead.
    """
    z = _three_d_tensor()
    inv = mohr_circle_invariants(z)
    z_p = z.real
    z_q = z.imag

    def _zl(zr):
        return 0.5 * np.sqrt(
            (zr[0, 0] + zr[1, 1]) ** 2 + (zr[0, 1] - zr[1, 0]) ** 2
        )

    def _c(zr):
        return 0.5 * np.sqrt(
            (zr[0, 0] - zr[1, 1]) ** 2 + (zr[0, 1] + zr[1, 0]) ** 2
        )

    i1 = _zl(z_p)
    i2 = _zl(z_q)
    i3 = _c(z_p) / i1
    i4 = _c(z_q) / i2
    mu_p = float(np.arctan2(z_p[0, 0] + z_p[1, 1], z_p[0, 1] - z_p[1, 0]))
    mu_q = float(np.arctan2(z_q[0, 0] + z_q[1, 1], z_q[0, 1] - z_q[1, 0]))
    i5 = float(np.sin(mu_p + mu_q))
    i6 = float(np.sin(mu_q - mu_p))

    assert np.isclose(inv["central_impedance_real"], i1)
    assert np.isclose(inv["central_impedance_imag"], i2)
    assert np.isclose(np.sin(inv["anisotropy_real_rad"]), i3)
    assert np.isclose(np.sin(inv["anisotropy_imag_rad"]), i4)
    assert np.isclose(
        np.sin(inv["threed_real_rad"] + inv["threed_imag_rad"]), i5
    )
    assert np.isclose(
        np.sin(inv["threed_imag_rad"] - inv["threed_real_rad"]), i6
    )


def test_noise_stability_strike_clean_2D():
    """Clean 2-D synthetic: tight strike distribution (std < 1°)."""
    z_2d = _two_d_tensor(theta_rad=np.radians(30.0))
    out = noise_stability_strike(
        z_2d, n_realisations=200, noise_fraction=0.005, seed=42
    )
    std_deg = float(np.degrees(out["std_rad"]))
    assert std_deg < 1.0, (
        f"clean 2-D should give a tight strike distribution, "
        f"got std = {std_deg:.4f}°"
    )


def test_noise_stability_strike_3D():
    """3-D synthetic with a near-flat in-phase circle: wide strike
    distribution (std > 5°).

    We construct an in-phase part whose Mohr circle has a very
    short radial arm so that the strike is poorly determined; the
    quadrature part is unconstrained but the strike is read from
    the in-phase circle.
    """
    # Construct a tensor whose in-phase part has a near-degenerate
    # Mohr circle (Zxxp - Zyyp ≈ 0 and Zxyp + Zyxp ≈ 0). Add modest
    # quadrature components so the tensor is non-trivial.
    z_p = np.array([[0.5, 1.0], [-1.0, 0.5]])
    z_q = np.array([[2.0, 5.0], [-5.0, 1.5]])
    z = z_p + 1j * z_q
    out = noise_stability_strike(
        z, n_realisations=300, noise_fraction=0.05, seed=42
    )
    std_deg = float(np.degrees(out["std_rad"]))
    assert std_deg > 5.0, (
        f"degenerate 3-D in-phase circle should give a wide "
        f"strike distribution, got std = {std_deg:.4f}°"
    )


def test_mohr_dimensionality_2D():
    z_2d = _two_d_tensor(theta_rad=np.radians(45.0))
    cls = mohr_circle_dimensionality(z_2d, threshold=0.05)
    assert cls == "2D", f"expected 2D classification, got {cls!r}"


def test_mohr_dimensionality_3D():
    z_3d = _three_d_tensor()
    cls = mohr_circle_dimensionality(z_3d, threshold=0.05)
    assert cls in ("3D", "3D-distorted"), (
        f"expected a 3-D classification, got {cls!r}"
    )


# ---------------------------------------------------------------------------
# Smoke test of decompose_lilley
# ---------------------------------------------------------------------------


def test_decompose_lilley_smoke():
    """``decompose_lilley`` returns a populated :class:`LilleyResult`
    on a small synthetic Z."""
    n_freqs = 8
    rng = np.random.default_rng(0)
    periods = np.logspace(-1.0, 2.0, n_freqs)
    z = np.empty((n_freqs, 2, 2), dtype=np.complex128)
    for k in range(n_freqs):
        # Use complex a and b so the quadrature part is non-trivial;
        # a 2-D tensor with zero quadrature is degenerate (no
        # induction) and would not classify as 2-D under any
        # reasonable test.
        a_re = 1.0 + 0.1 * rng.standard_normal()
        a_im = 2.0 + 0.1 * rng.standard_normal()
        b_re = 5.0 + 0.1 * rng.standard_normal()
        b_im = -1.5 + 0.1 * rng.standard_normal()
        z[k] = _two_d_tensor(
            a=complex(a_re, a_im),
            b=complex(b_re, b_im),
            theta_rad=np.radians(30.0),
        )
    sigma = np.maximum(0.01 * np.abs(z), 1e-15)
    z_obj = Z(z=z, z_error=sigma, frequency=1.0 / periods)

    res = decompose_lilley(z_obj, n_realisations=50, seed=42)
    assert isinstance(res, LilleyResult)
    assert res.periods.size == n_freqs
    assert res.center_real.size == n_freqs
    assert res.radius_real.size == n_freqs
    assert len(res.dimensionality) == n_freqs
    # All synthetic periods are 2-D; classifier should agree.
    assert all(d == "2D" for d in res.dimensionality), (
        f"expected all 2-D, got {res.dimensionality}"
    )
