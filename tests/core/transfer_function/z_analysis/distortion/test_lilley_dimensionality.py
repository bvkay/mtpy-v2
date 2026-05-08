"""Validation tests for the unified Lilley 2020 dimensionality
classifier (phase tensor + Bahr-eigenvector + Mohr-circle of the
phase tensor).

The seven required tests cover 1-D / 2-D-strike / 2-D-rotated /
2-D-distorted / 3-D synthetics, the Lilley 2020 formal-equivalence
identities (PT alpha = major-eigenvector strike;
``Mohr radius = (lambda_max − lambda_min) / 2``;
``Mohr mu = 2 * pt_beta``), and cross-tradition agreement with the
Marti / WALDIM classifier.

Reference
---------
Lilley, F. E. M. (2020). Magnetotellurics: the CBB or phase tensor
and Bahr's 1988 analysis. *Exploration Geophysics* 51(4), 401-421.
"""

from __future__ import annotations

import numpy as np

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    LilleyDimensionalityResult,
    classify_dimensionality,
    compare_lilley_marti,
    decompose_marti,
    eigenvector_strike,
    mohr_circle_phase_tensor,
    phase_tensor,
    phase_tensor_invariants,
)


_MU_0 = 4.0 * np.pi * 1.0e-7


def _periods(n: int = 8) -> np.ndarray:
    return np.logspace(-1.0, 2.0, n)


def _z_object(z_arr: np.ndarray, periods: np.ndarray) -> Z:
    sigma = np.maximum(0.005 * np.abs(z_arr), 1e-12)
    return Z(z=z_arr, z_error=sigma, frequency=1.0 / periods)


def _one_d_z(rho_ohm_m: float = 100.0, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    periods = _periods(n)
    omega = 2.0 * np.pi / periods
    z0 = np.sqrt(1j * omega * _MU_0 * rho_ohm_m)
    z = np.zeros((n, 2, 2), dtype=np.complex128)
    z[:, 0, 1] = z0
    z[:, 1, 0] = -z0
    return z, periods


def _two_d_z_strike(
    rho_te: float = 100.0,
    rho_tm: float = 400.0,
    phi_te_deg: float = 60.0,
    phi_tm_deg: float = 30.0,
    n: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    periods = _periods(n)
    omega = 2.0 * np.pi / periods
    a_mag = np.sqrt(omega * _MU_0 * rho_te)
    b_mag = np.sqrt(omega * _MU_0 * rho_tm)
    a = a_mag * np.exp(1j * np.radians(phi_te_deg))
    b = b_mag * np.exp(1j * np.radians(phi_tm_deg))
    z = np.zeros((n, 2, 2), dtype=np.complex128)
    z[:, 0, 1] = a
    z[:, 1, 0] = -b
    return z, periods


def _rotate_z(z: np.ndarray, theta_rad: float) -> np.ndarray:
    cs, sn = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[cs, -sn], [sn, cs]])
    return np.einsum("ij,kjl,lm->kim", R, z, R.T)


def _distort_z(z: np.ndarray, twist_deg: float, shear_deg: float) -> np.ndarray:
    t = np.tan(np.radians(twist_deg))
    s = np.tan(np.radians(shear_deg))
    T = np.array([[1.0, -t], [t, 1.0]])
    S = np.array([[1.0, s], [s, 1.0]])
    C = T @ S
    return np.einsum("ij,kjl->kil", C, z)


def _three_d_z(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Hand-crafted 3-D ``Z`` from test_marti.py case 5: in-phase
    and quadrature parts get *different* twist matrices, giving
    a non-symmetric phase tensor.
    """
    periods = _periods(n)
    omega = 2.0 * np.pi / periods
    a_mag = np.sqrt(omega * _MU_0 * 100.0)
    b_mag = np.sqrt(omega * _MU_0 * 400.0)
    z = np.zeros((n, 2, 2), dtype=np.complex128)
    for k in range(n):
        z_strike_p = np.array(
            [[0.0, 1.0 * a_mag[k]], [-3.0 * a_mag[k] / 4.0, 0.0]]
        )
        z_strike_q = np.array(
            [[0.0, 2.0 * a_mag[k] / 5.0], [-1.0 * b_mag[k] / 8.0, 0.0]]
        )
        twist_p = np.tan(np.radians(15.0))
        twist_q = np.tan(np.radians(-5.0))
        T_p = np.array([[1.0, -twist_p], [twist_p, 1.0]]) / np.sqrt(
            1 + twist_p**2
        )
        T_q = np.array([[1.0, -twist_q], [twist_q, 1.0]]) / np.sqrt(
            1 + twist_q**2
        )
        z[k] = T_p @ z_strike_p + 1j * (T_q @ z_strike_q)
    return z, periods


# ---------------------------------------------------------------------------
# Test 1: 1-D synthetic classifies as "1D"
# ---------------------------------------------------------------------------


def test_1d_classifies_as_1d():
    z, periods = _one_d_z()
    z_obj = _z_object(z, periods)
    result = classify_dimensionality(z_obj)
    assert all(c == "1D" for c in result.classification), (
        f"1-D synthetic should classify as 1D everywhere; got "
        f"{result.classification}"
    )
    assert np.max(np.abs(result.pt_beta_deg)) < 0.01
    assert np.max(np.abs(result.pt_ellipticity)) < 0.01


# ---------------------------------------------------------------------------
# Test 2: 2-D synthetic in strike frame classifies as "2D"
# ---------------------------------------------------------------------------


def test_2d_strike_frame_classifies_as_2d():
    z, periods = _two_d_z_strike()
    z_obj = _z_object(z, periods)
    result = classify_dimensionality(z_obj)
    assert all(c == "2D" for c in result.classification)
    # Eigenvectors are exactly perpendicular for a symmetric Phi.
    assert np.max(result.eigenvector_disagreement_deg) < 0.5
    # In strike frame, the major eigenvector is along ONE of the
    # strike axes (either 0° or 90°, depending on which mode has
    # the larger phase-tensor eigenvalue). Use mod-90°.
    alpha_mod_90 = np.mod(result.pt_alpha_deg, 90.0)
    d = np.minimum(alpha_mod_90, 90.0 - alpha_mod_90)
    assert np.max(d) < 1.0, (
        f"alpha mod 90° should be near 0; got {alpha_mod_90.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 3: 2-D rotated 30° — eigenvector_strike recovers the rotation
# ---------------------------------------------------------------------------


def test_2d_rotated_recovers_strike():
    z_strike, periods = _two_d_z_strike()
    z_rotated = _rotate_z(z_strike, np.radians(30.0))
    z_obj = _z_object(z_rotated, periods)
    result = classify_dimensionality(z_obj)
    assert all(c == "2D" for c in result.classification)

    # The two eigenvectors must be at {30°, 120°} (mod 180°) — i.e.
    # both eigenvector strikes mod 90° must equal 30° (mod 90°).
    for col_idx in (0, 1):
        s = result.eigenvector_strikes[:, col_idx]
        s_mod_90 = np.mod(s, 90.0)
        d = np.minimum(
            np.abs(s_mod_90 - 30.0), np.abs(90.0 - (s_mod_90 - 30.0))
        )
        d = np.minimum(d, np.abs(s_mod_90 + 60.0))
        assert np.max(d) < 1.0, (
            f"eigenvector column {col_idx} mod 90° should be 30°; "
            f"got {s.tolist()}"
        )


# ---------------------------------------------------------------------------
# Test 4: 2-D + galvanic distortion classifies as "2D"
# ---------------------------------------------------------------------------


def test_2d_galvanic_distortion_classifies_as_2d():
    """Galvanic distortion is gauge-invisible to the phase tensor
    by construction (Caldwell 2004); a 2-D regional + galvanic
    distortion thus classifies as 2-D, with the strike still
    recoverable.
    """
    z_strike, periods = _two_d_z_strike()
    z_rotated = _rotate_z(z_strike, np.radians(30.0))
    z_dist = _distort_z(z_rotated, twist_deg=15.0, shear_deg=20.0)
    z_obj = _z_object(z_dist, periods)
    result = classify_dimensionality(z_obj)
    # Should still classify as 2-D (PT is distortion-invariant).
    assert all(c == "2D" for c in result.classification), (
        f"2-D + galvanic distortion should classify as 2D; got "
        f"{result.classification}"
    )
    # Strike still recoverable to within 1° (mod 90°).
    s1 = result.eigenvector_strikes[:, 0]
    s1_mod_90 = np.mod(s1, 90.0)
    d1 = np.minimum(
        np.abs(s1_mod_90 - 30.0), np.abs(90.0 - (s1_mod_90 - 30.0))
    )
    d1 = np.minimum(d1, np.abs(s1_mod_90 + 60.0))
    assert np.max(d1) < 1.0


# ---------------------------------------------------------------------------
# Test 5: 3-D synthetic classifies as "3D" or "3D-2D"
# ---------------------------------------------------------------------------


def test_3d_classifies_as_three_d():
    z, periods = _three_d_z()
    z_obj = _z_object(z, periods)
    result = classify_dimensionality(z_obj)
    # At least some periods should be 3D-flavoured.
    three_d_like = sum(
        1 for c in result.classification if c in ("3D", "3D-2D")
    )
    assert three_d_like >= len(result.classification) // 2, (
        f"3-D synthetic should produce mostly 3-D / 3-D-2-D "
        f"classifications; got {result.classification}"
    )
    # |beta| substantially larger than the 2-D threshold.
    assert np.max(np.abs(result.pt_beta_deg)) > 1.0, (
        f"3-D synthetic should give |beta| > 1°; got "
        f"max={np.max(np.abs(result.pt_beta_deg)):.3f}"
    )


# ---------------------------------------------------------------------------
# Test 6: PT / eigenvector / Mohr-circle equivalence
# ---------------------------------------------------------------------------


class TestLilley2020Equivalences:
    """Verify the formal identities from Lilley 2020:

    * ``alpha`` = strike of the major eigenvector of (the
      symmetric part of) Phi
    * Mohr-circle radius = ``(lambda_max − lambda_min) / 2``
    * Mohr-circle ``mu`` = ``2 * beta_CBB``
    """

    def test_alpha_equals_major_eigenvector_strike_2d(self):
        """For symmetric Phi (clean 2-D), the major eigenvector
        strike coincides with the CBB principal-axis angle alpha
        modulo 180°.
        """
        z, periods = _two_d_z_strike()
        z_rotated = _rotate_z(z, np.radians(45.0))
        phi = phase_tensor(z_rotated)
        inv = phase_tensor_invariants(phi)
        eig = eigenvector_strike(phi)
        for k in range(periods.size):
            alpha = float(inv["alpha_deg"][k]) % 180.0
            s1 = float(eig["strike_alpha1_deg"][k]) % 180.0
            d = abs(alpha - s1) % 180.0
            d = min(d, 180.0 - d)
            assert d < 0.5, (
                f"period {k}: alpha {alpha:.3f}° vs major eigvec "
                f"strike {s1:.3f}°; angular distance {d:.4f}°"
            )

    def test_mohr_radius_equals_eigenvalue_half_difference(self):
        """For symmetric Phi, the Mohr-circle radius equals
        ``(lambda_max - lambda_min) / 2`` exactly.
        """
        z, periods = _two_d_z_strike()
        phi = phase_tensor(z)
        inv = phase_tensor_invariants(phi)
        mohr = mohr_circle_phase_tensor(phi)
        expected = 0.5 * (inv["lambda_max"] - inv["lambda_min"])
        np.testing.assert_allclose(
            mohr["radius"], expected, rtol=1e-10
        )

    def test_mohr_mu_equals_two_beta(self):
        """``mu_Mohr = 2 * beta_CBB`` (Lilley 2020 equivalence).

        Verified on a 3-D synthetic where ``beta`` is non-trivial.
        """
        z, periods = _three_d_z()
        phi = phase_tensor(z)
        inv = phase_tensor_invariants(phi)
        mohr = mohr_circle_phase_tensor(phi)
        np.testing.assert_allclose(
            mohr["mu_deg"], 2.0 * inv["beta_deg"], rtol=1e-9, atol=1e-9
        )


# ---------------------------------------------------------------------------
# Test 7: Lilley vs Marti agreement on canonical synthetics
# ---------------------------------------------------------------------------


def test_lilley_marti_agreement_on_canonical_synthetics():
    """Both classifiers should agree on dimensionality on the five
    canonical synthetics: 1-D, 2-D strike-frame, 2-D rotated, 2-D +
    galvanic distortion, 3-D. Galvanic distortion is gauge-
    invisible to the PT (Lilley sees 2-D); WALDIM cases 3 / 4 / 6 /
    7 (3-D-distorted-2-D) are mapped to "2D" in the
    :func:`compare_lilley_marti` mapping for this comparison.
    """
    cases = []

    # 1) 1-D
    z, p = _one_d_z()
    cases.append(("1D", _z_object(z, p)))
    # 2) 2-D in strike frame
    z, p = _two_d_z_strike()
    cases.append(("2D-strike", _z_object(z, p)))
    # 3) 2-D rotated
    z_strike, p = _two_d_z_strike()
    z_rotated = _rotate_z(z_strike, np.radians(30.0))
    cases.append(("2D-rotated", _z_object(z_rotated, p)))
    # 4) 2-D + galvanic distortion
    z_strike, p = _two_d_z_strike()
    z_rotated = _rotate_z(z_strike, np.radians(30.0))
    z_dist = _distort_z(z_rotated, twist_deg=15.0, shear_deg=20.0)
    cases.append(("2D-galvanic", _z_object(z_dist, p)))
    # 5) 3-D
    z, p = _three_d_z()
    cases.append(("3D", _z_object(z, p)))

    n_agree = 0
    n_total = 0
    diagnostic = []
    for name, z_obj in cases:
        lilley_result = classify_dimensionality(z_obj)
        marti_result = decompose_marti(z_obj)
        comparison = compare_lilley_marti(lilley_result, marti_result)
        rate = comparison["agreement_rate"]
        diagnostic.append(
            (name, rate, comparison["per_period"][0])
        )
        # Per-case: at least 50% of periods should agree (on the
        # 3-D synthetic the per-period codes can vary, so we don't
        # require 100%; the 5/5 across-cases agreement is what
        # matters).
        if rate >= 0.5:
            n_agree += 1
        n_total += 1

    assert n_agree == n_total, (
        f"expected {n_total}/{n_total} canonical cases with "
        f">= 50% Lilley/Marti agreement; got {n_agree}/{n_total}.\n"
        f"Per-case: {diagnostic}"
    )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_classify_returns_dataclass():
    z, periods = _two_d_z_strike()
    result = classify_dimensionality(_z_object(z, periods), site="S01")
    assert isinstance(result, LilleyDimensionalityResult)
    assert result.site == "S01"
    assert result.periods.size == periods.size
    assert "beta_2d_threshold_deg" in result.classification_thresholds
