"""Validation tests for the Gomez-Treviño 2018 rotational-invariant
TE / TM framework.

.. note::

    **Exploratory module.** These tests verify *internal
    mathematical consistency* and the *rotational-invariance
    property* of the Gomez-Treviño construction. They do **not**
    validate whether the framework produces geologically
    meaningful apparent resistivities for general 3-D data —
    that judgement is deferred to a benchmarking PR against
    forward-modelled 3-D synthetics. See the
    :mod:`...gomez_trevino` module docstring for the validation
    status of each component.

Reference
---------
Gómez-Treviño, E., Esparza, F. J., & Romo, J. M. (2018). On the
use of two new invariants of the magnetotelluric impedance tensor
as natural rotational invariant TE and TM modes. *Earth, Planets
and Space*, 70:35.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    decompose_gomez_trevino,
    determinant_resistivity,
    invariant_resistivities,
    iterative_chain,
    series_parallel_resistivities,
    GomezTrevinoResult,
)


_MU_0 = 4.0 * np.pi * 1.0e-7


def _periods_omega(n_freqs: int = 6) -> tuple[np.ndarray, np.ndarray]:
    periods = np.logspace(-1.0, 2.0, n_freqs)
    omega = 2.0 * np.pi / periods
    return periods, omega


def _one_d_z(rho_ohm_m: float = 100.0, n_freqs: int = 6) -> tuple[
    np.ndarray, np.ndarray
]:
    periods, omega = _periods_omega(n_freqs)
    z0 = np.sqrt(1j * omega * _MU_0 * rho_ohm_m)
    z = np.zeros((n_freqs, 2, 2), dtype=np.complex128)
    z[:, 0, 1] = z0
    z[:, 1, 0] = -z0
    return z, periods


def _two_d_z_strike(
    rho_te: float = 100.0,
    rho_tm: float = 400.0,
    phi_te_deg: float = 60.0,
    phi_tm_deg: float = 30.0,
    n_freqs: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """2-D anti-diagonal in strike frame."""
    periods, omega = _periods_omega(n_freqs)
    a_mag = np.sqrt(omega * _MU_0 * rho_te)
    b_mag = np.sqrt(omega * _MU_0 * rho_tm)
    a = a_mag * np.exp(1j * np.radians(phi_te_deg))
    b = b_mag * np.exp(1j * np.radians(phi_tm_deg))
    z = np.zeros((n_freqs, 2, 2), dtype=np.complex128)
    z[:, 0, 1] = a
    z[:, 1, 0] = -b
    return z, periods


def _rotate_z(z: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate every period's Z by R(theta).T @ Z @ R(theta) — the
    inverse-strike measurement-frame rotation.
    """
    cs, sn = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[cs, -sn], [sn, cs]])
    return np.einsum("ij,kjl,lm->kim", R, z, R.T)


# ---------------------------------------------------------------------------
# Test 1: 1-D Earth — rho_s = rho_p, rho_+ = rho_- = rho_d
# ---------------------------------------------------------------------------


class TestOneDimensional:
    """For a 1-D Earth the discriminant is zero and all four
    resistivities collapse to a single value.
    """

    def test_rho_s_equals_rho_p(self):
        z, periods = _one_d_z(rho_ohm_m=100.0)
        rho_s, rho_p = series_parallel_resistivities(z, periods)
        assert np.max(np.abs(rho_s - rho_p)) < 1e-10, (
            f"1-D should give rho_s == rho_p; got max|diff| = "
            f"{np.max(np.abs(rho_s - rho_p)):.3e}"
        )

    def test_discriminant_is_zero(self):
        z, periods = _one_d_z(rho_ohm_m=100.0)
        rho_s, rho_p = series_parallel_resistivities(z, periods)
        disc = rho_s * rho_s - rho_s * rho_p
        assert np.max(np.abs(disc)) < 1e-10, (
            f"1-D discriminant should vanish; got max|disc| = "
            f"{np.max(np.abs(disc)):.3e}"
        )

    def test_rho_plus_equals_rho_minus_equals_rho_d(self):
        z, periods = _one_d_z(rho_ohm_m=100.0)
        rho_plus, rho_minus, _, _ = invariant_resistivities(z, periods)
        rho_d, _ = determinant_resistivity(z, periods)
        # Tolerance set by sqrt(near-zero) numerical precision.
        assert np.max(np.abs(rho_plus - rho_minus)) < 1e-5
        assert np.max(np.abs(rho_plus - rho_d)) < 1e-5

    def test_apparent_resistivity_recovers_input(self):
        z, periods = _one_d_z(rho_ohm_m=250.0)
        rho_s, _ = series_parallel_resistivities(z, periods)
        # |rho_s| should equal the input rho (Cagniard).
        np.testing.assert_allclose(np.abs(rho_s), 250.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: 2-D Earth in strike frame — explicit reductions
# ---------------------------------------------------------------------------


class TestTwoDimensionalStrikeFrame:
    """In strike frame ``Z`` is anti-diagonal with entries ``a`` and
    ``-b``. The framework's algebra reduces to::

        rho_s = (a² + b²) / (2 omega mu_0)
        rho_p = 2 a² b² / (omega mu_0 (a² + b²))
        {rho_+, rho_-} = {a²/(omega mu_0), b²/(omega mu_0)}

    in some order. Verified to floating-point precision.
    """

    def test_rho_s_explicit(self):
        z, periods = _two_d_z_strike()
        rho_s, _ = series_parallel_resistivities(z, periods)
        omega = 2.0 * np.pi / periods
        a = z[:, 0, 1]
        b = -z[:, 1, 0]
        rho_s_expected = (a * a + b * b) / (2.0 * omega * _MU_0)
        np.testing.assert_allclose(rho_s, rho_s_expected, rtol=1e-12)

    def test_rho_p_explicit(self):
        z, periods = _two_d_z_strike()
        _, rho_p = series_parallel_resistivities(z, periods)
        omega = 2.0 * np.pi / periods
        a = z[:, 0, 1]
        b = -z[:, 1, 0]
        rho_p_expected = (
            2.0 * (a * a) * (b * b) / (omega * _MU_0 * (a * a + b * b))
        )
        np.testing.assert_allclose(rho_p, rho_p_expected, rtol=1e-12)

    def test_invariants_recover_te_tm(self):
        z, periods = _two_d_z_strike(rho_te=100.0, rho_tm=400.0)
        rho_plus, rho_minus, _, _ = invariant_resistivities(z, periods)
        # The magnitudes should be {100, 400} in some order at every
        # period. Sort per period and compare to (100, 400).
        per_period = np.sort(np.column_stack([rho_plus, rho_minus]), axis=1)
        expected = np.array([[100.0, 400.0]] * len(periods))
        np.testing.assert_allclose(per_period, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Test 3: 2-D rotated — invariants are rotation-invariant
# ---------------------------------------------------------------------------


class TestRotationInvariance:
    """The central claim of the framework: ``rho_+`` and ``rho_-``
    are rotational invariants of ``Z``. Rotating the measurement
    axis should leave both magnitudes and phases unchanged.
    """

    def test_invariants_unchanged_under_30_deg_rotation(self):
        z_strike, periods = _two_d_z_strike(rho_te=100.0, rho_tm=400.0)
        z_rotated = _rotate_z(z_strike, np.radians(30.0))

        rp_s, rm_s, _, _ = invariant_resistivities(z_strike, periods)
        rp_r, rm_r, _, _ = invariant_resistivities(z_rotated, periods)

        # The two pairs should match per period (sorted, since the
        # ± labels are mathematical and rotation may reorder them
        # in principle — though for a clean 2-D at strike frame both
        # orderings happen to agree).
        s_sorted = np.sort(np.column_stack([rp_s, rm_s]), axis=1)
        r_sorted = np.sort(np.column_stack([rp_r, rm_r]), axis=1)
        np.testing.assert_allclose(s_sorted, r_sorted, rtol=1e-10)

    def test_rho_d_unchanged_under_rotation(self):
        z_strike, periods = _two_d_z_strike(rho_te=100.0, rho_tm=400.0)
        z_rotated = _rotate_z(z_strike, np.radians(45.0))
        rd_s, _ = determinant_resistivity(z_strike, periods)
        rd_r, _ = determinant_resistivity(z_rotated, periods)
        np.testing.assert_allclose(rd_s, rd_r, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 4: 2-D + galvanic distortion — invariants biased but
# rotation-invariant
# ---------------------------------------------------------------------------


class TestDistortionBehaviour:
    """Galvanic distortion biases the apparent resistivities (this is
    well-known and shared with the determinant resistivity), but the
    framework is still rotationally invariant: rotating the
    *distorted* tensor must leave ``rho_+`` / ``rho_-`` unchanged.
    """

    @staticmethod
    def _construct_C(twist_deg: float, shear_deg: float) -> np.ndarray:
        t = np.tan(np.radians(twist_deg))
        s = np.tan(np.radians(shear_deg))
        T = np.array([[1.0, -t], [t, 1.0]])
        S = np.array([[1.0, s], [s, 1.0]])
        return T @ S

    def test_distortion_invariants_rotation_invariant(self):
        z_strike, periods = _two_d_z_strike(
            rho_te=100.0, rho_tm=400.0
        )
        c = self._construct_C(twist_deg=15.0, shear_deg=20.0)
        z_dist = np.einsum("ij,kjl->kil", c, z_strike)

        rp_orig, rm_orig, _, _ = invariant_resistivities(z_dist, periods)
        z_dist_rot = _rotate_z(z_dist, np.radians(30.0))
        rp_rot, rm_rot, _, _ = invariant_resistivities(z_dist_rot, periods)

        s_sorted = np.sort(np.column_stack([rp_orig, rm_orig]), axis=1)
        r_sorted = np.sort(np.column_stack([rp_rot, rm_rot]), axis=1)
        np.testing.assert_allclose(s_sorted, r_sorted, rtol=1e-10)

    def test_distortion_biases_rho_plus_minus(self):
        """Document that distortion *does* bias the values (this is
        expected, not a bug; like the determinant resistivity).
        """
        z_strike, periods = _two_d_z_strike(
            rho_te=100.0, rho_tm=400.0
        )
        c = self._construct_C(twist_deg=15.0, shear_deg=20.0)
        z_dist = np.einsum("ij,kjl->kil", c, z_strike)

        rp_clean, rm_clean, _, _ = invariant_resistivities(
            z_strike, periods
        )
        rp_dist, rm_dist, _, _ = invariant_resistivities(z_dist, periods)

        # Some bias must be present (otherwise the test is vacuous);
        # specifically the geometric mean rho_d must shift because
        # det(Z) and det(C·Z) differ by det(C) ≠ 1.
        rd_clean = np.sqrt(rp_clean * rm_clean)
        rd_dist = np.sqrt(rp_dist * rm_dist)
        rel_shift = np.median(np.abs(rd_dist / rd_clean - 1.0))
        assert rel_shift > 0.01, (
            f"distortion should noticeably shift rho_d; got median "
            f"relative shift {rel_shift:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: 3-D Earth — rho_+ != rho_-, chain converges
# ---------------------------------------------------------------------------


class TestThreeDimensional:
    """For a genuinely 3-D ``Z`` the framework gives distinct
    ``rho_+``, ``rho_-`` and the iterative chain converges to
    ``rho_d`` (geometric mean).
    """

    @staticmethod
    def _three_d_z(n_freqs: int = 6) -> tuple[np.ndarray, np.ndarray]:
        """Hand-crafted 3-D ``Z`` with non-zero diagonals (per the
        Marti 2009 case-5 construction in test_marti.py).
        """
        periods, omega = _periods_omega(n_freqs)
        # In-phase and quadrature parts have *different* twists,
        # which breaks the GB-real-distortion assumption and gives
        # a genuine 3-D Z.
        a_mag = np.sqrt(omega * _MU_0 * 100.0)
        b_mag = np.sqrt(omega * _MU_0 * 400.0)
        z = np.zeros((n_freqs, 2, 2), dtype=np.complex128)
        for k in range(n_freqs):
            z_strike_p = np.array(
                [[0.0, 1.0 * a_mag[k]], [-3.0 * a_mag[k] / 4.0, 0.0]]
            )
            z_strike_q = np.array(
                [[0.0, 2.0 * a_mag[k] / 5.0], [-1.0 * b_mag[k] / 8.0, 0.0]]
            )
            twist_p = np.tan(np.radians(15.0))
            twist_q = np.tan(np.radians(-5.0))
            T_p = np.array(
                [[1.0, -twist_p], [twist_p, 1.0]]
            ) / np.sqrt(1 + twist_p**2)
            T_q = np.array(
                [[1.0, -twist_q], [twist_q, 1.0]]
            ) / np.sqrt(1 + twist_q**2)
            z_p = T_p @ z_strike_p
            z_q = T_q @ z_strike_q
            z[k] = z_p + 1j * z_q
        return z, periods

    def test_3d_invariants_distinct(self):
        z, periods = self._three_d_z()
        rho_plus, rho_minus, _, _ = invariant_resistivities(z, periods)
        rel_split = np.abs(rho_plus - rho_minus) / np.maximum(
            np.abs(rho_plus), np.abs(rho_minus)
        )
        assert np.median(rel_split) > 0.05, (
            f"3-D synthetic should give clearly distinct rho_+ and "
            f"rho_-; got median relative split {np.median(rel_split):.4f}"
        )

    def test_3d_rho_d_between_rho_plus_minus(self):
        z, periods = self._three_d_z()
        rho_plus, rho_minus, _, _ = invariant_resistivities(z, periods)
        rho_d, _ = determinant_resistivity(z, periods)
        # rho_d = sqrt(rho_+ · rho_-) is the geometric mean, so it
        # lies between them (in magnitude). Allow numerical slack.
        lo = np.minimum(rho_plus, rho_minus)
        hi = np.maximum(rho_plus, rho_minus)
        assert np.all(rho_d >= lo - 1e-6 * hi), "rho_d below min(rho_+, rho_-)"
        assert np.all(rho_d <= hi + 1e-6 * hi), "rho_d above max(rho_+, rho_-)"

    def test_3d_chain_converges_to_rho_d(self):
        z, periods = self._three_d_z()
        rho_d, _ = determinant_resistivity(z, periods)
        rho_s_hist, rho_p_hist, iters = iterative_chain(
            z, periods, max_iter=20, tol=1e-10
        )
        # Pull the converged value: at iters[k]+1 the values are
        # close enough; or simpler, at the final non-NaN row.
        for k in range(periods.size):
            n_iter = int(iters[k])
            converged_rho_s = rho_s_hist[n_iter, k]
            assert abs(abs(converged_rho_s) - rho_d[k]) < 1e-6 * rho_d[k], (
                f"period {k}: chain converged to "
                f"|rho_s|={abs(converged_rho_s):.4f}, expected "
                f"rho_d={rho_d[k]:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 6: Geometric-mean invariance of the chain
# ---------------------------------------------------------------------------


class TestChainGeometricMeanInvariance:
    """At every step of the iterative chain, ``rho_s_i · rho_p_i =
    rho_+ · rho_- = rho_d²``. This is the framework's central
    structural identity and is exact (not asymptotic).
    """

    @pytest.mark.parametrize(
        "regime",
        ["1d", "2d_strike", "2d_distorted", "3d"],
    )
    def test_geometric_mean_preserved_every_iteration(self, regime: str):
        if regime == "1d":
            z, periods = _one_d_z()
        elif regime == "2d_strike":
            z, periods = _two_d_z_strike()
        elif regime == "2d_distorted":
            z, periods = _two_d_z_strike()
            t = np.tan(np.radians(15.0))
            s = np.tan(np.radians(20.0))
            c = np.array([[1.0, -t], [t, 1.0]]) @ np.array(
                [[1.0, s], [s, 1.0]]
            )
            z = np.einsum("ij,kjl->kil", c, z)
        else:  # 3d
            z, periods = TestThreeDimensional._three_d_z()

        rho_s_hist, rho_p_hist, iters = iterative_chain(
            z, periods, max_iter=20, tol=1e-10
        )
        rho_d, _ = determinant_resistivity(z, periods)
        rho_d_sq_expected = rho_d * rho_d

        # At every iteration up to convergence per period, the
        # geometric mean must equal rho_d^2.
        for k in range(periods.size):
            n_iter = int(iters[k])
            for i in range(n_iter + 1):
                product = rho_s_hist[i, k] * rho_p_hist[i, k]
                assert (
                    abs(abs(product) - rho_d_sq_expected[k])
                    < 1e-6 * rho_d_sq_expected[k]
                ), (
                    f"regime={regime}, period={k}, iter={i}: "
                    f"|rho_s · rho_p| = {abs(product):.4f}, "
                    f"expected rho_d² = {rho_d_sq_expected[k]:.4f}"
                )


# ---------------------------------------------------------------------------
# Top-level decompose smoke test
# ---------------------------------------------------------------------------


class TestDecomposeEntrypoint:
    """``decompose_gomez_trevino`` returns a populated
    :class:`GomezTrevinoResult` on a typical 2-D ``Z``."""

    def test_smoke(self):
        z_arr, periods = _two_d_z_strike()
        sigma = np.maximum(0.005 * np.abs(z_arr), 1e-12)
        z_obj = Z(z=z_arr, z_error=sigma, frequency=1.0 / periods)
        result = decompose_gomez_trevino(z_obj, site="S01")
        assert isinstance(result, GomezTrevinoResult)
        assert result.site == "S01"
        assert result.periods.size == periods.size
        assert result.rho_plus.size == periods.size
        assert result.convergence_iterations is not None
        assert result.metadata["exploratory"] is True
