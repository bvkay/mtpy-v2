"""Tests for the Bibby-Caldwell-Brown decomposition.

Covers exact recovery on 2-D synthetics in the BCB diagonal-unity
gauge, the no-distortion identity case, and consistency with
Groom-Bailey for synthetics with ``gain ~ 1`` and small twist /
shear (the regime where the BCB and GB gauges coincide).

References
----------
Bibby, H. M., Caldwell, T. G., & Brown, C. (2005). Determinable and
non-determinable parameters of galvanic distortion in
magnetotellurics. Geophysical Journal International, 163(3),
915-930.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    BibbyResult,
    _estim_imp,
    decompose,
    decompose_bibby,
)


def _R(theta_deg: float) -> np.ndarray:
    theta = np.radians(theta_deg)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _band_periods_and_ab(
    n_freqs: int = 12,
    rho_a: float = 100.0,
    rho_b: float = 10.0,
    phase_a_deg: float = 60.0,
    phase_b_deg: float = 45.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periods and TE/TM regional impedances for a clean 2-D site.

    Holds ``rho_a`` consistently larger than ``rho_b`` so the phase
    tensor's principal axis stays on the same branch across periods.
    """
    rng = np.random.default_rng(seed)
    periods = np.logspace(-1.0, 2.0, n_freqs)
    log10_rho_a = np.log10(rho_a) + 0.05 * rng.standard_normal(n_freqs)
    log10_rho_b = np.log10(rho_b) + 0.05 * rng.standard_normal(n_freqs)
    phase_a = np.radians(phase_a_deg) + 0.02 * rng.standard_normal(n_freqs)
    phase_b = np.radians(phase_b_deg) + 0.02 * rng.standard_normal(n_freqs)
    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0
    abs_a = np.sqrt(10.0**log10_rho_a * factor / periods)
    abs_b = np.sqrt(10.0**log10_rho_b * factor / periods)
    a = abs_a * np.exp(1j * phase_a)
    b = abs_b * np.exp(1j * phase_b)
    return periods, a, b


def _build_z_with_known_C(
    c_strike: np.ndarray,
    theta_deg: float = 30.0,
    n_freqs: int = 12,
    seed: int = 0,
) -> tuple[Z, np.ndarray]:
    """Synthetic 2-D site with ``Z_obs = C @ Z_R``, ``C`` real.

    ``c_strike`` is the distortion matrix in the strike frame; the
    measurement-frame ``C`` is ``R(theta) c_strike R(theta).T``. The
    regional ``Z_R`` is the standard log-spaced 2-D site.
    """
    R = _R(theta_deg)
    c_meas = R @ c_strike @ R.T

    periods, a, b = _band_periods_and_ab(n_freqs=n_freqs, seed=seed)
    z_obs = np.empty((n_freqs, 2, 2), dtype=np.complex128)
    for k in range(n_freqs):
        z_r_strike_k = np.array([[0.0, a[k]], [-b[k], 0.0]], dtype=np.complex128)
        z_r_meas_k = R @ z_r_strike_k @ R.T
        z_obs[k] = c_meas @ z_r_meas_k

    sigma = np.maximum(0.01 * np.abs(z_obs), 1e-15)
    return Z(z=z_obs, z_error=sigma, frequency=1.0 / periods), c_meas


def _gb_C_per_period(result) -> np.ndarray:
    """Reconstruct the measurement-frame ``C`` per period from a GB result."""
    p = result.parameters
    strikes = p["strike"].values
    twists = p["twist"].values
    shears = p["shear"].values
    gains = p["gain"].values
    n = strikes.size
    out = np.full((n, 2, 2), np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isfinite(strikes[i]):
            continue
        s = np.radians(strikes[i])
        t = np.radians(twists[i])
        e = np.radians(shears[i])
        cs, sn = np.cos(s), np.sin(s)
        ct, st = np.cos(t), np.sin(t)
        ce, se = np.cos(e), np.sin(e)
        R = np.array([[cs, -sn], [sn, cs]])
        T = np.array([[ct, -st], [st, ct]])
        S = np.array([[ce, se], [se, ce]])
        out[i] = gains[i] * R @ T @ S @ R.T
    return out


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_bibby_synthetic_2D_recovery():
    """BCB recovers a known ``C`` (in the diagonal-unity gauge) exactly.

    The synthetic's ``C`` is built with strike-frame diagonals = 1
    so it lives in BCB's gauge; the algorithm must therefore return
    it within numerical precision (well under the 0.01 Frobenius
    tolerance).
    """
    c_strike = np.array([[1.0, 0.15], [-0.10, 1.0]])
    z_obj, c_known = _build_z_with_known_C(c_strike, theta_deg=30.0)

    result = decompose_bibby(z_obj)

    assert isinstance(result, BibbyResult)
    assert result.C.shape == (2, 2)
    assert result.C.dtype == np.float64
    dist = np.linalg.norm(result.C - c_known, ord="fro")
    assert dist < 0.01, (
        f"BCB recovered C far from known: Frobenius distance = {dist:.6f}\n"
        f"  recovered:\n{result.C}\n  known:\n{c_known}"
    )


def test_bibby_no_distortion():
    """With ``C = I`` BCB recovers the identity matrix.

    Tolerance 0.05 (looser than the synthetic-recovery test) because
    the noiseless identity case still goes through the LS fit and
    can pick up O(1e-3) numerical noise.
    """
    c_strike = np.eye(2)
    z_obj, _ = _build_z_with_known_C(c_strike, theta_deg=30.0)

    result = decompose_bibby(z_obj)

    dist = np.linalg.norm(result.C - np.eye(2), ord="fro")
    assert dist < 0.05, (
        f"BCB did not recover identity: Frobenius distance = {dist:.6f}\n"
        f"  recovered:\n{result.C}"
    )


def test_bibby_consistency_with_gb():
    """BCB and GB agree on a small-distortion synthetic, gauge-normalised.

    The two methods describe the same physical galvanic distortion
    but pick different gauges for the structurally non-determinable
    overall scale of ``C`` (the GB ``gain`` is gauge-equivalent to a
    rescaling of the regional impedance — different bands of a
    single GB run can converge to different gains for the same
    physics; see the GB module docstring's note on gauge
    equivalences).

    The physically meaningful comparison is therefore on
    Frobenius-normalised C tensors (``C / ||C||_F``), which are
    gauge-invariant. This test builds a ``g=1``, ``twist=5°``,
    ``shear=3°`` synthetic and verifies BCB and GB produce
    normalised C tensors that agree within 0.05 Frobenius distance.
    """
    theta_deg = 30.0
    twist_deg = 5.0
    shear_deg = 3.0
    gain = 1.0
    periods, a, b = _band_periods_and_ab()
    n_freqs = a.size
    theta = np.radians(theta_deg)
    twist_tan = np.tan(np.radians(twist_deg))
    shear_tan = np.tan(np.radians(shear_deg))
    z_obs = np.empty((n_freqs, 2, 2), dtype=np.complex128)
    for k in range(n_freqs):
        z_obs[k] = gain * _estim_imp(a[k], b[k], twist_tan, shear_tan, theta)
    sigma = np.maximum(0.01 * np.abs(z_obs), 1e-15)
    z_obj = Z(z=z_obs, z_error=sigma, frequency=1.0 / periods)

    bcb = decompose_bibby(z_obj)

    gb = decompose(z_obj, seed=42)
    c_gb_per_period = _gb_C_per_period(gb)
    valid = np.all(
        np.isfinite(c_gb_per_period.reshape(c_gb_per_period.shape[0], -1)),
        axis=1,
    )
    c_gb_median = np.median(c_gb_per_period[valid], axis=0)

    def _normalise(c: np.ndarray) -> np.ndarray:
        return c / np.linalg.norm(c, ord="fro")

    dist = np.linalg.norm(_normalise(bcb.C) - _normalise(c_gb_median), ord="fro")
    assert dist < 0.05, (
        "BCB and GB normalised C tensors disagree: Frobenius distance "
        f"= {dist:.6f}\n  BCB:\n{bcb.C}\n  GB median:\n{c_gb_median}"
    )


def test_bibby_band_method_validation():
    """Unknown ``band_method`` is rejected with an informative error."""
    z_obj, _ = _build_z_with_known_C(np.eye(2))
    with pytest.raises(ValueError, match="band_method"):
        decompose_bibby(z_obj, band_method="harmonic")


def test_bibby_periods_window_validation():
    """A period window with no covered periods raises."""
    z_obj, _ = _build_z_with_known_C(np.eye(2))
    with pytest.raises(ValueError, match="no periods in window"):
        decompose_bibby(z_obj, periods=(1e10, 1e11))
