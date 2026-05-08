"""Tests for the Garcia-Jones (2002) extended decomposition.

Covers the multi-site shared-3D-regional algorithm on three synthetic
configurations:

1. Genuinely 3-D regional with non-trivial per-site distortion:
   Garcia-Jones recovers the distortion and the regional 3-D ``Z``.
2. Genuinely 2-D regional: Garcia-Jones still works and the
   recovered regional Z's diagonal entries are small relative to
   the off-diagonals (the 3-D parametrisation collapses to a 2-D
   form).
3. Same 3-D regional as test 1: Garcia-Jones achieves lower RMS
   misfit than the MJ 2-D-regional decomposition.

References
----------
Garcia, X., & Jones, A. G. (2002). Decomposition of three-
dimensional magnetotelluric data. In *Three-Dimensional
Electromagnetics* (M. S. Zhdanov & P. E. Wannamaker, eds.), Methods
in Geochemistry and Geophysics, 35, 235-250.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
tensor decomposition of magnetotelluric data. Geophysics, 66(1),
158-173.
"""

from __future__ import annotations

import numpy as np

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    GarciaJonesResult,
    decompose_garcia_jones,
    decompose_mcneice_jones,
)
from mtpy.core.transfer_function.z_analysis.decomposition.garcia_jones import (
    _distortion_matrix,
)


def _build_z_obs(
    z_reg: np.ndarray,
    distortions_deg: list[tuple[float, float]],
    noise_rel: float,
    seed: int,
) -> tuple[list[Z], list[str], np.ndarray]:
    """Forward-model observed Z at each site from a shared regional Z
    and per-site (twist, shear) in degrees, with optional noise.

    Returns (list of Z objects, list of site_ids, periods).
    """
    n_freqs = z_reg.shape[0]
    periods = np.logspace(1.0, 2.5, n_freqs)
    rng = np.random.default_rng(seed)

    z_objs: list[Z] = []
    site_ids: list[str] = []
    for site_idx, (twist_deg, shear_deg) in enumerate(distortions_deg):
        c = _distortion_matrix(np.radians(twist_deg), np.radians(shear_deg))
        z = np.empty((n_freqs, 2, 2), dtype=np.complex128)
        for k in range(n_freqs):
            z[k] = c @ z_reg[k]
        if noise_rel > 0:
            noise = noise_rel * np.abs(z) * (
                rng.standard_normal(z.shape)
                + 1j * rng.standard_normal(z.shape)
            )
            z = z + noise
        sigma = np.maximum(noise_rel * np.abs(z), 1e-6)
        z_objs.append(
            Z(z=z, z_error=sigma, frequency=1.0 / periods)
        )
        site_ids.append(f"S{site_idx + 1:02d}")
    return z_objs, site_ids, periods


def _three_d_regional(n_freqs: int = 6) -> np.ndarray:
    """Synthetic 3-D regional Z with non-trivial diagonals.

    Off-diagonals scaled like a typical TE / TM pair; diagonals are
    ~30 % of the off-diagonal magnitudes — well above the
    decomposition's noise floor and large enough that a 2-D fit
    cannot explain them.
    """
    z_reg = np.empty((n_freqs, 2, 2), dtype=np.complex128)
    for k in range(n_freqs):
        # Slowly varying with period index; nothing fancy.
        scale = 1.0 + 0.05 * k
        a = scale * 80.0 * np.exp(1j * np.radians(55.0 + 1.5 * k))
        b = scale * 12.0 * np.exp(1j * np.radians(40.0 + 0.8 * k))
        d_xx = scale * 25.0 * np.exp(1j * np.radians(70.0 - 1.0 * k))
        d_yy = scale * 18.0 * np.exp(1j * np.radians(35.0 + 0.5 * k))
        z_reg[k] = np.array([[d_xx, a], [-b, d_yy]], dtype=np.complex128)
    return z_reg


def _two_d_regional(n_freqs: int = 6) -> np.ndarray:
    """Synthetic 2-D regional Z (anti-diagonal in the strike frame).

    Built directly in the measurement frame with strike = 0 so the
    truth is anti-diagonal exactly.
    """
    z_reg = np.empty((n_freqs, 2, 2), dtype=np.complex128)
    for k in range(n_freqs):
        scale = 1.0 + 0.05 * k
        a = scale * 80.0 * np.exp(1j * np.radians(55.0 + 1.5 * k))
        b = scale * 12.0 * np.exp(1j * np.radians(40.0 + 0.8 * k))
        z_reg[k] = np.array([[0.0, a], [-b, 0.0]], dtype=np.complex128)
    return z_reg


_DISTORTIONS_5 = [
    (-10.0, 5.0),
    (15.0, -8.0),
    (0.0, 0.0),
    (-5.0, 12.0),
    (20.0, -3.0),
]


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_gj_synthetic_3D_regional():
    """Garcia-Jones recovers per-site distortion and the 3-D regional
    Z on a synthetic with known 3-D structure.

    Tolerances: twist / shear within 2 degrees of truth (the noise
    is 0.5 % per component); regional Z entries within 5 % of truth
    in absolute value.
    """
    z_reg_true = _three_d_regional(n_freqs=6)
    z_objs, site_ids, _ = _build_z_obs(
        z_reg_true, _DISTORTIONS_5, noise_rel=0.005, seed=0
    )

    res = decompose_garcia_jones(z_objs, site_ids, n_starts=4, seed=42)
    assert isinstance(res, GarciaJonesResult)
    assert len(res.per_band_3d_z_regional) == 1

    for sid, (twist_true, shear_true) in zip(site_ids, _DISTORTIONS_5):
        rec = res.per_site_distortion[sid]
        assert abs(rec["twist_deg"] - twist_true) < 2.0, (
            f"site {sid}: twist {rec['twist_deg']:.2f}° vs truth "
            f"{twist_true:.2f}°"
        )
        assert abs(rec["shear_deg"] - shear_true) < 2.0, (
            f"site {sid}: shear {rec['shear_deg']:.2f}° vs truth "
            f"{shear_true:.2f}°"
        )

    z_reg_rec = res.per_band_3d_z_regional[0]
    rel_err = np.abs(z_reg_rec - z_reg_true) / np.maximum(
        np.abs(z_reg_true), 1e-12
    )
    assert np.median(rel_err) < 0.05, (
        f"median |Z_reg - Z_true| / |Z_true| = {np.median(rel_err):.4f}, "
        f"expected < 0.05"
    )


def test_gj_collapses_to_mj_for_2D():
    """On a 2-D regional synthetic Garcia-Jones still works; the
    recovered 3-D regional Z's diagonal entries are small relative
    to its off-diagonals (the 3-D parametrisation collapses to the
    2-D form).
    """
    z_reg_true = _two_d_regional(n_freqs=6)
    z_objs, site_ids, _ = _build_z_obs(
        z_reg_true, _DISTORTIONS_5, noise_rel=0.005, seed=1
    )

    res = decompose_garcia_jones(z_objs, site_ids, n_starts=4, seed=42)

    # Distortion still recovered.
    for sid, (twist_true, shear_true) in zip(site_ids, _DISTORTIONS_5):
        rec = res.per_site_distortion[sid]
        assert abs(rec["twist_deg"] - twist_true) < 2.0, (
            f"site {sid}: twist {rec['twist_deg']:.2f}° vs truth "
            f"{twist_true:.2f}°"
        )
        assert abs(rec["shear_deg"] - shear_true) < 2.0, (
            f"site {sid}: shear {rec['shear_deg']:.2f}° vs truth "
            f"{shear_true:.2f}°"
        )

    # Recovered regional Z is approximately anti-diagonal.
    z_reg_rec = res.per_band_3d_z_regional[0]
    diag_mag = 0.5 * (
        np.abs(z_reg_rec[:, 0, 0]) + np.abs(z_reg_rec[:, 1, 1])
    )
    off_mag = 0.5 * (
        np.abs(z_reg_rec[:, 0, 1]) + np.abs(z_reg_rec[:, 1, 0])
    )
    ratio = diag_mag / np.maximum(off_mag, 1e-12)
    assert np.median(ratio) < 0.05, (
        f"diagonal / off-diagonal magnitude ratio median "
        f"{np.median(ratio):.4f} not small enough for 2-D collapse"
    )


def test_gj_better_than_mj_at_3d():
    """On a 3-D regional synthetic Garcia-Jones achieves lower RMS
    misfit than the MJ 2-D-regional decomposition.

    MJ is constrained to an anti-diagonal regional ``Z``, so on a
    genuinely 3-D regional it cannot fit the diagonal entries and
    leaves them in the residuals.
    """
    z_reg_true = _three_d_regional(n_freqs=6)
    z_objs, site_ids, _ = _build_z_obs(
        z_reg_true, _DISTORTIONS_5, noise_rel=0.005, seed=2
    )

    gj = decompose_garcia_jones(z_objs, site_ids, n_starts=4, seed=42)
    mj = decompose_mcneice_jones(z_objs, site_ids, n_starts=4, seed=42)

    gj_rms = float(np.mean(list(gj.rms_misfit_per_site.values())))
    mj_rms = float(np.mean(list(mj.rms_misfit_per_site.values())))

    assert gj_rms < mj_rms, (
        f"GJ RMS {gj_rms:.4f} not lower than MJ RMS {mj_rms:.4f} "
        f"on a 3-D regional synthetic"
    )
    # Quantitative gap: with diagonals at 30 % of off-diagonal scale
    # and 0.5 % noise, MJ should be at least 5x worse than GJ.
    assert mj_rms > 5.0 * gj_rms, (
        f"expected MJ RMS to be far worse than GJ RMS on 3-D synthetic "
        f"(gap factor = {mj_rms / max(gj_rms, 1e-12):.2f}, expected > 5)"
    )
