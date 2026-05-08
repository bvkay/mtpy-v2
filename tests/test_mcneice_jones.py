"""Tests for the McNeice-Jones (2001) multi-site joint decomposition.

This is the Phase-1 test suite — synthetic recovery, single-site
collapse to GB, and disambiguation pass-through. Validation against
published results (BC87 etc.) lives in a separate Phase-2 test
file.

References
----------
McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency tensor
decomposition of magnetotelluric data. Geophysics, 66(1), 158-173.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    JointDecompositionResult,
    _estim_imp,
    decompose,
    decompose_mcneice_jones,
)


def _build_site_z(
    theta_deg: float,
    twist_deg: float,
    shear_deg: float,
    gain: float = 1.0,
    n_freqs: int = 12,
    seed: int = 0,
    rho_a: float = 100.0,
    rho_b: float = 10.0,
) -> Z:
    """Synthetic ``Z`` for one site with shared-strike GB distortion.

    Holds ``rho_a > rho_b`` and uses the GB forward model
    ``_estim_imp`` so the synthetic is exactly a GB site with the
    requested parameters.
    """
    rng = np.random.default_rng(seed)
    periods = np.logspace(-1.0, 2.0, n_freqs)
    log10_rho_a = np.log10(rho_a) + 0.05 * rng.standard_normal(n_freqs)
    log10_rho_b = np.log10(rho_b) + 0.05 * rng.standard_normal(n_freqs)
    phase_a = np.radians(60.0) + 0.02 * rng.standard_normal(n_freqs)
    phase_b = np.radians(45.0) + 0.02 * rng.standard_normal(n_freqs)
    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0
    abs_a = np.sqrt(10.0**log10_rho_a * factor / periods)
    abs_b = np.sqrt(10.0**log10_rho_b * factor / periods)
    a = abs_a * np.exp(1j * phase_a)
    b = abs_b * np.exp(1j * phase_b)
    theta = np.radians(theta_deg)
    twist_tan = np.tan(np.radians(twist_deg))
    shear_tan = np.tan(np.radians(shear_deg))
    z_obs = np.empty((n_freqs, 2, 2), dtype=np.complex128)
    for k in range(n_freqs):
        z_obs[k] = gain * _estim_imp(a[k], b[k], twist_tan, shear_tan, theta)
    sigma = np.maximum(0.01 * np.abs(z_obs), 1e-15)
    return Z(z=z_obs, z_error=sigma, frequency=1.0 / periods)


def _C_from_GB(
    strike_deg: float, twist_deg: float, shear_deg: float, gain: float
) -> np.ndarray:
    s = np.radians(strike_deg)
    t = np.radians(twist_deg)
    e = np.radians(shear_deg)
    cs, sn = np.cos(s), np.sin(s)
    ct, st = np.cos(t), np.sin(t)
    ce, se = np.cos(e), np.sin(e)
    R = np.array([[cs, -sn], [sn, cs]])
    T = np.array([[ct, -st], [st, ct]])
    S = np.array([[ce, se], [se, ce]])
    return gain * R @ T @ S @ R.T


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_mj_synthetic_2D_recovery():
    """5 synthetic sites with a shared 30° strike and per-site distortion.

    The joint MJ optimiser must recover the shared strike to within
    1° and per-site C tensors to within Frobenius distance 0.05.
    The C-tensor comparison is against the synthetic's true GB C
    (``g·R·T·S·R.T``); recovery is on Frobenius-normalised tensors
    to remove the gain gauge ambiguity already documented for
    single-site GB.
    """
    shared_strike_deg = 30.0
    site_specs = [
        ("A", 6.0, 3.0, 1.0),
        ("B", -4.0, 5.0, 1.0),
        ("C", 8.0, -2.0, 1.0),
        ("D", 2.0, 4.0, 1.0),
        ("E", -3.0, -3.0, 1.0),
    ]
    z_objs = [
        _build_site_z(
            shared_strike_deg, twist_deg=t, shear_deg=e, gain=g, seed=i
        )
        for i, (_, t, e, g) in enumerate(site_specs)
    ]
    site_ids = [s[0] for s in site_specs]

    result = decompose_mcneice_jones(z_objs, site_ids, seed=42, n_starts=3)

    assert isinstance(result, JointDecompositionResult)
    assert len(result.per_band_strike) == 1
    band_id = next(iter(result.per_band_strike))
    recovered_strike = result.per_band_strike[band_id]

    diff_strike = abs(recovered_strike - shared_strike_deg) % 180.0
    diff_strike = min(diff_strike, 180.0 - diff_strike)
    assert diff_strike < 1.0, (
        f"shared strike not recovered: got {recovered_strike:.4f}, "
        f"expected {shared_strike_deg}; diff (mod 180) = {diff_strike:.4f}"
    )

    def _normalise(c: np.ndarray) -> np.ndarray:
        return c / np.linalg.norm(c, ord="fro")

    for sid, _, twist_deg, shear_deg, gain in (
        (s[0], None, s[1], s[2], s[3]) for s in site_specs
    ):
        c_true = _C_from_GB(shared_strike_deg, twist_deg, shear_deg, gain)
        c_rec = result.per_site_distortion[sid]["c_tensor"]
        dist = np.linalg.norm(_normalise(c_rec) - _normalise(c_true), ord="fro")
        assert dist < 0.05, (
            f"site {sid!r}: per-site C tensor recovery failed; "
            f"normalised Frobenius distance = {dist:.6f}"
        )


def test_mj_collapses_to_gb_at_single_site():
    """A single-site MJ run produces results consistent with GB.

    The joint algorithm at ``n_sites=1`` is just GB with extra
    book-keeping. We require:

    - The joint shared strike matches the single-site GB strike
      median within 1°.
    - The per-site twist / shear / gain medians match the GB
      medians at the same loose tolerance.
    """
    z = _build_site_z(theta_deg=42.0, twist_deg=7.0, shear_deg=4.0, gain=1.0)
    gb = decompose(z, seed=42, n_starts=3)
    mj = decompose_mcneice_jones([z], ["only"], seed=42, n_starts=3)

    band_id = next(iter(mj.per_band_strike))
    mj_strike = mj.per_band_strike[band_id]
    gb_strike_med = float(np.nanmedian(gb.parameters["strike"].values))
    diff_strike = abs(mj_strike - gb_strike_med) % 180.0
    diff_strike = min(diff_strike, 180.0 - diff_strike)
    assert diff_strike < 1.0, (
        f"MJ strike {mj_strike:.4f} differs from GB median "
        f"{gb_strike_med:.4f} by {diff_strike:.4f}° (mod 180)"
    )

    rec = mj.per_site_distortion["only"]
    gb_twist_med = float(np.nanmedian(gb.parameters["twist"].values))
    gb_shear_med = float(np.nanmedian(gb.parameters["shear"].values))
    gb_gain_med = float(np.nanmedian(gb.parameters["gain"].values))

    assert abs(rec["twist_deg"] - gb_twist_med) < 1.0
    assert abs(rec["shear_deg"] - gb_shear_med) < 1.0
    # Gain comparison is gauge-noisy; loosen to factor-of-3 in log10.
    assert abs(np.log10(rec["gain"]) - np.log10(gb_gain_med)) < 0.5


def test_mj_disambiguation_passes_through():
    """All four disambiguation strategies are accepted and produce
    consistent C tensors (the strategies pick different strike
    branches but the reconstructed C is GB-symmetry-invariant).
    """
    site_specs = [
        ("A", 6.0, 3.0, 1.0),
        ("B", -4.0, 5.0, 1.0),
        ("C", 8.0, -2.0, 1.0),
    ]
    z_objs = [
        _build_site_z(30.0, twist_deg=t, shear_deg=e, gain=g, seed=i)
        for i, (_, t, e, g) in enumerate(site_specs)
    ]
    site_ids = [s[0] for s in site_specs]

    strategies = ["geometric", "identity", "pt_aligned", "min_shear"]
    c_per_strategy: dict[str, np.ndarray] = {}
    for strat in strategies:
        result = decompose_mcneice_jones(
            z_objs, site_ids, seed=42, n_starts=3, disambiguation=strat
        )
        assert isinstance(result, JointDecompositionResult)
        assert result.metadata["disambiguation"] == strat
        c_per_strategy[strat] = result.per_site_distortion["A"]["c_tensor"]

    def _normalise(c: np.ndarray) -> np.ndarray:
        return c / np.linalg.norm(c, ord="fro")

    ref = _normalise(c_per_strategy["geometric"])
    for strat in strategies[1:]:
        dist = np.linalg.norm(_normalise(c_per_strategy[strat]) - ref, ord="fro")
        assert dist < 0.05, (
            f"normalised C tensors differ between geometric and "
            f"{strat!r}: Frobenius distance {dist:.6f}"
        )


# ---------------------------------------------------------------------------
# Smaller validation tests
# ---------------------------------------------------------------------------


def test_mj_input_validation():
    z = _build_site_z(theta_deg=30.0, twist_deg=5.0, shear_deg=3.0)

    with pytest.raises(NotImplementedError, match="share_strike_within_band"):
        decompose_mcneice_jones([z], ["A"], share_strike_within_band=False)

    with pytest.raises(NotImplementedError, match="per_site_distortion"):
        decompose_mcneice_jones([z], ["A"], per_site_distortion=False)

    with pytest.raises(ValueError, match="length mismatch"):
        decompose_mcneice_jones([z, z], ["A"])

    with pytest.raises(ValueError, match="unique"):
        decompose_mcneice_jones([z, z], ["A", "A"])
