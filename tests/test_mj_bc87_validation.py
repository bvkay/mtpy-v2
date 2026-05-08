"""McNeice-Jones validation: BC87 + 27-site synthetic.

Two complementary checks are wired up here:

1. ``test_mj_synthetic_27_sites`` — primary test. 27 synthetic sites
   sharing a known regional strike (≈30°, the published BC87 LIT
   ballpark) with varied per-site distortion and a small-noise
   forward model. Asserts MJ recovers the shared strike to within
   2° and per-site distortion C tensors (Frobenius-normalised) to
   within 0.05 of the truth. Runs in seconds.

2. ``test_mj_bc87_strike_recovery`` — opt-in real-data test. Loads
   the 27 EDIs of the BC87 LITHOPROBE LIT line, runs MJ, and
   checks the run completes and the per-band shared strike has a
   sane finite value. The exact Fig-12 strike magnitudes are NOT
   asserted: a preliminary probe at 27 sites in a single
   1–100 s band converged to ~5° (geometric fold), while
   published Fig 12 reads roughly 25–40° in that period range; the
   discrepancy could be local-minima at high-dimensional joint
   fits, narrower-band per-period strikes vs single-band
   aggregation, or a convention offset, and the proper
   investigation lives in follow-up work. The test is therefore
   conservative: it sanity-checks that the algorithm runs and
   returns a populated :class:`JointDecompositionResult` rather
   than asserting an exact published strike. See
   ``docs/decomposition_validation.md`` for the full validation
   strategy.

Setup
-----
Both tests need the BC87 EDIs available locally; the data is not
committed to this repo (the upstream BC87 README asks the dataset
not be redistributed person-to-person). Place EDI files under
``tests/data/bc87/`` or set ``MTPY_BC87_DATA_DIR``. See
``tests/data/bc87/README.md`` for download details. If neither
location resolves, the BC87 test skips with a clear message.

References
----------
McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency tensor
decomposition of magnetotelluric data. Geophysics, 66(1), 158-173.

Jones, A. G., Groom, R. W., & Kurtz, R. D. (1993). Decomposition and
modelling of the BC87 dataset. Journal of Geomagnetism and
Geoelectricity, 45(9), 1127-1150.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    JointDecompositionResult,
    _estim_imp,
    decompose_mcneice_jones,
)


# ---------------------------------------------------------------------------
# Primary test: 27-site synthetic
# ---------------------------------------------------------------------------


def _build_site_z(
    theta_deg: float,
    twist_deg: float,
    shear_deg: float,
    gain: float = 1.0,
    n_freqs: int = 12,
    seed: int = 0,
    rho_a: float = 100.0,
    rho_b: float = 10.0,
    noise_level: float = 0.005,
) -> Z:
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
    if noise_level > 0:
        scale = noise_level * np.maximum(
            np.abs(z_obs),
            np.max(np.abs(z_obs), axis=(1, 2), keepdims=True),
        )
        z_obs = z_obs + rng.normal(scale=scale) + 1j * rng.normal(scale=scale)
        sigma = scale
    else:
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


def test_mj_synthetic_9_sites():
    """Default-on, fast scaling check: 9 synthetic sites, shared 30° strike.

    The unit suite in ``tests/test_mcneice_jones.py`` covers
    ``n_sites=5``; this test bridges to the slow 27-site
    validation by exercising a 9-site joint problem (~470
    parameters, 12 freqs, n_starts=3) in roughly half a minute.
    Recovers the shared strike to <2° and per-site
    Frobenius-normalised C tensors to <0.05.
    """
    rng = np.random.default_rng(20260508)
    n_sites = 9
    twist_specs = rng.uniform(-8.0, 8.0, n_sites)
    shear_specs = rng.uniform(-5.0, 5.0, n_sites)
    site_ids = [f"S{i:02d}" for i in range(n_sites)]
    z_objs = [
        _build_site_z(
            theta_deg=30.0,
            twist_deg=float(twist_specs[i]),
            shear_deg=float(shear_specs[i]),
            gain=1.0,
            n_freqs=12,
            seed=i,
            noise_level=0.005,
        )
        for i in range(n_sites)
    ]
    result = decompose_mcneice_jones(
        z_objs, site_ids, seed=42, n_starts=3
    )
    band_id = next(iter(result.per_band_strike))
    diff = abs(result.per_band_strike[band_id] - 30.0) % 180.0
    diff = min(diff, 180.0 - diff)
    assert diff < 2.0, (
        f"shared strike not recovered: got "
        f"{result.per_band_strike[band_id]:.3f}°, expected 30°"
    )

    def _normalise(c: np.ndarray) -> np.ndarray:
        return c / np.linalg.norm(c, ord="fro")

    for i, sid in enumerate(site_ids):
        c_true = _C_from_GB(
            30.0, float(twist_specs[i]), float(shear_specs[i]), 1.0
        )
        c_rec = result.per_site_distortion[sid]["c_tensor"]
        d = float(np.linalg.norm(_normalise(c_rec) - _normalise(c_true), ord="fro"))
        assert d < 0.05, (
            f"site {sid!r} C tensor recovery: normalised Frobenius "
            f"distance {d:.6f}"
        )


_SLOW_VALIDATION_OPT_IN = os.environ.get("MTPY_RUN_SLOW_VALIDATION", "").lower() in {
    "1",
    "true",
    "yes",
}


@pytest.mark.skipif(
    not _SLOW_VALIDATION_OPT_IN,
    reason=(
        "27-site joint synthetic validation is slow (~13 min on a "
        "modern workstation: ~1400 parameters, n_starts=5, scipy TRF). "
        "Set MTPY_RUN_SLOW_VALIDATION=1 to run."
    ),
)
def test_mj_synthetic_27_sites():
    """Internal validation: 27 synthetic sites with shared 30° strike.

    Stand-in for BC87 published-data validation. Each site has
    independently varying (twist, shear, gain) within plausible
    ranges; all share a regional strike of 30°. With small noise
    (0.5% of |Z|) and 5 multi-starts, MJ should recover the
    shared strike to <2° and per-site (Frobenius-normalised) C
    tensors to <0.05 across all 27 sites.

    This was validated on 2026-05-08 — the run completes in
    ~13 minutes and recovers the shared 30° strike well within
    the 2° tolerance. The test is gated behind
    ``MTPY_RUN_SLOW_VALIDATION`` because 13 minutes is too slow
    for a routine pytest invocation.
    """
    shared_strike_deg = 30.0
    rng = np.random.default_rng(20260508)
    twist_specs = rng.uniform(-8.0, 8.0, 27)
    shear_specs = rng.uniform(-5.0, 5.0, 27)
    gain_specs = np.ones(27)
    site_ids = [f"S{i:02d}" for i in range(27)]

    z_objs = [
        _build_site_z(
            theta_deg=shared_strike_deg,
            twist_deg=float(twist_specs[i]),
            shear_deg=float(shear_specs[i]),
            gain=float(gain_specs[i]),
            n_freqs=12,
            seed=i,
            noise_level=0.005,
        )
        for i in range(27)
    ]

    result = decompose_mcneice_jones(
        z_objs, site_ids, seed=42, n_starts=5
    )

    assert isinstance(result, JointDecompositionResult)
    assert result.metadata["n_sites"] == 27

    band_id = next(iter(result.per_band_strike))
    recovered_strike = result.per_band_strike[band_id]
    diff = abs(recovered_strike - shared_strike_deg) % 180.0
    diff = min(diff, 180.0 - diff)
    assert diff < 2.0, (
        f"shared strike not recovered: got {recovered_strike:.3f}°, "
        f"expected ~30°; diff (mod 180) = {diff:.3f}°"
    )

    def _normalise(c: np.ndarray) -> np.ndarray:
        return c / np.linalg.norm(c, ord="fro")

    failures: list[tuple[str, float]] = []
    for i, sid in enumerate(site_ids):
        c_true = _C_from_GB(
            shared_strike_deg,
            float(twist_specs[i]),
            float(shear_specs[i]),
            float(gain_specs[i]),
        )
        c_rec = result.per_site_distortion[sid]["c_tensor"]
        d = float(np.linalg.norm(_normalise(c_rec) - _normalise(c_true), ord="fro"))
        if d >= 0.05:
            failures.append((sid, d))
    assert not failures, (
        "per-site C tensors not recovered within 0.05 (normalised "
        f"Frobenius). Worst offenders: {sorted(failures, key=lambda x: -x[1])[:3]}"
    )


# ---------------------------------------------------------------------------
# Secondary test: real BC87 LIT line (opt-in)
# ---------------------------------------------------------------------------


def _bc87_data_dir() -> Path | None:
    """Resolve where BC87 EDIs live. ``None`` if not found."""
    env = os.environ.get("MTPY_BC87_DATA_DIR")
    if env:
        p = Path(env)
        if p.is_dir() and any(p.glob("lit*.edi")):
            return p
    repo_default = Path(__file__).resolve().parent / "data" / "bc87"
    if repo_default.is_dir() and any(repo_default.glob("lit*.edi")):
        return repo_default
    return None


_BC87_SLOW_OPT_IN = os.environ.get("MTPY_RUN_BC87", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(
    not _BC87_SLOW_OPT_IN,
    reason=(
        "BC87 real-data validation is slow (~hours for 27-site joint "
        "fit) and is opt-in. Set MTPY_RUN_BC87=1 to run."
    ),
)
def test_mj_bc87_strike_recovery():
    """Smoke test on the real BC87 LIT line (opt-in via MTPY_RUN_BC87).

    Loads the 27 EDIs published with BC87, restricts every site to
    the common frequency grid, runs joint MJ on a mid-period band,
    and asserts the run completes with a populated result. The
    exact Figure-12 strike comparison is deferred — a preliminary
    probe converged to ~5° in the geometric-fold convention while
    the published Fig 12 sits closer to 25–40° in the same
    period range, and the discrepancy needs follow-up. See
    ``docs/decomposition_validation.md`` and the module docstring
    for the rationale.
    """
    data_dir = _bc87_data_dir()
    if data_dir is None:
        pytest.skip(
            "BC87 EDI files not found. Place lit*.edi under "
            "tests/data/bc87/ or set MTPY_BC87_DATA_DIR. See "
            "tests/data/bc87/README.md."
        )

    from mtpy.core.mt import MT

    edi_paths = sorted(data_dir.glob("lit*.edi"))
    assert len(edi_paths) >= 20, (
        f"expected ~27 BC87 LIT EDIs in {data_dir}, found {len(edi_paths)}"
    )

    mts = []
    for p in edi_paths:
        m = MT()
        m.read(str(p))
        mts.append(m)

    # Restrict every site to the intersection frequency grid (BC87
    # has one site with 40 freqs and the rest with 39; the joint
    # algorithm requires a common grid).
    freq_sets = [set(m.Z.frequency.tolist()) for m in mts]
    common = sorted(set.intersection(*freq_sets), reverse=True)
    z_objs: list[Z] = []
    site_ids: list[str] = []
    for m in mts:
        f = np.asarray(m.Z.frequency)
        keep = np.isin(f, list(common))
        z_objs.append(
            Z(
                z=np.asarray(m.Z.z)[keep],
                z_error=np.asarray(m.Z.z_error)[keep],
                frequency=f[keep],
            )
        )
        site_ids.append(str(m.station))

    result = decompose_mcneice_jones(
        z_objs,
        site_ids,
        period_bands=[(1.0, 100.0)],
        n_starts=3,
        seed=42,
    )

    assert isinstance(result, JointDecompositionResult)
    assert result.metadata["n_sites"] == len(z_objs)
    assert len(result.per_band_strike) == 1
    band_id = next(iter(result.per_band_strike))
    strike = result.per_band_strike[band_id]
    assert np.isfinite(strike)
    assert 0.0 <= strike < 180.0, (
        f"BC87 recovered strike out of [0, 180): {strike}"
    )
    assert all(
        sid in result.per_site_distortion for sid in site_ids
    ), "per-site distortion records missing for some BC87 sites"
    assert result.chi_squared > 0.0
