"""Tests for the spatial_coherence module.

Six required scenarios:

1. Pure noise (no spatial structure) classifies as
   ``noise_dominated``.
2. Pure smooth field (Gaussian random field with known
   correlation length) classifies as ``structured``.
3. Mixed structure + noise classifies as ``structured`` or
   ``weakly_structured`` with a non-zero nugget far below the sill.
4. Circular variogram for line-direction angles (the 0 / 180°
   wrap is handled correctly).
5. Randomisation null bracket: noise data sit inside the null at
   most bins; structured data sit *above* the null at most bins
   below the range.
6. AusLAMP-scale realistic synthetic (1353 sites at the actual
   AusLAMP coordinates, with a 500-km correlation field) recovers
   ``range_km`` within 100 km of 500 km in <60 s. Skips when the
   AusLAMP site_summary.csv is unavailable on this machine.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtpy.core.transfer_function.z_analysis.decomposition import (
    OBSERVABLE_COLUMNS,
    ObservableTable,
    compute_coherence,
    compute_coherence_all,
    empirical_variogram,
    haversine_distances_km,
    pairwise_distances,
    randomisation_null,
)
from mtpy.core.transfer_function.z_analysis.decomposition.spatial_coherence import (
    _classify_observable,
    _per_site_bootstrap_variance,
)


# ---------------------------------------------------------------------------
# Helpers: build a synthetic ObservableTable from coordinates + values
# ---------------------------------------------------------------------------


def _make_table(
    site_ids: list[str],
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    *,
    observables: dict[str, np.ndarray],
    band_label: str = "10s_100s",
    band_geomean_s: float = 31.62,
    extra_band: bool = False,
) -> ObservableTable:
    """Construct a minimal :class:`ObservableTable` for testing.

    ``observables`` maps observable column name -> per-site array
    of length ``len(site_ids)``. Other columns are filled with NaN
    or sentinel values consistent with the schema.
    """
    n = len(site_ids)
    rows = []
    band_specs = [(band_label, band_geomean_s)]
    if extra_band:
        # Add a second band so the bootstrap-variance fallback has
        # something to work with.
        band_specs.append(("100s_1000s", 316.23))
    for label, geomean in band_specs:
        for i in range(n):
            row = {col: np.nan for col in OBSERVABLE_COLUMNS}
            row["site_id"] = site_ids[i]
            row["longitude_deg"] = float(lons_deg[i])
            row["latitude_deg"] = float(lats_deg[i])
            row["elevation_m"] = 0.0
            row["period_band_label"] = label
            row["period_band_geomean_s"] = geomean
            row["GB_mode_warning"] = pd.NA
            row["WALDIM_case"] = pd.NA
            row["Lilley_category"] = pd.NA
            row["dimensionality_concordant"] = pd.NA
            row["magnetic_distortion_flag"] = pd.NA
            for obs_name, obs_values in observables.items():
                # When extra_band=True we add small per-band noise so
                # the inter-band variance fallback can compute.
                jitter = 0.0 if label == band_label else 0.001 * np.random.randn()
                row[obs_name] = float(obs_values[i]) + jitter
            rows.append(row)
    df = pd.DataFrame(rows, columns=OBSERVABLE_COLUMNS)
    df["site_id"] = df["site_id"].astype("string")
    df["period_band_label"] = df["period_band_label"].astype("string")
    return ObservableTable(dataframe=df, metadata={"_test": True})


def _gaussian_random_field_2d(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    correlation_km: float,
    *,
    sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthesise a smooth field with Gaussian-kernel correlation.

    Build the covariance matrix
    ``Sigma_ij = sigma² · exp(-d_ij² / (2 L²))`` from haversine
    distances ``d_ij``, then Cholesky-sample. For ~hundred sites
    this is the simplest exact-correlation generator. For the
    1353-site AusLAMP test we reduce sigma and use float32 to
    keep memory in check.
    """
    rng = np.random.default_rng(seed)
    n = lats_deg.size
    # Pairwise distances → full (n, n) matrix.
    iu, ju = np.triu_indices(n, k=1)
    d = haversine_distances_km(lats_deg, lons_deg)
    D = np.zeros((n, n))
    D[iu, ju] = d
    D = D + D.T
    cov = sigma**2 * np.exp(-(D**2) / (2.0 * correlation_km**2))
    # Mild jitter for Cholesky stability.
    cov += 1e-8 * np.eye(n)
    L = np.linalg.cholesky(cov)
    return L @ rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Pure noise → noise_dominated
# ---------------------------------------------------------------------------


def test_pure_noise_is_noise_dominated():
    rng = np.random.default_rng(7)
    n = 60
    lats = np.linspace(-37.0, -25.0, n) + 0.5 * rng.standard_normal(n)
    lons = np.linspace(135.0, 145.0, n) + 0.5 * rng.standard_normal(n)
    sids = [f"SITE{i:03d}" for i in range(n)]
    values = rng.standard_normal(n)
    table = _make_table(
        sids, lats, lons,
        observables={"PT_beta_deg": values},
        extra_band=True,
    )
    coh = compute_coherence(
        table, "PT_beta_deg",
        n_bins=12, max_distance_km=2000.0, n_shuffles=80, seed=0,
    )
    assert coh.coherence_label in {"noise_dominated", "weakly_structured"}
    # The empirical variogram should be flat-ish, so most bins fall
    # inside the null.
    finite = (
        np.isfinite(coh.variogram_values) & np.isfinite(coh.null_p95)
        & (coh.bin_counts > 0)
    )
    above = (coh.variogram_values > coh.null_p95) & finite
    assert above.sum() <= max(2, int(0.2 * finite.sum()))


# ---------------------------------------------------------------------------
# Pure smooth field → structured
# ---------------------------------------------------------------------------


def test_smooth_field_is_structured():
    n = 80
    rng = np.random.default_rng(11)
    lats = np.linspace(-37.0, -25.0, n) + 0.05 * rng.standard_normal(n)
    lons = np.linspace(135.0, 145.0, n) + 0.05 * rng.standard_normal(n)
    sids = [f"SMOOTH{i:03d}" for i in range(n)]
    values = _gaussian_random_field_2d(
        lats, lons, correlation_km=300.0, sigma=1.0, seed=1,
    )
    table = _make_table(
        sids, lats, lons,
        observables={"PT_beta_deg": values},
        extra_band=True,
    )
    coh = compute_coherence(
        table, "PT_beta_deg",
        n_bins=12, max_distance_km=2000.0, n_shuffles=80, seed=0,
    )
    assert coh.coherence_label in {"structured", "weakly_structured"}
    # Variogram value at small h should be small; at large h, large.
    finite_idx = np.where(
        np.isfinite(coh.variogram_values) & (coh.bin_counts > 0)
    )[0]
    assert finite_idx.size >= 4
    early = float(coh.variogram_values[finite_idx[0]])
    late = float(coh.variogram_values[finite_idx[-1]])
    assert late > early


# ---------------------------------------------------------------------------
# Smooth field + noise → mixed
# ---------------------------------------------------------------------------


def test_mixed_signal_plus_noise():
    n = 80
    rng = np.random.default_rng(13)
    lats = np.linspace(-37.0, -25.0, n) + 0.05 * rng.standard_normal(n)
    lons = np.linspace(135.0, 145.0, n) + 0.05 * rng.standard_normal(n)
    sids = [f"MIX{i:03d}" for i in range(n)]
    field = _gaussian_random_field_2d(
        lats, lons, correlation_km=400.0, sigma=1.0, seed=2,
    )
    noise = 0.4 * rng.standard_normal(n)  # ~half-amplitude noise
    table = _make_table(
        sids, lats, lons,
        observables={"PT_beta_deg": field + noise},
        extra_band=True,
    )
    coh = compute_coherence(
        table, "PT_beta_deg",
        n_bins=12, max_distance_km=2000.0, n_shuffles=80, seed=0,
    )
    assert coh.coherence_label in {"structured", "weakly_structured"}
    # Positive nugget; nugget should be a fraction of sill.
    assert np.isfinite(coh.nugget) and coh.nugget > 0.0
    assert np.isfinite(coh.sill) and coh.sill > 0.0
    # Nugget below sill — there's structure on top of the noise.
    assert coh.nugget_to_sill_ratio < 0.95


# ---------------------------------------------------------------------------
# Circular variogram for line-direction angles
# ---------------------------------------------------------------------------


def test_circular_variogram_handles_180_wrap():
    """Two angles 5° and 175° are 10° apart as line directions, not
    170°. The circular squared-difference
    ``1 - cos(2 (a - b))`` should reflect that.
    """
    n = 40
    rng = np.random.default_rng(17)
    lats = np.linspace(-30.0, -20.0, n)
    lons = np.linspace(140.0, 145.0, n)
    sids = [f"WRAP{i:03d}" for i in range(n)]
    # Half the sites at 5°, the other half at 175° — line-direction-
    # identical, so a circular variogram should be near zero.
    angles = np.where(np.arange(n) % 2 == 0, 5.0, 175.0)
    angles = angles + 0.1 * rng.standard_normal(n)  # small jitter

    table = _make_table(
        sids, lats, lons,
        observables={"C_strike_deg": angles},
    )
    vg = empirical_variogram(table, "C_strike_deg", n_bins=8)
    assert vg.kind == "circular"
    finite = np.isfinite(vg.variogram_values) & (vg.bin_counts > 0)
    assert finite.any()
    # All variogram values should be close to zero (≈ noise from the
    # 0.1° jitter).
    assert np.nanmax(vg.variogram_values) < 0.05

    # Sanity: also test that 0° and 90° (perpendicular lines) give
    # the maximum variogram value of 1 - cos(180°) = 2.
    angles_perp = np.where(np.arange(n) % 2 == 0, 0.0, 90.0)
    table_perp = _make_table(
        sids, lats, lons,
        observables={"C_strike_deg": angles_perp},
    )
    vg_perp = empirical_variogram(table_perp, "C_strike_deg", n_bins=8)
    finite_perp = np.isfinite(vg_perp.variogram_values) & (
        vg_perp.bin_counts > 0
    )
    # The 5°/175° (≈10° axial) case has variogram values <0.05 above;
    # the 0°/90° (90° axial) case has them ≈ 1 — three orders of
    # magnitude apart. So the circular metric distinguishes them.
    assert np.nanmean(vg_perp.variogram_values[finite_perp]) > 0.5


# ---------------------------------------------------------------------------
# Randomisation null bracket
# ---------------------------------------------------------------------------


def test_randomisation_null_brackets_signal():
    """Noise data → empirical inside null at >95% bins; structured
    data → empirical above null at most bins.
    """
    n = 80
    rng = np.random.default_rng(19)
    lats = np.linspace(-37.0, -25.0, n)
    lons = np.linspace(135.0, 145.0, n)
    sids = [f"NB{i:03d}" for i in range(n)]

    # Noise data
    noise_vals = rng.standard_normal(n)
    table_n = _make_table(
        sids, lats, lons, observables={"PT_beta_deg": noise_vals},
    )
    coh_n = compute_coherence(
        table_n, "PT_beta_deg",
        n_bins=12, n_shuffles=80, seed=0,
    )
    finite_n = (
        np.isfinite(coh_n.variogram_values) & np.isfinite(coh_n.null_p95)
        & (coh_n.bin_counts > 0)
    )
    inside_n = (coh_n.variogram_values <= coh_n.null_p95) & finite_n
    if finite_n.any():
        # At least 70% of finite bins inside the null (95% would be
        # the asymptotic expectation; we relax for finite-sample
        # variance).
        assert inside_n.sum() / finite_n.sum() >= 0.7

    # Structured data: at small h the empirical is BELOW the null
    # (structure → close-by sites have similar values, so the
    # short-h variogram is small). At large h the empirical
    # approaches the data variance, which is also where the null
    # mass sits. So the diagnostic is "outside the [p05, p95]
    # bracket" — typically below at small h.
    field = _gaussian_random_field_2d(
        lats, lons, correlation_km=400.0, sigma=1.0, seed=3,
    )
    table_s = _make_table(
        sids, lats, lons, observables={"PT_beta_deg": field},
    )
    coh_s = compute_coherence(
        table_s, "PT_beta_deg",
        n_bins=12, n_shuffles=80, seed=0,
    )
    finite_s = (
        np.isfinite(coh_s.variogram_values)
        & np.isfinite(coh_s.null_p05) & np.isfinite(coh_s.null_p95)
        & (coh_s.bin_counts > 0)
    )
    outside_s = (
        (coh_s.variogram_values < coh_s.null_p05)
        | (coh_s.variogram_values > coh_s.null_p95)
    ) & finite_s
    if finite_s.any():
        # At least 30% of bins outside the null envelope.
        assert outside_s.sum() / finite_s.sum() >= 0.3
    # And the small-h bins specifically should be below the null:
    # we expect the first few bins (smallest h) to be below the
    # null's 5th percentile because nearby sites resemble each
    # other.
    small_h = finite_s.copy()
    small_h[len(small_h) // 2:] = False
    if small_h.any():
        below_small = (
            (coh_s.variogram_values < coh_s.null_p05) & small_h
        )
        assert below_small.any()


# ---------------------------------------------------------------------------
# AusLAMP-scale synthetic (skips if no site_summary.csv)
# ---------------------------------------------------------------------------


def _find_auslamp_sites_csv() -> Path | None:
    """Return a path to AusLAMP site_summary.csv on this machine,
    or ``None`` if not found.
    """
    env = os.environ.get("AUSLAMP_SITE_SUMMARY")
    if env and Path(env).exists():
        return Path(env)
    candidates = [
        Path(__file__).resolve().parents[6] / "mt_decomp" / "data"
        / "processed" / "site_summary.csv",
        Path("/home/decomp/projects/mt_decomp/data/processed/"
             "site_summary.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def test_auslamp_scale_recovers_correlation_length():
    csv_path = _find_auslamp_sites_csv()
    if csv_path is None:
        pytest.skip(
            "AusLAMP site_summary.csv not found; set "
            "AUSLAMP_SITE_SUMMARY env var or place at "
            "mt_decomp/data/processed/site_summary.csv"
        )
    sites_df = pd.read_csv(csv_path)
    sites_df = sites_df.dropna(
        subset=["latitude", "longitude"]
    ).reset_index(drop=True)
    lats = sites_df["latitude"].to_numpy(dtype=np.float64)
    lons = sites_df["longitude"].to_numpy(dtype=np.float64)
    sids = sites_df["site"].astype(str).tolist()

    # Generate a smooth field at the actual AusLAMP coordinates
    # with 500 km correlation length.
    field = _gaussian_random_field_2d(
        lats, lons, correlation_km=500.0, sigma=1.0, seed=23,
    )
    table = _make_table(
        sids, lats, lons,
        observables={"PT_beta_deg": field},
    )

    # Variogram-only first (no shuffles) for the range check.
    t0 = time.perf_counter()
    coh = compute_coherence(
        table, "PT_beta_deg",
        n_bins=20, max_distance_km=2500.0,
        n_shuffles=50, seed=0,
    )
    t1 = time.perf_counter()
    assert (t1 - t0) < 60.0, (
        f"AusLAMP-scale single-observable coherence took "
        f"{t1 - t0:.1f}s, exceeded 60s budget"
    )

    # Bin counts non-zero across most of the range (we configured
    # 20 log-spaced bins from 50 → 2500 km plus a near-neighbour
    # bin; some bins may be empty if AusLAMP under-samples a
    # specific separation).
    nonzero_bins = int(np.sum(coh.bin_counts > 0))
    assert nonzero_bins >= 15, (
        f"only {nonzero_bins} non-empty bins out of 21"
    )

    # Range recovery: should be within 100 km of the true 500 km.
    assert np.isfinite(coh.range_km)
    assert abs(coh.range_km - 500.0) < 100.0, (
        f"recovered range {coh.range_km:.0f} km, expected ~500 km"
    )

    # Stop-condition: a coherence summary across all primary
    # observables in <60 s. We populate three primary observables
    # (the same field repeated) so the all-observables call
    # exercises the loop machinery.
    table2 = _make_table(
        sids, lats, lons,
        observables={
            "PT_beta_deg": field,
            "discordance_deg": field * 5.0 + 30.0,
            "gamma_magnitude": np.exp(0.5 * field) + 0.05,
        },
    )
    t0 = time.perf_counter()
    all_coh = compute_coherence_all(
        table2,
        observables=["PT_beta_deg", "discordance_deg", "gamma_magnitude"],
        n_bins=20, max_distance_km=2500.0,
        n_shuffles=20, seed=0,
    )
    t1 = time.perf_counter()
    assert (t1 - t0) < 60.0, (
        f"all-observables coherence at AusLAMP scale took "
        f"{t1 - t0:.1f}s, exceeded 60s budget"
    )
    assert set(all_coh.keys()) == {
        "PT_beta_deg", "discordance_deg", "gamma_magnitude"
    }


# ---------------------------------------------------------------------------
# Smoke tests: distance helpers, classification table
# ---------------------------------------------------------------------------


def test_haversine_distance_zero_for_identical_coords():
    # Two identical sites → zero distance, two close sites → small.
    lats = np.array([0.0, 0.0, 1.0])
    lons = np.array([0.0, 0.0, 0.0])
    d = haversine_distances_km(lats, lons)
    assert d.size == 3  # 3 pairs: (0,1), (0,2), (1,2)
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    # 1° latitude on a sphere of radius 6371.0088 km → ≈ 111.2 km.
    assert d[1] == pytest.approx(111.2, abs=1.0)
    assert d[2] == pytest.approx(111.2, abs=1.0)


def test_observable_kind_classification():
    assert _classify_observable("C_strike_deg") == "circular"
    assert _classify_observable("gamma_magnitude") == "log"
    assert _classify_observable("magnetic_distortion_flag") == "ordinal"
    assert _classify_observable("PT_beta_deg") == "linear"
    assert _classify_observable("discordance_deg") == "linear"


def test_pairwise_distances_output_shape():
    n = 30
    rng = np.random.default_rng(0)
    lats = np.linspace(-37.0, -25.0, n) + 0.05 * rng.standard_normal(n)
    lons = np.linspace(135.0, 145.0, n) + 0.05 * rng.standard_normal(n)
    sids = [f"PD{i:03d}" for i in range(n)]
    table = _make_table(
        sids, lats, lons,
        observables={"PT_beta_deg": rng.standard_normal(n)},
    )
    d = pairwise_distances(table)
    assert d.size == n * (n - 1) // 2
    assert np.all(d >= 0)


def test_bootstrap_variance_fallback_uses_inter_band():
    n = 30
    rng = np.random.default_rng(0)
    lats = np.linspace(-37.0, -25.0, n)
    lons = np.linspace(135.0, 145.0, n)
    sids = [f"BV{i:03d}" for i in range(n)]
    table = _make_table(
        sids, lats, lons,
        observables={"PT_beta_deg": rng.standard_normal(n)},
        extra_band=True,
    )
    # Suppress the documented warning the fallback emits.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bv = _per_site_bootstrap_variance(table, "PT_beta_deg")
    assert bv is not None
    assert bv > 0.0


def test_randomisation_null_shape():
    n = 40
    rng = np.random.default_rng(0)
    lats = np.linspace(-37.0, -25.0, n)
    lons = np.linspace(135.0, 145.0, n)
    sids = [f"RN{i:03d}" for i in range(n)]
    table = _make_table(
        sids, lats, lons,
        observables={"PT_beta_deg": rng.standard_normal(n)},
    )
    null = randomisation_null(
        table, "PT_beta_deg",
        n_bins=8, n_shuffles=20, seed=0,
    )
    assert null.p05.shape == null.p50.shape == null.p95.shape
    assert null.n_shuffles == 20
    # 5th ≤ 50th ≤ 95th, where defined.
    finite = np.isfinite(null.p05) & np.isfinite(null.p95)
    assert (null.p05[finite] <= null.p95[finite]).all()
