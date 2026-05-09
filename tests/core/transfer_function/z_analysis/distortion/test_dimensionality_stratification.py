"""Tests for the dimensionality_stratification module.

Six required scenarios:

1. Strata partition correctness (mutually exclusive, exhaustive).
2. ``trust_score`` continuity along a numeric criterion.
3. Default rules align with categorical stratification:
   high_trust rows → score ≥ 0.9; excluded rows → score ≤ 0.1.
4. Sensitivity sweep is monotonic in the threshold.
5. Filtering does not mutate the input table.
6. Round-trip with the continental-observables pipeline:
   stratification fractions sum to 1.0.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mtpy.core.transfer_function.z_analysis.decomposition import (
    DEFAULT_TRUST_RULES,
    OBSERVABLE_COLUMNS,
    STRATA_ORDER,
    ObservableTable,
    apply_trust_filter,
    compute_collection_observables,
    stratification_summary,
    stratify_table,
    threshold_sensitivity,
    trust_score,
)


# ---------------------------------------------------------------------------
# Synthetic-table helpers
# ---------------------------------------------------------------------------


def _empty_row() -> dict:
    """A row with all schema columns, blank-defaulted."""
    row: dict = {col: np.nan for col in OBSERVABLE_COLUMNS}
    row["site_id"] = "S"
    row["longitude_deg"] = 140.0
    row["latitude_deg"] = -30.0
    row["elevation_m"] = 0.0
    row["period_band_label"] = "10s_100s"
    row["period_band_geomean_s"] = 31.6
    row["WALDIM_case"] = pd.NA
    row["Lilley_category"] = pd.NA
    row["GB_mode_warning"] = pd.NA
    row["magnetic_distortion_flag"] = pd.NA
    row["dimensionality_concordant"] = pd.NA
    return row


def _make_high_trust_row(site_id: str = "HT001") -> dict:
    """A row that satisfies every ``high_trust`` criterion."""
    row = _empty_row()
    row["site_id"] = site_id
    row["WALDIM_case"] = 2
    row["Lilley_category"] = "2D"
    row["PT_abs_beta_deg"] = 1.0
    row["magnetic_distortion_flag"] = "low_risk"
    row["GB_mode_warning"] = False
    row["cross_method_strike_disagreement_deg"] = 2.0
    row["GB_rms_misfit"] = 1.2
    # Fill with sane numeric values for stat columns
    row["discordance_deg"] = 5.0
    row["gamma_magnitude"] = 0.1
    row["C_minus_I_F"] = 0.2
    return row


def _make_moderate_trust_row(site_id: str = "MOD001") -> dict:
    """Row that fails ``high_trust`` but satisfies
    ``moderate_trust``.
    """
    row = _empty_row()
    row["site_id"] = site_id
    row["WALDIM_case"] = 4
    row["Lilley_category"] = "3D-2D"  # not in high_trust set
    row["PT_abs_beta_deg"] = 4.0  # passes mod (<6) fails high (<3)
    row["magnetic_distortion_flag"] = "moderate_risk"  # not "low_risk"
    row["GB_mode_warning"] = False
    row["cross_method_strike_disagreement_deg"] = 12.0  # < 15
    row["GB_rms_misfit"] = 1.8
    row["discordance_deg"] = 12.0
    row["gamma_magnitude"] = 0.2
    return row


def _make_low_trust_row(site_id: str = "LOW001") -> dict:
    """Row that fails high and moderate but is not excluded.

    WALDIM=5 (3D) but PT_abs_beta < 6, magnetic_distortion not
    high_risk, Lilley_category not "indeterminate".
    """
    row = _empty_row()
    row["site_id"] = site_id
    row["WALDIM_case"] = 5
    row["Lilley_category"] = "3D"
    row["PT_abs_beta_deg"] = 5.0
    row["magnetic_distortion_flag"] = "moderate_risk"
    row["GB_mode_warning"] = True
    row["cross_method_strike_disagreement_deg"] = 25.0
    row["GB_rms_misfit"] = 3.0
    row["discordance_deg"] = 30.0
    row["gamma_magnitude"] = 0.4
    return row


def _make_excluded_row(site_id: str = "EX001", reason: str = "1d") -> dict:
    row = _empty_row()
    row["site_id"] = site_id
    row["GB_mode_warning"] = False
    row["magnetic_distortion_flag"] = "low_risk"
    row["Lilley_category"] = "2D"
    row["PT_abs_beta_deg"] = 1.0
    row["WALDIM_case"] = 2
    row["cross_method_strike_disagreement_deg"] = 2.0
    row["GB_rms_misfit"] = 1.0
    if reason == "1d":
        row["WALDIM_case"] = 1
    elif reason == "true_3d":
        row["WALDIM_case"] = 6
    elif reason == "high_magnetic":
        row["magnetic_distortion_flag"] = "high_risk"
    elif reason == "high_beta":
        row["PT_abs_beta_deg"] = 8.0
    elif reason == "indeterminate":
        row["Lilley_category"] = "indeterminate"
    else:
        raise ValueError(f"unknown reason {reason!r}")
    return row


def _table_from_rows(rows: list[dict]) -> ObservableTable:
    df = pd.DataFrame(rows, columns=OBSERVABLE_COLUMNS)
    df["site_id"] = df["site_id"].astype("string")
    df["period_band_label"] = df["period_band_label"].astype("string")
    df["WALDIM_case"] = df["WALDIM_case"].astype("Int64")
    df["Lilley_category"] = df["Lilley_category"].astype("string")
    df["magnetic_distortion_flag"] = df["magnetic_distortion_flag"].astype(
        "string"
    )
    df["GB_mode_warning"] = df["GB_mode_warning"].astype("boolean")
    df["dimensionality_concordant"] = df[
        "dimensionality_concordant"
    ].astype("boolean")
    return ObservableTable(dataframe=df, metadata={"_test": True})


# ---------------------------------------------------------------------------
# Test 1 — strata partition
# ---------------------------------------------------------------------------


def test_strata_partition_is_mutually_exclusive_and_exhaustive():
    rows = [
        _make_high_trust_row("A"),
        _make_high_trust_row("B"),
        _make_moderate_trust_row("C"),
        _make_low_trust_row("D"),
        _make_excluded_row("E", "1d"),
        _make_excluded_row("F", "true_3d"),
        _make_excluded_row("G", "high_magnetic"),
        _make_excluded_row("H", "high_beta"),
        _make_excluded_row("I", "indeterminate"),
    ]
    table = _table_from_rows(rows)
    strata = stratify_table(table)

    # Every stratum key present.
    assert set(strata.keys()) == set(STRATA_ORDER)
    # Counts add up to n_total.
    n_total = sum(len(t.dataframe) for t in strata.values())
    assert n_total == len(rows)
    # Mutually exclusive: each site_id appears in exactly one stratum.
    site_to_stratum: dict[str, str] = {}
    for s, t in strata.items():
        for sid in t.dataframe["site_id"].dropna().tolist():
            assert sid not in site_to_stratum, (
                f"site {sid} in both {site_to_stratum.get(sid)} and {s}"
            )
            site_to_stratum[sid] = s
    assert set(site_to_stratum.keys()) == {r["site_id"] for r in rows}

    # Spot-checks on tier assignments.
    assert site_to_stratum["A"] == "high_trust"
    assert site_to_stratum["C"] == "moderate_trust"
    assert site_to_stratum["D"] == "low_trust"
    for sid in ("E", "F", "G", "H", "I"):
        assert site_to_stratum[sid] == "excluded"


# ---------------------------------------------------------------------------
# Test 2 — trust_score continuity
# ---------------------------------------------------------------------------


def test_trust_score_smooth_in_numeric_threshold():
    base = _make_high_trust_row()
    pt_betas = np.linspace(0.0, 10.0, 41)
    scores = []
    for b in pt_betas:
        row = dict(base)
        row["PT_abs_beta_deg"] = float(b)
        scores.append(trust_score(row))
    scores = np.asarray(scores, dtype=np.float64)

    # No NaNs.
    assert np.all(np.isfinite(scores))
    # Strictly monotonically decreasing as beta increases (further
    # from the threshold of 3°).
    assert np.all(np.diff(scores) <= 1e-9), (
        "trust_score should be monotonically non-increasing as "
        "PT_abs_beta_deg moves away from satisfying."
    )
    # Endpoint sanity: low beta well above 0.5; high beta well below.
    assert scores[0] > 0.85
    assert scores[-1] < 0.5
    # Smooth: largest single-step drop is well under a step-function
    # height (which would be ≈ scores[0] − scores[-1]).
    max_step = float(np.max(np.abs(np.diff(scores))))
    total_drop = float(scores[0] - scores[-1])
    assert max_step < 0.5 * total_drop, (
        f"largest step {max_step:.3f} exceeds half the total drop "
        f"{total_drop:.3f} — score is too step-like, not a sigmoid."
    )


# ---------------------------------------------------------------------------
# Test 3 — default rules: high_trust >= 0.9, excluded <= 0.1
# ---------------------------------------------------------------------------


def test_high_trust_score_above_threshold():
    row = _make_high_trust_row()
    score = trust_score(row)
    assert score >= 0.9, (
        f"high-trust row scored {score:.3f}, expected >= 0.9"
    )


def test_excluded_score_below_threshold():
    # Each excluded reason should produce a low score because at
    # least one categorical/numeric criterion is in violation.
    for reason in ["1d", "true_3d", "high_magnetic", "high_beta",
                   "indeterminate"]:
        row = _make_excluded_row(site_id=f"EX_{reason}", reason=reason)
        score = trust_score(row)
        assert score <= 0.1, (
            f"excluded row reason={reason!r} scored {score:.3f}; "
            f"expected <= 0.1"
        )


# ---------------------------------------------------------------------------
# Test 4 — sensitivity sweep is monotonic in the threshold
# ---------------------------------------------------------------------------


def test_threshold_sensitivity_is_monotonic_in_pt_beta():
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for i in range(40):
        # Build a population with PT_abs_beta uniform on [0, 10].
        row = _make_high_trust_row(site_id=f"S{i:03d}")
        row["PT_abs_beta_deg"] = float(rng.uniform(0.0, 10.0))
        rows.append(row)
    table = _table_from_rows(rows)

    thresholds = np.linspace(1.0, 10.0, 10)
    out = threshold_sensitivity(
        table, observable_name="discordance_deg",
        rule_column="PT_abs_beta_deg",
        threshold_range=thresholds,
    )
    assert out.shape == (10, 3)
    fractions = out[:, 1]
    # Monotonically non-decreasing as the threshold rises (looser
    # threshold → more sites pass).
    assert np.all(np.diff(fractions) >= -1e-9), (
        "fraction_high_trust should be non-decreasing as the "
        "PT_abs_beta_deg threshold rises."
    )
    # Endpoints behave: at threshold=1, fewer sites pass than at
    # threshold=10.
    assert fractions[0] <= fractions[-1]


# ---------------------------------------------------------------------------
# Test 5 — filtering does not mutate the input
# ---------------------------------------------------------------------------


def test_apply_trust_filter_does_not_mutate_input():
    rows = [
        _make_high_trust_row("A"),
        _make_moderate_trust_row("B"),
        _make_low_trust_row("C"),
        _make_excluded_row("D", "high_magnetic"),
    ]
    table = _table_from_rows(rows)
    df_before = table.dataframe.copy(deep=True)
    meta_before = dict(table.metadata)

    sub = apply_trust_filter(table, min_trust=0.5)

    # Input dataframe unchanged.
    pd.testing.assert_frame_equal(table.dataframe, df_before)
    assert table.metadata == meta_before
    # Output is a different object.
    assert sub.dataframe is not table.dataframe
    # Output keeps at least the high-trust row.
    assert "A" in sub.dataframe["site_id"].astype(str).tolist()


def test_stratify_does_not_mutate_input():
    rows = [_make_high_trust_row("A"), _make_excluded_row("X", "1d")]
    table = _table_from_rows(rows)
    df_before = table.dataframe.copy(deep=True)
    _ = stratify_table(table)
    pd.testing.assert_frame_equal(table.dataframe, df_before)


# ---------------------------------------------------------------------------
# Test 6 — round-trip via continental_observables
# ---------------------------------------------------------------------------


def _make_mt(z_obj, station: str):
    return SimpleNamespace(
        Z=z_obj, Tipper=None,
        station=station, longitude=140.0, latitude=-30.0, elevation=0.0,
    )


def test_round_trip_with_continental_observables():
    """Running the continental pipeline and then stratifying gives a
    summary whose fractions sum to 1.0.
    """
    pytest.importorskip("scipy")
    from .synthetics import generate_synthetic_z

    periods = np.logspace(0.0, np.log10(3000.0), 24)
    sites = []
    for sid in ["S1", "S2", "S3"]:
        syn = generate_synthetic_z(
            regional_type="2D",
            distortion_strength="moderate",
            distortion_shear="low",
            noise_level="clean",
            periods=periods,
            site_id=sid, seed=42,
        )
        sites.append(_make_mt(syn["z_obj"], station=sid))

    bands = [
        {"label": "1s_10s", "period_min": 1.0, "period_max": 10.0},
        {"label": "10s_100s", "period_min": 10.0, "period_max": 100.0},
    ]
    table = compute_collection_observables(
        sites, period_bands=bands, n_starts=3, seed=42,
    )
    summary = stratification_summary(table)
    total_fraction = sum(summary.fraction_per_stratum.values())
    assert abs(total_fraction - 1.0) < 1e-9
    assert summary.n_total_rows == sum(summary.n_per_stratum.values())
    # Every canonical stratum is present in the dict, even if zero.
    assert set(summary.n_per_stratum.keys()) == set(STRATA_ORDER)


# ---------------------------------------------------------------------------
# Bonus smoke tests
# ---------------------------------------------------------------------------


def test_default_trust_rules_structure_is_well_formed():
    for tier in ("high_trust", "moderate_trust", "excluded"):
        assert tier in DEFAULT_TRUST_RULES
        block = DEFAULT_TRUST_RULES[tier]
        assert block["logic"] in ("all", "any")
        if block["logic"] == "all":
            assert isinstance(block["criteria"], dict)
        else:
            assert isinstance(block["criteria"], list)


def test_trust_score_handles_missing_values_gracefully():
    """A row with NA on a high_trust criterion zeroes that
    criterion's contribution, yielding a low overall score.
    """
    row = _make_high_trust_row()
    row["WALDIM_case"] = pd.NA
    score = trust_score(row)
    assert score < 0.5


def test_pipeline_speed_under_5s_for_synthetic_table():
    """Stratification on a 1000-row synthetic table runs in under 5
    seconds — comfortably below the AusLAMP-scale (~6500-row) budget.
    """
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for i in range(1000):
        # Uniformly draw values that span all four tiers.
        row = _make_high_trust_row(site_id=f"X{i:04d}")
        row["WALDIM_case"] = int(rng.integers(1, 8))
        row["PT_abs_beta_deg"] = float(rng.uniform(0.0, 10.0))
        row["GB_rms_misfit"] = float(rng.uniform(0.5, 3.0))
        row["cross_method_strike_disagreement_deg"] = float(
            rng.uniform(0.0, 30.0)
        )
        rows.append(row)
    table = _table_from_rows(rows)
    t0 = time.perf_counter()
    summary = stratification_summary(table)
    t1 = time.perf_counter()
    assert (t1 - t0) < 5.0, (
        f"stratification on 1000 rows took {t1 - t0:.2f}s; "
        f"AusLAMP-scale (~6500 rows) would not meet the 5s target."
    )
    # Sanity: row count preserved.
    assert summary.n_total_rows == 1000
