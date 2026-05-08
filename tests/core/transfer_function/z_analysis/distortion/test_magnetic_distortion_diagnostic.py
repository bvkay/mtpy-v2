"""Tests for the magnetic-distortion heuristic flag.

Six controlled synthetics exercise each of the three sub-
diagnostics individually and the combined-flag rule. The tests
verify *flag classification*, not the diagnostic's underlying
correctness as a magnetic-distortion detector — see the module
docstring for the heuristic-not-corrective scope.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mtpy.core.transfer_function.tipper import Tipper
from mtpy.core.transfer_function.z_analysis.decomposition import (
    MagneticDistortionFlag,
    compute_magnetic_distortion_flag,
    frequency_dependence_diagnostic,
    method_inconsistency_diagnostic,
    tipper_diagnostic,
)


def _make_mt(
    tipper_amplitude: float | np.ndarray | None = 0.05,
    n_freqs: int = 8,
    station: str = "S01",
) -> SimpleNamespace:
    """Build a minimal MT mock with a populated Tipper attribute.

    ``tipper_amplitude`` may be:
      - ``None`` → no Tipper attribute at all (insufficient-data
        case).
      - scalar float → constant per-period |T| (clean-2-D-like).
      - ``(n_freqs,)`` array → per-period |T| profile.
    """
    if tipper_amplitude is None:
        return SimpleNamespace(station=station, Tipper=None)

    freq = np.logspace(0.0, -2.0, n_freqs)
    if np.isscalar(tipper_amplitude):
        amp = np.full(n_freqs, float(tipper_amplitude))
    else:
        amp = np.asarray(tipper_amplitude, dtype=np.float64)
    # |T_zx|² + |T_zy|² = amp² → set each to amp / sqrt(2).
    per_axis = amp / np.sqrt(2.0)
    t = np.zeros((n_freqs, 1, 2), dtype=np.complex128)
    # Real-only tipper for simplicity (just for the diagnostic).
    t[:, 0, 0] = per_axis
    t[:, 0, 1] = per_axis
    err = np.full((n_freqs, 1, 2), 0.01, dtype=np.float64)
    tipper = Tipper(tipper=t, tipper_error=err, frequency=freq)
    return SimpleNamespace(station=station, Tipper=tipper)


# ---------------------------------------------------------------------------
# Test 1: clean 2-D, no Tipper anomaly -> low_risk
# ---------------------------------------------------------------------------


def test_clean_2d_low_risk():
    """Clean 2-D site: |T| ≈ 0.05 across all bands, no other
    diagnostics supplied (so they're skipped, not flagged) →
    low_risk."""
    mt = _make_mt(tipper_amplitude=0.05)
    # Provide a flat C across 4 bands to give the freq-dep
    # diagnostic data to operate on (so the overall flag is not
    # 'indeterminate' for missing input).
    c_per_band = np.tile(np.eye(2)[None, :, :], (4, 1, 1)).astype(
        np.float64
    )
    # And a cross-method input where everyone agrees.
    cross_method = {
        "strikes_deg": {"gb": 30.0, "mj": 30.0, "gj": 30.5},
    }
    flag = compute_magnetic_distortion_flag(
        mt,
        c_per_band=c_per_band,
        cross_method_result=cross_method,
    )
    assert isinstance(flag, MagneticDistortionFlag)
    assert flag.overall_flag == "low_risk", (
        f"expected low_risk; got {flag.overall_flag!r}; "
        f"contributing={flag.contributing_factors}"
    )


# ---------------------------------------------------------------------------
# Test 2: strong Tipper |T|=0.6 -> high_risk via override
# ---------------------------------------------------------------------------


def test_strong_tipper_high_risk():
    """Peak |T| = 0.6 exceeds the strong-tipper override (0.5);
    flag must be ``high_risk`` even with no other diagnostics
    flagging."""
    n = 8
    profile = np.full(n, 0.05)
    profile[n // 2] = 0.6  # one period with anomalously high |T|
    mt = _make_mt(tipper_amplitude=profile)
    flag = compute_magnetic_distortion_flag(mt)
    assert flag.overall_flag == "high_risk", (
        f"expected high_risk via strong-tipper override; got "
        f"{flag.overall_flag!r}"
    )
    assert flag.tipper_diagnostic["strong_override"] is True


# ---------------------------------------------------------------------------
# Test 3: frequency-dependent C -> moderate_risk
# ---------------------------------------------------------------------------


def test_frequency_dependent_c_moderate_risk():
    """C tensor varies strongly across bands; tipper is clean →
    moderate_risk."""
    mt = _make_mt(tipper_amplitude=0.05)
    # Per-band C with twist increasing linearly across 4 bands;
    # use the GB tan form so |C| varies noticeably.
    c_per_band = np.empty((4, 2, 2))
    twist_per_band = np.radians(np.array([0.0, 10.0, 20.0, 30.0]))
    for k, t_rad in enumerate(twist_per_band):
        h = np.tan(t_rad)
        T_mat = np.array([[1.0, -h], [h, 1.0]])
        S_mat = np.array([[1.0, 0.1], [0.1, 1.0]])
        c_per_band[k] = T_mat @ S_mat
    flag = compute_magnetic_distortion_flag(
        mt, c_per_band=c_per_band
    )
    assert flag.overall_flag == "moderate_risk", (
        f"expected moderate_risk via frequency-dependence; got "
        f"{flag.overall_flag!r}; contributing="
        f"{flag.contributing_factors}"
    )


# ---------------------------------------------------------------------------
# Test 4: cross-method disagreement -> moderate_risk
# ---------------------------------------------------------------------------


def test_method_inconsistency_moderate_risk():
    """Two methods disagree on strike by 25° (well above the 10°
    threshold) → moderate_risk."""
    mt = _make_mt(tipper_amplitude=0.05)
    cross_method = {
        "strikes_deg": {"gb": 30.0, "gj": 55.0},
    }
    flag = compute_magnetic_distortion_flag(
        mt, cross_method_result=cross_method
    )
    assert flag.overall_flag == "moderate_risk", (
        f"expected moderate_risk via method inconsistency; got "
        f"{flag.overall_flag!r}; contributing="
        f"{flag.contributing_factors}"
    )


# ---------------------------------------------------------------------------
# Test 5: two diagnostics flag -> high_risk
# ---------------------------------------------------------------------------


def test_two_diagnostics_high_risk():
    """Flag (a) Tipper amplitude > 0.3 across the band, AND
    (b) frequency-dependent C, two flags → high_risk."""
    n = 8
    # Tipper that's flagged but does NOT exceed the strong-override
    # (so the flag combination logic, not the override, gives
    # high_risk).
    profile = np.full(n, 0.4)
    mt = _make_mt(tipper_amplitude=profile)
    c_per_band = np.empty((4, 2, 2))
    twist_per_band = np.radians(np.array([0.0, 10.0, 20.0, 30.0]))
    for k, t_rad in enumerate(twist_per_band):
        h = np.tan(t_rad)
        T_mat = np.array([[1.0, -h], [h, 1.0]])
        S_mat = np.array([[1.0, 0.1], [0.1, 1.0]])
        c_per_band[k] = T_mat @ S_mat
    flag = compute_magnetic_distortion_flag(
        mt, c_per_band=c_per_band
    )
    assert flag.overall_flag == "high_risk", (
        f"expected high_risk via 2+ diagnostics; got "
        f"{flag.overall_flag!r}; contributing="
        f"{flag.contributing_factors}"
    )


# ---------------------------------------------------------------------------
# Test 6: insufficient data -> indeterminate
# ---------------------------------------------------------------------------


def test_insufficient_data_indeterminate():
    """No Tipper, no C, no cross-method input → indeterminate
    (NOT low_risk — distinct case)."""
    mt = _make_mt(tipper_amplitude=None)  # Tipper attribute is None
    flag = compute_magnetic_distortion_flag(mt)
    assert flag.overall_flag == "indeterminate", (
        f"expected indeterminate; got {flag.overall_flag!r}; "
        f"contributing={flag.contributing_factors}"
    )


# ---------------------------------------------------------------------------
# Sub-diagnostic unit tests (for thresholds and edge cases)
# ---------------------------------------------------------------------------


class TestTipperDiagnostic:
    def test_no_tipper_attribute(self):
        result = tipper_diagnostic(SimpleNamespace(Tipper=None))
        assert result["available"] is False
        assert result["flagged"] is False

    def test_amplitude_above_threshold_flags(self):
        mt = _make_mt(tipper_amplitude=0.4)
        result = tipper_diagnostic(mt)
        assert result["flagged"] is True

    def test_strong_override_at_0p5(self):
        mt = _make_mt(
            tipper_amplitude=np.array([0.05, 0.05, 0.5, 0.05]),
            n_freqs=4,
        )
        result = tipper_diagnostic(mt)
        assert result["strong_override"] is True


class TestFrequencyDependenceDiagnostic:
    def test_too_few_bands_unavailable(self):
        c = np.tile(np.eye(2)[None, :, :], (2, 1, 1))
        result = frequency_dependence_diagnostic(c)
        assert result["available"] is False

    def test_constant_c_not_flagged(self):
        c = np.tile(np.eye(2)[None, :, :], (5, 1, 1))
        result = frequency_dependence_diagnostic(c)
        assert result["available"] is True
        assert result["flagged"] is False

    def test_dominant_period_reported(self):
        c = np.tile(np.eye(2)[None, :, :], (5, 1, 1))
        c[2, 0, 1] = 5.0  # one band has anomalous off-diagonal
        bp = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
        result = frequency_dependence_diagnostic(c, band_periods=bp)
        assert result["flagged"] is True
        assert result["dominant_period"] == pytest.approx(100.0)


class TestMethodInconsistencyDiagnostic:
    def test_single_method_unavailable(self):
        result = method_inconsistency_diagnostic(
            {"strikes_deg": {"gb": 30.0}}
        )
        assert result["available"] is False

    def test_strike_agreement_not_flagged(self):
        result = method_inconsistency_diagnostic(
            {"strikes_deg": {"gb": 30.0, "mj": 31.0}}
        )
        assert result["flagged"] is False

    def test_z_te_disagreement_flags(self):
        result = method_inconsistency_diagnostic(
            {
                "strikes_deg": {"gb": 30.0, "mj": 30.0},
                "z_te_per_period": {
                    "gb": np.array([1.0, 1.0, 1.0]),
                    "mj": np.array([1.5, 1.5, 1.5]),
                },
            }
        )
        assert result["flagged"] is True
