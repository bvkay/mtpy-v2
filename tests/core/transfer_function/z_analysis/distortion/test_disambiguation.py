"""Tests for the GB disambiguation strategy framework.

Covers the four named fold strategies (``geometric``, ``identity``,
``pt_aligned``, ``min_shear``) and the dispatcher in
:mod:`mtpy.core.transfer_function.z_analysis.decomposition.symmetries`,
plus their integration into :func:`decompose`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    _estim_imp,
    _geometric_fold,
    _identity_fold,
    _min_shear_fold,
    _pt_aligned_fold,
    _resolve_disambiguation,
    decompose,
)


def _synthetic_z(
    theta_deg: float = 30.0,
    twist_deg: float = 10.0,
    shear_deg: float = 5.0,
    log10_gain: float = 0.0,
    n_freqs: int = 12,
    seed: int = 0,
    rho_a: float = 10.0,
    rho_b: float = 100.0,
    phase_a: float = np.radians(60.0),
    phase_b: float = np.radians(45.0),
) -> Z:
    """Noiseless synthetic ``Z`` with one distortion triple over ~3 decades.

    Defaults produce a clean 2D-regional site where ``rho_a`` is held
    consistently larger than ``rho_b`` so the phase tensor's
    principal axis stays on the same branch across periods. Mirrors
    the helper used by the main decomposition test suite, kept local
    so this file is self-contained.
    """
    rng = np.random.default_rng(seed)
    periods = np.logspace(-1.0, 2.0, n_freqs)
    log10_rho_a = np.full(n_freqs, np.log10(rho_a)) + 0.1 * rng.standard_normal(n_freqs)
    log10_rho_b = np.full(n_freqs, np.log10(rho_b)) + 0.1 * rng.standard_normal(n_freqs)
    phase_a = np.full(n_freqs, phase_a) + 0.05 * rng.standard_normal(n_freqs)
    phase_b = np.full(n_freqs, phase_b) + 0.05 * rng.standard_normal(n_freqs)

    theta = np.radians(theta_deg)
    twist = np.radians(twist_deg)
    shear = np.radians(shear_deg)

    mu0 = 4.0 * np.pi * 1.0e-7
    factor = 2.0 * np.pi * mu0
    rho_a = 10.0**log10_rho_a
    rho_b = 10.0**log10_rho_b
    abs_a = np.sqrt(rho_a * factor / periods)
    abs_b = np.sqrt(rho_b * factor / periods)
    a = abs_a * np.exp(1j * phase_a)
    b = abs_b * np.exp(1j * phase_b)
    gain = 10.0**log10_gain
    t = np.tan(twist)
    e = np.tan(shear)

    z_obs = np.empty((n_freqs, 2, 2), dtype=np.complex128)
    for k in range(n_freqs):
        z_obs[k] = gain * _estim_imp(a[k], b[k], t, e, theta)

    sigma = np.maximum(
        0.01 * np.abs(z_obs),
        0.01 * np.max(np.abs(z_obs), axis=(1, 2), keepdims=True),
    )
    return Z(z=z_obs, z_error=sigma, frequency=1.0 / periods)


def _c_meas_from_params_deg(
    strike_deg: np.ndarray,
    twist_deg: np.ndarray,
    shear_deg: np.ndarray,
    gain: np.ndarray,
) -> np.ndarray:
    """Reconstruct ``C_meas = g * R(strike) T(twist) S(shear) R(strike).T``.

    The C tensor is the GB-symmetry-invariant; differing strategies
    that pick different (strike, shear) branches must reconstruct the
    same C per period.
    """
    s = np.radians(strike_deg)
    t = np.radians(twist_deg)
    e = np.radians(shear_deg)
    n = s.size
    out = np.empty((n, 2, 2), dtype=np.float64)
    for k in range(n):
        if not np.isfinite(s[k]):
            out[k] = np.nan
            continue
        cs, sn = np.cos(s[k]), np.sin(s[k])
        ct, st = np.cos(t[k]), np.sin(t[k])
        ce, se = np.cos(e[k]), np.sin(e[k])
        R = np.array([[cs, -sn], [sn, cs]])
        T = np.array([[ct, -st], [st, ct]])
        S = np.array([[ce, se], [se, ce]])
        out[k] = gain[k] * R @ T @ S @ R.T
    return out


# ---------------------------------------------------------------------------
# Unit tests on the fold callables
# ---------------------------------------------------------------------------


class TestFoldUnits:
    def test_geometric_folds_upper_branch(self):
        s, _, sh = _geometric_fold(np.radians(120.0), 0.0, np.radians(5.0))
        assert np.isclose(np.degrees(s), 30.0, atol=1e-9)
        assert np.isclose(np.degrees(sh), -5.0, atol=1e-9)

    def test_identity_keeps_branch(self):
        s, _, sh = _identity_fold(np.radians(120.0), 0.0, np.radians(5.0))
        assert np.isclose(np.degrees(s), 120.0, atol=1e-9)
        assert np.isclose(np.degrees(sh), 5.0, atol=1e-9)

    def test_min_shear_picks_smaller_magnitude(self):
        # Both branches share |shear|, but our implementation flips
        # only when the rotated branch's |-shear| is *strictly* less.
        # Equality => keep original branch.
        s, _, sh = _min_shear_fold(np.radians(40.0), 0.0, np.radians(5.0))
        assert np.isclose(np.degrees(s), 40.0)
        assert np.isclose(np.degrees(sh), 5.0)

    def test_pt_aligned_chooses_closest_branch(self):
        # Optimiser-frame strike is upper-branch (120 deg). PT
        # reference is at 35 deg → closer to the rotated branch
        # (120 deg + 90 deg mod 180 = 30 deg). Expect strike->30 deg
        # and shear sign flipped.
        s, _, sh = _pt_aligned_fold(
            np.radians(120.0),
            0.0,
            np.radians(5.0),
            pt_strike_rad=np.radians(35.0),
        )
        assert np.isclose(np.degrees(s), 30.0, atol=1e-9)
        assert np.isclose(np.degrees(sh), -5.0, atol=1e-9)

    def test_pt_aligned_keeps_close_branch(self):
        # Original branch already close to PT → no flip.
        s, _, sh = _pt_aligned_fold(
            np.radians(35.0),
            0.0,
            np.radians(5.0),
            pt_strike_rad=np.radians(35.0),
        )
        assert np.isclose(np.degrees(s), 35.0, atol=1e-9)
        assert np.isclose(np.degrees(sh), 5.0, atol=1e-9)

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown disambiguation"):
            _resolve_disambiguation("not_a_strategy")

    def test_resolve_pt_aligned_requires_strike(self):
        with pytest.raises(ValueError, match="pt_strike_rad"):
            _resolve_disambiguation("pt_aligned")


# ---------------------------------------------------------------------------
# decompose() integration tests
# ---------------------------------------------------------------------------


def test_disambiguation_geometric_unchanged():
    """Default disambiguation matches the pre-refactor `canonicalise=True`
    behaviour byte-for-byte on a deterministic synthetic case.

    The previous task's `canonicalise=True` flag is the historical
    geometric fold; the new default `disambiguation='geometric'` must
    be a drop-in equivalent.
    """
    # canonical_gauge="rms_best" so this test exercises only the
    # disambiguation flag — the F4 default ("pt_aligned") would
    # additionally rewrite (strike, shear) per-band based on PT
    # alpha, which is a separate feature with its own tests
    # (TestCanonicalGauge / TestGbSymmetryRoundtrip in
    # test_symmetries.py).
    z = _synthetic_z(theta_deg=120.0)
    res_default = decompose(z, seed=42, canonical_gauge="rms_best")
    res_legacy = decompose(
        z, seed=42, canonicalise=True, canonical_gauge="rms_best"
    )
    res_explicit = decompose(
        z, seed=42, disambiguation="geometric", canonical_gauge="rms_best"
    )

    np.testing.assert_array_equal(
        res_default.parameters["strike"].values,
        res_legacy.parameters["strike"].values,
    )
    np.testing.assert_array_equal(
        res_default.parameters["shear"].values,
        res_legacy.parameters["shear"].values,
    )
    np.testing.assert_array_equal(
        res_default.parameters["strike"].values,
        res_explicit.parameters["strike"].values,
    )
    assert res_default.parameters["strike"].attrs["range"] == "[0, 90)"
    assert res_default.metadata["disambiguation"] == "geometric"


def test_disambiguation_identity_strike_range():
    """`disambiguation='identity'` reports strike in [0, 180).

    Constrain the optimiser to the upper branch so the identity fold
    actually reports values in [90, 180); confirms the [0, 180) range
    is exercised, not just permitted.
    """
    z = _synthetic_z(theta_deg=120.0)
    res = decompose(
        z,
        seed=7,
        disambiguation="identity",
        bounds_override={"strike": (np.pi / 2.0, np.pi)},
    )
    strikes = res.parameters["strike"].values
    finite = strikes[np.isfinite(strikes)]
    assert finite.size > 0
    assert np.all(finite >= 0.0 - 1e-9)
    assert np.all(finite < 180.0 + 1e-9)
    # Forced to upper branch → reported strikes must stay there.
    assert np.all(finite >= 90.0 - 1e-6)
    assert res.parameters["strike"].attrs["range"] == "[0, 180)"
    assert res.metadata["strike_range_degrees"] == "[0, 180)"
    assert res.metadata["disambiguation"] == "identity"


def test_disambiguation_pt_aligned_consistency():
    """`disambiguation='pt_aligned'` selects the GB branch matching
    the input phase-tensor strike.

    Build a 2D-regional site with true regional strike 30°. The
    phase tensor's principal axis ``alpha`` follows mtpy's
    convention of being 90° off the GB strike, so PT alpha
    median ≈ 120° (mod 180). Force the GB optimiser into the
    *lower* branch via bounds (strike in (0, 90°)) and verify that
    ``pt_aligned`` flips the reported strike to the *upper* branch
    (~120°) so it matches the PT principal axis.
    """
    z = _synthetic_z(theta_deg=30.0, twist_deg=8.0, shear_deg=4.0)
    pt_alpha = np.asarray(z.phase_tensor.alpha)
    pt_alpha = pt_alpha[np.isfinite(pt_alpha)]
    pt_median_mod180 = float(np.median(pt_alpha) % 180.0)
    # Sanity: with these inputs the PT principal axis is consistent
    # across periods (clean 2D regional structure) and lands at
    # ~120° (= true strike + 90° under mtpy's convention).
    assert pt_median_mod180 == pytest.approx(120.0, abs=5.0)

    # canonical_gauge="rms_best": this test exercises the
    # disambiguation='pt_aligned' fold; the F4 canonical_gauge
    # would re-flip after the fold, which is tested separately.
    res = decompose(
        z,
        seed=11,
        disambiguation="pt_aligned",
        canonical_gauge="rms_best",
        bounds_override={"strike": (0.0, np.pi / 2.0)},
    )
    strikes = res.parameters["strike"].values
    finite = strikes[np.isfinite(strikes)]
    assert finite.size > 0
    median_strike = float(np.nanmedian(strikes))
    # pt_aligned must pick the branch within angular distance (mod
    # 180) of the PT median.
    diff = abs(median_strike - pt_median_mod180) % 180.0
    diff = min(diff, 180.0 - diff)
    assert diff < 5.0, (
        f"pt_aligned strike {median_strike:.3f} not aligned with "
        f"PT strike {pt_median_mod180:.3f}"
    )
    # The optimiser is bounded to (0, 90°) but the fold flipped to
    # the rotated branch, so the reported strike must be in [90°, 180°).
    assert np.all(finite >= 90.0 - 1e-6)
    assert res.metadata["disambiguation"] == "pt_aligned"


def test_disambiguation_C_invariance():
    """All four strategies reconstruct the same C tensor.

    The GB 90-deg / shear-sign symmetry is exactly that — different
    fold choices pick different (strike, shear) branches, but the
    measurement-frame galvanic distortion tensor
    ``C = g * R T S R.T`` is invariant. If the four strategies all
    decompose the same Z (with the same seed), the reconstructed C
    per period should match across strategies to numerical
    precision.
    """
    z = _synthetic_z(theta_deg=120.0, twist_deg=8.0, shear_deg=4.0)
    bounds = {"strike": (np.pi / 2.0, np.pi)}
    strategies = ("geometric", "identity", "pt_aligned", "min_shear")

    c_per_strategy: dict[str, np.ndarray] = {}
    for name in strategies:
        res = decompose(
            z, seed=99, disambiguation=name, bounds_override=bounds
        )
        c_per_strategy[name] = _c_meas_from_params_deg(
            res.parameters["strike"].values,
            res.parameters["twist"].values,
            res.parameters["shear"].values,
            res.parameters["gain"].values,
        )

    ref = c_per_strategy["geometric"]
    for name in strategies[1:]:
        np.testing.assert_allclose(
            c_per_strategy[name],
            ref,
            atol=1e-9,
            rtol=1e-9,
            err_msg=f"C tensor differs between geometric and {name}",
        )


def test_disambiguation_round_trip():
    """``_identity_fold`` is idempotent: applying it twice equals
    applying it once.

    This is the trivial round-trip; the function only wraps strike
    into [0, pi) and is a no-op on already-wrapped inputs.
    """
    rng = np.random.default_rng(1234)
    for _ in range(20):
        s_in = float(rng.uniform(-3.0 * np.pi, 3.0 * np.pi))
        t_in = float(rng.uniform(-1.0, 1.0))
        sh_in = float(rng.uniform(-1.0, 1.0))

        s1, t1, sh1 = _identity_fold(s_in, t_in, sh_in)
        s2, t2, sh2 = _identity_fold(s1, t1, sh1)
        assert s1 == s2
        assert t1 == t2
        assert sh1 == sh2
        assert 0.0 <= s1 < np.pi
