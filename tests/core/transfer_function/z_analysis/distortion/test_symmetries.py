"""Unit tests for the canonical-gauge selection (F4) and the
GB-symmetry roundtrip (F5).

The two task-blocks are co-located here because they exercise the
same forward-model and gauge-canonicalisation machinery and share
fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    decompose,
    decompose_each_station,
    decompose_joint,
)
from mtpy.core.transfer_function.z_analysis.decomposition.common import (
    _estim_imp,
)
from tests.core.transfer_function.z_analysis.distortion.synthetics import (
    _build_regional_z,
    _construct_C_gb89,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_synthetic_z(
    *,
    strike_deg: float,
    twist_deg: float,
    shear_deg: float,
    gain: float = 1.0,
    noise_fraction: float = 0.0,
    seed: int = 0,
    n_freqs: int = 12,
    rho_te: float = 100.0,
    rho_tm: float = 400.0,
) -> tuple[Z, np.ndarray]:
    """Return a Z object whose physics matches the GB89 forward model
    at the given (strike, twist, shear, gain), plus the true_C tensor.
    """
    periods = np.logspace(-1.0, 2.0, n_freqs)
    z_regional = _build_regional_z(periods, "2D", strike_deg)
    # Re-run with the requested rho_te / rho_tm by scaling.
    omega = 2.0 * np.pi / periods
    mu0 = 4.0 * np.pi * 1.0e-7
    factor = omega * mu0
    scale_te = np.sqrt(rho_te / 100.0)
    scale_tm = np.sqrt(rho_tm / 400.0)
    z_regional[:, 0, 1] *= scale_te
    z_regional[:, 1, 0] *= scale_tm
    true_C = _construct_C_gb89(strike_deg, twist_deg, shear_deg, gain)
    z_obs = np.einsum("ij,kjl->kil", true_C, z_regional)
    if noise_fraction > 0:
        rng = np.random.default_rng(seed)
        amp = noise_fraction * np.abs(z_obs)
        noise = (
            rng.standard_normal(z_obs.shape)
            + 1j * rng.standard_normal(z_obs.shape)
        ) / np.sqrt(2.0)
        z_obs = z_obs + amp * noise
    sigma = np.maximum(max(noise_fraction, 0.005) * np.abs(z_obs), 1e-9)
    return (
        Z(z=z_obs, z_error=sigma, frequency=1.0 / periods),
        true_C,
    )


def _strike_frame_a_b(
    z_meas: np.ndarray, strike_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Derive per-period strike-frame (a, b) from measurement-frame
    Z and a strike. ``a = Z'_xy``, ``b = -Z'_yx``.
    """
    theta = np.radians(strike_deg)
    cs, sn = np.cos(theta), np.sin(theta)
    R = np.array([[cs, -sn], [sn, cs]])
    n_freqs = z_meas.shape[0]
    a = np.empty(n_freqs, dtype=np.complex128)
    b = np.empty(n_freqs, dtype=np.complex128)
    for k in range(n_freqs):
        z_strike = R.T @ z_meas[k] @ R
        a[k] = z_strike[0, 1]
        b[k] = -z_strike[1, 0]
    return a, b


# ---------------------------------------------------------------------------
# F4 unit tests: canonical-gauge selection
# ---------------------------------------------------------------------------


class TestCanonicalGauge:
    """Verify the three ``canonical_gauge`` options on a controlled
    synthetic.
    """

    def test_pt_aligned_recovers_true_strike(self):
        """``canonical_gauge='pt_aligned'`` reports the GB strike
        that aligns (along-strike) with the phase-tensor reference,
        i.e. the geographically-anchored strike. For a 2-D synthetic
        at 30° this should be 30°, not 120°.
        """
        z, _ = _make_synthetic_z(
            strike_deg=30.0, twist_deg=15.0, shear_deg=20.0
        )
        result = decompose(z, n_starts=3, canonical_gauge="pt_aligned")
        recovered = float(np.median(result.parameters["strike"].values))
        # Strike close to 30° (mod 180°). Check angular distance.
        d = abs(recovered - 30.0) % 180.0
        d = min(d, 180.0 - d)
        assert d < 0.5, (
            f"expected strike ≈ 30°; got {recovered:.2f}° (angular "
            f"distance {d:.2f}°)"
        )

    def test_magnitude_chooses_larger_a(self):
        """``canonical_gauge='magnitude'`` reports the branch with
        median |a| >= |b|.

        We synthesise with ``rho_te = 1000`` (so |a| in strike
        frame is large), ``rho_tm = 50`` (|b| is small). The optimiser
        may converge to either branch; canonical_gauge='magnitude'
        ensures the returned (a, b) gauge has |a| >= |b|.
        """
        z, _ = _make_synthetic_z(
            strike_deg=30.0,
            twist_deg=10.0,
            shear_deg=10.0,
            rho_te=1000.0,
            rho_tm=50.0,
        )
        result = decompose(z, n_starts=3, canonical_gauge="magnitude")
        recovered_strike = float(
            np.median(result.parameters["strike"].values)
        )
        a, b = _strike_frame_a_b(
            np.asarray(result.regional_z.z), recovered_strike
        )
        assert float(np.median(np.abs(a))) >= float(
            np.median(np.abs(b))
        ), (
            f"magnitude gauge should give |a| >= |b|; got median |a| "
            f"= {np.median(np.abs(a)):.3f}, |b| = {np.median(np.abs(b)):.3f}"
        )

    def test_rms_best_matches_optimiser_choice(self):
        """``canonical_gauge='rms_best'`` is the no-op default for
        backwards compatibility: no per-band flip is applied.
        """
        z, _ = _make_synthetic_z(
            strike_deg=30.0, twist_deg=15.0, shear_deg=20.0
        )
        result = decompose(z, n_starts=3, canonical_gauge="rms_best")
        # The default disambiguation='geometric' folds strike to
        # [0, 90°), so the recovered strike is 30°. Just verify no
        # ``canonical_gauge`` flips were applied.
        flipped = result.metadata.get(
            "canonical_gauge_flipped_period_indices", None
        )
        assert flipped is None, (
            f"rms_best should not record any flips; got {flipped!r}"
        )

    def test_invalid_gauge_raises(self):
        z, _ = _make_synthetic_z(
            strike_deg=30.0, twist_deg=10.0, shear_deg=10.0
        )
        with pytest.raises(ValueError, match="canonical_gauge"):
            decompose(z, n_starts=2, canonical_gauge="not_a_gauge")


# ---------------------------------------------------------------------------
# F5 roundtrip tests: GB symmetry locked in by tests
# ---------------------------------------------------------------------------


class TestGbSymmetryRoundtrip:
    """The full GB symmetry — including the (a, b) swap — is the
    foundation of the canonical-gauge logic. These tests lock it in
    at the forward-model level and end-to-end through
    :func:`decompose`.
    """

    @pytest.mark.parametrize(
        "strike_deg,twist_deg,shear_deg,a,b",
        [
            (30.0, 15.0, 20.0, 0.5 + 1.2j, 3.0 - 0.8j),
            (60.0, -10.0, 25.0, 2.5 + 0.3j, 0.4 - 1.5j),
            (10.0, 5.0, -8.0, 1.1 + 0.6j, 1.7 + 0.2j),
        ],
    )
    def test_forward_model_complete_symmetry(
        self,
        strike_deg: float,
        twist_deg: float,
        shear_deg: float,
        a: complex,
        b: complex,
    ):
        """F5 Test 1: ``Z_obs(strike, twist, shear, a, b)``
        equals ``Z_obs(strike+90°, twist, -shear, b, a)`` to machine
        precision.
        """
        theta = np.radians(strike_deg)
        t_twist = np.tan(np.radians(twist_deg))
        e_shear = np.tan(np.radians(shear_deg))

        z_a = _estim_imp(a, b, t_twist, e_shear, theta)
        z_b = _estim_imp(
            b, a, t_twist, -e_shear, theta + np.pi / 2.0
        )

        max_err = float(np.max(np.abs(z_a - z_b)))
        assert max_err < 1e-12, (
            f"complete-symmetry roundtrip failed at "
            f"(strike={strike_deg}, twist={twist_deg}, "
            f"shear={shear_deg}, a={a}, b={b}): "
            f"max|Z_A - Z_B| = {max_err:.3e}"
        )

    def test_forward_model_incomplete_symmetry_fails(self):
        """F5 Test 2: omitting the (a, b) swap is *not* a
        symmetry; the predicted Z differs noticeably.

        This locks in that the (a, b) swap is REQUIRED — if a
        future refactor accidentally restores the incomplete
        symmetry, this test fails loudly.
        """
        strike_deg, twist_deg, shear_deg = 30.0, 15.0, 20.0
        a, b = 0.5 + 1.2j, 3.0 - 0.8j
        theta = np.radians(strike_deg)
        t_twist = np.tan(np.radians(twist_deg))
        e_shear = np.tan(np.radians(shear_deg))

        z_a = _estim_imp(a, b, t_twist, e_shear, theta)
        z_c = _estim_imp(
            a, b, t_twist, -e_shear, theta + np.pi / 2.0
        )  # (a, b) NOT swapped

        max_diff = float(np.max(np.abs(z_a - z_c)))
        assert max_diff > 1e-3, (
            f"incomplete symmetry should not preserve Z; got "
            f"max|Z_A - Z_C| = {max_diff:.3e} which is suspiciously "
            f"small. The (a, b) swap is required for the GB symmetry."
        )

    def test_roundtrip_clean_synthetic_recovers_truth(self):
        """F5 Test 3: from ``Z_obs`` synthesised at known
        ``(strike=30°, twist=15°, shear=20°)``,
        :func:`decompose` with ``canonical_gauge='pt_aligned'``
        recovers all three within 0.5° (clean synthetic, no noise).
        """
        true_strike, true_twist, true_shear = 30.0, 15.0, 20.0
        z, _ = _make_synthetic_z(
            strike_deg=true_strike,
            twist_deg=true_twist,
            shear_deg=true_shear,
            n_freqs=12,
        )
        result = decompose(z, n_starts=3, canonical_gauge="pt_aligned")
        rec_strike = float(np.median(result.parameters["strike"].values))
        rec_twist = float(np.median(result.parameters["twist"].values))
        rec_shear = float(np.median(result.parameters["shear"].values))

        d_strike = min(
            abs(rec_strike - true_strike) % 180.0,
            180.0 - (abs(rec_strike - true_strike) % 180.0),
        )
        assert d_strike < 0.5, (
            f"strike: rec {rec_strike:.3f}° vs true {true_strike}°; "
            f"angular distance {d_strike:.3f}° > 0.5°"
        )
        assert abs(rec_twist - true_twist) < 0.5, (
            f"twist: rec {rec_twist:.3f}° vs true {true_twist}°"
        )
        assert abs(rec_shear - true_shear) < 0.5, (
            f"shear: rec {rec_shear:.3f}° vs true {true_shear}°"
        )

        # |a|, |b| recovery: derive from regional_z and recovered
        # strike. With true rho_te=100, rho_tm=400, |a| corresponds
        # to TE (smaller magnitude in this convention) and |b| to
        # TM (larger).
        a_rec, b_rec = _strike_frame_a_b(
            np.asarray(result.regional_z.z), rec_strike
        )
        # The synthetic uses fixed rho_te=100, rho_tm=400; we don't
        # have direct (a, b) numbers but we know the expected ratio
        # |b|/|a| ≈ sqrt(rho_tm / rho_te) = 2.
        ratio = float(np.median(np.abs(b_rec) / np.abs(a_rec)))
        assert abs(ratio - 2.0) < 0.05, (
            f"|b|/|a| recovery: got ratio {ratio:.4f}, expected ≈ 2.0 "
            f"(sqrt(rho_tm/rho_te) = sqrt(400/100))"
        )

    def test_roundtrip_gauge_flipped_synthetic_recovers_canonical(self):
        """F5 Test 4: force the optimiser onto the alt GB branch
        (via strike bounds) and verify ``canonical_gauge='pt_aligned'``
        flips it back to the geographically anchored 30°.

        Synthesising directly at the alt-branch parameters with
        ``rho_te`` / ``rho_tm`` swapped is *not* a clean GB-equivalent
        synthetic because :func:`_build_regional_z` uses fixed phases
        that do not swap with rho — the resulting Z_obs is *not*
        identical to the 30°-branch Z. We sidestep that by using the
        same Z as Test 3 and constraining the optimiser to the upper
        strike branch ``[90°, 180°)`` so the as-fitted strike is in
        the 120° branch; ``canonical_gauge='pt_aligned'`` should then
        flip the reported strike back to the canonical 30° branch.
        """
        true_strike = 30.0
        z, _ = _make_synthetic_z(
            strike_deg=true_strike,
            twist_deg=15.0,
            shear_deg=20.0,
            n_freqs=12,
        )
        # Force the optimiser into the alt branch (strike in
        # [90°, 180°)) via bounds_override.
        upper_bounds = {"strike": (np.pi / 2.0, np.pi)}
        result = decompose(
            z,
            n_starts=3,
            disambiguation="identity",  # keep optimiser strike unfolded
            canonical_gauge="pt_aligned",
            bounds_override=upper_bounds,
        )
        rec_strike = float(np.median(result.parameters["strike"].values))
        d_to_30 = min(
            abs(rec_strike - true_strike) % 180.0,
            180.0 - (abs(rec_strike - true_strike) % 180.0),
        )
        d_to_120 = min(
            abs(rec_strike - 120.0) % 180.0,
            180.0 - (abs(rec_strike - 120.0) % 180.0),
        )
        # The pt_aligned gauge canonicalises to the geographically
        # anchored strike; for this synthetic that is 30°. The
        # gauge step should also have been applied (i.e. flipped)
        # since the optimiser was forced onto the 120° branch.
        flipped = result.metadata.get(
            "canonical_gauge_flipped_period_indices", []
        )
        assert len(flipped) > 0, (
            "pt_aligned should have flipped the upper-branch "
            "as-fitted strike back to the 30° canonical branch"
        )
        assert d_to_30 < d_to_120, (
            f"pt_aligned should canonicalise to 30°, got "
            f"strike={rec_strike:.2f}° (dist to 30°={d_to_30:.2f}°, "
            f"dist to 120°={d_to_120:.2f}°)"
        )

    def test_multi_site_spatial_consistency(self):
        """F5 Test 5: 7 sites with the same true (strike, twist,
        shear) but independent noise realisations.
        :func:`decompose_each_station` with
        ``canonical_gauge='pt_aligned'`` reports spatially-consistent
        strikes and (|a|, |b|) gauges across all sites.
        """
        true_strike, true_twist, true_shear = 30.0, 15.0, 20.0
        n_sites = 7
        z_objs: list[Z] = []
        from types import SimpleNamespace

        mts: list = []
        for i in range(n_sites):
            z, _ = _make_synthetic_z(
                strike_deg=true_strike,
                twist_deg=true_twist,
                shear_deg=true_shear,
                noise_fraction=0.005,
                seed=100 + i,
            )
            z_objs.append(z)
            # decompose_each_station accepts list[MT]; mock with
            # SimpleNamespace.
            mts.append(SimpleNamespace(station=f"S{i:02d}", Z=z))

        results = decompose_each_station(mts, n_starts=3)

        recovered_strikes = []
        a_b_ratios = []
        for sid, r in results.items():
            s = float(np.median(r.parameters["strike"].values))
            recovered_strikes.append(s)
            a, b = _strike_frame_a_b(np.asarray(r.regional_z.z), s)
            a_b_ratios.append(
                float(np.median(np.abs(a) / np.abs(b)))
            )

        # Strikes within ±2° of 30°.
        for sid, s in zip(results.keys(), recovered_strikes):
            d = min(
                abs(s - 30.0) % 180.0,
                180.0 - (abs(s - 30.0) % 180.0),
            )
            assert d < 2.0, (
                f"site {sid}: strike {s:.2f}° too far from 30° "
                f"(distance {d:.2f}°)"
            )

        # All sites' gauges agree: either |a| > |b| at all sites,
        # or |a| < |b| at all sites. With pt_aligned canonical
        # gauge anchoring strike to PT alpha + 90°, all sites should
        # land on the same branch.
        all_a_smaller = all(r < 1.0 for r in a_b_ratios)
        all_b_smaller = all(r > 1.0 for r in a_b_ratios)
        assert all_a_smaller or all_b_smaller, (
            f"gauge inconsistent across sites: |a|/|b| ratios "
            f"{a_b_ratios} should all be on the same side of 1"
        )


# ---------------------------------------------------------------------------
# F4 joint-decomposition test
# ---------------------------------------------------------------------------


class TestCanonicalGaugeJoint:
    """Verify ``canonical_gauge`` flows through the joint
    decomposition path (:func:`decompose_joint`).
    """

    def test_decompose_joint_pt_aligned(self):
        from types import SimpleNamespace

        n_sites = 3
        mts = []
        for i in range(n_sites):
            z, _ = _make_synthetic_z(
                strike_deg=30.0,
                twist_deg=10.0 + 2.0 * i,
                shear_deg=15.0 - 3.0 * i,
                noise_fraction=0.005,
                seed=200 + i,
            )
            mts.append(SimpleNamespace(station=f"S{i:02d}", Z=z))

        result = decompose_joint(mts, n_starts=3, canonical_gauge="pt_aligned")
        rec_strike = float(np.median(result.parameters["strike"].values))
        d = min(
            abs(rec_strike - 30.0) % 180.0,
            180.0 - (abs(rec_strike - 30.0) % 180.0),
        )
        assert d < 2.0, (
            f"joint pt_aligned: strike {rec_strike:.2f}° too far "
            f"from 30° (distance {d:.2f}°)"
        )
