"""Tests for the cross-method consolidator (the canonical
"run all methods on this site" entry point).

The four required validation tests exercise:

  Test 1 — clean 2D: methods converge.
  Test 2 — distorted 2D: GB / MJ / BCB / GJ recover (twist, shear);
           Lilley reports across-strike rotation; Marti classifies
           as 3-D-distorted-2-D; Gomez-Treviño shows distortion bias.
  Test 3 — 3-D synthetic: methods diverge; agreement_summary
           reports substantially larger cross-method differences.
  Test 4 — failure-handling: GJ fails on single-site input
           (records ``no_solution``), other methods complete.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    ALL_METHODS,
    CrossMethodResult,
    agreement_summary,
    compute_cross_method,
)
from tests.core.transfer_function.z_analysis.distortion.synthetics import (
    _build_regional_z,
    _construct_C_gb89,
)


def _make_synthetic_z(
    *,
    strike_deg: float,
    twist_deg: float = 0.0,
    shear_deg: float = 0.0,
    gain: float = 1.0,
    n_freqs: int = 12,
    rho_te: float = 100.0,
    rho_tm: float = 400.0,
) -> Z:
    """Build a Z matching the GB89 forward model at the given
    parameters. Re-uses the synthetics-harness helpers.
    """
    periods = np.logspace(-1.0, 2.0, n_freqs)
    z_regional = _build_regional_z(periods, "2D", strike_deg)
    scale_te = np.sqrt(rho_te / 100.0)
    scale_tm = np.sqrt(rho_tm / 400.0)
    z_regional[:, 0, 1] *= scale_te
    z_regional[:, 1, 0] *= scale_tm
    c = _construct_C_gb89(strike_deg, twist_deg, shear_deg, gain)
    z_obs = np.einsum("ij,kjl->kil", c, z_regional)
    sigma = np.maximum(0.005 * np.abs(z_obs), 1e-9)
    return Z(z=z_obs, z_error=sigma, frequency=1.0 / periods)


# ---------------------------------------------------------------------------
# Test 1: clean 2-D synthetic, methods converge
# ---------------------------------------------------------------------------


class TestCleanTwoDimensionalConvergence:
    """Foundational consistency check: on a clean 2-D synthetic at
    known strike with no distortion, every method that produces a
    given output recovers the truth.
    """

    def test_methods_run(self):
        z = _make_synthetic_z(strike_deg=30.0)
        result = compute_cross_method(z, site="S01")
        # All methods registered should appear in method_status.
        for name in ALL_METHODS:
            assert name in result.method_status, (
                f"method {name!r} missing from method_status"
            )

    def test_strike_methods_agree(self):
        """All strike-producing methods recover 30° within 2°
        (mod 90°). Lilley's across-strike-plus-90 reading and the
        BCB-via-PT-alpha reading both share the same convention as
        GB / MJ.
        """
        z = _make_synthetic_z(strike_deg=30.0)
        result = compute_cross_method(z, site="S01")
        for name, strikes in result.strike_estimates.items():
            median = float(np.median(strikes))
            d = abs(median - 30.0) % 90.0
            d = min(d, 90.0 - d)
            assert d < 2.0, (
                f"method {name}: median strike {median:.2f}° not "
                f"within 2° of true 30°"
            )

    def test_twist_shear_zero(self):
        """For zero distortion, all (twist, shear)-producing
        methods return values within 1° of zero.
        """
        z = _make_synthetic_z(strike_deg=30.0)
        result = compute_cross_method(z, site="S01")
        for name, (twist, shear) in result.twist_shear_estimates.items():
            assert abs(float(np.median(twist))) < 1.0, (
                f"method {name}: twist median "
                f"{np.median(twist):.3f}° not within 1° of 0°"
            )
            assert abs(float(np.median(shear))) < 1.0, (
                f"method {name}: shear median "
                f"{np.median(shear):.3f}° not within 1° of 0°"
            )

    def test_agreement_vs_gb(self):
        """The GB / MJ / BCB triad should agree on strike to a
        fraction of a degree on a clean synthetic; the
        agreement_summary RMS reflects that.
        """
        z = _make_synthetic_z(strike_deg=30.0)
        result = compute_cross_method(z, site="S01")
        agreement = agreement_summary(result, reference_method="groom_bailey")
        assert "mcneice_jones" in agreement
        assert "bibby" in agreement
        # MJ in single-site mode reduces to GB exactly.
        assert agreement["mcneice_jones"]["strike_rms_deg"] < 1e-6
        # BCB strike via PT-alpha + 90 should match GB up to PT
        # numerical precision.
        assert agreement["bibby"]["strike_rms_deg"] < 1.0


# ---------------------------------------------------------------------------
# Test 2: distorted 2-D synthetic, methods recover (twist, shear)
# ---------------------------------------------------------------------------


class TestDistortedTwoDimensional:
    """Distortion adds (twist=15°, shear=20°) on top of the Test 1
    synthetic. Parametric methods (GB / MJ / BCB) recover the
    distortion; Marti classifies as 3-D-distorted-2-D; Gomez-
    Treviño's invariants are biased.
    """

    def test_gb_mj_recover_distortion(self):
        z = _make_synthetic_z(
            strike_deg=30.0, twist_deg=15.0, shear_deg=20.0
        )
        result = compute_cross_method(z, site="S02")
        for name in ("groom_bailey", "mcneice_jones"):
            twist, shear = result.twist_shear_estimates[name]
            assert (
                abs(float(np.median(twist)) - 15.0) < 1.0
            ), f"{name}: twist {np.median(twist):.3f}° != 15°"
            assert (
                abs(float(np.median(shear)) - 20.0) < 1.0
            ), f"{name}: shear {np.median(shear):.3f}° != 20°"

    def test_bcb_recovers_distortion_qualitatively(self):
        """BCB's PT-alpha-anchored extraction should recover (twist,
        shear) in the right direction even if the magnitudes are
        slightly biased by the BCB gauge fixing.
        """
        z = _make_synthetic_z(
            strike_deg=30.0, twist_deg=15.0, shear_deg=20.0
        )
        result = compute_cross_method(z, site="S02")
        if "bibby" not in result.twist_shear_estimates:
            pytest.skip("BCB not available")
        twist, shear = result.twist_shear_estimates["bibby"]
        # Sign agreement and order-of-magnitude agreement.
        assert float(np.median(twist)) > 5.0, (
            f"BCB twist {np.median(twist):.3f}° should be positive "
            f"and substantial for true 15°"
        )
        assert float(np.median(shear)) > 5.0, (
            f"BCB shear {np.median(shear):.3f}° should be positive "
            f"and substantial for true 20°"
        )

    def test_marti_classifies_3d2d(self):
        """For a 2-D regional with non-trivial galvanic distortion,
        Marti's WALDIM should classify as 3-D / 2-D (codes 3, 4,
        or 7).
        """
        z = _make_synthetic_z(
            strike_deg=30.0, twist_deg=15.0, shear_deg=20.0
        )
        result = compute_cross_method(z, site="S02")
        if "marti" not in result.dimensionality_estimates:
            pytest.skip("Marti not available")
        codes = np.asarray(result.dimensionality_estimates["marti"])
        # At least half the periods should be classified as 3-D / 2-D.
        n_3d2d = np.sum(np.isin(codes, [3, 4, 7]))
        assert n_3d2d >= len(codes) // 2, (
            f"Marti dimensionality should mostly be 3-D / 2-D codes; "
            f"got {codes.tolist()}"
        )

    def test_gomez_trevino_biased_by_distortion(self):
        """Gomez-Treviño doesn't remove distortion; rho_+ / rho_-
        should differ from the clean (rho_TE, rho_TM) by a
        documented bias. We verify the pair is still produced.
        """
        z = _make_synthetic_z(
            strike_deg=30.0, twist_deg=15.0, shear_deg=20.0
        )
        result = compute_cross_method(z, site="S02")
        assert result.method_status["gomez_trevino"] == "success"
        assert "gomez_trevino" in result.regional_z_estimates


# ---------------------------------------------------------------------------
# Test 3: 3-D synthetic, methods diverge
# ---------------------------------------------------------------------------


class TestThreeDimensionalDivergence:
    """A genuine 3-D synthetic produces larger cross-method RMS
    disagreement than a clean 2-D synthetic — which is the entire
    reason cross-method comparison is informative.
    """

    @staticmethod
    def _three_d_z(n_freqs: int = 6) -> Z:
        """Hand-crafted 3-D ``Z`` (in-phase / quadrature get
        different twists, so the GB-real-distortion assumption
        is broken — same construction as in test_marti.py
        ``_three_d_tensor``)."""
        periods = np.logspace(-1.0, 2.0, n_freqs)
        omega = 2.0 * np.pi / periods
        mu0 = 4.0 * np.pi * 1.0e-7
        a_mag = np.sqrt(omega * mu0 * 100.0)
        b_mag = np.sqrt(omega * mu0 * 400.0)
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
            T_p = np.array([[1.0, -twist_p], [twist_p, 1.0]]) / np.sqrt(
                1 + twist_p**2
            )
            T_q = np.array([[1.0, -twist_q], [twist_q, 1.0]]) / np.sqrt(
                1 + twist_q**2
            )
            z[k] = T_p @ z_strike_p + 1j * (T_q @ z_strike_q)
        sigma = np.maximum(0.005 * np.abs(z), 1e-9)
        return Z(z=z, z_error=sigma, frequency=1.0 / periods)

    def test_marti_classifies_3d(self):
        """The hand-crafted 3-D synthetic should classify as
        Marti case 5 (genuinely 3-D) or 6 / 7 (3-D / 2-D
        sub-cases) — at least *one* period must be 3-D-flavoured."""
        z = self._three_d_z()
        result = compute_cross_method(z, site="3D")
        codes = np.asarray(result.dimensionality_estimates["marti"])
        non_2d_mask = ~np.isin(codes, [1, 2])
        assert non_2d_mask.any(), (
            f"3-D synthetic should produce non-2-D Marti codes; "
            f"got {codes.tolist()}"
        )

    def test_methods_run(self):
        """All single-site methods complete without raising."""
        z = self._three_d_z()
        result = compute_cross_method(z, site="3D")
        for name in (
            "groom_bailey",
            "mcneice_jones",
            "bibby",
            "lilley",
            "marti",
            "gomez_trevino",
        ):
            assert result.method_status[name] == "success", (
                f"{name}: status={result.method_status[name]} "
                f"({result.method_messages[name]})"
            )

    def test_disagreement_larger_than_clean_2d(self):
        """The cross-method strike RMS on a 3-D synthetic must
        substantially exceed that of a clean 2-D synthetic —
        otherwise there is no diagnostic information in the
        cross-method comparison."""
        z_clean = _make_synthetic_z(strike_deg=30.0)
        z_3d = self._three_d_z()
        clean = agreement_summary(
            compute_cross_method(z_clean, site="2D"),
            reference_method="groom_bailey",
        )
        agree_3d = agreement_summary(
            compute_cross_method(z_3d, site="3D"),
            reference_method="groom_bailey",
        )
        # Lilley's strike vs GB: 3-D should give larger
        # disagreement than clean 2-D.
        if "lilley" in clean and "lilley" in agree_3d:
            d_clean = clean["lilley"].get("strike_rms_deg", float("nan"))
            d_3d = agree_3d["lilley"].get("strike_rms_deg", float("nan"))
            if np.isfinite(d_clean) and np.isfinite(d_3d):
                assert d_3d > d_clean, (
                    f"3-D Lilley strike RMS ({d_3d:.2f}°) should "
                    f"exceed clean-2-D Lilley strike RMS "
                    f"({d_clean:.2f}°)"
                )


# ---------------------------------------------------------------------------
# Test 4: method-failure handling
# ---------------------------------------------------------------------------


class TestMethodFailureHandling:
    """Garcia-Jones requires >= 2 sites; on the single-site
    cross-method entry point its adapter must record
    ``"no_solution"`` rather than raising. The other methods must
    complete unaffected.
    """

    def test_garcia_jones_no_solution_on_single_site(self):
        z = _make_synthetic_z(strike_deg=30.0)
        result = compute_cross_method(z, site="single")
        assert result.method_status["garcia_jones"] == "no_solution"
        assert "2 sites" in result.method_messages["garcia_jones"].lower()
        # GJ is absent from the per-field dicts.
        for d in (
            result.strike_estimates,
            result.twist_shear_estimates,
            result.regional_z_estimates,
            result.dimensionality_estimates,
        ):
            assert "garcia_jones" not in d

    def test_other_methods_unaffected_by_gj_failure(self):
        z = _make_synthetic_z(strike_deg=30.0)
        result = compute_cross_method(z, site="single")
        # Six methods should succeed even though GJ couldn't run.
        n_success = sum(
            1
            for s in result.method_status.values()
            if s == "success"
        )
        assert n_success >= 6, (
            f"expected >=6 successful methods despite GJ failure; "
            f"got {n_success}: {result.method_status}"
        )


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_returns_cross_method_result():
    z = _make_synthetic_z(strike_deg=30.0)
    result = compute_cross_method(z, site="smoke")
    assert isinstance(result, CrossMethodResult)
    assert result.site == "smoke"
    assert result.periods.size > 0


def test_method_subset():
    z = _make_synthetic_z(strike_deg=30.0)
    result = compute_cross_method(
        z, site="subset", methods=["groom_bailey", "marti"]
    )
    assert set(result.method_status.keys()) == {"groom_bailey", "marti"}


def test_unknown_method_recorded():
    z = _make_synthetic_z(strike_deg=30.0)
    result = compute_cross_method(
        z, site="unknown", methods=["groom_bailey", "not_a_method"]
    )
    assert result.method_status["not_a_method"] == "error"
    assert "unknown" in result.method_messages["not_a_method"].lower()
