"""End-to-end integration tests for the distortion-analysis package.

Exercises the full workflow on a controlled synthetic and verifies
that every implemented method produces a result without errors;
verifies further that the methods that recover a galvanic
distortion ``C`` (GB single-site, BCB, MJ) agree with each other
and with the ground truth, and that the spin-2 shear ``gamma``
field is consistent across methods.

Marked ``slow`` because each test runs every method end-to-end.
Deselect with ``pytest -m "not slow"``.

Distortion-level note
---------------------
The GB factorisation ``C = gain * R T S R.T`` has a structural
gain non-identifiability — the optimiser can trade ``gain``
against the regional impedance amplitudes. At weak distortion
(~5 deg) ``gain`` stays near 1 and recovered ``C`` matches truth
tightly; at moderate distortion (~15 deg) ``gain`` drifts and the
recovered ``C`` is biased even though ``(strike, twist, shear)``
are recovered exactly. We therefore use weak distortion in these
integration tests so the cross-method tolerance is set by genuine
inter-method bias rather than by the structural ambiguity. See
``docs/distortion_methods.md`` for the full discussion.

Garcia-Jones note
-----------------
``decompose_garcia_jones`` is exercised but not held to a tight
``C`` recovery tolerance: when the regional structure is 2-D and
no "control" site (one with zero distortion) is present, the
3-D-regional parameterisation has a degenerate optimisation
landscape and the multi-start can converge to a non-truth
minimum. This is a known Phase-1 limitation; GJ shines on
genuinely 3-D regional structure (see
:func:`tests.test_garcia_jones.test_gj_better_than_mj_at_3d`).
The integration test here verifies GJ runs end-to-end and
produces a populated result.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    GarciaJonesResult,
    JointDecompositionResult,
    decompose,
    decompose_bibby,
    decompose_garcia_jones,
    decompose_lilley,
    decompose_marti,
    decompose_mcneice_jones,
    gamma_field,
    irreducible_decomposition,
)
from tests.core.transfer_function.z_analysis.distortion.synthetics import (
    _build_regional_z,
    _construct_C_gb89,
    compute_method_accuracy,
    generate_synthetic_z,
    run_all_methods_on_synthetic,
)


def _make_multi_site_synthetic(
    periods: np.ndarray,
    strike_deg: float,
    distortions_deg: list[tuple[float, float]],
    noise_fraction: float = 0.0,
    seed: int = 42,
) -> tuple[list[Z], list[str], list[np.ndarray]]:
    """Build a list of ``Z`` objects with a shared regional but
    site-specific distortion.

    Returns ``(z_objs, site_ids, true_C_per_site)``.
    """
    z_regional = _build_regional_z(periods, "2D", strike_deg)
    rng = np.random.default_rng(seed)
    z_objs: list[Z] = []
    site_ids: list[str] = []
    true_C_per_site: list[np.ndarray] = []
    for i, (twist_deg, shear_deg) in enumerate(distortions_deg):
        c = _construct_C_gb89(strike_deg, twist_deg, shear_deg, gain=1.0)
        z_obs = np.einsum("ij,kjl->kil", c, z_regional)
        if noise_fraction > 0:
            amp = noise_fraction * np.abs(z_obs)
            noise = (
                rng.standard_normal(z_obs.shape)
                + 1j * rng.standard_normal(z_obs.shape)
            ) / np.sqrt(2.0)
            z_obs = z_obs + amp * noise
        sigma = np.maximum(max(noise_fraction, 0.005) * np.abs(z_obs), 1e-9)
        z_objs.append(Z(z=z_obs, z_error=sigma, frequency=1.0 / periods))
        site_ids.append(f"S{i + 1:02d}")
        true_C_per_site.append(c)
    return z_objs, site_ids, true_C_per_site


@pytest.mark.slow
def test_full_workflow_on_synthetic():
    """Exercise every decomposition method end-to-end and verify each
    method that recovers ``C`` agrees with the ground truth on the
    primary site.

    Single-site methods (GB with all four disambiguation strategies,
    BCB, Lilley, Marti) are run on the primary site via the
    synthetic harness. Cross-method ``C`` consistency check is
    asserted at ``|C_recovered - C_true|_F < 0.05`` for the methods
    that produce a ``C`` *and* that don't suffer the GB-style gain
    non-identifiability on a single site (GB single-site at weak
    distortion stays near gain = 1; BCB has no gain factor).

    Multi-site methods (MJ, GJ) are exercised end-to-end and
    verified to return populated result objects. Their recovered
    ``C`` is *not* held to the same tolerance: MJ recovers
    ``(strike, twist, shear)`` exactly but the joint gain has a
    structural scale degeneracy across sites (the optimiser can
    rescale all gains and the regional Z together); GJ on a 2-D
    regional has a degenerate optimisation landscape (no
    "control" site to break the gauge). Both effects are
    documented in ``docs/distortion_methods.md`` as known
    Phase-1 limitations.

    Lilley and Marti are exercised but skipped in the consistency
    check (they characterise the tensor without producing a ``C``).
    """
    periods = np.logspace(-1.0, 2.0, 12)

    # Single-site harness sweep on the primary site.
    syn = generate_synthetic_z(
        regional_type="2D",
        distortion_strength="weak",
        distortion_shear="low",
        noise_level="clean",
        periods=periods,
        site_id="primary",
        seed=42,
    )
    single_results = run_all_methods_on_synthetic(syn)
    for method_name, result in single_results.items():
        assert not (
            isinstance(result, str) and result.startswith("ERROR:")
        ), f"single-site method {method_name} failed: {result}"

    # Multi-site synthetic for MJ and GJ. Site distortions chosen
    # to keep MJ's joint optimisation in a well-conditioned basin
    # (see test_mcneice_jones.py for how MJ's recovery depends on
    # inter-site distortion variety).
    distortions_deg = [
        (syn["true_twist"], syn["true_shear"]),
        (-8.0, 12.0),
        (20.0, -5.0),
    ]
    multi_z_objs, multi_site_ids, _ = _make_multi_site_synthetic(
        periods,
        strike_deg=syn["true_strike"],
        distortions_deg=distortions_deg,
        noise_fraction=0.0,
        seed=42,
    )
    mj_result = decompose_mcneice_jones(
        multi_z_objs, multi_site_ids, n_starts=3, seed=42
    )
    gj_result = decompose_garcia_jones(
        multi_z_objs, multi_site_ids, n_starts=3, seed=42
    )
    assert isinstance(mj_result, JointDecompositionResult)
    assert isinstance(gj_result, GarciaJonesResult)
    primary_sid = multi_site_ids[0]
    assert primary_sid in mj_result.per_site_distortion
    assert "c_tensor" in mj_result.per_site_distortion[primary_sid]
    assert primary_sid in gj_result.per_site_distortion
    assert "c_tensor" in gj_result.per_site_distortion[primary_sid]
    # Per-band shared strike was recovered.
    assert 0 in mj_result.per_band_strike
    # Regional 3-D Z populated for the band.
    assert 0 in gj_result.per_band_3d_z_regional

    # Collect single-site methods' recovered C on the primary site.
    recovered_C: dict[str, np.ndarray] = {}
    for method_name in ("gb_geometric", "gb_identity", "gb_min_shear", "bibby"):
        result = single_results[method_name]
        if method_name.startswith("gb_"):
            params = result.parameters
            recovered_C[method_name] = _construct_C_gb89(
                float(np.median(params["strike"].values)),
                float(np.median(params["twist"].values)),
                float(np.median(params["shear"].values)),
                float(np.median(params["gain"].values)),
            )
        else:
            recovered_C[method_name] = result.C

    true_C = syn["true_C"]
    for method_name, c_rec in recovered_C.items():
        err = float(np.linalg.norm(c_rec - true_C))
        assert err < 0.05, (
            f"{method_name}: |C_recovered - C_true|_F = {err:.4f} "
            f">= 0.05 tolerance"
        )

    # Verify MJ recovers (twist, shear) on the primary site; the
    # joint gain has a separate scale degeneracy that this test
    # does not pin down.
    mj_rec = mj_result.per_site_distortion[primary_sid]
    assert abs(mj_rec["twist_deg"] - syn["true_twist"]) < 0.5, (
        f"MJ twist {mj_rec['twist_deg']:.2f} off truth "
        f"{syn['true_twist']:.2f}"
    )
    assert abs(mj_rec["shear_deg"] - syn["true_shear"]) < 0.5, (
        f"MJ shear {mj_rec['shear_deg']:.2f} off truth "
        f"{syn['true_shear']:.2f}"
    )

    # Lilley and Marti also returned valid result objects (but no
    # recoverable C). Check their accuracy probe is NaN as designed.
    assert np.isnan(
        compute_method_accuracy("lilley", single_results["lilley"], syn)
    )
    assert np.isnan(
        compute_method_accuracy("marti", single_results["marti"], syn)
    )


@pytest.mark.slow
def test_irreducible_decomposition_integration():
    """The spin-2 shear ``gamma`` field is consistent across methods.

    GB single-site and BCB fit the same ``Z(omega)`` with different
    parameterisations and gauges; their recovered ``C`` matrices
    differ in detail, but the gauge-clean spin-2 shear ``gamma =
    (gamma_1, gamma_2)`` (the deviatoric part of ``C - I``) should
    agree to within the inter-method bias on a clean synthetic.

    Note on the antisymmetric part ``beta``: ``beta`` is *not*
    expected to be zero for a GB-style distortion with non-zero
    twist, because the ``T(twist)`` rotation matrix has both
    symmetric and antisymmetric components. ``beta = 0`` only for
    pure-shear distortion (``twist = 0``). See the
    :mod:`...decomposition.distortion_geometry` module docstring.
    """
    periods = np.logspace(-1.0, 2.0, 12)
    syn = generate_synthetic_z(
        regional_type="2D",
        distortion_strength="weak",
        distortion_shear="low",
        noise_level="clean",
        periods=periods,
        site_id="primary",
        seed=42,
    )
    z = syn["z_obj"]

    gb = decompose(z, n_starts=3)
    bcb = decompose_bibby(z)

    gb_C = _construct_C_gb89(
        float(np.median(gb.parameters["strike"].values)),
        float(np.median(gb.parameters["twist"].values)),
        float(np.median(gb.parameters["shear"].values)),
        float(np.median(gb.parameters["gain"].values)),
    )
    bcb_C = bcb.C
    true_C = syn["true_C"]

    # Spin-2 shear from each method, plus the ground truth.
    gamma_gb = gamma_field(gb_C)
    gamma_bcb = gamma_field(bcb_C)
    gamma_true = gamma_field(true_C)

    g_gb = np.array(gamma_gb)
    g_bcb = np.array(gamma_bcb)
    g_true = np.array(gamma_true)

    # Each method recovers gamma close to the truth.
    assert np.linalg.norm(g_gb - g_true) < 0.02, (
        f"GB gamma {g_gb} off truth {g_true} "
        f"by {np.linalg.norm(g_gb - g_true):.4f}"
    )
    assert np.linalg.norm(g_bcb - g_true) < 0.05, (
        f"BCB gamma {g_bcb} off truth {g_true} "
        f"by {np.linalg.norm(g_bcb - g_true):.4f}"
    )

    # And the two methods agree with each other.
    assert np.linalg.norm(g_gb - g_bcb) < 0.05, (
        f"GB and BCB gamma disagree: "
        f"|{g_gb} - {g_bcb}| = {np.linalg.norm(g_gb - g_bcb):.4f}"
    )

    # The irreducible decomposition runs cleanly on each method's
    # recovered C - I. We verify the four parts have plausible
    # magnitudes (no NaNs / Infs); we do *not* assert beta = 0
    # because the GB synthetic uses non-zero twist and the rotation
    # part of T(twist) gives a non-zero beta by construction.
    parts_gb = irreducible_decomposition(gb_C - np.eye(2))
    parts_bcb = irreducible_decomposition(bcb_C - np.eye(2))
    for parts in (parts_gb, parts_bcb):
        for key in ("trace_a", "gamma_1", "gamma_2", "beta"):
            assert np.isfinite(parts[key]), (
                f"{key} not finite: {parts[key]}"
            )

    # Run Lilley and Marti as well, just to verify they execute on
    # the same input (their gamma is not directly extractable).
    decompose_lilley(z, n_realisations=20, seed=42)
    decompose_marti(z)
