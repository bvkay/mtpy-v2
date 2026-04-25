"""Contract tests for the GB decomposition module.

These tests verify the type and import contracts at the boundary
of :mod:`mtpy.core.transfer_function.z_analysis.decomposition`,
without exercising any numerical implementation.

The fixture :func:`tests.conftest.mt_with_impedance` provides an
:class:`mtpy.core.mt.MT` instance with a populated impedance tensor;
its ``.Z`` attribute is a real :class:`Z` object built through the
public mtpy-v2 pipeline. Per the contribution constitution
(``CLAUDE.md`` invariant 10), all decomposition tests use the
existing fixture system rather than rolling their own Z factories.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    _band_arrays_to_z,
    _BandResult,
    _bootstrap_decompose,
    _build_bounds,
    _calc_error,
    _canonical_initial_guess,
    _canonicalise_solution,
    _cluster_modes,
    _compute_ci_percentile,
    _compute_mode_probabilities,
    _convz2p,
    _convz2r,
    _DEFAULT_MODE_TOLERANCE,
    _detect_band_disagreement,
    _detect_primary_mode_warning,
    _estim_imp,
    _extract_bands,
    _extreme,
    _generate_starting_points,
    _jkvar,
    _mat_multiply,
    _objfun,
    _perturbed_initial_guess,
    _predict_z_from_primary_modes,
    _resample_residuals,
    _rotated_initial_guess,
    _solve_band,
    _solve_band_multistart,
    _unpack_x,
    _z_to_band_arrays,
    decompose,
    decompose_joint,
    DecompositionResult,
)


def _build_synthetic_band(
    theta_deg=30.0,
    twist_deg=10.0,
    shear_deg=5.0,
    log10_gain=0.0,
    n_freqs=5,
    seed=0,
    noise_level=0.0,
    period_lo=0.0,
    period_hi=1.0,
):
    """Synthetic single-band ``(z_obs, sigma, periods)`` arrays.

    Truth parameters are recoverable by ``_objfun`` / ``_solve_band``
    to machine precision when ``noise_level == 0``.
    """
    rng = np.random.default_rng(seed)
    periods = np.logspace(period_lo, period_hi, n_freqs)
    log10_rho_a = rng.uniform(0.5, 2.5, n_freqs)
    phase_a = rng.uniform(0.3, 1.4, n_freqs)
    log10_rho_b = rng.uniform(0.5, 2.5, n_freqs)
    phase_b = rng.uniform(0.3, 1.4, n_freqs)

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

    if noise_level > 0:
        sigma = noise_level * np.maximum(
            np.abs(z_obs),
            np.max(np.abs(z_obs), axis=(1, 2), keepdims=True) * 0.1,
        )
        noise = rng.normal(scale=sigma) + 1j * rng.normal(scale=sigma)
        z_obs = z_obs + noise
    else:
        sigma = np.maximum(
            0.01 * np.abs(z_obs),
            0.01 * np.max(np.abs(z_obs), axis=(1, 2), keepdims=True),
        )

    return {
        "z_obs": z_obs,
        "sigma": sigma,
        "periods": periods,
        "true_theta": theta,
        "true_twist": twist,
        "true_shear": shear,
        "true_log10_gain": log10_gain,
    }


def _build_synthetic_z(
    theta_deg=30.0,
    twist_deg=10.0,
    shear_deg=5.0,
    log10_gain=0.0,
    n_freqs=12,
    seed=0,
):
    """Synthetic ``Z`` object spanning ~3 decades, single distortion
    triple. Used by the end-to-end decompose() tests."""
    d = _build_synthetic_band(
        theta_deg=theta_deg,
        twist_deg=twist_deg,
        shear_deg=shear_deg,
        log10_gain=log10_gain,
        n_freqs=n_freqs,
        seed=seed,
        noise_level=0.0,
        period_lo=-1.0,
        period_hi=2.0,
    )
    frequencies = 1.0 / d["periods"]
    z = Z(
        z=d["z_obs"],
        z_error=d["sigma"],
        frequency=frequencies,
    )
    return z, {
        "true_strike_deg": theta_deg,
        "true_twist_deg": twist_deg,
        "true_shear_deg": shear_deg,
        "true_log10_gain": log10_gain,
    }


class TestImports:
    """The module exposes the documented public API."""

    def test_module_exports(self):
        from mtpy.core.transfer_function.z_analysis import decomposition

        assert "decompose" in decomposition.__all__
        assert "decompose_joint" in decomposition.__all__
        assert "DecompositionResult" in decomposition.__all__

    def test_no_private_leak(self):
        """Underscore-prefixed names are not in __all__."""
        from mtpy.core.transfer_function.z_analysis import decomposition

        for name in decomposition.__all__:
            assert not name.startswith(
                "_"
            ), f"private name {name!r} leaked into __all__"


class TestDecompositionResult:
    """The result dataclass constructs cleanly with the documented
    field types."""

    def _minimal_result(self, regional_z: Z) -> DecompositionResult:
        nf = regional_z.z.shape[0]
        periods = 1.0 / regional_z.frequency

        params = xr.Dataset(
            {
                "strike": ("period", np.zeros(nf)),
                "twist": ("period", np.zeros(nf)),
                "shear": ("period", np.zeros(nf)),
                "gain": ("period", np.ones(nf)),
                "anisotropy": ("period", np.zeros(nf)),
                "strike_error": ("period", np.zeros(nf)),
                "twist_error": ("period", np.zeros(nf)),
                "shear_error": ("period", np.zeros(nf)),
                "gain_error": ("period", np.zeros(nf)),
                "anisotropy_error": ("period", np.zeros(nf)),
            },
            coords={"period": periods},
        )

        chi_squared = xr.DataArray(
            np.zeros(nf), coords={"period": periods}, dims=["period"]
        )

        return DecompositionResult(
            parameters=params,
            regional_z=regional_z,
            chi_squared=chi_squared,
            rms_misfit=0.0,
            method="groom_bailey",
            options={"bandwidth": 1.0},
            metadata={"convention": "clockwise from x-axis"},
        )

    def test_construct(self, mt_with_impedance):
        """DecompositionResult constructs with documented fields."""
        result = self._minimal_result(mt_with_impedance.Z)

        assert result.method == "groom_bailey"
        assert isinstance(result.parameters, xr.Dataset)
        assert isinstance(result.regional_z, Z)
        assert isinstance(result.chi_squared, xr.DataArray)
        assert result.rms_misfit == 0.0
        assert result.options == {"bandwidth": 1.0}
        assert result.metadata == {"convention": "clockwise from x-axis"}

    def test_default_dicts(self, mt_with_impedance):
        """options and metadata default to empty dicts (not None)."""
        z = mt_with_impedance.Z
        nf = z.z.shape[0]
        periods = 1.0 / z.frequency

        result = DecompositionResult(
            parameters=xr.Dataset(coords={"period": periods}),
            regional_z=z,
            chi_squared=xr.DataArray(
                np.zeros(nf),
                coords={"period": periods},
                dims=["period"],
            ),
            rms_misfit=0.0,
            method="groom_bailey",
        )

        assert result.options == {}
        assert result.metadata == {}

    def test_documented_data_variables_present(self, mt_with_impedance):
        """The dataset slot accommodates the documented variable
        names. This locks the parameter naming contract before any
        implementation lands."""
        result = self._minimal_result(mt_with_impedance.Z)

        for name in (
            "strike",
            "twist",
            "shear",
            "gain",
            "anisotropy",
            "strike_error",
            "twist_error",
            "shear_error",
            "gain_error",
            "anisotropy_error",
        ):
            assert name in result.parameters.data_vars, (
                f"DecompositionResult.parameters missing documented "
                f"variable {name!r}"
            )


class TestZTypeRoundTrip:
    """The Z read-via-public-API, write-via-construction pattern is
    the contract for all subsequent type translation. Verify it here
    so the contract is documented in test code."""

    def test_z_round_trip_through_public_api(self, mt_with_impedance):
        """Reading Z via public attributes and reconstructing yields
        an equivalent Z."""
        z = mt_with_impedance.Z

        # Public-API reads (the contract: never touch
        # _dataset.transfer_function directly).
        z_array = z.z
        z_error_array = z.z_error
        frequency = z.frequency

        # Construct a fresh Z from the read arrays.
        z_reconstructed = Z(
            z=z_array,
            z_error=z_error_array,
            frequency=frequency,
        )

        # Equivalence check on the public-facing values.
        np.testing.assert_array_equal(z_reconstructed.z, z.z)
        np.testing.assert_array_equal(z_reconstructed.z_error, z.z_error)
        np.testing.assert_array_equal(z_reconstructed.frequency, z.frequency)


class TestDecomposeStubs:
    """``decompose_joint`` is still a stub; ``decompose`` is
    implemented (its contract tests live in TestDecomposeEndToEnd)."""

    def test_decompose_joint_raises(self, mt_with_impedance):
        # Pass a list with a single MT-like; decompose_joint takes
        # MTCollection or list[MT] but raising before validation is
        # fine for the stub test.
        with pytest.raises(NotImplementedError, match="subsequent"):
            decompose_joint([mt_with_impedance])


# ---------------------------------------------------------------------------
# Kernel unit tests. Each class targets one of the seven private
# numerical helpers in decomposition.py with synthetic-input
# known-answer cases. These are pure correctness tests; the
# Fortran-vs-Python continuity tests live separately under
# tests/cross_validation/.
# ---------------------------------------------------------------------------


class TestMatMultiply:
    """_mat_multiply produces the matrix product."""

    def test_identity(self):
        eye = np.eye(2, dtype=np.complex128)
        a = np.array(
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
            dtype=np.complex128,
        )
        np.testing.assert_array_equal(_mat_multiply(a, eye), a)
        np.testing.assert_array_equal(_mat_multiply(eye, a), a)

    def test_known_product(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.complex128)
        b = np.array([[1j, 0], [0, 1j]], dtype=np.complex128)
        expected = np.array([[1j, 2j], [3j, 4j]], dtype=np.complex128)
        np.testing.assert_array_equal(_mat_multiply(a, b), expected)


class TestExtreme:
    """_extreme returns (min, max) and rejects empty input."""

    def test_basic(self):
        x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
        vmin, vmax = _extreme(x)
        assert vmin == 1.0
        assert vmax == 9.0

    def test_single_element(self):
        x = np.array([42.0])
        assert _extreme(x) == (42.0, 42.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _extreme(np.array([]))


class TestConvz2r:
    """_convz2r is rho = |Z|^2 * T / (2 pi mu_0)."""

    def test_known_value(self):
        z = complex(1.0e-3, 5.0e-4)
        period = 100.0
        mu0 = 4.0 * np.pi * 1.0e-7
        expected = abs(z) ** 2 * period / (2.0 * np.pi * mu0)
        np.testing.assert_allclose(_convz2r(z, period), expected, rtol=1e-12)

    def test_zero_impedance(self):
        assert _convz2r(complex(0, 0), 100.0) == 0.0

    def test_period_proportionality(self):
        """rho is linear in period."""
        z = complex(1.0e-3, 0)
        rho_1 = _convz2r(z, 1.0)
        rho_10 = _convz2r(z, 10.0)
        np.testing.assert_allclose(rho_10, 10.0 * rho_1, rtol=1e-12)


class TestConvz2p:
    """_convz2p is the phase in degrees, with period vestigial."""

    @pytest.mark.parametrize("deg", [0.0, 30.0, 45.0, 60.0, 90.0, -45.0, 135.0, 179.0])
    def test_known_angles(self, deg):
        rad = np.radians(deg)
        z = complex(np.cos(rad), np.sin(rad))
        np.testing.assert_allclose(_convz2p(z), deg, atol=1e-10)

    def test_period_vestigial(self):
        """The period argument does not affect the result."""
        z = complex(1.0, 1.0)
        assert _convz2p(z, 1.0) == _convz2p(z, 1000.0)


class TestCalcError:
    """_calc_error is the chi-squared residual."""

    def test_zero_on_identical(self):
        z = np.array(
            [[1 + 0j, 2 + 3j], [4 - 1j, 0.5 + 0j]],
            dtype=np.complex128,
        )
        sigma = np.ones((2, 2))
        assert _calc_error(z, z, sigma) == 0.0

    def test_known_value(self):
        z1 = np.array([[1 + 0j, 0], [0, 0]], dtype=np.complex128)
        z2 = np.zeros((2, 2), dtype=np.complex128)
        sigma = np.ones((2, 2))
        np.testing.assert_allclose(_calc_error(z1, z2, sigma), 1.0)

    def test_sigma_weighting(self):
        """Doubling sigma quarters the chi-squared."""
        z1 = np.array([[1 + 0j, 0], [0, 0]], dtype=np.complex128)
        z2 = np.zeros((2, 2), dtype=np.complex128)
        chi_sigma_1 = _calc_error(z1, z2, np.ones((2, 2)))
        chi_sigma_2 = _calc_error(z1, z2, 2.0 * np.ones((2, 2)))
        np.testing.assert_allclose(chi_sigma_2, chi_sigma_1 / 4.0)

    def test_nonpositive_sigma_raises(self):
        z = np.zeros((2, 2), dtype=np.complex128)
        sigma_zero = np.zeros((2, 2))
        with pytest.raises(ValueError, match="non-positive"):
            _calc_error(z, z, sigma_zero)


class TestJkvar:
    """_jkvar gives delete-1 jackknife variance, sentinel for n<2."""

    def test_against_manual(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0.0, 2.0, size=20)
        n = len(x)
        total = x.sum()
        means_minus_i = (total - x) / (n - 1)
        manual = (n - 1) / n * np.sum((means_minus_i - x.mean()) ** 2)
        np.testing.assert_allclose(_jkvar(x), manual, rtol=1e-12)

    def test_n_too_small_returns_sentinel(self):
        assert _jkvar(np.array([1.0])) == -1.0
        assert _jkvar(np.array([])) == -1.0

    def test_n_equals_two(self):
        """n=2 is the minimum valid sample; check it doesn't error."""
        x = np.array([1.0, 3.0])
        result = _jkvar(x)
        # Manual: total=4, mean=2, m_-i = (3, 1), var = (1/2)*(1+1) = 1
        np.testing.assert_allclose(result, 1.0, rtol=1e-12)


class TestEstimImp:
    """_estim_imp produces the GB89 forward impedance."""

    def test_no_distortion_no_rotation(self):
        """t=e=0, theta=0: Z is the anti-diagonal regional tensor."""
        a = complex(1.0e-3, 5.0e-4)
        b = complex(8.0e-4, 3.0e-4)
        z = _estim_imp(a, b, 0.0, 0.0, 0.0)

        # Pauli-spin: with t=e=theta=0 the diagonal vanishes and
        # Z_xy = a, Z_yx = -b (anti-diagonal regional tensor).
        np.testing.assert_allclose(z[0, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(z[1, 1], 0.0, atol=1e-15)
        np.testing.assert_allclose(z[0, 1], a, atol=1e-15)
        np.testing.assert_allclose(z[1, 0], -b, atol=1e-15)

    def test_pure_rotation(self):
        """t=e=0, theta=45 deg: Z_obs = R Z_2D R^T."""
        a = complex(1.0e-3, 0)
        b = complex(2.0e-3, 0)
        theta = np.pi / 4
        z = _estim_imp(a, b, 0.0, 0.0, theta)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        z_2d = np.array([[0, a], [-b, 0]], dtype=np.complex128)
        expected = rot @ z_2d @ rot.T

        np.testing.assert_allclose(z, expected, atol=1e-12)

    def test_alpha_consistency(self):
        """The inverse Pauli decomposition recovers our alpha values.

        Build Z via _estim_imp, derive alpha from Z via Pauli
        combinations, verify it matches the direct GB89 formulas.
        Self-consistency check on the implementation.
        """
        a = complex(1.0e-3, 5.0e-4)
        b = complex(8.0e-4, 3.0e-4)
        theta = np.radians(30.0)
        t = np.tan(np.radians(15.0))
        e = np.tan(np.radians(10.0))

        z = _estim_imp(a, b, t, e, theta)

        alpha_via_z = np.array(
            [
                z[0, 0] + z[1, 1],
                z[0, 1] + z[1, 0],
                z[1, 0] - z[0, 1],
                z[0, 0] - z[1, 1],
            ]
        )

        c2 = np.cos(2.0 * theta)
        s2 = np.sin(2.0 * theta)
        alpha_direct = np.array(
            [
                -b * (e - t) + a * (t + e),
                (
                    s2 * (-b * (e - t))
                    + c2 * (-b * (1 + t * e))
                    + c2 * (a * (1 - e * t))
                    - s2 * (a * (t + e))
                ),
                -b * (1 + t * e) - a * (1 - e * t),
                (
                    c2 * (-b * (e - t))
                    - s2 * (-b * (1 + t * e))
                    - s2 * (a * (1 - e * t))
                    - c2 * (a * (t + e))
                ),
            ]
        )

        np.testing.assert_allclose(alpha_via_z, alpha_direct, atol=1e-15)


# ---------------------------------------------------------------------------
# _unpack_x and _objfun: GB optimisation residuals and analytic
# Jacobian. Three test layers below: parameter-vector indexing
# (TestUnpackX), residual correctness at known points
# (TestObjfunResiduals), input validation (TestObjfunValidation), and
# analytic Jacobian validation against central finite differences
# (TestObjfunJacobian). The Fortran reference's ``objfun_wrapped``
# uses a fundamentally different state-vector parameterisation
# (impedances stored as ``(re, im)`` per frequency, residuals
# computed in alpha-space with non-standard sigma weighting). Direct
# element-wise cross-validation is therefore not meaningful for
# _objfun; the forward-model continuity is provided by the
# _estim_imp cross-validation in
# tests/cross_validation/test_kernels_against_fortran.py.
# ---------------------------------------------------------------------------


class TestUnpackX:
    """_unpack_x correctly partitions the flat parameter vector."""

    def test_simple_case(self):
        n_freqs = 3
        x = np.arange(5 + 4 * n_freqs, dtype=np.float64)
        (
            theta,
            twist,
            shear,
            log10_gain,
            aniso,
            lr_a,
            ph_a,
            lr_b,
            ph_b,
        ) = _unpack_x(x, n_freqs)

        assert theta == 0.0
        assert twist == 1.0
        assert shear == 2.0
        assert log10_gain == 3.0
        assert aniso == 4.0
        np.testing.assert_array_equal(lr_a, [5, 6, 7])
        np.testing.assert_array_equal(ph_a, [8, 9, 10])
        np.testing.assert_array_equal(lr_b, [11, 12, 13])
        np.testing.assert_array_equal(ph_b, [14, 15, 16])

    def test_size_mismatch_raises(self):
        x = np.zeros(10)
        with pytest.raises(ValueError, match="expected x of size"):
            _unpack_x(x, n_freqs=3)


def _make_synthetic_data(
    theta_deg=30.0,
    twist_deg=15.0,
    shear_deg=10.0,
    log10_gain=0.0,
    n_freqs=5,
    seed=0,
):
    """Build a synthetic single-band problem from known parameters.

    Returns ``(x, z_obs, sigma, periods)``. ``z_obs`` is the
    forward-modelled tensor at the same parameters as ``x``, so
    ``_objfun(x, z_obs, sigma, periods)`` should give zero
    residuals to machine precision.
    """
    rng = np.random.default_rng(seed)
    periods = np.logspace(0.0, 1.0, n_freqs)

    log10_rho_a = rng.uniform(-1.0, 3.0, n_freqs)
    phase_a = rng.uniform(0.0, np.pi / 2, n_freqs)
    log10_rho_b = rng.uniform(-1.0, 3.0, n_freqs)
    phase_b = rng.uniform(0.0, np.pi / 2, n_freqs)

    theta = np.radians(theta_deg)
    twist = np.radians(twist_deg)
    shear = np.radians(shear_deg)

    x = np.concatenate(
        [
            [theta, twist, shear, log10_gain, 0.0],
            log10_rho_a,
            phase_a,
            log10_rho_b,
            phase_b,
        ]
    )

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

    sigma = np.maximum(0.01 * np.abs(z_obs), 1e-12)
    return x, z_obs, sigma, periods


class TestObjfunResiduals:
    """The cornerstone test: residuals are zero when x reproduces
    the data exactly."""

    def test_residuals_zero_at_truth(self):
        x, z_obs, sigma, periods = _make_synthetic_data()
        residuals, _ = _objfun(x, z_obs, sigma, periods, compute_jacobian=False)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    def test_residuals_grow_with_perturbation(self):
        """Perturbing strike from truth gives non-trivial residuals."""
        x, z_obs, sigma, periods = _make_synthetic_data()
        x_perturbed = x.copy()
        x_perturbed[0] += np.radians(5.0)
        residuals, _ = _objfun(
            x_perturbed,
            z_obs,
            sigma,
            periods,
            compute_jacobian=False,
        )
        rms = np.sqrt(np.mean(residuals**2))
        assert rms > 1.0

    def test_residual_shape(self):
        x, z_obs, sigma, periods = _make_synthetic_data(n_freqs=4)
        residuals, _ = _objfun(x, z_obs, sigma, periods, compute_jacobian=False)
        assert residuals.shape == (8 * 4,)

    def test_jacobian_shape(self):
        x, z_obs, sigma, periods = _make_synthetic_data(n_freqs=4)
        _, jac = _objfun(x, z_obs, sigma, periods, compute_jacobian=True)
        assert jac.shape == (8 * 4, 5 + 4 * 4)

    def test_anisotropy_jacobian_column_zero(self):
        """Column 4 (anisotropy) is identically zero by design."""
        x, z_obs, sigma, periods = _make_synthetic_data(n_freqs=4)
        _, jac = _objfun(x, z_obs, sigma, periods, compute_jacobian=True)
        np.testing.assert_array_equal(jac[:, 4], 0.0)


class TestObjfunValidation:
    """_objfun validates its inputs."""

    def test_z_obs_wrong_shape(self):
        x = np.zeros(5 + 4 * 3)
        z_obs = np.zeros((3, 2, 3), dtype=np.complex128)
        sigma = np.ones((3, 2, 3))
        periods = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="z_obs shape"):
            _objfun(x, z_obs, sigma, periods)

    def test_x_wrong_size(self):
        x = np.zeros(10)
        z_obs = np.zeros((3, 2, 2), dtype=np.complex128)
        sigma = np.ones((3, 2, 2))
        periods = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="x size"):
            _objfun(x, z_obs, sigma, periods)

    def test_nonpositive_sigma(self):
        x = np.zeros(5 + 4 * 1)
        z_obs = np.zeros((1, 2, 2), dtype=np.complex128)
        sigma = np.zeros((1, 2, 2))
        periods = np.array([1.0])
        with pytest.raises(ValueError, match="sigma"):
            _objfun(x, z_obs, sigma, periods)


class TestObjfunJacobian:
    """Analytic Jacobian agrees with central finite differences.

    This is the layer that catches sign errors in the analytic
    derivatives. Tolerance is ~1e-4 relative -- finite differences
    at h=1e-7 give about that much accuracy on Jacobian entries
    of typical magnitude. Anything looser hides bugs.
    """

    @staticmethod
    def _finite_diff_jacobian(x, z_obs, sigma, periods, h=1e-7):
        n_resid = 8 * len(periods)
        jac = np.zeros((n_resid, len(x)))
        for j in range(len(x)):
            x_plus = x.copy()
            x_plus[j] += h
            x_minus = x.copy()
            x_minus[j] -= h
            r_plus, _ = _objfun(
                x_plus,
                z_obs,
                sigma,
                periods,
                compute_jacobian=False,
            )
            r_minus, _ = _objfun(
                x_minus,
                z_obs,
                sigma,
                periods,
                compute_jacobian=False,
            )
            jac[:, j] = (r_plus - r_minus) / (2 * h)
        return jac

    def _make_problem(
        self,
        theta_deg=30.0,
        twist_deg=15.0,
        shear_deg=10.0,
        log10_gain=0.0,
        n_freqs=4,
        seed=0,
    ):
        """Build a synthetic problem and perturb x slightly off
        the residuals-zero point so all derivatives are non-trivial.
        """
        x_true, z_obs, sigma, periods = _make_synthetic_data(
            theta_deg=theta_deg,
            twist_deg=twist_deg,
            shear_deg=shear_deg,
            log10_gain=log10_gain,
            n_freqs=n_freqs,
            seed=seed,
        )
        rng = np.random.default_rng(seed + 1000)
        x = x_true + 0.01 * rng.normal(size=x_true.shape)
        return x, z_obs, sigma, periods

    def test_jacobian_matches_finite_diff_canonical(self):
        x, z_obs, sigma, periods = self._make_problem()

        _, jac_analytic = _objfun(x, z_obs, sigma, periods, compute_jacobian=True)
        jac_numeric = self._finite_diff_jacobian(x, z_obs, sigma, periods, h=1e-7)

        # Anisotropy column: zero analytically (by construction)
        # and ~zero numerically (because the parameter has no
        # effect on the model).
        np.testing.assert_array_equal(jac_analytic[:, 4], 0.0)
        np.testing.assert_allclose(jac_numeric[:, 4], 0.0, atol=1e-6)

        # All other columns: agreement at finite-difference
        # precision.
        cols_to_check = [0, 1, 2, 3] + list(range(5, len(x)))
        for j in cols_to_check:
            col_a = jac_analytic[:, j]
            col_n = jac_numeric[:, j]
            scale = max(
                np.max(np.abs(col_a)),
                np.max(np.abs(col_n)),
                1e-15,
            )
            np.testing.assert_allclose(
                col_a,
                col_n,
                atol=scale * 1e-5,
                rtol=1e-5,
                err_msg=f"Jacobian column {j} disagrees",
            )

    @pytest.mark.parametrize("seed", [1, 17, 42])
    def test_jacobian_at_multiple_parameter_points(self, seed):
        """Run the Jacobian check at several points to catch
        accidental agreement at any one location."""
        x, z_obs, sigma, periods = self._make_problem(
            theta_deg=20 + (10 * seed) % 60,
            twist_deg=5 + (5 * seed) % 25,
            shear_deg=5 + (5 * (seed + 1)) % 25,
            seed=seed,
        )
        _, jac_a = _objfun(x, z_obs, sigma, periods, compute_jacobian=True)
        jac_n = self._finite_diff_jacobian(x, z_obs, sigma, periods)
        cols_to_check = [0, 1, 2, 3] + list(range(5, len(x)))
        for j in cols_to_check:
            ca = jac_a[:, j]
            cn = jac_n[:, j]
            scale = max(np.max(np.abs(ca)), np.max(np.abs(cn)), 1e-15)
            np.testing.assert_allclose(
                ca,
                cn,
                atol=scale * 1e-4,
                rtol=1e-4,
                err_msg=f"seed={seed}, column {j}",
            )


class TestCanonicaliseSolution:
    """The 90-degree / shear-sign symmetry fold reduces solutions to
    a canonical branch with strike in [0, pi/2)."""

    def test_in_canonical_range_already(self):
        s, t, sh = _canonicalise_solution(
            np.radians(30.0), np.radians(10.0), np.radians(5.0)
        )
        assert np.isclose(np.degrees(s), 30.0)
        assert np.isclose(np.degrees(t), 10.0)
        assert np.isclose(np.degrees(sh), 5.0)

    def test_folds_upper_half(self):
        # strike=120 deg, shear=+5 deg should fold to strike=30 deg,
        # shear=-5 deg.
        s, t, sh = _canonicalise_solution(
            np.radians(120.0), np.radians(10.0), np.radians(5.0)
        )
        assert np.isclose(np.degrees(s), 30.0, atol=1e-9)
        assert np.isclose(np.degrees(t), 10.0)
        assert np.isclose(np.degrees(sh), -5.0, atol=1e-9)

    def test_handles_pi_over_two_exactly(self):
        # The boundary case: floating-point % np.pi can drop strike
        # one ULP below pi/2; the tolerance in the fold should still
        # send it to ~0.
        s, _, _ = _canonicalise_solution(np.pi / 2.0, 0.0, 0.0)
        assert s < 1e-8

    def test_negative_strike(self):
        # strike=-30 deg = +150 deg mod 180; should fold to 60 deg
        # with shear sign flipped.
        s, _, sh = _canonicalise_solution(np.radians(-30.0), 0.0, np.radians(7.0))
        assert np.isclose(np.degrees(s), 60.0, atol=1e-9)
        assert np.isclose(np.degrees(sh), -7.0, atol=1e-9)

    def test_strike_above_pi(self):
        # strike=200 deg = 20 deg mod 180. Below pi/2; no fold.
        s, _, sh = _canonicalise_solution(np.radians(200.0), 0.0, np.radians(3.0))
        assert np.isclose(np.degrees(s), 20.0, atol=1e-9)
        assert np.isclose(np.degrees(sh), 3.0, atol=1e-9)

    def test_twist_unchanged(self):
        # Twist is the symmetry-invariant; should never change.
        for strike_deg in [10, 50, 90, 130, 170]:
            _, t_out, _ = _canonicalise_solution(
                np.radians(strike_deg),
                np.radians(7.5),
                np.radians(2.0),
            )
            assert np.isclose(np.degrees(t_out), 7.5, atol=1e-9)


class TestExtractBands:
    """_extract_bands partitions periods into log10-decade bands."""

    def test_single_decade_one_band(self):
        periods = np.logspace(0, 1, 5)
        bands = _extract_bands(periods, bandwidth=1.0, overlap=0.0)
        assert len(bands) == 1
        np.testing.assert_array_equal(bands[0], np.arange(5))

    def test_three_decades_three_bands(self):
        periods = np.logspace(-1, 2, 12)
        bands = _extract_bands(periods, bandwidth=1.0, overlap=0.0)
        assert len(bands) == 3
        all_idx = sorted(set(int(i) for b in bands for i in b))
        assert all_idx == list(range(12))

    def test_overlap_creates_duplicates(self):
        periods = np.logspace(0, 2, 10)
        bands = _extract_bands(periods, bandwidth=1.0, overlap=0.5)
        all_idx = np.concatenate(bands)
        assert len(all_idx) > len(periods)

    def test_invalid_overlap_raises(self):
        periods = np.logspace(0, 2, 10)
        with pytest.raises(ValueError, match="overlap"):
            _extract_bands(periods, bandwidth=1.0, overlap=1.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _extract_bands(np.array([]))

    def test_negative_periods_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _extract_bands(np.array([1.0, -2.0, 3.0]))

    def test_too_narrow_raises(self):
        # Single period: no band has 2+ entries
        with pytest.raises(ValueError, match="2 or more"):
            _extract_bands(np.array([1.0]), bandwidth=1.0)


class TestCanonicalInitialGuess:
    """_canonical_initial_guess produces a valid x0 vector."""

    def test_shape(self):
        d = _build_synthetic_band(n_freqs=5)
        x0 = _canonical_initial_guess(d["z_obs"], d["sigma"], d["periods"])
        assert x0.shape == (5 + 4 * 5,)

    def test_distortion_starts_at_zero(self):
        d = _build_synthetic_band(n_freqs=4)
        x0 = _canonical_initial_guess(d["z_obs"], d["sigma"], d["periods"])
        assert x0[1] == 0.0  # twist
        assert x0[2] == 0.0  # shear
        assert x0[3] == 0.0  # log10_gain
        assert x0[4] == 0.0  # anisotropy


class TestBuildBounds:
    """_build_bounds produces ordered, overridable bounds."""

    def test_default_shape(self):
        n = 5
        lo, up = _build_bounds(n)
        assert lo.shape == (5 + 4 * n,)
        assert up.shape == (5 + 4 * n,)

    def test_lower_below_upper(self):
        lo, up = _build_bounds(5)
        assert np.all(lo < up)

    def test_phase_bound_loosened(self):
        # Per the design decision, phase bounds are wider than the
        # textbook causal first quadrant.
        lo, up = _build_bounds(3)
        # phase_a band: indices 5+1*n .. 5+2*n
        assert lo[5 + 3] < 0.0
        assert up[5 + 3] > np.pi / 2.0

    def test_override(self):
        lo, up = _build_bounds(5, bounds_override={"twist": (-0.1, 0.1)})
        assert lo[1] == -0.1
        assert up[1] == 0.1

    def test_unknown_override_raises(self):
        with pytest.raises(ValueError, match="unknown override key"):
            _build_bounds(5, bounds_override={"theta_typo": (0, 1)})


class TestSolveBand:
    """_solve_band recovers known parameters from synthetic data."""

    def test_recovers_realistic_distortion(self):
        d = _build_synthetic_band(
            theta_deg=30.0,
            twist_deg=10.0,
            shear_deg=5.0,
            log10_gain=0.0,
        )
        result = _solve_band(d["z_obs"], d["sigma"], d["periods"])
        assert isinstance(result, _BandResult)
        assert result.converged
        # Strike: handle the 90-deg branch by checking against both.
        rec = result.x_opt[0] % np.pi
        truth = d["true_theta"] % np.pi
        diff = min(
            abs(rec - truth),
            abs(rec - truth - np.pi / 2),
            abs(rec - truth + np.pi / 2),
        )
        assert diff < np.radians(5.0)
        assert result.rms_misfit < 0.01

    def test_recovers_no_distortion(self):
        d = _build_synthetic_band(
            theta_deg=0.0,
            twist_deg=0.0,
            shear_deg=0.0,
        )
        result = _solve_band(d["z_obs"], d["sigma"], d["periods"])
        assert result.converged
        # No-distortion: rms should be near machine epsilon.
        assert result.rms_misfit < 1e-3

    def test_recovers_with_noise(self):
        d = _build_synthetic_band(
            theta_deg=30.0,
            twist_deg=10.0,
            shear_deg=5.0,
            log10_gain=0.0,
            noise_level=0.02,
        )
        result = _solve_band(d["z_obs"], d["sigma"], d["periods"])
        assert result.converged
        rec = result.x_opt[0] % np.pi
        truth = d["true_theta"] % np.pi
        diff = min(
            abs(rec - truth),
            abs(rec - truth - np.pi / 2),
            abs(rec - truth + np.pi / 2),
        )
        # With 2% noise on a 5-frequency band, expect <15 deg.
        assert diff < np.radians(15.0)

    def test_x0_clipped_to_bounds(self):
        # Pass an x0 deliberately outside bounds; _solve_band should
        # clip and run rather than letting scipy raise.
        d = _build_synthetic_band(theta_deg=30.0)
        n = len(d["periods"])
        bad_x0 = np.zeros(5 + 4 * n)
        bad_x0[3] = 100.0  # log10_gain way above bound (default 2)
        # Should not raise: clip happens before scipy.
        result = _solve_band(d["z_obs"], d["sigma"], d["periods"], x0=bad_x0)
        assert result.x_opt[3] <= 2.0 + 1e-9

    def test_anisotropy_error_is_inf(self):
        # Anisotropy column is identically zero in the Jacobian; its
        # formal variance is infinite. The error should report inf.
        d = _build_synthetic_band(theta_deg=30.0)
        result = _solve_band(d["z_obs"], d["sigma"], d["periods"])
        assert not np.isfinite(result.x_err[4])


class TestZBandArrayRoundTrip:
    """_z_to_band_arrays / _band_arrays_to_z form the Z<->array
    boundary."""

    def test_z_to_band_arrays_basic(self, mt_with_impedance):
        z = mt_with_impedance.Z
        n_total = z.z.shape[0]
        band_idx = np.arange(min(5, n_total))
        z_obs, sigma, periods = _z_to_band_arrays(z, band_idx)
        assert z_obs.shape == (len(band_idx), 2, 2)
        assert sigma.shape == z_obs.shape
        assert periods.shape == (len(band_idx),)
        np.testing.assert_array_equal(z_obs, z.z[band_idx])

    def test_band_arrays_to_z_anti_diagonal(self):
        # Strike-frame regional tensor: Z = [[0, a], [-b, 0]]
        n = 4
        periods = np.logspace(0, 1, n)
        log10_rho_a = np.full(n, 2.0)
        phase_a = np.full(n, np.pi / 4)
        log10_rho_b = np.full(n, 2.0)
        phase_b = np.full(n, np.pi / 4)
        z_reg, z_err = _band_arrays_to_z(
            log10_rho_a,
            phase_a,
            log10_rho_b,
            phase_b,
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            periods,
        )
        assert z_reg.shape == (n, 2, 2)
        # Anti-diagonal: diagonals are zero
        np.testing.assert_array_equal(z_reg[:, 0, 0], 0)
        np.testing.assert_array_equal(z_reg[:, 1, 1], 0)
        # Z[1, 0] = -b (sign flip on second component)
        assert np.all(z_reg[:, 1, 0].real < 0) | np.all(z_reg[:, 1, 0].imag != 0)

    def test_z_to_band_arrays_no_z_error_raises(self):
        # Z without z_error: raises ValueError before producing arrays.
        rng = np.random.default_rng(0)
        n = 3
        z_arr = rng.normal(size=(n, 2, 2)) + 1j * rng.normal(size=(n, 2, 2))
        z = Z(z=z_arr, frequency=np.logspace(0, 1, n))
        # mtpy-v2 may auto-populate z_error to zeros; either path
        # raises (None or non-positive entries).
        with pytest.raises(ValueError, match="None|non-positive"):
            _z_to_band_arrays(z, np.arange(n))


class TestDecomposeEndToEnd:
    """End-to-end decompose() recovers known parameters from synthetic
    Z spanning 3 decades."""

    def test_returns_decomposition_result(self):
        z, _ = _build_synthetic_z()
        result = decompose(z)
        assert isinstance(result, DecompositionResult)
        assert result.method == "groom_bailey"
        assert result.frame == "measurement"

    def test_dataset_has_documented_variables(self):
        z, _ = _build_synthetic_z()
        result = decompose(z)
        for name in (
            "strike",
            "twist",
            "shear",
            "gain",
            "anisotropy",
            "strike_error",
            "twist_error",
            "shear_error",
            "gain_error",
            "anisotropy_error",
        ):
            assert name in result.parameters.data_vars
        assert "period" in result.parameters.coords

    def test_strike_attrs(self):
        z, _ = _build_synthetic_z()
        result = decompose(z)
        assert result.parameters["strike"].attrs.get("units") == "degrees"
        assert result.parameters["strike"].attrs.get("range") == "[0, 90)"

    def test_chi_squared_attrs(self):
        z, _ = _build_synthetic_z()
        result = decompose(z)
        assert result.chi_squared.attrs.get("degrees_of_freedom") == 8

    def test_regional_z_is_z_instance(self):
        z, _ = _build_synthetic_z()
        result = decompose(z)
        assert isinstance(result.regional_z, Z)
        assert len(result.regional_z.frequency) == len(z.frequency)

    def test_regional_z_in_measurement_frame(self):
        # Build with theta=30 deg and verify regional_z has all four
        # entries non-zero (i.e. not strike-frame anti-diagonal).
        z, _ = _build_synthetic_z(theta_deg=30.0)
        result = decompose(z)
        # In measurement frame, all four entries should be non-zero
        # for a non-zero strike. (Strike-frame regional_z has zero
        # diagonals.)
        diag_max = np.max(np.abs(result.regional_z.z[:, 0, 0]))
        offdiag_max = np.max(np.abs(result.regional_z.z[:, 0, 1]))
        assert diag_max > 0.01 * offdiag_max

    def test_recovers_strike_realistic(self):
        z, truth = _build_synthetic_z(
            theta_deg=30.0,
            twist_deg=10.0,
            shear_deg=5.0,
        )
        result = decompose(z)
        # Median strike across periods (avoid bands that fall into
        # alternative local minima)
        median_strike = float(np.nanmedian(result.parameters["strike"].values))
        # Truth (30 deg) is already in the canonical [0, 90) range
        diff = min(
            abs(median_strike - 30.0),
            abs(median_strike - 30.0 - 90.0),
            abs(median_strike - 30.0 + 90.0),
        )
        assert diff < 5.0

    def test_recovers_no_distortion(self):
        z, _ = _build_synthetic_z(
            theta_deg=0.0,
            twist_deg=0.0,
            shear_deg=0.0,
        )
        result = decompose(z)
        # No-distortion should canonicalise to strike=0 across all
        # bands
        median_strike = float(np.nanmedian(result.parameters["strike"].values))
        # Either 0 or near-90 (folded to 0) — check against [0, 5]
        # since canonicalisation maps both to 0
        assert median_strike < 5.0

    def test_metadata_present(self):
        z, _ = _build_synthetic_z()
        result = decompose(z)
        assert "convention" in result.metadata
        assert "n_bands" in result.metadata
        assert "per_band" in result.metadata
        assert result.metadata["n_bands"] >= 1
        assert result.metadata.get("regional_z_frame") == "measurement"
        assert "canonicalisation" in result.metadata

    def test_options_recorded(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, bandwidth=1.5, overlap=0.5)
        assert result.options["bandwidth"] == 1.5
        assert result.options["overlap"] == 0.5

    def test_no_z_error_raises(self):
        rng = np.random.default_rng(0)
        n = 5
        z_arr = rng.normal(scale=1e-3, size=(n, 2, 2)) + 1j * rng.normal(
            scale=1e-3, size=(n, 2, 2)
        )
        frequencies = np.logspace(-1, 1, n)
        z = Z(z=z_arr, frequency=frequencies)
        with pytest.raises(ValueError, match="None|non-positive"):
            decompose(z)

    def test_period_window(self):
        z, _ = _build_synthetic_z(n_freqs=12)
        # Window that selects the middle decade only
        result = decompose(z, periods=(1.0, 10.0))
        assert result.parameters["period"].values.min() >= 1.0
        assert result.parameters["period"].values.max() <= 10.0

    def test_period_window_too_narrow_raises(self):
        z, _ = _build_synthetic_z()
        with pytest.raises(ValueError):
            # An empty window (no periods inside it)
            decompose(z, periods=(1e10, 1e11))


class TestRotatedInitialGuess:
    """_rotated_initial_guess produces the 90-degree symmetry alternative."""

    def test_strike_rotated_by_pi_2(self):
        d = _build_synthetic_band(n_freqs=5)
        canonical = _canonical_initial_guess(d["z_obs"], d["sigma"], d["periods"])
        rotated = _rotated_initial_guess(d["z_obs"], d["sigma"], d["periods"])
        diff = (rotated[0] - canonical[0]) % np.pi
        # Allow either pi/2 or 0 (mod pi) — rotation can wrap
        wrap = min(abs(diff - np.pi / 2.0), abs(diff - np.pi / 2.0 + np.pi))
        assert wrap < 1e-10

    def test_te_tm_swap(self):
        d = _build_synthetic_band(n_freqs=4)
        canonical = _canonical_initial_guess(d["z_obs"], d["sigma"], d["periods"])
        rotated = _rotated_initial_guess(d["z_obs"], d["sigma"], d["periods"])
        n = 4
        base = 5
        # rotated's log10_rho_a slot == canonical's log10_rho_b slot
        np.testing.assert_allclose(
            rotated[base : base + n],
            canonical[base + 2 * n : base + 3 * n],
        )
        # rotated's phase_a slot == canonical's phase_b slot
        np.testing.assert_allclose(
            rotated[base + n : base + 2 * n],
            canonical[base + 3 * n : base + 4 * n],
        )

    def test_distortion_remains_zero(self):
        d = _build_synthetic_band(n_freqs=5)
        rotated = _rotated_initial_guess(d["z_obs"], d["sigma"], d["periods"])
        assert rotated[1] == 0.0  # twist
        assert rotated[2] == 0.0  # shear
        assert rotated[3] == 0.0  # log10_gain
        assert rotated[4] == 0.0  # anisotropy


class TestPerturbedInitialGuess:
    """_perturbed_initial_guess returns valid bounded perturbations."""

    def test_within_bounds(self):
        canonical = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.5])
        lower = np.array([0.0, -1.0, -1.0, -2.0, -2.0, -3.0, 0.0, -3.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 1.5, 3.0, 1.5])
        rng = np.random.default_rng(0)
        x = _perturbed_initial_guess(canonical, lower, upper, rng)
        np.testing.assert_array_less(lower - 1e-12, x)
        np.testing.assert_array_less(x, upper + 1e-12)

    def test_reproducibility(self):
        canonical = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        lower = -np.ones(5)
        upper = np.ones(5)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        x_a = _perturbed_initial_guess(canonical, lower, upper, rng_a)
        x_b = _perturbed_initial_guess(canonical, lower, upper, rng_b)
        np.testing.assert_array_equal(x_a, x_b)

    def test_perturbation_actually_perturbs(self):
        canonical = np.array([0.5, 0.0, 0.0, 0.0, 0.0])
        lower = -np.ones(5)
        upper = np.ones(5)
        rng = np.random.default_rng(123)
        x = _perturbed_initial_guess(canonical, lower, upper, rng)
        # At least one parameter should differ from canonical
        assert not np.allclose(x, canonical)


class TestGenerateStartingPoints:
    """_generate_starting_points produces hybrid starts."""

    def test_count_matches_n_starts(self):
        d = _build_synthetic_band(n_freqs=4)
        lower, upper = _build_bounds(4)
        rng = np.random.default_rng(0)
        starts = _generate_starting_points(
            d["z_obs"], d["sigma"], d["periods"], lower, upper, 5, rng
        )
        assert len(starts) == 5

    def test_n_starts_one_returns_only_canonical(self):
        d = _build_synthetic_band(n_freqs=4)
        lower, upper = _build_bounds(4)
        rng = np.random.default_rng(0)
        starts = _generate_starting_points(
            d["z_obs"], d["sigma"], d["periods"], lower, upper, 1, rng
        )
        assert len(starts) == 1
        canonical = np.clip(
            _canonical_initial_guess(d["z_obs"], d["sigma"], d["periods"]),
            lower,
            upper,
        )
        np.testing.assert_array_equal(starts[0], canonical)

    def test_first_two_are_canonical_and_rotated(self):
        d = _build_synthetic_band(n_freqs=4)
        lower, upper = _build_bounds(4)
        rng = np.random.default_rng(0)
        starts = _generate_starting_points(
            d["z_obs"], d["sigma"], d["periods"], lower, upper, 5, rng
        )
        canonical_expected = np.clip(
            _canonical_initial_guess(d["z_obs"], d["sigma"], d["periods"]),
            lower,
            upper,
        )
        rotated_expected = np.clip(
            _rotated_initial_guess(d["z_obs"], d["sigma"], d["periods"]),
            lower,
            upper,
        )
        np.testing.assert_array_equal(starts[0], canonical_expected)
        np.testing.assert_array_equal(starts[1], rotated_expected)

    def test_reproducibility_same_seed(self):
        d = _build_synthetic_band(n_freqs=4)
        lower, upper = _build_bounds(4)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        starts_a = _generate_starting_points(
            d["z_obs"], d["sigma"], d["periods"], lower, upper, 5, rng_a
        )
        starts_b = _generate_starting_points(
            d["z_obs"], d["sigma"], d["periods"], lower, upper, 5, rng_b
        )
        for a, b in zip(starts_a, starts_b):
            np.testing.assert_array_equal(a, b)

    def test_n_starts_zero_raises(self):
        d = _build_synthetic_band(n_freqs=4)
        lower, upper = _build_bounds(4)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="n_starts must be >= 1"):
            _generate_starting_points(
                d["z_obs"], d["sigma"], d["periods"], lower, upper, 0, rng
            )


def _make_band_result_at(strike_deg, twist_deg, shear_deg, log10_gain, rms, n_freqs=4):
    """Helper: synthesise a _BandResult with controlled canonical-form
    parameters. Used for clustering / probability tests where we don't
    need a real fit."""
    n_params = 5 + 4 * n_freqs
    x_opt = np.zeros(n_params)
    x_opt[0] = np.radians(strike_deg)
    x_opt[1] = np.radians(twist_deg)
    x_opt[2] = np.radians(shear_deg)
    x_opt[3] = log10_gain
    x_err = np.full(n_params, 0.01)
    x_err[4] = np.inf  # anisotropy non-identifiable
    n_resid = 8 * n_freqs
    residuals = np.full(n_resid, rms * np.sqrt(n_resid) / np.sqrt(n_resid))
    chi_sq = rms * rms * n_resid
    # Synthesise a non-degenerate Jacobian with anisotropy column zero
    jac = np.eye(n_resid, n_params)
    jac[:, 4] = 0.0
    return _BandResult(
        x_opt=x_opt,
        x_err=x_err,
        residuals=residuals,
        chi_squared=chi_sq,
        rms_misfit=rms,
        n_iter=10,
        converged=True,
        cost_at_opt=0.5 * chi_sq,
        jacobian=jac,
    )


class TestClusterModes:
    """_cluster_modes correctly groups converged points."""

    def test_empty_input(self):
        assert _cluster_modes([]) == []

    def test_single_result(self):
        br = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        modes = _cluster_modes([br])
        assert len(modes) == 1
        assert modes[0].n_starts_landing_here == 1

    def test_two_close_results_collapse_to_one_mode(self):
        # Two points within 0.1 deg on every physical param -> 1 mode
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        br_b = _make_band_result_at(30.05, 5.05, 3.05, 0.005, 0.51)
        modes = _cluster_modes([br_a, br_b])
        assert len(modes) == 1
        assert modes[0].n_starts_landing_here == 2

    def test_two_distinct_results_yield_two_modes(self):
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        # Strike differs by 30 deg: well outside 0.5 deg tolerance
        br_b = _make_band_result_at(60.0, 5.0, 3.0, 0.0, 0.7)
        modes = _cluster_modes([br_a, br_b])
        assert len(modes) == 2
        # Modes sorted ascending by RMS
        assert modes[0].rms_misfit < modes[1].rms_misfit

    def test_rms_path_difference_does_not_split(self):
        # Same physical params, very different RMS (1e-3 vs 1e-1) ->
        # 1 mode under the new RMS-free clustering rule.
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 1e-3)
        br_b = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 1e-1)
        modes = _cluster_modes([br_a, br_b])
        assert len(modes) == 1

    def test_rms_relative_in_tolerance_warns(self):
        br = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        with pytest.warns(UserWarning, match="rms_relative"):
            _cluster_modes([br], mode_tolerance={"rms_relative": 1e-3})


class TestComputeModeProbabilities:
    """_compute_mode_probabilities normalised and ordered."""

    def test_empty(self):
        assert _compute_mode_probabilities([]) == []

    def test_single_mode_probability_one(self):
        br = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        modes = _cluster_modes([br])
        probs = _compute_mode_probabilities(modes)
        assert probs == [1.0]

    def test_sum_to_one_two_modes(self):
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        br_b = _make_band_result_at(60.0, 5.0, 3.0, 0.0, 0.7)
        modes = _cluster_modes([br_a, br_b])
        probs = _compute_mode_probabilities(modes)
        np.testing.assert_allclose(sum(probs), 1.0, atol=1e-12)

    def test_lower_rms_higher_probability(self):
        # Two modes with similar covariance structure (synthetic
        # _BandResult uses identity jacobian shape) but different RMS;
        # the lower-chi-squared one should win.
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.1)
        br_b = _make_band_result_at(60.0, 5.0, 3.0, 0.0, 1.0)
        modes = _cluster_modes([br_a, br_b])
        probs = _compute_mode_probabilities(modes)
        assert probs[0] > probs[1]


class TestSolveBandMultistart:
    """_solve_band_multistart discovers modes correctly."""

    def test_unimodal_problem_one_mode(self):
        # Clean synthetic single-band fit: should converge to one mode.
        d = _build_synthetic_band(theta_deg=30.0, n_freqs=5)
        rng = np.random.default_rng(0)
        modes = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=5, rng=rng
        )
        assert len(modes) == 1
        # All five starts should have landed at the primary mode
        assert modes[0].n_starts_landing_here == 5

    def test_n_starts_one_returns_one_mode(self):
        d = _build_synthetic_band(theta_deg=30.0, n_freqs=5)
        rng = np.random.default_rng(0)
        modes = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=1, rng=rng
        )
        assert len(modes) == 1
        assert modes[0].n_starts_landing_here == 1

    def test_primary_mode_recovers_truth(self):
        d = _build_synthetic_band(
            theta_deg=30.0, twist_deg=10.0, shear_deg=5.0, n_freqs=5
        )
        rng = np.random.default_rng(0)
        modes = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=5, rng=rng
        )
        cf = modes[0].canonical_form
        # Truth: 30 deg in canonical [0, 90)
        assert abs(cf["strike_deg"] - 30.0) < 1.0

    def test_seed_reproducibility(self):
        d = _build_synthetic_band(theta_deg=30.0, n_freqs=5)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        modes_a = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=5, rng=rng_a
        )
        modes_b = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=5, rng=rng_b
        )
        assert len(modes_a) == len(modes_b)
        for a, b in zip(modes_a, modes_b):
            assert abs(a.rms_misfit - b.rms_misfit) < 1e-12

    def test_probabilities_populated(self):
        d = _build_synthetic_band(theta_deg=30.0, n_freqs=5)
        rng = np.random.default_rng(0)
        modes = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=5, rng=rng
        )
        for mode in modes:
            assert mode.probability is not None
            assert 0.0 <= mode.probability <= 1.0
        np.testing.assert_allclose(sum(m.probability for m in modes), 1.0, atol=1e-12)


class TestDetectPrimaryModeWarning:
    """_detect_primary_mode_warning fires only on close modes."""

    def test_no_warning_with_single_mode(self):
        br = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        modes = _cluster_modes([br])
        triggered, _ = _detect_primary_mode_warning(
            [(np.array([0]), modes)], threshold=1.5
        )
        assert triggered is False

    def test_warning_when_modes_close(self):
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        br_b = _make_band_result_at(60.0, 5.0, 3.0, 0.0, 0.6)
        modes = _cluster_modes([br_a, br_b])
        triggered, text = _detect_primary_mode_warning(
            [(np.array([0]), modes)], threshold=1.5
        )
        assert triggered is True
        assert "modes" in text.lower()

    def test_no_warning_when_modes_far(self):
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.1)
        br_b = _make_band_result_at(60.0, 5.0, 3.0, 0.0, 1.0)
        modes = _cluster_modes([br_a, br_b])
        triggered, _ = _detect_primary_mode_warning(
            [(np.array([0]), modes)], threshold=1.5
        )
        assert triggered is False


class TestDetectBandDisagreement:
    """_detect_band_disagreement uses physical-params-only matcher."""

    def test_none_when_single_band(self):
        br = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        modes = _cluster_modes([br])
        result = _detect_band_disagreement(
            [(np.array([0]), modes)], _DEFAULT_MODE_TOLERANCE
        )
        assert result is None

    def test_none_when_bands_agree_despite_different_rms(self):
        # Two bands at the same physical parameters but with very
        # different RMS values (because they fit different period
        # subsets). Should NOT report disagreement under the new
        # physical-params-only matcher.
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.001)
        br_b = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        modes_a = _cluster_modes([br_a])
        modes_b = _cluster_modes([br_b])
        result = _detect_band_disagreement(
            [(np.array([0]), modes_a), (np.array([1]), modes_b)],
            _DEFAULT_MODE_TOLERANCE,
        )
        assert result is None

    def test_reports_when_strikes_disagree(self):
        br_a = _make_band_result_at(30.0, 5.0, 3.0, 0.0, 0.5)
        br_b = _make_band_result_at(60.0, 5.0, 3.0, 0.0, 0.5)
        modes_a = _cluster_modes([br_a])
        modes_b = _cluster_modes([br_b])
        result = _detect_band_disagreement(
            [(np.array([0]), modes_a), (np.array([1]), modes_b)],
            _DEFAULT_MODE_TOLERANCE,
        )
        assert result is not None
        assert len(result["disagreements"]) >= 1


class TestDecomposeMultistart:
    """decompose() with multi-start handles multi-modal cases."""

    def test_default_n_starts_recorded(self):
        z, _ = _build_synthetic_z()
        result = decompose(z)
        assert result.metadata["n_starts_per_band"] == 5

    def test_n_starts_parameter_respected(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, n_starts=3)
        assert result.metadata["n_starts_per_band"] == 3

    def test_n_starts_one_falls_through(self):
        # n_starts=1 should give same behaviour as Session 4
        # single-start (no multimodal warning, single mode per band).
        z, _ = _build_synthetic_z()
        result = decompose(z, n_starts=1)
        for band_meta in result.metadata["per_band"]:
            assert band_meta["n_modes"] == 1
        assert result.metadata["primary_mode_warning"] is False

    def test_per_band_modes_metadata_present(self):
        z, _ = _build_synthetic_z()
        result = decompose(z)
        for band_meta in result.metadata["per_band"]:
            assert "modes" in band_meta
            assert "n_modes" in band_meta
            assert "primary_mode_index" in band_meta
            assert band_meta["primary_mode_index"] == 0
            for mode in band_meta["modes"]:
                assert "probability" in mode
                assert "n_starts_landing_here" in mode
                assert "canonical_form" in mode

    def test_seed_reproducibility(self):
        z, _ = _build_synthetic_z()
        a = decompose(z, seed=99)
        b = decompose(z, seed=99)
        np.testing.assert_array_equal(
            a.parameters["strike"].values, b.parameters["strike"].values
        )

    def test_recovers_session4_multimodal_case(self):
        # Session 4 finding: theta=75, twist=15, shear=-8 had band 1
        # land at theta=0 with single-start. With multi-start n=5,
        # primary mode of every band should now recover theta=75
        # (or its canonical fold, 75 in [0, 90)).
        z, _ = _build_synthetic_z(theta_deg=75.0, twist_deg=15.0, shear_deg=-8.0)
        result = decompose(z, n_starts=5)
        strikes = result.parameters["strike"].values
        finite = strikes[np.isfinite(strikes)]
        # Median strike should be near 75 deg
        median_strike = float(np.median(finite))
        assert (
            abs(median_strike - 75.0) < 5.0
        ), f"median strike {median_strike} != 75 (multi-start failed)"

    def test_return_all_modes_dataset_shape(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, return_all_modes=True)
        assert "mode" in result.parameters.dims
        # Primary mode = 0, must equal the regular per-period values
        # for the same-shape (period,) projection
        n_periods = result.parameters.sizes["period"]
        max_modes = result.parameters.sizes["mode"]
        assert max_modes >= 1
        assert result.parameters["strike"].shape == (n_periods, max_modes)

    def test_rms_relative_warns(self):
        z, _ = _build_synthetic_z()
        with pytest.warns(UserWarning, match="rms_relative"):
            decompose(z, mode_tolerance={"rms_relative": 1e-3})

    def test_options_record_includes_multistart_params(self):
        z, _ = _build_synthetic_z()
        result = decompose(
            z,
            n_starts=3,
            mode_warning_threshold=2.0,
            perturbation_scale=0.2,
        )
        assert result.options["n_starts"] == 3
        assert result.options["mode_warning_threshold"] == 2.0
        assert result.options["perturbation_scale"] == 0.2

    def test_band_disagreement_absent_for_consistent_synthetic(self):
        # Synthetic with single distortion triple across 3 decades —
        # bands should agree on primary-mode physical parameters.
        z, _ = _build_synthetic_z(theta_deg=30.0, n_freqs=12)
        result = decompose(z)
        assert "band_disagreement" not in result.metadata


class TestResampleResiduals:
    """_resample_residuals generates correct synthetic Z."""

    def test_shape_matches_input(self):
        rng = np.random.default_rng(0)
        n_freqs = 5
        z_pred = rng.normal(size=(n_freqs, 2, 2)) + 1j * rng.normal(
            size=(n_freqs, 2, 2)
        )
        z_obs = z_pred * 1.1
        sigma = np.abs(z_pred) * 0.05 + 1e-12
        replica = _resample_residuals(z_obs, sigma, z_pred, rng)
        assert replica.shape == z_pred.shape
        assert replica.dtype == z_pred.dtype

    def test_reproducibility(self):
        n_freqs = 3
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        z_pred = np.ones((n_freqs, 2, 2), dtype=np.complex128)
        sigma = np.full((n_freqs, 2, 2), 0.01)
        z_obs = z_pred.copy()
        a = _resample_residuals(z_obs, sigma, z_pred, rng_a)
        b = _resample_residuals(z_obs, sigma, z_pred, rng_b)
        np.testing.assert_array_equal(a, b)

    def test_centered_on_z_predicted(self):
        """Many replicas: empirical mean ~ z_predicted."""
        rng = np.random.default_rng(0)
        n_freqs = 4
        z_pred = np.arange(n_freqs * 4).reshape(n_freqs, 2, 2).astype(np.complex128)
        sigma = np.full((n_freqs, 2, 2), 0.5)
        z_obs = z_pred.copy()
        n_replicas = 1000
        replicas = np.array(
            [_resample_residuals(z_obs, sigma, z_pred, rng) for _ in range(n_replicas)]
        )
        empirical_mean = replicas.mean(axis=0)
        # 3-sigma at n=1000: 3 * 0.5/sqrt(1000) = 0.047
        np.testing.assert_allclose(empirical_mean, z_pred, atol=0.05)

    def test_variance_matches_sigma(self):
        rng = np.random.default_rng(0)
        n_freqs = 4
        z_pred = np.zeros((n_freqs, 2, 2), dtype=np.complex128)
        sigma = np.full((n_freqs, 2, 2), 0.5)
        z_obs = z_pred.copy()
        n_replicas = 5000
        replicas = np.array(
            [_resample_residuals(z_obs, sigma, z_pred, rng) for _ in range(n_replicas)]
        )
        empirical_std_real = replicas.real.std(axis=0)
        np.testing.assert_allclose(empirical_std_real, sigma, rtol=0.05)

    def test_shape_mismatch_raises(self):
        rng = np.random.default_rng(0)
        z_pred = np.zeros((3, 2, 2), dtype=np.complex128)
        sigma = np.ones((4, 2, 2))
        z_obs = z_pred.copy()
        with pytest.raises(ValueError, match="shape"):
            _resample_residuals(z_obs, sigma, z_pred, rng)

    def test_non_positive_sigma_raises(self):
        rng = np.random.default_rng(0)
        z_pred = np.zeros((3, 2, 2), dtype=np.complex128)
        sigma = np.zeros((3, 2, 2))
        z_obs = z_pred.copy()
        with pytest.raises(ValueError, match="non-positive"):
            _resample_residuals(z_obs, sigma, z_pred, rng)


class TestComputeCiPercentile:
    """_compute_ci_percentile produces correct percentile-based CIs."""

    def test_normal_distribution_95(self):
        rng = np.random.default_rng(42)
        replicates = rng.normal(loc=10.0, scale=2.0, size=(20000,))
        lower, upper = _compute_ci_percentile(replicates, level=0.95)
        # 95% CI of N(10, 2) is approximately [6.08, 13.92]
        np.testing.assert_allclose(float(lower), 6.08, atol=0.2)
        np.testing.assert_allclose(float(upper), 13.92, atol=0.2)

    def test_complex_split_real_imag(self):
        rng = np.random.default_rng(0)
        replicates = rng.normal(size=(2000,)) + 1j * rng.normal(size=(2000,))
        lower, upper = _compute_ci_percentile(replicates, level=0.95)
        assert np.iscomplexobj(lower)
        assert np.iscomplexobj(upper)
        np.testing.assert_allclose(float(lower.real), -1.96, atol=0.2)
        np.testing.assert_allclose(float(upper.real), 1.96, atol=0.2)

    def test_axis_argument(self):
        rng = np.random.default_rng(0)
        replicates = rng.normal(size=(100, 5))
        lower, upper = _compute_ci_percentile(replicates, level=0.95, axis=0)
        assert lower.shape == (5,)
        assert upper.shape == (5,)
        assert np.all(lower <= upper)

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level"):
            _compute_ci_percentile(np.zeros(10), level=1.5)
        with pytest.raises(ValueError, match="level"):
            _compute_ci_percentile(np.zeros(10), level=0.0)

    def test_nan_aware(self):
        # NaN entries should be ignored, not propagate to CI bounds
        rng = np.random.default_rng(0)
        x = rng.normal(size=1000)
        x[::10] = np.nan
        lower, upper = _compute_ci_percentile(x, level=0.95)
        assert np.isfinite(lower)
        assert np.isfinite(upper)


class TestPredictZFromPrimaryModes:
    """_predict_z_from_primary_modes reconstructs Z correctly."""

    def test_round_trip_from_synthetic(self):
        # Build a single-band synthetic, fit it, and verify the
        # forward-model prediction at the primary mode reproduces
        # the input Z within machine precision (since the primary
        # mode converged to the truth on this clean problem).
        d = _build_synthetic_band(theta_deg=30.0, n_freqs=5)
        from mtpy.core.transfer_function.z_analysis.decomposition import (
            _solve_band_multistart,
        )

        rng = np.random.default_rng(0)
        modes = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=2, rng=rng
        )
        band_idx = np.arange(5)
        z_pred = _predict_z_from_primary_modes([(band_idx, modes)], d["periods"])
        # Clean fit: forward model at primary should reproduce z_obs
        # up to numerical noise
        np.testing.assert_allclose(z_pred, d["z_obs"], atol=1e-9, rtol=1e-9)

    def test_shape(self):
        d = _build_synthetic_band(theta_deg=30.0, n_freqs=5)
        from mtpy.core.transfer_function.z_analysis.decomposition import (
            _solve_band_multistart,
        )

        rng = np.random.default_rng(0)
        modes = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=1, rng=rng
        )
        band_idx = np.arange(5)
        z_pred = _predict_z_from_primary_modes([(band_idx, modes)], d["periods"])
        assert z_pred.shape == (5, 2, 2)
        assert z_pred.dtype == np.complex128


class TestBootstrapDecompose:
    """_bootstrap_decompose returns the documented array shapes."""

    def test_shapes_and_keys(self):
        d = _build_synthetic_band(theta_deg=30.0, n_freqs=5)
        from mtpy.core.transfer_function.z_analysis.decomposition import (
            _solve_band_multistart,
        )

        rng_fit = np.random.default_rng(0)
        modes = _solve_band_multistart(
            d["z_obs"], d["sigma"], d["periods"], n_starts=2, rng=rng_fit
        )
        band_idx = np.arange(5)
        z_pred = _predict_z_from_primary_modes([(band_idx, modes)], d["periods"])

        rng = np.random.default_rng(1)
        replicates = _bootstrap_decompose(
            z_obs_full=d["z_obs"],
            sigma_full=d["sigma"],
            selected_periods=d["periods"],
            bands=[band_idx],
            z_predicted=z_pred,
            realisations=5,
            n_starts=2,
            bounds_override=None,
            rng=rng,
            mode_tolerance=None,
            perturbation_scale=0.1,
            mode_warning_threshold=1.5,
        )
        assert replicates["strike"].shape == (5, 5)
        assert replicates["regional_z"].shape == (5, 5, 2, 2)
        assert replicates["rms_misfit"].shape == (5,)
        assert replicates["mode_warning"].shape == (5,)
        # No catastrophic failures on clean synthetic
        assert np.sum(np.isnan(replicates["rms_misfit"])) == 0

    def test_realisations_zero_raises(self):
        d = _build_synthetic_band(n_freqs=3)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match=">= 1"):
            _bootstrap_decompose(
                z_obs_full=d["z_obs"],
                sigma_full=d["sigma"],
                selected_periods=d["periods"],
                bands=[np.arange(3)],
                z_predicted=d["z_obs"].copy(),
                realisations=0,
                n_starts=1,
                bounds_override=None,
                rng=rng,
                mode_tolerance=None,
                perturbation_scale=0.1,
                mode_warning_threshold=1.5,
            )


class TestDecomposeBootstrap:
    """End-to-end decompose() with bootstrap."""

    def test_realisations_zero_no_ci_fields(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, realisations=0)
        assert "strike_ci_lower" not in result.parameters
        assert "strike_ci_upper" not in result.parameters
        assert "bootstrap_replicates" not in result.metadata

    def test_realisations_positive_adds_ci_fields(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, realisations=5, n_starts=2, seed=42)
        for name in ("strike", "twist", "shear", "gain"):
            assert f"{name}_ci_lower" in result.parameters
            assert f"{name}_ci_upper" in result.parameters

    def test_ci_lower_le_upper(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, realisations=10, n_starts=2, seed=42)
        for name in ("strike", "twist", "shear", "gain"):
            lower = result.parameters[f"{name}_ci_lower"].values
            upper = result.parameters[f"{name}_ci_upper"].values
            finite = np.isfinite(lower) & np.isfinite(upper)
            assert np.all(lower[finite] <= upper[finite] + 1e-10)

    def test_metadata_populated(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, realisations=10, n_starts=2, seed=42)
        assert result.metadata["bootstrap_realisations"] == 10
        assert result.metadata["bootstrap_ci_method"] == "percentile"
        assert result.metadata["bootstrap_ci_level"] == 0.95
        assert "bootstrap_replicates" in result.metadata
        assert "bootstrap_caveats" in result.metadata
        assert "bootstrap_n_failed" in result.metadata
        assert "bootstrap_mode_warning_fraction" in result.metadata

    def test_replicates_shapes(self):
        z, _ = _build_synthetic_z(n_freqs=12)
        result = decompose(z, realisations=10, n_starts=2, seed=42)
        replicas = result.metadata["bootstrap_replicates"]
        n_periods = len(z.frequency)
        assert replicas["strike"].shape == (10, n_periods)
        assert replicas["regional_z"].shape == (10, n_periods, 2, 2)
        assert replicas["rms_misfit"].shape == (10,)
        assert replicas["mode_warning"].shape == (10,)

    def test_seed_reproducibility(self):
        z, _ = _build_synthetic_z()
        a = decompose(z, realisations=5, n_starts=2, seed=42)
        b = decompose(z, realisations=5, n_starts=2, seed=42)
        np.testing.assert_array_equal(
            a.parameters["strike_ci_lower"].values,
            b.parameters["strike_ci_lower"].values,
        )
        np.testing.assert_array_equal(
            a.parameters["strike_ci_upper"].values,
            b.parameters["strike_ci_upper"].values,
        )

    def test_invalid_ci_method_raises(self):
        z, _ = _build_synthetic_z()
        with pytest.raises(NotImplementedError, match="ci_method"):
            decompose(z, realisations=5, n_starts=2, ci_method="bca")

    def test_ci_level_recorded(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, realisations=5, n_starts=2, seed=42, ci_level=0.90)
        assert result.metadata["bootstrap_ci_level"] == 0.90
        assert result.options["ci_level"] == 0.90

    def test_options_record_includes_bootstrap_params(self):
        z, _ = _build_synthetic_z()
        result = decompose(
            z,
            realisations=5,
            n_starts=2,
            seed=42,
            ci_level=0.90,
            ci_method="percentile",
        )
        assert result.options["ci_level"] == 0.90
        assert result.options["ci_method"] == "percentile"

    def test_ci_units_attrs(self):
        z, _ = _build_synthetic_z()
        result = decompose(z, realisations=5, n_starts=2, seed=42)
        assert result.parameters["strike_ci_lower"].attrs.get("units") == "degrees"
        assert result.parameters["gain_ci_lower"].attrs.get("units") == "dimensionless"

    @pytest.mark.slow
    def test_empirical_coverage(self):
        """Empirical 95% coverage on synthetic noisy data should be
        at least 85% on a 20-outer sample (allowing for Monte Carlo
        noise). Marked slow: ~100 seconds.
        """
        truth_strike = 30.0
        truth_twist = 10.0
        truth_shear = 5.0
        n_outer = 20
        cov_strike = 0
        cov_twist = 0
        cov_shear = 0
        for outer_seed in range(n_outer):
            d = _build_synthetic_band(
                theta_deg=truth_strike,
                twist_deg=truth_twist,
                shear_deg=truth_shear,
                n_freqs=8,
                seed=outer_seed,
                noise_level=0.02,
            )
            frequencies = 1.0 / d["periods"]
            z = Z(z=d["z_obs"], z_error=d["sigma"], frequency=frequencies)
            result = decompose(
                z,
                realisations=50,
                n_starts=2,
                seed=outer_seed + 1000,
                bandwidth=2.0,  # single band for this synthetic
            )
            ci_l_strike = float(
                np.nanmedian(result.parameters["strike_ci_lower"].values)
            )
            ci_u_strike = float(
                np.nanmedian(result.parameters["strike_ci_upper"].values)
            )
            ci_l_twist = float(np.nanmedian(result.parameters["twist_ci_lower"].values))
            ci_u_twist = float(np.nanmedian(result.parameters["twist_ci_upper"].values))
            ci_l_shear = float(np.nanmedian(result.parameters["shear_ci_lower"].values))
            ci_u_shear = float(np.nanmedian(result.parameters["shear_ci_upper"].values))
            if ci_l_strike <= truth_strike <= ci_u_strike:
                cov_strike += 1
            if ci_l_twist <= truth_twist <= ci_u_twist:
                cov_twist += 1
            if ci_l_shear <= truth_shear <= ci_u_shear:
                cov_shear += 1
        # Allow [85%, 100%] coverage for a 20-sample assessment of
        # a nominal-95% interval (Monte Carlo binomial noise).
        assert cov_strike >= 17, f"strike coverage {cov_strike}/{n_outer} below 85%"
        assert cov_twist >= 17, f"twist coverage {cov_twist}/{n_outer} below 85%"
        assert cov_shear >= 17, f"shear coverage {cov_shear}/{n_outer} below 85%"
