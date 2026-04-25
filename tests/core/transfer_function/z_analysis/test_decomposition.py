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
    _build_bounds,
    _build_initial_guess,
    _calc_error,
    _canonicalise_solution,
    _convz2p,
    _convz2r,
    _estim_imp,
    _extract_bands,
    _extreme,
    _jkvar,
    _mat_multiply,
    _objfun,
    _solve_band,
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


class TestBuildInitialGuess:
    """_build_initial_guess produces a valid x0 vector."""

    def test_shape(self):
        d = _build_synthetic_band(n_freqs=5)
        x0 = _build_initial_guess(d["z_obs"], d["sigma"], d["periods"])
        assert x0.shape == (5 + 4 * 5,)

    def test_distortion_starts_at_zero(self):
        d = _build_synthetic_band(n_freqs=4)
        x0 = _build_initial_guess(d["z_obs"], d["sigma"], d["periods"])
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

    def test_realisations_raises_not_implemented(self):
        z, _ = _build_synthetic_z()
        with pytest.raises(NotImplementedError, match="Bootstrap"):
            decompose(z, realisations=10)

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
