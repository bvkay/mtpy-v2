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
    _calc_error,
    _convz2p,
    _convz2r,
    _estim_imp,
    _extreme,
    _jkvar,
    _mat_multiply,
    decompose,
    decompose_joint,
    DecompositionResult,
)


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
    """The stub functions raise NotImplementedError with informative
    messages."""

    def test_decompose_raises(self, mt_with_impedance):
        with pytest.raises(NotImplementedError, match="subsequent"):
            decompose(mt_with_impedance.Z)

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
