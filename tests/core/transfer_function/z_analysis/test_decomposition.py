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
