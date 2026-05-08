"""Tests for ``DecompositionResult.alternate_branch``.

The Groom-Bailey decomposition has an exact discrete symmetry:
``(strike, twist, shear)`` and ``(strike + 90 mod 180, twist, -shear)``
fit the data identically. The :meth:`alternate_branch` method exposes
this by returning a new result on the alternate branch.

References
----------
Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    DecompositionResult,
    _estim_imp,
    decompose,
)


def _synthetic_z(
    theta_deg: float = 47.5,
    twist_deg: float = 8.0,
    shear_deg: float = 4.0,
    log10_gain: float = 0.1,
    n_freqs: int = 12,
    seed: int = 0,
) -> Z:
    """Noiseless synthetic ``Z`` used by the alternate-branch tests.

    Defaults pick a generic (non-canonical) strike so the GB 90-deg
    fold actually triggers and the alternate branch is genuinely
    distinct from the original.
    """
    rng = np.random.default_rng(seed)
    periods = np.logspace(-1.0, 2.0, n_freqs)
    log10_rho_a = np.full(n_freqs, 1.0) + 0.1 * rng.standard_normal(n_freqs)
    log10_rho_b = np.full(n_freqs, 2.0) + 0.1 * rng.standard_normal(n_freqs)
    phase_a = np.full(n_freqs, np.radians(60.0)) + 0.05 * rng.standard_normal(n_freqs)
    phase_b = np.full(n_freqs, np.radians(45.0)) + 0.05 * rng.standard_normal(n_freqs)

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


def _c_meas(strike_deg, twist_deg, shear_deg, gain) -> np.ndarray:
    """Reconstruct ``C_meas = g * R(strike) T(twist) S(shear) R(strike).T``.

    The C tensor is the GB-symmetry invariant; the two branches must
    produce algebraically identical C per period. Numerically the
    reconstructed entries differ by at most a ULP because the
    formula goes through different sin/cos calls on each branch.
    """
    s = np.radians(np.asarray(strike_deg, dtype=np.float64))
    t = np.radians(np.asarray(twist_deg, dtype=np.float64))
    e = np.radians(np.asarray(shear_deg, dtype=np.float64))
    g = np.asarray(gain, dtype=np.float64)
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
        out[k] = g[k] * R @ T @ S @ R.T
    return out


@pytest.fixture(scope="module")
def baseline_result() -> DecompositionResult:
    """A finished decomposition used by all tests in this module."""
    z = _synthetic_z()
    return decompose(z, seed=42)


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_alternate_branch_idempotent(baseline_result: DecompositionResult):
    """Applying ``alternate_branch`` twice round-trips to the original."""
    twice = baseline_result.alternate_branch().alternate_branch()

    # Parameters: every numeric variable matches the original to 1e-10.
    for var in (
        "strike",
        "twist",
        "shear",
        "gain",
        "anisotropy",
        "strike_error",
        "twist_error",
        "shear_error",
        "gain_error",
    ):
        np.testing.assert_allclose(
            twice.parameters[var].values,
            baseline_result.parameters[var].values,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"parameters[{var!r}] not idempotent",
        )

    # Regional Z: the swap is its own inverse, so two swaps reproduce
    # the original arrays exactly.
    np.testing.assert_array_equal(
        twice.regional_z.z, baseline_result.regional_z.z
    )
    np.testing.assert_array_equal(
        twice.regional_z.z_error, baseline_result.regional_z.z_error
    )

    # Misfit and chi-squared come straight through unchanged.
    assert twice.rms_misfit == baseline_result.rms_misfit
    np.testing.assert_array_equal(
        twice.chi_squared.values, baseline_result.chi_squared.values
    )


def test_alternate_branch_C_unchanged(baseline_result: DecompositionResult):
    """The galvanic-distortion C tensor is invariant under the symmetry.

    C is reconstructed from the GB parameters as
    ``g * R(strike) T(twist) S(shear) R(strike).T``. Algebraically
    the two branches give identical C; numerically the entries
    differ by at most one ULP because the formula evaluates
    different sin/cos arguments on each branch. ``np.array_equal``
    is therefore inappropriate; we use a 1e-12 tolerance to
    confirm the identity is upheld to numerical precision.
    """
    alt = baseline_result.alternate_branch()

    finite = np.isfinite(baseline_result.parameters["strike"].values)
    p0 = baseline_result.parameters
    p1 = alt.parameters
    c0 = _c_meas(
        p0["strike"].values[finite],
        p0["twist"].values[finite],
        p0["shear"].values[finite],
        p0["gain"].values[finite],
    )
    c1 = _c_meas(
        p1["strike"].values[finite],
        p1["twist"].values[finite],
        p1["shear"].values[finite],
        p1["gain"].values[finite],
    )
    np.testing.assert_allclose(c0, c1, atol=1e-12, rtol=0.0)


def test_alternate_branch_chi_squared_unchanged(
    baseline_result: DecompositionResult,
):
    """Both branches fit the data identically: chi-squared and RMS match."""
    alt = baseline_result.alternate_branch()
    assert alt.rms_misfit == baseline_result.rms_misfit
    np.testing.assert_array_equal(
        alt.chi_squared.values, baseline_result.chi_squared.values
    )
    # The chi_squared DataArray attrs (degrees_of_freedom, description)
    # are also pure provenance and survive the branch swap.
    assert alt.chi_squared.attrs == baseline_result.chi_squared.attrs


def test_alternate_branch_metadata_preserved(
    baseline_result: DecompositionResult,
):
    """Errors, uncertainties, and metadata pass through unchanged."""
    alt = baseline_result.alternate_branch()

    # Errors and any bootstrap CI fields are 1-sigma quantities or
    # percentile bounds in degrees; they describe sensitivity, not
    # the absolute value, so the symmetry preserves them.
    for var in (
        "strike_error",
        "twist_error",
        "shear_error",
        "gain_error",
        "anisotropy_error",
    ):
        np.testing.assert_array_equal(
            alt.parameters[var].values,
            baseline_result.parameters[var].values,
            err_msg=f"parameters[{var!r}] not preserved",
        )

    # Twist/gain/anisotropy are symmetry-invariant.
    for var in ("twist", "gain", "anisotropy"):
        np.testing.assert_array_equal(
            alt.parameters[var].values,
            baseline_result.parameters[var].values,
            err_msg=f"parameters[{var!r}] not preserved",
        )

    # method, options, metadata, frame come straight through.
    assert alt.method == baseline_result.method
    assert alt.options == baseline_result.options
    assert alt.metadata == baseline_result.metadata
    assert alt.frame == baseline_result.frame

    # And we did not mutate self.
    assert baseline_result.parameters["strike"].attrs.get("range") == "[0, 90)"


def test_alternate_branch_strike_in_range(baseline_result: DecompositionResult):
    """After ``alternate_branch``, strike values lie in [0, 180)."""
    alt = baseline_result.alternate_branch()
    strikes = alt.parameters["strike"].values
    finite = strikes[np.isfinite(strikes)]
    assert finite.size > 0
    assert np.all(finite >= 0.0 - 1e-12)
    assert np.all(finite < 180.0 + 1e-12)
    assert alt.parameters["strike"].attrs["range"] == "[0, 180)"

    # Sanity: the baseline used the geometric fold, so its strikes
    # were in [0, 90); the alternate branch must put each strike
    # exactly 90 deg away (mod 180).
    base = baseline_result.parameters["strike"].values
    base_finite = base[np.isfinite(base)]
    assert np.all(base_finite < 90.0 + 1e-9)
    np.testing.assert_allclose(
        finite, (base_finite + 90.0) % 180.0, atol=1e-12
    )
