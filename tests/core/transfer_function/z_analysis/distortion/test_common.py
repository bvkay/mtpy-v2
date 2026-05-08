"""Tests for the method-agnostic helpers in
:mod:`mtpy.core.transfer_function.z_analysis.decomposition.common`.

Covers the three plumbing helpers that were lifted out of
``groom_bailey.py`` because they are not GB-specific and are needed
by every decomposition method that does multi-start, bootstrap, or
parameter-bound perturbation:

* :func:`_resample_residuals` (parametric bootstrap noise generator)
* :func:`_compute_ci_percentile` (percentile-bootstrap CI builder)
* :func:`_perturbed_initial_guess` (bound-clipped Gaussian
  perturbation of an initial guess)
"""

from __future__ import annotations

import numpy as np
import pytest

from mtpy.core.transfer_function.z_analysis.decomposition.common import (
    _compute_ci_percentile,
    _perturbed_initial_guess,
    _resample_residuals,
)


# ---------------------------------------------------------------------------
# _resample_residuals
# ---------------------------------------------------------------------------


class TestResampleResiduals:
    def _inputs(self, n_freqs: int = 6):
        rng_data = np.random.default_rng(0)
        z_obs = (
            rng_data.standard_normal((n_freqs, 2, 2))
            + 1j * rng_data.standard_normal((n_freqs, 2, 2))
        ).astype(np.complex128)
        sigma = np.full((n_freqs, 2, 2), 0.1, dtype=np.float64)
        z_predicted = z_obs * 1.0  # arbitrary; bootstrap is around this
        return z_obs, sigma, z_predicted

    def test_shape_preserved(self):
        z_obs, sigma, z_predicted = self._inputs(n_freqs=8)
        rng = np.random.default_rng(42)
        replica = _resample_residuals(z_obs, sigma, z_predicted, rng)
        assert replica.shape == z_obs.shape
        assert replica.dtype == np.complex128

    def test_noise_distribution(self):
        """Many replicas: per-component (real, imag) residuals match
        N(0, sigma^2). The check is a coarse moment match — mean
        within 0.05 sigma, std within 5 % — so a fixed-seed run is
        deterministic.
        """
        n_freqs = 4
        sigma_scalar = 0.2
        z_predicted = np.zeros((n_freqs, 2, 2), dtype=np.complex128)
        z_obs = z_predicted.copy()
        sigma = np.full(z_predicted.shape, sigma_scalar, dtype=np.float64)

        rng = np.random.default_rng(123)
        n_replicas = 5000
        replicas = np.empty((n_replicas,) + z_predicted.shape, dtype=np.complex128)
        for k in range(n_replicas):
            replicas[k] = _resample_residuals(z_obs, sigma, z_predicted, rng)

        residuals_real = replicas.real - z_predicted.real
        residuals_imag = replicas.imag - z_predicted.imag
        assert abs(residuals_real.mean()) < 0.05 * sigma_scalar
        assert abs(residuals_imag.mean()) < 0.05 * sigma_scalar
        assert abs(residuals_real.std() - sigma_scalar) < 0.05 * sigma_scalar
        assert abs(residuals_imag.std() - sigma_scalar) < 0.05 * sigma_scalar

    def test_reproducible_under_fixed_seed(self):
        z_obs, sigma, z_predicted = self._inputs()
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        replica_a = _resample_residuals(z_obs, sigma, z_predicted, rng_a)
        replica_b = _resample_residuals(z_obs, sigma, z_predicted, rng_b)
        np.testing.assert_array_equal(replica_a, replica_b)

    def test_shape_mismatch_raises(self):
        z_obs, sigma, z_predicted = self._inputs(n_freqs=6)
        sigma_wrong = np.full((4, 2, 2), 0.1, dtype=np.float64)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="shape mismatch"):
            _resample_residuals(z_obs, sigma_wrong, z_predicted, rng)

    def test_nonpositive_sigma_raises(self):
        z_obs, sigma, z_predicted = self._inputs(n_freqs=4)
        sigma[0, 0, 0] = 0.0
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="non-positive"):
            _resample_residuals(z_obs, sigma, z_predicted, rng)


# ---------------------------------------------------------------------------
# _compute_ci_percentile
# ---------------------------------------------------------------------------


class TestComputeCiPercentile:
    def test_matches_nanpercentile_real(self):
        rng = np.random.default_rng(0)
        replicates = rng.standard_normal((500, 4))
        lower, upper = _compute_ci_percentile(replicates, level=0.95, axis=0)
        np.testing.assert_allclose(
            lower, np.nanpercentile(replicates, 2.5, axis=0)
        )
        np.testing.assert_allclose(
            upper, np.nanpercentile(replicates, 97.5, axis=0)
        )

    def test_complex_real_and_imag_independent(self):
        rng = np.random.default_rng(1)
        n = 400
        replicates = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        lower, upper = _compute_ci_percentile(replicates, level=0.9, axis=0)
        np.testing.assert_allclose(
            lower.real, np.nanpercentile(replicates.real, 5.0, axis=0)
        )
        np.testing.assert_allclose(
            upper.imag, np.nanpercentile(replicates.imag, 95.0, axis=0)
        )

    def test_nan_aware(self):
        replicates = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
        lower, upper = _compute_ci_percentile(replicates, level=0.6, axis=0)
        # Equivalent to nanpercentile on the cleaned array.
        clean = replicates[~np.isnan(replicates)]
        assert np.isclose(lower, np.percentile(clean, 20.0))
        assert np.isclose(upper, np.percentile(clean, 80.0))

    def test_axis_respected(self):
        rng = np.random.default_rng(2)
        replicates = rng.standard_normal((100, 3, 5))
        lower_axis0, _ = _compute_ci_percentile(replicates, level=0.9, axis=0)
        lower_axis1, _ = _compute_ci_percentile(replicates, level=0.9, axis=1)
        assert lower_axis0.shape == (3, 5)
        assert lower_axis1.shape == (100, 5)

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="level must be in"):
            _compute_ci_percentile(np.array([1.0, 2.0, 3.0]), level=1.0)
        with pytest.raises(ValueError, match="level must be in"):
            _compute_ci_percentile(np.array([1.0, 2.0, 3.0]), level=0.0)


# ---------------------------------------------------------------------------
# _perturbed_initial_guess
# ---------------------------------------------------------------------------


class TestPerturbedInitialGuess:
    def test_within_bounds(self):
        canonical = np.array([0.5, 1.5, -0.5])
        lower = np.array([0.0, 0.0, -1.0])
        upper = np.array([1.0, 3.0, 1.0])
        rng = np.random.default_rng(42)
        for _ in range(50):
            x = _perturbed_initial_guess(canonical, lower, upper, rng)
            assert np.all(x >= lower - 1e-12)
            assert np.all(x <= upper + 1e-12)

    def test_perturbation_scales_with_bound_width(self):
        """Larger bound widths produce proportionally larger
        perturbation magnitudes (in expectation).

        We compare two parameters with different bound widths and
        check the larger-width parameter has noticeably larger
        spread across many draws.
        """
        canonical = np.array([0.0, 0.0])
        lower = np.array([-0.1, -10.0])
        upper = np.array([0.1, 10.0])
        rng = np.random.default_rng(0)
        n = 2000
        samples = np.empty((n, 2))
        for i in range(n):
            samples[i] = _perturbed_initial_guess(
                canonical, lower, upper, rng, perturbation_scale=0.1
            )
        # Param 1 has 100x wider bounds than param 0; clipped
        # std should be much larger.
        assert samples[:, 1].std() > 10.0 * samples[:, 0].std()

    def test_reproducible_under_fixed_seed(self):
        canonical = np.array([0.0, 1.0, -1.0, 2.5])
        lower = np.array([-1.0, 0.0, -2.0, 1.0])
        upper = np.array([1.0, 2.0, 0.0, 4.0])
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        x_a = _perturbed_initial_guess(canonical, lower, upper, rng_a)
        x_b = _perturbed_initial_guess(canonical, lower, upper, rng_b)
        np.testing.assert_array_equal(x_a, x_b)

    def test_zero_perturbation_returns_clipped_canonical(self):
        canonical = np.array([0.5, 2.0, -0.5])
        lower = np.array([0.0, 0.0, -1.0])
        upper = np.array([1.0, 3.0, 1.0])
        rng = np.random.default_rng(0)
        x = _perturbed_initial_guess(
            canonical, lower, upper, rng, perturbation_scale=0.0
        )
        np.testing.assert_array_equal(x, canonical)
