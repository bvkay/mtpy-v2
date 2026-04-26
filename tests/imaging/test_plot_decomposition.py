"""Tests for mtpy.imaging.plot_decomposition.

Element-presence smoke tests for the six plot classes plus the
top-level summary function. No pixel-snapshot regression — tests
verify that figures are created with the expected axes, lines,
annotations, and structure.

Test fixtures build small DecompositionResult instances via the
existing decompose() / decompose_joint() machinery on synthetic Z.
"""

from __future__ import annotations

import matplotlib


matplotlib.use("Agg")  # No display backend in CI
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from mtpy.core.transfer_function.z import Z
from mtpy.core.transfer_function.z_analysis.decomposition import (
    decompose,
    decompose_joint,
)
from mtpy.imaging.plot_decomposition import (
    plot_decomposition_summary,
    PlotDecompositionApparentResistivity,
    PlotDecompositionBootstrap,
    PlotDecompositionChiSquared,
    PlotDecompositionModeLandscape,
    PlotDecompositionStrikeRose,
    PlotDecompositionTwistShear,
)


# ============================================================
# Synthetic data fixtures (mirroring patterns from test_decomposition.py)
# ============================================================


def _build_synthetic_z(
    theta_deg=30.0, twist_deg=10.0, shear_deg=5.0, n_freqs=12, seed=0
):
    """Build a synthetic Z spanning ~3 decades."""
    from mtpy.core.transfer_function.z_analysis.decomposition import _estim_imp

    rng = np.random.default_rng(seed)
    periods = np.logspace(-1, 2, n_freqs)
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
    t = np.tan(twist)
    e = np.tan(shear)

    z_obs = np.empty((n_freqs, 2, 2), dtype=np.complex128)
    for k in range(n_freqs):
        z_obs[k] = _estim_imp(a[k], b[k], t, e, theta)
    sigma = np.maximum(0.01 * np.abs(z_obs), 1e-12)
    return Z(z=z_obs, z_error=sigma, frequency=1.0 / periods)


def _make_simple_result(seed=0):
    z = _build_synthetic_z(seed=seed)
    return decompose(z, n_starts=2, seed=seed)


def _make_simple_result_with_bootstrap(seed=0):
    z = _build_synthetic_z(seed=seed)
    return decompose(z, n_starts=2, realisations=3, seed=seed)


def _make_simple_joint_result(n_stations=3, seed=0):
    class _MTLike:
        def __init__(self, station, z):
            self.station = station
            self.Z = z

    mts = [
        _MTLike(f"s{i}", _build_synthetic_z(seed=seed + i, n_freqs=8))
        for i in range(n_stations)
    ]
    # All synthetic stations share the same period grid (same n_freqs/range)
    return decompose_joint(mts, n_starts=2, seed=seed)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ============================================================
# Plot 1: Strike rose
# ============================================================


class TestPlotDecompositionStrikeRose:
    def test_returns_figure(self):
        result = _make_simple_result()
        fig = PlotDecompositionStrikeRose(result).plot()
        assert isinstance(fig, Figure)

    def test_uses_provided_axes(self):
        result = _make_simple_result()
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        returned = PlotDecompositionStrikeRose(result, ax=ax).plot()
        assert returned is fig

    def test_dict_input_with_station_id(self):
        results = {
            "STA001": _make_simple_result(seed=0),
            "STA002": _make_simple_result(seed=1),
        }
        fig = PlotDecompositionStrikeRose(results, station_id="STA002").plot()
        assert isinstance(fig, Figure)

    def test_dict_input_unknown_station_raises(self):
        results = {"STA001": _make_simple_result()}
        with pytest.raises(KeyError, match="not in results dict"):
            PlotDecompositionStrikeRose(results, station_id="MISSING").plot()

    def test_rms_annotation_present(self):
        result = _make_simple_result()
        fig = PlotDecompositionStrikeRose(result).plot()
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("RMS" in t for t in texts)

    def test_show_alternates_false(self):
        result = _make_simple_result()
        fig = PlotDecompositionStrikeRose(result, show_alternates=False).plot()
        # Smoke: figure created and bars present
        ax = fig.axes[0]
        assert len(ax.patches) > 0


# ============================================================
# Plot 2: Twist and shear
# ============================================================


class TestPlotDecompositionTwistShear:
    def test_returns_figure(self):
        result = _make_simple_result()
        fig = PlotDecompositionTwistShear(result).plot()
        assert isinstance(fig, Figure)

    def test_two_subplots(self):
        result = _make_simple_result()
        fig = PlotDecompositionTwistShear(result).plot()
        assert len(fig.axes) == 2

    def test_data_lines_present(self):
        result = _make_simple_result()
        fig = PlotDecompositionTwistShear(result).plot()
        # Twist and shear panels each plot at least one line
        assert any(len(ax.lines) >= 1 for ax in fig.axes)

    def test_with_bootstrap_uses_ci_bands(self):
        result = _make_simple_result_with_bootstrap()
        fig = PlotDecompositionTwistShear(result, use_bootstrap=True).plot()
        # fill_between adds collections (PolyCollection)
        any_filled = any(len(ax.collections) >= 1 for ax in fig.axes)
        assert any_filled

    def test_joint_result_overlays_stations(self):
        result = _make_simple_joint_result(n_stations=3)
        fig = PlotDecompositionTwistShear(result).plot()
        # Each panel should have one line per station
        ax_twist = fig.axes[0]
        # 3 stations + 0/horizontal lines (axhline doesn't add Line2D
        # to lines list in same way? actually it does) -> n_stations
        # plus 3 horizontal reference lines
        assert len(ax_twist.lines) >= 3


# ============================================================
# Plot 3: Apparent resistivity
# ============================================================


class TestPlotDecompositionApparentResistivity:
    def test_returns_figure(self):
        result = _make_simple_result()
        fig = PlotDecompositionApparentResistivity(result).plot()
        assert isinstance(fig, Figure)

    def test_two_subplots(self):
        result = _make_simple_result()
        fig = PlotDecompositionApparentResistivity(result).plot()
        assert len(fig.axes) == 2

    def test_te_and_tm_lines(self):
        result = _make_simple_result()
        fig = PlotDecompositionApparentResistivity(result).plot()
        ax_rho = fig.axes[0]
        # Two lines (TE and TM) on the rho panel
        assert len(ax_rho.lines) == 2

    def test_log_scale_x_and_y_on_rho(self):
        result = _make_simple_result()
        fig = PlotDecompositionApparentResistivity(result).plot()
        ax_rho = fig.axes[0]
        assert ax_rho.get_xscale() == "log"
        assert ax_rho.get_yscale() == "log"

    def test_joint_picks_first_station_when_no_id(self):
        result = _make_simple_joint_result(n_stations=2)
        fig = PlotDecompositionApparentResistivity(result).plot()
        assert isinstance(fig, Figure)


# ============================================================
# Plot 4: Chi-squared
# ============================================================


class TestPlotDecompositionChiSquared:
    def test_returns_figure(self):
        result = _make_simple_result()
        fig = PlotDecompositionChiSquared(result).plot()
        assert isinstance(fig, Figure)

    def test_regime_bands_present(self):
        result = _make_simple_result()
        fig = PlotDecompositionChiSquared(result).plot()
        ax = fig.axes[0]
        # axhspan adds Polygon patches (3 regime bands)
        assert len(ax.patches) >= 3

    def test_dof_reference_line(self):
        result = _make_simple_result()
        fig = PlotDecompositionChiSquared(result).plot()
        ax = fig.axes[0]
        # Legend should mention 'dof'
        legend = ax.get_legend()
        assert legend is not None
        assert any("dof" in t.get_text() for t in legend.get_texts())

    def test_log_x_log_y(self):
        result = _make_simple_result()
        fig = PlotDecompositionChiSquared(result).plot()
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"


# ============================================================
# Plot 5: Mode landscape
# ============================================================


class TestPlotDecompositionModeLandscape:
    def test_returns_figure(self):
        result = _make_simple_result()
        fig = PlotDecompositionModeLandscape(result).plot()
        assert isinstance(fig, Figure)

    def test_log_scales(self):
        result = _make_simple_result()
        fig = PlotDecompositionModeLandscape(result).plot()
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"

    def test_no_per_band_renders_placeholder(self):
        result = _make_simple_result()
        # Strip per_band metadata to test placeholder path
        result.metadata = {**result.metadata, "per_band": []}
        fig = PlotDecompositionModeLandscape(result).plot()
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert any("No per-band metadata" in t for t in texts)


# ============================================================
# Plot 6: Bootstrap distributions
# ============================================================


class TestPlotDecompositionBootstrap:
    def test_returns_figure_no_bootstrap(self):
        result = _make_simple_result()
        fig = PlotDecompositionBootstrap(result).plot()
        assert isinstance(fig, Figure)
        # All four placeholder axes show the no-bootstrap message
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        assert any("No bootstrap data" in t for t in texts)

    def test_returns_figure_with_bootstrap(self):
        result = _make_simple_result_with_bootstrap()
        fig = PlotDecompositionBootstrap(result).plot()
        assert isinstance(fig, Figure)
        # 2x2 grid → 4 axes
        assert len(fig.axes) == 4

    def test_period_indices_subset(self):
        result = _make_simple_result_with_bootstrap()
        fig = PlotDecompositionBootstrap(result, period_indices=[0, 1]).plot()
        assert isinstance(fig, Figure)


# ============================================================
# Top-level summary function
# ============================================================


class TestPlotDecompositionSummary:
    def test_returns_figure(self):
        result = _make_simple_result()
        fig = plot_decomposition_summary(result)
        assert isinstance(fig, Figure)

    def test_show_diagnostics_false_fewer_axes(self):
        result = _make_simple_result()
        fig_with = plot_decomposition_summary(result, show_diagnostics=True)
        fig_without = plot_decomposition_summary(result, show_diagnostics=False)
        assert len(fig_with.axes) > len(fig_without.axes)

    def test_save_path_writes_file(self, tmp_path):
        result = _make_simple_result()
        path = tmp_path / "summary.png"
        fig = plot_decomposition_summary(result, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_dict_input_with_station_id(self):
        results = {
            "STA1": _make_simple_result(seed=0),
            "STA2": _make_simple_result(seed=1),
        }
        fig = plot_decomposition_summary(results, station_id="STA2")
        assert isinstance(fig, Figure)
