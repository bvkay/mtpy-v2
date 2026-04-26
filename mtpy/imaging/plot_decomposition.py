"""Plotting for Groom-Bailey decomposition results.

Six plot classes plus a top-level convenience function. All plotting
takes :class:`mtpy.core.transfer_function.z_analysis.decomposition.DecompositionResult`
as input (or ``dict[station_id, DecompositionResult]`` for multi-
station overlays); plots never re-run decomposition.

Classes
-------
:class:`PlotDecompositionStrikeRose`
    Polar histogram of recovered strike across periods, with all
    discovered modes shown.
:class:`PlotDecompositionTwistShear`
    Twist and shear vs period.
:class:`PlotDecompositionApparentResistivity`
    Apparent resistivity and phase derived from regional_z.
:class:`PlotDecompositionChiSquared`
    Per-period chi-squared with fit-quality regime indicator.
:class:`PlotDecompositionModeLandscape`
    Per-band per-mode RMS comparison; surfaces multi-modal structure.
:class:`PlotDecompositionBootstrap`
    Per-parameter bootstrap distribution histograms.

Functions
---------
:func:`plot_decomposition_summary`
    Multi-panel figure with the four mandatory plots (strike rose,
    twist/shear, apparent resistivity, chi-squared) plus optional
    diagnostic rows.

Design notes
------------
The real-data exploration in ``~/MT_Decomp/findings/`` shaped these
designs:

- Per-station single-site is the default. Joint plotting is opt-in
  for users with curated common-grid data.
- ``primary_mode_warning`` fires on every real dataset, so multi-
  mode rendering is mandatory (strike rose shows all modes; mode
  landscape exists as a regular-use diagnostic).
- Bootstrap distributions are visualised as histograms, not
  symmetric CI bands, so users can see distribution shape.
- All plotting takes precomputed ``DecompositionResult`` and never
  re-runs decomposition (joint can take 18+ minutes on real data).
- RMS misfit is annotated on every plot; fit-quality regimes are
  colour-coded on the chi-squared plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from matplotlib.figure import Figure

from mtpy.core.transfer_function.z_analysis.decomposition import DecompositionResult


__all__ = [
    "PlotDecompositionStrikeRose",
    "PlotDecompositionTwistShear",
    "PlotDecompositionApparentResistivity",
    "PlotDecompositionChiSquared",
    "PlotDecompositionModeLandscape",
    "PlotDecompositionBootstrap",
    "plot_decomposition_summary",
]


DecompInput = Union[DecompositionResult, dict[str, DecompositionResult]]


def _is_dict_input(obj) -> bool:
    return isinstance(obj, dict)


def _select_result(
    obj: DecompInput, station_id: Optional[str] = None
) -> DecompositionResult:
    """Pick a single result for plots that take one station at a time."""
    if _is_dict_input(obj):
        if station_id is None:
            station_id = next(iter(obj))
            logger.warning(
                f"Multi-station input given without station_id; "
                f"plotting first station: {station_id!r}"
            )
        if station_id not in obj:
            raise KeyError(
                f"station_id={station_id!r} not in results dict; "
                f"available: {sorted(obj.keys())}"
            )
        return obj[station_id]
    return obj


def _rms_quality_color(rms: float) -> str:
    """Colour code for chi-squared / RMS regime indicators.

    Thresholds match the regimes observed in ~/MT_Decomp real-data
    exploration: BBMT data lands in 0.7-1.4, distorted Burra-style
    in 1.4-2.8, heavily-distorted in >2.8.
    """
    if rms < 1.4:
        return "#88cc88"  # green
    if rms < 2.8:
        return "#ddcc66"  # amber
    return "#cc6666"  # red


def _annotate_rms(ax, result, x=0.95, y=0.05, ha="right", va="bottom"):
    """Standard RMS + warning annotation for plots."""
    rms = result.rms_misfit
    text = f"RMS = {rms:.3g}"
    if result.metadata.get("primary_mode_warning", False):
        text += "\n(primary mode warning)"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=_rms_quality_color(rms),
            edgecolor="grey",
            alpha=0.85,
        ),
    )


def _ensure_polar_axes(ax, figsize):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure
    return fig, ax


def _ensure_axes(ax, figsize, n_subplots=1):
    if ax is None:
        fig, axes = plt.subplots(
            n_subplots, 1, figsize=figsize, sharex=(n_subplots > 1)
        )
        if n_subplots == 1:
            axes = [axes]
        return fig, list(axes)
    if isinstance(ax, (list, tuple, np.ndarray)):
        return ax[0].figure, list(ax)
    return ax.figure, [ax]


# ---------------------------------------------------------------------------
# Plot 1: Strike rose
# ---------------------------------------------------------------------------


class PlotDecompositionStrikeRose:
    """Polar histogram of recovered strike across periods.

    Primary mode is plotted in the foreground; alternate modes
    (when ``primary_mode_warning`` fires and ``show_alternates=True``)
    are overlaid in lighter shading. The 180-degree ambiguity inherent
    to strike is handled by reflecting each value (each strike
    plotted at θ and θ+180).

    Parameters
    ----------
    result : DecompositionResult or dict[str, DecompositionResult]
    station_id : str, optional
        Required for dict input.
    ax : polar matplotlib Axes, optional
    figsize : tuple, default (8, 8)
    n_bins : int, default 36
    show_alternates : bool, default True
    """

    def __init__(
        self,
        result,
        *,
        station_id=None,
        ax=None,
        figsize=(8, 8),
        n_bins=36,
        show_alternates=True,
    ):
        self.result = _select_result(result, station_id)
        self.ax = ax
        self.figsize = figsize
        self.n_bins = n_bins
        self.show_alternates = show_alternates
        self.fig: Optional[Figure] = None

    def plot(self) -> Figure:
        fig, ax = _ensure_polar_axes(self.ax, self.figsize)
        self.fig = fig

        strike_deg = np.asarray(self.result.parameters["strike"].values, dtype=float)
        # Strike Dataset shape may be (period,) or (period, mode);
        # take the primary (first mode).
        if strike_deg.ndim == 2:
            strike_deg = strike_deg[:, 0]
        valid = ~np.isnan(strike_deg)
        strike_deg = strike_deg[valid]

        if len(strike_deg) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid strike data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        bins = np.linspace(-np.pi, np.pi, self.n_bins + 1)
        # MT azimuth: clockwise from north. Convert to standard
        # mathematical angle: pi/2 - mt_az_rad. Reflect by adding pi
        # to account for 180-deg ambiguity.
        primary_rad = np.radians(90.0 - strike_deg)
        all_rad = np.concatenate([primary_rad, primary_rad + np.pi])
        wrapped = np.mod(all_rad + np.pi, 2 * np.pi) - np.pi
        counts, edges = np.histogram(wrapped, bins=bins)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_widths = np.diff(edges)
        ax.bar(
            bin_centers,
            counts,
            width=bin_widths,
            bottom=0,
            color="steelblue",
            alpha=0.75,
            edgecolor="white",
            label="primary mode",
        )

        warning = bool(self.result.metadata.get("primary_mode_warning", False))
        if self.show_alternates and warning:
            alt_strikes = []
            for band_info in self.result.metadata.get("per_band", []):
                for mode in band_info.get("modes", [])[1:]:
                    cf = mode.get("canonical_form", {})
                    if "strike_deg" in cf:
                        alt_strikes.append(float(cf["strike_deg"]))
            if alt_strikes:
                alt_rad = np.radians(90.0 - np.array(alt_strikes))
                alt_all = np.concatenate([alt_rad, alt_rad + np.pi])
                alt_wrapped = np.mod(alt_all + np.pi, 2 * np.pi) - np.pi
                alt_counts, _ = np.histogram(alt_wrapped, bins=bins)
                ax.bar(
                    bin_centers,
                    alt_counts,
                    width=bin_widths,
                    bottom=0,
                    color="lightcoral",
                    alpha=0.4,
                    edgecolor="white",
                    label="alternate modes",
                )

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

        _annotate_rms(ax, self.result)
        ax.set_title("Recovered regional strike")
        if (self.show_alternates and warning) or len(ax.patches) > self.n_bins:
            ax.legend(loc="upper right", fontsize=8)
        return fig


# ---------------------------------------------------------------------------
# Plot 2: Twist and shear vs period
# ---------------------------------------------------------------------------


class PlotDecompositionTwistShear:
    """Per-period twist (top) and shear (bottom) vs period.

    Error bars come from analytical Jacobian errors, or from
    bootstrap CI bounds when present and ``use_bootstrap=True``.
    Bound lines at ±π/4 (the parameterisation bound, ±45°) are
    drawn for visual reference.

    For joint results (``twist`` has a station dim) every station is
    overlaid by default.

    Parameters
    ----------
    result : DecompositionResult or dict
    station_id : str, optional
    ax : list of two Axes, optional
    figsize : tuple, default (8, 8)
    use_bootstrap : bool, default True
    """

    def __init__(
        self,
        result,
        *,
        station_id=None,
        ax=None,
        figsize=(8, 8),
        use_bootstrap=True,
    ):
        self.result = _select_result(result, station_id)
        self.ax = ax
        self.figsize = figsize
        self.use_bootstrap = use_bootstrap
        self.fig: Optional[Figure] = None

    def plot(self) -> Figure:
        fig, axes = _ensure_axes(self.ax, self.figsize, n_subplots=2)
        self.fig = fig
        ax_twist, ax_shear = axes[0], axes[1]

        params = self.result.parameters
        periods = np.asarray(params.coords["period"].values)
        joint = "station" in params["twist"].dims

        for ax_, name, ylabel in (
            (ax_twist, "twist", "Twist (deg)"),
            (ax_shear, "shear", "Shear (deg)"),
        ):
            values = np.asarray(params[name].values, dtype=float)
            err_name = f"{name}_error"
            err = (
                np.asarray(params[err_name].values, dtype=float)
                if err_name in params
                else None
            )
            ci_lo = (
                np.asarray(params[f"{name}_ci_lower"].values, dtype=float)
                if (self.use_bootstrap and f"{name}_ci_lower" in params)
                else None
            )
            ci_hi = (
                np.asarray(params[f"{name}_ci_upper"].values, dtype=float)
                if (self.use_bootstrap and f"{name}_ci_upper" in params)
                else None
            )
            if joint:
                station_ids = list(params.coords["station"].values)
                for i, sid in enumerate(station_ids):
                    v = values[:, i]
                    (line,) = ax_.plot(periods, v, marker="o", ms=3, label=str(sid))
                    color = line.get_color()
                    if ci_lo is not None and ci_hi is not None:
                        ax_.fill_between(
                            periods,
                            ci_lo[:, i],
                            ci_hi[:, i],
                            color=color,
                            alpha=0.18,
                        )
                    elif err is not None:
                        ax_.errorbar(
                            periods,
                            v,
                            yerr=err[:, i],
                            fmt="none",
                            color=color,
                            alpha=0.4,
                            capsize=2,
                        )
            else:
                ax_.plot(
                    periods,
                    values,
                    marker="o",
                    ms=3,
                    color="steelblue",
                )
                if ci_lo is not None and ci_hi is not None:
                    ax_.fill_between(
                        periods,
                        ci_lo,
                        ci_hi,
                        color="steelblue",
                        alpha=0.18,
                    )
                elif err is not None:
                    ax_.errorbar(
                        periods,
                        values,
                        yerr=err,
                        fmt="none",
                        color="steelblue",
                        alpha=0.4,
                        capsize=2,
                    )
            ax_.axhline(45, color="grey", ls=":", alpha=0.5)
            ax_.axhline(-45, color="grey", ls=":", alpha=0.5)
            ax_.axhline(0, color="grey", ls="-", alpha=0.2)
            ax_.set_xscale("log")
            ax_.set_ylabel(ylabel)
            ax_.grid(True, which="both", alpha=0.3)
        ax_shear.set_xlabel("Period (s)")
        if joint:
            ax_twist.legend(fontsize=7, loc="best", ncol=2)

        _annotate_rms(ax_twist, self.result, x=0.02, y=0.95, ha="left", va="top")
        ax_twist.set_title("Recovered distortion: twist and shear")
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Plot 3: Apparent resistivity and phase
# ---------------------------------------------------------------------------


class PlotDecompositionApparentResistivity:
    """Apparent resistivity (top, log scale) and phase (bottom) from
    the recovered regional Z.

    The xy and yx components of regional_z give the TE and TM modes
    respectively. Apparent resistivity follows the standard MT
    formula ``rho = period / (2*pi*mu0) * |Z|^2``.

    Parameters
    ----------
    result : DecompositionResult or dict
    station_id : str, optional
    ax : list of two Axes, optional
    figsize : tuple, default (8, 8)
    """

    def __init__(
        self,
        result,
        *,
        station_id=None,
        ax=None,
        figsize=(8, 8),
    ):
        self.result = _select_result(result, station_id)
        self.station_id = station_id
        self.ax = ax
        self.figsize = figsize
        self.fig: Optional[Figure] = None

    def plot(self) -> Figure:
        fig, axes = _ensure_axes(self.ax, self.figsize, n_subplots=2)
        self.fig = fig
        ax_rho, ax_phase = axes[0], axes[1]

        # Resolve regional_z (single Z or dict[station_id, Z])
        rz = self.result.regional_z
        if isinstance(rz, dict):
            sid = self.station_id if self.station_id is not None else next(iter(rz))
            z_obj = rz[sid]
        else:
            z_obj = rz

        z_arr = np.asarray(z_obj.z)
        freq = np.asarray(z_obj.frequency)
        periods = 1.0 / freq
        mu0 = 4.0 * np.pi * 1.0e-7
        factor = 2.0 * np.pi * mu0

        z_xy = z_arr[:, 0, 1]
        z_yx = z_arr[:, 1, 0]

        rho_xy = (np.abs(z_xy) ** 2) * periods / factor
        rho_yx = (np.abs(z_yx) ** 2) * periods / factor
        phase_xy = np.degrees(np.arctan2(z_xy.imag, z_xy.real))
        # yx phase conventionally rotated by 180 to align with xy
        phase_yx = np.degrees(np.arctan2(z_yx.imag, z_yx.real)) + 180.0

        ax_rho.loglog(periods, rho_xy, "o-", color="C0", ms=4, label="TE (xy)")
        ax_rho.loglog(periods, rho_yx, "s-", color="C3", ms=4, label="TM (yx)")
        ax_rho.set_ylabel(r"$\rho_a$ ($\Omega \cdot$m)")
        ax_rho.grid(True, which="both", alpha=0.3)
        ax_rho.legend(loc="best", fontsize=8)

        ax_phase.semilogx(periods, phase_xy, "o-", color="C0", ms=4)
        ax_phase.semilogx(periods, phase_yx, "s-", color="C3", ms=4)
        ax_phase.axhline(45, color="grey", ls=":", alpha=0.5)
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_xlabel("Period (s)")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.set_ylim(-10, 100)

        _annotate_rms(ax_rho, self.result, x=0.02, y=0.05, ha="left", va="bottom")
        ax_rho.set_title("Recovered regional apparent resistivity and phase")
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Plot 4: Chi-squared per period
# ---------------------------------------------------------------------------


class PlotDecompositionChiSquared:
    """Per-period chi-squared with fit-quality regime shading.

    Background bands indicate the three regimes observed in real
    data: green (RMS < 1.4, high quality), amber (RMS 1.4-2.8,
    moderate), red (RMS > 2.8, high misfit). Reference line at
    chi-squared = 8 (the degrees of freedom for GB, when RMS = 1).
    """

    def __init__(
        self,
        result,
        *,
        station_id=None,
        ax=None,
        figsize=(8, 6),
    ):
        self.result = _select_result(result, station_id)
        self.ax = ax
        self.figsize = figsize
        self.fig: Optional[Figure] = None

    def plot(self) -> Figure:
        fig, axes = _ensure_axes(self.ax, self.figsize, n_subplots=1)
        self.fig = fig
        ax = axes[0]

        chi = np.asarray(self.result.chi_squared.values, dtype=float)
        periods = np.asarray(self.result.chi_squared.coords["period"].values)
        dof = self.result.chi_squared.attrs.get("degrees_of_freedom", 8)

        # Regime shading (chi-squared thresholds = dof * RMS^2)
        ax.axhspan(0, dof * 1.96, color="#88cc88", alpha=0.18, label="RMS < 1.4")
        ax.axhspan(
            dof * 1.96,
            dof * 7.84,
            color="#ddcc66",
            alpha=0.18,
            label="RMS 1.4-2.8",
        )
        ax.axhspan(
            dof * 7.84,
            dof * 100.0,
            color="#cc6666",
            alpha=0.15,
            label="RMS > 2.8",
        )
        ax.axhline(dof, color="black", ls="--", alpha=0.5, label=f"dof = {dof}")
        ax.semilogx(periods, chi, "o-", color="black", ms=4)
        ax.set_xlabel("Period (s)")
        ax.set_ylabel(r"$\chi^2$")
        ax.set_yscale("log")
        ax.set_ylim(
            bottom=max(0.1, np.nanmin(chi) * 0.5) if np.any(np.isfinite(chi)) else 0.1
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

        _annotate_rms(ax, self.result, x=0.02, y=0.95, ha="left", va="top")
        ax.set_title("Per-period chi-squared")
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Plot 5: Mode landscape (diagnostic)
# ---------------------------------------------------------------------------


class PlotDecompositionModeLandscape:
    """Per-band per-mode RMS comparison.

    For each band, horizontal bars at the band's period range show
    each discovered mode's RMS. Bar heights scale with mode
    probability (taller = more probable). Multi-modal bands have
    multiple competing bars; unimodal bands have a single tall bar.
    """

    def __init__(
        self,
        result,
        *,
        station_id=None,
        ax=None,
        figsize=(10, 6),
    ):
        self.result = _select_result(result, station_id)
        self.ax = ax
        self.figsize = figsize
        self.fig: Optional[Figure] = None

    def plot(self) -> Figure:
        fig, axes = _ensure_axes(self.ax, self.figsize, n_subplots=1)
        self.fig = fig
        ax = axes[0]

        per_band = self.result.metadata.get("per_band", [])
        if not per_band:
            ax.text(
                0.5,
                0.5,
                "No per-band metadata",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        for band_info in per_band:
            band_periods = band_info.get("band_periods", [])
            if not band_periods:
                continue
            p_lo = float(min(band_periods))
            p_hi = float(max(band_periods))
            modes = band_info.get("modes", [])
            for mode_idx, mode in enumerate(modes):
                rms = float(mode.get("rms_misfit", float("nan")))
                prob = float(mode.get("probability") or 0.0)
                if not np.isfinite(rms):
                    continue
                # Bar drawn from rms-low to rms-high in y, spanning
                # band's period range in x; alpha encodes probability.
                color = "C0" if mode_idx == 0 else "C1"
                ax.fill_betweenx(
                    [rms * 0.95, rms * 1.05],
                    p_lo,
                    p_hi,
                    color=color,
                    alpha=0.25 + 0.7 * prob,
                    edgecolor=color,
                )
                ax.plot(
                    [(p_lo + p_hi) / 2.0],
                    [rms],
                    "o",
                    color=color,
                    ms=6 + 4 * prob,
                    label="primary" if mode_idx == 0 else "alternate",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Period (s)")
        ax.set_ylabel("Per-band RMS misfit")
        ax.grid(True, which="both", alpha=0.3)
        # De-duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        seen: dict[str, object] = {}
        for h, lbl in zip(handles, labels):
            seen.setdefault(lbl, h)
        if seen:
            ax.legend(seen.values(), seen.keys(), loc="best", fontsize=8)
        ax.set_title("Per-band mode landscape")
        _annotate_rms(ax, self.result, x=0.98, y=0.95, ha="right", va="top")
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Plot 6: Bootstrap distributions
# ---------------------------------------------------------------------------


class PlotDecompositionBootstrap:
    """Per-parameter bootstrap distribution histograms.

    Renders only when ``realisations > 0``. For each scalar parameter
    (strike, twist, shear, gain), a small-multiples grid shows
    histograms of the bootstrap replicate values across periods,
    revealing skew and multimodality that point-plus-CI hides.
    """

    PARAM_NAMES = ("strike", "twist", "shear", "gain")

    def __init__(
        self,
        result,
        *,
        station_id=None,
        ax=None,
        figsize=(10, 8),
        period_indices=None,
    ):
        self.result = _select_result(result, station_id)
        self.station_id = station_id
        self.ax = ax
        self.figsize = figsize
        self.period_indices = period_indices
        self.fig: Optional[Figure] = None

    def plot(self) -> Figure:
        if self.ax is None:
            fig, axes_arr = plt.subplots(2, 2, figsize=self.figsize)
            axes = axes_arr.ravel().tolist()
        else:
            fig = (
                self.ax[0].figure
                if isinstance(self.ax, (list, tuple))
                else self.ax.figure
            )
            axes = list(self.ax) if isinstance(self.ax, (list, tuple)) else [self.ax]
        self.fig = fig

        replicates = self.result.metadata.get("bootstrap_replicates")
        if replicates is None:
            for ax_ in axes:
                ax_.text(
                    0.5,
                    0.5,
                    "No bootstrap data\n(realisations=0)",
                    ha="center",
                    va="center",
                    transform=ax_.transAxes,
                    fontsize=10,
                )
                ax_.set_xticks([])
                ax_.set_yticks([])
            return fig

        # Pick representative period indices
        n_periods = replicates["strike"].shape[1]
        if self.period_indices is None:
            n_show = min(4, n_periods)
            chosen = np.linspace(0, n_periods - 1, n_show, dtype=int)
        else:
            chosen = list(self.period_indices)
        periods_coord = np.asarray(self.result.parameters.coords["period"].values)

        joint = replicates["twist"].ndim == 3
        # Plot one parameter per subplot
        for ax_, name in zip(axes, self.PARAM_NAMES):
            arr = np.asarray(replicates[name])
            for k in chosen:
                if joint and arr.ndim == 3:
                    # Multi-station; collapse station axis (overlay)
                    samples = arr[:, k, :].ravel()
                else:
                    samples = arr[:, k]
                samples = samples[np.isfinite(samples)]
                if samples.size == 0:
                    continue
                ax_.hist(
                    samples,
                    bins=20,
                    alpha=0.5,
                    label=f"{periods_coord[k]:.3g}s",
                )
            ax_.set_title(name)
            ax_.set_xlabel(name)
            ax_.set_ylabel("count")
            ax_.legend(fontsize=7)

        fig.suptitle(
            f"Bootstrap distributions ({replicates['strike'].shape[0]} reps)",
            fontsize=12,
        )
        if fig.get_layout_engine() is None:
            fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Top-level summary function
# ---------------------------------------------------------------------------


def plot_decomposition_summary(
    result: DecompInput,
    *,
    station_id: Optional[str] = None,
    figsize: tuple = (14, 12),
    show_diagnostics: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """Multi-panel summary figure.

    Includes the four mandatory plots (strike rose, twist/shear,
    apparent resistivity/phase, chi-squared). When
    ``show_diagnostics=True``, an additional row of mode landscape
    + bootstrap distributions is added (bootstrap row only when
    realisations > 0).

    Parameters
    ----------
    result : DecompositionResult or dict
    station_id : str, optional
    figsize : tuple, default (14, 12)
    show_diagnostics : bool, default True
    save_path : path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    selected = _select_result(result, station_id)
    has_bootstrap = selected.metadata.get("bootstrap_replicates") is not None
    n_extra = (1 if show_diagnostics else 0) + (
        1 if (show_diagnostics and has_bootstrap) else 0
    )

    # constrained_layout handles the polar+Cartesian axis combinations
    # cleanly. Per-plotter tight_layout calls are guarded to skip when
    # a layout engine is already set on the figure (i.e., when this
    # summary is the parent).
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    nrows = 2 + n_extra
    gs = fig.add_gridspec(nrows, 2, hspace=0.38, wspace=0.28)

    ax_rose = fig.add_subplot(gs[0, 0], projection="polar")
    ax_chi = fig.add_subplot(gs[0, 1])
    ax_twist = fig.add_subplot(gs[1, 0])
    ax_shear = fig.add_subplot(gs[1, 1])

    PlotDecompositionStrikeRose(selected, station_id=station_id, ax=ax_rose).plot()
    PlotDecompositionChiSquared(selected, station_id=station_id, ax=ax_chi).plot()
    # Twist+shear plotted into the two cells in row 1
    PlotDecompositionTwistShear(
        selected, station_id=station_id, ax=[ax_twist, ax_shear]
    ).plot()

    if show_diagnostics:
        # Diagnostic row: mode landscape + apparent rho/phase split
        ax_landscape = fig.add_subplot(gs[2, 0])
        ax_rho_phase = fig.add_subplot(gs[2, 1])
        PlotDecompositionModeLandscape(
            selected, station_id=station_id, ax=ax_landscape
        ).plot()
        # Apparent rho-phase as single panel (collapsed)
        # We render it on a sub-gridspec so rho and phase share the
        # cell vertically.
        sub_gs = gs[2, 1].subgridspec(2, 1, hspace=0.15)
        ax_rho = fig.add_subplot(sub_gs[0])
        ax_phase = fig.add_subplot(sub_gs[1], sharex=ax_rho)
        ax_rho_phase.set_visible(False)
        PlotDecompositionApparentResistivity(
            selected, station_id=station_id, ax=[ax_rho, ax_phase]
        ).plot()

        if has_bootstrap:
            ax_b1 = fig.add_subplot(gs[3, 0])
            ax_b2 = fig.add_subplot(gs[3, 1])
            ax_b3 = fig.add_subplot(gs[3, 0])
            ax_b4 = fig.add_subplot(gs[3, 1])
            # We need 4 axes for 2x2; use sub-gridspec across [3, :]
            ax_b1.set_visible(False)
            ax_b2.set_visible(False)
            ax_b3.set_visible(False)
            ax_b4.set_visible(False)
            sub_gs_b = gs[3, :].subgridspec(2, 2, hspace=0.45, wspace=0.25)
            bs_axes = [
                fig.add_subplot(sub_gs_b[0, 0]),
                fig.add_subplot(sub_gs_b[0, 1]),
                fig.add_subplot(sub_gs_b[1, 0]),
                fig.add_subplot(sub_gs_b[1, 1]),
            ]
            PlotDecompositionBootstrap(
                selected, station_id=station_id, ax=bs_axes
            ).plot()

    title = "GB decomposition summary"
    if station_id:
        title += f" — {station_id}"
    fig.suptitle(title, fontsize=14)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Decomposition summary saved to {save_path}")

    return fig
