"""Visualisation helpers for :mod:`...spatial_coherence` outputs.

Kept separate from the core :mod:`...spatial_coherence` module so
that the main API has *no* matplotlib dependency. This module is
imported only when a caller explicitly asks for a plot, and at
that point matplotlib is loaded on first use.

Usage
-----
::

    from mtpy.core.transfer_function.z_analysis.decomposition import (
        compute_coherence,
    )
    from mtpy.core.transfer_function.z_analysis.decomposition.spatial_coherence_plots import (
        plot_variogram,
    )

    coh = compute_coherence(table, "gamma_magnitude")
    fig, ax = plot_variogram(coh)

The matplotlib figure is returned for further customisation
(adding subplots, saving, etc.); we deliberately do *not* call
``plt.show`` ourselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover -- type-only imports
    from .results import CoherenceResult


__all__ = [
    "plot_coherence_summary",
    "plot_variogram",
]


def _label_color(label: str) -> str:
    return {
        "structured": "#1b7837",
        "weakly_structured": "#fdae61",
        "noise_dominated": "#9e0142",
        "insufficient_data": "#888888",
    }.get(label, "#444444")


def plot_variogram(
    coherence: "CoherenceResult",
    *,
    ax: Any = None,
    show_null: bool = True,
    show_bootstrap_reference: bool = True,
    show_geostat_summary: bool = True,
    title: str | None = None,
):
    """Publication-style variogram plot for one observable.

    Plots the empirical variogram in solid colour, the
    randomisation-null 5-50-95 band as a shaded grey envelope, and
    (when present) the bootstrap-variance reference as a horizontal
    dashed line. The estimated nugget, sill, and range are
    annotated.

    Parameters
    ----------
    coherence : CoherenceResult
        Output of :func:`...spatial_coherence.compute_coherence`.
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. ``None`` (default) creates a fresh
        figure.
    show_null : bool, default True
    show_bootstrap_reference : bool, default True
    show_geostat_summary : bool, default True
        Annotate ``nugget``, ``sill``, and ``range`` in the
        bottom-right.
    title : str, optional
        Axis title; defaults to the observable name plus the
        coherence label.

    Returns
    -------
    (fig, ax) : tuple
        The matplotlib figure and axis the data was drawn on.
    """
    import matplotlib.pyplot as plt  # local: keep optional dependency

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
    else:
        fig = ax.figure

    bin_centers = np.asarray(coherence.bin_centers_km, dtype=np.float64)
    gamma = np.asarray(coherence.variogram_values, dtype=np.float64)
    counts = np.asarray(coherence.bin_counts, dtype=np.int64)

    if show_null:
        p05 = np.asarray(coherence.null_p05, dtype=np.float64)
        p95 = np.asarray(coherence.null_p95, dtype=np.float64)
        finite = np.isfinite(p05) & np.isfinite(p95)
        if finite.any():
            ax.fill_between(
                bin_centers[finite], p05[finite], p95[finite],
                color="#cccccc", alpha=0.55,
                label="null 5-95 %", zorder=1,
            )
            ax.plot(
                bin_centers[finite], coherence.null_p50[finite],
                color="#777777", linestyle=":", linewidth=1.0,
                label="null median", zorder=2,
            )

    finite = np.isfinite(gamma) & (counts > 0)
    color = _label_color(coherence.coherence_label)
    if finite.any():
        ax.plot(
            bin_centers[finite], gamma[finite],
            marker="o", color=color, linewidth=1.6,
            label="empirical γ(h)", zorder=4,
        )

    if show_bootstrap_reference and coherence.bootstrap_variance is not None:
        ax.axhline(
            coherence.bootstrap_variance,
            linestyle="--", linewidth=1.0, color="#444444",
            label="bootstrap variance", zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Separation h (km)")
    if coherence.metadata.get("kind") == "log":
        ax.set_ylabel("Semivariance γ(h)  [log10 units]")
    elif coherence.metadata.get("kind") == "circular":
        ax.set_ylabel("Semivariance γ(h)  [1 - cos(2 Δθ)]")
    else:
        ax.set_ylabel("Semivariance γ(h)")

    if title is None:
        title = (
            f"{coherence.observable_name}  "
            f"({coherence.period_band_label}; "
            f"{coherence.coherence_label})"
        )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    if show_geostat_summary:
        nugget = coherence.nugget
        sill = coherence.sill
        range_km = coherence.range_km
        text_lines = [
            f"nugget = {nugget:.3g}",
            f"sill   = {sill:.3g}",
            f"range  = {range_km:.0f} km" if np.isfinite(range_km)
            else "range  = (none)",
            f"nugget/sill = {coherence.nugget_to_sill_ratio:.2f}",
        ]
        ax.text(
            0.98, 0.02,
            "\n".join(text_lines),
            transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom",
            family="monospace",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"),
        )

    return fig, ax


def plot_coherence_summary(
    coherence_results: dict[str, "CoherenceResult"],
    *,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
):
    """Grid of variogram plots — one per observable.

    Layout: ``ncols`` columns × ``ceil(n / ncols)`` rows. Useful as
    a one-figure summary of how *every* primary observable behaves
    spatially across the array.

    Parameters
    ----------
    coherence_results : dict[str, CoherenceResult]
        Output of :func:`...spatial_coherence.compute_coherence_all`
        (or any dict mapping observable name → result).
    ncols : int, default 2
    figsize : (width, height), optional
        Defaults to ``(7 * ncols, 3.5 * nrows)``.

    Returns
    -------
    (fig, axes) : tuple
    """
    import matplotlib.pyplot as plt  # local: keep optional dependency

    n = len(coherence_results)
    if n == 0:
        raise ValueError(
            "plot_coherence_summary: empty coherence_results dict"
        )
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (7.0 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    flat_axes = np.asarray(axes).reshape(-1)

    for ax_idx, (name, coh) in enumerate(coherence_results.items()):
        plot_variogram(coh, ax=flat_axes[ax_idx])
    # Blank axes for empty grid cells.
    for ax_idx in range(n, flat_axes.size):
        flat_axes[ax_idx].axis("off")
    fig.tight_layout()
    return fig, axes
