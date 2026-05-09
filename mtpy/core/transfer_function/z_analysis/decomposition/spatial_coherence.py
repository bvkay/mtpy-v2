"""Spatial coherence of distortion observables across an MT array.

Quantifies whether each per-site, per-band observable produced by
:mod:`...continental_observables` is *geographically structured*
versus per-site noise. This is the empirical foundation for the
Paper 1 "distortion is signal, not nuisance" claim: an observable
whose values vary smoothly across an MT array carries information
about the underlying conductivity heterogeneity field; an
observable whose values are uncorrelated across nearby sites is
dominated by per-site noise (instrumentation, processing, or
optimiser-driven variance) and cannot support continental-scale
phenomenology.

Three diagnostics combine into a single classification per
observable:

1. **Empirical variogram**
   ``γ(h) = (1/2) · mean[(z_i - z_j)²]``
   over all site pairs at separation ``h``. The variogram rises
   from a small value at ``h ≈ 0`` (the *nugget*) toward a plateau
   at large ``h`` (the *sill*); the distance at which it reaches
   half the sill is the *range*, the spatial autocorrelation length.
2. **Randomisation null**
   Shuffle the observable values across sites and recompute the
   variogram. Repeat 100×. The 95th-percentile envelope is the
   noise-only-null bound; empirical variogram values above this
   envelope are *significantly* spatially structured.
3. **Nugget vs noise**
   The variogram nugget is *the* small-distance variance: it
   absorbs both pure measurement noise and any sub-grid spatial
   structure. Compared against an independent per-site noise
   estimate (bootstrap CI when available, inter-band variance as
   a fallback), the nugget tells us whether close-by-site
   disagreement comes from genuine sub-grid heterogeneity or from
   the per-site noise floor.

Per-observable special handling
-------------------------------
* **Magnitudes** (``gamma_magnitude``, ``C_minus_I_F``,
  ``GB_rms_misfit``, …): variogram on ``log10(value)`` so
  multiplicative scaling is captured additively. Non-positive
  values are filtered out.
* **Line-direction angles** (``C_strike_deg``,
  ``gamma_principal_axis_deg``, ``PT_alpha_deg``): circular
  variogram with squared difference replaced by
  ``1 - cos(2 (θ_i - θ_j))`` (in radians). The factor of 2 inside
  the cosine respects the 180°-periodicity of line directions: a
  5°/175° pair contributes ``≈ 1 - cos(2 · 10°) ≈ 0.06``
  (small, ≈ "10° apart"), not the ``≈ 1 - cos(340°)``
  ("close to opposite") that the 360°-periodic form would give.
* **Linear angles** (``C_twist_deg``, ``C_shear_deg``,
  ``PT_beta_deg``, …) and other linear observables: standard
  squared-difference variogram.
* **Ordinal flags** (``magnetic_distortion_flag``,
  ``Lilley_category``, ``WALDIM_case``,
  ``dimensionality_concordant``, ``GB_mode_warning``): converted
  to integer codes per the maps in :data:`_FLAG_ORDINAL_MAP` /
  :data:`_LILLEY_ORDINAL_MAP`. Variograms on ordinal data are
  interpretable but care should be taken with absolute scale —
  the bin distances are conventional, not natural.

Bin layout
----------
Default 20 logarithmically spaced bins from 50 km to 2000 km,
matching the AusLAMP characteristic separations. Site-pairs at
< 50 km are pooled into a single "near-neighbour" bin appended at
the bottom (so the actual returned bin count is 21). Pass
``n_bins`` / ``max_distance_km`` to override.

Distance metric
---------------
Great-circle haversine. Sufficient to mm-level accuracy at the
AusLAMP continental scale (~3500 km maximum separation); for
global-scale data a more accurate geodesic (Vincenty,
Karney 2013) would be needed.

References
----------
Cressie, N. A. C. (1993). *Statistics for Spatial Data*. Wiley.
Section 2.4 (variograms), 2.6 (nugget / sill / range).

Schneider, P., et al. (2002). Detection of shear due to weak
lensing by large-scale structure. *Astronomy & Astrophysics* 396,
1-19. (For the spin-2 / E-mode-B-mode formalism on a flat array;
the variogram is one slice of that decomposition.)

See Also
--------
:mod:`...continental_observables` : the per-site, per-band table
    that this module consumes.
:mod:`...spatial_coherence_plots` : visualisation helpers
    (matplotlib-based) for variograms with the null bound and the
    bootstrap reference line.
"""

from __future__ import annotations

import datetime as _datetime
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from .results import CoherenceResult

if TYPE_CHECKING:  # pragma: no cover -- type-only imports
    pass


__all__ = [
    "ANGULAR_LINE_OBSERVABLES",
    "DEFAULT_BIN_EDGES_KM",
    "MAGNITUDE_OBSERVABLES",
    "ORDINAL_OBSERVABLES",
    "PRIMARY_OBSERVABLES",
    "Variogram",
    "compute_coherence",
    "compute_coherence_all",
    "default_bin_edges_km",
    "empirical_variogram",
    "haversine_distances_km",
    "pairwise_distances",
    "randomisation_null",
]


# ---------------------------------------------------------------------------
# Per-observable type tables
# ---------------------------------------------------------------------------


ANGULAR_LINE_OBSERVABLES: set[str] = {
    "C_strike_deg",
    "gamma_principal_axis_deg",
    "PT_alpha_deg",
}
"""Observables that are line directions (mod 180°). Their
variograms use the circular metric ``1 - cos(2 Δθ)``.
"""

MAGNITUDE_OBSERVABLES: set[str] = {
    "gamma_magnitude",
    "C_minus_I_F",
    "GB_rms_misfit",
    "MJ_rms_misfit",
    "PT_lambda_max",
    "PT_lambda_min",
    "tipper_max_amplitude",
}
"""Strictly-positive magnitudes; variograms operate on
``log10(value)``.
"""

ORDINAL_OBSERVABLES: dict[str, dict[Any, int]] = {
    "magnetic_distortion_flag": {
        "low_risk": 0,
        "moderate_risk": 1,
        "high_risk": 2,
        "indeterminate": -1,
    },
    "Lilley_category": {
        "1D": 0,
        "2D": 1,
        "3D-2D": 2,
        "3D": 3,
        "indeterminate": -1,
    },
}
"""Categorical observables converted to integer codes for the
variogram. Negative codes (``"indeterminate"``) are filtered out.
"""

PRIMARY_OBSERVABLES: list[str] = [
    "discordance_deg",
    "gamma_magnitude",
    "gamma_principal_axis_deg",
    "C_minus_I_F",
    "C_strike_deg",
    "PT_alpha_deg",
    "PT_abs_beta_deg",
    "PT_ellipticity",
]
"""The eight observables :func:`compute_coherence_all` runs by
default. Hand-picked as the most informative per-site outputs of
the continental pipeline; pass ``observables=`` to override.
"""


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


_EARTH_RADIUS_KM = 6371.0088  # IUGG mean radius


def haversine_distances_km(
    lats_deg: np.ndarray, lons_deg: np.ndarray
) -> np.ndarray:
    """All pairwise great-circle distances among the given sites.

    Parameters
    ----------
    lats_deg, lons_deg : ndarray, shape ``(n,)``
        Site coordinates in degrees.

    Returns
    -------
    ndarray, shape ``(n*(n-1)/2,)``
        Pair distances in kilometres, in the order returned by
        ``np.triu_indices(n, k=1)``: ``(0,1), (0,2), ..., (n-2,
        n-1)``.

    Notes
    -----
    Uses the haversine formula with the IUGG mean Earth radius
    ``R = 6371.0088 km``. Accuracy is ≈ 0.5 % over continental
    scales; sub-metre at all separations relevant to AusLAMP.
    For global-scale data the Karney 2013 / Vincenty geodesic
    would be needed.
    """
    lat = np.radians(np.asarray(lats_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lons_deg, dtype=np.float64))
    n = lat.size
    iu, ju = np.triu_indices(n, k=1)
    dlat = lat[ju] - lat[iu]
    dlon = lon[ju] - lon[iu]
    a = (
        np.sin(0.5 * dlat) ** 2
        + np.cos(lat[iu]) * np.cos(lat[ju]) * np.sin(0.5 * dlon) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    return _EARTH_RADIUS_KM * c


def pairwise_distances(observable_table) -> np.ndarray:
    """Pairwise great-circle distances for all sites in the table.

    The table is expected to follow the
    :class:`...results.ObservableTable` schema (one row per
    ``(site, band)``). We deduplicate by ``site_id`` to get one
    coordinate per site, then haversine.

    Returns
    -------
    ndarray, shape ``(n_sites*(n_sites-1)/2,)``
    """
    df = observable_table.dataframe
    sites = (
        df[["site_id", "longitude_deg", "latitude_deg"]]
        .drop_duplicates(subset=["site_id"])
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    return haversine_distances_km(
        sites["latitude_deg"].to_numpy(dtype=np.float64),
        sites["longitude_deg"].to_numpy(dtype=np.float64),
    )


def default_bin_edges_km(
    n_bins: int = 20,
    max_distance_km: float = 2000.0,
    near_threshold_km: float = 50.0,
) -> np.ndarray:
    """Default variogram bin edges.

    Returns ``n_bins + 1`` log-spaced edges from
    ``near_threshold_km`` to ``max_distance_km``, plus an extra
    leading edge at 0 km to capture the "near-neighbour" pool of
    pairs at < ``near_threshold_km``. The first bin therefore covers
    ``[0, near_threshold_km]``; the remaining ``n_bins`` cover
    ``[near_threshold_km, max_distance_km]`` log-spaced.
    """
    log_edges = np.logspace(
        np.log10(near_threshold_km),
        np.log10(max_distance_km),
        n_bins + 1,
    )
    return np.concatenate([[0.0], log_edges])


DEFAULT_BIN_EDGES_KM: np.ndarray = default_bin_edges_km()


# ---------------------------------------------------------------------------
# Per-observable extraction and squared-difference helpers
# ---------------------------------------------------------------------------


def _classify_observable(name: str) -> str:
    if name in ANGULAR_LINE_OBSERVABLES:
        return "circular"
    if name in MAGNITUDE_OBSERVABLES:
        return "log"
    if name in ORDINAL_OBSERVABLES:
        return "ordinal"
    return "linear"


def _extract_values(df, name: str) -> tuple[np.ndarray, str]:
    """Return ``(values, kind)``.

    ``values`` is a float ndarray of length ``len(df)``; ``NaN`` is
    used for any row whose original value is missing or non-coercible.
    """
    kind = _classify_observable(name)
    if kind == "ordinal":
        mapping = ORDINAL_OBSERVABLES[name]
        out = np.full(len(df), np.nan)
        col = df[name]
        for i, v in enumerate(col):
            if v is None:
                continue
            try:
                key = str(v) if not isinstance(v, str) else v
            except Exception:
                continue
            if key in mapping:
                code = mapping[key]
                out[i] = float(code) if code >= 0 else np.nan
        return out, "ordinal"
    # All numeric kinds: rely on pandas coercion.
    raw = df[name]
    try:
        v = raw.astype(np.float64).to_numpy()
    except (TypeError, ValueError):
        # Some pandas dtypes (e.g. object holding ints) refuse
        # direct astype; fall back to numeric coercion.
        import pandas as pd

        v = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
    if kind == "log":
        v = np.where(v > 0, v, np.nan)
    return v, kind


def _squared_diff(v_i: np.ndarray, v_j: np.ndarray, kind: str) -> np.ndarray:
    """Per-pair squared difference for the variogram.

    Returns the per-pair quantity that, halved-and-averaged, gives
    the semivariance. For circular angles this is
    ``1 - cos(2 (θ_i - θ_j))`` (range ``[0, 2]``); for linear and
    ordinal observables the squared linear difference; for log
    magnitudes the squared difference of ``log10(value)``.
    """
    if kind == "circular":
        diff_rad = np.radians(v_i - v_j)
        return 1.0 - np.cos(2.0 * diff_rad)
    if kind == "log":
        with np.errstate(invalid="ignore"):
            l_i = np.log10(np.abs(v_i))
            l_j = np.log10(np.abs(v_j))
        return (l_i - l_j) ** 2
    # linear / ordinal share the standard form.
    return (v_i - v_j) ** 2


# ---------------------------------------------------------------------------
# Empirical variogram
# ---------------------------------------------------------------------------


class Variogram:
    """Lightweight container for a single variogram computation.

    A struct-shaped class (rather than a dataclass) so we can
    expose convenience read-only properties without introducing a
    new module-level dependency on, say, ``attrs``.
    """

    __slots__ = (
        "bin_edges_km", "bin_centers_km", "bin_counts",
        "variogram_values", "n_pairs_total", "kind",
    )

    def __init__(
        self,
        *,
        bin_edges_km: np.ndarray,
        bin_centers_km: np.ndarray,
        bin_counts: np.ndarray,
        variogram_values: np.ndarray,
        n_pairs_total: int,
        kind: str,
    ):
        self.bin_edges_km = np.asarray(bin_edges_km, dtype=np.float64)
        self.bin_centers_km = np.asarray(bin_centers_km, dtype=np.float64)
        self.bin_counts = np.asarray(bin_counts, dtype=np.int64)
        self.variogram_values = np.asarray(variogram_values, dtype=np.float64)
        self.n_pairs_total = int(n_pairs_total)
        self.kind = str(kind)

    def __repr__(self) -> str:  # pragma: no cover -- repr only
        return (
            f"Variogram(kind={self.kind!r}, n_bins={self.bin_centers_km.size},"
            f" n_pairs={self.n_pairs_total})"
        )


def _compute_variogram_arrays(
    sq_diff: np.ndarray,
    bin_idx: np.ndarray,
    valid: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin-mean of the half-squared-difference: γ(h) per bin."""
    sd = np.where(valid, sq_diff, 0.0)
    sums = np.bincount(
        np.where(valid, bin_idx, n_bins), weights=sd, minlength=n_bins + 1
    )[:n_bins]
    counts = np.bincount(
        np.where(valid, bin_idx, n_bins), minlength=n_bins + 1
    )[:n_bins]
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sums / counts
    gamma = np.where(counts > 0, gamma, np.nan)
    return gamma, counts


def _site_index_lookup(observable_table) -> tuple[np.ndarray, np.ndarray]:
    """Map ``(site_id -> per-site lat/lon)`` ordered by sorted site_id.

    Returns the unique-sorted ``site_ids`` and a parallel
    coordinate array of shape ``(n_sites, 2)`` columns ``[lat, lon]``.
    """
    df = observable_table.dataframe
    sites = (
        df[["site_id", "longitude_deg", "latitude_deg"]]
        .drop_duplicates(subset=["site_id"])
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    sids = sites["site_id"].to_numpy()
    coords = np.column_stack(
        [
            sites["latitude_deg"].to_numpy(dtype=np.float64),
            sites["longitude_deg"].to_numpy(dtype=np.float64),
        ]
    )
    return sids, coords


def _per_site_values_for_band(
    observable_table,
    observable_name: str,
    period_band_label: str | None,
) -> tuple[np.ndarray, str, str]:
    """Pull a per-site value array for a single band.

    Returns ``(values, kind, band_label_used)``. ``values`` is in
    the same order as :func:`_site_index_lookup`'s site list.
    """
    df = observable_table.dataframe
    bands = list(df["period_band_label"].dropna().unique())
    if period_band_label is None:
        if not bands:
            raise ValueError(
                "spatial_coherence: empty observable table; cannot "
                "compute variograms."
            )
        # Default: the band closest to 100 s (a reasonable AusLAMP
        # mid-range), or the first available.
        try:
            target = min(
                bands,
                key=lambda b: abs(
                    df.loc[df["period_band_label"] == b,
                           "period_band_geomean_s"].iloc[0] - 100.0
                ),
            )
        except (IndexError, KeyError):
            target = bands[0]
        band_used = target
    else:
        if period_band_label not in bands:
            raise ValueError(
                f"spatial_coherence: period_band_label="
                f"{period_band_label!r} not in table; available "
                f"bands: {bands}"
            )
        band_used = period_band_label

    df_b = df[df["period_band_label"] == band_used]
    if observable_name not in df.columns:
        raise ValueError(
            f"spatial_coherence: observable {observable_name!r} not "
            f"in table columns {list(df.columns)}"
        )

    # Re-sort to match _site_index_lookup ordering.
    sids_lookup, _ = _site_index_lookup(observable_table)
    df_indexed = df_b.set_index("site_id").reindex(sids_lookup)

    values, kind = _extract_values(df_indexed.reset_index(), observable_name)
    return values, kind, band_used


def empirical_variogram(
    observable_table,
    observable_name: str,
    *,
    period_band_label: str | None = None,
    n_bins: int = 20,
    max_distance_km: float = 2000.0,
    near_threshold_km: float = 50.0,
) -> Variogram:
    """Compute the empirical semivariogram for one observable.

    Parameters
    ----------
    observable_table : ObservableTable
        Source long-format table.
    observable_name : str
        Column name. Special handling per the module-level type
        tables.
    period_band_label : str, optional
        Restrict to one band. Defaults to the band whose
        ``period_band_geomean_s`` is nearest 100 s.
    n_bins : int, default 20
        Number of *log-spaced* bins between ``near_threshold_km``
        and ``max_distance_km``. A leading bin from 0 to
        ``near_threshold_km`` is added automatically, so the
        returned :attr:`Variogram.bin_centers_km` has length
        ``n_bins + 1``.
    max_distance_km : float, default 2000.0
    near_threshold_km : float, default 50.0
        Pairs at < ``near_threshold_km`` go in the bottom bin.

    Returns
    -------
    Variogram
    """
    values, kind, band_used = _per_site_values_for_band(
        observable_table, observable_name, period_band_label
    )
    sids, coords = _site_index_lookup(observable_table)
    if values.size != sids.size:  # pragma: no cover -- sanity guard
        raise RuntimeError(
            "internal: site / value array size mismatch "
            f"({values.size} vs {sids.size})"
        )

    bin_edges = default_bin_edges_km(
        n_bins=n_bins,
        max_distance_km=max_distance_km,
        near_threshold_km=near_threshold_km,
    )
    n_total_bins = bin_edges.size - 1

    distances = haversine_distances_km(coords[:, 0], coords[:, 1])
    iu, ju = np.triu_indices(values.size, k=1)
    sq = _squared_diff(values[iu], values[ju], kind)
    valid = (
        np.isfinite(sq)
        & np.isfinite(distances)
        & (distances <= max_distance_km)
    )
    bin_idx = np.searchsorted(bin_edges, distances, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_total_bins - 1)
    gamma, counts = _compute_variogram_arrays(
        sq, bin_idx, valid, n_total_bins
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return Variogram(
        bin_edges_km=bin_edges,
        bin_centers_km=bin_centers,
        bin_counts=counts,
        variogram_values=gamma,
        n_pairs_total=int(valid.sum()),
        kind=kind,
    )


# ---------------------------------------------------------------------------
# Randomisation null
# ---------------------------------------------------------------------------


class NullDistribution:
    __slots__ = (
        "bin_centers_km", "p05", "p50", "p95", "n_shuffles", "kind",
    )

    def __init__(
        self,
        *,
        bin_centers_km: np.ndarray,
        p05: np.ndarray,
        p50: np.ndarray,
        p95: np.ndarray,
        n_shuffles: int,
        kind: str,
    ):
        self.bin_centers_km = np.asarray(bin_centers_km, dtype=np.float64)
        self.p05 = np.asarray(p05, dtype=np.float64)
        self.p50 = np.asarray(p50, dtype=np.float64)
        self.p95 = np.asarray(p95, dtype=np.float64)
        self.n_shuffles = int(n_shuffles)
        self.kind = str(kind)


def randomisation_null(
    observable_table,
    observable_name: str,
    *,
    period_band_label: str | None = None,
    n_bins: int = 20,
    max_distance_km: float = 2000.0,
    near_threshold_km: float = 50.0,
    n_shuffles: int = 100,
    seed: int = 42,
) -> NullDistribution:
    """Permutation null for the empirical variogram.

    The site-to-coordinate mapping is held fixed and the observable
    values are shuffled across sites. With site labels broken, the
    expected variogram is *flat* across all separations (no spatial
    information left in the values). The 5th / 50th / 95th
    percentiles of the shuffled-variogram distribution per bin
    bracket the no-spatial-structure null.

    Performance: distances and bin assignments are computed once
    and reused for every shuffle. With 1353 sites and 100 shuffles
    the call typically runs in ≈ 5 seconds.
    """
    values, kind, _ = _per_site_values_for_band(
        observable_table, observable_name, period_band_label
    )
    sids, coords = _site_index_lookup(observable_table)
    bin_edges = default_bin_edges_km(
        n_bins=n_bins,
        max_distance_km=max_distance_km,
        near_threshold_km=near_threshold_km,
    )
    n_total_bins = bin_edges.size - 1

    distances = haversine_distances_km(coords[:, 0], coords[:, 1])
    iu, ju = np.triu_indices(values.size, k=1)
    bin_idx = np.searchsorted(bin_edges, distances, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, n_total_bins - 1)
    dist_valid = np.isfinite(distances) & (distances <= max_distance_km)

    rng = np.random.default_rng(int(seed))
    samples = np.full((n_shuffles, n_total_bins), np.nan)
    finite_orig = np.isfinite(values)
    if int(finite_orig.sum()) < 2:
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        return NullDistribution(
            bin_centers_km=bin_centers,
            p05=np.full(n_total_bins, np.nan),
            p50=np.full(n_total_bins, np.nan),
            p95=np.full(n_total_bins, np.nan),
            n_shuffles=n_shuffles,
            kind=kind,
        )

    finite_vals = values[finite_orig]

    for s in range(n_shuffles):
        # Shuffle the finite values into a fresh array; non-finite
        # positions stay NaN. This is the per-spec "shuffle the
        # observable values across sites".
        shuffled = np.full(values.shape, np.nan)
        perm = rng.permutation(finite_vals)
        shuffled[finite_orig] = perm
        sq = _squared_diff(shuffled[iu], shuffled[ju], kind)
        valid = np.isfinite(sq) & dist_valid
        gamma, _ = _compute_variogram_arrays(
            sq, bin_idx, valid, n_total_bins
        )
        samples[s] = gamma

    # Some bins may be entirely NaN across all shuffles when no
    # pairs fall in them; nanpercentile's "All-NaN slice" warning
    # is then expected and benign — suppress it locally.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered"
        )
        p05 = np.nanpercentile(samples, 5, axis=0)
        p50 = np.nanpercentile(samples, 50, axis=0)
        p95 = np.nanpercentile(samples, 95, axis=0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return NullDistribution(
        bin_centers_km=bin_centers,
        p05=p05, p50=p50, p95=p95,
        n_shuffles=n_shuffles, kind=kind,
    )


# ---------------------------------------------------------------------------
# Bootstrap-variance fallback
# ---------------------------------------------------------------------------


def _per_site_bootstrap_variance(
    observable_table,
    observable_name: str,
) -> float | None:
    """Best-available per-site noise variance.

    Tries (in order):

    1. A column named ``f"{observable_name}_bootstrap_var"`` in the
       table. (Reserved for a future Phase-2
       :func:`...continental_observables.compute_site_observables`
       output.)
    2. The inter-band variance of the observable at each site,
       averaged across sites. Less ideal — band-to-band differences
       reflect both noise and genuine frequency-dependence — but
       bounds the noise floor from above.

    Returns ``None`` when neither is computable (single band only,
    no finite values, etc.).
    """
    df = observable_table.dataframe
    boot_col = f"{observable_name}_bootstrap_var"
    if boot_col in df.columns:
        v = (
            df[[boot_col]]
            .astype(np.float64)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if not v.empty:
            return float(v[boot_col].mean())
    # Fallback: inter-band variance per site.
    if df["period_band_label"].nunique() < 2:
        warnings.warn(
            "spatial_coherence: cannot estimate bootstrap variance — "
            "no bootstrap column and only one period band in the "
            "table. Returning None.",
            stacklevel=2,
        )
        return None
    grouped = df.groupby("site_id")[observable_name]
    var_per_site = grouped.apply(
        lambda s: float(np.nanvar(s.astype(np.float64))) if s.notna().any()
        else np.nan
    ).dropna()
    if var_per_site.empty:
        return None
    warnings.warn(
        "spatial_coherence: per-site bootstrap CI not present in "
        "the observable table; using inter-band variance as a "
        "noise-floor fallback. The fallback over-estimates noise "
        "by absorbing genuine frequency dependence; treat as an "
        "upper bound.",
        stacklevel=2,
    )
    return float(np.nanmedian(var_per_site))


# ---------------------------------------------------------------------------
# Coherence summary
# ---------------------------------------------------------------------------


def _summarise_geostat(variogram: Variogram) -> tuple[float, float, float, float]:
    """Compute (nugget, sill, range_km, nugget_to_sill_ratio).

    * nugget : variogram value at the smallest non-empty bin.
    * sill   : median of the upper-quartile of bins.
    * range  : first bin centre at which γ ≥ nugget + 0.5 (sill - nugget).
    """
    gamma = variogram.variogram_values
    counts = variogram.bin_counts
    centres = variogram.bin_centers_km
    valid = np.isfinite(gamma) & (counts > 0)
    if not valid.any():
        return float("nan"), float("nan"), float("nan"), float("nan")
    finite_gamma = gamma[valid]
    finite_centres = centres[valid]
    nugget = float(finite_gamma[0])
    upper_q = max(1, int(np.ceil(finite_gamma.size / 4.0)))
    sill = float(np.median(finite_gamma[-upper_q:]))
    if sill <= nugget:
        # Pathological: sill ≤ nugget means no growth; report
        # range as NaN.
        ratio = 1.0 if sill > 0 else float("nan")
        return nugget, sill, float("nan"), ratio
    half_sill = nugget + 0.5 * (sill - nugget)
    above = finite_gamma >= half_sill
    if not above.any():
        range_km = float("nan")
    else:
        range_km = float(finite_centres[int(np.argmax(above))])
    ratio = float(min(max(nugget / sill, 0.0), 1.0)) if sill > 0 else float("nan")
    return nugget, sill, range_km, ratio


def _classify_coherence(
    variogram: Variogram,
    null: NullDistribution,
    nugget_to_sill_ratio: float,
) -> tuple[str, int]:
    """Apply the structured / weakly_structured / noise_dominated rule."""
    gamma = variogram.variogram_values
    counts = variogram.bin_counts
    valid = np.isfinite(gamma) & np.isfinite(null.p95) & (counts > 0)
    above = (gamma > null.p95) & valid
    n_valid = int(valid.sum())
    n_above = int(above.sum())
    if n_valid < 2:
        return "insufficient_data", n_above
    if not np.isfinite(nugget_to_sill_ratio):
        return "noise_dominated", n_above
    fraction_above = n_above / max(n_valid, 1)
    if nugget_to_sill_ratio >= 0.9 or n_above < 2:
        return "noise_dominated", n_above
    if fraction_above >= 0.5 and nugget_to_sill_ratio < 0.4:
        return "structured", n_above
    return "weakly_structured", n_above


def compute_coherence(
    observable_table,
    observable_name: str,
    *,
    period_band_label: str | None = None,
    n_bins: int = 20,
    max_distance_km: float = 2000.0,
    near_threshold_km: float = 50.0,
    n_shuffles: int = 100,
    seed: int = 42,
) -> CoherenceResult:
    """One-shot coherence summary for a single observable.

    Combines :func:`empirical_variogram`, :func:`randomisation_null`,
    a nugget-vs-noise reference, and the
    structured / weakly-structured / noise-dominated classifier
    into a single :class:`CoherenceResult`.

    Parameters
    ----------
    observable_table : ObservableTable
    observable_name : str
        Column to analyse. Type is auto-detected per
        :data:`ANGULAR_LINE_OBSERVABLES`,
        :data:`MAGNITUDE_OBSERVABLES`,
        :data:`ORDINAL_OBSERVABLES`.
    period_band_label : str, optional
    n_bins, max_distance_km, near_threshold_km
        Forwarded to :func:`empirical_variogram`.
    n_shuffles : int, default 100
        Number of shuffles for the randomisation null.
    seed : int, default 42

    Returns
    -------
    CoherenceResult
    """
    vg = empirical_variogram(
        observable_table, observable_name,
        period_band_label=period_band_label,
        n_bins=n_bins, max_distance_km=max_distance_km,
        near_threshold_km=near_threshold_km,
    )
    null = randomisation_null(
        observable_table, observable_name,
        period_band_label=period_band_label,
        n_bins=n_bins, max_distance_km=max_distance_km,
        near_threshold_km=near_threshold_km,
        n_shuffles=n_shuffles, seed=seed,
    )
    nugget, sill, range_km, nts = _summarise_geostat(vg)
    label, n_above = _classify_coherence(vg, null, nts)
    boot_var = _per_site_bootstrap_variance(observable_table, observable_name)

    # The band actually used (default-resolved if caller passed
    # None).
    _, _, band_used = _per_site_values_for_band(
        observable_table, observable_name, period_band_label
    )

    metadata = {
        "n_bins": int(n_bins),
        "near_threshold_km": float(near_threshold_km),
        "max_distance_km": float(max_distance_km),
        "n_shuffles": int(n_shuffles),
        "seed": int(seed),
        "kind": vg.kind,
        "n_bins_above_null_p95": int(n_above),
        "timestamp_utc": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat(),
    }
    return CoherenceResult(
        observable_name=observable_name,
        bin_centers_km=vg.bin_centers_km,
        bin_counts=vg.bin_counts,
        variogram_values=vg.variogram_values,
        null_p05=null.p05, null_p50=null.p50, null_p95=null.p95,
        nugget=nugget,
        sill=sill,
        range_km=range_km,
        nugget_to_sill_ratio=nts,
        bootstrap_variance=boot_var,
        coherence_label=label,
        n_pairs_total=vg.n_pairs_total,
        period_band_label=band_used,
        metadata=metadata,
    )


def compute_coherence_all(
    observable_table,
    *,
    observables: list[str] | None = None,
    period_band_label: str | None = None,
    n_bins: int = 20,
    max_distance_km: float = 2000.0,
    near_threshold_km: float = 50.0,
    n_shuffles: int = 100,
    seed: int = 42,
) -> dict[str, CoherenceResult]:
    """Run :func:`compute_coherence` on a set of observables.

    Defaults to :data:`PRIMARY_OBSERVABLES`. Failures on individual
    observables (missing column, all-NaN values) are caught and
    recorded in a :class:`CoherenceResult` with
    ``coherence_label="insufficient_data"``; the call as a whole
    never raises.
    """
    obs_list = observables if observables is not None else PRIMARY_OBSERVABLES
    out: dict[str, CoherenceResult] = {}
    for name in obs_list:
        try:
            out[name] = compute_coherence(
                observable_table, name,
                period_band_label=period_band_label,
                n_bins=n_bins, max_distance_km=max_distance_km,
                near_threshold_km=near_threshold_km,
                n_shuffles=n_shuffles, seed=seed,
            )
        except Exception as exc:  # noqa: BLE001 -- we degrade, not raise
            n_total_bins = n_bins + 1
            zero = np.zeros(n_total_bins)
            nan_arr = np.full(n_total_bins, np.nan)
            out[name] = CoherenceResult(
                observable_name=name,
                bin_centers_km=nan_arr.copy(),
                bin_counts=zero.astype(np.int64),
                variogram_values=nan_arr.copy(),
                null_p05=nan_arr.copy(), null_p50=nan_arr.copy(),
                null_p95=nan_arr.copy(),
                nugget=float("nan"), sill=float("nan"),
                range_km=float("nan"),
                nugget_to_sill_ratio=float("nan"),
                bootstrap_variance=None,
                coherence_label="insufficient_data",
                n_pairs_total=0,
                period_band_label=period_band_label or "",
                metadata={"error": repr(exc)},
            )
    return out
