"""Continental-scale distortion-as-signal observables pipeline.

This module is the canonical "compute everything for a station
collection" entry point for the Paper 1 empirical foundation. It
takes a list of :class:`mtpy.core.mt.MT` objects (or an
:class:`mtpy.core.mt_collection.MTCollection`), runs the existing
decomposition modules on each site at each configured period band,
and emits a tidy long-format table whose schema is fixed by
:data:`OBSERVABLE_COLUMNS`. Downstream analyses (mapping,
variograms, stratification) consume the table directly.

The pipeline is deliberately a *composition* layer: every
observable is computed by the existing per-method modules
(``groom_bailey``, ``bibby``, ``lilley``, ``lilley_dimensionality``,
``marti``, ``cross_method``, ``distortion_geometry``,
``magnetic_distortion_diagnostic``). No decomposition logic lives
here. If a desired observable cannot be cleanly produced by an
existing module the gap is recorded as a ``NaN`` / sentinel value
and documented as a Phase-2 follow-up rather than papered over with
inline math.

The "Paper 1 hero observable" is :func:`compute_site_observables`'s
``discordance_deg`` field — the angle between the recovered
distortion principal axis (``arg(gamma) / 2`` in spin-2 language)
and the phase-tensor principal axis ``alpha``. Galvanic physics
predicts these two should be tightly aligned (the distortion
ellipse points along the strike of the near-surface heterogeneity
that produces it, which the phase tensor also tracks); systematic
deviation is the empirical signal mapped continentally.

Cross-method disagreement columns
=================================
Three columns report how strongly the cross-tradition methods
disagree at each (site, band):

* ``cross_method_strike_disagreement_deg``
* ``cross_method_twist_disagreement_deg``
* ``cross_method_shear_disagreement_deg``

**Semantics (current, post-F1).** For each period ``p`` in the
band, take all pair-wise distances ``d(m_i, m_j, p)`` across the
methods that produced a finite value at ``p``, square-mean to
obtain the per-period statistic
``S(p) = sqrt(mean_pairs(d²))``. Aggregate across periods within
the band as
``column_value = sqrt(mean_p_in_band(S(p)²))``.

Periods at which fewer than two methods produced a finite output
are dropped from the per-band aggregation (the disagreement
between a single method and itself is meaningless).

* For the strike column: ``d`` is circular distance modulo 90°
  (``abs(((d1 - d2 + 45) mod 90) - 45)``) — strikes are
  90°-ambiguous in the parameterised methods (the GB-symmetry
  branch).
* For twist / shear: ``d`` is linear absolute distance.

**Changed in F4.** Pre-F4, the columns were "median per method,
then RMS of (each method - GB)" because the upstream Lilley
adapter returned full-grid arrays even when a period window was
requested, forcing pre-aggregation to per-method scalars. With
the F1 cross-method shape-bug fix every adapter now returns
arrays at the requested period grid, so per-period pair-RMS is
sound. The column is now a per-band *uncertainty estimate*
(scale of cross-method disagreement) rather than a per-band
central tendency. Downstream consumers should re-interpret
existing values accordingly: large values still indicate
disagreement, but the units are now "RMS pair-wise distance
across periods" not "RMS deviation of method medians from GB".

Two definitions of the spin-2 magnitude
=======================================
The schema carries two complementary magnitude columns. Both are
physically meaningful; including both supports sensitivity
analysis and direct comparison.

* ``gamma_magnitude``
  For each band, reconstruct a single ``C`` tensor from the
  band-aggregate ``(strike, twist, shear, gain)`` and compute
  ``|γ|`` of ``C - I`` via
  :func:`...distortion_geometry.irreducible_decomposition`.
  Answers: *"What is |γ| of the band-representative C?"*
  Internally consistent with the ``C_strike_deg`` /
  ``C_twist_deg`` / ``C_shear_deg`` columns in the same row —
  reconstructing those parameters into a ``C`` and taking ``γ``
  reproduces this number.

* ``gamma_magnitude_periodwise``
  For each period in the band, build a per-period ``C`` from
  GB's per-period optimised parameters, take ``|γ|``, then
  return the **geometric mean** across periods within the band.
  Periods with non-finite parameters (failed convergence) are
  excluded; if all periods are excluded the value is ``NaN``.
  Answers: *"What is the typical per-period distortion magnitude
  in this band?"* Robust to outlier periods and decoupled from
  the band's angular-aggregation choice.

The two definitions agree when GB resolves the band into a
single GB-internal sub-band (so per-period parameters are
identical) and diverge when the continental band spans multiple
GB sub-bands or when the optimiser picks competing modes at
different periods. The divergence is itself diagnostic: a large
``|gamma_magnitude − gamma_magnitude_periodwise|`` flags
within-band parameter instability.

Usage
-----
::

    from mtpy.core.transfer_function.z_analysis.decomposition import (
        compute_collection_observables,
    )

    table = compute_collection_observables(mt_collection)
    table.to_netcdf("auslamp_observables.nc")    # round-trippable
    table.to_csv("auslamp_observables.csv")      # human-readable export
    df = table.dataframe                         # long-format DataFrame

netCDF is the canonical serialisation format — round-trippable with
full dtype preservation (nullable ``Int64``, ``boolean``, string,
complex columns all survive intact); ``netcdf4`` is already in
the project closure. CSV is a one-way export for humans, GIS
overlay, and publication; it does not preserve nullable dtypes.
``ObservableTable.from_netcdf`` validates the loaded schema against
the current :data:`OBSERVABLE_COLUMNS`: extra columns load with a
:class:`UserWarning` (forward compatibility), missing columns
raise :class:`ValueError`.

The default :data:`DEFAULT_PERIOD_BANDS` is six 1-decade-wide bands
tiled across ``[0.01, 10000]`` s (the AusLAMP design range). Pass
``period_bands=`` for a different specification, or pass
``band_overlap_fraction=f`` (with ``f ∈ [0, 1)``) to widen the
default bands so adjacent bands share a fraction ``f`` of a unit
band width — useful for smoother spatial-coherence and
period-vs-observable plots.

**Paper-1 reproducibility note.** The choice of band geometry
(``n_bands`` *and* ``band_overlap_fraction``) materially affects
every downstream observable: with overlap > 0, a single per-period
GB / Lilley / Marti output contributes to multiple bands' aggregations.
Both numbers are captured in :class:`ObservableTable.metadata` (under
``n_bands`` and ``band_overlap_fraction``) so the long-format table
is reproducible from the metadata alone. **Any publication using
these outputs MUST report both values** — defaults today
(``n_bands=6``, ``band_overlap_fraction=0.0``) are documented but
are not assumed.

Reproducibility
---------------
Given a fixed ``seed`` and identical inputs, the observable values
are bit-identical across runs. The metadata sidecar carries the
mtpy version, the decomposition module git sha (when discoverable),
the timestamp, the method versions, the period-band specification,
the ``canonical_gauge`` choice, the RNG seed, and an input hash
(file list + modification times when the source is an
``MTCollection``).

References
----------
See the per-method module docstrings (``groom_bailey``,
``lilley_dimensionality``, ``marti``, ``cross_method``,
``distortion_geometry``, ``magnetic_distortion_diagnostic``) for the
foundational decomposition citations. The Paper 1 framing — the
distortion tensor as a signal rather than a nuisance — is the
project-level motivation; see ``CLAUDE.md`` and the
``docs/notes/Paper_01.md`` working notes in the parent
``mt_decomp`` repository.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib as _hashlib
import os as _os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .cross_method import compute_cross_method
from .distortion_geometry import irreducible_decomposition
from .groom_bailey import decompose
from .lilley_dimensionality import classify_dimensionality
from .magnetic_distortion_diagnostic import compute_magnetic_distortion_flag
from .marti import decompose_marti
from .results import ObservableTable, SiteObservables

if TYPE_CHECKING:  # pragma: no cover -- type-only imports
    from mtpy.core.mt import MT


__all__ = [
    "DEFAULT_PERIOD_BANDS",
    "OBSERVABLE_COLUMNS",
    "OBSERVABLE_DTYPES",
    "compute_collection_observables",
    "compute_site_observables",
    "default_period_bands",
]


# ---------------------------------------------------------------------------
# Period-band specification
# ---------------------------------------------------------------------------


@dataclass
class _PeriodBand:
    """Internal period-band representation.

    Each band has an explicit ``(period_min, period_max)`` window
    and a label / geomean. The aggregation routines treat the
    geomean as the band's representative period.
    """

    label: str
    period_min: float
    period_max: float

    @property
    def geomean(self) -> float:
        return float(np.sqrt(self.period_min * self.period_max))

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "period_min": self.period_min,
            "period_max": self.period_max,
            "geomean": self.geomean,
        }


def _format_band_label(p_min: float, p_max: float) -> str:
    return f"{p_min:g}s_{p_max:g}s"


def default_period_bands(
    band_overlap_fraction: float = 0.0,
    *,
    n_bands: int = 6,
    log_min: float = -2.0,
    log_max: float = 4.0,
) -> list[_PeriodBand]:
    """Six log-spaced period bands across ``[0.01, 10000]`` s.

    Parameters
    ----------
    band_overlap_fraction : float, default 0.0
        Overlap between adjacent bands as a fraction of the unit
        band width ``L / n_bands``. Must be in ``[0, 1)``.

        * ``0.0`` (default) — bands tile with no overlap. Each
          band is exactly ``L / n_bands`` decades wide.
        * ``> 0`` — each band is widened by a factor ``(1 + f)``
          on each side, so adjacent bands share an ``f``-fraction
          of a unit width. Useful for smoother sweeps in spatial
          coherence and period-vs-observable plots, at the cost
          of one period contributing to multiple bands.
    n_bands : int, default 6
    log_min, log_max : float
        Total log10-period span. Defaults give the AusLAMP design
        range ``log10(period) ∈ [-2, 4]`` (0.01 s to 10000 s).

    Returns
    -------
    list[_PeriodBand]
        Length ``n_bands``. Each band's ``[period_min, period_max]``
        is constructed from the worked formula below.

    Worked example
    --------------
    With ``L = log_max - log_min``, the formula for band ``i``
    (``i = 0 ... n_bands-1``) is::

        start_i = log_min + i · L / n_bands - f · L / (2 · n_bands)
        end_i   = start_i + (1 + f) · L / n_bands

    so each band has *log-width* ``(1 + f) · L / n_bands`` and
    centres are spaced by ``L / n_bands``. The shared overlap
    between adjacent bands is ``f · L / n_bands``.

    Default case (``f = 0``)::

        L = 6, n_bands = 6, log_min = -2, log_max = 4
        band 0: [10^-2,   10^-1] = [0.01,    0.1] s
        band 1: [10^-1,   10^0 ] = [0.1,     1  ] s
        band 2: [10^0,    10^1 ] = [1,       10 ] s
        band 3: [10^1,    10^2 ] = [10,      100] s
        band 4: [10^2,    10^3 ] = [100,    1000] s
        band 5: [10^3,    10^4 ] = [1000,  10000] s

    With ``f = 0.5``::

        Each band is 1.5 decades wide; adjacent bands overlap by
        0.5 decade (the ``f · L / n_bands`` overlap).
        band 0: [10^-2.25, 10^-0.75] ≈ [0.0056,    0.178] s
        band 1: [10^-1.25, 10^ 0.25] ≈ [0.0562,    1.778] s
        band 2: [10^-0.25, 10^ 1.25] ≈ [0.562,    17.78 ] s
        band 3: [10^ 0.75, 10^ 2.25] ≈ [5.62,    177.8  ] s
        band 4: [10^ 1.75, 10^ 3.25] ≈ [56.2,   1778    ] s
        band 5: [10^ 2.75, 10^ 4.25] ≈ [562,   17783    ] s

        A period at 1 s (``log10 = 0``) lies in bands 1 and 2.

    Raises
    ------
    ValueError
        If ``band_overlap_fraction`` is outside ``[0, 1)``.
    """
    if not (0.0 <= band_overlap_fraction < 1.0):
        raise ValueError(
            f"default_period_bands: band_overlap_fraction must be "
            f"in [0, 1); got {band_overlap_fraction}"
        )
    f = float(band_overlap_fraction)
    L = float(log_max - log_min)
    width = (1.0 + f) * L / n_bands
    out: list[_PeriodBand] = []
    for i in range(n_bands):
        start = log_min + i * L / n_bands - f * L / (2.0 * n_bands)
        end = start + width
        p_min = float(10.0**start)
        p_max = float(10.0**end)
        out.append(
            _PeriodBand(
                label=_format_band_label(p_min, p_max),
                period_min=p_min,
                period_max=p_max,
            )
        )
    return out


DEFAULT_PERIOD_BANDS: list[_PeriodBand] = default_period_bands()
"""Module-level default (no overlap). See
:func:`default_period_bands`. Mutating this list in place is
unsupported — pass ``period_bands=`` to override per-call.
"""


def _coerce_period_bands(
    period_bands,
    band_overlap_fraction: float = 0.0,
) -> list[_PeriodBand]:
    """Accept a variety of band specifications and normalise to
    ``list[_PeriodBand]``.

    Accepted forms:

    * ``None`` -> :func:`default_period_bands` with the supplied
      ``band_overlap_fraction``. The cached
      :data:`DEFAULT_PERIOD_BANDS` is returned only when the
      caller explicitly accepts ``band_overlap_fraction == 0.0``.
    * ``list[_PeriodBand]`` -> returned as-is (overlap kwarg is
      ignored — explicit specifications win).
    * ``list[(p_min, p_max)]`` -> labels auto-generated.
    * ``list[dict]`` with keys ``label``, ``period_min``,
      ``period_max`` -> coerced.
    """
    if period_bands is None:
        if band_overlap_fraction == 0.0:
            return DEFAULT_PERIOD_BANDS
        return default_period_bands(
            band_overlap_fraction=band_overlap_fraction
        )
    out: list[_PeriodBand] = []
    for spec in period_bands:
        if isinstance(spec, _PeriodBand):
            out.append(spec)
        elif isinstance(spec, dict):
            out.append(
                _PeriodBand(
                    label=str(
                        spec.get(
                            "label",
                            _format_band_label(
                                float(spec["period_min"]),
                                float(spec["period_max"]),
                            ),
                        )
                    ),
                    period_min=float(spec["period_min"]),
                    period_max=float(spec["period_max"]),
                )
            )
        else:
            p_min, p_max = spec
            out.append(
                _PeriodBand(
                    label=_format_band_label(float(p_min), float(p_max)),
                    period_min=float(p_min),
                    period_max=float(p_max),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Observable-table schema
# ---------------------------------------------------------------------------


# Canonical column names (and human-readable units / docstrings live
# in the module docstring; this list is the schema contract).
OBSERVABLE_COLUMNS: list[str] = [
    # Site identification
    "site_id",
    "longitude_deg",
    "latitude_deg",
    "elevation_m",
    "period_band_label",
    "period_band_geomean_s",
    # Distortion-tensor primary observables
    "C_strike_deg",
    "C_twist_deg",
    "C_shear_deg",
    "C_minus_I_F",
    "C_determinant_real",
    "C_determinant_imag",
    # Irreducible distortion decomposition
    "gamma_1",
    "gamma_2",
    "gamma_magnitude",
    "gamma_magnitude_periodwise",
    "gamma_principal_axis_deg",
    "beta_antisymmetric",
    "trace_a",
    # Phase-tensor observables
    "PT_alpha_deg",
    "PT_beta_deg",
    "PT_abs_beta_deg",
    "PT_ellipticity",
    "PT_lambda_max",
    "PT_lambda_min",
    # Cross-method consistency
    "GB_rms_misfit",
    "GB_mode_warning",
    "MJ_rms_misfit",
    "cross_method_strike_disagreement_deg",
    "cross_method_twist_disagreement_deg",
    "cross_method_shear_disagreement_deg",
    # Dimensionality classification
    "WALDIM_case",
    "Lilley_category",
    "dimensionality_concordant",
    # Diagnostics
    "magnetic_distortion_flag",
    "tipper_max_amplitude",
    # Discordance metrics (Paper 1 hero)
    "discordance_deg",
    "discordance_significance",
]

# Per-column dtype hints (used to coerce on output). Float64 is the
# default; we name the exceptions.
OBSERVABLE_DTYPES: dict[str, str] = {
    "site_id": "string",
    "period_band_label": "string",
    "GB_mode_warning": "boolean",
    "WALDIM_case": "Int64",  # nullable int
    "Lilley_category": "string",
    "dimensionality_concordant": "boolean",
    "magnetic_distortion_flag": "string",
}


def _empty_row(
    *,
    site_id: str,
    longitude_deg: float,
    latitude_deg: float,
    elevation_m: float,
    band: _PeriodBand,
) -> dict[str, Any]:
    """A new row pre-populated with sentinel values.

    All numeric columns default to ``NaN``; the boolean / int / str
    columns get sensible "missing" sentinels (``None`` / ``pd.NA``).
    Filling proceeds by overwriting these defaults in place.
    """
    row: dict[str, Any] = {col: np.nan for col in OBSERVABLE_COLUMNS}
    row["site_id"] = site_id
    row["longitude_deg"] = float(longitude_deg)
    row["latitude_deg"] = float(latitude_deg)
    row["elevation_m"] = (
        float(elevation_m) if elevation_m is not None else np.nan
    )
    row["period_band_label"] = band.label
    row["period_band_geomean_s"] = band.geomean
    row["GB_mode_warning"] = pd.NA
    row["WALDIM_case"] = pd.NA
    row["Lilley_category"] = pd.NA
    row["dimensionality_concordant"] = pd.NA
    row["magnetic_distortion_flag"] = pd.NA
    return row


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _geometric_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values) & (values > 0)
    if not finite.any():
        return float("nan")
    return float(np.exp(np.mean(np.log(values[finite]))))


def _circular_weighted_mean_deg(
    angles_deg: np.ndarray,
    weights: np.ndarray,
    period: float = 180.0,
) -> float:
    """Circular weighted mean of angles modulo ``period``.

    Maps each input angle ``a`` (degrees) onto the unit circle via
    ``theta = 2π a / period`` and returns the angle of the
    weighted vector mean. ``period=180°`` is the right choice for
    line-direction quantities (strike, principal axis); ``period=
    360°`` is the standard angular mean.
    """
    finite = np.isfinite(angles_deg) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float("nan")
    a = angles_deg[finite]
    w = weights[finite]
    theta = 2.0 * np.pi * a / period
    s = float(np.sum(w * np.sin(theta)))
    c = float(np.sum(w * np.cos(theta)))
    if s == 0.0 and c == 0.0:
        return float("nan")
    mean_theta_rad = np.arctan2(s, c)
    mean_deg = float(np.degrees(mean_theta_rad) * period / 360.0)
    return mean_deg % period


def _arithmetic_mean(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan")
    return float(np.mean(values[finite]))


def _geomean_finite(values) -> float:
    """Geometric mean of a sequence of non-negative floats.

    Non-finite entries (``NaN``, ``inf``, ``-inf``) are excluded.
    Zero entries are clipped to ``1e-30`` before the log so they
    pull the geomean toward zero without producing ``log(0)``.
    Returns ``NaN`` when no finite values are available.

    Used for the per-band aggregation of per-period ``|γ|``
    values — see :data:`OBSERVABLE_COLUMNS` and the module
    docstring's "Two definitions of the spin-2 magnitude"
    section.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    pos = np.maximum(arr[finite], 1e-30)
    return float(np.exp(np.mean(np.log(pos))))


def _axial_angular_distance_deg(a: float, b: float) -> float:
    """Smaller-arc distance between two line directions (mod 180°).

    Both inputs are angles in degrees treated as line directions
    (i.e. ``a`` and ``a + 180°`` represent the same line). The
    distance is the angle between the two lines, in ``[0°, 90°]``.
    Used for discordance (angle between the C principal axis and
    the phase-tensor alpha).
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def _strike_distance_mod_90_deg(a: float, b: float) -> float:
    """Circular angular distance modulo 90° between two strikes.

    The parameterised methods (GB / MJ / BCB / Lilley) report
    strike with a 90° ambiguity (the GB-symmetry branch). The
    natural distance is therefore on a 90°-period circle, range
    ``[0°, 45°]``. Equivalent to:

        abs(((a - b + 45) mod 90) - 45)

    A 5° / 85° pair gives 10° (close on the 90°-circle), not 80°.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(abs(((a - b + 45.0) % 90.0) - 45.0))


def _per_period_pair_rms(
    arrays: list[np.ndarray],
    distance_fn,
) -> np.ndarray:
    """Per-period RMS of the pair-wise distances across method
    arrays.

    For each period ``k``, collect the values from each method's
    array at index ``k``, drop non-finite entries, and compute the
    RMS of all pair-wise distances among the surviving values.
    Periods where fewer than two methods contributed a finite
    value yield ``NaN`` (cannot compute disagreement from one
    method). All input arrays must have the same length (the
    cross-method shape contract enforced by
    :func:`compute_cross_method` post-fix; see the F1 commit).
    """
    if not arrays:
        return np.empty(0, dtype=np.float64)
    n_periods = arrays[0].size
    for arr in arrays:
        if arr.size != n_periods:  # pragma: no cover -- F1 contract
            raise ValueError(
                "_per_period_pair_rms: input arrays differ in length "
                f"({[a.size for a in arrays]}). Cross-method adapters "
                "must return matching shapes (see cross_method.py "
                "shape-bug fix)."
            )
    out = np.full(n_periods, np.nan, dtype=np.float64)
    for k in range(n_periods):
        finite_vals = [
            float(arr[k]) for arr in arrays if np.isfinite(arr[k])
        ]
        if len(finite_vals) < 2:
            continue
        sq_sum = 0.0
        n_pairs = 0
        for i in range(len(finite_vals)):
            for j in range(i + 1, len(finite_vals)):
                d = distance_fn(finite_vals[i], finite_vals[j])
                if np.isfinite(d):
                    sq_sum += d * d
                    n_pairs += 1
        if n_pairs > 0:
            out[k] = float(np.sqrt(sq_sum / n_pairs))
    return out


def _per_band_rms_of_finite(values: np.ndarray) -> float:
    """Root-mean-square of the finite entries of ``values``.

    Used to aggregate per-period statistics into a single per-band
    scalar (the second stage of the per-period-pair-RMS,
    per-band-RMS pipeline). NaN periods are treated as missing
    and excluded from the mean.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(arr[finite] ** 2)))


# ---------------------------------------------------------------------------
# Per-band observable computation
# ---------------------------------------------------------------------------


_MJ_NOT_RUN = -1.0
"""Sentinel: MJ has not been requested in single-site mode.

We do not run MJ in single-site mode by default — it duplicates GB
and the field would be redundant. ``MJ_rms_misfit`` therefore stays
NaN in the standard pipeline; pass ``methods=['groom_bailey',
'mcneice_jones', ...]`` to override.
"""


def _construct_C_gb89(
    strike_deg: float, twist_deg: float, shear_deg: float, gain: float
) -> np.ndarray:
    """Reconstruct the GB89 distortion matrix in the measurement frame.

    Matches the synthetic-test harness convention (``synthetics.py``)
    and the :func:`...common._estim_imp` inner formulation:
    ``C = gain · R(strike) · T(twist) · S(shear) · R(strike).T``.
    """
    s = float(np.radians(strike_deg))
    t = float(np.radians(twist_deg))
    sh = float(np.radians(shear_deg))
    R = np.array([[np.cos(s), -np.sin(s)], [np.sin(s), np.cos(s)]])
    T = np.array([[1.0, -np.tan(t)], [np.tan(t), 1.0]])
    S = np.array([[1.0, np.tan(sh)], [np.tan(sh), 1.0]])
    return float(gain) * R @ T @ S @ R.T


def _gb_band_observables(z, band: _PeriodBand, *, n_starts: int, seed: int,
                         canonical_gauge: str) -> dict[str, Any]:
    """Run GB on a band's period window and reduce to per-band scalars.

    Failure modes (no periods in window, bounds violation, optimiser
    exception) are caught; in each case the observables are left as
    NaN and a status string is returned so the caller can record the
    failure.
    """
    out: dict[str, Any] = {
        "status": "ok",
        "message": "",
        "C_strike_deg": float("nan"),
        "C_twist_deg": float("nan"),
        "C_shear_deg": float("nan"),
        "C_minus_I_F": float("nan"),
        "C_determinant_real": float("nan"),
        "C_determinant_imag": float("nan"),
        "gamma_1": float("nan"),
        "gamma_2": float("nan"),
        "gamma_magnitude": float("nan"),
        "gamma_magnitude_periodwise": float("nan"),
        "gamma_principal_axis_deg": float("nan"),
        "beta_antisymmetric": float("nan"),
        "trace_a": float("nan"),
        "GB_rms_misfit": float("nan"),
        "GB_mode_warning": pd.NA,
    }

    # Restrict GB to periods that fall in the band.
    z_periods = 1.0 / np.asarray(z.frequency, dtype=np.float64)
    in_band = (z_periods >= band.period_min) & (z_periods <= band.period_max)
    if int(in_band.sum()) < 2:
        out["status"] = "no_periods"
        out["message"] = (
            f"only {int(in_band.sum())} period(s) in band "
            f"{band.label}; GB needs at least 2."
        )
        return out

    try:
        gb_result = decompose(
            z,
            periods=(band.period_min, band.period_max),
            n_starts=n_starts,
            seed=seed,
            canonical_gauge=canonical_gauge,
        )
    except Exception as exc:
        out["status"] = "gb_error"
        out["message"] = f"decompose raised: {exc!r}"
        return out

    p = gb_result.parameters
    strikes = np.asarray(p["strike"].values, dtype=np.float64)
    twists = np.asarray(p["twist"].values, dtype=np.float64)
    shears = np.asarray(p["shear"].values, dtype=np.float64)
    gains = np.asarray(p["gain"].values, dtype=np.float64)
    rms = float(gb_result.rms_misfit)
    weights = np.full_like(strikes, 1.0 / max(rms * rms, 1e-30))

    # Aggregate per-period -> per-band.
    out["C_strike_deg"] = _circular_weighted_mean_deg(
        strikes, weights, period=180.0
    )
    out["C_twist_deg"] = _arithmetic_mean(twists)
    out["C_shear_deg"] = _arithmetic_mean(shears)
    out["GB_rms_misfit"] = rms
    out["GB_mode_warning"] = bool(
        gb_result.metadata.get("primary_mode_warning", False)
    )

    # Reconstruct C from the band-aggregated parameters and pull the
    # gauge-trapped magnitude / determinant. We use the band-mean
    # gain (linear average of the per-period values, since gain is
    # in linear space here, not log space).
    gain_band = _arithmetic_mean(gains)
    C_band = _construct_C_gb89(
        out["C_strike_deg"],
        out["C_twist_deg"],
        out["C_shear_deg"],
        gain_band,
    )
    eye = np.eye(2)
    out["C_minus_I_F"] = float(np.linalg.norm(C_band - eye, ord="fro"))
    det_C = complex(np.linalg.det(C_band))
    out["C_determinant_real"] = float(det_C.real)
    out["C_determinant_imag"] = float(det_C.imag)

    # Spin-2 / spin-0 split of D = C - I.
    irrep = irreducible_decomposition(C_band - eye)
    out["gamma_1"] = float(irrep["gamma_1"])
    out["gamma_2"] = float(irrep["gamma_2"])
    out["gamma_magnitude"] = float(
        np.hypot(out["gamma_1"], out["gamma_2"])
    )
    out["gamma_principal_axis_deg"] = float(
        (np.degrees(0.5 * np.arctan2(out["gamma_2"], out["gamma_1"]))) % 180.0
    )
    out["beta_antisymmetric"] = float(irrep["beta"])
    out["trace_a"] = float(irrep["trace_a"])

    # Periodwise |γ|: per-period C from per-period (strike, twist,
    # shear, gain) → |γ| from each → geometric mean across periods.
    # This is the magnitude observable that does *not* depend on
    # the band's angular aggregation choice — see the module
    # docstring's "Two definitions of the spin-2 magnitude".
    per_period_gamma_mag = []
    for k in range(strikes.size):
        if not (
            np.isfinite(strikes[k])
            and np.isfinite(twists[k])
            and np.isfinite(shears[k])
            and np.isfinite(gains[k])
        ):
            continue
        C_k = _construct_C_gb89(
            float(strikes[k]),
            float(twists[k]),
            float(shears[k]),
            float(gains[k]),
        )
        irrep_k = irreducible_decomposition(C_k - eye)
        mag = float(np.hypot(irrep_k["gamma_1"], irrep_k["gamma_2"]))
        if np.isfinite(mag):
            per_period_gamma_mag.append(mag)
    out["gamma_magnitude_periodwise"] = _geomean_finite(per_period_gamma_mag)
    return out


def _pt_band_observables(z, band: _PeriodBand) -> dict[str, Any]:
    """Phase-tensor invariants aggregated over the band."""
    out: dict[str, Any] = {
        "status": "ok",
        "message": "",
        "PT_alpha_deg": float("nan"),
        "PT_beta_deg": float("nan"),
        "PT_abs_beta_deg": float("nan"),
        "PT_ellipticity": float("nan"),
        "PT_lambda_max": float("nan"),
        "PT_lambda_min": float("nan"),
        "Lilley_category": pd.NA,
        "_lilley_periods": np.empty(0),
        "_lilley_classification": [],
    }
    try:
        ld = classify_dimensionality(
            z, periods=(band.period_min, band.period_max)
        )
    except Exception as exc:
        out["status"] = "lilley_error"
        out["message"] = f"classify_dimensionality raised: {exc!r}"
        return out

    out["PT_alpha_deg"] = _circular_weighted_mean_deg(
        np.asarray(ld.pt_alpha_deg, dtype=np.float64),
        np.ones_like(ld.pt_alpha_deg, dtype=np.float64),
        period=180.0,
    )
    out["PT_beta_deg"] = _arithmetic_mean(
        np.asarray(ld.pt_beta_deg, dtype=np.float64)
    )
    out["PT_abs_beta_deg"] = _arithmetic_mean(
        np.abs(np.asarray(ld.pt_beta_deg, dtype=np.float64))
    )
    out["PT_ellipticity"] = _arithmetic_mean(
        np.asarray(ld.pt_ellipticity, dtype=np.float64)
    )
    out["PT_lambda_max"] = _arithmetic_mean(
        np.asarray(ld.pt_lambda_max, dtype=np.float64)
    )
    out["PT_lambda_min"] = _arithmetic_mean(
        np.asarray(ld.pt_lambda_min, dtype=np.float64)
    )

    # Worst-case aggregation for the dimensionality label: 3D > 3D-2D
    # > 2D > 1D > indeterminate.
    rank = {"1D": 0, "2D": 1, "3D-2D": 2, "3D": 3, "indeterminate": -1}
    if ld.classification:
        worst = max(
            ld.classification,
            key=lambda c: rank.get(c, -1),
        )
        out["Lilley_category"] = worst
    out["_lilley_periods"] = ld.periods
    out["_lilley_classification"] = list(ld.classification)
    return out


def _marti_band_observables(z, band: _PeriodBand) -> dict[str, Any]:
    """Marti / WALDIM dominant case over the band.

    Aggregation rule: integer mode (most common WALDIM code) within
    the band. ``0`` (undetermined) is filtered out before voting if
    any non-zero codes are present.
    """
    out: dict[str, Any] = {
        "status": "ok",
        "message": "",
        "WALDIM_case": pd.NA,
    }
    try:
        marti = decompose_marti(z)
    except Exception as exc:
        out["status"] = "marti_error"
        out["message"] = f"decompose_marti raised: {exc!r}"
        return out

    pers = np.asarray(marti.periods, dtype=np.float64)
    dim = np.asarray(marti.dimensionality, dtype=np.int64)
    in_band = (pers >= band.period_min) & (pers <= band.period_max)
    if not in_band.any():
        out["status"] = "no_periods"
        out["message"] = "no Marti periods in band"
        return out
    band_dim = dim[in_band]
    nonzero = band_dim[band_dim != 0]
    if nonzero.size == 0:
        out["WALDIM_case"] = 0
    else:
        # Mode (most common value).
        vals, counts = np.unique(nonzero, return_counts=True)
        out["WALDIM_case"] = int(vals[int(np.argmax(counts))])
    return out


def _cross_method_disagreements(
    z,
    band: _PeriodBand,
    *,
    canonical_gauge: str,
) -> dict[str, Any]:
    """Per-band cross-method disagreement summaries.

    For each per-period observable (strike, twist, shear), compute
    the per-period RMS of the pair-wise distances across all
    methods that produced a finite value at that period, then take
    the RMS across periods within the band. This is a per-period-
    pair-RMS, per-band-RMS pipeline; see the module docstring's
    "cross-method disagreement columns" section for the rigorous
    definition.

    The cross-method shape contract (each adapter returns arrays
    of length ``n_periods`` matching :attr:`CrossMethodResult.periods`)
    is required for this function to be sound. It was established
    by the cross_method shape-bug fix (commit ``48a3280``); see the
    docstring of :func:`...cross_method._adapter_lilley` for the
    detailed history.
    """
    out: dict[str, Any] = {
        "status": "ok",
        "message": "",
        "cross_method_strike_disagreement_deg": float("nan"),
        "cross_method_twist_disagreement_deg": float("nan"),
        "cross_method_shear_disagreement_deg": float("nan"),
    }
    try:
        cm = compute_cross_method(
            z,
            methods=["groom_bailey", "bibby", "lilley"],
            periods=(band.period_min, band.period_max),
            method_kwargs={
                "groom_bailey": {"canonical_gauge": canonical_gauge}
            },
        )
    except Exception as exc:
        out["status"] = "cross_method_error"
        out["message"] = f"compute_cross_method raised: {exc!r}"
        return out

    # Strike: all methods that produced a strike, all pair-wise
    # circular distances mod 90°, RMS at each period, RMS across
    # periods.
    strike_arrays = [
        np.asarray(arr, dtype=np.float64)
        for arr in cm.strike_estimates.values()
    ]
    if len(strike_arrays) >= 2:
        per_period = _per_period_pair_rms(
            strike_arrays, _strike_distance_mod_90_deg
        )
        out["cross_method_strike_disagreement_deg"] = (
            _per_band_rms_of_finite(per_period)
        )

    # Twist / shear: linear distance (no circular wrap); same
    # pair-RMS pipeline. Methods that don't produce twist/shear
    # (Lilley, Marti) are absent from cm.twist_shear_estimates,
    # so the pair count is naturally smaller.
    twist_arrays = [
        np.asarray(twist, dtype=np.float64)
        for twist, _shear in cm.twist_shear_estimates.values()
    ]
    if len(twist_arrays) >= 2:
        per_period = _per_period_pair_rms(twist_arrays, _linear_abs_diff)
        out["cross_method_twist_disagreement_deg"] = (
            _per_band_rms_of_finite(per_period)
        )

    shear_arrays = [
        np.asarray(shear, dtype=np.float64)
        for _twist, shear in cm.twist_shear_estimates.values()
    ]
    if len(shear_arrays) >= 2:
        per_period = _per_period_pair_rms(shear_arrays, _linear_abs_diff)
        out["cross_method_shear_disagreement_deg"] = (
            _per_band_rms_of_finite(per_period)
        )

    return out


def _linear_abs_diff(a: float, b: float) -> float:
    """Absolute linear distance ``|a - b|``. Used for the
    twist / shear disagreement, which has no circular wrap.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(abs(a - b))


def _diagnostics_band_observables(
    mt_object: "MT",
    band: _PeriodBand,
) -> dict[str, Any]:
    """Magnetic-distortion flag + Tipper amplitude for the band."""
    out: dict[str, Any] = {
        "magnetic_distortion_flag": pd.NA,
        "tipper_max_amplitude": float("nan"),
    }
    try:
        flag = compute_magnetic_distortion_flag(
            mt_object,
            period_range=(band.period_min, band.period_max),
        )
        out["magnetic_distortion_flag"] = str(flag.overall_flag)
        if (
            flag.tipper_diagnostic is not None
            and flag.tipper_diagnostic.get("available")
        ):
            out["tipper_max_amplitude"] = float(
                flag.tipper_diagnostic.get("max_amplitude", float("nan"))
            )
    except Exception:
        # Non-fatal: leave NaN / NA sentinels.
        pass
    return out


# ---------------------------------------------------------------------------
# Public per-site entry point
# ---------------------------------------------------------------------------


def compute_site_observables(
    mt_object: "MT",
    *,
    period_bands=None,
    canonical_gauge: str = "pt_aligned",
    n_starts: int = 5,
    seed: int = 42,
    band_overlap_fraction: float = 0.0,
) -> SiteObservables:
    """All distortion-as-signal observables for a single site.

    Parameters
    ----------
    mt_object : MT
        mtpy ``MT`` instance carrying ``Z`` (required) and optional
        ``Tipper``, ``latitude``, ``longitude``, ``elevation``.
    period_bands : sequence, optional
        Band specification; see :func:`default_period_bands` for the
        default. Each entry can be a :class:`_PeriodBand`, a
        ``(p_min, p_max)`` tuple, or a dict with keys ``label``,
        ``period_min``, ``period_max``. When supplied,
        ``band_overlap_fraction`` is ignored.
    canonical_gauge : str, default ``'pt_aligned'``
        GB canonical-gauge selection. Forwarded verbatim to
        :func:`...groom_bailey.decompose`.
    n_starts : int, default 5
        Multi-start count for GB.
    seed : int, default 42
        RNG seed for any stochastic component.
    band_overlap_fraction : float, default 0.0
        Adjacent-band overlap as a fraction of the unit band
        width. Forwarded to :func:`default_period_bands` only
        when ``period_bands is None``. ``0.0`` keeps the
        backward-compatible tiled default; ``> 0`` widens each
        band so adjacent bands share data — useful for smoother
        spatial-coherence and period-vs-observable sweeps.

    Returns
    -------
    SiteObservables
    """
    bands = _coerce_period_bands(period_bands, band_overlap_fraction)

    site_id = str(getattr(mt_object, "station", "") or "")
    longitude = float(getattr(mt_object, "longitude", float("nan")) or 0.0)
    latitude = float(getattr(mt_object, "latitude", float("nan")) or 0.0)
    elevation_attr = getattr(mt_object, "elevation", None)
    elevation = (
        float(elevation_attr)
        if elevation_attr is not None and np.isfinite(elevation_attr)
        else float("nan")
    )

    z = mt_object.Z

    band_rows: list[dict[str, Any]] = []
    for band in bands:
        row = _empty_row(
            site_id=site_id,
            longitude_deg=longitude,
            latitude_deg=latitude,
            elevation_m=elevation,
            band=band,
        )

        gb = _gb_band_observables(
            z, band, n_starts=n_starts, seed=seed,
            canonical_gauge=canonical_gauge,
        )
        for k in (
            "C_strike_deg", "C_twist_deg", "C_shear_deg",
            "C_minus_I_F", "C_determinant_real", "C_determinant_imag",
            "gamma_1", "gamma_2", "gamma_magnitude",
            "gamma_magnitude_periodwise",
            "gamma_principal_axis_deg", "beta_antisymmetric", "trace_a",
            "GB_rms_misfit", "GB_mode_warning",
        ):
            row[k] = gb[k]

        pt = _pt_band_observables(z, band)
        for k in (
            "PT_alpha_deg", "PT_beta_deg", "PT_abs_beta_deg",
            "PT_ellipticity", "PT_lambda_max", "PT_lambda_min",
            "Lilley_category",
        ):
            row[k] = pt[k]

        marti = _marti_band_observables(z, band)
        row["WALDIM_case"] = marti["WALDIM_case"]

        cross = _cross_method_disagreements(
            z, band, canonical_gauge=canonical_gauge
        )
        for k in (
            "cross_method_strike_disagreement_deg",
            "cross_method_twist_disagreement_deg",
            "cross_method_shear_disagreement_deg",
        ):
            row[k] = cross[k]

        diags = _diagnostics_band_observables(mt_object, band)
        for k in ("magnetic_distortion_flag", "tipper_max_amplitude"):
            row[k] = diags[k]

        # Dimensionality concordance: WALDIM and Lilley both produced
        # something, and the WALDIM case maps to the same Lilley
        # category (approximately).
        waldim = row["WALDIM_case"]
        lilley = row["Lilley_category"]
        if waldim is pd.NA or lilley is pd.NA:
            row["dimensionality_concordant"] = pd.NA
        else:
            # Mapping: WALDIM 1 -> 1D; 2 -> 2D; 3,4,6,7 -> 3D-2D or
            # 2D (galvanic distortion); 5 -> 3D.
            waldim_int = int(waldim)
            mapping = {1: {"1D"}, 2: {"2D"}, 3: {"2D", "3D-2D"},
                       4: {"2D", "3D-2D"}, 5: {"3D"},
                       6: {"2D", "3D-2D"}, 7: {"2D", "3D-2D"}, 0: set()}
            row["dimensionality_concordant"] = bool(
                str(lilley) in mapping.get(waldim_int, set())
            )

        # Discordance: angle between the GB C principal axis and the
        # phase-tensor alpha as line directions (range [0°, 90°]).
        row["discordance_deg"] = _axial_angular_distance_deg(
            float(row["gamma_principal_axis_deg"]),
            float(row["PT_alpha_deg"]),
        )
        # Bootstrap-based significance is a follow-up; report NaN.
        row["discordance_significance"] = float("nan")

        band_rows.append(row)

    return SiteObservables(
        site_id=site_id,
        longitude_deg=longitude,
        latitude_deg=latitude,
        elevation_m=elevation,
        band_observables=band_rows,
        metadata=_build_metadata(
            seed=seed,
            canonical_gauge=canonical_gauge,
            n_starts=n_starts,
            period_bands=bands,
            input_identifier=site_id,
            band_overlap_fraction=band_overlap_fraction,
        ),
    )


# ---------------------------------------------------------------------------
# Public collection entry point
# ---------------------------------------------------------------------------


def _iter_mt_objects(mt_collection_or_list):
    """Yield ``MT``-like objects from either an MTCollection or a list."""
    if hasattr(mt_collection_or_list, "dataframe") and hasattr(
        mt_collection_or_list, "get_tf"
    ):
        df = mt_collection_or_list.dataframe
        if df is None:
            return
        for row in df.itertuples():
            tf_id = getattr(row, "tf_id", getattr(row, "station", None))
            if tf_id is None:
                continue
            yield mt_collection_or_list.get_tf(tf_id)
    else:
        yield from mt_collection_or_list


def _hash_collection(mt_collection_or_list) -> str:
    """Hash of file list + modification times for the input.

    A best-effort fingerprint: when a collection-style object has
    ``.fn`` / ``.dataframe`` we hash the file paths and mtimes; for
    a plain list we hash the station ids. The result is for
    reproducibility tracking, not cryptographic verification.
    """
    h = _hashlib.sha256()
    if hasattr(mt_collection_or_list, "dataframe") and hasattr(
        mt_collection_or_list, "get_tf"
    ):
        df = mt_collection_or_list.dataframe
        if df is None:
            return "empty_collection"
        for row in df.itertuples():
            tf_id = str(getattr(row, "tf_id", getattr(row, "station", "")))
            fn = getattr(row, "fn", "") or ""
            mtime = ""
            try:
                if fn and _os.path.exists(str(fn)):
                    mtime = str(int(_os.path.getmtime(str(fn))))
            except OSError:  # pragma: no cover -- defensive
                mtime = ""
            h.update(f"{tf_id}|{fn}|{mtime}".encode())
    else:
        for mt in mt_collection_or_list:
            sid = str(getattr(mt, "station", id(mt)))
            h.update(sid.encode())
    return h.hexdigest()[:16]


def _git_sha() -> str:
    """Decomposition module's git sha when available; ``"unknown"``
    otherwise.
    """
    try:
        import subprocess

        sha = subprocess.run(
            ["git", "-C", _os.path.dirname(__file__), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if sha.returncode == 0:
            return sha.stdout.strip()[:12]
    except Exception:  # pragma: no cover -- defensive
        pass
    return "unknown"


def _mtpy_version() -> str:
    try:
        import mtpy

        return str(getattr(mtpy, "__version__", "unknown"))
    except Exception:  # pragma: no cover -- defensive
        return "unknown"


def _build_metadata(
    *,
    seed: int,
    canonical_gauge: str,
    n_starts: int,
    period_bands: list[_PeriodBand],
    input_identifier: str,
    band_overlap_fraction: float = 0.0,
) -> dict[str, Any]:
    return {
        "mtpy_version": _mtpy_version(),
        "decomposition_git_sha": _git_sha(),
        "timestamp_utc": _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat(),
        "method_versions": {
            "groom_bailey": "phase_1",
            "bibby": "phase_1",
            "lilley": "phase_1",
            "marti": "phase_1",
            "lilley_dimensionality": "phase_1",
            "magnetic_distortion_diagnostic": "phase_1",
        },
        "period_bands": [b.to_dict() for b in period_bands],
        "n_bands": len(period_bands),
        "band_overlap_fraction": float(band_overlap_fraction),
        "canonical_gauge": canonical_gauge,
        "rng_seed": int(seed),
        "n_starts": int(n_starts),
        "input_identifier": str(input_identifier),
    }


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame uses :data:`OBSERVABLE_DTYPES` for
    schema-fixed columns. Float64 stays the default for everything
    else.
    """
    for col, dtype in OBSERVABLE_DTYPES.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (TypeError, ValueError):  # pragma: no cover -- defensive
                pass
    return df


def compute_collection_observables(
    mt_collection_or_list,
    *,
    period_bands=None,
    canonical_gauge: str = "pt_aligned",
    n_starts: int = 5,
    seed: int = 42,
    skip_failed: bool = True,
    band_overlap_fraction: float = 0.0,
) -> ObservableTable:
    """Continental-scale observable table across an MT collection.

    Iterates over every site in ``mt_collection_or_list``, calls
    :func:`compute_site_observables`, and assembles the results into
    a long-format :class:`ObservableTable` whose schema is fixed by
    :data:`OBSERVABLE_COLUMNS`.

    Parameters
    ----------
    mt_collection_or_list : MTCollection or list[MT]
        Source of sites. If an :class:`MTCollection`, sites are
        iterated via ``.dataframe`` + ``.get_tf``; otherwise the
        argument is treated as an iterable of ``MT`` objects.
    period_bands, canonical_gauge, n_starts, seed,
    band_overlap_fraction
        See :func:`compute_site_observables`.
    skip_failed : bool, default True
        If True, sites whose pipeline raises are logged into the
        metadata's ``failed_sites`` list and skipped. If False,
        the first raise propagates.

    Returns
    -------
    ObservableTable
    """
    bands = _coerce_period_bands(period_bands, band_overlap_fraction)
    all_rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    site_count = 0
    for mt_object in _iter_mt_objects(mt_collection_or_list):
        site_count += 1
        try:
            site_obs = compute_site_observables(
                mt_object,
                period_bands=bands,
                canonical_gauge=canonical_gauge,
                n_starts=n_starts,
                seed=seed,
                band_overlap_fraction=band_overlap_fraction,
            )
        except Exception as exc:
            failed.append(
                {
                    "site_id": str(getattr(mt_object, "station", "")),
                    "reason": repr(exc),
                }
            )
            if not skip_failed:
                raise
            continue
        all_rows.extend(site_obs.band_observables)

    df = pd.DataFrame(all_rows, columns=OBSERVABLE_COLUMNS)
    df = _coerce_dtypes(df)

    metadata = _build_metadata(
        seed=seed,
        canonical_gauge=canonical_gauge,
        n_starts=n_starts,
        period_bands=bands,
        input_identifier=_hash_collection(mt_collection_or_list),
        band_overlap_fraction=band_overlap_fraction,
    )
    metadata["site_count"] = int(site_count)
    metadata["failed_sites"] = failed

    return ObservableTable(dataframe=df, metadata=metadata)
