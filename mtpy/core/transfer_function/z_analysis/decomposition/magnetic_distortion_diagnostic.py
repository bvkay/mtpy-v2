"""Heuristic flag for sites where magnetic galvanic distortion may
violate the standard MT decomposition's "magnetic distortion is
negligible" assumption.

.. warning::

    **Heuristic, not corrective.** This module's outputs flag sites
    where the standard galvanic-distortion model (Groom-Bailey,
    McNeice-Jones, Bibby-Caldwell-Brown — all of which assume
    magnetic distortion ``Q_h``, ``Q_z`` are negligible) may be
    violated. The flag does *not* prove magnetic distortion is
    present, and it does *not* correct for it. Two intended uses:

    1. Exclude flagged sites from continental aggregation.
    2. Annotate flagged sites in summary plots / data quality
       overlays.

    Do **not** use the flag to "correct" individual-site analyses
    or to drive site-specific modelling. The proper handling of
    magnetic galvanic distortion requires Bayesian inversion with
    explicit ``Q_h``, ``Q_z`` priors (Garcia, Boerner & Pedersen
    2003) and is Paper 6 / a future PR; this module is a
    pre-Paper-6 triage tool.

The three diagnostics
=====================

The literature (Chave & Smith 1994; Garcia 2003) identifies three
practical signatures of breakdown of the magnetic-distortion-
negligible assumption:

1. **Anomalous Tipper** — the vertical-magnetic transfer function
   ``T = (T_zx, T_zy)`` should be small (``|T| < ~0.1-0.2``) for
   sites in regional 1-D / 2-D structure. Strong Tippers at
   intermediate periods (where galvanic-magnetic coupling is
   non-negligible but inductive coupling has not yet dominated
   the regional response) are a Garcia-2003 signature of
   near-surface 3-D conductivity contrasts in resistive
   environments.
2. **Strong frequency dependence of the recovered ``C`` tensor**
   beyond what frequency-dependent regional structure can
   explain. ``C`` is the *galvanic* distortion and is by
   definition frequency-independent in the static-galvanic
   model; large band-to-band variation indicates either magnetic
   distortion contaminating the inversion or a violated
   underlying assumption.
3. **Cross-method inconsistency**: methods that allow 3-D
   regional structure (Garcia-Jones) recover a different ``Z``
   than methods that assume 2-D regional + electric-only
   distortion (GB, MJ). The disagreement is informative even
   when neither method is "correct" in absolute terms.

The combined flag follows a simple rule:

    high_risk     : 2 or more diagnostics flag, OR peak Tipper
                    magnitude exceeds the strong-tipper override
                    (default 0.5)
    moderate_risk : exactly one diagnostic flags
    low_risk      : zero diagnostics flag
    indeterminate : insufficient data — Tipper missing, or fewer
                    than 3 bands for the C frequency-dependence
                    test, or no cross-method comparison supplied

Thresholds are configurable via keyword arguments and exposed as
module-level constants for tuning by researchers.

References
----------
Chave, A. D., & Smith, J. T. (1994). On electric and magnetic
galvanic distortion tensor decompositions. *Journal of Geophysical
Research* 99(B3), 4669-4682.

Garcia, X., Boerner, D., & Pedersen, L. B. (2003). Electric and
magnetic galvanic distortion decomposition of tensor CSAMT data.
*Geophysical Journal International* 154, 957-969.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .results import MagneticDistortionFlag

if TYPE_CHECKING:
    from mtpy.core.mt import MT


__all__ = [
    "compute_magnetic_distortion_flag",
    "frequency_dependence_diagnostic",
    "method_inconsistency_diagnostic",
    "tipper_diagnostic",
    # Configurable thresholds — exposed as module-level constants
    # so researchers can tune without editing this file's API.
    "DEFAULT_TIPPER_FLAG_THRESHOLD",
    "DEFAULT_TIPPER_STRONG_OVERRIDE",
    "DEFAULT_TIPPER_FREQ_DEP_THRESHOLD",
    "DEFAULT_C_CV_THRESHOLD",
    "DEFAULT_METHOD_STRIKE_DISAGREE_DEG",
    "DEFAULT_METHOD_Z_TE_REL_DISAGREE",
]


# ---------------------------------------------------------------------------
# Default thresholds — tunable. Numbers chosen as Garcia 2003 /
# Chave 1994 informed defaults; researchers should re-tune when
# the survey-typical noise floor differs.
# ---------------------------------------------------------------------------

DEFAULT_TIPPER_FLAG_THRESHOLD: float = 0.3
"""Per-period |T| above this triggers the Tipper diagnostic.

Below 0.1 typically indicates 1-D / 2-D regional structure with no
magnetic-distortion concern; 0.3 is a conservative cut for
"unusually large" without yet being "definitively 3-D".
"""

DEFAULT_TIPPER_STRONG_OVERRIDE: float = 0.5
"""Peak |T| at or above this immediately upgrades the overall flag
to ``high_risk``, regardless of the other two diagnostics.

Garcia 2003 sites with magnetic distortion typically show |T| in
[0.4, 0.8] at intermediate periods.
"""

DEFAULT_TIPPER_FREQ_DEP_THRESHOLD: float = 0.5
"""Fractional band-to-band change in mean |T| above this triggers
the Tipper diagnostic via the frequency-dependence sub-test.

Computed as ``(max_band_mean − min_band_mean) / mean_band_mean``.
"""

DEFAULT_C_CV_THRESHOLD: float = 0.2
"""Coefficient of variation (std / mean of ``|C_ij|``) across bands
above this for any tensor element triggers the C-frequency-
dependence diagnostic.
"""

DEFAULT_METHOD_STRIKE_DISAGREE_DEG: float = 10.0
"""Cross-method strike disagreement (in degrees, after the GB
90-degree symmetry is resolved) above this triggers the cross-
method-inconsistency diagnostic.
"""

DEFAULT_METHOD_Z_TE_REL_DISAGREE: float = 0.2
"""Median fractional disagreement of |Z_TE| between methods above
this also triggers the cross-method-inconsistency diagnostic.
"""


def tipper_diagnostic(
    mt_object: "MT",
    period_range: tuple[float, float] | None = None,
    *,
    flag_threshold: float = DEFAULT_TIPPER_FLAG_THRESHOLD,
    strong_override: float = DEFAULT_TIPPER_STRONG_OVERRIDE,
    freq_dep_threshold: float = DEFAULT_TIPPER_FREQ_DEP_THRESHOLD,
) -> dict[str, Any]:
    """Tipper-based magnetic-distortion heuristic.

    Returns a dict with the per-period |T| statistics and a Boolean
    ``flagged`` summarising whether the diagnostic triggers.

    Parameters
    ----------
    mt_object : MT
        mtpy ``MT`` instance. Must have a populated ``Tipper``
        attribute. If absent or empty, returns ``{"available":
        False, ...}`` and ``flagged=False``.
    period_range : (pmin, pmax), optional
        Restrict the diagnostic to a period sub-window. ``None``
        (default) uses the full Tipper period range.

    Returns
    -------
    dict
        Keys: ``available`` (bool), ``mean_amplitude``,
        ``max_amplitude``, ``period_of_max`` (seconds),
        ``frequency_dependence`` (fractional change of |T| across
        the band), ``flagged`` (bool), ``strong_override``
        (whether peak amplitude triggers the high-risk override).
    """
    tipper = getattr(mt_object, "Tipper", None)
    if tipper is None:
        return {
            "available": False,
            "flagged": False,
            "strong_override": False,
            "reason": "mt_object has no Tipper attribute",
        }

    try:
        t_arr = np.asarray(tipper.tipper, dtype=np.complex128)
        freq = np.asarray(tipper.frequency, dtype=np.float64)
    except Exception as exc:  # pragma: no cover -- defensive
        return {
            "available": False,
            "flagged": False,
            "strong_override": False,
            "reason": f"Tipper data unreadable: {exc!r}",
        }

    if t_arr.size == 0 or not np.any(np.isfinite(t_arr)):
        return {
            "available": False,
            "flagged": False,
            "strong_override": False,
            "reason": "Tipper has no finite values",
        }

    periods = 1.0 / freq

    if period_range is not None:
        pmin, pmax = period_range
        mask = (periods >= pmin) & (periods <= pmax)
        if not mask.any():
            return {
                "available": False,
                "flagged": False,
                "strong_override": False,
                "reason": (
                    f"no Tipper periods in range [{pmin}, {pmax}]"
                ),
            }
        t_arr = t_arr[mask]
        periods = periods[mask]

    # Per-period |T| = sqrt(|T_zx|² + |T_zy|²). The Tipper array's
    # last dim is (1, 2) per mtpy convention; flatten conservatively.
    t_per_period = np.linalg.norm(
        t_arr.reshape(t_arr.shape[0], -1), axis=1
    )
    finite_mask = np.isfinite(t_per_period)
    if not finite_mask.any():
        return {
            "available": False,
            "flagged": False,
            "strong_override": False,
            "reason": "no finite |T| in window",
        }
    t_per_period = t_per_period[finite_mask]
    periods = periods[finite_mask]

    mean_amp = float(np.mean(t_per_period))
    max_amp = float(np.max(t_per_period))
    period_of_max = float(periods[int(np.argmax(t_per_period))])

    if mean_amp > 0:
        freq_dep = float(
            (np.max(t_per_period) - np.min(t_per_period)) / mean_amp
        )
    else:
        freq_dep = 0.0

    flag_amp = mean_amp > flag_threshold or max_amp > flag_threshold
    flag_freq_dep = freq_dep > freq_dep_threshold
    flagged = bool(flag_amp or flag_freq_dep)
    # Float tolerance on the override boundary: a value one ULP
    # below the threshold should still trigger.
    override_triggered = bool(
        max_amp >= strong_override - 1e-12
    )

    return {
        "available": True,
        "mean_amplitude": mean_amp,
        "max_amplitude": max_amp,
        "period_of_max": period_of_max,
        "frequency_dependence": freq_dep,
        "flagged": flagged,
        "strong_override": override_triggered,
        "thresholds": {
            "flag_threshold": flag_threshold,
            "strong_override": strong_override,
            "freq_dep_threshold": freq_dep_threshold,
        },
    }


def frequency_dependence_diagnostic(
    c_per_band: np.ndarray | list[np.ndarray],
    *,
    cv_threshold: float = DEFAULT_C_CV_THRESHOLD,
    band_periods: np.ndarray | None = None,
) -> dict[str, Any]:
    """Coefficient-of-variation diagnostic on per-band ``C``.

    A frequency-independent galvanic distortion gives ``C`` invariant
    across bands. Strong band-to-band variation in ``|C_ij|`` is
    informative — either the regional structure varies (handled by
    the GB / MJ band model) or magnetic distortion contributes
    frequency-dependent terms.

    Parameters
    ----------
    c_per_band : ndarray or list of ndarrays
        Per-band 2x2 distortion matrices, shape ``(n_bands, 2, 2)``
        or a list of ``(2, 2)`` arrays.
    cv_threshold : float
        Coefficient of variation threshold; the diagnostic flags
        when any element's CV exceeds this.
    band_periods : (n_bands,) ndarray, optional
        Band-representative periods (seconds). Used only to report
        the dominant period of variation if flagged.

    Returns
    -------
    dict
        Keys: ``available`` (bool), ``element_cv`` (``(2, 2)`` of
        per-element CV), ``max_element_cv`` (float),
        ``flagged`` (bool), ``dominant_period`` (seconds, or
        ``None`` if not flagged or not supplied).
    """
    c_arr = np.asarray(c_per_band, dtype=np.float64)
    if c_arr.ndim != 3 or c_arr.shape[-2:] != (2, 2):
        return {
            "available": False,
            "flagged": False,
            "reason": (
                f"expected c_per_band shape (n_bands, 2, 2); got "
                f"{c_arr.shape}"
            ),
        }
    if c_arr.shape[0] < 3:
        return {
            "available": False,
            "flagged": False,
            "reason": (
                f"frequency-dependence diagnostic needs >= 3 bands; "
                f"got {c_arr.shape[0]}"
            ),
        }

    abs_c = np.abs(c_arr)
    mean_per_element = np.mean(abs_c, axis=0)
    std_per_element = np.std(abs_c, axis=0, ddof=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(
            mean_per_element > 1e-12,
            std_per_element / mean_per_element,
            0.0,
        )

    max_cv = float(np.max(cv))
    flagged = bool(max_cv > cv_threshold)

    dominant_period: float | None = None
    if flagged and band_periods is not None:
        # Argmax of element with largest CV: which band has its
        # extreme value? Use the band with the largest deviation
        # from the mean.
        i, j = np.unravel_index(int(np.argmax(cv)), cv.shape)
        deviations = np.abs(abs_c[:, i, j] - mean_per_element[i, j])
        band_periods_arr = np.asarray(band_periods, dtype=np.float64)
        if band_periods_arr.size == c_arr.shape[0]:
            dominant_period = float(
                band_periods_arr[int(np.argmax(deviations))]
            )

    return {
        "available": True,
        "element_cv": cv,
        "max_element_cv": max_cv,
        "flagged": flagged,
        "dominant_period": dominant_period,
        "thresholds": {"cv_threshold": cv_threshold},
    }


def method_inconsistency_diagnostic(
    cross_method_result: dict[str, Any],
    *,
    strike_disagree_deg: float = DEFAULT_METHOD_STRIKE_DISAGREE_DEG,
    z_te_rel_disagree: float = DEFAULT_METHOD_Z_TE_REL_DISAGREE,
) -> dict[str, Any]:
    """Cross-method consistency check between GB / MJ and Garcia-
    Jones (or any pair of methods on the same site).

    Parameters
    ----------
    cross_method_result : dict
        Caller-supplied dict with keys:

        * ``strikes_deg`` : dict[str, float] — per-method strike
          (degrees). At least two entries required.
        * ``z_te_per_period`` : dict[str, ndarray] (optional) —
          per-method ``|Z_TE|`` array (per period). Two entries
          required for the |Z_TE| sub-test.

        Missing or single-method input → ``available=False``.
    strike_disagree_deg : float
        Strike disagreement (after resolving the GB 90° symmetry,
        i.e. mod 90°) above which the diagnostic flags.
    z_te_rel_disagree : float
        Median fractional disagreement of ``|Z_TE|`` between
        methods above which the diagnostic flags.

    Returns
    -------
    dict
        Keys: ``available``, ``strike_rms_disagree_deg``,
        ``z_te_median_rel_disagree``, ``flagged``,
        ``thresholds``.
    """
    strikes = cross_method_result.get("strikes_deg", {})
    if len(strikes) < 2:
        return {
            "available": False,
            "flagged": False,
            "reason": (
                f"cross-method comparison requires >= 2 methods' "
                f"strikes_deg; got {len(strikes)}"
            ),
        }

    methods = list(strikes.keys())
    deltas: list[float] = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a = float(strikes[methods[i]])
            b = float(strikes[methods[j]])
            d = abs(a - b) % 90.0
            d = min(d, 90.0 - d)
            deltas.append(d)
    strike_rms = float(np.sqrt(np.mean(np.array(deltas) ** 2)))

    z_te_med_rel: float | None = None
    z_te_dict = cross_method_result.get("z_te_per_period")
    if isinstance(z_te_dict, dict) and len(z_te_dict) >= 2:
        z_te_methods = list(z_te_dict.keys())
        rel_diffs: list[float] = []
        for i in range(len(z_te_methods)):
            for j in range(i + 1, len(z_te_methods)):
                a = np.abs(np.asarray(z_te_dict[z_te_methods[i]]))
                b = np.abs(np.asarray(z_te_dict[z_te_methods[j]]))
                if a.shape != b.shape or a.size == 0:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = np.abs(a - b) / np.maximum(a, b)
                d = d[np.isfinite(d)]
                if d.size:
                    rel_diffs.append(float(np.median(d)))
        if rel_diffs:
            z_te_med_rel = float(np.max(rel_diffs))

    flag_strike = strike_rms > strike_disagree_deg
    flag_z_te = z_te_med_rel is not None and z_te_med_rel > z_te_rel_disagree
    flagged = bool(flag_strike or flag_z_te)

    return {
        "available": True,
        "strike_rms_disagree_deg": strike_rms,
        "z_te_median_rel_disagree": z_te_med_rel,
        "flagged": flagged,
        "thresholds": {
            "strike_disagree_deg": strike_disagree_deg,
            "z_te_rel_disagree": z_te_rel_disagree,
        },
    }


def compute_magnetic_distortion_flag(
    mt_object: "MT",
    *,
    site: str | None = None,
    period_range: tuple[float, float] | None = None,
    c_per_band: np.ndarray | list[np.ndarray] | None = None,
    band_periods: np.ndarray | None = None,
    cross_method_result: dict[str, Any] | None = None,
    tipper_kwargs: dict[str, Any] | None = None,
    frequency_kwargs: dict[str, Any] | None = None,
    method_kwargs: dict[str, Any] | None = None,
) -> MagneticDistortionFlag:
    """Combine the three sub-diagnostics into a single per-site flag.

    See :class:`MagneticDistortionFlag` for the flag-combination
    rule and the heuristic-not-corrective disclaimer.

    Parameters
    ----------
    mt_object : MT
        mtpy ``MT`` instance for the site.
    site : str, optional
        Identifier; defaults to ``mt_object.station`` if available
        else ``""``.
    period_range : (pmin, pmax), optional
        Restrict all per-period diagnostics to this window.
    c_per_band : ndarray, optional
        Per-band ``(n_bands, 2, 2)`` ``C`` for the frequency-
        dependence diagnostic. ``None`` skips that diagnostic.
    band_periods : ndarray, optional
        Per-band representative periods, used only for reporting
        the dominant period of variation.
    cross_method_result : dict, optional
        Cross-method comparison input for
        :func:`method_inconsistency_diagnostic`. ``None`` skips
        that diagnostic.
    tipper_kwargs, frequency_kwargs, method_kwargs : dict, optional
        Per-diagnostic threshold overrides; forwarded to the
        respective functions.
    """
    site_id = site or getattr(mt_object, "station", "") or ""

    tip = tipper_diagnostic(
        mt_object, period_range=period_range,
        **(tipper_kwargs or {}),
    )
    freq = (
        frequency_dependence_diagnostic(
            c_per_band, band_periods=band_periods,
            **(frequency_kwargs or {}),
        )
        if c_per_band is not None
        else None
    )
    method = (
        method_inconsistency_diagnostic(
            cross_method_result, **(method_kwargs or {})
        )
        if cross_method_result is not None
        else None
    )

    contributing: list[str] = []
    flag_count = 0

    if tip.get("available"):
        if tip.get("flagged"):
            flag_count += 1
            contributing.append(
                f"Tipper: mean_amp={tip['mean_amplitude']:.3f}, "
                f"max_amp={tip['max_amplitude']:.3f}, "
                f"freq_dep={tip['frequency_dependence']:.3f}"
            )
        if tip.get("strong_override"):
            contributing.append(
                f"Tipper strong-override: max |T| = "
                f"{tip['max_amplitude']:.3f} >= "
                f"{tip['thresholds']['strong_override']}"
            )

    if freq is None:
        contributing.append(
            "frequency_dependence: not run (no c_per_band supplied)"
        )
    elif freq.get("available"):
        if freq.get("flagged"):
            flag_count += 1
            contributing.append(
                f"C frequency-dependence: max element CV = "
                f"{freq['max_element_cv']:.3f}"
            )
    else:
        contributing.append(
            f"frequency_dependence: {freq.get('reason', 'unavailable')}"
        )

    if method is None:
        contributing.append(
            "method_inconsistency: not run (no cross_method_result)"
        )
    elif method.get("available"):
        if method.get("flagged"):
            flag_count += 1
            contributing.append(
                f"method inconsistency: strike RMS = "
                f"{method['strike_rms_disagree_deg']:.2f}°"
            )
    else:
        contributing.append(
            f"method_inconsistency: "
            f"{method.get('reason', 'unavailable')}"
        )

    # Indeterminate if no diagnostic could even run.
    diagnostics_runnable = sum(
        d is not None and d.get("available", False)
        for d in (tip, freq, method)
    )

    if diagnostics_runnable == 0:
        overall = "indeterminate"
    elif tip.get("strong_override"):
        overall = "high_risk"
    elif flag_count >= 2:
        overall = "high_risk"
    elif flag_count == 1:
        overall = "moderate_risk"
    else:
        overall = "low_risk"

    return MagneticDistortionFlag(
        site=site_id,
        overall_flag=overall,
        tipper_diagnostic=tip,
        frequency_dependence_diagnostic=freq,
        method_inconsistency_diagnostic=method,
        contributing_factors=contributing,
    )
