"""Unified phase-tensor / Bahr-eigenvector / Mohr-circle
dimensionality classifier per Lilley 2020.

Lilley 2020 (*Exploration Geophysics* 51:4, 401-421) establishes
the formal equivalence between three apparently-distinct
dimensionality tools:

    1. **Phase-tensor analysis** (Caldwell-Bibby-Brown 2004): the
       phase tensor ``Phi = X^{-1} Y`` (with ``X = Re(Z)``,
       ``Y = Im(Z)``) is a real 2x2 dimensionality-revealing
       tensor whose principal-axis angle ``alpha`` is the regional
       strike candidate and whose skew angle ``beta`` measures
       3-D-ness.
    2. **Bahr 1988 strike determination**: a strike is the
       direction in which the impedance tensor's regional 2-D
       ``Z_TE`` and ``Z_TM`` modes are decoupled. Bahr derived this
       from rotational invariants of ``Z`` itself.
    3. **Mohr-circle representation of the phase tensor** (Lilley
       1976, 2018, 2020): the phase tensor traces a circle in
       ``(Phi'_xy, Phi'_xx)`` space under measurement-axis
       rotation; the centroid offset from the horizontal axis is
       the same 3-D-ness measure as the CBB skew.

Lilley 2020 shows these three give *the same* dimensionality
information — the eigenvectors of ``Phi`` are Bahr's strike
directions, ``alpha`` is the angle of the major eigenvector, and
``mu_Mohr = 2 * beta_CBB`` to numerical precision. This module
exposes all three views from a single ``Phi`` and provides a
unified classifier.

This complements :mod:`...marti` (which classifies via the WAL
rotational invariants). The two traditions should agree on
classification when the data are well-conditioned; disagreement
is itself diagnostic. The :func:`compare_lilley_marti` helper
produces a quantitative agreement rate.

Caveat: galvanic distortion is gauge-invisible to the phase tensor
by construction (Caldwell et al. 2004). A 2-D regional + galvanic
distortion thus classifies as "2D" via this module, even though
Marti's WAL invariants would classify it as "3D-2D" (since WAL
sees the distortion). The user-facing comparison helper documents
this mapping so the apparent disagreement is interpreted
correctly.

References
----------
Caldwell, T. G., Bibby, H. M., & Brown, C. (2004). The
magnetotelluric phase tensor. *Geophysical Journal International*
158, 457-469.

Bahr, K. (1988). Interpretation of the magnetotelluric impedance
tensor: regional induction and local telluric distortion. *Journal
of Geophysics* 62(2), 119-127.

Lilley, F. E. M. (2020). Magnetotellurics: the CBB or phase tensor
and Bahr's 1988 analysis. *Exploration Geophysics* 51(4), 401-421.
doi:10.1080/08123985.2020.1717333

Caveats
=======
* **WALDIM and Lilley categories disagree by design** for
  galvanically-distorted 2-D sites. Galvanic distortion is
  gauge-invisible to the phase tensor (Caldwell et al. 2004), so
  a 2-D + galvanic site is ``"2D"`` to Lilley but a 3-D-flavoured
  case (3, 4, 6, 7) in Marti's WALDIM. **The disagreement is
  expected, not a data quality flag**; the
  :func:`compare_lilley_marti` helper makes this mapping
  explicit and the
  :data:`...continental_observables.OBSERVABLE_COLUMNS`
  ``dimensionality_concordant`` column will read ``False`` on
  clean 2-D-distorted sites for the same reason.
* **3-D sites with non-symmetric Φ produce complex eigenvalues**;
  :func:`eigenvector_strike` returns ``NaN`` in that case. The
  classifier prioritises ``|β|`` for 3-D detection and treats
  the NaN eigenvector-disagreement as a 3-D signature (mapped
  to ``"3D-2D"`` when ``β`` is small).
* **Configurable threshold defaults**:
  :data:`BETA_2D_THRESHOLD_DEG` = 3.0,
  :data:`BETA_1D_THRESHOLD_DEG` = 1.0,
  :data:`ELL_1D_THRESHOLD` = 0.05. Reasonable but not measured
  against ground truth. Sensitivity to threshold choice is part
  of Paper 1's robustness analysis — pass kwargs to
  :func:`classify_dimensionality` to vary them, and
  :func:`...dimensionality_stratification.threshold_sensitivity`
  to sweep stratification thresholds (those are different).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .results import LilleyDimensionalityResult

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


__all__ = [
    "BETA_1D_THRESHOLD_DEG",
    "BETA_2D_THRESHOLD_DEG",
    "ELL_1D_THRESHOLD",
    "EIGENVECTOR_DISAGREE_THRESHOLD_DEG",
    "classify_dimensionality",
    "compare_lilley_marti",
    "eigenvector_strike",
    "mohr_circle_phase_tensor",
    "phase_tensor",
    "phase_tensor_invariants",
]


# Default thresholds (configurable via classify_dimensionality kwargs).
BETA_1D_THRESHOLD_DEG: float = 1.0
BETA_2D_THRESHOLD_DEG: float = 3.0
ELL_1D_THRESHOLD: float = 0.05
EIGENVECTOR_DISAGREE_THRESHOLD_DEG: float = 5.0


# ---------------------------------------------------------------------------
# Phase tensor and its invariants
# ---------------------------------------------------------------------------


def phase_tensor(z: np.ndarray) -> np.ndarray:
    """Per-period CBB phase tensor ``Phi = X^{-1} Y``.

    Parameters
    ----------
    z : (n_periods, 2, 2) complex ndarray

    Returns
    -------
    phi : (n_periods, 2, 2) float ndarray
        Real-valued (``Phi`` is real by construction). Per-period
        ``X = Re(Z)``, ``Y = Im(Z)``, ``Phi = X^{-1} Y``.
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    if z_arr.ndim != 3 or z_arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"phase_tensor: expected z of shape (n_periods, 2, 2); "
            f"got {z_arr.shape}"
        )
    n_periods = z_arr.shape[0]
    phi = np.empty((n_periods, 2, 2), dtype=np.float64)
    for k in range(n_periods):
        X = z_arr[k].real
        Y = z_arr[k].imag
        try:
            phi[k] = np.linalg.solve(X, Y)
        except np.linalg.LinAlgError:
            phi[k] = np.full((2, 2), np.nan)
    return phi


def phase_tensor_invariants(phi: np.ndarray) -> dict[str, np.ndarray]:
    """CBB phase-tensor invariants per period.

    Implements the Caldwell-Bibby-Brown 2004 angle definitions:

        alpha = (1/2) arctan2(Phi_xy + Phi_yx, Phi_xx - Phi_yy)
        beta  = (1/2) arctan2(Phi_xy - Phi_yx, Phi_xx + Phi_yy)

    Eigenvalues of the symmetric part are returned as
    ``lambda_max``, ``lambda_min``; the ``ellipticity`` is the
    standard ``(lambda_max - lambda_min) / (lambda_max + lambda_min)``.

    Returns
    -------
    dict with keys ``alpha_deg``, ``beta_deg``, ``lambda_max``,
    ``lambda_min``, ``ellipticity``.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    n_periods = phi_arr.shape[0]
    alpha = np.empty(n_periods)
    beta = np.empty(n_periods)
    lam_max = np.empty(n_periods)
    lam_min = np.empty(n_periods)
    ell = np.empty(n_periods)
    for k in range(n_periods):
        p = phi_arr[k]
        alpha[k] = np.degrees(
            0.5 * np.arctan2(p[0, 1] + p[1, 0], p[0, 0] - p[1, 1])
        )
        beta[k] = np.degrees(
            0.5 * np.arctan2(p[0, 1] - p[1, 0], p[0, 0] + p[1, 1])
        )
        sym = 0.5 * (p + p.T)
        eigvals = np.linalg.eigvalsh(sym)
        lam_max[k] = float(eigvals[1])
        lam_min[k] = float(eigvals[0])
        denom = lam_max[k] + lam_min[k]
        ell[k] = (
            (lam_max[k] - lam_min[k]) / denom
            if abs(denom) > 1e-12
            else 0.0
        )
    return {
        "alpha_deg": alpha,
        "beta_deg": beta,
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "ellipticity": ell,
    }


def eigenvector_strike(phi: np.ndarray) -> dict[str, np.ndarray]:
    """Real eigenvectors of ``Phi`` and their angle of disagreement.

    Per Lilley 2020, the eigenvectors of the (generally non-
    symmetric) phase tensor are Bahr 1988's regional strike
    directions. In 2-D the two eigenvectors are exactly
    perpendicular (``angle = 90°``); in 3-D they may deviate.

    Returns
    -------
    dict with keys:
      ``strike_alpha1_deg`` : (n_periods,) float ndarray
        Major-axis eigenvector angle (degrees, mod 180).
      ``strike_alpha2_deg`` : (n_periods,) float ndarray
        Minor-axis eigenvector angle (degrees, mod 180).
      ``eigenvector_angle_deg`` : (n_periods,) float ndarray
        Actual angle between the two eigenvectors (90° in 2-D).
      ``eigenvector_disagreement_deg`` : (n_periods,) float ndarray
        ``|90 − eigenvector_angle|`` — deviation from
        perpendicularity. Zero in 2-D, larger in 3-D. This is the
        primary 3-D-ness summary across the band.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    n_periods = phi_arr.shape[0]
    s1 = np.full(n_periods, np.nan)
    s2 = np.full(n_periods, np.nan)
    angle = np.full(n_periods, np.nan)
    disagree = np.full(n_periods, np.nan)

    for k in range(n_periods):
        p = phi_arr[k]
        eigvals, eigvecs = np.linalg.eig(p)
        if not np.all(np.isreal(eigvals)):
            # Complex eigenvalues — strike undefined in this
            # standard sense; leave NaN.
            continue
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        order = np.argsort(-eigvals)
        v1 = eigvecs[:, order[0]]
        v2 = eigvecs[:, order[1]]
        s1[k] = float(np.degrees(np.arctan2(v1[1], v1[0])) % 180.0)
        s2[k] = float(np.degrees(np.arctan2(v2[1], v2[0])) % 180.0)
        nrm1 = np.linalg.norm(v1)
        nrm2 = np.linalg.norm(v2)
        if nrm1 < 1e-12 or nrm2 < 1e-12:
            continue
        # Use |dot| to compare lines (eigenvectors are direction-
        # ambiguous): dot=|v1||v2| → parallel (angle=0°); dot=0 →
        # perpendicular (angle=90°). For 2-D Phi the eigenvectors
        # are exactly perpendicular so angle≈90° and
        # disagreement≈0°.
        cos_angle = abs(np.dot(v1, v2)) / (nrm1 * nrm2)
        cos_angle = float(np.clip(cos_angle, 0.0, 1.0))
        angle[k] = float(np.degrees(np.arccos(cos_angle)))
        disagree[k] = abs(90.0 - angle[k])
    return {
        "strike_alpha1_deg": s1,
        "strike_alpha2_deg": s2,
        "eigenvector_angle_deg": angle,
        "eigenvector_disagreement_deg": disagree,
    }


def mohr_circle_phase_tensor(phi: np.ndarray) -> dict[str, np.ndarray]:
    """Mohr circle of the phase tensor.

    The Mohr circle of ``Phi`` parametrises the per-rotation-angle
    change of ``Phi`` as a circle in ``(Phi'_xy, Phi'_xx)`` space.
    Centroid coordinates are

        center_x = (Phi_xx + Phi_yy) / 2
        center_y = (Phi_xy − Phi_yx) / 2

    and the radius is

        C = (1/2) sqrt((Phi_xx − Phi_yy)² + (Phi_xy + Phi_yx)²).

    Lilley 2020 establishes the equivalences

        mu_deg     = 2 * beta_CBB    (centroid angle from horizontal)
        radius     = (lambda_max − lambda_min) / 2  (for symmetric Phi)
        lambda_a   = arcsin(C / Z^L)  (anisotropy / 2-D-ness angle)

    where ``Z^L = sqrt(center_x² + center_y²)``.

    Returns
    -------
    dict with keys ``center_x``, ``center_y``, ``radius``,
    ``mu_deg``, ``lambda_a_deg``.
    """
    phi_arr = np.asarray(phi, dtype=np.float64)
    n_periods = phi_arr.shape[0]
    cx = np.empty(n_periods)
    cy = np.empty(n_periods)
    rad = np.empty(n_periods)
    mu = np.empty(n_periods)
    lam_a = np.empty(n_periods)
    for k in range(n_periods):
        p = phi_arr[k]
        cx[k] = 0.5 * (p[0, 0] + p[1, 1])
        cy[k] = 0.5 * (p[0, 1] - p[1, 0])
        rad[k] = 0.5 * np.sqrt(
            (p[0, 0] - p[1, 1]) ** 2 + (p[0, 1] + p[1, 0]) ** 2
        )
        mu[k] = float(np.degrees(np.arctan2(cy[k], cx[k])))
        z_l = float(np.hypot(cx[k], cy[k]))
        if z_l > 1e-12:
            ratio = float(np.clip(rad[k] / z_l, -1.0, 1.0))
            lam_a[k] = float(np.degrees(np.arcsin(ratio)))
        else:
            lam_a[k] = float("nan")
    return {
        "center_x": cx,
        "center_y": cy,
        "radius": rad,
        "mu_deg": mu,
        "lambda_a_deg": lam_a,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify_one_period(
    *,
    beta_deg: float,
    ellipticity: float,
    eigenvector_disagreement_deg: float,
    beta_1d_threshold: float,
    beta_2d_threshold: float,
    ell_1d_threshold: float,
    eigenvector_disagree_threshold: float,
) -> str:
    # Without finite beta / ellipticity we cannot say anything.
    if not np.isfinite(beta_deg) or not np.isfinite(ellipticity):
        return "indeterminate"

    abs_beta = abs(beta_deg)

    # 3-D first: large skew. (Strongly 3-D phase tensors often have
    # complex eigenvalues — eigenvector_disagreement_deg may be NaN
    # — but the dimensionality is still unambiguously 3-D given
    # large beta.)
    if abs_beta >= beta_2d_threshold:
        return "3D"

    # 1-D: small skew and small ellipticity.
    if (
        abs_beta < beta_1d_threshold
        and ellipticity < ell_1d_threshold
    ):
        return "1D"

    # Approximately 2-D regime (small skew). Eigenvectors should be
    # close to perpendicular; disagreement above threshold pushes
    # to "3D-2D" (Lilley's "approximately 2-D" sub-case where the
    # eigenvectors deviate but skew is still small).
    if not np.isfinite(eigenvector_disagreement_deg):
        # Symmetric Phi has perpendicular eigenvectors by
        # construction; if the call returned NaN here it means the
        # eigvecs of the full (non-symmetric) Phi were complex,
        # which is a 3-D-flavoured signature even though beta is
        # small. Map to "3D-2D".
        return "3D-2D"

    if eigenvector_disagreement_deg < eigenvector_disagree_threshold:
        return "2D"
    return "3D-2D"


def classify_dimensionality(
    z_object: "Z",
    *,
    site: str = "",
    periods: tuple[float, float] | None = None,
    beta_1d_threshold_deg: float = BETA_1D_THRESHOLD_DEG,
    beta_2d_threshold_deg: float = BETA_2D_THRESHOLD_DEG,
    ell_1d_threshold: float = ELL_1D_THRESHOLD,
    eigenvector_disagree_threshold_deg: float = EIGENVECTOR_DISAGREE_THRESHOLD_DEG,
) -> LilleyDimensionalityResult:
    """Per-period dimensionality classification via the unified
    Lilley 2020 framework.

    Returns
    -------
    LilleyDimensionalityResult
    """
    z_arr = np.asarray(z_object.z, dtype=np.complex128)
    freq = np.asarray(z_object.frequency, dtype=np.float64)
    pers = 1.0 / freq

    if periods is not None:
        pmin, pmax = periods
        mask = (pers >= pmin) & (pers <= pmax)
        if not mask.any():
            raise ValueError(
                f"classify_dimensionality: no periods in window "
                f"[{pmin}, {pmax}]; available "
                f"[{pers.min():.3g}, {pers.max():.3g}]"
            )
        z_arr = z_arr[mask]
        pers = pers[mask]

    sort_idx = np.argsort(pers)
    pers = pers[sort_idx]
    z_arr = z_arr[sort_idx]

    phi = phase_tensor(z_arr)
    inv = phase_tensor_invariants(phi)
    eig = eigenvector_strike(phi)
    mohr = mohr_circle_phase_tensor(phi)

    n_periods = pers.size
    classification: list[str] = []
    for k in range(n_periods):
        classification.append(
            _classify_one_period(
                beta_deg=float(inv["beta_deg"][k]),
                ellipticity=float(inv["ellipticity"][k]),
                eigenvector_disagreement_deg=float(
                    eig["eigenvector_disagreement_deg"][k]
                ),
                beta_1d_threshold=beta_1d_threshold_deg,
                beta_2d_threshold=beta_2d_threshold_deg,
                ell_1d_threshold=ell_1d_threshold,
                eigenvector_disagree_threshold=(
                    eigenvector_disagree_threshold_deg
                ),
            )
        )

    eigenvector_strikes = np.column_stack(
        [eig["strike_alpha1_deg"], eig["strike_alpha2_deg"]]
    )

    return LilleyDimensionalityResult(
        site=site,
        periods=pers,
        phase_tensor=phi,
        pt_alpha_deg=inv["alpha_deg"],
        pt_beta_deg=inv["beta_deg"],
        pt_ellipticity=inv["ellipticity"],
        pt_lambda_max=inv["lambda_max"],
        pt_lambda_min=inv["lambda_min"],
        eigenvector_strikes=eigenvector_strikes,
        eigenvector_disagreement_deg=eig["eigenvector_disagreement_deg"],
        mohr_lambda_a_deg=mohr["lambda_a_deg"],
        mohr_mu_deg=mohr["mu_deg"],
        classification=classification,
        classification_thresholds={
            "beta_1d_threshold_deg": beta_1d_threshold_deg,
            "beta_2d_threshold_deg": beta_2d_threshold_deg,
            "ell_1d_threshold": ell_1d_threshold,
            "eigenvector_disagree_threshold_deg": (
                eigenvector_disagree_threshold_deg
            ),
        },
    )


# ---------------------------------------------------------------------------
# Cross-tradition agreement: Lilley vs Marti
# ---------------------------------------------------------------------------


# Documented mapping from Marti WALDIM codes to the Lilley categories.
# WALDIM 0 → indeterminate (per Marti 2009).
# WALDIM 1 → "1D".
# WALDIM 2 → "2D".
# WALDIM 3, 4, 6, 7 → "2D" (these are 2-D regional + galvanic
#   distortion sub-cases; galvanic distortion is gauge-invisible to
#   the phase tensor, so Lilley correctly classifies them as "2D").
# WALDIM 5 → "3D".
_WALDIM_TO_LILLEY: dict[int, str] = {
    0: "indeterminate",
    1: "1D",
    2: "2D",
    3: "2D",
    4: "2D",
    5: "3D",
    6: "2D",
    7: "2D",
}


def compare_lilley_marti(
    lilley_result: LilleyDimensionalityResult,
    marti_result,
) -> dict[str, Any]:
    """Cross-tradition agreement between Lilley 2020 (this module)
    and Marti / WALDIM.

    Galvanic distortion is gauge-invisible to the phase tensor by
    construction; WALDIM cases 3, 4, 6, 7 (3-D-distorted-2-D
    regimes) therefore map to Lilley ``"2D"`` for the comparison.
    The mapping is documented at module level
    (``_WALDIM_TO_LILLEY``).

    Parameters
    ----------
    lilley_result : LilleyDimensionalityResult
    marti_result : MartiResult
        From :func:`...marti.decompose_marti`.

    Returns
    -------
    dict
        Keys:

        * ``n_periods`` (int)
        * ``n_agree`` (int)
        * ``agreement_rate`` (float, 0 to 1)
        * ``per_period`` : list of dicts with
          ``{period, lilley, marti_code, marti_mapped, agree}``.
    """
    marti_codes = np.asarray(marti_result.dimensionality, dtype=np.int64)
    lilley_labels = list(lilley_result.classification)
    n = min(len(lilley_labels), marti_codes.size)
    if n == 0:
        return {
            "n_periods": 0,
            "n_agree": 0,
            "agreement_rate": float("nan"),
            "per_period": [],
        }

    n_agree = 0
    per_period = []
    for k in range(n):
        lab_l = lilley_labels[k]
        code_m = int(marti_codes[k])
        lab_m = _WALDIM_TO_LILLEY.get(code_m, "indeterminate")
        agree = lab_l == lab_m
        if agree:
            n_agree += 1
        per_period.append(
            {
                "period": float(lilley_result.periods[k]),
                "lilley": lab_l,
                "marti_code": code_m,
                "marti_mapped": lab_m,
                "agree": agree,
            }
        )
    return {
        "n_periods": n,
        "n_agree": n_agree,
        "agreement_rate": n_agree / n,
        "per_period": per_period,
    }
