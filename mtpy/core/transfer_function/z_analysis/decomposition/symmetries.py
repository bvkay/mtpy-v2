"""GB symmetry handling and mode disambiguation.

The Groom-Bailey forward model has a 90-degree strike / shear-sign
symmetry that produces an exact gauge equivalence:
``(strike, twist, shear) <-> (strike + 90 mod 180, twist, -shear)``
yields identical predicted impedances and identical galvanic-
distortion tensors. The two branches are observationally
indistinguishable; which one a single optimiser run reports depends
on starting point alone. Multi-start optimisation can additionally
converge to physically-distinct local minima ("modes") that are not
GB-symmetry-equivalent.

Why a strategy pattern, not a single default
--------------------------------------------
Different downstream applications need different conventions, and
those needs are mutually incompatible:

- Reproducing historical strike_py / McNeice-Jones Fortran output
  requires the geometric fold to ``[0, 90)``.
- Cross-tool comparison with the phase tensor (``alpha``, mod 180)
  requires the identity fold so both axes share the same strike
  range.
- Combining GB strikes with an external prior (a phase-tensor strike,
  a regional geological strike) requires picking the branch closest
  to that prior — a separate fold per band.
- Low-shear-preferring tiebreakers are useful when the data is
  consistent with no shear and the optimiser landed on a noisy
  non-zero value.

The empirical-reproducibility goal of the surrounding research
project additionally requires that this choice be *explicit*: a
silent default would mean two analysts with the same data and the
same package version can publish different strikes, with no
recoverable provenance for the discrepancy. This module therefore
exposes the choice as a named strategy with no hidden default.

Module contents
---------------
- Disambiguation strategies that resolve the GB 90-degree symmetry
  by choosing one of the two equivalent branches per band:

  * :func:`_geometric_fold` — historical default, folds strike into
    ``[0, pi/2)`` and flips shear sign when the fold triggers.
    Matches strike_py and the McNeice-Jones Fortran convention.
  * :func:`_identity_fold` — no fold; strike wraps to ``[0, pi)`` and
    shear keeps its optimised sign. Both branches remain accessible.
  * :func:`_pt_aligned_fold` — choose the branch whose strike is
    closest (mod pi) to a supplied phase-tensor strike.
  * :func:`_min_shear_fold` — choose the branch with smaller
    ``|shear|``, a low-shear-preferring tiebreaker.
  * :func:`_resolve_disambiguation` — dispatch a string identifier or
    a user-supplied callable to one of the above.

- :class:`_Mode` and clustering / probability helpers — group
  converged multi-start runs into modes for both single-site and
  joint decompositions. Clustering always uses :func:`_geometric_fold`
  internally so two starts differing only by the GB symmetry collapse
  to a single mode regardless of the user's reporting choice.
- Disagreement / primary-mode-warning detectors used by the public
  decompose path to surface ambiguous fits to the caller.

References
----------
Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.

McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency tensor
decomposition of magnetotelluric data. Geophysics, 66(1), 158-173.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .common import _BandResult, _unpack_x_joint


_DEFAULT_MODE_TOLERANCE = {
    "strike_deg": 0.5,
    "twist_deg": 0.5,
    "shear_deg": 0.5,
}
# Note: log10_gain was originally intended as a fourth discriminator
# but is gauge-equivalent at the band level (the forward model
# satisfies (a, b, gain) -> (gain*a, gain*b, 1)), so different starts
# converge to physically-equivalent solutions with different
# gain/|a|/|b| splits. Including it as a clustering criterion would
# fragment a single physical mode into multiple gauge-equivalent
# clusters; we exclude it.


def _geometric_fold(
    strike: float,
    twist: float,
    shear: float,
) -> tuple[float, float, float]:
    """Geometric / "canonical" fold of the GB 90-degree symmetry.

    Historical default. Folds strike into ``[0, pi/2)`` and flips the
    sign of shear when the fold triggers. Matches the strike_py
    reference's ``symmetry.compare_band_against_dcmp`` convention and
    the McNeice-Jones Fortran output.

    The Groom-Bailey decomposition has a discrete symmetry: the
    triple ``(strike, twist, shear)`` and ``(strike + 90 mod 180,
    twist, -shear)`` generate identical predicted impedances (twist
    is invariant; the strike shift by 90 degrees is paired with a
    sign flip on shear). This function chooses the branch with
    strike in ``[0, pi/2)``.

    Parameters
    ----------
    strike, twist, shear : float
        Radians.

    Returns
    -------
    strike_c, twist_c, shear_c : float
        Strike in ``[0, pi/2)``. Twist unchanged. Shear sign flipped
        iff the fold triggered.
    """
    half_pi = np.pi / 2.0
    # Tolerance for the boundary at strike = pi/2 (90 deg). Without
    # this, a strike that lands exactly at pi/2 can be rounded to one
    # ULP below pi/2 by ``% np.pi`` and miss the fold.
    fold_tol = 1e-9
    strike_mod_pi = strike % np.pi
    if strike_mod_pi >= half_pi - fold_tol:
        strike_c = strike_mod_pi - half_pi
        shear_c = -shear
        # Re-fold strike_c into [0, pi/2): if the input was very close
        # to pi/2 from below, strike_c is a tiny negative; map to 0.
        if strike_c < 0.0:
            strike_c = 0.0
    else:
        strike_c = strike_mod_pi
        shear_c = shear
    return float(strike_c), float(twist), float(shear_c)


def _identity_fold(
    strike: float,
    twist: float,
    shear: float,
) -> tuple[float, float, float]:
    """Identity fold: wrap strike to ``[0, pi)`` and leave shear alone.

    The GB 90-degree symmetry is *not* applied; both branches remain
    accessible in the reported parameters. Useful when the caller
    wants to compare the unfolded GB output directly against another
    tool that uses a 180-degree convention (e.g. the phase tensor).
    """
    return float(strike % np.pi), float(twist), float(shear)


def _pt_aligned_fold(
    strike: float,
    twist: float,
    shear: float,
    *,
    pt_strike_rad: float,
) -> tuple[float, float, float]:
    """Choose the GB branch whose strike is closest to ``pt_strike_rad``.

    Of the two equivalent branches
    ``(strike % pi, twist, shear)`` and
    ``((strike + pi/2) % pi, twist, -shear)``, return whichever has
    the smaller angular distance (mod pi) to ``pt_strike_rad``. Ties
    resolve to the un-rotated branch.

    Parameters
    ----------
    strike, twist, shear : float
        Radians.
    pt_strike_rad : float, keyword-only
        Phase-tensor strike (radians) for the same band/period.
        Required: callers without a PT reference should use
        :func:`_geometric_fold` or :func:`_min_shear_fold`.

    Returns
    -------
    strike_c, twist_c, shear_c : float
        Strike in ``[0, pi)``. Twist unchanged. Shear sign flipped
        iff the rotated branch was selected.
    """
    s0 = float(strike % np.pi)
    s1 = float((strike + np.pi / 2.0) % np.pi)
    target = float(pt_strike_rad % np.pi)

    def _dist_mod_pi(a: float, b: float) -> float:
        d = abs(a - b) % np.pi
        return min(d, np.pi - d)

    if _dist_mod_pi(s1, target) < _dist_mod_pi(s0, target):
        return s1, float(twist), float(-shear)
    return s0, float(twist), float(shear)


def _min_shear_fold(
    strike: float,
    twist: float,
    shear: float,
) -> tuple[float, float, float]:
    """Choose the GB branch with the smaller ``|shear|``.

    Useful as a low-shear-preferring tiebreaker when no external
    reference (PT strike, geological prior) is available. Since the
    rotated branch flips the sign of shear, ``|shear|`` is the same
    on both branches in pure form — but with finite-precision
    optimisation output the two are not exactly equal, so this fold
    is well-defined operationally and stable to a single canonical
    pick when fed downstream-rounded values.

    Returns
    -------
    strike_c, twist_c, shear_c : float
        Strike in ``[0, pi)``. Twist unchanged. Shear sign flipped
        iff the rotated branch had smaller ``|shear|``.
    """
    s0 = float(strike % np.pi)
    s1 = float((strike + np.pi / 2.0) % np.pi)
    if abs(-shear) < abs(shear):
        return s1, float(twist), float(-shear)
    return s0, float(twist), float(shear)


def _resolve_disambiguation(
    disambiguation: "str | callable",
    *,
    pt_strike_rad: float | None = None,
):
    """Map a disambiguation identifier to a fold callable.

    Parameters
    ----------
    disambiguation : str or callable
        One of ``"geometric"``, ``"identity"``, ``"pt_aligned"``,
        ``"min_shear"``, or a callable with the signature
        ``(strike, twist, shear) -> (strike_c, twist_c, shear_c)``
        (radians).
    pt_strike_rad : float, optional
        Required when ``disambiguation == "pt_aligned"``. Bound into
        the returned callable as the keyword argument.

    Returns
    -------
    fold : callable
        ``fold(strike, twist, shear) -> (strike_c, twist_c, shear_c)``.

    Raises
    ------
    ValueError
        If the string identifier is unknown, or ``"pt_aligned"`` is
        requested without ``pt_strike_rad``.
    """
    if callable(disambiguation):
        return disambiguation
    if disambiguation == "geometric":
        return _geometric_fold
    if disambiguation == "identity":
        return _identity_fold
    if disambiguation == "min_shear":
        return _min_shear_fold
    if disambiguation == "pt_aligned":
        if pt_strike_rad is None:
            raise ValueError(
                "_resolve_disambiguation: disambiguation='pt_aligned' "
                "requires pt_strike_rad (radians) to be supplied."
            )
        pt = float(pt_strike_rad)

        def _bound(strike, twist, shear):
            return _pt_aligned_fold(strike, twist, shear, pt_strike_rad=pt)

        return _bound
    raise ValueError(
        f"_resolve_disambiguation: unknown disambiguation "
        f"{disambiguation!r}. Expected one of 'geometric', 'identity', "
        f"'pt_aligned', 'min_shear', or a callable."
    )

@dataclass
class _Mode:
    """Internal type: one discovered mode of a multi-start fit.

    Each mode collects the converged starts that landed in the same
    physical basin (matching strike/twist/shear/gain within the
    user-configurable tolerances). The representative
    :class:`_BandResult` is the lowest-RMS member of the cluster.
    """

    rms_misfit: float
    chi_squared: float
    n_starts_landing_here: int
    band_result: _BandResult
    canonical_form: dict
    probability: float | None = None

def _canonical_form_summary(br: _BandResult) -> dict:
    """Extract canonical-form scalar parameters for clustering.

    Applies :func:`_geometric_fold` so that two converged points
    differing only by the GB 90-degree symmetry produce identical
    summaries. Mode clustering always uses the geometric fold
    regardless of the user's reporting choice.

    Returns a dict with keys: ``strike_deg``, ``twist_deg``,
    ``shear_deg``, ``log10_gain``, ``rms_misfit``.
    """
    strike_rad, twist_rad, shear_rad = _geometric_fold(
        float(br.x_opt[0]), float(br.x_opt[1]), float(br.x_opt[2])
    )
    return {
        "strike_deg": float(np.degrees(strike_rad)),
        "twist_deg": float(np.degrees(twist_rad)),
        "shear_deg": float(np.degrees(shear_rad)),
        "log10_gain": float(br.x_opt[3]),
        "rms_misfit": float(br.rms_misfit),
    }

def _modes_match_within_band(cf_a: dict, cf_b: dict, tolerance: dict) -> bool:
    """Three physical-parameter conditions, used during clustering
    of starts within a single band.

    Two converged points within a band are the same mode when
    strike, twist, and shear all agree within their respective
    tolerances. We deliberately exclude log10_gain (gauge-equivalent;
    see :data:`_DEFAULT_MODE_TOLERANCE` notes) and RMS (TRF's path-
    dependent numerical noise produces O(1e-4) relative RMS
    differences between starts converging to the same physical
    basin).
    """
    return (
        abs(cf_a["strike_deg"] - cf_b["strike_deg"]) <= tolerance["strike_deg"]
        and abs(cf_a["twist_deg"] - cf_b["twist_deg"]) <= tolerance["twist_deg"]
        and abs(cf_a["shear_deg"] - cf_b["shear_deg"]) <= tolerance["shear_deg"]
    )

def _modes_match_across_bands(cf_a: dict, cf_b: dict, tolerance: dict) -> bool:
    """Same four physical-parameter conditions, used to compare
    bands' primary modes.

    The within-band and across-band matchers check the same fields
    today; the function exists as a separate name because the
    intents at the two call sites differ. Within a band, RMS is
    the natural tie-breaker but we deliberately don't use it (see
    :func:`_modes_match_within_band` notes). Across bands, RMS is
    meaningless to compare because each band fits a different
    period subset, so any RMS-aware matcher would always report
    disagreement.
    """
    return _modes_match_within_band(cf_a, cf_b, tolerance)

def _cluster_modes(
    band_results: list[_BandResult],
    mode_tolerance: dict | None = None,
) -> list[_Mode]:
    """Greedy clustering of converged points into modes.

    Algorithm:
      1. Compute canonical-form summary for each
         :class:`_BandResult`.
      2. Sort summaries ascending by RMS misfit.
      3. The lowest-RMS point seeds mode 0 and becomes its
         representative.
      4. Each subsequent point is compared to all existing modes
         via :func:`_modes_match_within_band`; if it matches any,
         increment that mode's count, else open a new mode.

    By sorting first, each mode's representative is always the
    best converged point in its cluster.

    Parameters
    ----------
    band_results : list of _BandResult
    mode_tolerance : dict, optional
        Overrides for the per-parameter tolerances. Keys:
        ``strike_deg``, ``twist_deg``, ``shear_deg``. Defaults
        applied for missing keys. The deprecated ``rms_relative``
        and ``log10_gain`` keys raise :class:`UserWarning` and are
        ignored.

    Returns
    -------
    list of _Mode
        Sorted ascending by RMS misfit. Mode 0 is primary.
    """
    if not band_results:
        return []

    tol = dict(_DEFAULT_MODE_TOLERANCE)
    if mode_tolerance:
        deprecated = {"rms_relative", "log10_gain"} & set(mode_tolerance)
        if deprecated:
            warnings.warn(
                f"mode_tolerance keys {sorted(deprecated)} are no longer "
                "used and will be ignored. Clustering uses strike_deg, "
                "twist_deg, and shear_deg only: rms_relative was "
                "fragile under TRF's path-dependent noise, and "
                "log10_gain is gauge-equivalent with the regional "
                "impedance magnitudes at the band level.",
                UserWarning,
                stacklevel=2,
            )
            mode_tolerance = {
                k: v for k, v in mode_tolerance.items() if k not in deprecated
            }
        tol.update(mode_tolerance)

    summaries = [(br, _canonical_form_summary(br)) for br in band_results]
    summaries.sort(key=lambda x: x[1]["rms_misfit"])

    modes: list[_Mode] = []
    for br, cf in summaries:
        matched = False
        for mode in modes:
            if _modes_match_within_band(mode.canonical_form, cf, tol):
                mode.n_starts_landing_here += 1
                matched = True
                break
        if not matched:
            modes.append(
                _Mode(
                    rms_misfit=br.rms_misfit,
                    chi_squared=br.chi_squared,
                    n_starts_landing_here=1,
                    band_result=br,
                    canonical_form=cf,
                )
            )
    return modes

def _compute_mode_probabilities(modes: list[_Mode]) -> list[float]:
    """Laplace approximation of mode probabilities.

    For each mode i, the unnormalised log-weight is::

        log_w_i = -0.5 * chi_squared_i + 0.5 * log det(cov_i)

    where ``cov_i = (J_eff^T J_eff)^{-1}`` is the parameter
    covariance restricted to identifiable parameters (the
    anisotropy column of the Jacobian is identically zero, so we
    drop it from the determinant calculation).

    Probabilities are obtained by max-subtract on the log-weights
    for numerical stability::

        log_w_max = max(log_w_i)
        w_i = exp(log_w_i - log_w_max)
        p_i = w_i / sum_j w_j

    Parameters
    ----------
    modes : list of _Mode
        Each mode's ``band_result.jacobian`` must be populated.

    Returns
    -------
    list of float
        Mode probabilities, summing to 1, in the same order as
        the input.

    Notes
    -----
    The Laplace approximation assumes a Gaussian likelihood near
    each mode. For MT decomposition the residuals are complex-
    Gaussian by assumption, so the approximation is reasonable.

    For modes whose effective Gram matrix is rank-deficient (more
    than the anisotropy column unidentifiable), the determinant
    is evaluated via eigenvalues of the pseudo-inverse with a
    small floor.
    """
    if not modes:
        return []
    if len(modes) == 1:
        return [1.0]

    log_weights: list[float] = []
    for mode in modes:
        chi_sq = mode.chi_squared
        J = mode.band_result.jacobian
        if J is None:
            # Should not happen in normal use; fall back to a
            # likelihood-only weight.
            log_weights.append(-0.5 * chi_sq)
            continue
        # Drop anisotropy column (index 4) from the Jacobian for
        # the determinant: it's identically zero so det(J^T J)
        # would otherwise be exactly zero.
        J_eff = np.delete(J, 4, axis=1)
        gram = J_eff.T @ J_eff
        sign, log_det_gram = np.linalg.slogdet(gram)
        if sign > 0:
            log_det_cov = -log_det_gram
        else:
            cov = np.linalg.pinv(gram)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-30]
            if eigenvalues.size == 0:
                log_det_cov = -np.inf
            else:
                log_det_cov = float(np.sum(np.log(eigenvalues)))
        log_weights.append(-0.5 * chi_sq + 0.5 * log_det_cov)

    log_w_max = max(log_weights)
    weights = [float(np.exp(lw - log_w_max)) for lw in log_weights]
    total = sum(weights)
    if total <= 0.0:
        # Degenerate; fall back to uniform.
        return [1.0 / len(modes)] * len(modes)
    return [w / total for w in weights]

def _detect_band_disagreement(
    band_modes_list: list[tuple[np.ndarray, list[_Mode]]],
    mode_tolerance: dict,
) -> dict | None:
    """Check whether bands' primary modes agree on physical params.

    Uses :func:`_modes_match_across_bands`, which checks
    strike/twist/shear/gain only (RMS comparison is meaningless
    across bands fitting different period subsets).

    Returns ``None`` if all primaries agree. Otherwise returns a
    dict listing each disagreeing band pair with their canonical-
    form summaries.
    """
    if len(band_modes_list) < 2:
        return None

    tol = dict(_DEFAULT_MODE_TOLERANCE)
    if mode_tolerance:
        # Tolerate deprecated keys silently here; _cluster_modes has
        # already warned upstream.
        tol.update(
            {
                k: v
                for k, v in mode_tolerance.items()
                if k not in ("rms_relative", "log10_gain")
            }
        )

    primaries = [modes[0].canonical_form for _, modes in band_modes_list]

    disagreements: list[dict] = []
    for i in range(len(primaries)):
        for j in range(i + 1, len(primaries)):
            if not _modes_match_across_bands(primaries[i], primaries[j], tol):
                disagreements.append(
                    {
                        "bands": [i, j],
                        "values_i": primaries[i],
                        "values_j": primaries[j],
                    }
                )

    if not disagreements:
        return None

    return {
        "disagreements": disagreements,
        "interpretation": (
            "Bands' primary modes disagree on canonical-form "
            "physical parameters. This may indicate model "
            "misspecification (deeper/shallower structure differs "
            "across the period range) or that one band's primary "
            "mode is not the global minimum. Inspect "
            "metadata['per_band'] to see all modes per band."
        ),
    }

def _detect_primary_mode_warning(
    band_modes_list: list[tuple[np.ndarray, list[_Mode]]],
    threshold: float,
) -> tuple[bool, str]:
    """Warn when any band's second-best mode is within ``threshold``
    of its primary by RMS ratio.

    Returns ``(True, text)`` if any band has a competitive second
    mode, else ``(False, "")``.
    """
    for _, modes in band_modes_list:
        if len(modes) < 2:
            continue
        ratio = modes[1].rms_misfit / max(modes[0].rms_misfit, 1e-30)
        if ratio <= threshold:
            text = (
                f"At least one band discovered multiple modes within "
                f"{threshold}x RMS of each other. Single-mode "
                f"confidence intervals are misleading; the data does "
                f"not strongly distinguish between competing physical "
                f"solutions. See metadata['per_band']."
            )
            return True, text
    return False, ""

def _canonical_form_summary_joint(br: _BandResult, n_sites: int, n_freqs: int) -> dict:
    """Joint canonical-form summary for clustering converged points.

    Strike is shared across sites, so canonicalisation is decided by
    the shared theta alone; if it rotates by pi/2, every site's shear
    flips sign simultaneously.

    Returns
    -------
    dict with keys:
        strike_deg : float
        per_site_twist_deg : (n_sites,) ndarray
        per_site_shear_deg : (n_sites,) ndarray
        per_site_log10_gain : (n_sites,) ndarray (gauge — not used
                              for clustering, kept for diagnostics)
        rms_misfit : float
    """
    (
        theta_shared,
        twist,
        shear,
        log10_gain,
        _aniso,
        _,
        _,
        _,
        _,
    ) = _unpack_x_joint(br.x_opt, n_sites, n_freqs)

    strike_can, _, _ = _geometric_fold(
        theta_shared, float(twist[0]), float(shear[0])
    )
    rotated = abs(strike_can - (theta_shared % np.pi)) > 1e-9
    canonical_shear = -shear if rotated else shear

    return {
        "strike_deg": float(np.degrees(strike_can)),
        "per_site_twist_deg": np.degrees(twist).copy(),
        "per_site_shear_deg": np.degrees(canonical_shear).copy(),
        "per_site_log10_gain": log10_gain.copy(),
        "rms_misfit": float(br.rms_misfit),
    }

def _modes_match_within_band_joint(cf_a: dict, cf_b: dict, tolerance: dict) -> bool:
    """Two joint canonical-form summaries match if shared strike
    agrees within tolerance and every site's (twist, shear) agree.

    log10_gain and rms are NOT checked — gauge-equivalent and path-
    dependent respectively (see Sessions 5 and 6).
    """
    if abs(cf_a["strike_deg"] - cf_b["strike_deg"]) > tolerance["strike_deg"]:
        return False
    if np.any(
        np.abs(cf_a["per_site_twist_deg"] - cf_b["per_site_twist_deg"])
        > tolerance["twist_deg"]
    ):
        return False
    if np.any(
        np.abs(cf_a["per_site_shear_deg"] - cf_b["per_site_shear_deg"])
        > tolerance["shear_deg"]
    ):
        return False
    return True

def _modes_match_across_bands_joint(cf_a: dict, cf_b: dict, tolerance: dict) -> bool:
    """Cross-band joint mode-equivalence predicate.

    Same as :func:`_modes_match_within_band_joint`. Exists as a
    separate name for documented intent: cross-band RMS comparison
    is meaningless because bands fit different period subsets.
    """
    return _modes_match_within_band_joint(cf_a, cf_b, tolerance)

def _cluster_modes_joint(
    band_results: list[_BandResult],
    n_sites: int,
    n_freqs: int,
    mode_tolerance: dict | None = None,
) -> list[_Mode]:
    """Greedy single-pass clustering of joint converged points."""
    if not band_results:
        return []

    tol = dict(_DEFAULT_MODE_TOLERANCE)
    if mode_tolerance:
        deprecated = {"rms_relative", "log10_gain"} & set(mode_tolerance)
        if deprecated:
            warnings.warn(
                f"mode_tolerance keys {sorted(deprecated)} are no longer "
                "used and will be ignored. Joint clustering uses "
                "strike_deg, twist_deg, and shear_deg only.",
                UserWarning,
                stacklevel=2,
            )
            mode_tolerance = {
                k: v for k, v in mode_tolerance.items() if k not in deprecated
            }
        tol.update(mode_tolerance)

    summaries = [
        (br, _canonical_form_summary_joint(br, n_sites, n_freqs)) for br in band_results
    ]
    summaries.sort(key=lambda x: x[1]["rms_misfit"])

    modes: list[_Mode] = []
    for br, cf in summaries:
        matched = False
        for mode in modes:
            if _modes_match_within_band_joint(mode.canonical_form, cf, tol):
                mode.n_starts_landing_here += 1
                matched = True
                break
        if not matched:
            modes.append(
                _Mode(
                    rms_misfit=br.rms_misfit,
                    chi_squared=br.chi_squared,
                    n_starts_landing_here=1,
                    band_result=br,
                    canonical_form=cf,
                )
            )
    return modes

def _compute_mode_probabilities_joint(modes: list[_Mode], n_sites: int) -> list[float]:
    """Laplace approximation, dropping per-site anisotropy columns.

    Same algebra as :func:`_compute_mode_probabilities` but with
    n_sites zero columns removed from the Gram matrix instead of one.
    """
    if not modes:
        return []
    if len(modes) == 1:
        return [1.0]

    aniso_cols = [1 + 4 * i + 3 for i in range(n_sites)]

    log_weights: list[float] = []
    for mode in modes:
        chi_sq = mode.chi_squared
        J = mode.band_result.jacobian
        if J is None:
            log_weights.append(-0.5 * chi_sq)
            continue
        J_eff = np.delete(J, aniso_cols, axis=1)
        gram = J_eff.T @ J_eff
        sign, log_det_gram = np.linalg.slogdet(gram)
        if sign > 0:
            log_det_cov = -log_det_gram
        else:
            cov = np.linalg.pinv(gram)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-30]
            if eigenvalues.size == 0:
                log_det_cov = -np.inf
            else:
                log_det_cov = float(np.sum(np.log(eigenvalues)))
        log_weights.append(-0.5 * chi_sq + 0.5 * log_det_cov)

    log_w_max = max(log_weights)
    weights = [float(np.exp(lw - log_w_max)) for lw in log_weights]
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / len(modes)] * len(modes)
    return [w / total for w in weights]
