"""McNeice-Jones (2001) multi-site joint Groom-Bailey decomposition.

The McNeice-Jones extension of GB takes the geological observation that
a regional 2-D structure has a single regional strike — not a
station-dependent strike — and ties the strike across multiple sites
within each frequency band. Per-site distortion (twist, shear, gain)
remains per-site, but the joint fit pools information about the
regional strike from all stations, dramatically improving its
identifiability when single-site GB hits the strike / shear-sign
symmetry as multiple comparable modes.

Algorithm
---------
This module is a thin façade over the joint-optimisation machinery in
:mod:`.groom_bailey`. It exposes a cleaner API specifically for the
multi-site case (parallel ``z_objs`` / ``site_ids`` lists, per-band
result dicts) without re-implementing the cost function, Jacobian, or
multi-start clustering. The core call sequence is:

1. ``_validate_joint_input`` (from :mod:`.common`) checks that all
   sites share a frequency grid and have valid ``z_error``.
2. ``_solve_band_joint_multistart`` (from :mod:`.common`) runs the
   joint TRF nonlinear least-squares fit per band over ``n_starts``
   starting points, with mode clustering by canonical form so
   multi-modal fits are detected. The orchestration is method-
   agnostic; the GB-specific cost function, bounds, and initial
   guesses (``_objfun_joint``, ``_build_bounds_joint``,
   ``_canonical_initial_guess_joint``,
   ``_rotated_initial_guess_joint``) are also imported from
   :mod:`.common` and passed in as factory callables.
   ``mcneice_jones`` no longer reaches into ``groom_bailey``.
3. The chosen disambiguation strategy
   (:mod:`.symmetries._resolve_disambiguation`) is applied
   *post-hoc* to each band's primary mode: the shared strike is
   folded to its canonical branch, with every site's shear sign
   flipped together if the rotated branch is selected (a property
   of the GB symmetry — strike rotates by 90° and every site's
   shear flips sign in lock-step).
4. Per-site reconstructed C tensors and per-band per-site regional
   Z are assembled into a :class:`JointDecompositionResult`.

This is "Phase 1" — the core algorithm. Validation against published
results (BC87 etc.) and the more delicate edge cases (ragged
frequency grids, sparse coverage, ``per_site_distortion=False``
shared distortion) are deferred to a follow-up pass.

References
----------
McNeice, G. W., & Jones, A. G. (2001). Multisite, multifrequency
tensor decomposition of magnetotelluric data. Geophysics, 66(1),
158-173.

Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional galvanic
distortion. Journal of Geophysical Research: Solid Earth, 94(B2),
1913-1925.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .common import (
    _build_bounds_joint,
    _canonical_initial_guess_joint,
    _objfun_joint,
    _rotated_initial_guess_joint,
    _solve_band_joint_multistart,
    _unpack_x_joint,
    _validate_joint_input,
)
from .results import JointDecompositionResult
from .symmetries import _geometric_fold, _resolve_disambiguation

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


__all__ = ["decompose_mcneice_jones"]


def decompose_mcneice_jones(
    z_objs: "list[Z]",
    site_ids: list[str],
    *,
    period_bands: list[tuple[float, float]] | None = None,
    share_strike_within_band: bool = True,
    per_site_distortion: bool = True,
    n_starts: int = 5,
    seed: int = 42,
    disambiguation: "str | Callable" = "geometric",
    max_iter: int = 200,
) -> JointDecompositionResult:
    """McNeice-Jones (2001) multi-site joint GB decomposition.

    Fits a single shared regional strike per band across all sites,
    plus per-site distortion (``twist``, ``shear``, ``gain``) and
    per-site per-period regional impedance.

    Parameters
    ----------
    z_objs : list of Z
        Parallel list of mtpy ``Z`` objects (impedance tensors), one
        per site. Every ``Z`` must share the same frequency grid;
        :func:`._validate_joint_input` raises if not.
    site_ids : list of str
        Identifiers parallel to ``z_objs``. Must be unique.
    period_bands : list of (float, float), optional
        Period bands as ``(min, max)`` tuples in seconds. ``None``
        (default) treats the full period range as a single band.
    share_strike_within_band : bool, default True
        Only ``True`` is currently supported. Per-site or smoothed-
        strike sharing is future scope.
    per_site_distortion : bool, default True
        Only ``True`` is currently supported. A shared distortion
        across sites is methodologically different (closer to
        Bibby-Caldwell-Brown semantics) and is future scope.
    n_starts : int, default 5
        Number of multi-start initial guesses per band. The
        disambiguation step is applied per primary-mode-per-band; if
        a band has multiple competitive modes,
        ``metadata['per_band'][band_id]['n_modes']`` will record
        that.
    seed : int, default 42
        RNG seed for reproducibility of the multi-start
        perturbations.
    disambiguation : str or callable, default ``'geometric'``
        Strategy for resolving the GB 90-degree symmetry on the
        shared strike. Same set as :func:`decompose`:
        ``'geometric'``, ``'identity'``, ``'pt_aligned'``,
        ``'min_shear'``, or a callable
        ``fold(strike, twist, shear) -> (strike, twist, shear)``
        (radians). For the joint case, a single fold decision is
        made per band based on the shared strike; if the rotated
        branch is selected, every site's shear sign flips
        together.
    max_iter : int, default 200
        Maximum scipy ``least_squares`` function evaluations per
        band per start. Recorded in metadata; passed through as
        ``max_nfev``.

    Returns
    -------
    JointDecompositionResult

    Raises
    ------
    NotImplementedError
        If ``share_strike_within_band`` or ``per_site_distortion``
        is ``False`` (future scope).
    ValueError
        If the input lists have mismatched lengths or non-unique
        ``site_ids``, or no period falls inside any band.

    See Also
    --------
    decompose : Single-site GB.
    decompose_joint : Older joint API returning a
        :class:`DecompositionResult`. Same algorithm, different
        result shape.
    """
    if not share_strike_within_band:
        raise NotImplementedError(
            "decompose_mcneice_jones: share_strike_within_band=False "
            "is future scope; currently only the per-band shared-strike "
            "configuration is supported."
        )
    if not per_site_distortion:
        raise NotImplementedError(
            "decompose_mcneice_jones: per_site_distortion=False "
            "(shared distortion across sites) is future scope."
        )
    if len(z_objs) != len(site_ids):
        raise ValueError(
            f"decompose_mcneice_jones: z_objs ({len(z_objs)}) and "
            f"site_ids ({len(site_ids)}) length mismatch"
        )
    if len(set(site_ids)) != len(site_ids):
        raise ValueError(
            "decompose_mcneice_jones: site_ids must be unique; got "
            f"{site_ids!r}"
        )

    stations = list(zip(site_ids, z_objs))
    _validate_joint_input(stations)

    n_sites = len(stations)
    ref_z = stations[0][1]
    frequencies = np.asarray(ref_z.frequency, dtype=np.float64)
    all_periods = 1.0 / frequencies
    sort_idx = np.argsort(all_periods)
    selected_periods = all_periods[sort_idx]
    n_periods = len(selected_periods)

    z_per_site = np.empty((n_sites, n_periods, 2, 2), dtype=np.complex128)
    sigma_per_site = np.empty((n_sites, n_periods, 2, 2), dtype=np.float64)
    for i, (_, z) in enumerate(stations):
        z_per_site[i] = np.asarray(z.z, dtype=np.complex128)[sort_idx]
        sigma_per_site[i] = np.asarray(z.z_error, dtype=np.float64)[sort_idx]

    # Resolve period bands to per-band period-index arrays.
    if period_bands is None:
        bands: list[np.ndarray] = [np.arange(n_periods)]
        bands_actually_fitted: list[tuple[float, float]] = [
            (float(selected_periods[0]), float(selected_periods[-1]))
        ]
    else:
        bands = []
        bands_actually_fitted = []
        for pmin, pmax in period_bands:
            mask = (selected_periods >= pmin) & (selected_periods <= pmax)
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            bands.append(idx)
            bands_actually_fitted.append((float(pmin), float(pmax)))
        if not bands:
            raise ValueError(
                "decompose_mcneice_jones: no periods fall inside any "
                f"of period_bands={period_bands!r}; available range is "
                f"({selected_periods[0]:.3g}, {selected_periods[-1]:.3g}) s"
            )

    rng = np.random.default_rng(seed)

    per_site_dist: dict[str, dict] = {sid: _empty_site_record() for sid in site_ids}
    per_band_strike: dict[int, float] = {}
    per_band_per_site_z_regional: dict[tuple[int, str], np.ndarray] = {}
    per_band_meta: list[dict] = []
    site_chi_sq: dict[str, float] = {sid: 0.0 for sid in site_ids}
    site_n_resid: dict[str, int] = {sid: 0 for sid in site_ids}
    total_chi_sq = 0.0

    for band_id, band_idx in enumerate(bands):
        z_b = z_per_site[:, band_idx, :, :]
        sigma_b = sigma_per_site[:, band_idx, :, :]
        periods_b = selected_periods[band_idx]
        n_band = periods_b.size

        modes = _solve_band_joint_multistart(
            z_obs_per_site=z_b,
            sigma_per_site=sigma_b,
            periods=periods_b,
            objfun_joint=_objfun_joint,
            build_bounds_joint=_build_bounds_joint,
            canonical_initial_guess_joint=_canonical_initial_guess_joint,
            rotated_initial_guess_joint=_rotated_initial_guess_joint,
            n_starts=n_starts,
            rng=rng,
            max_nfev=int(max_iter),
        )
        primary = modes[0]
        br = primary.band_result

        unpacked = _unpack_x_joint(br.x_opt, n_sites, n_band)
        theta_raw = float(unpacked[0])
        twist_raw = unpacked[1].copy()
        shear_raw = unpacked[2].copy()
        log10_gain = unpacked[3].copy()
        log10_rho_a_band = unpacked[5]
        phase_a_band = unpacked[6]
        log10_rho_b_band = unpacked[7]
        phase_b_band = unpacked[8]

        theta_can, _, shear_sign = _apply_joint_fold(
            theta_raw,
            twist_raw,
            shear_raw,
            disambiguation,
            z_per_site,
            sort_idx_view=band_idx,
            stations=stations,
        )
        per_site_shear_canonical = shear_sign * shear_raw

        per_band_strike[band_id] = float(np.degrees(theta_can))
        c_a, s_a = np.cos(theta_can), np.sin(theta_can)
        R = np.array([[c_a, -s_a], [s_a, c_a]])

        mu0 = 4.0 * np.pi * 1.0e-7
        factor = 2.0 * np.pi * mu0
        for i, sid in enumerate(site_ids):
            rho_a_i = 10.0 ** log10_rho_a_band[i]
            rho_b_i = 10.0 ** log10_rho_b_band[i]
            abs_a_i = np.sqrt(rho_a_i * factor / periods_b)
            abs_b_i = np.sqrt(rho_b_i * factor / periods_b)
            a_i = abs_a_i * np.exp(1j * phase_a_band[i])
            b_i = abs_b_i * np.exp(1j * phase_b_band[i])
            z_r_meas = np.empty((n_band, 2, 2), dtype=np.complex128)
            for k in range(n_band):
                z_r_strike = np.array(
                    [[0.0, a_i[k]], [-b_i[k], 0.0]], dtype=np.complex128
                )
                z_r_meas[k] = R @ z_r_strike @ R.T
            per_band_per_site_z_regional[(band_id, sid)] = z_r_meas

            twist_deg_i = float(np.degrees(twist_raw[i]))
            shear_deg_i = float(np.degrees(per_site_shear_canonical[i]))
            gain_i = float(10.0 ** log10_gain[i])
            rec = per_site_dist[sid]
            rec["twist_deg_per_band"].append(twist_deg_i)
            rec["shear_deg_per_band"].append(shear_deg_i)
            rec["gain_per_band"].append(gain_i)
            rec["strike_deg_per_band"].append(float(np.degrees(theta_can)))

        # Per-site residual accounting. The joint residual vector is
        # packed [site, freq, ReXX, ImXX, ReXY, ImXY, ReYX, ImYX, ReYY, ImYY].
        for i, sid in enumerate(site_ids):
            site_block = br.residuals[
                i * n_band * 8 : (i + 1) * n_band * 8
            ]
            site_chi_sq[sid] += float(np.sum(site_block**2))
            site_n_resid[sid] += int(site_block.size)
        total_chi_sq += float(br.chi_squared)

        per_band_meta.append(
            {
                "band_id": band_id,
                "band_period_indices": band_idx.tolist(),
                "band_periods_seconds": periods_b.tolist(),
                "n_modes": len(modes),
                "primary_rms_misfit": primary.rms_misfit,
                "primary_chi_squared": primary.chi_squared,
                "n_iter": int(br.n_iter),
                "converged": bool(br.converged),
                "shear_branch_flip": bool(shear_sign < 0),
            }
        )

    # Finalise per-site records: band-averaged scalars (median) and
    # the reconstructed measurement-frame C tensor.
    rms_per_site: dict[str, float] = {}
    for sid in site_ids:
        rec = per_site_dist[sid]
        rec["twist_deg_per_band"] = np.array(
            rec["twist_deg_per_band"], dtype=np.float64
        )
        rec["shear_deg_per_band"] = np.array(
            rec["shear_deg_per_band"], dtype=np.float64
        )
        rec["gain_per_band"] = np.array(rec["gain_per_band"], dtype=np.float64)
        rec["strike_deg_per_band"] = np.array(
            rec["strike_deg_per_band"], dtype=np.float64
        )
        rec["twist_deg"] = float(np.median(rec["twist_deg_per_band"]))
        rec["shear_deg"] = float(np.median(rec["shear_deg_per_band"]))
        rec["gain"] = float(np.median(rec["gain_per_band"]))
        strike_deg_med = float(np.median(rec["strike_deg_per_band"]))
        rec["c_tensor"] = _reconstruct_c_meas(
            strike_deg_med, rec["twist_deg"], rec["shear_deg"], rec["gain"]
        )
        rms_per_site[sid] = float(
            np.sqrt(site_chi_sq[sid] / max(site_n_resid[sid], 1))
        )

    metadata = {
        "method": "mcneice_jones_joint",
        "n_sites": n_sites,
        "n_bands": len(bands),
        "n_starts": n_starts,
        "seed": seed,
        "disambiguation": (
            disambiguation if isinstance(disambiguation, str) else "callable"
        ),
        "share_strike_within_band": share_strike_within_band,
        "per_site_distortion": per_site_distortion,
        "max_iter": max_iter,
        "period_bands": bands_actually_fitted,
        "site_ids": list(site_ids),
        "per_band": per_band_meta,
    }

    return JointDecompositionResult(
        per_site_distortion=per_site_dist,
        per_band_strike=per_band_strike,
        per_band_per_site_z_regional=per_band_per_site_z_regional,
        chi_squared=float(total_chi_sq),
        rms_misfit_per_site=rms_per_site,
        metadata=metadata,
    )


def _empty_site_record() -> dict:
    return {
        "twist_deg_per_band": [],
        "shear_deg_per_band": [],
        "gain_per_band": [],
        "strike_deg_per_band": [],
    }


def _apply_joint_fold(
    theta_rad: float,
    twist_rad: np.ndarray,
    shear_rad: np.ndarray,
    disambiguation: "str | Callable",
    z_per_site: np.ndarray,
    *,
    sort_idx_view,
    stations,
) -> tuple[float, np.ndarray, float]:
    """Apply a disambiguation fold to the joint shared strike.

    Returns ``(theta_canonical_rad, twist_unchanged, shear_sign)``
    where ``shear_sign`` is ``+1`` if the original branch was kept
    or ``-1`` if the rotated branch was selected (in which case
    every site's shear sign must flip in lock-step).

    For ``disambiguation='pt_aligned'`` the per-band PT strike used
    as the target is the median of the per-site PT alphas at the
    band's median period.
    """
    if isinstance(disambiguation, str) and disambiguation == "geometric":
        # Use the canonical _geometric_fold result; sign flip is
        # implicit in whether shear changed sign.
        rep_shear = float(shear_rad[0]) if shear_rad.size else 0.0
        s_can, _, sh_can = _geometric_fold(
            theta_rad, float(twist_rad[0]) if twist_rad.size else 0.0, rep_shear
        )
        sign = 1.0 if np.isclose(sh_can, rep_shear) else -1.0
        return s_can, twist_rad, sign

    if isinstance(disambiguation, str) and disambiguation == "identity":
        return float(theta_rad % np.pi), twist_rad, 1.0

    if isinstance(disambiguation, str) and disambiguation == "pt_aligned":
        pt_strike_rad = _band_pt_strike(z_per_site, sort_idx_view, stations)
        fold = _resolve_disambiguation("pt_aligned", pt_strike_rad=pt_strike_rad)
        rep_shear = float(shear_rad[0]) if shear_rad.size else 1.0
        s_can, _, sh_can = fold(theta_rad, 0.0, rep_shear)
        sign = 1.0 if np.isclose(sh_can, rep_shear) else -1.0
        return s_can, twist_rad, sign

    if isinstance(disambiguation, str) and disambiguation == "min_shear":
        # For joint, "min |shear|" is interpreted across sites: pick
        # the branch that minimises the sum of |shear| over sites.
        # The rotated branch flips every shear's sign, so |shear|
        # values are identical on the two branches in pure form.
        # For finite-precision optimiser output the two are not
        # exactly equal, so we still get a deterministic pick.
        sum_orig = float(np.sum(np.abs(shear_rad)))
        sum_flip = float(np.sum(np.abs(-shear_rad)))
        if sum_flip < sum_orig:
            return float((theta_rad + np.pi / 2.0) % np.pi), twist_rad, -1.0
        return float(theta_rad % np.pi), twist_rad, 1.0

    if callable(disambiguation):
        rep_shear = float(shear_rad[0]) if shear_rad.size else 1.0
        s_can, _, sh_can = disambiguation(
            theta_rad,
            float(twist_rad[0]) if twist_rad.size else 0.0,
            rep_shear,
        )
        sign = 1.0 if np.isclose(sh_can, rep_shear) else -1.0
        return float(s_can), twist_rad, sign

    raise ValueError(
        f"decompose_mcneice_jones: unknown disambiguation "
        f"{disambiguation!r}"
    )


def _band_pt_strike(z_per_site, band_idx, stations) -> float:
    """Median across sites of the PT strike at the band's median period.

    Used by the ``pt_aligned`` disambiguation strategy in the joint
    case. Returns radians.
    """
    median_local = int(len(band_idx) // 2)
    global_period_idx = int(band_idx[median_local])
    pt_alphas: list[float] = []
    for _, z in stations:
        try:
            alpha = np.asarray(z.phase_tensor.alpha, dtype=np.float64)
        except Exception:
            continue
        # The Z's frequency order may not match selected_periods; in
        # the joint context the validation step has already ensured
        # all stations share a frequency grid, so the order matches
        # after the sort applied in decompose_mcneice_jones — which
        # we mirror by re-sorting the ``alpha`` array here.
        freqs = np.asarray(z.frequency, dtype=np.float64)
        sort_idx = np.argsort(1.0 / freqs)
        alpha_sorted = alpha[sort_idx]
        if global_period_idx < alpha_sorted.size:
            val = alpha_sorted[global_period_idx]
            if np.isfinite(val):
                pt_alphas.append(float(val))
    if not pt_alphas:
        raise ValueError(
            "decompose_mcneice_jones: disambiguation='pt_aligned' "
            "needs at least one station with a finite phase-tensor "
            "strike at the band's median period."
        )
    return float(np.radians(np.median(pt_alphas)))


def _reconstruct_c_meas(
    strike_deg: float, twist_deg: float, shear_deg: float, gain: float
) -> np.ndarray:
    """Reconstruct ``C_meas = g R(strike) T(twist) S(shear) R(strike).T``."""
    s = np.radians(strike_deg)
    t = np.radians(twist_deg)
    e = np.radians(shear_deg)
    cs, sn = np.cos(s), np.sin(s)
    ct, st = np.cos(t), np.sin(t)
    ce, se = np.cos(e), np.sin(e)
    R = np.array([[cs, -sn], [sn, cs]])
    T = np.array([[ct, -st], [st, ct]])
    S = np.array([[ce, se], [se, ce]])
    return gain * R @ T @ S @ R.T
