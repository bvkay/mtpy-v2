"""Garcia-Jones (2002) extended distortion decomposition.

The Garcia-Jones tradition extends the Groom-Bailey / McNeice-Jones
parameterised 2-D-regional decomposition to a *3-D regional* Earth.
Where GB / MJ assume the regional impedance is a 2-D anti-diagonal
tensor in some strike frame (so 4 real numbers per period: TE
amplitude / phase + TM amplitude / phase), Garcia-Jones leaves all
four complex components of the regional ``Z`` free (8 real numbers
per period). The geological context is mineral exploration over a
genuinely 3-D ore body in a 3-D regional fabric — see Figure 1 of
Garcia & Jones (2002): at long enough periods that the regional
inductive scale has resolved, only galvanic distortion remains, but
the regional response itself is 3-D.

Identifiability and the two-site assumption
-------------------------------------------
A single MT station provides 8 real numbers per period (4 complex
``Z_obs`` components). Solving for a 3-D regional ``Z`` (8 real)
plus a real distortion matrix (4 real) is impossible from one site
alone (12 unknowns vs 8 equations per period; 12 unknowns vs ``8N``
equations once we collapse across periods is also under-determined
because the regional ``Z`` is per-period and the distortion is
shared across periods).

Garcia & Jones 2002 break the deadlock by assuming **two adjacent
stations share the same regional response but have different
galvanic distortion** (Section 2):

    Z_obs^i = C^i Z_regional       i = 1, 2, ..., N

For ``N`` sites and ``F`` frequencies:

* Equations: ``8 N F`` (8 real numbers per site per period).
* Unknowns: ``8 F`` (regional Z, complex 2x2 per period) plus
  per-site distortion parameters.

For ``N >= 2`` and the basic GB-distortion parameterisation (twist,
shear; gain and anisotropy excluded — see below), the unknowns are
``8F + 2N`` and the system is over-determined.

Phase 1 parameterisation
------------------------
The 2002 paper (Section 4) demonstrates two empirical results that
constrain a robust Phase 1 implementation:

1. **Gain is unrecoverable.** With the gain ``g^i`` free, the
   inverse problem is unstable; the recovered (gain, regional Z)
   pair drifts substantially while still fitting the observations
   to within tolerance (their Figure 7c, 8). We therefore fix
   ``g^i = 1`` for all sites in Phase 1; whatever absolute scaling
   is present in ``Z_obs`` is absorbed into the recovered regional
   Z.
2. **Anisotropy is unrecoverable for the same reason.** The
   anisotropy parameter ``a^i`` (which would give ``g^i_1 != g^i_2``
   in eq 13.8 of the paper) is degenerate with the off-diagonal
   amplitudes of the regional Z. We fix ``a^i = 0``.

This leaves per-site real ``twist`` and ``shear`` as the only
distortion parameters. The distortion matrix in this Phase 1
restriction is

    C(twist, shear) = T(twist) S(shear) =
        [[1, -tan(t)], [tan(t), 1]] @ [[1, sin(s)/cos(s)],
                                       [sin(s)/cos(s), 1]]

where ``t = twist`` and ``s = shear`` are real angles in radians.
We use the GB ``twist_tan`` / ``shear_tan`` form so that the
forward model is the same matrix expression used elsewhere in the
package (compare :func:`._estim_imp` and
:func:`mcneice_jones._reconstruct_c_meas`).

Algorithm
---------
For each period band:

1. Pack parameters into a flat vector
   ``x = [twist_per_site (N), shear_per_site (N),
   real(Z_reg) per period (4F), imag(Z_reg) per period (4F)]``.
2. Initial guess: zero distortion (``twist = shear = 0`` for all
   sites) and ``Z_reg[k] = mean over sites of Z_obs[i, k]``. This
   is exact when there is no distortion and a reasonable starting
   point otherwise.
3. Multi-start: ``n_starts`` perturbations of the twist / shear
   block; the regional Z block is left at the initial mean. The
   perturbations cover ``[-30, +30]`` degrees uniformly.
4. Per-start: scipy ``least_squares`` with TRF and the
   ``z_error``-normalised residual vector.
5. Pick the start with lowest chi-squared as primary; record the
   spread across starts for diagnostics.

This is "Phase 1" — the core algorithm. Validation against
multi-instrument real data (BC87 etc.), gain-recovery experiments
(Section 4.2 of the paper), and the magnetic-distortion extension
of Garcia, Boerner, & Pedersen (2003) are deferred to follow-up
sessions.

References
----------
Garcia, X., & Jones, A. G. (2002). Decomposition of three-
dimensional magnetotelluric data. In *Three-Dimensional
Electromagnetics* (M. S. Zhdanov & P. E. Wannamaker, eds.), Methods
in Geochemistry and Geophysics, 35, 235-250.

Garcia, X., Boerner, D., & Pedersen, L. B. (2003). Electric and
magnetic galvanic distortion decomposition of tensor CSAMT data.
Application to data from the Buchans Mine (Newfoundland, Canada).
Geophysical Journal International, 154, 957-969.

Groom, R. W., & Bailey, R. C. (1989). Decomposition of magnetotelluric
impedance tensors in the presence of local three-dimensional
galvanic distortion. Journal of Geophysical Research: Solid Earth,
94(B2), 1913-1925.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from .common import _validate_joint_input
from .results import GarciaJonesResult

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


__all__ = ["decompose_garcia_jones"]


# ---------------------------------------------------------------------------
# Forward model and parameter packing
# ---------------------------------------------------------------------------


def _distortion_matrix(twist_rad: float, shear_rad: float) -> np.ndarray:
    """Real GB distortion matrix ``C = T(twist) S(shear)``.

    Phase 1: gain = 1, anisotropy = 0. Returns a real 2x2 matrix.
    """
    t = np.tan(twist_rad)
    s = np.tan(shear_rad)
    T = np.array([[1.0, -t], [t, 1.0]])
    S = np.array([[1.0, s], [s, 1.0]])
    return T @ S


def _pack_x_gj(
    twist_rad: np.ndarray,
    shear_rad: np.ndarray,
    z_regional: np.ndarray,
) -> np.ndarray:
    """Pack twist, shear, and per-period 3-D regional Z into a flat
    parameter vector.

    Parameters
    ----------
    twist_rad, shear_rad : (n_sites,) float ndarray
    z_regional : (n_freqs, 2, 2) complex ndarray

    Returns
    -------
    x : (2 * n_sites + 8 * n_freqs,) float ndarray
        Layout: ``[twist (N), shear (N), Re(Z_reg) flat (4F),
        Im(Z_reg) flat (4F)]``.
    """
    n_sites = twist_rad.size
    n_freqs = z_regional.shape[0]
    x = np.empty(2 * n_sites + 8 * n_freqs, dtype=np.float64)
    x[:n_sites] = twist_rad
    x[n_sites : 2 * n_sites] = shear_rad
    re_block = z_regional.real.reshape(-1)
    im_block = z_regional.imag.reshape(-1)
    base = 2 * n_sites
    x[base : base + 4 * n_freqs] = re_block
    x[base + 4 * n_freqs : base + 8 * n_freqs] = im_block
    return x


def _unpack_x_gj(
    x: np.ndarray, n_sites: int, n_freqs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`_pack_x_gj`."""
    expected = 2 * n_sites + 8 * n_freqs
    if x.size != expected:
        raise ValueError(
            f"_unpack_x_gj: expected x of size {expected} for "
            f"n_sites={n_sites}, n_freqs={n_freqs}, got {x.size}"
        )
    twist_rad = x[:n_sites].copy()
    shear_rad = x[n_sites : 2 * n_sites].copy()
    base = 2 * n_sites
    re_block = x[base : base + 4 * n_freqs].reshape(n_freqs, 2, 2)
    im_block = x[base + 4 * n_freqs : base + 8 * n_freqs].reshape(n_freqs, 2, 2)
    z_regional = re_block + 1j * im_block
    return twist_rad, shear_rad, z_regional


def _residual_vector(
    x: np.ndarray,
    z_obs: np.ndarray,
    sigma: np.ndarray,
    n_sites: int,
    n_freqs: int,
) -> np.ndarray:
    """TRF residual vector for Garcia-Jones.

    Layout matches MJ: per (site, period, component) we emit a
    ``(real, imag)`` pair, normalised by ``sigma``.

    Parameters
    ----------
    x : (2*n_sites + 8*n_freqs,) float ndarray
    z_obs : (n_sites, n_freqs, 2, 2) complex ndarray
    sigma : (n_sites, n_freqs, 2, 2) float ndarray
    """
    twist_rad, shear_rad, z_reg = _unpack_x_gj(x, n_sites, n_freqs)

    res = np.empty(n_sites * n_freqs * 8, dtype=np.float64)
    out_idx = 0
    for i in range(n_sites):
        c_i = _distortion_matrix(twist_rad[i], shear_rad[i])
        for k in range(n_freqs):
            z_pred = c_i @ z_reg[k]
            diff = z_pred - z_obs[i, k]
            inv_sigma = 1.0 / sigma[i, k]
            res[out_idx + 0] = diff[0, 0].real * inv_sigma[0, 0]
            res[out_idx + 1] = diff[0, 0].imag * inv_sigma[0, 0]
            res[out_idx + 2] = diff[0, 1].real * inv_sigma[0, 1]
            res[out_idx + 3] = diff[0, 1].imag * inv_sigma[0, 1]
            res[out_idx + 4] = diff[1, 0].real * inv_sigma[1, 0]
            res[out_idx + 5] = diff[1, 0].imag * inv_sigma[1, 0]
            res[out_idx + 6] = diff[1, 1].real * inv_sigma[1, 1]
            res[out_idx + 7] = diff[1, 1].imag * inv_sigma[1, 1]
            out_idx += 8
    return res


# ---------------------------------------------------------------------------
# Per-band optimisation
# ---------------------------------------------------------------------------


def _initial_guess(
    z_obs: np.ndarray, n_sites: int, n_freqs: int
) -> np.ndarray:
    """Starting parameter vector: zero distortion, mean-of-sites Z."""
    twist0 = np.zeros(n_sites)
    shear0 = np.zeros(n_sites)
    z_reg0 = np.mean(z_obs, axis=0)
    return _pack_x_gj(twist0, shear0, z_reg0)


def _solve_band_gj(
    z_obs: np.ndarray,
    sigma: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
    max_nfev: int,
    perturb_deg: float = 30.0,
) -> dict:
    """Solve one band: multi-start TRF, return primary fit metadata.

    Parameters
    ----------
    z_obs : (n_sites, n_freqs, 2, 2) complex ndarray
    sigma : (n_sites, n_freqs, 2, 2) float ndarray
    n_starts : int
        Number of starting points for the multi-start procedure.
        The first start is the unperturbed mean-of-sites guess.
    rng : np.random.Generator
    max_nfev : int
        Maximum scipy ``least_squares`` function evaluations.
    perturb_deg : float, default 30.0
        Half-range (in degrees) of the uniform perturbation applied
        to the twist / shear block on starts 2 ... ``n_starts``.

    Returns
    -------
    dict with keys
        ``x_opt``, ``residuals``, ``chi_squared``, ``rms_misfit``,
        ``n_iter``, ``converged``, ``n_starts``, ``starts_chi``
        (list of the chi-squared from each start).
    """
    n_sites, n_freqs = z_obs.shape[:2]

    x0_base = _initial_guess(z_obs, n_sites, n_freqs)

    starts: list[np.ndarray] = [x0_base]
    perturb_rad = np.radians(perturb_deg)
    for _ in range(max(0, n_starts - 1)):
        x0 = x0_base.copy()
        x0[: 2 * n_sites] += rng.uniform(
            -perturb_rad, perturb_rad, size=2 * n_sites
        )
        starts.append(x0)

    best = None
    starts_chi: list[float] = []
    for x0 in starts:
        try:
            sol = least_squares(
                _residual_vector,
                x0,
                args=(z_obs, sigma, n_sites, n_freqs),
                method="trf",
                max_nfev=int(max_nfev),
            )
        except Exception:
            starts_chi.append(float("inf"))
            continue
        chi2 = float(np.sum(sol.fun**2))
        starts_chi.append(chi2)
        if best is None or chi2 < best["chi_squared"]:
            best = {
                "x_opt": sol.x,
                "residuals": sol.fun,
                "chi_squared": chi2,
                "n_iter": int(sol.nfev),
                "converged": bool(sol.success),
            }

    if best is None:
        raise RuntimeError(
            "_solve_band_gj: every multi-start failed; check input."
        )

    n_resid = best["residuals"].size
    best["rms_misfit"] = float(np.sqrt(best["chi_squared"] / max(n_resid, 1)))
    best["n_starts"] = len(starts)
    best["starts_chi"] = starts_chi
    return best


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decompose_garcia_jones(
    z_objs: "list[Z]",
    site_ids: list[str],
    *,
    period_bands: list[tuple[float, float]] | None = None,
    share_distortion_within_period: bool = True,
    per_band_3d: bool = True,
    n_starts: int = 5,
    seed: int = 42,
    max_iter: int = 400,
) -> GarciaJonesResult:
    """Garcia-Jones (2002) extended 3-D regional decomposition.

    Fits a single shared 3-D regional impedance per period and
    per-site real distortion (``twist``, ``shear``) by joint TRF
    nonlinear least-squares across all sites.

    Parameters
    ----------
    z_objs : list of Z
        Parallel list of mtpy ``Z`` objects, one per site. All must
        share the same frequency grid.
    site_ids : list of str
        Identifiers parallel to ``z_objs``. Must be unique.
    period_bands : list of (float, float), optional
        Period bands in seconds. ``None`` (default) treats the full
        range as a single band.
    share_distortion_within_period : bool, default True
        Currently must be ``True``: distortion is per-site and
        constant across periods within a band, the standard GB
        assumption. ``False`` (per-period distortion) is future
        scope.
    per_band_3d : bool, default True
        Currently must be ``True``: regional Z is fitted as 3-D per
        period within each band. ``False`` (per-band 2-D) reduces
        to MJ and the user should call :func:`decompose_mcneice_jones`
        directly.
    n_starts : int, default 5
        Number of multi-start initial guesses per band. The first
        start is unperturbed; remaining starts perturb the twist /
        shear block uniformly in ``[-30, +30]`` degrees.
    seed : int, default 42
        RNG seed for reproducibility of the multi-start
        perturbations.
    max_iter : int, default 400
        Maximum scipy ``least_squares`` function evaluations per
        band per start.

    Returns
    -------
    GarciaJonesResult

    Raises
    ------
    NotImplementedError
        If ``share_distortion_within_period`` or ``per_band_3d`` is
        ``False`` (future scope).
    ValueError
        If the input lists have mismatched lengths or non-unique
        ``site_ids``, or no period falls inside any band, or fewer
        than two sites are provided.

    See Also
    --------
    decompose_mcneice_jones : Multi-site 2-D-regional decomposition.
        Use that when the regional structure is genuinely 2-D —
        Garcia-Jones is biased / unstable in that case because the
        free 3-D regional fits noise into the diagonal entries.
    """
    if not share_distortion_within_period:
        raise NotImplementedError(
            "decompose_garcia_jones: share_distortion_within_period=False "
            "is future scope; currently distortion is per-site and "
            "shared across all periods within a band."
        )
    if not per_band_3d:
        raise NotImplementedError(
            "decompose_garcia_jones: per_band_3d=False reduces to MJ; "
            "call decompose_mcneice_jones directly."
        )
    if len(z_objs) != len(site_ids):
        raise ValueError(
            f"decompose_garcia_jones: z_objs ({len(z_objs)}) and "
            f"site_ids ({len(site_ids)}) length mismatch"
        )
    if len(set(site_ids)) != len(site_ids):
        raise ValueError(
            "decompose_garcia_jones: site_ids must be unique; got "
            f"{site_ids!r}"
        )
    if len(z_objs) < 2:
        raise ValueError(
            "decompose_garcia_jones: at least two sites are required "
            "for the Garcia-Jones two-site assumption to be applicable."
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
                "decompose_garcia_jones: no periods fall inside any "
                f"of period_bands={period_bands!r}; available range is "
                f"({selected_periods[0]:.3g}, {selected_periods[-1]:.3g}) s"
            )

    rng = np.random.default_rng(seed)

    per_site_dist: dict[str, dict] = {sid: _empty_site_record() for sid in site_ids}
    per_band_3d_z: dict[int, np.ndarray] = {}
    per_band_meta: list[dict] = []
    site_chi_sq: dict[str, float] = {sid: 0.0 for sid in site_ids}
    site_n_resid: dict[str, int] = {sid: 0 for sid in site_ids}
    total_chi_sq = 0.0

    for band_id, band_idx in enumerate(bands):
        z_b = z_per_site[:, band_idx, :, :]
        sigma_b = sigma_per_site[:, band_idx, :, :]
        periods_b = selected_periods[band_idx]
        n_band = periods_b.size

        fit = _solve_band_gj(
            z_obs=z_b,
            sigma=sigma_b,
            n_starts=n_starts,
            rng=rng,
            max_nfev=max_iter,
        )

        twist_rad, shear_rad, z_reg = _unpack_x_gj(
            fit["x_opt"], n_sites, n_band
        )
        per_band_3d_z[band_id] = z_reg

        for i, sid in enumerate(site_ids):
            twist_deg_i = float(np.degrees(twist_rad[i]))
            shear_deg_i = float(np.degrees(shear_rad[i]))
            rec = per_site_dist[sid]
            rec["twist_deg_per_band"].append(twist_deg_i)
            rec["shear_deg_per_band"].append(shear_deg_i)

        # Per-site residual accounting. The residual vector is laid
        # out [site, freq, ReXX, ImXX, ReXY, ImXY, ReYX, ImYX, ReYY,
        # ImYY], 8 reals per (site, freq).
        for i, sid in enumerate(site_ids):
            site_block = fit["residuals"][
                i * n_band * 8 : (i + 1) * n_band * 8
            ]
            site_chi_sq[sid] += float(np.sum(site_block**2))
            site_n_resid[sid] += int(site_block.size)
        total_chi_sq += float(fit["chi_squared"])

        per_band_meta.append(
            {
                "band_id": band_id,
                "band_period_indices": band_idx.tolist(),
                "band_periods_seconds": periods_b.tolist(),
                "n_starts": fit["n_starts"],
                "rms_misfit": fit["rms_misfit"],
                "chi_squared": fit["chi_squared"],
                "n_iter": fit["n_iter"],
                "converged": fit["converged"],
                "starts_chi": fit["starts_chi"],
            }
        )

    rms_per_site: dict[str, float] = {}
    for sid in site_ids:
        rec = per_site_dist[sid]
        rec["twist_deg_per_band"] = np.array(
            rec["twist_deg_per_band"], dtype=np.float64
        )
        rec["shear_deg_per_band"] = np.array(
            rec["shear_deg_per_band"], dtype=np.float64
        )
        rec["twist_deg"] = float(np.median(rec["twist_deg_per_band"]))
        rec["shear_deg"] = float(np.median(rec["shear_deg_per_band"]))
        rec["gain"] = 1.0
        rec["c_tensor"] = _distortion_matrix(
            np.radians(rec["twist_deg"]), np.radians(rec["shear_deg"])
        )
        rms_per_site[sid] = float(
            np.sqrt(site_chi_sq[sid] / max(site_n_resid[sid], 1))
        )

    metadata = {
        "method": "garcia_jones",
        "n_sites": n_sites,
        "n_bands": len(bands),
        "n_starts": n_starts,
        "seed": seed,
        "share_distortion_within_period": share_distortion_within_period,
        "per_band_3d": per_band_3d,
        "max_iter": max_iter,
        "period_bands": bands_actually_fitted,
        "site_ids": list(site_ids),
        "per_band": per_band_meta,
    }

    return GarciaJonesResult(
        per_site_distortion=per_site_dist,
        per_band_3d_z_regional=per_band_3d_z,
        chi_squared=float(total_chi_sq),
        rms_misfit_per_site=rms_per_site,
        metadata=metadata,
    )


def _empty_site_record() -> dict:
    return {
        "twist_deg_per_band": [],
        "shear_deg_per_band": [],
    }
