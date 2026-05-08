"""Bibby-Caldwell-Brown (BCB) decomposition.

Algorithm
---------
Bibby et al. (2005) decompose the observed measurement-frame
impedance tensor as

    Z_obs = C @ Z_R

with ``C`` a real 2x2 galvanic-distortion tensor and ``Z_R`` a
purely two-dimensional regional impedance (anti-diagonal in the
regional strike frame). Unlike Groom-Bailey, BCB does not factorise
``C`` into named geometric parameters — it returns the 2x2 matrix
directly. This makes BCB the methodologically simplest comparison
point for GB's parameterised approach: a single real matrix per
site, frequency-averaged over a user-chosen period band.

This module implements the following per-period procedure:

1. The regional strike ``alpha`` is read from the phase tensor of
   ``Z_obs`` (``z_obj.phase_tensor.alpha``). The phase tensor is
   distortion-invariant (Caldwell-Bibby-Brown, 2004) so its
   principal axis is a property of the regional 2-D structure
   alone.
2. ``Z_obs`` is rotated into the strike frame: ``M = R(-alpha) Z_obs
   R(alpha)``.
3. The regional 2-D impedance in the strike frame is taken to be
   anti-diagonal with off-diagonals equal to ``M``'s off-diagonals
   (the diagonals of ``M``, which carry distortion-induced
   components, are zeroed out): ``Z_R_strike = [[0, M[0, 1]],
   [M[1, 0], 0]]``. This step fixes BCB's gauge (see "Gauge" below).
4. The regional ``Z_R`` is rotated back to the measurement frame.
5. The real 2x2 ``C`` is recovered as the least-squares solution to
   the real-stacked system ``[Re Z_obs | Im Z_obs] = C [Re Z_R | Im
   Z_R]``. Because ``C`` is constrained to be real, this is exact
   for noiseless 2-D-regional data and a meaningful LS estimate
   otherwise.

The per-period ``C`` tensors are then averaged across the band
using ``band_method`` ('median' or 'mean'), and the per-period
residuals ``Z_obs - C_period @ Z_R_period`` are RMS-aggregated.

Gauge
-----
BCB has a known non-determinable scale: replacing ``(C, Z_R)`` with
``(C/k, k Z_R)`` for any real ``k`` leaves ``Z_obs`` unchanged (this
is the static-shift / gain ambiguity). The convention used here is
``gauge="diagonal_unity"``: ``Z_R``'s strike-frame off-diagonals
match the observed strike-frame off-diagonals, so ``C``'s
strike-frame diagonals are ~1. This convention coincides with the
Groom-Bailey-reconstructed ``C`` only when the GB ``gain`` is
itself ~1 and the GB ``twist`` and ``shear`` are small (so
``cos(twist ± shear) ~ 1``); in that regime ``BCB(Z) ~ GB(Z)``
within Frobenius distance ~0.02 on noiseless synthetics. Outside
that regime the two methods report different gauges, not different
physics.

References
----------
Bibby, H. M., Caldwell, T. G., & Brown, C. (2005). Determinable and
non-determinable parameters of galvanic distortion in
magnetotellurics. Geophysical Journal International, 163(3),
915-930.

Caldwell, T. G., Bibby, H. M., & Brown, C. (2004). The
magnetotelluric phase tensor. Geophysical Journal International,
158, 457-469.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .results import BibbyResult

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


__all__ = ["decompose_bibby"]


def decompose_bibby(
    z_obj: "Z",
    periods: tuple[float, float] | None = None,
    band_method: str = "median",
) -> BibbyResult:
    """Bibby-Caldwell-Brown single-site decomposition.

    Per period in the band, fit a real 2x2 distortion matrix ``C``
    such that ``||Z_obs - C @ Z_R||_F`` is minimised for ``Z_R``
    constructed as the anti-diagonal regional 2-D impedance in the
    phase tensor's strike frame; band-average the resulting
    per-period ``C`` tensors.

    Parameters
    ----------
    z_obj : Z
        Impedance tensor object with ``z`` of shape
        ``(n_freqs, 2, 2)``, ``z_error`` of matching shape, and
        ``phase_tensor`` available (i.e., ``Re(Z)`` invertible at
        the periods of interest).
    periods : tuple of float, optional
        ``(min_period, max_period)`` in seconds. Periods outside
        this window are ignored. Default: full period range of
        ``z_obj``.
    band_method : str, default ``"median"``
        Aggregation method for the per-period ``C`` tensors over
        the band. Must be ``"median"`` or ``"mean"``. ``"median"``
        is the recommended default — it is robust to per-period
        outliers driven by phase-tensor noise at periods where the
        regional 2-D assumption is weakest.

    Returns
    -------
    BibbyResult
        Band-averaged ``C``, per-period ``Z_regional`` (in the
        measurement frame), RMS misfit, period count, and
        ``band_method`` / ``gauge`` provenance.

    Raises
    ------
    ValueError
        If ``band_method`` is not ``"median"`` or ``"mean"``, or no
        periods fall inside the requested window.

    See Also
    --------
    decompose : Parameterised Groom-Bailey decomposition; richer
        output but more assumptions.
    DecompositionResult.alternate_branch : The GB symmetry that BCB
        does not need to disambiguate (BCB returns ``C`` directly,
        without strike / shear angles).
    """
    if band_method not in ("median", "mean"):
        raise ValueError(
            f"decompose_bibby: band_method must be 'median' or 'mean', "
            f"got {band_method!r}"
        )

    z_arr = np.asarray(z_obj.z, dtype=np.complex128)
    frequencies = np.asarray(z_obj.frequency, dtype=np.float64)
    all_periods = 1.0 / frequencies

    if periods is not None:
        permin, permax = periods
        mask = (all_periods >= permin) & (all_periods <= permax)
    else:
        mask = np.ones(len(all_periods), dtype=bool)
    if not mask.any():
        raise ValueError(
            f"decompose_bibby: no periods in window {periods}; "
            f"available range is "
            f"({all_periods.min():.3g}, {all_periods.max():.3g}) s"
        )

    z_band = z_arr[mask]
    periods_band = all_periods[mask]
    sort_idx = np.argsort(periods_band)
    z_band = z_band[sort_idx]
    periods_band = periods_band[sort_idx]
    n_band = len(periods_band)

    pt_alpha_full = np.asarray(z_obj.phase_tensor.alpha, dtype=np.float64)
    pt_alpha_band = pt_alpha_full[mask][sort_idx]

    c_per_period = np.full((n_band, 2, 2), np.nan, dtype=np.float64)
    z_r_per_period = np.full((n_band, 2, 2), np.nan + 0j, dtype=np.complex128)
    residuals = np.full((n_band, 2, 2), np.nan + 0j, dtype=np.complex128)

    for i in range(n_band):
        if not np.isfinite(pt_alpha_band[i]):
            continue
        c_per_period[i], z_r_per_period[i], residuals[i] = _bibby_single_period(
            z_band[i], np.radians(pt_alpha_band[i])
        )

    valid = np.all(np.isfinite(c_per_period.reshape(n_band, -1)), axis=1)
    if not valid.any():
        raise ValueError(
            "decompose_bibby: every period in the band has an "
            "undefined phase-tensor strike (singular Re(Z))"
        )

    if band_method == "median":
        c_avg = np.median(c_per_period[valid], axis=0)
    else:
        c_avg = np.mean(c_per_period[valid], axis=0)

    valid_residuals = residuals[valid]
    rms = float(np.sqrt(np.mean(np.abs(valid_residuals) ** 2)))

    return BibbyResult(
        C=c_avg,
        Z_regional=z_r_per_period,
        rms_misfit=rms,
        n_periods=int(valid.sum()),
        band_method=band_method,
        periods=periods_band,
        gauge="diagonal_unity",
    )


def _bibby_single_period(
    z_obs: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-period BCB decomposition with the diagonal-unity gauge.

    Inputs ``z_obs`` (complex 2x2) and ``alpha`` (regional strike,
    radians). Returns ``(C, Z_regional, residual)`` all 2x2 in the
    measurement frame; ``C`` is real, the others complex.
    """
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    R_pos = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    R_neg = R_pos.T

    m = R_neg @ z_obs @ R_pos

    z_r_strike = np.zeros((2, 2), dtype=np.complex128)
    z_r_strike[0, 1] = m[0, 1]
    z_r_strike[1, 0] = m[1, 0]
    z_r = R_pos @ z_r_strike @ R_neg

    a_real = np.column_stack([z_obs.real, z_obs.imag])
    b_real = np.column_stack([z_r.real, z_r.imag])
    c_t, *_ = np.linalg.lstsq(b_real.T, a_real.T, rcond=None)
    c = c_t.T
    residual = z_obs - c @ z_r
    return c, z_r, residual
