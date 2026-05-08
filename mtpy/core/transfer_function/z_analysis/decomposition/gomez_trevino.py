"""Rotational-invariant TE / TM resistivities (Gomez-Treviño 2018).

.. warning::

    **Exploratory module.** The Gomez-Treviño et al. (2018)
    framework constructs rotational-invariant resistivities
    ``rho_+``, ``rho_-`` from the magnetotelluric impedance
    tensor as proposed natural analogues of the 2-D TE / TM modes
    in the 3-D case. The framework's correctness for generic 3-D
    data has not been independently verified in the published
    literature beyond the original paper, and the mode assignment
    of ``rho_+`` / ``rho_-`` to TE / TM is mathematically
    ambiguous (an artefact of the symmetric quadratic that
    defines them). This module is included for *exploration* —
    its outputs should be treated as diagnostic, not as
    production-quality apparent-resistivity estimates, until the
    framework is benchmarked against forward-modelled 3-D
    synthetics in a follow-up PR.

Theoretical background
======================

For an MT impedance tensor ``Z(omega)`` and its admittance
``Y = Z^{-1}``, the framework defines two scalar rotational
invariants (Gomez-Treviño et al. 2018, eqs. 7-8):

    Z_s² = trace(Z^T Z)        (sum of squared entries)
    Y_s² = trace(Y^T Y)

(Here ``Z^T`` is the transpose, *not* the conjugate transpose;
``Z_s²`` is therefore complex in general.) These give the *series*
and *parallel* complex resistivities:

    rho_s = Z_s² / (2 omega mu_0)
    rho_p = 2 / (omega mu_0 · Y_s²)

In 1-D (``Z`` proportional to ``[[0, Z_0], [-Z_0, 0]]``) both
collapse to the Cagniard apparent resistivity ``|Z_0|² / (omega
mu_0)``. In 2-D (anti-diagonal in some strike frame, with TE
``a`` and TM ``b`` impedances) the algebra gives::

    rho_s = (a² + b²) / (2 omega mu_0)
    rho_p = 2 a² b² / (omega mu_0 (a² + b²))

The invariant TE / TM analogues ``rho_+``, ``rho_-`` are the
solutions of the quadratic (Gomez-Treviño eq. 12):

    lambda² − 2 rho_s lambda + rho_s rho_p = 0

with explicit form

    rho_pm = rho_s ± sqrt(rho_s² − rho_s · rho_p)

In 2-D this reduces (algebraically) to ``{a²/(omega mu_0),
b²/(omega mu_0)}`` — i.e. the standard TE and TM apparent
resistivities, in some order. The ``±`` labelling is symmetric
and does not select TE vs TM on its own; that assignment is left
to the consumer.

The Berdichevsky-Dmitriev (1976) determinant resistivity ``rho_d
= sqrt(rho_s · rho_p) = sqrt(rho_+ · rho_-)`` is included for
comparison.

Iterative-chain demonstration
-----------------------------
A central observation of Gomez-Treviño et al. (2018, fig. 1) is
that the recurrence

    rho_s_{i+1} = (rho_s_i + rho_p_i) / 2          (arithmetic mean)
    rho_p_{i+1} = 2 rho_s_i rho_p_i / (rho_s_i + rho_p_i)  (harmonic mean)

with ``rho_s_1 = rho_s``, ``rho_p_1 = rho_p`` converges to the
geometric mean ``rho_d = sqrt(rho_s · rho_p)``. The product
``rho_s_i · rho_p_i`` is invariant at every step (it equals
``rho_s · rho_p`` for all ``i``), so ``rho_+`` and ``rho_-``
are the "first link" in a chain whose limit is the determinant
resistivity. :func:`iterative_chain` exposes the chain.

Numerical verification
----------------------
The forward-model formulas above are verified by the unit-test
module ``tests/.../distortion/test_gomez_trevino.py``:

* Test 1 (1-D Earth) — verifies ``rho_s = rho_p``, the
  discriminant ``rho_s² − rho_s · rho_p`` is zero, and
  ``rho_+ = rho_- = rho_s = rho_d``.
* Test 2 (2-D in strike frame) — verifies the explicit 2-D
  reductions for ``rho_s``, ``rho_p`` and that
  ``{|rho_+|, |rho_-|} = {|a|², |b|²} / (omega mu_0)``.
* Test 3 (2-D rotated) — verifies that ``rho_+`` and ``rho_-``
  are *invariant* under measurement-axis rotation (the central
  rotational-invariance claim).
* Test 4 (2-D + galvanic distortion) — verifies that distortion
  biases ``rho_+`` / ``rho_-`` (as expected — they are not
  distortion-invariant) but they remain invariant under
  measurement-axis rotation of the *distorted* tensor.
* Test 5 (3-D) — verifies that ``rho_+ ≠ rho_-`` in 3-D, that
  ``rho_d`` lies between them, and that the iterative chain
  converges to ``rho_d``.
* Test 6 (chain invariance) — verifies that the geometric-mean
  invariant ``rho_+ · rho_- = rho_d²`` holds at every iteration
  of the chain across all four regimes.

Module status
-------------
* Forward-model algebra: verified.
* Rotational-invariance property: verified.
* 1-D / 2-D reductions: verified.
* 2-D + galvanic distortion behaviour: characterised (biased,
  but rotationally invariant).
* General 3-D behaviour: only mathematically characterised —
  field-data validation deferred.
* Mode assignment of ``+`` / ``−`` to TE / TM: not implemented;
  the labels are symmetric.

Reference
---------
Gómez-Treviño, E., Esparza, F. J., & Romo, J. M. (2018). On the
use of two new invariants of the magnetotelluric impedance tensor
as natural rotational invariant TE and TM modes. *Earth, Planets
and Space*, 70:35. doi:10.1186/s40623-018-0900-y
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .results import GomezTrevinoResult

if TYPE_CHECKING:
    from mtpy.core.transfer_function.z import Z


__all__ = [
    "decompose_gomez_trevino",
    "determinant_resistivity",
    "invariant_resistivities",
    "iterative_chain",
    "series_parallel_resistivities",
]


_MU_0 = 4.0 * np.pi * 1.0e-7


def _omega_from_periods(periods: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi / np.asarray(periods, dtype=np.float64)


def series_parallel_resistivities(
    z: np.ndarray,
    periods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Gomez-Treviño series and parallel resistivities.

    Implements::

        rho_s = trace(Z^T Z) / (2 omega mu_0)
        rho_p = 2 / (omega mu_0 trace(Y^T Y)),  Y = Z^{-1}

    Parameters
    ----------
    z : (n_periods, 2, 2) complex ndarray
        Impedance tensor per period.
    periods : (n_periods,) float ndarray
        Periods in seconds.

    Returns
    -------
    rho_s, rho_p : (n_periods,) complex ndarrays
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    if z_arr.ndim != 3 or z_arr.shape[-2:] != (2, 2):
        raise ValueError(
            f"series_parallel_resistivities: expected z of shape "
            f"(n_periods, 2, 2); got {z_arr.shape}"
        )
    omega = _omega_from_periods(periods)
    if omega.shape != (z_arr.shape[0],):
        raise ValueError(
            f"series_parallel_resistivities: periods length "
            f"{omega.shape[0]} != z first dim {z_arr.shape[0]}"
        )

    # Z^T Z trace = sum of squared entries (complex, no conjugate).
    z_sq_trace = np.sum(z_arr * z_arr, axis=(-2, -1))

    # Y = Z^{-1} per period.
    det_z = (
        z_arr[..., 0, 0] * z_arr[..., 1, 1]
        - z_arr[..., 0, 1] * z_arr[..., 1, 0]
    )
    y = np.empty_like(z_arr)
    y[..., 0, 0] = z_arr[..., 1, 1] / det_z
    y[..., 1, 1] = z_arr[..., 0, 0] / det_z
    y[..., 0, 1] = -z_arr[..., 0, 1] / det_z
    y[..., 1, 0] = -z_arr[..., 1, 0] / det_z

    y_sq_trace = np.sum(y * y, axis=(-2, -1))

    rho_s = z_sq_trace / (2.0 * omega * _MU_0)
    rho_p = 2.0 / (omega * _MU_0 * y_sq_trace)
    return rho_s, rho_p


def determinant_resistivity(
    z: np.ndarray,
    periods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Determinant resistivity ``rho_d = sqrt(rho_s · rho_p)``.

    For 2-D this equals ``sqrt(a² b²) / (omega mu_0) = |a||b|/(omega
    mu_0) = sqrt(rho_TE · rho_TM)`` — the geometric mean of the
    two modes.

    Returns
    -------
    rho_d, phi_d : (n_periods,) float ndarrays
        Magnitude and phase (radians) of the determinant
        resistivity.
    """
    rho_s, rho_p = series_parallel_resistivities(z, periods)
    rho_d_complex = np.sqrt(rho_s * rho_p)
    return np.abs(rho_d_complex), np.angle(rho_d_complex)


def invariant_resistivities(
    z: np.ndarray,
    periods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Invariant TE / TM resistivities ``rho_+``, ``rho_-``.

    Implements::

        rho_pm = rho_s ± sqrt(rho_s² − rho_s · rho_p)

    The ``+`` / ``−`` labels are mathematical (the two roots of
    the symmetric quadratic) and do not encode TE vs TM. In 2-D
    the two values reduce to the standard TE and TM apparent
    resistivities ``a²/(omega mu_0)``, ``b²/(omega mu_0)`` — in
    some order — but the framework provides no general rule for
    which root corresponds to which mode in 3-D.

    Returns
    -------
    rho_plus, rho_minus : (n_periods,) float ndarrays
        Magnitudes ``|rho_pm|``.
    phi_plus, phi_minus : (n_periods,) float ndarrays
        Phases ``arg(rho_pm)`` (radians).
    """
    rho_s, rho_p = series_parallel_resistivities(z, periods)
    discriminant = rho_s * rho_s - rho_s * rho_p
    sqrt_disc = np.sqrt(discriminant)
    rho_pm_plus = rho_s + sqrt_disc
    rho_pm_minus = rho_s - sqrt_disc
    return (
        np.abs(rho_pm_plus),
        np.abs(rho_pm_minus),
        np.angle(rho_pm_plus),
        np.angle(rho_pm_minus),
    )


def iterative_chain(
    z: np.ndarray,
    periods: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate ``(arithmetic, harmonic) mean`` until convergence.

    The recurrence

        rho_s_{i+1} = (rho_s_i + rho_p_i) / 2
        rho_p_{i+1} = 2 rho_s_i rho_p_i / (rho_s_i + rho_p_i)

    converges (quadratically for positive reals; complex case
    typically also quadratic) to the geometric mean
    ``rho_d = sqrt(rho_s_1 · rho_p_1)``. The product
    ``rho_s_i · rho_p_i`` is exactly preserved at every step.

    Parameters
    ----------
    z : (n_periods, 2, 2) complex ndarray
    periods : (n_periods,) float ndarray
    max_iter : int, default 20
    tol : float, default 1e-10
        Stop when ``max |rho_s_i − rho_p_i| / |rho_s_i| < tol``
        across all periods.

    Returns
    -------
    rho_s_history : (max_iter+1, n_periods) complex ndarray
        Sequence of ``rho_s_i`` for ``i = 1 ... iters_used + 1``.
        Rows beyond ``iters_used`` are NaN-filled.
    rho_p_history : (max_iter+1, n_periods) complex ndarray
        Sequence of ``rho_p_i``, same shape.
    iters_per_period : (n_periods,) int ndarray
        Iterations to convergence per period.
    """
    rho_s_0, rho_p_0 = series_parallel_resistivities(z, periods)
    n_periods = rho_s_0.size
    rho_s_hist = np.full((max_iter + 1, n_periods), np.nan, dtype=np.complex128)
    rho_p_hist = np.full((max_iter + 1, n_periods), np.nan, dtype=np.complex128)
    rho_s_hist[0] = rho_s_0
    rho_p_hist[0] = rho_p_0
    iters_per_period = np.full(n_periods, max_iter, dtype=int)

    rho_s = rho_s_0.copy()
    rho_p = rho_p_0.copy()
    converged = np.zeros(n_periods, dtype=bool)
    for i in range(max_iter):
        new_rho_s = 0.5 * (rho_s + rho_p)
        new_rho_p = (
            2.0
            * rho_s
            * rho_p
            / (rho_s + rho_p)
        )
        rho_s = new_rho_s
        rho_p = new_rho_p
        rho_s_hist[i + 1] = rho_s
        rho_p_hist[i + 1] = rho_p

        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(rho_s - rho_p) / np.maximum(np.abs(rho_s), 1e-30)
        newly_converged = (rel < tol) & (~converged)
        iters_per_period[newly_converged] = i + 1
        converged |= rel < tol
        if converged.all():
            break

    return rho_s_hist, rho_p_hist, iters_per_period


def decompose_gomez_trevino(
    z_object: "Z",
    *,
    periods: tuple[float, float] | None = None,
    site: str = "",
    iterative: bool = True,
    max_iter: int = 20,
    tol: float = 1e-10,
) -> GomezTrevinoResult:
    """Compute Gomez-Treviño rotational invariants for a Z object.

    Parameters
    ----------
    z_object : Z
        mtpy ``Z`` instance. Reads ``z.z`` (per-period 2x2 complex)
        and ``z.frequency``.
    periods : (pmin, pmax), optional
        Period window in seconds; selects a sub-range of
        ``z_object.frequency``. ``None`` (default) uses the full
        range.
    site : str, default ``""``
        Identifier for the source site, stored on the result.
    iterative : bool, default True
        If True, also run :func:`iterative_chain` and store the
        per-period iteration count on the result.
    max_iter, tol
        Passed to :func:`iterative_chain` when ``iterative`` is
        True.

    Returns
    -------
    GomezTrevinoResult
    """
    z_arr = np.asarray(z_object.z, dtype=np.complex128)
    freqs = np.asarray(z_object.frequency, dtype=np.float64)
    pers = 1.0 / freqs

    if periods is not None:
        pmin, pmax = periods
        mask = (pers >= pmin) & (pers <= pmax)
        if not mask.any():
            raise ValueError(
                f"decompose_gomez_trevino: no periods in window "
                f"[{pmin}, {pmax}]; available range "
                f"[{pers.min():.3g}, {pers.max():.3g}]"
            )
        z_arr = z_arr[mask]
        pers = pers[mask]

    sort_idx = np.argsort(pers)
    pers = pers[sort_idx]
    z_arr = z_arr[sort_idx]

    rho_s, rho_p = series_parallel_resistivities(z_arr, pers)
    rho_plus, rho_minus, phi_plus, phi_minus = invariant_resistivities(
        z_arr, pers
    )
    rho_d, phi_d = determinant_resistivity(z_arr, pers)

    iters: np.ndarray | None = None
    if iterative:
        _, _, iters = iterative_chain(
            z_arr, pers, max_iter=max_iter, tol=tol
        )

    return GomezTrevinoResult(
        site=site,
        periods=pers,
        rho_s=rho_s,
        rho_p=rho_p,
        rho_plus=rho_plus,
        rho_minus=rho_minus,
        phi_plus=phi_plus,
        phi_minus=phi_minus,
        rho_d=rho_d,
        phi_d=phi_d,
        convergence_iterations=iters,
        metadata={
            "method": "gomez_trevino",
            "exploratory": True,
            "validated": (
                "1-D and 2-D forward-model algebra; rotational "
                "invariance under measurement-axis rotation; "
                "iterative-chain convergence to determinant "
                "resistivity. 3-D field-data interpretation NOT "
                "validated."
            ),
        },
    )
